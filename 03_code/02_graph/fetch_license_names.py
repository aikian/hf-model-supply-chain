"""Fetch the real license name of license:other models from their model cards (task B11).

A model tagged `license:other` on HF falls into class "other" in license_map.csv, but the model-card
metadata often names the real license in `license_name` / `license_link`
(e.g. black-forest-labs/FLUX.1-dev -> flux-1-dev-non-commercial-license).

Step 1  Selection: models whose effective license (attributes.parquet, with inheritance) is 'other' and that
        (a) have at least --min-desc descendants (eco.reach, self excluded), or
        (b) are among the top --top-dl 'other' models by downloads_all.
Step 2  GET https://huggingface.co/api/models/<id>?expand[]=cardData and collect cardData.license_name and
        cardData.license_link. To respect the anonymous limit (500 per 5 min), sleep --sleep seconds between
        requests; on 429, wait the t= seconds from the RateLimit header and retry.

Output is JSONL, one line per model (model_id, status, license_name, license_link, error, selection info).
Already-recorded model_ids are skipped, so rerunning after an interruption resumes.

Usage (imports removal_sim.py from 03_code/03_simulation):
    python fetch_license_names.py ../../02_data/processed/2026-09-25 \
        --out ../../02_data/raw/license_names_2026-09-25.jsonl
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SIM = Path(__file__).resolve().parents[1] / "03_simulation"
sys.path.insert(0, str(SIM))
from removal_sim import add_common_args, load_inputs  # noqa: E402

API = "https://huggingface.co/api/models/{}"


def select_candidates(processed, min_desc=50, top_dl=100):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    eco, _ = load_inputs(ap.parse_args([str(processed)]))
    # effective license string (same rule as load_inputs: attributes.parquet if present)
    nodes = pd.read_parquet(processed / "nodes.parquet", columns=["model_id", "license", "downloads_all"])
    attr = processed / "attributes.parquet"
    if attr.exists():
        a = pd.read_parquet(attr, columns=["model_id", "license", "inherited_license"])
        nodes = nodes.rename(columns={"license": "license_tag"}).merge(a, on="model_id", how="left")
    else:
        nodes["license_tag"], nodes["inherited_license"] = nodes["license"], False
    nodes = nodes.set_index("model_id").loc[eco.ids].reset_index()
    is_other = nodes["license"].eq("other").to_numpy()

    rows = []
    for i in np.flatnonzero(is_other & (eco.outdeg > 0)):
        r = eco.reach([i])
        d = r[r != i]
        if len(d) >= min_desc:
            rows.append((i, len(d), int(eco.provider[d].sum())))
    sel = {i: (nd, nt) for i, nd, nt in rows}
    dl = nodes["downloads_all"].fillna(0).to_numpy()
    other_idx = np.flatnonzero(is_other)
    top = other_idx[np.argsort(-dl[other_idx], kind="stable")[:top_dl]]
    out = []
    for i in sorted(set(sel) | set(top.tolist())):
        if i in sel:
            nd, nt = sel[i]
        else:
            r = eco.reach([i])
            d = r[r != i]
            nd, nt = len(d), int(eco.provider[d].sum())
        by = "+".join(k for k, on in [("desc", i in sel), ("topdl", i in set(top.tolist()))] if on)
        out.append({"model_id": eco.ids[i], "license_tag": nodes.at[i, "license_tag"],
                    "inherited_license": bool(nodes.at[i, "inherited_license"]),
                    "in_T1": bool(eco.provider[i]), "downloads_all": float(dl[i]),
                    "n_desc": nd, "n_desc_T1": nt, "selected_by": by})
    return pd.DataFrame(out).sort_values("n_desc", ascending=False)


def fetch(model_id, session, max_tries=5):
    err = "429 retries exhausted"
    for _ in range(max_tries):
        try:
            r = session.get(API.format(model_id), params={"expand[]": "cardData"}, timeout=30)
        except requests.RequestException as e:
            err = repr(e)
            time.sleep(5)
            continue
        if r.status_code == 429:
            m = re.search(r"t=(\d+)", r.headers.get("RateLimit", ""))
            wait = int(m.group(1)) + 1 if m else 60
            print(f"  429 → sleep {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            return {"status": r.status_code, "license_name": None, "license_link": None,
                    "error": r.text[:200]}
        cd = (r.json() or {}).get("cardData") or {}
        return {"status": 200, "license_name": cd.get("license_name"),
                "license_link": cd.get("license_link"), "card_license": cd.get("license"),
                "error": None}
    return {"status": None, "license_name": None, "license_link": None, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="JSONL output (supports resuming)")
    ap.add_argument("--min-desc", type=int, default=50)
    ap.add_argument("--top-dl", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.7)
    args = ap.parse_args()

    cand = select_candidates(args.processed, args.min_desc, args.top_dl)
    print(f"candidates {len(cand)}  (desc>={args.min_desc}: {cand.selected_by.str.contains('desc').sum()}, "
          f"topdl: {cand.selected_by.str.contains('topdl').sum()})", flush=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["model_id"] for l in args.out.open(encoding="utf-8") if l.strip()}
    s = requests.Session()
    s.headers["User-Agent"] = "hf-supply-chain-study/1.0 (license_name provenance)"
    with args.out.open("a", encoding="utf-8") as f:
        for k, row in enumerate(cand.to_dict("records")):
            if row["model_id"] in done:
                continue
            res = fetch(row["model_id"], s)
            rec = {"model_id": row["model_id"], **res, **{c: row[c] for c in row if c != "model_id"},
                   "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"{k}/{len(cand)} {row['model_id']} → {res['status']} {res['license_name']}", flush=True)
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
