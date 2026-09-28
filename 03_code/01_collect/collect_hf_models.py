"""Collect metadata for every model on the Hugging Face Hub.

Appends the snapshot to a jsonl.gz using createdAt ascending order and cursor pagination.
If interrupted, rerunning the same command resumes from the cursor in the state file.

Usage:
    python collect_hf_models.py                    # full collection
    python collect_hf_models.py --max-pages 3      # pilot (3,000 models)
HF_TOKEN is used if set in the environment (relaxes the rate limit).
"""
import argparse
import datetime as dt
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "02_data" / "raw"
API = "https://huggingface.co/api/models"
EXPAND = ["createdAt", "lastModified", "author", "tags", "pipeline_tag",
          "library_name", "downloads", "downloadsAllTime", "likes", "gated", "private"]


def first_url(page_size):
    params = [("limit", page_size), ("sort", "createdAt"), ("direction", 1)]
    params += [("expand[]", e) for e in EXPAND]
    return requests.Request("GET", API, params=params).prepare().url


def next_link(resp):
    m = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
    return m.group(1) if m else None


def wait_seconds(resp):
    # RateLimit: "api";r=499;t=257  (t = seconds until the window resets)
    m = re.search(r"t=(\d+)", resp.headers.get("RateLimit", ""))
    return int(m.group(1)) + 1 if m else 60


def fetch(session, url):
    for attempt in range(8):
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as e:
            time.sleep(min(2 ** attempt, 60))
            print(f"  network error ({e}); retry", file=sys.stderr)
            continue
        if resp.status_code == 429:
            s = wait_seconds(resp)
            print(f"  rate limited; sleeping {s}s", file=sys.stderr)
            time.sleep(s)
            continue
        if resp.status_code >= 500:
            time.sleep(min(2 ** attempt, 60))
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"failed after retries: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=dt.date.today().isoformat(),
                    help="Snapshot date tag (use the same value when resuming)")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=None, help="Page limit for pilots")
    ap.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.snapshot}" + (f"_pilot{args.max_pages}" if args.max_pages else "")
    out_path = args.out_dir / f"hf_models_{tag}.jsonl.gz"
    state_path = args.out_dir / f"hf_models_{tag}.state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("done"):
            print(f"already complete: {out_path} ({state['count']} models)")
            return
        print(f"resuming at page {state['pages']} ({state['count']} models)")
    else:
        state = {"snapshot": args.snapshot, "started": dt.datetime.now().isoformat(),
                 "url": first_url(args.page_size), "pages": 0, "count": 0, "done": False}

    session = requests.Session()
    if os.environ.get("HF_TOKEN"):
        session.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

    t0 = time.time()
    while state["url"]:
        if args.max_pages and state["pages"] >= args.max_pages:
            break
        resp = fetch(session, state["url"])
        rows = resp.json()
        # gzip allows member-wise append, so each page is appended as it arrives
        with gzip.open(out_path, "at", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        state["pages"] += 1
        state["count"] += len(rows)
        state["url"] = next_link(resp)
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        if state["pages"] % 10 == 0 or not state["url"]:
            last = rows[-1]["createdAt"] if rows else "-"
            rate = state["count"] / max(time.time() - t0, 1)
            print(f"page {state['pages']:5d}  models {state['count']:9,d}  "
                  f"last createdAt {last}  ({rate:,.0f}/s)")

    state["done"] = state["url"] is None
    state["finished"] = dt.datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"saved {state['count']:,} models -> {out_path}  (complete={state['done']})")


if __name__ == "__main__":
    main()
