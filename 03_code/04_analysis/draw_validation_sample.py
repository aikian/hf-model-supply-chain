"""Draw the cleaning validation sample for the paper (manuscript section "Data Cleaning and Its Validation"). Run in Stage 2, after the rules are frozen.

- 50 models per exclusion rule (f_bot, f_boilerplate, f_course, f_test) + 100 retained (T1) models
- models seen during rule development (dev_rule_development/*.csv) are excluded from the candidate pool
- a model that matches several strata stays only in the stratum drawn first
- writes a labeling file that hides stratum and flags from the rater, and a separate key file with the strata

Output (02_data/validation/)
    paper_validation_to_label.csv   model_id, url, label, type, note  (for the rater, shuffled)
    paper_validation_key.csv        model_id, stratum                  (open only after labeling is finished)
    paper_validation_meta.json      seed, rule file hashes, counts per stratum

Usage
    python draw_validation_sample.py ../../02_data/processed/2026-09-25 --seed 20270205
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "02_data" / "validation"
STRATA = {"f_bot": 50, "f_boilerplate": 50, "f_course": 50, "f_test": 50, "kept_T1": 100}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--seed", type=int, required=True, help="seed fixed in advance and reported in the manuscript")
    args = ap.parse_args()

    out_label = VAL / "paper_validation_to_label.csv"
    if out_label.exists():
        raise SystemExit(f"already exists: {out_label} (redrawing would invalidate the validation)")

    f = pd.read_parquet(args.processed / "model_flags.parquet")
    seen = set()
    for csv in (VAL / "dev_rule_development").glob("*.csv"):
        d = pd.read_csv(csv)
        if "model_id" in d:
            seen |= set(d["model_id"])
    f = f[~f["model_id"].isin(seen)]

    taken, parts = set(), []
    for i, (stratum, size) in enumerate(STRATA.items()):
        pool = f[f["in_T1"]] if stratum == "kept_T1" else f[f[stratum]]
        pool = pool[~pool["model_id"].isin(taken)]
        s = pool.sample(n=size, random_state=args.seed + i)
        taken |= set(s["model_id"])
        parts.append(pd.DataFrame({"model_id": s["model_id"], "stratum": stratum}))
    key = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    lab = key[["model_id"]].assign(url="https://huggingface.co/" + key["model_id"],
                                   label="", type="", note="")
    lab.to_csv(out_label, index=False, encoding="utf-8-sig")
    key.to_csv(VAL / "paper_validation_key.csv", index=False, encoding="utf-8-sig")
    meta = {"seed": args.seed, "excluded_dev_models": len(seen),
            "strata": key["stratum"].value_counts().to_dict(),
            "rules_sha256": sha256(ROOT / "03_code" / "02_graph" / "clean_models.py"),
            "flags_sha256": sha256(args.processed / "model_flags.parquet")}
    (VAL / "paper_validation_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
