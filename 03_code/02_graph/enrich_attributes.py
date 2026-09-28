"""Attribute inheritance: fill the empty task/language/license of quantized and mirror models from the parent.

Rationale: quantized models and mirrors share the parent's weights, so their function (task, language) is the same,
      and the parent's license applies to them as well.
      Fine-tunes, adapters and merges may change function or license, so they do not inherit.
Rule: fill only when the child's value is **empty** (declared values are never overwritten).
      Multi-step chains (quantization of a quantization, quantization of a mirror) are handled by iterating.

Input   nodes.parquet, edges_all.parquet
Output  attributes.parquet   model_id, task, languages, license (with inheritance) + inherited_* markers
      enrich_report.json

Usage
    python enrich_attributes.py ../../02_data/processed/2026-09-25
"""
import argparse
import json
from pathlib import Path

import pandas as pd

INHERIT_RELATIONS = {"quantized", "mirror"}
FIELDS = {"task": "pipeline_tag", "languages": "languages", "license": "license"}
MAX_DEPTH = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()

    n = pd.read_parquet(args.processed / "nodes.parquet", columns=["model_id", *FIELDS.values()])
    e = pd.read_parquet(args.processed / "edges_all.parquet",
                        columns=["parent_id", "child_id", "relation", "parent_in_snapshot", "in_cycle"])
    e = e[e["relation"].isin(INHERIT_RELATIONS) & e["parent_in_snapshot"] & ~e["in_cycle"]]
    # a quantized/mirror model with several parents is ambiguous -> no inheritance
    e = e[~e["child_id"].duplicated(keep=False)]

    base = n.rename(columns={v: k for k, v in FIELDS.items()}).set_index("model_id")
    before = {k: int(base[k].isna().sum()) for k in FIELDS}

    def inherit(a):
        a = a.copy()
        for k in FIELDS:
            a[f"inherited_{k}"] = False
        rounds = 0
        for rounds in range(1, MAX_DEPTH + 1):
            changed = 0
            for k in FIELDS:
                pv = e["parent_id"].map(a[k])
                cv = e["child_id"].map(a[k])
                fill = cv.isna() & pv.notna()
                if fill.any():
                    ids = e.loc[fill, "child_id"].to_numpy()
                    a.loc[ids, k] = pv[fill].to_numpy()
                    a.loc[ids, f"inherited_{k}"] = True
                    changed += int(fill.sum())
            if not changed:
                break
        return a, rounds

    # License reclassification (license_overrides.csv; model-card license_name checked with fetch_license_names.py):
    # the real license class of large roots tagged 'other'. Written as "override:<class>" so license_class() reads it directly.
    # Applied before inheritance, so quantized/mirror models with an empty license inherit the corrected value.
    ov_path = Path(__file__).resolve().parents[1] / "03_simulation" / "license_overrides.csv"
    a_raw, _ = inherit(base)
    b = base.copy()
    n_ov = 0
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        ov = ov[(ov["class"] != "other") & ov["model_id"].isin(b.index)]
        b.loc[ov["model_id"].to_numpy(), "license"] = ("override:" + ov["class"]).to_numpy()
        n_ov = len(ov)
    a, rounds = inherit(b)
    a["license_no_override"] = a_raw["license"]
    a.reset_index().to_parquet(args.processed / "attributes.parquet", index=False)
    rep = {
        "inherit_relations": sorted(INHERIT_RELATIONS),
        "eligible_edges": int(len(e)),
        "rounds": rounds,
        "license_overrides_applied": n_ov,
        "models_with_override_license": int(a["license"].astype(str).str.startswith("override:").sum()),
        "missing_before": before,
        "missing_after": {k: int(a[k].isna().sum()) for k in FIELDS},
        "filled": {k: int(a[f"inherited_{k}"].sum()) for k in FIELDS},
    }
    (args.processed / "enrich_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
