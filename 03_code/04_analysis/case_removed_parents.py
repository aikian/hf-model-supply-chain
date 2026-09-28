"""Case analysis: upstream models that have already disappeared (natural experiment).

A declared parent missing from the snapshot = an upstream source that was deleted, made private, or renamed.
For each missing parent, report the following (descriptive statistics, not a hypothesis test):
  - number of declared children, and how many of them are in T1
  - timing of child uploads: date of the last child upload (did derivation continue after the parent vanished?)
  - substitute sources: models in the snapshot with the same model name (organization ignored) = candidate
    mirrors / earlier repositories. Their creation date, downloads, and child counts show where the ecosystem
    moved after the original disappeared
  - among the functional options (task x license class) provided by the missing parent's children, the share
    that models outside the missing lineage still provide (a real case of substitutability)

Caveat: the HF API gives no deletion dates. "After the parent vanished" is relative to the snapshot date.

Output: 04_results/tables/case_removed_parents_<snap>.csv, case_removed_parents_<snap>.md

Usage
    python case_removed_parents.py ../../02_data/processed/2026-09-25 --min-children 20
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code" / "03_simulation"))
from removal_sim import license_class  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--min-children", type=int, default=20)
    args = ap.parse_args()
    P, snap = args.processed, args.processed.name

    n = pd.read_parquet(P / "nodes.parquet", columns=["model_id", "created_at", "downloads_all", "pipeline_tag"])
    attr = pd.read_parquet(P / "attributes.parquet", columns=["model_id", "task", "license"])
    flags = pd.read_parquet(P / "model_flags.parquet", columns=["model_id", "in_T1"])
    e = pd.read_parquet(P / "edges_all.parquet", columns=["parent_id", "child_id", "relation", "parent_in_snapshot", "source"])
    n = n.merge(attr, on="model_id").merge(flags, on="model_id")
    n["opt"] = n["task"].fillna("unk") + "|" + n["license"].map(license_class)
    n["name"] = n["model_id"].str.split("/", n=1).str[1].str.lower()
    idx = n.set_index("model_id")
    kids = e[e["parent_in_snapshot"]].groupby("parent_id").size()

    missing = e[~e["parent_in_snapshot"] & e["source"].eq("declared")]
    big = missing.groupby("parent_id").size()
    big = big[big >= args.min_children].sort_values(ascending=False)

    # providers per option among T1 providers (used to judge substitutability)
    t1 = n[n["in_T1"]]
    rows = []
    for pid, n_decl in big.items():
        ch = missing.loc[missing["parent_id"] == pid, "child_id"]
        c = idx.loc[idx.index.intersection(ch)]
        c1 = c[c["in_T1"]]
        pname = pid.split("/", 1)[1].lower()
        alt = n[(n["name"] == pname) & (n["model_id"] != pid)]
        alt_best = alt.sort_values("downloads_all", ascending=False).head(1)
        # share of the options provided by the missing parent's family that T1 models outside these children still provide
        opts = set(c1["opt"])
        outside = t1[~t1["model_id"].isin(c1.index)]
        still = sum(1 for o in opts if (outside["opt"] == o).any()) if opts else 0
        rows.append({
            "missing_parent": pid,
            "declared_children": int(n_decl),
            "t1_children": int(len(c1)),
            "first_child": str(c["created_at"].min())[:10],
            "last_child": str(c["created_at"].max())[:10],
            "children_downloads": int(c["downloads_all"].fillna(0).sum()),
            "same_name_copies": int(len(alt)),
            "top_copy": alt_best["model_id"].iat[0] if len(alt_best) else "",
            "top_copy_created": str(alt_best["created_at"].iat[0])[:10] if len(alt_best) else "",
            "top_copy_children": int(kids.get(alt_best["model_id"].iat[0], 0)) if len(alt_best) else 0,
            "options_of_children": len(opts),
            "pct_options_still_provided_elsewhere": round(100 * still / len(opts), 1) if opts else None,
        })
    d = pd.DataFrame(rows)
    out = ROOT / "04_results" / "tables"
    d.to_csv(out / f"case_removed_parents_{snap}.csv", index=False)
    md = [f"# Missing upstream model cases ({snap}, at least {args.min_children} declared children)", "",
          f"Missing parents: {len(d)}, declared by {int(d['declared_children'].sum()):,} children "
          f"(T1: {int(d['t1_children'].sum()):,}).", "", d.head(30).to_markdown(index=False)]
    (out / f"case_removed_parents_{snap}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:3]))
    print(d.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
