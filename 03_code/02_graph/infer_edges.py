"""Infer undeclared lineage edges and remove cycles.

Input   nodes.parquet, edges.parquet (build_graph.py output)
Output  edges_all.parquet   declared + inferred edges. Column source in {declared, inferred_mirror, inferred_name}
                            edges with in_cycle = True are excluded from analysis
      inference_report.json

Inference rules (conservative: only when the name maps uniquely inside the snapshot)
    inferred_name    RichardErkhov format "<author>_-_<model>[-gguf|-4bits|-8bits|...]"
                     → <author>/<model>
    inferred_mirror  a model with no declared parent whose name (author stripped) exactly equals the name
                     of a base model with 20+ children, and which was uploaded after that base model
                     -> that base model as parent (relation = "mirror")
    inferred_quant   a model with a quantization suffix (-GGUF, -AWQ, -GPTQ, -EXL2, -MLX, -Nbit, -bnb-4bit),
                     no declared parent, and whose suffix-stripped name maps to exactly one model in the snapshot
                     -> that model as parent (relation = "quantized")

Usage
    python infer_edges.py ../../02_data/processed/2026-09-25
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

MIRROR_MIN_CHILDREN = 20
QUANT_SUFFIX = re.compile(
    r"[-_.](?:i1-)?(?:gguf|awq|gptq|exl2|mlx(?:-\d+bits?)?|\d+(?:\.\d+)?-?bits?|\d+bpw|bnb-4bit|"
    r"q\d(?:_k)?(?:_[msl])?|fp8|int[48])(?:[-_].*)?$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()

    n = pd.read_parquet(args.processed / "nodes.parquet",
                        columns=["model_id", "author", "created_at", "created_at_legacy", "n_parents"])
    e = pd.read_parquet(args.processed / "edges.parquet")
    e["source"] = "declared"
    # 2026-09-26: declared parents missing from the snapshot that were confirmed as renamed / old IDs / local paths
    # are reconnected to their real IDs (resolve_missing_parents.py). Before, these children were cut off from their
    # parents and counted as independent lineages.
    e["parent_resolution"] = None
    rp = args.processed / "missing_parents_resolved.csv"
    if rp.exists():
        rs = pd.read_csv(rp)
        rs = rs[rs["in_snapshot"].astype(bool) & rs["status"].isin(["renamed", "alias", "local_path"])]
        mp = rs.set_index("parent_id")["resolved_id"]
        st = rs.set_index("parent_id")["status"]
        hit = ~e["parent_in_snapshot"] & e["parent_id"].isin(mp.index)
        e.loc[hit, "parent_resolution"] = e.loc[hit, "parent_id"].map(st)
        e.loc[hit, "parent_id_declared"] = e.loc[hit, "parent_id"]
        e.loc[hit, "parent_id"] = e.loc[hit, "parent_id"].map(mp)
        e.loc[hit, "parent_in_snapshot"] = True
        e = e[e["parent_id"] != e["child_id"]].drop_duplicates(["parent_id", "child_id"])
        created0 = n.set_index("model_id")["created_at"]
        legacy0 = n.set_index("model_id")["created_at_legacy"]
        # Reconnected edges get no temporal check (NA): the linked repository's creation time is not the original's upload time.
        # E.g. runwayml/stable-diffusion-v1-5 -> stable-diffusion-v1-5/stable-diffusion-v1-5 is a redistribution created after
        # the original vanished, so it is dated later. With the check, 2,540 edges would drop as 'time reversed', voiding the reconnection.
        h = e["parent_resolution"].notna()
        e.loc[h, "temporal_ok"] = pd.NA
        print(f"reconnected {int(h.sum()):,} declared edges via name resolution "
              f"({e.loc[h, 'parent_resolution'].value_counts().to_dict()})")

    name = n["model_id"].str.split("/", n=1).str[1]
    low_id = n["model_id"].str.lower()
    id_by_low = pd.Series(n["model_id"].to_numpy(), index=low_id)
    id_by_low = id_by_low[~id_by_low.index.duplicated()]
    created = n.set_index("model_id")["created_at"]
    orphan = n["n_parents"].eq(0)

    # name_low -> model list (for uniqueness checks)
    name_low = name.str.lower()
    name_count = name_low.value_counts()

    inferred = []

    # 1) RichardErkhov format
    m = n["author"].eq("RichardErkhov") & orphan
    parts = name[m].str.extract(r"^(?P<a>.+?)_-_(?P<b>.+)$")
    base = parts["b"].str.replace(QUANT_SUFFIX, "", regex=True)
    cand = (parts["a"] + "/" + base).str.lower().map(id_by_low)
    ok = cand.notna()
    inferred.append(pd.DataFrame({"parent_id": cand[ok].to_numpy(), "child_id": n.loc[m, "model_id"][ok].to_numpy(),
                                  "relation": "quantized", "source": "inferred_name"}))

    # 2) mirrors
    outdeg = e[e["parent_in_snapshot"]].groupby("parent_id").size()
    popular = outdeg[outdeg >= MIRROR_MIN_CHILDREN].index
    pop = pd.DataFrame({"parent_id": popular, "nl": pd.Series(popular).str.split("/", n=1).str[1].str.lower()})
    pop = pop[pop["nl"].map(pop["nl"].value_counts()).eq(1)]          # several popular models sharing a name are ambiguous -> excluded
    pmap = pop.set_index("nl")["parent_id"]
    m = orphan & name_low.isin(pmap.index) & ~n["model_id"].isin(popular)
    par = name_low[m].map(pmap)
    later = n.loc[m, "created_at"].to_numpy() >= par.map(created).to_numpy()
    ch = n.loc[m, "model_id"][later]
    inferred.append(pd.DataFrame({"parent_id": par[later].to_numpy(), "child_id": ch.to_numpy(),
                                  "relation": "mirror", "source": "inferred_mirror"}))

    # 3) quantization suffix
    done = set(pd.concat(inferred)["child_id"])
    has_q = name.str.contains(QUANT_SUFFIX, regex=True)
    m = orphan & has_q & ~n["model_id"].isin(done)
    stripped = name[m].str.replace(QUANT_SUFFIX, "", regex=True).str.lower()
    uniq = stripped.map(name_count).eq(1)                              # name is unique in the snapshot
    nl_to_id = pd.Series(n["model_id"].to_numpy(), index=name_low)
    nl_to_id = nl_to_id[~nl_to_id.index.duplicated(keep=False)]
    par = stripped[uniq].map(nl_to_id)
    ok = par.notna() & (par != n.loc[m, "model_id"][uniq])
    ch = n.loc[m, "model_id"][uniq][ok]
    par = par[ok]
    later = ch.map(created).to_numpy() >= par.map(created).to_numpy()
    inferred.append(pd.DataFrame({"parent_id": par[later].to_numpy(), "child_id": ch[later].to_numpy(),
                                  "relation": "quantized", "source": "inferred_quant"}))

    inf = pd.concat(inferred, ignore_index=True).drop_duplicates(["parent_id", "child_id"])
    inf = inf[inf["parent_id"] != inf["child_id"]]
    inf["parent_in_snapshot"] = True
    pc, cc = inf["parent_id"].map(created), inf["child_id"].map(created)
    inf["temporal_ok"] = (pc <= cc)
    allE = pd.concat([e, inf], ignore_index=True).drop_duplicates(["parent_id", "child_id"])

    # 4) cycles: edges inside a strongly connected component (size >= 2) are excluded from analysis
    idx = pd.Series(np.arange(len(n)), index=n["model_id"])
    inside = allE[allE["parent_in_snapshot"]]
    A = sparse.csr_matrix((np.ones(len(inside)), (idx[inside["parent_id"]].to_numpy(),
                                                  idx[inside["child_id"]].to_numpy())), shape=(len(n), len(n)))
    _, lab = connected_components(A, directed=True, connection="strong")
    big = np.bincount(lab) > 1
    allE["in_cycle"] = False
    cs = allE.index[allE["parent_in_snapshot"]]
    lp = lab[idx[allE.loc[cs, "parent_id"]].to_numpy()]
    lc = lab[idx[allE.loc[cs, "child_id"]].to_numpy()]
    allE.loc[cs, "in_cycle"] = (lp == lc) & big[lp]

    allE.to_parquet(args.processed / "edges_all.parquet", index=False)
    rep = {
        "declared_edges": int(len(e)),
        "declared_edges_reconnected_by_name_resolution": int(e["parent_resolution"].notna().sum()),
        "inferred_edges": inf["source"].value_counts().to_dict(),
        "inferred_temporal_violations": int((~inf["temporal_ok"]).sum()),
        "models_gaining_parent": int(inf["child_id"].nunique()),
        "cycle_components": int(big.sum()),
        "edges_in_cycles": int(allE["in_cycle"].sum()),
        "total_edges": int(len(allE)),
        "examples": {s: g[["parent_id", "child_id"]].head(5).values.tolist() for s, g in inf.groupby("source")},
    }
    (args.processed / "inference_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
