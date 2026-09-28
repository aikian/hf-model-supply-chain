"""RQ1: number of independent root lineages per functional option (substitutability).

Root = a model with no parent inside the snapshot. A merge model can have several roots.
Independent lineages of option o = size of the union of roots over the models providing o.

Usage
    python substitutability.py ../../02_data/processed/2026-09-25 [--sample 0.05]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from removal_sim import add_common_args, load_inputs


def model_roots(eco):
    """Root set of each model as (model, root) pair arrays. BFS from each root toward descendants."""
    indeg = np.asarray(eco.children.sum(axis=0)).ravel()
    roots = np.flatnonzero(indeg == 0)
    # an isolated root with no children reaches only itself -> skip the BFS
    lone = roots[eco.outdeg[roots] == 0]
    m, r = [lone], [lone]
    for root in roots[eco.outdeg[roots] > 0]:
        desc = eco.reach([root])
        m.append(desc)
        r.append(np.full(len(desc), root))
    return np.concatenate(m), np.concatenate(r)


def bootstrap_share(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def lineage_counts(eco, mm, rr, root_ok=None):
    """Independent root lineages per option. With the wildcard (eco.wildcard), lineages of untagged-language providers
    are added to every language option of the same (task, license). root_ok: mask of roots that count as lineages (sensitivity)."""
    mr = pd.DataFrame({"m": mm, "root": rr})
    if root_ok is not None:
        mr = mr[root_ok[mr["root"].to_numpy()]]
    mo = pd.DataFrame({"m": eco.pair_model, "o": eco.pair_option}).merge(mr, on="m")
    spec = {o: np.unique(g.to_numpy()) for o, g in mo.groupby("o")["root"]}
    empty = np.zeros(0, dtype=np.int64)
    if not eco.wildcard:
        return np.array([len(spec.get(o, empty)) for o in range(eco.n_options)])
    mo["g"] = eco.option_group[mo["o"].to_numpy()]
    unk = mo[eco.option_is_unk[mo["o"].to_numpy()]]
    unk_roots = {g: np.unique(x.to_numpy()) for g, x in unk.groupby("g")["root"]}
    all_roots = {g: np.unique(x.to_numpy()) for g, x in mo.groupby("g")["root"]}
    out = np.zeros(eco.n_options, dtype=np.int64)
    for o in range(eco.n_options):
        g = eco.option_group[o]
        if eco.option_is_unk[o]:
            out[o] = len(all_roots.get(g, empty))          # unk option: a provider of any language in the group substitutes
        else:
            a, b = spec.get(o, empty), unk_roots.get(g, empty)
            out[o] = len(a) + len(b) - len(np.intersect1d(a, b, assume_unique=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    eco, out = load_inputs(args)
    mm, rr = model_roots(eco)
    n_all = lineage_counts(eco, mm, rr)
    indeg = np.asarray(eco.children.sum(axis=0)).ravel()
    isolated = (indeg == 0) & (eco.outdeg == 0)       # models with neither parent nor child: lineage may be undeclared (M9)
    n_known = lineage_counts(eco, mm, rr, root_ok=~isolated)
    # Lower bound (gap 2, 2026-09-26): roots with the same architecture tag (llama, qwen2, ...) are merged into one lineage.
    # This conservatively prevents fine-tunes with undeclared parents from counting as independent lineages.
    # Models trained separately on the same architecture are merged too, so independent lineages are underestimated = lower bound on substitutability.
    n_arch = np.full(eco.n_options, -1)
    ap_ = args.processed / "arch.parquet"
    if ap_.exists():
        a = pd.read_parquet(ap_)
        code = pd.Series(np.arange(eco.n), index=eco.ids)
        fam = np.arange(eco.n) + 1_000_000                    # a root without a tag stays its own lineage
        hit = a["model_id"].isin(code.index)
        cats = pd.factorize(a.loc[hit, "arch"].astype(str))[0]
        fam[code.loc[a.loc[hit, "model_id"]].to_numpy()] = cats
        n_arch = lineage_counts(eco, mm, fam[rr])
    res = pd.DataFrame({"option": eco.option_names, "n_root_lineages": n_all,
                        "n_root_lineages_known": n_known, "n_root_lineages_arch": n_arch,
                        "n_providers": eco.providers})
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "substitutability.csv", index=False)
    s = res[res["n_root_lineages"] > 0]
    print(f"options: {eco.n_options:,}  single-lineage: {100 * (s['n_root_lineages'] == 1).mean():.2f}%")


if __name__ == "__main__":
    main()
