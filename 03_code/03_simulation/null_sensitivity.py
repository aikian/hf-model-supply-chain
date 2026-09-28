"""Sensitivity of the size-matched null to its matching threshold (review response: the 95% threshold is arbitrary).

Under the main-analysis setting (T1-main: test rule off, language wildcard), the null is redrawn in three ways.
  tol90    add random candidates until removed providers reach at least 90% of the target
  tol99    at least 99%
  nearest  stop at the count closest to the target (the closer of just below and just above)
For each way, the H2 tests (Holm over 20 per shock) are rerun and the null-size / target-size distribution is recorded.
Partial results are saved after each step, so an interrupted run resumes.

Usage: python null_sensitivity.py ../../02_data/processed/2026-09-25 --setting {tol90,tol99,nearest} [--seeds 500]
Output: 04_results/tables/2026-09-25_full_null_<setting>/ (null_results.csv, null_tests.csv, null_verdicts.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from removal_sim import add_common_args, load_inputs, single_closure_sizes, union_matched  # noqa: E402

KS = [1, 5, 10, 50, 100]
TARGETED = ["descendants", "descendant_authors", "downloads", "outdegree"]
SETTINGS = ["tol90", "tol99", "nearest"]


def nearest_matched(rng, eco, cand, dsize, target_n, semantics):
    """Add candidates in random order and stop at the count whose removed-provider total is closest to the target.
    The legal shock uses the incremental method (fast); the availability shock uses bisection. Both pick the same count:
    the removed total never decreases as candidates are added, so both choose between 'the first m* reaching the target'
    and the one just before it, whichever is closer."""
    if semantics == "legal":
        return _nearest_legal_incremental(rng, eco, cand, target_n)
    return _nearest_bisect(rng, eco, cand, dsize, target_n, semantics)


def _reach_new(eco, seeds, mask):
    """Set reached from seeds by following only nodes outside mask (mask itself is not modified).
    A legal-shock mask is closed under descendants (descendants of a marked node are already marked), so stopping at marked nodes is safe."""
    ip, ix = eco.children.indptr, eco.children.indices
    vis = getattr(eco, "_vis_buf", None)
    if vis is None:
        vis = eco._vis_buf = np.zeros(eco.n, dtype=bool)          # reusable buffer (only the cells used are reset at the end)
    frontier = np.unique(np.asarray(seeds, dtype=np.int64))
    frontier = frontier[~mask[frontier]]
    vis[frontier] = True
    out = [frontier]
    while len(frontier):
        starts, lens = ip[frontier], ip[frontier + 1] - ip[frontier]
        if lens.sum() == 0:
            break
        offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        nxt = np.unique(ix[np.arange(lens.sum()) + offs])
        nxt = nxt[~mask[nxt] & ~vis[nxt]]
        vis[nxt] = True
        out.append(nxt)
        frontier = nxt
    res = np.concatenate(out)
    vis[res] = False
    return res


def _nearest_legal_incremental(rng, eco, cand, target_n, chunk=512):
    """Add candidates in random order, chunk at a time, accumulating removed providers. Inside the chunk that crosses
    the target, add one at a time. Picks the same m* as adding singly (the union of a prefix does not depend on chunking)."""
    perm = rng.permutation(len(cand))
    mask = np.zeros(eco.n, dtype=bool)
    count, pos = 0, 0
    while pos < len(cand):
        block = cand[perm[pos:pos + chunk]]
        new = _reach_new(eco, block, mask)
        add = int(eco.provider[new].sum())
        if count + add < target_n:
            mask[new] = True
            count += add
            pos += len(block)
            continue
        prev, last_new, m = count, np.zeros(0, dtype=np.int64), pos
        for c in block:                                   # the target is crossed inside this chunk -> one at a time
            last_new = _reach_new(eco, [c], mask)
            mask[last_new] = True
            prev, count = count, count + int(eco.provider[last_new].sum())
            m += 1
            if count >= target_n:
                break
        if m > 1 and abs(count - target_n) > abs(target_n - prev):   # if the previous count is closer, drop the last candidate
            mask[last_new] = False
            return mask, prev, m - 1
        return mask, count, m
    return mask, count, len(cand)


def _nearest_bisect(rng, eco, cand, dsize, target_n, semantics):
    perm = rng.permutation(len(cand))

    def got(m):
        return int((eco.closure(cand[perm[:m]], semantics) & eco.provider).sum())

    cs = np.cumsum(dsize[perm])
    hi = min(int(np.searchsorted(cs, target_n)) + 1, len(cand))
    while got(hi) < target_n and hi < len(cand):         # grow if overlap leaves it short
        hi = min(int(hi * 1.15) + 1, len(cand))
    lo = 0                                                # bisect while keeping got(lo) < target <= got(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if got(mid) >= target_n:
            hi = mid
        else:
            lo = mid
    g_hi, g_lo = got(hi), got(lo)
    m = hi if abs(g_hi - target_n) <= abs(target_n - g_lo) or lo == 0 else lo
    return eco.closure(cand[perm[:m]], semantics), (g_hi if m == hi else g_lo), m


def empirical_p(obs, null):
    return (1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null))


def holm(p, alpha=0.05):
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj, adj <= alpha


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--setting", choices=SETTINGS, required=True)
    args = ap.parse_args(sys.argv[1:] + ["--drop-rule", "f_test"])
    out = Path(__file__).resolve().parents[2] / "04_results" / "tables" / f"{args.processed.name}_full_null_{args.setting}"
    args.out = out
    eco, _ = load_inputs(args)
    part = out / "_partial"
    part.mkdir(parents=True, exist_ok=True)
    cand = np.flatnonzero((eco.outdeg > 0) & eco.provider)
    cand_desc, cand_auth = eco.descendants_count(cand)
    rank = {"descendants": cand[np.argsort(-cand_desc, kind="stable")],
            "descendant_authors": cand[np.argsort(-cand_auth, kind="stable")],
            "downloads": cand[np.argsort(-eco.downloads[cand], kind="stable")],
            "outdegree": cand[np.argsort(-eco.outdeg[cand], kind="stable")]}
    if args.setting == "nearest":                         # check that the incremental method matches saved bisection results
        f5 = part / "nearest_legal_k5.csv"
        if f5.exists():
            saved = pd.read_csv(f5)
            dsize = single_closure_sizes(eco, cand, "legal", cand_desc)
            tn = int((eco.closure(rank["descendants"][:5], "legal") & eco.provider).sum())
            for seed in range(10):
                rng = np.random.default_rng([seed, 5, len("descendants"), 7])
                mm, got, m = nearest_matched(rng, eco, cand, dsize, tn, "legal")
                row = saved[(saved.matched_to == "descendants") & (saved.seed == seed)].iloc[0]
                assert (got, eco.loss(mm)["options_lost"]) == (row.removed_providers, row.options_lost), (seed, got, row)
            print("verified: incremental nearest matches saved bisection results (10 seeds)", flush=True)
        vr = np.random.default_rng(99)                    # availability shock: vectorized mirror check = original implementation
        for size in [10, 1000, 20000]:
            seeds = np.unique(vr.choice(cand, size=size, replace=False))
            mask = np.zeros(eco.n, dtype=bool)
            mask[seeds] = True
            assert (eco._mirror_substituted(seeds, mask) == eco._mirror_substituted_reference(seeds, mask)).all()
        print("verified: vectorized mirror check matches reference (3 random removal sets)", flush=True)
    rows = []
    for sem in ["legal", "availability"]:
        dsize = single_closure_sizes(eco, cand, sem, cand_desc)
        for k in KS:
            targets = {}
            for s in TARGETED:
                mask = eco.closure(rank[s][:k], sem)
                targets[s] = (int((mask & eco.provider).sum()), eco.loss(mask)["options_lost"])
            for setting in [args.setting]:
                f = part / f"{setting}_{sem}_k{k}.csv"
                if f.exists():
                    rows.extend(pd.read_csv(f).to_dict("records"))
                    print(f"[{sem}] k={k} done (resumed)", flush=True)
                    continue
                block = []
                for s, (tn, tloss) in targets.items():
                    block.append({"setting": setting, "semantics": sem, "k": k, "strategy": s, "seed": None,
                                  "removed_providers": tn, "options_lost": tloss})
                    for seed in range(args.seeds):
                        rng = np.random.default_rng([seed, k, len(s), 7])
                        if setting == "nearest":
                            mm, got, m = nearest_matched(rng, eco, cand, dsize, tn, sem)
                        else:
                            mm, got, m = union_matched(rng, eco, cand, dsize, tn, sem,
                                                       tol=0.90 if setting == "tol90" else 0.99)
                        block.append({"setting": setting, "semantics": sem, "k": k, "strategy": "null",
                                      "matched_to": s, "seed": seed, "removed_providers": got,
                                      "size_ratio": got / tn if tn else np.nan,
                                      "options_lost": eco.loss(mm)["options_lost"]})
                pd.DataFrame(block).to_csv(f.with_suffix(".tmp"), index=False)
                f.with_suffix(".tmp").replace(f)
                rows.extend(block)
                print(f"[{sem}] k={k} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "null_results.csv", index=False)

    tests, verdicts = [], {}
    for (setting, sem, k), g in df.groupby(["setting", "semantics", "k"]):
        for s in TARGETED:
            # null rows are identified by matched_to: pandas reads the CSV string "null" as missing, so strategy cannot be used
            obs = g[(g.strategy == s) & g.matched_to.isna()].options_lost.iat[0]
            null = g[g.matched_to == s]
            tests.append({"setting": setting, "semantics": sem, "k": k, "strategy": s, "observed": obs,
                          "null_mean": null.options_lost.mean(), "p": empirical_p(obs, null.options_lost),
                          "size_ratio_median": null.size_ratio.median(),
                          "size_ratio_p5": null.size_ratio.quantile(0.05),
                          "size_ratio_p95": null.size_ratio.quantile(0.95)})
    t = pd.DataFrame(tests)
    t["p_holm"], t["reject"] = np.nan, False
    for key, idx in t.groupby(["setting", "semantics"]).groups.items():
        t.loc[idx, "p_holm"], t.loc[idx, "reject"] = holm(t.loc[idx, "p"].to_numpy())
    for (setting, sem), g in t.groupby(["setting", "semantics"]):
        ks = int(g.groupby("k")["reject"].any().sum())
        verdicts[f"{setting}_{sem}"] = {"k_significant": ks, "supported": bool(ks > 2),
                                        "min_p": float(g.p.min()),
                                        "size_ratio_median_range": [float(g.size_ratio_median.min()),
                                                                    float(g.size_ratio_median.max())]}
    t.to_csv(out / "null_tests.csv", index=False)
    (out / "null_verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    (out / "hypothesis_verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")   # completion marker for the progress window
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
