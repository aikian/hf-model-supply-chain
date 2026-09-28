"""Hypothesis verdicts (v2, after the 2026-09-26 mock review). Reads the simulation result CSVs and writes the verdict table.

H1   share of functional options with a single lineage > 50%                         <- substitutability.csv
     Full-population data, so no bootstrap CI (the share is exact). A root-definition sensitivity (isolated models excluded) is reported too.
H2   per shock type (legal, availability): at each k, the targeted strategy's loss > the distribution of the
     matched null (random removals with the same number of removed providers), one-sided empirical p. Holm correction
     within each shock-type family (4 strategies x 5 k = 20).
     Supported if significant (for any strategy) at a majority of k.
     Effect size: loss ratio = observed / null mean, and the percentile of the observed value in the null distribution.
     Reference values (not used for the verdict): ratio vs. uniform random (same k), greedy maximal loss.
H3   under the legal shock, for each of the noncommercial and vendor-custom license classes:
     (share of collateral commercial option loss) / (share of models in that class) > 1 (both)

Caveat: the smallest possible empirical p is 1/(seeds+1). With a family of 20, significance needs seeds >= 400.

Usage
    python hypothesis_tests.py ../../04_results/tables/2026-09-25_full_main
    python hypothesis_tests.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05
METRIC = "pct_options_lost"
TARGETED = ["descendants", "descendant_authors", "downloads", "outdegree"]
H3_CLASSES = ["noncommercial", "vendor_custom"]


def empirical_p(observed, null):
    """One-sided empirical p: (1 + #{null ≥ obs}) / (1 + N)."""
    null = np.asarray(null)
    return (1 + np.sum(null >= observed)) / (1 + len(null))


def holm(pvals, alpha=ALPHA):
    """Holm–Bonferroni. Returns (adjusted p, reject flags) in input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj, adj <= alpha


def test_h1(subst):
    s = subst[subst["n_root_lineages"] > 0]
    out = {"share_single_lineage": float((s["n_root_lineages"] == 1).mean()), "n_options": int(len(s))}
    out["supported"] = bool(out["share_single_lineage"] > 0.5)
    if "n_root_lineages_known" in subst:                     # sensitivity: isolated models (no parent, no child) not counted as lineages
        k = subst[subst["n_root_lineages_known"] > 0]
        out["sensitivity_isolated_excluded"] = {
            "share_single_lineage": float((k["n_root_lineages_known"] == 1).mean()),
            "n_options": int(len(k)),
            "options_only_from_isolated_models": int((subst["n_root_lineages_known"] == 0).sum())}
    if "n_root_lineages_arch" in subst and (subst["n_root_lineages_arch"] >= 0).any():   # lower bound: roots grouped by architecture tag
        a = subst[subst["n_root_lineages_arch"] > 0]
        out["lower_bound_architecture_grouped"] = {
            "share_single_lineage": float((a["n_root_lineages_arch"] == 1).mean()), "n_options": int(len(a))}
    return out


def test_h2(res):
    rows = []
    for (sem, k), g in res.groupby(["semantics", "k"]):
        uni = g.loc[g["strategy"] == "random", METRIC].to_numpy()
        greedy = g.loc[g["strategy"] == "greedy", METRIC]
        for s in TARGETED:
            obs = g.loc[g["strategy"] == s, METRIC]
            null = g.loc[(g["strategy"] == "random_matched") & (g["matched_to"] == s), METRIC].to_numpy()
            if obs.empty or not len(null):
                continue
            obs = float(obs.iat[0])
            rows.append({"semantics": sem, "k": k, "strategy": s, "observed": obs,
                         "removed_providers": int(g.loc[g["strategy"] == s, "removed_providers"].iat[0]),
                         "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                         "ratio_vs_matched": obs / null.mean() if null.mean() > 0 else np.inf if obs > 0 else 1.0,
                         "percentile_in_null": float((null < obs).mean() * 100),
                         "p": empirical_p(obs, null), "n_null": len(null),
                         "ratio_vs_uniform_ref": obs / uni.mean() if len(uni) and uni.mean() > 0 else np.nan,
                         "greedy_ref": float(greedy.iat[0]) if len(greedy) else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, {}
    t["p_holm"], t["reject"] = np.nan, False
    for sem, idx in t.groupby("semantics").groups.items():
        t.loc[idx, "p_holm"], t.loc[idx, "reject"] = holm(t.loc[idx, "p"].to_numpy())
    t["reject"] = t["reject"].astype(bool)
    verdict = {}
    for sem, g in t.groupby("semantics"):
        by_k = g.groupby("k")["reject"].any()
        verdict[f"H2_{sem}"] = {"k_significant": int(by_k.sum()), "k_total": int(len(by_k)),
                                "supported": bool(by_k.sum() > len(by_k) / 2)}
    return t, verdict


def test_h3(res):
    rows = []
    for _, r in res[res["strategy"].astype(str).str.startswith("license:")].iterrows():
        cls = r["strategy"].split(":", 1)[1]
        share = r["pct_models_in_class"]
        lost = r["pct_collateral_commercial_lost"]
        rows.append({"class": cls, "pct_models": share, "pct_collateral_commercial_lost": lost,
                     "pct_collateral_options_lost": r["pct_collateral_options_lost"],
                     "ratio": lost / share if share else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, {"supported": None}
    focus = t[t["class"].isin(H3_CLASSES)]
    return t, {"supported": bool(len(focus) == len(H3_CLASSES) and (focus["ratio"] > 1).all()),
               "ratios": {c: round(float(v), 3) for c, v in zip(focus["class"], focus["ratio"])}}


def selftest():
    rng = np.random.default_rng(1)
    assert empirical_p(10, np.zeros(99)) == 0.01
    adj, rej = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and list(rej) == [True, False, False]
    rows = []
    for sem, effect in [("legal", 5.0), ("availability", 0.0)]:
        for k in [1, 5, 10]:
            for s in TARGETED:
                rows.append({"semantics": sem, "k": k, "strategy": s, METRIC: 1.0 + effect, "removed_providers": 100})
                for seed in range(1000):
                    rows.append({"semantics": sem, "k": k, "strategy": "random_matched", "matched_to": s,
                                 METRIC: rng.normal(1.0, 0.3), "removed_providers": 100})
    t, v = test_h2(pd.DataFrame(rows))
    assert v["H2_legal"]["supported"] and not v["H2_availability"]["supported"], v
    h1 = test_h1(pd.DataFrame({"n_root_lineages": [1] * 70 + [2] * 30}))
    assert h1["supported"] and abs(h1["share_single_lineage"] - 0.7) < 1e-9
    print("selftest passed:", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path, nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or args.result_dir is None:
        selftest()
        return
    d = args.result_dir
    out = {}
    if (d / "substitutability.csv").exists():
        out["H1"] = test_h1(pd.read_csv(d / "substitutability.csv"))
    if (d / "removal_results.csv").exists():
        res = pd.read_csv(d / "removal_results.csv")
        t, v = test_h2(res)
        t.to_csv(d / "h2_tests.csv", index=False)
        out.update(v)
        t3, v3 = test_h3(res)
        t3.to_csv(d / "h3_tests.csv", index=False)
        out["H3"] = v3
    (d / "hypothesis_verdicts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
