"""Rerun the original plan (the analysis fixed before the first full run) with only implementation bugs corrected and the final data.

Review response: the RQ1-RQ3 figures in the manuscript come from a 'revised exploratory analysis' changed after seeing results.
This script separately produces the 'implementation-corrected planned analysis'.

Original plan (same as removal_sim.py / hypothesis_tests.py in the first commit 21a16a8)
  population  T1-original (all cleaning rules, including the test rule)
  option      task x language x license, missing language = separate value (now --option strict)
  shock       contagion: removed models + all descendants (now legal)
  H1          bootstrap 95% CI lower bound of the share of single-lineage options > 50%
  H2a         per k, loss of the 4 targeted strategies > k uniform random picks (one-sided empirical p, Holm over 20); supported if significant for a majority of k
  H2b         per k, loss of the descendants target > k random picks matched on the log2(descendants+1) bin distribution (Holm over 5)
  effect      Cliff's delta
  H3          for at least one class, (% commercial options lost) / (% models in class) > 1
  seeds    1,000
Implementation fixes kept: (1) shared-name rule exceptions, (2) count provider descendants only, (4) no-derivatives is noncommercial.
Uses the final data (including parent reconnection). This is a data change and is disclosed in the manuscript.

Usage: python planned_analysis.py ../../02_data/processed/2026-09-25 [--seeds 1000]
Output: 04_results/tables/2026-09-25_planned/ (removal_results.csv, substitutability.csv, planned_tests.csv, planned_verdicts.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "04_analysis"))
from removal_sim import LICENSE_SCENARIOS, add_common_args, load_inputs  # noqa: E402
from substitutability import lineage_counts, model_roots  # noqa: E402

ALPHA = 0.05
KS = [1, 5, 10, 50, 100]
TARGETED = ["descendants", "descendant_authors", "downloads", "outdegree"]


def matched_random(rng, cand, cand_desc, target, k):
    """Same as v1: random draw matching the target set's per-bin counts of log2(descendants+1)."""
    bins = np.floor(np.log2(cand_desc + 1)).astype(int)
    in_t = np.isin(cand, target)
    tbins = np.floor(np.log2(cand_desc[in_t] + 1)).astype(int)
    pick = []
    for b, cnt in zip(*np.unique(tbins, return_counts=True)):
        pool = cand[(bins == b) & ~in_t]
        if len(pool) < cnt:
            pool = cand[(bins >= b - 1) & ~in_t]
        pick.extend(rng.choice(pool, size=min(cnt, len(pool)), replace=False))
    return np.array(pick[:k])


def empirical_p(obs, null):
    return (1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null))


def cliffs_delta(x, y):
    y = np.asarray(y)
    return float((x > y).mean() - (x < y).mean())


def holm(p, alpha=ALPHA):
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj, adj <= alpha


def simulate(eco, seeds, out):
    part = out / "_partial"
    part.mkdir(parents=True, exist_ok=True)
    cand = np.flatnonzero((eco.outdeg > 0) & eco.provider)
    cand_desc, cand_auth = eco.descendants_count(cand)
    rank = {"descendants": cand[np.argsort(-cand_desc, kind="stable")],
            "descendant_authors": cand[np.argsort(-cand_auth, kind="stable")],
            "downloads": cand[np.argsort(-eco.downloads[cand], kind="stable")],
            "outdegree": cand[np.argsort(-eco.outdeg[cand], kind="stable")]}
    rows = []
    for k in KS:
        f = part / f"k{k}.csv"
        if f.exists():
            rows.extend(pd.read_csv(f).to_dict("records"))
            print(f"[legal] k={k} done (resumed)", flush=True)
            continue
        block = []
        for name, order in rank.items():
            block.append({"k": k, "strategy": name, "seed": None, **eco.loss(eco.closure(order[:k], "legal"))})
        target = rank["descendants"][:k]
        for s in range(seeds):
            rng = np.random.default_rng(s)
            rnd = rng.choice(cand, size=min(k, len(cand)), replace=False)
            block.append({"k": k, "strategy": "random", "seed": s, **eco.loss(eco.closure(rnd, "legal"))})
            mr = matched_random(rng, cand, cand_desc, target, k)
            block.append({"k": k, "strategy": "random_matched", "seed": s, **eco.loss(eco.closure(mr, "legal"))})
        pd.DataFrame(block).to_csv(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        rows.extend(block)
        print(f"[legal] k={k} done", flush=True)
    for cls in LICENSE_SCENARIOS:
        m = (eco.license_class == cls) & eco.provider
        rows.append({"k": int(m.sum()), "strategy": f"license:{cls}", "seed": None,
                     "pct_models_in_class": 100 * m.sum() / max(eco.provider.sum(), 1),
                     **eco.loss(eco.closure(np.flatnonzero(m), "legal"), removed_class=cls)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "removal_results.csv", index=False)
    return df


def tests(res, sub):
    out = {}
    single = (sub.loc[sub.n_root_lineages > 0, "n_root_lineages"] == 1).to_numpy(float)
    rng = np.random.default_rng(0)
    boots = rng.choice(single, size=(5000, len(single))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["H1"] = {"share_single_lineage": float(single.mean()), "ci95": [float(lo), float(hi)],
                 "n_options": int(len(single)), "supported": bool(lo > 0.5)}
    rows = []
    for k, g in res[~res.strategy.astype(str).str.startswith("license:")].groupby("k"):
        nu = g.loc[g.strategy == "random", "pct_options_lost"].to_numpy()
        nm = g.loc[g.strategy == "random_matched", "pct_options_lost"].to_numpy()
        for s in TARGETED:
            obs = float(g.loc[g.strategy == s, "pct_options_lost"].iat[0])
            rows.append({"hyp": "H2a", "k": k, "strategy": s, "observed": obs, "null_mean": nu.mean(),
                         "p": empirical_p(obs, nu), "cliffs_delta": cliffs_delta(obs, nu)})
            if s == "descendants":
                rows.append({"hyp": "H2b", "k": k, "strategy": s, "observed": obs, "null_mean": nm.mean(),
                             "p": empirical_p(obs, nm), "cliffs_delta": cliffs_delta(obs, nm)})
    t = pd.DataFrame(rows)
    t["p_holm"], t["reject"] = np.nan, False
    for h, idx in t.groupby("hyp").groups.items():
        t.loc[idx, "p_holm"], t.loc[idx, "reject"] = holm(t.loc[idx, "p"].to_numpy())
    for h, g in t.groupby("hyp"):
        by_k = g.groupby("k")["reject"].any()
        out[h] = {"k_significant": int(by_k.sum()), "k_total": int(len(by_k)),
                  "supported": bool(by_k.sum() > len(by_k) / 2),
                  "cliffs_delta_range": [float(g.cliffs_delta.min()), float(g.cliffs_delta.max())]}
    lic = res[res.strategy.astype(str).str.startswith("license:")]
    ratios = {r.strategy.split(":", 1)[1]: (r.pct_commercial_options_lost / r.pct_models_in_class
                                            if r.pct_models_in_class else float("nan"))
              for r in lic.itertuples()}
    out["H3"] = {"ratios_including_own_class": {c: round(float(v), 3) for c, v in ratios.items()},
                 "supported": bool(any(v > 1 for v in ratios.values()))}
    return t, out


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--seeds", type=int, default=1000)
    extra = ["--option", "strict"] + ([] if "--tier" in sys.argv else ["--tier", "T1"])
    args = ap.parse_args(sys.argv[1:] + extra)
    suffix = "" if args.tier == "T1" else f"_{args.tier}"          # T0: the population whose H2b verdict disagreed with T1 in the first run
    args.out = args.out or Path(__file__).resolve().parents[2] / "04_results" / "tables" / f"{args.processed.name}_full_planned{suffix}"
    eco, out = load_inputs(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()}, indent=1), encoding="utf-8")
    res = simulate(eco, args.seeds, out)
    mm, rr = model_roots(eco)
    sub = pd.DataFrame({"option": eco.option_names, "n_root_lineages": lineage_counts(eco, mm, rr)})
    sub.to_csv(out / "substitutability.csv", index=False)
    t, v = tests(res, sub)
    t.to_csv(out / "planned_tests.csv", index=False)
    (out / "planned_verdicts.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    (out / "hypothesis_verdicts.json").write_text(json.dumps(v, indent=1), encoding="utf-8")   # completion marker for the progress window
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
