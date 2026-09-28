"""Smallest excess loss the H2 tests could have detected (review response: what a non-significant result rules out).

For each test (shock, k, strategy), find the smallest observed loss that passes the first Holm step (p ≤ α/20),
and report its difference from the null mean (minimum detectable excess loss) in option counts and as a share. Observed minus null mean is reported too.

Usage: python detectable_effects.py ../../04_results/tables/2026-09-25_full_main_notest
Output: <dir>/detectable_effects.csv, summary printed to stdout
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA, M = 0.05, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    d = ap.parse_args().result_dir
    r = pd.read_csv(d / "removal_results.csv")
    tot = int(r.options_total.iloc[0])
    rows = []
    for (sem, k, s), g in r[r.strategy == "random_matched"].groupby(["semantics", "k", "matched_to"]):
        obs = r[(r.semantics == sem) & (r.k == k) & (r.strategy == s)].options_lost.iat[0]
        null = np.sort(g.options_lost.to_numpy())[::-1]
        n = len(null)
        # (1 + #{null ≥ x}) / (n + 1) ≤ α/M  ⇔  #{null ≥ x} ≤ α/M·(n+1) − 1
        m = int(np.floor(ALPHA / M * (n + 1) - 1))
        crit = null[m] + 1 if m >= 0 else np.inf
        rows.append({"semantics": sem, "k": k, "strategy": s, "observed": obs, "null_mean": null.mean(),
                     "critical": crit, "min_detectable_excess": crit - null.mean(),
                     "observed_minus_null": obs - null.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(d / "detectable_effects.csv", index=False)
    for sem, g in t.groupby("semantics"):
        lo, hi = g.min_detectable_excess.min(), g.min_detectable_excess.max()
        print(f"{sem:12s} min detectable excess {lo:.1f}-{hi:.1f} options "
              f"({100 * lo / tot:.2f}-{100 * hi / tot:.2f}% of {tot:,}); "
              f"largest observed excess {g.observed_minus_null.max():.1f}")


if __name__ == "__main__":
    main()
