"""H2 검정이 잡아낼 수 있었던 최소 초과 손실 (심사 대응: '유의하지 않음'이 무엇을 배제하는가).

각 검정 (충격, k, 전략) 에서 Holm 첫 단계(p ≤ α/20)를 통과하는 가장 작은 관측 손실을 구하고,
대조군 평균과의 차이(최소 검출 초과 손실)를 옵션 수와 비율로 보고한다. 관측값 − 대조군 평균도 함께.

사용: python detectable_effects.py ../../04_results/tables/2026-09-25_full_main_notest
출력: <dir>/detectable_effects.csv, 요약 출력
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
