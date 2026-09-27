"""Fig: RQ2 — 모델 손실 대 기능 손실. 충격 종류별 두 패널.

x = 사용 불가가 된 제공자(T1 모델) 비율 (%, 로그)
y = 제공자가 하나도 남지 않은 기능 옵션 수 (전체 대비 %는 오른쪽 축)
점: 표적 전략 4개 × k 5개 (점 크기 = k). 회색: 같은 규모로 맞춘 무작위 대조군의 평균과 5–95% 범위.
검은 선: 탐욕적 최대 손실 (참고).

출력: 04_results/figures/fig_rq2_contrast.{pdf,png} (+ _gray.png)
사용: python fig_rq2_contrast.py ../../04_results/tables/2026-09-25_full_main
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STRAT = [("descendants", "descendants", "#23395B", "o"), ("descendant_authors", "descendant accounts", "#B03A2E", "s"),
         ("downloads", "downloads", "#2E7D5B", "P"), ("outdegree", "out-degree", "#C7862F", "D")]
SEM = [("legal", "(a) Legal shock: restriction propagates to all descendants"),
       ("availability", "(b) Availability shock: adapters only; mirrors substitute")]
KSIZE = {1: 9, 5: 14, 10: 20, 50: 30, 100: 42}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--providers", type=int, default=None, help="T1 제공자 수 (기본: 결과에서 추정 불가 → 필수)")
    args = ap.parse_args()
    d = pd.read_csv(args.result_dir / "removal_results.csv")
    n_opt = int(d["options_total"].iloc[0])
    n_prov = args.providers
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9), gridspec_kw={"height_ratios": [1.7, 1]})
    ymax = max(1, d.loc[d["strategy"].isin([s for s, *_ in STRAT] + ["greedy"]), "options_lost"].max(),
               d.loc[d["strategy"] == "random_matched"].groupby(["semantics", "k", "matched_to"])["options_lost"]
               .quantile(0.95).max())
    for ax, (sem, title) in zip(axes, SEM):
        g = d[d["semantics"] == sem]
        for s, lab, col, mk in STRAT:
            obs = g[g["strategy"] == s]
            null = g[(g["strategy"] == "random_matched") & (g["matched_to"] == s)]
            for k, o in obs.set_index("k").iterrows():
                nk = null[null["k"] == k]
                x = 100 * o["removed_providers"] / n_prov
                nx = 100 * nk["removed_providers"].mean() / n_prov
                lo, hi = nk["options_lost"].quantile([0.05, 0.95])
                ax.plot([nx, nx], [lo, hi], color="#9AA3AD", lw=0.8, zorder=1)
                ax.scatter([nx], [nk["options_lost"].mean()], s=KSIZE[k] * 0.55, color="#9AA3AD", marker="_", lw=1.1,
                           zorder=2)
                ax.scatter([x], [o["options_lost"]], s=KSIZE[k], marker=mk, facecolor="white", edgecolor=col,
                           lw=0.9, zorder=3)
        gr = g[g["strategy"] == "greedy"].sort_values("k")
        ax.plot(100 * gr["removed_providers"] / n_prov, gr["options_lost"], color="#1F2328", lw=0.8,
                ls=(0, (3, 1.5)), marker=".", ms=3, zorder=2)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(-0.002, 30)
        ax.set_xticks([0, 0.01, 0.1, 1, 10], ["0", "0.01", "0.1", "1", "10"])
        ax.set_ylim(-0.5, ymax * 1.15 + 1)
        ax.set_ylabel("Functional options lost\n(of %s)" % f"{n_opt:,}")
        sec = ax.secondary_yaxis("right", functions=(lambda v: 100 * v / n_opt, lambda p: p * n_opt / 100))
        sec.set_ylabel("% of options", fontsize=5.8)
        sec.tick_params(labelsize=5.6)
        ax.set_title(title, fontsize=6.3, loc="left")
        ax.grid(color="#E5E7EB", lw=0.4)
        for sp in ["top"]:
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("Models made unavailable (% of T1 models)")
    handles = [Line2D([0], [0], marker=mk, ls="", mfc="white", mec=col, ms=4, label=lab) for _, lab, col, mk in STRAT]
    handles += [Line2D([0], [0], color="#9AA3AD", lw=0.9, marker="_", ms=6, label="size-matched random (mean, 5–95%)"),
                Line2D([0], [0], color="#1F2328", lw=0.8, ls=(0, (3, 1.5)), marker=".", ms=3, label="greedy (reference)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=0.9)
    fig.text(0.5, 0.093, "marker size = removal budget k (1, 5, 10, 50, 100)", ha="center", fontsize=5.2, color="#57606A")
    fig.tight_layout(rect=(0, 0.115, 1, 1), pad=0.3)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    probs = []
    leg = fig.legends[0].get_window_extent(r)
    for ax in axes:
        if leg.overlaps(ax.get_window_extent(r)):
            probs.append("legend overlaps an axes")
        for t in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            bb = t.get_window_extent(r)
            if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1:
                probs.append(f"outside: {t.get_text()[:30]}")
    for t in fig.texts:
        if t.get_window_extent(r).overlaps(leg):
            probs.append("note overlaps legend")
    if probs:
        raise SystemExit("LAYOUT CHECK FAILED:\n  " + "\n  ".join(probs))
    out = ROOT / "04_results" / "figures" / "fig_rq2_contrast"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    Image.open(out.with_suffix(".png")).convert("L").save(out.parent / "fig_rq2_contrast_gray.png")
    print("layout check passed ->", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
