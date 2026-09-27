"""Fig: RQ2 — 모델 손실 대 기능 손실을 같은 단위(%)의 로그 축에 그린다. 대각선 y = x 는 '모델이 사라지는 만큼 기능도
사라지는' 기준선이고, 점이 대각선에서 아래로 멀수록 기능이 모델보다 덜 사라진다.

x = 사용 불가가 된 제공자 비율 (% of T1-main models, 로그)
y = 제공자가 하나도 남지 않은 기능 옵션 비율 (% of options, 로그). 손실 0 은 아래쪽 띠에 표시.
점: 표적 전략 4개 × k 5개. 채운 점 = 법적 충격(최대 전파), 빈 점 = 가용성 충격. 검은 점선: 탐욕적 참고선.
점 뒤의 연회색 세로선: 같은 x 에 맞춘 대조군(제공자 수 일치 무작위 제거)의 5–95% 범위. 수치 비교는 표 5.

색: dataviz 기준 팔레트 1–4번 (validate_palette.js 통과). 대비가 낮은 청록·노랑은 모양과 범례로 보완.
출력: 04_results/figures/fig_rq2_contrast.{pdf,png} (+ _gray.png)
사용: python fig_rq2_contrast.py ../../04_results/tables/2026-09-25_full_main_notest --providers 1656701
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
STRAT = [("descendants", "descendants", "#2a78d6", "o"), ("descendant_authors", "descendant accounts", "#eb6834", "s"),
         ("downloads", "downloads", "#1baf7a", "^"), ("outdegree", "out-degree", "#eda100", "D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FLOOR = 0.002                       # 손실 0 을 그릴 높이 (% of options)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.4})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--providers", type=int, required=True, help="T1-main 제공자 수")
    args = ap.parse_args()
    d = pd.read_csv(args.result_dir / "removal_results.csv")
    n_prov = args.providers

    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    xs = np.logspace(-5, 2, 50)
    for f, lab in [(1, "option loss = model loss"), (1e-2, "1/100"), (1e-3, "1/1,000")]:
        ax.plot(xs, xs * f, color=MUTED if f == 1 else "#b9b8b2", lw=0.8 if f == 1 else 0.6,
                ls="-" if f == 1 else (0, (2, 2)), zorder=1)
    ax.axhspan(FLOOR / 1.8, FLOOR * 1.8, color="#f0efec", zorder=0)
    ax.text(55, FLOOR, "no option lost", va="center", ha="right", fontsize=5.4, color=MUTED)

    for sem, filled in [("legal", True), ("availability", False)]:
        g = d[d["semantics"] == sem]
        for s, _, col, mk in STRAT:
            o = g[g["strategy"] == s].sort_values("k")
            x = 100 * o["removed_providers"].to_numpy() / n_prov
            y = np.where(o["pct_options_lost"].to_numpy() > 0, o["pct_options_lost"].to_numpy(), FLOOR)
            for kk, xx in zip(o["k"], x):          # 대조군은 제공자 수를 맞췄으므로 같은 x 에 그린다
                n = g[(g["strategy"] == "random_matched") & (g["matched_to"] == s) & (g["k"] == kk)]["pct_options_lost"]
                ax.plot([xx, xx], [max(n.quantile(0.05), FLOOR), max(n.quantile(0.95), FLOOR)], color="#cfcec9",
                        lw=1.4, solid_capstyle="butt", zorder=1.5)
            ax.scatter(x, y, s=22, marker=mk, facecolor=col if filled else "white", edgecolor=col,
                       lw=1.1, zorder=3 if filled else 4)
        gr = g[g["strategy"] == "greedy"].sort_values("k")
        gy = np.where(gr["pct_options_lost"] > 0, gr["pct_options_lost"], FLOOR)
        ax.plot(100 * gr["removed_providers"] / n_prov, gy, color=INK, lw=0.7, ls=(0, (3, 1.5)),
                marker="o", ms=2.6, mfc=INK if filled else "white", mec=INK, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 60)
    ax.set_ylim(FLOOR / 2.2, 6)
    fig.canvas.draw()                                     # 선의 화면상 기울기에 맞춰 이름표를 돌린다
    def label_line(x, f, text, dy=1.7):
        p1 = ax.transData.transform((x, x * f)); p2 = ax.transData.transform((x * 2, x * 2 * f))
        ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        ax.text(x, x * f * dy, text, rotation=ang, rotation_mode="anchor", fontsize=5.5, color=MUTED, zorder=5)
    label_line(4e-3, 1, "option loss = model loss")
    label_line(12, 1e-2, "1/100")
    label_line(12, 1e-3, "1/1,000")
    ax.set_xlabel("Providers unavailable after propagation (% of T1-main)", color=INK)
    ax.set_ylabel("Options with no provider left (% of all options)", color=INK)
    ax.grid(color=GRID, lw=0.4, which="major")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#9a9994")
    ax.tick_params(colors=MUTED, labelcolor=INK)

    handles = [Line2D([0], [0], marker=mk, ls="", mfc=col, mec=col, ms=4.2, label=lab) for _, lab, col, mk in STRAT]
    handles += [Line2D([0], [0], marker="o", ls="", mfc=MUTED, mec=MUTED, ms=4.2, label="filled: legal shock"),
                Line2D([0], [0], marker="o", ls="", mfc="white", mec=MUTED, ms=4.2, label="hollow: availability"),
                Line2D([0], [0], color=INK, lw=0.7, ls=(0, (3, 1.5)), marker="o", ms=2.6, label="greedy (reference)"),
                Line2D([0], [0], color="#cfcec9", lw=1.4, label="matched random, 5-95% range")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.3, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.16, 1, 1), pad=0.3)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb, leg = fig.bbox, fig.legends[0].get_window_extent(r)
    probs = []
    if leg.overlaps(ax.get_window_extent(r)):
        probs.append("legend overlaps the axes")
    for t in [ax.xaxis.label, ax.yaxis.label, *ax.texts]:
        bb = t.get_window_extent(r)
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1:
            probs.append(f"outside: {t.get_text()[:30]}")
    if probs:
        raise SystemExit("LAYOUT CHECK FAILED:\n  " + "\n  ".join(probs))
    out = ROOT / "04_results" / "figures" / "fig_rq2_contrast"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    Image.open(out.with_suffix(".png")).convert("L").save(out.parent / "fig_rq2_contrast_gray.png")
    print("layout check passed ->", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
