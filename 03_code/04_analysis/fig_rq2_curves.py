"""Fig: RQ2 — 표적 제거의 손실 배율 (표적 / 같은 규모 무작위 대조군 평균), 제거 예산 k 별. 충격 종류별 두 패널.

y = 1 이면 표적이 같은 규모의 무작위 제거와 같은 만큼 잃는다. 속이 찬 점 = Holm 보정 후 유의 (H2).
대조군 평균이 0 인데 표적 손실 > 0 이면 배율이 무한대 → 위쪽 경계에 삼각형으로 표시.
절대 손실값은 표(Table)로 보고한다.

입력: <result_dir>/h2_tests.csv (hypothesis_tests.py 출력)
출력: 04_results/figures/fig_rq2_curves.{pdf,png} (+ _gray.png)
사용: python fig_rq2_curves.py ../../04_results/tables/2026-09-25_full_main
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STRAT = [("descendants", "descendants", "#23395B", "o"), ("descendant_authors", "descendant accounts", "#B03A2E", "s"),
         ("downloads", "downloads", "#2E7D5B", "^"), ("outdegree", "out-degree", "#C7862F", "D")]
SEM = [("legal", "Legal shock (all descendants)"), ("availability", "Availability shock (adapters; mirrors substitute)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})   # 눈금·축 이름 7.5pt (캡션 8pt 기준)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    args = ap.parse_args()
    t = pd.read_csv(args.result_dir / "h2_tests.csv")
    finite = t["ratio_vs_matched"].replace([np.inf], np.nan)
    top = max(2.0, float(np.nanmax(finite)) * 1.6) if finite.notna().any() else 2.0
    bottom = min(0.5, float(np.nanmin(finite[finite > 0])) / 1.6) if (finite > 0).any() else 0.5
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    for ax, (sem, title) in zip(axes, SEM):
        g = t[t["semantics"] == sem]
        ks = sorted(g["k"].unique())
        for j, (s, lab, col, mk) in enumerate(STRAT):
            x = g[g["strategy"] == s].set_index("k").reindex(ks)
            xs = np.array(ks, dtype=float) * (1 + 0.06 * (j - 1.5))          # 겹침 방지용 가로 이동
            r = x["ratio_vs_matched"].to_numpy(dtype=float)
            r_plot = np.where(np.isinf(r), top, np.where(r <= 0, bottom, r))
            ax.plot(xs, r_plot, lw=0.8, color=col, zorder=2)
            ax.plot([], [], lw=0.8, color=col, marker=mk, ms=3.2, mfc="white", mec=col, label=lab)   # 범례용
            sig = x["reject"].fillna(False).to_numpy(dtype=bool)
            inf = np.isinf(r)
            zero_ = (r <= 0) & ~np.isnan(r)
            for mask, face in [(sig & ~inf & ~zero_, col), (~sig & ~inf & ~zero_, "white")]:
                ax.scatter(xs[mask], r_plot[mask], marker=mk, s=11, facecolor=face, edgecolor=col, lw=0.8, zorder=3)
            if inf.any():
                ax.scatter(xs[inf], r_plot[inf], marker="^", s=16, facecolor=np.where(sig[inf], col, "white").tolist(),
                           edgecolor=col, lw=0.8, zorder=3)
            zero = (r <= 0) & ~np.isnan(r)
            if zero.any():
                ax.scatter(xs[zero], r_plot[zero], marker="v", s=16, facecolor="white", edgecolor=col, lw=0.8, zorder=3)
        ax.axhline(1, color="#57606A", lw=0.7, ls=(0, (3, 1.5)), zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom / 1.15, top * 1.15)
        ax.set_xticks(ks, [str(k) for k in ks])
        yt = [v for v in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] if bottom / 1.15 <= v <= top * 1.15]
        ax.set_yticks(yt, [f"{v:g}" for v in yt])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=8.0, loc="left")
        ax.grid(color="#E5E7EB", lw=0.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("Removal budget k (upstream models)")
    # y축 이름은 7.5pt에서 패널 하나보다 길어 두 패널이 공유하는 이름 하나로 둔다
    ylab = fig.supylabel("Loss ratio vs. matched random", fontsize=7.5, x=0.0, ha="left")
    # 범례 7pt: 한 줄(ncol=4)로는 3.5in을 넘으므로 두 줄. 주석 7pt도 두 줄로 나눈다.
    # 범례 → 그 위에 주석 → 그 위에 축, 순서로 실측 높이를 재어 쌓는다.
    h, lab = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 0.0),
                     handlelength=1.6, columnspacing=1.5, labelspacing=0.25, borderpad=0, borderaxespad=0.3)
    fig.canvas.draw()
    leg_top = leg.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height
    note = fig.text(0.5, leg_top + 0.01,
                    "filled = significant (Holm)\n▲ top: random lost nothing; ▼ bottom: targeted lost nothing",
                    ha="center", va="bottom", fontsize=7.0, color="#57606A", linespacing=1.3)
    fig.canvas.draw()
    note_top = note.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height
    ylab_right = ylab.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
    fig.tight_layout(rect=(ylab_right + 0.005, note_top + 0.01, 1, 1), pad=0.3)
    # 공유 y축 이름을 두 패널의 세로 중앙에 맞춘다
    ylab.set_y((axes[0].get_position().y1 + axes[1].get_position().y0) / 2)

    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    fb = fig.bbox
    probs = []
    leg = fig.legends[0].get_window_extent(rnd)
    for ax in axes:
        if leg.overlaps(ax.get_window_extent(rnd)):
            probs.append("legend overlaps an axes")
        lo, hi = sorted(ax.get_ylim())
        yt = [tk.label1 for tk in ax.yaxis.get_major_ticks() if lo <= tk.get_loc() <= hi]
        for tx in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title] + yt:
            if tx.get_text() and tx.get_visible():
                bb = tx.get_window_extent(rnd)
                if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
                    probs.append(f"outside: {tx.get_text()}")
    for tx in fig.texts:
        bb = tx.get_window_extent(rnd)
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1:
            probs.append(f"outside: {tx.get_text()[:30]}")
        if bb.overlaps(leg):
            probs.append("note overlaps legend")
    if leg.x0 < fb.x0 - 1 or leg.x1 > fb.x1 + 1 or leg.y0 < fb.y0 - 1:
        probs.append("legend outside figure")
    yb = ylab.get_window_extent(rnd)
    if yb.x0 < fb.x0 - 1 or yb.y0 < fb.y0 - 1 or yb.y1 > fb.y1 + 1:
        probs.append("shared y label outside figure")
    for ax in axes:
        for tk in ax.yaxis.get_major_ticks():
            if tk.label1.get_visible() and tk.label1.get_window_extent(rnd).overlaps(yb):
                probs.append("shared y label overlaps y tick labels")
                break
    if probs:
        raise SystemExit("LAYOUT CHECK FAILED:\n  " + "\n  ".join(probs))
    out = ROOT / "04_results" / "figures" / "fig_rq2_curves"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    Image.open(out.with_suffix(".png")).convert("L").save(out.parent / "fig_rq2_curves_gray.png")
    print("layout check passed ->", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
