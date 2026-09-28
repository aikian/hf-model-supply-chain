"""Fig: RQ2 — model loss vs. functional loss on log axes in the same unit (%). The diagonal y = x is the
'options vanish as fast as models' reference line; the further a point sits below it, the less functionality is lost per model.

x = share of providers made unavailable (% of T1-main models, log)
y = share of functional options with no provider left (% of options, log). Zero loss is drawn in the band at the bottom.
Points: 4 targeted strategies x 5 k. Filled = legal shock (maximal propagation), hollow = availability shock. Black dotted line: greedy reference.
Light gray vertical bar behind each point: 5-95% range of the matched null (random removal with the same provider count) at the same x. Numbers are in Table 5.

Colors: dataviz base palette 1-4 (passes validate_palette.js). The low-contrast teal and yellow are backed up by marker shape and the legend.
Output: 04_results/figures/fig_rq2_contrast.{pdf,png} (+ _gray.png)
Usage: python fig_rq2_contrast.py ../../04_results/tables/2026-09-25_full_main_notest --providers 1656701
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
FLOOR = 0.002                       # height at which zero loss is drawn (% of options)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--providers", type=int, required=True, help="number of T1-main providers")
    args = ap.parse_args()
    d = pd.read_csv(args.result_dir / "removal_results.csv")
    n_prov = args.providers

    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    xs = np.logspace(-5, 2, 50)
    for f, lab in [(1, "option loss = model loss"), (1e-2, "1/100"), (1e-3, "1/1,000")]:
        ax.plot(xs, xs * f, color=MUTED if f == 1 else "#b9b8b2", lw=0.8 if f == 1 else 0.6,
                ls="-" if f == 1 else (0, (2, 2)), zorder=1)
    ax.axhspan(FLOOR / 1.8, FLOOR * 1.8, color="#f0efec", zorder=0)
    ax.text(55, FLOOR, "no option lost", va="center", ha="right", fontsize=6.5, color=MUTED)

    for sem, filled in [("legal", True), ("availability", False)]:
        g = d[d["semantics"] == sem]
        for s, _, col, mk in STRAT:
            o = g[g["strategy"] == s].sort_values("k")
            x = 100 * o["removed_providers"].to_numpy() / n_prov
            y = np.where(o["pct_options_lost"].to_numpy() > 0, o["pct_options_lost"].to_numpy(), FLOOR)
            for kk, xx in zip(o["k"], x):          # the null matches the provider count, so draw it at the same x
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
    fig.canvas.draw()                                     # rotate each label to the on-screen slope of its line
    def label_line(x, f, text, dy=1.7):
        p1 = ax.transData.transform((x, x * f)); p2 = ax.transData.transform((x * 2, x * 2 * f))
        ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        ax.text(x, x * f * dy, text, rotation=ang, rotation_mode="anchor", fontsize=6.5, color=MUTED, zorder=5)
    label_line(4e-3, 1, "option loss = provider loss")
    label_line(30, 1e-2, "1/100")
    label_line(30, 1e-3, "1/1,000")
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
                Line2D([0], [0], color="#cfcec9", lw=1.4, label="matched random, 5–95%")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.25, columnspacing=0.6, handlelength=1.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1), pad=0.3)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb, leg = fig.bbox, fig.legends[0].get_window_extent(r)
    probs = []
    if leg.overlaps(ax.get_window_extent(r)):
        probs.append("legend overlaps the axes")
    if leg.x0 < fb.x0 - 1 or leg.x1 > fb.x1 + 1 or leg.y0 < fb.y0 - 1:
        probs.append("legend outside the figure")
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
