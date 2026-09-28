"""Fig. 1: how a removal shock propagates to descendants in one real model family (conceptual figure).

No hypothesis results (functional option loss rates) are shown, only structure (descendant counts).
Boxes have fixed sizes and are placed on a slot grid so nothing overlaps.

Output: 04_results/figures/fig1_lineage_example.{pdf,png}

Usage
    python fig_lineage_example.py ../../02_data/processed/2026-09-25
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType (IEEE PDF eXpress rejects Type 3)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code" / "03_simulation"))
from removal_sim import add_common_args, load_inputs  # noqa: E402

BASE = "meta-llama/Llama-3.1-8B"
REMOVED = "meta-llama/Llama-3.1-8B-Instruct"
N_SHOW_L1, N_SHOW_L2 = 2, 3
# Display names: model IDs are unchanged and only shortened in the figure; the caption gives the full IDs
DISPLAY = {
    "meta-llama/Llama-3.1-8B": ("Llama-3.1-8B", "meta-llama"),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1-8B-Instruct", "meta-llama"),
    "NousResearch/Hermes-3-Llama-3.1-8B": ("Hermes-3", "NousResearch"),
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit": ("Base 4-bit", "unsloth"),
    "unsloth/Meta-Llama-3.1-8B-Instruct": ("Instruct re-upload", "unsloth"),
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": ("Instruct 4-bit", "unsloth"),
    "arcee-ai/Llama-3.1-SuperNova-Lite": ("SuperNova-Lite", "arcee-ai"),
}
LS = {"finetune": "-", "quantized": (0, (3, 1.5)), "adapter": ":", "merge": "-.", "mirror": "--"}
HIT_FC, HIT_EC = "#F4C7BE", "#A8321F"     # unavailable: darker fill, thicker edge (for grayscale print)
OK_FC, OK_EC = "#FFFFFF", "#6B7785"       # unaffected: white fill, thin gray edge
INK, MUTED = "#1F2328", "#57606A"
# Coordinates are in inches (IEEE column width 3.5in).
# With 7.5/7pt text (matching the 8pt caption) in two-line boxes, three fixed columns would exceed 3.5in.
# So rows are staggered instead: the removed node sits just left of the generation-2 column (using the empty
# slots above the root), and the root's other children go in rows 4-6 below the generation-2 boxes.
FIG_W = 3.5
GAP = 0.12
W_BASE, W_MID, W_RIGHT = 1.27, 1.36, 1.36  # box widths: root / generation 1 / generation 2 (longest line + side padding)
X_BASE = W_BASE / 2                        # root center
X_RIGHT = FIG_W - W_RIGHT / 2              # generation 2 (right edge)
X_MID = FIG_W - W_MID / 2                  # root's other children: below generation 2, right-aligned (diagonal arrows from root)
X_REM = FIG_W - W_RIGHT - GAP - W_MID / 2  # removed node (left of generation 2)
BH, DY = 0.36, 0.44                        # box height, slot spacing
FS_TITLE, FS_SUB, FS_NOTE = 7.5, 7.0, 7.0


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    eco, _ = load_inputs(args)
    idx = {m: i for i, m in enumerate(eco.ids)}
    e = pd.read_parquet(args.processed / "edges_all.parquet", columns=["parent_id", "child_id", "relation", "in_cycle"])
    e = e[~e["in_cycle"]]
    desc = lambda mid: len(eco.reach([idx[mid]])) - 1

    def children(p):
        k = e[e["parent_id"] == p][["child_id", "relation"]].drop_duplicates("child_id")
        k = k[k["child_id"].isin(idx)].copy()
        k["desc"] = [desc(c) for c in k["child_id"]]
        return k.sort_values("desc", ascending=False)

    base_desc = desc(BASE)
    l1 = children(BASE)
    rem = l1[l1["child_id"] == REMOVED].iloc[0]
    others1 = l1[l1["child_id"] != REMOVED]
    show1, rest1 = others1.head(N_SHOW_L1), others1.iloc[N_SHOW_L1:]
    l2 = children(REMOVED)
    show2, rest2 = l2.head(N_SHOW_L2), l2.iloc[N_SHOW_L2:]
    rem_total = int(rem["desc"]) + 1

    # ------------------------------------------------ layout (slot 0 = top)
    nodes, edges = {}, []

    def node(key, x, w, slot, lines, hit, bold=False):
        nodes[key] = dict(x=x, w=w, y=-slot * DY, lines=lines, hit=hit, bold=bold)

    def label(mid, n):
        name, org = DISPLAY.get(mid, (mid.split("/", 1)[1], mid.split("/", 1)[0]))
        return [name, f"{org} · n = {n:,}"]

    for j, (_, r) in enumerate(show2.iterrows()):
        node(f"a{j}", X_RIGHT, W_RIGHT, j, label(r["child_id"], int(r["desc"])), True)
        edges.append(("rem", f"a{j}", r["relation"], True))
    # descendant sets overlap (merges, quantized mirrors, ...), so count the union rather than the sum
    n_rest2 = len(eco.reach([idx[c] for c in rest2["child_id"]]))
    node("a_rest", X_RIGHT, W_RIGHT, 3, [f"+{len(rest2):,} other children", f"n = {n_rest2:,} incl. subtrees"], True)
    edges.append(("rem", "a_rest", "finetune", True))
    node("rem", X_REM, W_MID, 1.5, label(REMOVED, int(rem["desc"])), True, bold=True)
    edges.append(("base", "rem", rem["relation"], True))
    node("base", X_BASE, W_BASE, 3.0, label(BASE, base_desc), False, bold=True)
    # the root's other children overlap the generation-2 column horizontally, so they start at row 4 below a_rest (row 3)
    for j, (_, r) in enumerate(show1.iterrows()):
        node(f"b{j}", X_MID, W_MID, 4 + j, label(r["child_id"], int(r["desc"])), False)
        edges.append(("base", f"b{j}", r["relation"], False))
    node("b_rest", X_MID, W_MID, 6, [f"+{len(rest1):,} other children", "all relation types"], False)
    edges.append(("base", "b_rest", "finetune", False))

    # ------------------------------------------------ drawing
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    y_top, y_bot = BH / 2 + 0.03, -6 * DY - BH / 2 - 0.03
    fig = plt.figure(figsize=(FIG_W, (y_top - y_bot) * FIG_W / (FIG_W + 0.04)))
    ax = fig.add_axes([0, 0, 1, 1])
    M = 0.02                                   # side margin so box edges are not clipped
    ax.set_xlim(-M, FIG_W + M); ax.set_ylim(y_bot, y_top); ax.set_aspect("equal"); ax.axis("off")
    for a, b, rel, hit in edges:
        A, B = nodes[a], nodes[b]
        src = (A["x"] + A["w"] / 2, A["y"])
        if B["x"] - B["w"] / 2 >= src[0] + 0.05:            # child to the right: enter at its left edge
            dst = (B["x"] - B["w"] / 2, B["y"])
        else:                                                # child overlapping above: enter at its bottom edge
            dst = (B["x"], B["y"] - BH / 2)
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="-|>", lw=0.75, linestyle=LS.get(rel, "-"),
                                    mutation_scale=5, shrinkA=0, shrinkB=0.5,
                                    color=HIT_EC if hit else OK_EC), zorder=1)
    texts = []
    for k, n in nodes.items():
        w = n["w"]
        n["box"] = (n["x"] - w / 2, n["y"] - BH / 2, w, BH)
        ax.add_patch(FancyBboxPatch(n["box"][:2], w, BH, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=HIT_FC if n["hit"] else OK_FC, ec=HIT_EC if n["hit"] else OK_EC,
                                    lw=1.4 if k == "rem" else (0.8 if n["hit"] else 0.6), zorder=2))
        t1 = ax.text(n["x"], n["y"] + 0.078, n["lines"][0], ha="center", va="center", color=INK,
                     fontsize=FS_TITLE, fontweight="bold" if n["bold"] else "normal", zorder=3)
        t2 = ax.text(n["x"], n["y"] - 0.085, n["lines"][1], ha="center", va="center", color=MUTED,
                     fontsize=FS_SUB, zorder=3)
        texts += [(k, t1), (k, t2)]
    r = nodes["rem"]
    rm_lab = ax.text(r["x"], r["y"] + BH / 2 + 0.05, "✕  removed", ha="center", va="bottom",
                     color=HIT_EC, fontsize=FS_NOTE, fontweight="bold")
    # the note and legend go in the free slots below the root (rows 3.4-6.4); boxes and arrows occupy the rest
    pct = 100 * rem_total / (base_desc + 1)
    note = ax.text(X_BASE, -4.25 * DY, f"1 removal → {rem_total:,}\nmodels unavailable\n({pct:.0f}% of the family)",
                   ha="center", va="center", color=HIT_EC, fontsize=FS_NOTE, linespacing=1.35)
    handles = [Line2D([0], [0], color=INK, lw=0.75, ls=LS["finetune"], label="fine-tune"),
               Line2D([0], [0], color=INK, lw=0.75, ls=LS["quantized"], label="quantized"),
               Line2D([0], [0], marker="s", ls="", ms=6.5, mfc=HIT_FC, mec=HIT_EC, mew=0.8, label="unavailable"),
               Line2D([0], [0], marker="s", ls="", ms=6.5, mfc=OK_FC, mec=OK_EC, mew=0.6, label="unaffected")]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(X_BASE, y_bot + 0.03),
                    bbox_transform=ax.transData, frameon=False, fontsize=FS_SUB, handlelength=1.8,
                    labelspacing=0.3, borderpad=0, title="n = descendants", title_fontsize=FS_SUB)

    # ------------------------------------------------ automatic layout check: text inside boxes, no overlapping elements
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    def ext(artist):
        bb = artist.get_window_extent(rend).transformed(inv)
        return bb.x0, bb.y0, bb.x1, bb.y1
    problems = []
    for k, t in texts:
        x0, y0, x1, y1 = ext(t)
        bx, by, bw, bh = nodes[k]["box"]
        if x0 < bx + 0.02 or x1 > bx + bw - 0.02 or y0 < by or y1 > by + bh:
            problems.append(f"text overflows box '{k}': '{t.get_text()}'")
    boxes = [nodes[k]["box"] for k in nodes]
    free = {"note": ext(note), "legend": ext(leg), "removed label": ext(rm_lab)}
    for name, (x0, y0, x1, y1) in free.items():
        if x0 < -0.02 or x1 > FIG_W + 0.02 or y0 < y_bot or y1 > y_top:
            problems.append(f"{name} outside figure")
        for k, (bx, by, bw, bh) in zip(nodes, boxes):
            if x0 < bx + bw and x1 > bx and y0 < by + bh and y1 > by:
                problems.append(f"{name} overlaps box '{k}'")
    # box-box and free-text overlaps
    keys = list(nodes)
    for i, ki in enumerate(keys):
        bx, by, bw, bh = nodes[ki]["box"]
        for kj in keys[i + 1:]:
            cx, cy, cw, ch = nodes[kj]["box"]
            if bx < cx + cw and bx + bw > cx and by < cy + ch and by + bh > cy:
                problems.append(f"box '{ki}' overlaps box '{kj}'")
    names = list(free)
    for i, ni in enumerate(names):
        x0, y0, x1, y1 = free[ni]
        for nj in names[i + 1:]:
            u0, v0, u1, v1 = free[nj]
            if x0 < u1 and x1 > u0 and y0 < v1 and y1 > v0:
                problems.append(f"{ni} overlaps {nj}")
    if problems:
        raise SystemExit("LAYOUT CHECK FAILED:\n  " + "\n  ".join(problems))
    print("layout check passed: all text inside boxes, no overlaps")

    out = ROOT / "04_results" / "figures"
    fig.savefig(out / "fig1_lineage_example.pdf")
    fig.savefig(out / "fig1_lineage_example.png", dpi=300)
    from PIL import Image
    Image.open(out / "fig1_lineage_example.png").convert("L").save(out / "fig1_lineage_example_gray.png")
    print("saved", out / "fig1_lineage_example.png", f"({FIG_W:.2f} x {y_top - y_bot:.2f} in)",
          "| removed subtree:", rem_total, "of", base_desc + 1)


if __name__ == "__main__":
    main()
