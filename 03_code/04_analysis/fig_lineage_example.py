"""Fig. 1: 실제 계보 한 집안에서 제거 충격이 후손으로 전파되는 모습 (개념도).

가설 결과(기능 옵션 손실률)는 넣지 않는다. 보여 주는 것은 구조(후손 수)뿐이다.
박스 크기를 고정하고 칸(slot) 단위로 배치해 겹침이 생기지 않게 한다.

출력: 04_results/figures/fig1_lineage_example.{pdf,png}

사용
    python fig_lineage_example.py ../../02_data/processed/2026-09-25
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType 내장 (IEEE PDF eXpress는 Type 3 거부)
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
# 표시용 이름 (모델 ID는 그대로 두고 그림에서만 줄인다. 캡션에 원래 ID를 밝힌다)
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
HIT_FC, HIT_EC = "#F4C7BE", "#A8321F"     # 사용 불가: 채움이 더 진하고 테두리 굵게 (흑백 인쇄 대비)
OK_FC, OK_EC = "#FFFFFF", "#6B7785"       # 영향 없음: 흰 채움, 얇은 회색 테두리
INK, MUTED = "#1F2328", "#57606A"
# 좌표 단위 = 인치 (IEEE 단 폭 3.5in)
FIG_W = 3.5
COL_W = [1.00, 1.13, 1.13]                 # 열별 박스 폭
GAP = 0.12
XS = [COL_W[0] / 2, COL_W[0] + GAP + COL_W[1] / 2, COL_W[0] + COL_W[1] + 2 * GAP + COL_W[2] / 2]
BH, DY = 0.32, 0.43                        # 박스 높이, 칸 간격
FS_TITLE, FS_SUB, FS_NOTE = 6.0, 5.3, 5.6


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

    # ------------------------------------------------ 배치 (slot 0 = 맨 위)
    nodes, edges = {}, []

    def node(key, col, slot, lines, hit, bold=False):
        nodes[key] = dict(col=col, x=XS[col], y=-slot * DY, lines=lines, hit=hit, bold=bold)

    def label(mid, n):
        name, org = DISPLAY.get(mid, (mid.split("/", 1)[1], mid.split("/", 1)[0]))
        return [name, f"{org} · n = {n:,}"]

    for j, (_, r) in enumerate(show2.iterrows()):
        node(f"a{j}", 2, j, label(r["child_id"], int(r["desc"])), True)
        edges.append(("rem", f"a{j}", r["relation"], True))
    # 후손 집합은 서로 겹친다 (병합, 재배포본의 양자화 등) → 합이 아니라 합집합으로 센다
    n_rest2 = len(eco.reach([idx[c] for c in rest2["child_id"]]))
    node("a_rest", 2, 3, [f"+{len(rest2):,} other children", f"n = {n_rest2:,} incl. subtrees"], True)
    edges.append(("rem", "a_rest", "finetune", True))
    node("rem", 1, 1.5, label(REMOVED, int(rem["desc"])), True, bold=True)
    edges.append(("base", "rem", rem["relation"], True))
    node("base", 0, 3.0, label(BASE, base_desc), False, bold=True)
    for j, (_, r) in enumerate(show1.iterrows()):
        node(f"b{j}", 1, 3.5 + j, label(r["child_id"], int(r["desc"])), False)
        edges.append(("base", f"b{j}", r["relation"], False))
    node("b_rest", 1, 5.5, [f"+{len(rest1):,} other children", "all relation types"], False)
    edges.append(("base", "b_rest", "finetune", False))

    # ------------------------------------------------ 그리기
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    y_top, y_bot = BH / 2 + 0.03, -5.5 * DY - BH / 2 - 0.03
    fig = plt.figure(figsize=(FIG_W, (y_top - y_bot) * FIG_W / (FIG_W + 0.04)))
    ax = fig.add_axes([0, 0, 1, 1])
    M = 0.02                                   # 테두리가 잘리지 않게 좌우 여백
    ax.set_xlim(-M, FIG_W + M); ax.set_ylim(y_bot, y_top); ax.set_aspect("equal"); ax.axis("off")
    for a, b, rel, hit in edges:
        A, B = nodes[a], nodes[b]
        ax.annotate("", xy=(B["x"] - COL_W[B["col"]] / 2, B["y"]), xytext=(A["x"] + COL_W[A["col"]] / 2, A["y"]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.75, linestyle=LS.get(rel, "-"),
                                    mutation_scale=5, shrinkA=0, shrinkB=0.5,
                                    color=HIT_EC if hit else OK_EC), zorder=1)
    texts = []
    for k, n in nodes.items():
        w = COL_W[n["col"]]
        n["box"] = (n["x"] - w / 2, n["y"] - BH / 2, w, BH)
        ax.add_patch(FancyBboxPatch(n["box"][:2], w, BH, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=HIT_FC if n["hit"] else OK_FC, ec=HIT_EC if n["hit"] else OK_EC,
                                    lw=1.4 if k == "rem" else (0.8 if n["hit"] else 0.6), zorder=2))
        t1 = ax.text(n["x"], n["y"] + 0.065, n["lines"][0], ha="center", va="center", color=INK,
                     fontsize=FS_TITLE, fontweight="bold" if n["bold"] else "normal", zorder=3)
        t2 = ax.text(n["x"], n["y"] - 0.08, n["lines"][1], ha="center", va="center", color=MUTED,
                     fontsize=FS_SUB, zorder=3)
        texts += [(k, t1), (k, t2)]
    r = nodes["rem"]
    ax.text(r["x"], r["y"] + BH / 2 + 0.05, "✕  removed", ha="center", va="bottom",
            color=HIT_EC, fontsize=FS_NOTE, fontweight="bold")
    pct = 100 * rem_total / (base_desc + 1)
    note = ax.text(XS[2], -4.75 * DY, f"1 removal →\n{rem_total:,} models unavailable\n({pct:.0f}% of the family)",
                   ha="center", va="center", color=HIT_EC, fontsize=FS_NOTE, linespacing=1.35)
    handles = [Line2D([0], [0], color=INK, lw=0.75, ls=LS["finetune"], label="fine-tune"),
               Line2D([0], [0], color=INK, lw=0.75, ls=LS["quantized"], label="quantized"),
               Line2D([0], [0], marker="s", ls="", ms=5.5, mfc=HIT_FC, mec=HIT_EC, mew=0.8, label="unavailable"),
               Line2D([0], [0], marker="s", ls="", ms=5.5, mfc=OK_FC, mec=OK_EC, mew=0.6, label="unaffected")]
    leg = ax.legend(handles=handles, loc="center", bbox_to_anchor=(XS[0], -4.85 * DY), bbox_transform=ax.transData,
                    frameon=False, fontsize=FS_SUB, handlelength=1.8, labelspacing=0.3, borderpad=0,
                    title="n = descendants", title_fontsize=FS_SUB)

    # ------------------------------------------------ 자동 레이아웃 검사: 글자가 박스 안에 있는가, 요소끼리 겹치는가
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
    for name, art in [("note", note), ("legend", leg)]:
        x0, y0, x1, y1 = ext(art)
        if x0 < -0.02 or x1 > FIG_W + 0.02 or y0 < y_bot or y1 > y_top:
            problems.append(f"{name} outside figure")
        for k, (bx, by, bw, bh) in zip(nodes, boxes):
            if x0 < bx + bw and x1 > bx and y0 < by + bh and y1 > by:
                problems.append(f"{name} overlaps box '{k}'")
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
