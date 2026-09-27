"""RR Stage 1 파일럿 그림/표 (기술 통계만: 가설 검정 결과는 포함하지 않음).

출력 (04_results/figures, 04_results/tables):
    fig_outdegree_ccdf.{pdf,png}      기반 모델의 직계 자식 수 분포 (log-log CCDF), 정제 전(T0) vs 후(T1)
    fig_relations_by_year.{pdf,png}   연도별 새 계보 엣지의 관계 유형 (T1 자식 기준)
    pilot_table_<snap>.tex            논문 파일럿 표 (LaTeX)

사용
    python pilot_figures.py ../../02_data/processed/2026-09-25
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType 내장 (IEEE PDF eXpress는 Type 3 거부)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, GRID = "#1F2328", "#57606A", "#E5E7EB"
# 관계 유형: 흑백에서도 구분되도록 밝기 차이가 큰 순서 + 무늬
REL_ORDER = ["finetune", "adapter", "quantized", "merge", "other"]
REL_LABEL = {"finetune": "fine-tune", "adapter": "adapter", "quantized": "quantized", "merge": "merge", "other": "mirror / other"}
REL_COLOR = {"finetune": "#23395B", "adapter": "#5C7FA8", "quantized": "#A9C1DB", "merge": "#D9824B", "other": "#EFE3D0"}
REL_HATCH = {"finetune": "", "adapter": "", "quantized": "", "merge": "////", "other": ".."}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2, "hatch.linewidth": 0.4})


def check_layout(fig, extra=()):
    """모든 글자(축 제목, 눈금, 범례, 주석)가 그림 안에 있는지, 범례가 데이터와 겹치지 않는지 검사."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    problems = []
    for ax in fig.axes:
        def visible_ticks(axis, lim):
            lo, hi = sorted(lim)
            return [t.label1 for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi]
        arts = (ax.texts + visible_ticks(ax.xaxis, ax.get_xlim()) + visible_ticks(ax.yaxis, ax.get_ylim())
                + [ax.xaxis.label, ax.yaxis.label, ax.title])
        leg = ax.get_legend()
        if leg:
            arts += leg.get_texts()
        for t in arts:
            if not t.get_text() or not t.get_visible():
                continue
            bb = t.get_window_extent(r)
            if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5:
                problems.append(f"outside figure: '{t.get_text()}'")
    for a, b, name in extra:
        if a.get_window_extent(r).overlaps(b.get_window_extent(r)):
            problems.append(name)
    if problems:
        raise SystemExit("LAYOUT CHECK FAILED:\n  " + "\n  ".join(problems))


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    Image.open(path.with_suffix(".png")).convert("L").save(path.parent / f"{path.name}_gray.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()
    P, snap = args.processed, args.processed.name
    figs, tabs = ROOT / "04_results" / "figures", ROOT / "04_results" / "tables"

    nodes = pd.read_parquet(P / "nodes.parquet", columns=["model_id", "created_at", "created_at_legacy"])
    flags = pd.read_parquet(P / "model_flags.parquet", columns=["model_id", "in_T1"])
    edges = pd.read_parquet(P / "edges_all.parquet")
    # 분석 그래프와 같은 기준: 스냅샷 안의 부모, 순환 제외, 시간 역전 엣지 제외 (removal_sim.Ecosystem)
    edges = edges[edges["parent_in_snapshot"] & ~edges["in_cycle"]
                  & edges["temporal_ok"].astype("boolean").fillna(True).astype(bool)]
    t1 = set(flags.loc[flags["in_T1"], "model_id"])
    e1 = edges[edges["child_id"].isin(t1)]
    summary = json.loads((P / "summary.json").read_text(encoding="utf-8"))
    infer = json.loads((P / "inference_report.json").read_text(encoding="utf-8"))

    # ------------------------------------------------ 1) 직계 자식 수 CCDF: T0 vs T1
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for data, lab, st in [(edges, "all models (T0)", dict(color="#A9B4C2", ms=1.6)),
                          (e1, "after cleaning (T1)", dict(color="#23395B", ms=1.6))]:
        deg = data.groupby("parent_id").size().to_numpy()
        vals, counts = np.unique(deg, return_counts=True)            # 값마다 점 하나
        ccdf = 1 - np.concatenate(([0], np.cumsum(counts)[:-1])) / len(deg)
        ax.loglog(vals, ccdf, "o", mew=0, label=f"{lab}, {len(deg):,} parents", **st)
    ax.set_xlabel("Direct children per parent model")
    ax.set_ylabel("P(children ≥ x)")
    ax.grid(color=GRID, lw=0.4, which="major")
    ax.legend(frameon=False, loc="lower left", markerscale=2.5, fontsize=5.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    check_layout(fig)
    save(fig, figs / f"fig_outdegree_ccdf_{snap}")

    # ------------------------------------------------ 2) 연도별 관계 유형 (T1 자식, 부모·자식 모두 실제 업로드 시각)
    created = nodes.set_index("model_id")["created_at"]
    legacy = nodes.set_index("model_id")["created_at_legacy"]
    e = e1[~e1["child_id"].map(legacy)].copy()
    e["year"] = e["child_id"].map(created).dt.year
    first_year = 2023                     # 2022년 모델은 대부분 이관 날짜(2022-03-02)라 연도를 알 수 없음
    e = e[e["year"] >= first_year]
    e["rel"] = e["relation"].where(e["relation"].isin(REL_ORDER[:-1]), "other")
    tab = e.pivot_table(index="year", columns="rel", values="child_id", aggfunc="count").fillna(0)
    tab = tab.reindex(columns=REL_ORDER, fill_value=0) / 1000
    last = created.max()
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    bottom = np.zeros(len(tab))
    xs = np.arange(len(tab))
    for r in REL_ORDER:
        ax.bar(xs, tab[r].to_numpy(), bottom=bottom, width=0.72, color=REL_COLOR[r], hatch=REL_HATCH[r],
               edgecolor="white" if not REL_HATCH[r] else "#8A5A3C", lw=0.3, label=REL_LABEL[r], zorder=2)
        bottom += tab[r].to_numpy()
    labels = [str(y) if y != last.year else f"{y}*" for y in tab.index]
    ax.set_xticks(xs, labels)
    ax.set_ylabel("New lineage edges (thousands)")
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1.2, ncol=1)
    # 각주(2026년은 부분 연도, 2022년 제외 이유)는 캡션에 쓴다
    fig.tight_layout(pad=0.25)
    # 범례가 막대와 겹치지 않는지 검사
    check_layout(fig, extra=[(a, p, f"{n} overlaps a bar") for a, n in [(leg, "legend")]
                             for p in ax.patches if p.get_height() > 0])
    save(fig, figs / f"fig_relations_by_year_{snap}")
    print((tab * 1000).astype(int).to_string())

    # ------------------------------------------------ 3) 파일럿 표
    n_t1 = len(t1)
    child_t1 = e1["child_id"].nunique()
    rows = [
        ("Models in snapshot (T0)", f"{summary['n_models']:,}"),
        ("Models after original cleaning (T1-original)", f"{n_t1:,}"),
        ("Declared lineage edges", f"{summary['n_edges']:,}"),
        ("Inferred edges: undeclared mirrors", f"{infer['inferred_edges'].get('inferred_mirror', 0):,}"),
        ("Inferred edges: parents in repository names", f"{infer['inferred_edges'].get('inferred_name', 0):,}"),
        ("Inferred edges: quantization suffixes", f"{infer['inferred_edges'].get('inferred_quant', 0):,}"),
        ("T1-original models with $\\geq$1 parent", f"{child_t1:,} ({100 * child_t1 / n_t1:.1f}\\%)"),
        ("Declared parents missing from snapshot", f"{100 - summary['pct_edges_parent_in_snapshot']:.2f}\\%"),
        ("Parent uploaded before child", f"{summary['pct_temporal_ok']}\\%"),
        ("Multi-parent (merged) models", f"{summary['multi_parent_models']:,}"),
    ]
    tex = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           f"\\caption{{Lineage graph statistics (snapshot {snap}; provider counts use T1-original).}}",
           "\\label{tab:pilot}", "\\begin{tabular}{lr}", "\\toprule"]
    tex += [f"{a} & {b} \\\\" for a, b in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (tabs / f"pilot_table_{snap}.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(f"{a}: {b}" for a, b in rows))
    print("layout checks passed ->", figs)


if __name__ == "__main__":
    main()
