"""표: 정제 전(T0)과 후(T1)의 '직계 자식이 많은 부모 모델' 순위 비교 → 04_results/tables/cleaning_effect_<snap>.tex

분석 그래프와 같은 엣지 기준(스냅샷 안의 부모, 순환·시간 역전 제외)을 쓴다.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOP = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()
    P, snap = args.processed, args.processed.name
    e = pd.read_parquet(P / "edges_all.parquet")
    e = e[e["parent_in_snapshot"] & ~e["in_cycle"] & e["temporal_ok"].astype("boolean").fillna(True).astype(bool)]
    f = pd.read_parquet(P / "model_flags.parquet", columns=["model_id", "in_T1"])
    t1 = set(f.loc[f["in_T1"], "model_id"])

    c0 = e.groupby("parent_id").size().rename("t0")
    c1 = e[e["child_id"].isin(t1)].groupby("parent_id").size().rename("t1")
    d = pd.concat([c0, c1], axis=1).fillna(0).astype(int)
    d["rank0"] = d["t0"].rank(ascending=False, method="min").astype(int)
    d["rank1"] = d["t1"].rank(ascending=False, method="min").astype(int)
    d = d.sort_values("t0", ascending=False).head(TOP)
    d["removed_pct"] = 100 * (1 - d["t1"] / d["t0"])
    d.to_csv(ROOT / "04_results" / "tables" / f"cleaning_effect_{snap}.csv")

    # iterrows 는 행을 float 로 바꾸므로 열을 직접 순회하고 int 로 표기한다.
    # 단 폭(3.5in)에 맞추려고 조직명은 빼고 모델명만 쓴다 (캡션에 명시). sdxl 은 이름이 길어 줄인다.
    short = {"stabilityai/stable-diffusion-xl-base-1.0": "stable-diffusion-xl-base"}
    rows = []
    for pid, t0, r0, t1, r1, rm in zip(d.index, d["t0"], d["rank0"], d["t1"], d["rank1"], d["removed_pct"]):
        name = short.get(pid, pid.split("/", 1)[1])
        rows.append(f"\\texttt{{{name}}} & {int(t0):,} ({int(r0)}) & {int(t1):,} ({int(r1)}) & {rm:.0f}\\% \\\\")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Parents with the most direct children before (T0) and after (T1-original) cleaning, "
        r"with ranks in parentheses. Several apparent hubs are artifacts of automated "
        r"reward-scheme uploads. Organization prefixes are omitted.}",
        r"\label{tab:cleaning}", r"\begin{tabular}{lrrr}", r"\toprule",
        r"Parent model & T0 (rank) & T1-orig. (rank) & Removed \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "04_results" / "tables" / f"cleaning_effect_{snap}.tex").write_text(tex, encoding="utf-8")
    print(d.to_string())


if __name__ == "__main__":
    main()
