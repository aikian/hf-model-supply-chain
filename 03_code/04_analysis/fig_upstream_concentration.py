"""Fig. 2: 파생 모델의 상위 공급 조직 집중도 (기술 통계. 가설 검정 결과 아님).

파생 모델 = T1 제공자 중 조상이 1개 이상 있는 모델.
공급 조직 = 그 모델의 루트 계보를 올린 조직. 같은 회사의 HF 계정은 COMPANY 표로 묶는다.
한 모델이 여러 회사 계보를 가질 수 있으므로(병합) 막대 합은 100%를 넘을 수 있다.
누적선은 합집합 비율이다.

출력: 04_results/figures/fig2_upstream_concentration.{pdf,png}, 04_results/tables/upstream_orgs.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType 내장 (IEEE PDF eXpress는 Type 3 거부)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code" / "03_simulation"))
from removal_sim import add_common_args, load_inputs  # noqa: E402
from substitutability import model_roots  # noqa: E402

# 같은 회사의 HF 계정 (HF가 이전 저장소를 조직 계정으로 옮기면서 생긴 별칭 포함)
COMPANY = {
    "meta-llama": "Meta", "facebook": "Meta", "FacebookAI": "Meta",
    "google": "Google", "google-bert": "Google", "google-t5": "Google",
    "openai": "OpenAI", "openai-community": "OpenAI",
    "Qwen": "Alibaba Qwen", "mistralai": "Mistral AI", "microsoft": "Microsoft",
    "black-forest-labs": "Black Forest Labs", "stabilityai": "Stability AI",
    "deepseek-ai": "DeepSeek", "distilbert": "Hugging Face", "HuggingFaceTB": "Hugging Face",
    "lerobot": "Hugging Face", "nvidia": "NVIDIA", "unsloth": "Unsloth (re-uploads)",
}
TOP = 12
BAR, LINE, INK, MUTED = "#4C6A92", "#B03A2E", "#1F2328", "#57606A"


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    eco, _ = load_inputs(args)
    mm, rr = model_roots(eco)
    mr = pd.DataFrame({"m": mm, "root": rr})
    mr = mr[(mr["m"] != mr["root"]) & eco.provider[mr["m"].to_numpy()]]
    org = pd.Series(eco.ids).str.split("/").str[0].to_numpy()
    mr["company"] = pd.Series(org[mr["root"].to_numpy()]).map(lambda o: COMPANY.get(o, o)).to_numpy()
    mr = mr.drop_duplicates(["m", "company"])
    n_der = mr["m"].nunique()

    share = (mr.groupby("company")["m"].nunique() / n_der * 100).sort_values(ascending=False)
    top = share.head(TOP)
    cum = [100 * mr.loc[mr["company"].isin(top.index[:i + 1]), "m"].nunique() / n_der for i in range(TOP)]
    tab = pd.DataFrame({"company": top.index, "pct_derived": top.values.round(2), "cum_union_pct": np.round(cum, 2)})
    tab.to_csv(ROOT / "04_results" / "tables" / "upstream_orgs.csv", index=False)
    print(tab.to_string(index=False))
    print("derived T1 models:", n_der)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2})
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    y = np.arange(TOP)[::-1]
    ax.barh(y, top.values, color=BAR, height=0.68, zorder=2)
    for yi, v in zip(y, top.values):
        ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", ha="left", fontsize=5.6, color=INK)
    ax.set_yticks(y, top.index, fontsize=6.0)
    ax.set_xlabel("Derived models descending from the company (%)", fontsize=6.0)
    ax.set_xlim(0, max(top.values) * 1.22)
    ax.set_xticks([t for t in range(0, 101, 5) if t <= max(top.values) * 1.22])
    ax.tick_params(axis="x", labelsize=5.8, length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E5E7EB", lw=0.5, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9AA3AD")
    # 누적 합집합 주석: 상위 5·10개
    note = (f"Top 5 companies: {cum[4]:.0f}% of derived models\n"
            f"Top 10 companies: {cum[9]:.0f}%\n(n = {n_der:,} derived models)")
    txt = ax.text(0.97, 0.06, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
                  color=LINE, linespacing=1.35)
    fig.tight_layout(pad=0.2)

    # 레이아웃 검사: 막대 값 라벨이 축 범위 안, 주석이 막대·값 라벨과 겹치지 않음
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = txt.get_window_extent(r)
    problems = []
    for art in ax.patches + [t for t in ax.texts if t is not txt]:
        bb = art.get_window_extent(r)
        if bb.overlaps(tb):
            problems.append(f"note overlaps {getattr(art, 'get_text', lambda: 'bar')()}")
    fb = fig.bbox
    for t in ax.texts + ax.get_yticklabels() + ax.get_xticklabels() + [ax.xaxis.label, ax.yaxis.label]:
        bb = t.get_window_extent(r)
        if t.get_text() and (bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1):
            problems.append(f"outside figure: {t.get_text()}")
    if problems:
        raise SystemExit("LAYOUT CHECK FAILED:\n  " + "\n  ".join(problems))
    print("layout check passed")
    out = ROOT / "04_results" / "figures"
    fig.savefig(out / "fig2_upstream_concentration.pdf")
    fig.savefig(out / "fig2_upstream_concentration.png", dpi=300)
    from PIL import Image
    Image.open(out / "fig2_upstream_concentration.png").convert("L").save(out / "fig2_upstream_concentration_gray.png")


if __name__ == "__main__":
    main()
