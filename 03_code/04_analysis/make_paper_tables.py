r"""결과 CSV → 원고용 LaTeX 표와 숫자 매크로. 변형이 끝날 때마다 다시 돌리면 원고 숫자가 갱신된다.

출력: 05_paper_tse/generated/
    numbers.tex      \newcommand 숫자 매크로 (본문에서 \HOneWild 처럼 쓴다)
    tab_rq1.tex      RQ1 표 본문 (tabular 안쪽 행)
    tab_rq2.tex      RQ2 표 본문
    tab_rq3.tex      RQ3 표 본문
    tab_robust.tex   강건성 표 본문
아직 끝나지 않은 변형은 \RES{pending} 으로 남긴다.

사용: python make_paper_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "04_results" / "tables"
OUT = ROOT / "05_paper_tse" / "generated"
SNAP = "2026-09-25"
KS = [1, 5, 10, 50, 100]
STRATS = [("descendants", "Descendants"), ("descendant_authors", "Descendant accounts"),
          ("downloads", "Downloads"), ("outdegree", "Out-degree")]
VARIANTS = [("main_notest", "Main (T1-main)"), ("main", "T1-original (test rule kept)"), ("tierT0", "Tier T0"), ("tierT2", "Tier T2-original"),
            ("declared", "Declared edges only"), ("keeptv", "Keep temporal violations"),
            ("noquant", "No quantized models"), ("coarse", r"Coarse (task $\times$ license)"),
            ("strictlang", "Strict language"), ("other2unk", r"Other $\rightarrow$ unknown"),
            ("strictcom", "Strict commercial"), ("noov", "No license overrides")]
H3_ROWS = [("noncommercial", "Non-commercial"), ("vendor_custom", "Vendor-custom"), None,
           ("no_derivatives", "No-derivatives"), ("responsible_ai", "RAIL"), ("copyleft", "Copyleft")]
PENDING = r"\RES{pending}"
# 주 분석: 사전 결정 규칙(정밀도 80% 미만 규칙 끄기)에 따라 test 규칙을 끈 실행. 끝나기 전에는 원래 T1 을 쓰고 경고한다.
PROVIDERS = {"main": 1_631_600, "main_notest": 1_656_701}


def d(name):
    return TAB / f"{SNAP}_full_{name}"


def done(name):
    return (d(name) / "hypothesis_verdicts.json").exists()


def verdicts(name):
    return json.loads((d(name) / "hypothesis_verdicts.json").read_text(encoding="utf-8-sig"))


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}\\%"


def num(x, nd=2):
    return f"{x:.{nd}f}"


def bins(s):
    s = s[s > 0]
    n = len(s)
    return [(s == 1).sum() / n, s.between(2, 5).sum() / n, s.between(6, 20).sum() / n, (s > 20).sum() / n], n


def rq1(macros):
    sub = pd.read_csv(d(MAIN) / "substitutability.csv")
    cols = [bins(sub["n_root_lineages"])]
    strict_main = TAB / f"{SNAP}_subst_T1main_strict" / "substitutability.csv"     # 같은 모집단(T1-main)의 strict
    if strict_main.exists():
        cols.append(bins(pd.read_csv(strict_main)["n_root_lineages"]))
    elif done("strictlang"):
        cols.append(bins(pd.read_csv(d("strictlang") / "substitutability.csv")["n_root_lineages"]))
    else:
        cols.append(None)
    cols.append(bins(sub["n_root_lineages_known"]))
    cols.append(bins(sub["n_root_lineages_arch"]))
    labels = ["1", "2--5", "6--20", "$>20$"]
    lines = []
    for i, lab in enumerate(labels):
        cells = [pct(c[0][i], 1) if c else PENDING for c in cols]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Options (count) & " + " & ".join(f"{c[1]:,}" if c else PENDING for c in cols) + r" \\")
    v = verdicts(MAIN)["H1"]
    macros.update({
        "nOptions": f"{v['n_options']:,}",
        "HOneWild": pct(v["share_single_lineage"]),
        "HOneStrict": pct(cols[1][0][0]) if cols[1] else PENDING,
        "HOneStrictOriginal": pct(verdicts("strictlang")["H1"]["share_single_lineage"]) if done("strictlang") else PENDING,
        "HOneIso": pct(v["sensitivity_isolated_excluded"]["share_single_lineage"]),
        "nIsoOnly": f"{v['sensitivity_isolated_excluded']['options_only_from_isolated_models']:,}",
        "HOneArch": pct(v["lower_bound_architecture_grouped"]["share_single_lineage"]),
        "HOneVerdict": "supported" if v["supported"] else "not supported",
        "medianLineages": f"{int(np.median(sub.loc[sub.n_root_lineages > 0, 'n_root_lineages'])):,}",
    })
    return "\n".join(lines)


def rq2(macros):
    res = pd.read_csv(d(MAIN) / "removal_results.csv")
    h2 = pd.read_csv(d(MAIN) / "h2_tests.csv")
    lines = []
    for sem, title in [("legal", r"(a) Legal shock (all descendants)"),
                       ("availability", r"(b) Availability shock (adapters without a surviving mirror)")]:
        if sem == "availability":
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for s, lab in STRATS:
            t = h2[(h2.semantics == sem) & (h2.strategy == s)].set_index("k")
            loss = [num(t.loc[k, "observed"], 3) for k in KS]
            rat = []
            for k in KS:
                star = "$^{*}$" if bool(t.loc[k, "reject"]) else ""
                if t.loc[k, "null_mean"] == 0 and t.loc[k, "observed"] == 0:   # 둘 다 0: 배율 정의 안 됨
                    rat.append("--")
                    continue
                rat.append(f"{t.loc[k, 'ratio_vs_matched']:.2f} ({t.loc[k, 'percentile_in_null']:.0f}){star}")
            lines.append(f"{lab} & Loss (\\%) & " + " & ".join(loss) + r" \\")
            lines.append(" & Ratio (pctl) & " + " & ".join(rat) + r" \\")
        g = res[(res.semantics == sem) & (res.strategy == "greedy")].set_index("k")["pct_options_lost"]
        u = res[(res.semantics == sem) & (res.strategy == "random")].groupby("k")["pct_options_lost"].mean()
        lines.append("Greedy (reference) & Loss (\\%) & " + " & ".join(num(g[k], 3) for k in KS) + r" \\")
        lines.append("Uniform random (reference) & Mean loss (\\%) & " + " & ".join(num(u[k], 3) for k in KS) + r" \\")
    v = verdicts(MAIN)
    fin = h2[np.isfinite(h2.ratio_vs_matched) & (h2.null_mean > 0)]
    k100 = h2[h2.k == 100]
    total = int(res["options_total"].iloc[0])
    T1 = PROVIDERS[MAIN]
    for sem, key in [("legal", "Legal"), ("availability", "Avail")]:
        kk = k100[k100.semantics == sem]
        g = res[(res.semantics == sem) & (res.strategy == "greedy") & (res.k == 100)].iloc[0]
        macros[f"HTwo{key}K"] = str(v[f"H2_{sem}"]["k_significant"])
        macros[f"HTwo{key}Verdict"] = "supported" if v[f"H2_{sem}"]["supported"] else "not supported"
        macros[f"Loss{key}KHundredMin"] = pct(kk.observed.min() / 100, 3)
        macros[f"Loss{key}KHundredMax"] = pct(kk.observed.max() / 100, 3)
        macros[f"Greedy{key}KHundred"] = pct(g.pct_options_lost / 100, 3)
        macros[f"GreedyOpts{key}KHundred"] = f"{int(round(g.pct_options_lost / 100 * total)):,}"
    macros["RatioMin"] = num(fin.ratio_vs_matched.min())
    macros["RatioMax"] = num(fin.ratio_vs_matched.max())
    # 대비: 다운로드 상위 100 (법적 충격)
    dl = res[(res.semantics == "legal") & (res.strategy == "downloads") & (res.k == 100)].iloc[0]
    macros["DlHundredModels"] = pct(dl.removed_providers / T1, 1)
    macros["DlHundredDownloads"] = pct(dl.pct_downloads_lost / 100, 1)
    macros["DlHundredOptions"] = pct(dl.pct_options_lost / 100, 3)
    macros["DlHundredDegraded"] = pct(dl.pct_options_degraded90 / 100, 1)
    ds = res[(res.semantics == "legal") & (res.strategy == "descendants") & (res.k == 100)].iloc[0]
    macros["DescHundredModels"] = pct(ds.removed_providers / T1, 1)
    macros["DescHundredOptions"] = pct(ds.pct_options_lost / 100, 3)
    macros["nOptionsTotal"] = f"{total:,}"
    return "\n".join(lines)


def rq3(macros):
    t = pd.read_csv(d(MAIN) / "h3_tests.csv").set_index("class")
    lines = []
    for row in H3_ROWS:
        if row is None:
            lines.append(r"\midrule")
            continue
        c, lab = row
        r = t.loc[c]
        lines.append(f"{lab} & {r.pct_models:.2f} & {r.pct_collateral_options_lost:.3f} & "
                     f"{r.pct_collateral_commercial_lost:.3f} & {r.ratio:.3f}" + r" \\")
    v = verdicts(MAIN)["H3"]
    macros.update({"HThreeNCLost": pct(t.loc["noncommercial", "pct_collateral_commercial_lost"] / 100, 3),
                   "HThreeNCRatio": num(v["ratios"]["noncommercial"], 3),
                   "HThreeVCRatio": num(v["ratios"]["vendor_custom"], 3),
                   "HThreeVerdict": "supported" if v["supported"] else "not supported"})
    return "\n".join(lines)


def robust(macros):
    lines, n_done, flips = [], 0, []
    base = verdicts(MAIN)
    for name, lab in VARIANTS:
        if not done(name):
            lines.append(f"{lab} & " + " & ".join([PENDING] * 4) + r" \\")
            if name == "main_notest":
                lines.append(r"\midrule")
            continue
        n_done += 1
        v = verdicts(name)
        h3 = v["H3"].get("ratios", {})
        cells = [pct(v["H1"]["share_single_lineage"], 1),
                 f"{v['H2_legal']['k_significant']}/5", f"{v['H2_availability']['k_significant']}/5",
                 f"{h3.get('noncommercial', float('nan')):.2f} / {h3.get('vendor_custom', float('nan')):.2f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        for h in ["H1", "H2_legal", "H2_availability", "H3"]:
            if bool(v[h]["supported"]) != bool(base[h]["supported"]):
                flips.append(f"{name}:{h}")
        if name == "main_notest":
            lines.append(r"\midrule")
    macros["nVariantsDone"] = str(n_done - (1 if done(MAIN) else 0))
    macros["nVerdictFlips"] = str(len(flips))
    (OUT / "robust_flips.txt").write_text("\n".join(flips) or "none", encoding="utf-8")
    return "\n".join(lines)


def main():
    global MAIN
    MAIN = "main_notest" if done("main_notest") else "main"
    if MAIN != "main_notest":
        print("WARNING: main_notest not finished — numbers below are from the original T1 run (main)")
    OUT.mkdir(parents=True, exist_ok=True)
    macros = {}
    for fn, f in [("tab_rq1.tex", rq1), ("tab_rq2.tex", rq2), ("tab_rq3.tex", rq3), ("tab_robust.tex", robust)]:
        (OUT / fn).write_text(f"% generated by make_paper_tables.py — do not edit\n{f(macros)}\n", encoding="utf-8")
    macros["mainProviders"] = f"{PROVIDERS[MAIN]:,}"
    body = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items())
    (OUT / "numbers.tex").write_text(f"% generated by make_paper_tables.py — do not edit\n{body}\n", encoding="utf-8")
    for k, v in macros.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
