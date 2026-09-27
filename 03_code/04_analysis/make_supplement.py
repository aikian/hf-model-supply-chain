"""보충 자료(supplement.tex)용 LaTeX 표를 결과 파일에서 만든다. 결과가 바뀌면 다시 돌린다.

출력: 05_paper_tse/supplement_generated/*.tex
사용: python make_supplement.py
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "04_results" / "tables"
OUT = ROOT / "05_paper_tse" / "supplement_generated"
S = "2026-09-25"
KS = [1, 5, 10, 50, 100]
NAMES = {"descendants": "Descendants", "descendant_authors": "Desc.\\ accounts", "downloads": "Downloads",
         "outdegree": "Out-degree"}


def d(name):
    return TAB / f"{S}_full_{name}"


def js(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def planned_table():
    lines = []
    for tier, name in [("T1-original", "planned"), ("T0", "planned_T0")]:
        t = pd.read_csv(d(name) / "planned_tests.csv")
        r = pd.read_csv(d(name) / "removal_results.csv")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{tier}}}}} \\")
        for _, x in t.iterrows():
            g = r[r.k == x.k]
            tgt = g[g.strategy == x.strategy].models_lost.iat[0]
            null = g[g.strategy == ("random" if x.hyp == "H2a" else "random_matched")].models_lost.mean()
            star = "$^{*}$" if x.reject else ""
            lines.append(f"{x.hyp} & {int(x.k)} & {NAMES[x.strategy]} & {x.observed:.3f} & {x.null_mean:.3f} & "
                         f"{x.p_holm:.3f}{star} & {x.cliffs_delta:.2f} & "
                         f"{(f'{null / tgt:.3f}' if null / tgt >= 0.01 else f'{null / tgt:.4f}')} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    v1, v0 = js(d("planned") / "planned_verdicts.json"), js(d("planned_T0") / "planned_verdicts.json")
    summary = []
    for lab, v in [("T1-original", v1), ("T0", v0)]:
        h1 = v["H1"]
        h3 = ", ".join(f"{esc(c)} {r:.2f}" for c, r in v["H3"]["ratios_including_own_class"].items())
        summary.append(f"{lab} & {100 * h1['share_single_lineage']:.2f}\\% [{100 * h1['ci95'][0]:.2f}, "
                       f"{100 * h1['ci95'][1]:.2f}] & {v['H2a']['k_significant']}/5 & {v['H2b']['k_significant']}/5 & "
                       f"{'supported' if v['H3']['supported'] else 'not supported'} ({h3}) \\\\")
    return "\n".join(lines), "\n".join(summary)


def null_table():
    rows = []
    main = pd.read_csv(d("main_notest") / "removal_results.csv")
    h2 = pd.read_csv(d("main_notest") / "h2_tests.csv")
    for sem in ["legal", "availability"]:
        ratios = []
        for (k, s), g in main[main.strategy == "random_matched"].groupby(["k", "matched_to"]):
            if g.semantics.iat[0] != sem:
                continue
        sub = main[(main.strategy == "random_matched") & (main.semantics == sem)]
        tgt = main[(main.semantics == sem) & main.strategy.isin(NAMES)].set_index(["k", "strategy"]).removed_providers
        ratio = sub.removed_providers.to_numpy() / tgt.loc[list(zip(sub.k, sub.matched_to))].to_numpy()
        g = h2[h2.semantics == sem]
        ks = int(g.groupby("k")["reject"].any().sum())
        rows.append(("95\\% (main, 1,000 runs)", sem, ks, g.p.min(), np.median(ratio), np.quantile(ratio, 0.05),
                     np.quantile(ratio, 0.95), ratio.max()))
    for setting, lab in [("tol90", "90\\%"), ("tol99", "99\\%"), ("nearest", "nearest")]:
        f = d(f"null_{setting}") / "null_results.csv"
        if not f.exists():
            rows.append((lab, "pending", None, None, None, None, None, None))
            continue
        res = pd.read_csv(f)
        tests = pd.read_csv(d(f"null_{setting}") / "null_tests.csv")
        for sem in ["legal", "availability"]:
            n = res[(res.semantics == sem) & res.matched_to.notna()]
            g = tests[tests.semantics == sem]
            ks = int(g.groupby("k")["reject"].any().sum())
            rows.append((f"{lab} (500 runs)", sem, ks, g.p.min(), n.size_ratio.median(), n.size_ratio.quantile(0.05),
                         n.size_ratio.quantile(0.95), n.size_ratio.max()))
    out = []
    for lab, sem, ks, p, med, lo, hi, mx in rows:
        if ks is None:
            out.append(f"{lab} & \\multicolumn{{6}}{{l}}{{(pending)}} \\\\")
        else:
            out.append(f"{lab} & {sem} & {ks}/5 & {p:.3f} & {med:.3f} & {lo:.2f}--{hi:.2f} & {mx:.1f} \\\\")
    return "\n".join(out)


def mde_table():
    t = pd.read_csv(d("main_notest") / "detectable_effects.csv")
    out = []
    for sem in ["legal", "availability"]:
        for k in KS:
            g = t[(t.semantics == sem) & (t.k == k)]
            out.append(f"{sem} & {k} & {g.null_mean.min():.1f}--{g.null_mean.max():.1f} & "
                       f"{g.min_detectable_excess.min():.1f}--{g.min_detectable_excess.max():.1f} & "
                       f"{g.observed_minus_null.max():.1f} \\\\")
    return "\n".join(out)


def diff_table():
    """본 분석의 H2 검정별: 관측 손실, 대조군 평균, 차이, 대조군 95% 구간, 제거 규모 비율 중앙값 (옵션 수 단위)."""
    r = pd.read_csv(d("main_notest") / "removal_results.csv")
    crit = pd.read_csv(d("main_notest") / "detectable_effects.csv").set_index(["semantics", "k", "strategy"]).critical
    out = []
    for sem in ["legal", "availability"]:
        for k in KS:
            g = r[(r.semantics == sem) & (r.k == k)]
            for s in NAMES:
                t = g[g.strategy == s].iloc[0]
                n = g[(g.strategy == "random_matched") & (g.matched_to == s)]
                lo, hi = n.options_lost.quantile([0.025, 0.975])
                ratio = (n.removed_providers / t.removed_providers).median() if t.removed_providers else np.nan
                out.append(f"{sem} & {k} & {NAMES[s]} & {int(t.options_lost)} & {n.options_lost.mean():.1f} & "
                           f"{t.options_lost - n.options_lost.mean():+.1f} & {lo:.0f}--{hi:.0f} & "
                           f"{int(crit.loc[(sem, k, s)])} & {ratio:.2f} \\\\")
    return "\n".join(out)


def rq1_table():
    def sh(s):
        s = s[s.n_root_lineages > 0]
        return 100 * (s.n_root_lineages == 1).mean(), len(s)
    m = pd.read_csv(d("main_notest") / "substitutability.csv")
    lic = m.option.str.split("|").str[-1]
    rows = [("Main (T1-main, wildcard)", *sh(m)),
            ("Isolated models not counted", 100 * (m[m.n_root_lineages_known > 0].n_root_lineages_known == 1).mean(),
             int((m.n_root_lineages_known > 0).sum())),
            ("Architecture-grouped roots", 100 * (m[m.n_root_lineages_arch > 0].n_root_lineages_arch == 1).mean(),
             int((m.n_root_lineages_arch > 0).sum())),
            ("Known license class only", *sh(m[lic != "unknown"])),
            ("Commercial-compatible classes only",
             *sh(m[lic.isin(["permissive", "copyleft", "responsible_ai", "vendor_custom"])]))]
    for name, lab in [("declared", "Declared edges only"), ("noquant", "No quantized providers"),
                      ("tierT0", "Tier T0"), ("tierT2", "Tier T2-original"), ("coarse", "Coarse (task $\\times$ license)"),
                      ("strictlang", "Strict language (T1-original)")]:
        rows.append((lab, *sh(pd.read_csv(d(name) / "substitutability.csv"))))
    sm = TAB / f"{S}_subst_T1main_strict" / "substitutability.csv"
    if sm.exists():
        st = pd.read_csv(sm)
        rows.append(("Strict language (T1-main)", *sh(st)))
        rows.append(("Strict language, untagged models excluded (T1-main)", *sh(st[st.option.str.split("|").str[1] != "unk"])))
    return "\n".join(f"{a} & {b:.2f}\\% & {c:,} \\\\" for a, b, c in rows)


def license_table():
    lm = pd.read_csv(ROOT / "03_code" / "03_simulation" / "license_map.csv").fillna("")
    # 분석에서 쓰는 정의 (removal_sim.COMMERCIAL_CLASSES) 를 따른다. license_map.csv 의 commercial 열은
    # 라이선스 원문의 허용 여부라서 no-derivatives 가 yes 로 되어 있다 — 분석 정의와 다르므로 쓰지 않는다.
    import sys
    sys.path.insert(0, str(ROOT / "03_code" / "03_simulation"))
    from removal_sim import COMMERCIAL_CLASSES
    return "\n".join(f"\\texttt{{{esc(r.license)}}} & {esc(r['class'])} & "
                     f"{'yes' if r['class'] in COMMERCIAL_CLASSES else 'no'} \\\\" for _, r in lm.iterrows())


def company_table():
    src = (ROOT / "03_code" / "04_analysis" / "fig_upstream_concentration.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "COMPANY")
    comp = ast.literal_eval(node.value)
    orgs = pd.read_csv(TAB / "upstream_orgs.csv")
    by = {}
    for acct, c in comp.items():
        by.setdefault(c, []).append(acct)
    out = []
    for _, r in orgs.iterrows():
        accts = ", ".join(f"\\texttt{{{esc(a)}}}" for a in by.get(r.company, [r.company]))
        out.append(f"{esc(r.company)} & {accts} & {r.pct_derived:.2f} & {r.cum_union_pct:.2f} \\\\")
    return "\n".join(out)


def validation_table():
    v = js(ROOT / "02_data" / "validation" / "validation_agreement.json")
    names = {"f_bot": "Automated upload", "f_course": "Course assignment", "f_boilerplate": "Boilerplate card",
             "f_test": "Test name", "kept_T1": "Retained (genuine)"}
    out = [f"{names[r['stratum']]} & {r['n']} & {r['A_correct']} ({r['A_pct']}\\%, {r['A_ci95']}) & "
           f"{r['B_correct']} ({r['B_pct']}\\%, {r['B_ci95']}) \\\\" for r in v["by_stratum"]]
    ag = v["agreement"]
    return "\n".join(out), (f"{ag['percent_agree']}\\% on the {ag['n_both_decidable']} models that both labelers "
                            f"could decide; Cohen's $\\kappa$ = {ag['kappa_binary']} (binary) and "
                            f"{ag['kappa_3cat_all']} (three categories, all 300).%")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    body, summ = planned_table()
    parts = {"planned_tests": body, "planned_summary": summ, "null_sensitivity": null_table(),
             "mde": mde_table(), "h2_differences": diff_table(), "rq1_sensitivity": rq1_table(), "license_map": license_table(),
             "companies": company_table()}
    parts["validation"], agree = validation_table()
    parts["validation_agreement"] = agree
    for k, v in parts.items():
        (OUT / f"{k}.tex").write_text(f"% generated by make_supplement.py\n{v}\n", encoding="utf-8")
    print("wrote", ", ".join(parts))


if __name__ == "__main__":
    main()
