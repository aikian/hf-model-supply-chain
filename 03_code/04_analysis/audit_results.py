"""전체 실행 결과 점검: 빠진 것·꼬인 것·설정 불일치를 찾는다. 문제마다 한 줄씩 출력하고, 마지막에 요약.

점검 항목 (변형마다)
  A 파일      필수 결과 파일이 모두 있는가
  B 설정      run_config.json 의 옵션이 변형 이름과 맞는가 (seeds, k, 변형 플래그)
  C 행 수     (충격, k, 전략) 마다 표적 1행, 규모 맞춤 대조군 = seeds 행, 균등 무작위 = seeds 행, 라이선스 시나리오 5행
  D 중복      같은 (충격, k, 전략, 비교 대상, seed) 가 두 번 나오지 않는가 (이어서 계산한 실행 확인)
  E 결측·범위 핵심 지표에 NaN 없음, 비율 0–100, 손실 ≤ 전체 옵션
  F 대조군 크기 규모 맞춤 대조군의 제거 제공자 수가 표적의 95% 이상인가
  G 단조성    탐욕 손실이 k 에 따라 줄지 않는가, 표적 전략의 제거 수가 k 에 따라 줄지 않는가
  H 검정      h2_tests 가 충격마다 20행, Holm 조정 p ≥ 원 p, 판정 JSON 과 h2_tests 가 일치
  I RQ1       substitutability 의 옵션 수 = 제거 시뮬레이션의 전체 옵션 수, 고립 제외·구조 묶음 ≤ 기본 계보 수
  J 로그      마지막 실행의 세 단계가 모두 exit 0
  K 변형 효과 변형의 제공자 수·옵션 수가 기준과 예상대로 다른가

사용: python audit_results.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "04_results" / "tables"
LOGS = ROOT / "04_results" / "logs"
SNAP = "2026-09-25"
KS = [1, 5, 10, 50, 100]
TARGETED = ["descendants", "descendant_authors", "downloads", "outdegree"]
# 변형 → (장소, seeds, run_config 에서 기대하는 값)
EXPECT = {
    "main": ("laptop", 1000, {}),
    "main_notest": ("laptop", 1000, {"drop_rule": ["f_test"]}),
    "tierT0": ("laptop", 500, {"tier": "T0"}),
    "tierT2": ("laptop", 500, {"tier": "T2"}),
    "declared": ("laptop", 500, {"declared_only": True, "no_inherit": True}),
    "keeptv": ("colab", 500, {"keep_temporal_violations": True}),
    "noquant": ("colab", 500, {"exclude_quantized": True}),
    "coarse": ("colab", 500, {"option": "coarse"}),
    "strictlang": ("colab", 500, {"option": "strict"}),
    "other2unk": ("colab", 500, {"other_as_unknown": True}),
    "strictcom": ("colab", 500, {"strict_commercial": True}),
    "noov": ("colab", 500, {"no_license_overrides": True}),
}
DEFAULTS = {"tier": "T1", "declared_only": False, "no_inherit": False, "no_license_overrides": False,
            "option": "full", "exclude_quantized": False, "other_as_unknown": False,
            "strict_commercial": False, "keep_temporal_violations": False, "drop_rule": []}
FILES = ["removal_results.csv", "removal_summary.csv", "run_config.json", "substitutability.csv",
         "h2_tests.csv", "h3_tests.csv", "hypothesis_verdicts.json"]


def audit(name):
    host, seeds, exp = EXPECT[name]
    d = TAB / f"{SNAP}_full_{name}"
    issues, info = [], {}
    miss = [f for f in FILES if not (d / f).exists()]
    if miss:
        return [f"A missing files: {miss}"], info
    cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8-sig"))
    want = {**DEFAULTS, **exp}
    for k, v in want.items():
        got = cfg.get(k, DEFAULTS.get(k))
        if got != v and not (k == "drop_rule" and got is None and v == []):
            issues.append(f"B config {k}={got!r}, expected {v!r}")
    if cfg.get("seeds") != seeds:
        issues.append(f"B seeds={cfg.get('seeds')}, expected {seeds}")
    if cfg.get("k") != KS:
        issues.append(f"B k={cfg.get('k')}")
    if cfg.get("sample"):
        issues.append(f"B sample={cfg['sample']} (should be full data)")

    r = pd.read_csv(d / "removal_results.csv")
    sims = r[~r.strategy.astype(str).str.startswith("license:")]
    for sem in ["legal", "availability"]:
        for k in KS:
            g = sims[(sims.semantics == sem) & (sims.k == k)]
            for s in TARGETED + ["greedy"]:
                n = int((g.strategy == s).sum())
                if n != 1:
                    issues.append(f"C {sem} k={k} {s}: {n} rows (expected 1)")
            for s in TARGETED:
                n = int(((g.strategy == "random_matched") & (g.matched_to == s)).sum())
                if n != seeds:
                    issues.append(f"C {sem} k={k} matched_to={s}: {n} rows (expected {seeds})")
            n = int((g.strategy == "random").sum())
            if n != seeds:
                issues.append(f"C {sem} k={k} uniform random: {n} rows (expected {seeds})")
    lic = r[r.strategy.astype(str).str.startswith("license:")]
    if len(lic) != 5:
        issues.append(f"C license scenarios: {len(lic)} rows (expected 5)")
    dup = r.duplicated(["semantics", "k", "strategy", "matched_to", "seed"], keep=False)
    if dup.any():
        issues.append(f"D duplicated rows: {int(dup.sum())}")
    for c in ["pct_options_lost", "options_lost", "removed_providers", "pct_downloads_lost",
              "pct_options_degraded90", "demand_weighted_degradation"]:
        if r[c].isna().any():
            issues.append(f"E NaN in {c}: {int(r[c].isna().sum())}")
    for c in ["pct_options_lost", "pct_downloads_lost", "pct_options_degraded90", "pct_commercial_options_lost"]:
        if ((r[c] < 0) | (r[c] > 100)).any():
            issues.append(f"E {c} out of [0,100]")
    tot = r["options_total"].unique()
    if len(tot) != 1:
        issues.append(f"E options_total varies: {tot}")
    if (r.options_lost > r.options_total).any():
        issues.append("E options_lost > options_total")

    worst = 1.0
    for (sem, k, s), g in sims[sims.strategy == "random_matched"].groupby(["semantics", "k", "matched_to"]):
        t = sims[(sims.semantics == sem) & (sims.k == k) & (sims.strategy == s)]["removed_providers"].iat[0]
        if t > 0:
            ratio = g.removed_providers.min() / t
            worst = min(worst, ratio)
    info["matched_min_size_ratio"] = round(worst, 3)
    if worst < 0.95 - 1e-9:
        issues.append(f"F smallest matched null reaches only {worst:.3f} of target size")

    for sem in ["legal", "availability"]:
        gr = sims[(sims.semantics == sem) & (sims.strategy == "greedy")].set_index("k").loc[KS, "pct_options_lost"]
        if (np.diff(gr.to_numpy()) < -1e-12).any():
            issues.append(f"G greedy loss decreases with k ({sem}): {gr.round(3).tolist()}")
        for s in TARGETED:
            rp = sims[(sims.semantics == sem) & (sims.strategy == s)].set_index("k").loc[KS, "removed_providers"]
            if (np.diff(rp.to_numpy()) < 0).any():
                issues.append(f"G {s} removed providers decrease with k ({sem})")

    h2 = pd.read_csv(d / "h2_tests.csv")
    for sem in ["legal", "availability"]:
        n = int((h2.semantics == sem).sum())
        if n != 20:
            issues.append(f"H h2_tests {sem}: {n} rows (expected 20)")
    if (h2.p_holm + 1e-12 < h2.p).any():
        issues.append("H Holm-adjusted p smaller than raw p")
    if (h2.n_null != seeds).any():
        issues.append(f"H n_null != seeds in {int((h2.n_null != seeds).sum())} rows")
    v = json.loads((d / "hypothesis_verdicts.json").read_text(encoding="utf-8-sig"))
    for sem in ["legal", "availability"]:
        ks = int(h2[h2.semantics == sem].groupby("k")["reject"].any().sum())
        if ks != v[f"H2_{sem}"]["k_significant"]:
            issues.append(f"H verdict mismatch {sem}: tests {ks} vs json {v[f'H2_{sem}']['k_significant']}")
    info["min_p"] = float(h2.p.min())

    sub = pd.read_csv(d / "substitutability.csv")
    n_opt = int((sub.n_root_lineages > 0).sum())
    if n_opt != int(tot[0]):
        issues.append(f"I options: substitutability {n_opt} vs removal {int(tot[0])}")
    if (sub.n_root_lineages_known > sub.n_root_lineages).any():
        issues.append("I isolated-excluded lineages > lineages")
    if "n_root_lineages_arch" in sub and (sub.n_root_lineages_arch > sub.n_root_lineages).any():
        issues.append("I arch-grouped lineages > lineages")
    if sub.option.duplicated().any():
        issues.append("I duplicated options in substitutability.csv")

    log = LOGS / f"{SNAP}_{name}_{host}.log"
    if not log.exists():
        issues.append(f"J log missing: {log.name}")
    else:
        txt = log.read_text(encoding="utf-8", errors="replace")
        seg = txt[max(txt.rfind("03_simulation/removal_sim.py"), 0):]
        exits = re.findall(r"exit (-?\d+)", seg)
        if exits != ["0", "0", "0"]:
            issues.append(f"J last run exits {exits} (expected three 0)")
        m = re.search(r"providers ([\d,]+)", seg)
        info["providers"] = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"edges ([\d,]+)", seg)
        info["edges"] = int(m.group(1).replace(",", "")) if m else None
    info["options"] = int(tot[0])
    return issues, info


def main():
    rows, all_issues = [], {}
    for name in EXPECT:
        if not (TAB / f"{SNAP}_full_{name}" / "hypothesis_verdicts.json").exists():
            print(f"{name:12s} NOT FINISHED")
            continue
        issues, info = audit(name)
        all_issues[name] = issues
        rows.append({"variant": name, **info, "issues": len(issues)})
    t = pd.DataFrame(rows).set_index("variant")
    # K: 변형 효과 (기준 = 원래 T1)
    if "main" in t.index:
        b = t.loc["main"]
        checks = {"tierT0": ("providers", ">"), "tierT2": ("providers", "<"), "main_notest": ("providers", ">"),
                  "noquant": ("providers", "<"), "declared": ("edges", "<"), "coarse": ("options", "<")}
        for n, (col, op) in checks.items():
            if n in t.index and t.loc[n, col] is not None:
                ok = t.loc[n, col] > b[col] if op == ">" else t.loc[n, col] < b[col]
                if not ok:
                    all_issues[n].append(f"K {col} {t.loc[n, col]} not {op} main {b[col]}")
        for n in ["keeptv", "strictlang", "other2unk", "strictcom", "noov"]:
            if n in t.index and t.loc[n, "providers"] != b["providers"]:
                all_issues[n].append(f"K providers {t.loc[n, 'providers']} differ from main {b['providers']} (should be equal)")
        t["issues"] = [len(all_issues[n]) for n in t.index]
    print(t.to_string())
    print()
    for n, iss in all_issues.items():
        for i in iss:
            print(f"[{n}] {i}")
    n = sum(len(v) for v in all_issues.values())
    print(f"\n{len(all_issues)} variants audited, {n} issue(s)")


if __name__ == "__main__":
    main()
