"""최초 계획(첫 전체 실행 전에 정한 분석) 그대로, 구현 오류만 고친 코드와 최종 데이터로 다시 돌린다.

심사 대응: 원고의 RQ1–RQ3 수치는 결과를 본 뒤 바뀐 '수정 탐색 분석'이다. 이 스크립트는 그와 별도로
'구현 수정된 최초 계획 분석'(implementation-corrected planned analysis)을 낸다.

최초 계획 (첫 커밋 21a16a8 의 removal_sim.py / hypothesis_tests.py 와 같다)
  모집단   T1-original (정제 규칙 전부, test 규칙 포함)
  옵션     task × language × license, 언어 없음 = 별개 값 (현재 --option strict)
  충격     전염: 제거 모델 + 모든 후손 (현재 legal)
  H1       계보가 하나뿐인 옵션 비율의 부트스트랩 95% CI 하한 > 50%
  H2a      k 마다 표적 4전략 손실 > 균등 무작위 k 개 (단측 경험적 p, Holm 20개), k 의 과반에서 유의하면 지지
  H2b      k 마다 descendants 표적 손실 > log2(후손+1) 구간 분포를 맞춘 무작위 k 개 (Holm 5개)
  효과     Cliff's δ
  H3       어느 한 계열이라도 (상업 옵션 손실 %) / (계열 모델 %) > 1
  seeds    1,000
유지되는 구현 수정: (1) 공유 이름 규칙 예외, (2) 제공자 후손만 세기, (4) no-derivatives 는 비상업.
최종 데이터(부모 재연결 포함)를 쓴다. 이는 데이터 수정이며 원고에 밝힌다.

사용: python planned_analysis.py ../../02_data/processed/2026-09-25 [--seeds 1000]
출력: 04_results/tables/2026-09-25_planned/ (removal_results.csv, substitutability.csv, planned_tests.csv, planned_verdicts.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "04_analysis"))
from removal_sim import LICENSE_SCENARIOS, add_common_args, load_inputs  # noqa: E402
from substitutability import lineage_counts, model_roots  # noqa: E402

ALPHA = 0.05
KS = [1, 5, 10, 50, 100]
TARGETED = ["descendants", "descendant_authors", "downloads", "outdegree"]


def matched_random(rng, cand, cand_desc, target, k):
    """v1 과 같다: 표적 집합의 log2(후손+1) 구간별 개수를 맞춰 무작위 추출."""
    bins = np.floor(np.log2(cand_desc + 1)).astype(int)
    in_t = np.isin(cand, target)
    tbins = np.floor(np.log2(cand_desc[in_t] + 1)).astype(int)
    pick = []
    for b, cnt in zip(*np.unique(tbins, return_counts=True)):
        pool = cand[(bins == b) & ~in_t]
        if len(pool) < cnt:
            pool = cand[(bins >= b - 1) & ~in_t]
        pick.extend(rng.choice(pool, size=min(cnt, len(pool)), replace=False))
    return np.array(pick[:k])


def empirical_p(obs, null):
    return (1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null))


def cliffs_delta(x, y):
    y = np.asarray(y)
    return float((x > y).mean() - (x < y).mean())


def holm(p, alpha=ALPHA):
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj, adj <= alpha


def simulate(eco, seeds, out):
    part = out / "_partial"
    part.mkdir(parents=True, exist_ok=True)
    cand = np.flatnonzero((eco.outdeg > 0) & eco.provider)
    cand_desc, cand_auth = eco.descendants_count(cand)
    rank = {"descendants": cand[np.argsort(-cand_desc, kind="stable")],
            "descendant_authors": cand[np.argsort(-cand_auth, kind="stable")],
            "downloads": cand[np.argsort(-eco.downloads[cand], kind="stable")],
            "outdegree": cand[np.argsort(-eco.outdeg[cand], kind="stable")]}
    rows = []
    for k in KS:
        f = part / f"k{k}.csv"
        if f.exists():
            rows.extend(pd.read_csv(f).to_dict("records"))
            print(f"[legal] k={k} done (resumed)", flush=True)
            continue
        block = []
        for name, order in rank.items():
            block.append({"k": k, "strategy": name, "seed": None, **eco.loss(eco.closure(order[:k], "legal"))})
        target = rank["descendants"][:k]
        for s in range(seeds):
            rng = np.random.default_rng(s)
            rnd = rng.choice(cand, size=min(k, len(cand)), replace=False)
            block.append({"k": k, "strategy": "random", "seed": s, **eco.loss(eco.closure(rnd, "legal"))})
            mr = matched_random(rng, cand, cand_desc, target, k)
            block.append({"k": k, "strategy": "random_matched", "seed": s, **eco.loss(eco.closure(mr, "legal"))})
        pd.DataFrame(block).to_csv(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        rows.extend(block)
        print(f"[legal] k={k} done", flush=True)
    for cls in LICENSE_SCENARIOS:
        m = (eco.license_class == cls) & eco.provider
        rows.append({"k": int(m.sum()), "strategy": f"license:{cls}", "seed": None,
                     "pct_models_in_class": 100 * m.sum() / max(eco.provider.sum(), 1),
                     **eco.loss(eco.closure(np.flatnonzero(m), "legal"), removed_class=cls)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "removal_results.csv", index=False)
    return df


def tests(res, sub):
    out = {}
    single = (sub.loc[sub.n_root_lineages > 0, "n_root_lineages"] == 1).to_numpy(float)
    rng = np.random.default_rng(0)
    boots = rng.choice(single, size=(5000, len(single))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["H1"] = {"share_single_lineage": float(single.mean()), "ci95": [float(lo), float(hi)],
                 "n_options": int(len(single)), "supported": bool(lo > 0.5)}
    rows = []
    for k, g in res[~res.strategy.astype(str).str.startswith("license:")].groupby("k"):
        nu = g.loc[g.strategy == "random", "pct_options_lost"].to_numpy()
        nm = g.loc[g.strategy == "random_matched", "pct_options_lost"].to_numpy()
        for s in TARGETED:
            obs = float(g.loc[g.strategy == s, "pct_options_lost"].iat[0])
            rows.append({"hyp": "H2a", "k": k, "strategy": s, "observed": obs, "null_mean": nu.mean(),
                         "p": empirical_p(obs, nu), "cliffs_delta": cliffs_delta(obs, nu)})
            if s == "descendants":
                rows.append({"hyp": "H2b", "k": k, "strategy": s, "observed": obs, "null_mean": nm.mean(),
                             "p": empirical_p(obs, nm), "cliffs_delta": cliffs_delta(obs, nm)})
    t = pd.DataFrame(rows)
    t["p_holm"], t["reject"] = np.nan, False
    for h, idx in t.groupby("hyp").groups.items():
        t.loc[idx, "p_holm"], t.loc[idx, "reject"] = holm(t.loc[idx, "p"].to_numpy())
    for h, g in t.groupby("hyp"):
        by_k = g.groupby("k")["reject"].any()
        out[h] = {"k_significant": int(by_k.sum()), "k_total": int(len(by_k)),
                  "supported": bool(by_k.sum() > len(by_k) / 2),
                  "cliffs_delta_range": [float(g.cliffs_delta.min()), float(g.cliffs_delta.max())]}
    lic = res[res.strategy.astype(str).str.startswith("license:")]
    ratios = {r.strategy.split(":", 1)[1]: (r.pct_commercial_options_lost / r.pct_models_in_class
                                            if r.pct_models_in_class else float("nan"))
              for r in lic.itertuples()}
    out["H3"] = {"ratios_including_own_class": {c: round(float(v), 3) for c, v in ratios.items()},
                 "supported": bool(any(v > 1 for v in ratios.values()))}
    return t, out


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--seeds", type=int, default=1000)
    extra = ["--option", "strict"] + ([] if "--tier" in sys.argv else ["--tier", "T1"])
    args = ap.parse_args(sys.argv[1:] + extra)
    suffix = "" if args.tier == "T1" else f"_{args.tier}"          # T0: 최초 실행에서 T1 과 H2b 가 엇갈렸던 모집단
    args.out = args.out or Path(__file__).resolve().parents[2] / "04_results" / "tables" / f"{args.processed.name}_full_planned{suffix}"
    eco, out = load_inputs(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()}, indent=1), encoding="utf-8")
    res = simulate(eco, args.seeds, out)
    mm, rr = model_roots(eco)
    sub = pd.DataFrame({"option": eco.option_names, "n_root_lineages": lineage_counts(eco, mm, rr)})
    sub.to_csv(out / "substitutability.csv", index=False)
    t, v = tests(res, sub)
    t.to_csv(out / "planned_tests.csv", index=False)
    (out / "planned_verdicts.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    (out / "hypothesis_verdicts.json").write_text(json.dumps(v, indent=1), encoding="utf-8")   # 진행 창의 완료 표시용
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
