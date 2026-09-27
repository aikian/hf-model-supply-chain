"""RQ1: 기능 옵션별 독립 루트 계보 수 (대체 가능성).

루트 = 스냅샷 안에서 부모가 없는 모델. 병합 모델은 여러 루트를 가질 수 있다.
옵션 o의 독립 계보 수 = o를 제공하는 모델들의 루트 합집합 크기.

사용
    python substitutability.py ../../02_data/processed/2026-09-25 [--sample 0.05]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from removal_sim import add_common_args, load_inputs


def model_roots(eco):
    """각 모델의 루트 집합 → (model, root) 쌍 배열. 루트에서 후손 방향 BFS."""
    indeg = np.asarray(eco.children.sum(axis=0)).ravel()
    roots = np.flatnonzero(indeg == 0)
    # 자식 없는 고립 루트는 자기 자신만 → BFS 생략
    lone = roots[eco.outdeg[roots] == 0]
    m, r = [lone], [lone]
    for root in roots[eco.outdeg[roots] > 0]:
        desc = eco.reach([root])
        m.append(desc)
        r.append(np.full(len(desc), root))
    return np.concatenate(m), np.concatenate(r)


def bootstrap_share(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def lineage_counts(eco, mm, rr, root_ok=None):
    """옵션별 독립 루트 계보 수. 와일드카드(eco.wildcard)이면 언어 없는 제공자의 계보를 같은
    (태스크, 라이선스)의 모든 언어 옵션에 더한다. root_ok: 계보로 인정할 루트 마스크 (민감도)."""
    mr = pd.DataFrame({"m": mm, "root": rr})
    if root_ok is not None:
        mr = mr[root_ok[mr["root"].to_numpy()]]
    mo = pd.DataFrame({"m": eco.pair_model, "o": eco.pair_option}).merge(mr, on="m")
    spec = {o: np.unique(g.to_numpy()) for o, g in mo.groupby("o")["root"]}
    empty = np.zeros(0, dtype=np.int64)
    if not eco.wildcard:
        return np.array([len(spec.get(o, empty)) for o in range(eco.n_options)])
    mo["g"] = eco.option_group[mo["o"].to_numpy()]
    unk = mo[eco.option_is_unk[mo["o"].to_numpy()]]
    unk_roots = {g: np.unique(x.to_numpy()) for g, x in unk.groupby("g")["root"]}
    all_roots = {g: np.unique(x.to_numpy()) for g, x in mo.groupby("g")["root"]}
    out = np.zeros(eco.n_options, dtype=np.int64)
    for o in range(eco.n_options):
        g = eco.option_group[o]
        if eco.option_is_unk[o]:
            out[o] = len(all_roots.get(g, empty))          # 언어 미상 옵션: 그룹 내 어떤 언어의 제공자든 대체
        else:
            a, b = spec.get(o, empty), unk_roots.get(g, empty)
            out[o] = len(a) + len(b) - len(np.intersect1d(a, b, assume_unique=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    eco, out = load_inputs(args)
    mm, rr = model_roots(eco)
    n_all = lineage_counts(eco, mm, rr)
    indeg = np.asarray(eco.children.sum(axis=0)).ravel()
    isolated = (indeg == 0) & (eco.outdeg == 0)       # 부모도 자식도 없는 모델: 계보 선언이 빠졌을 수 있다 (M9)
    n_known = lineage_counts(eco, mm, rr, root_ok=~isolated)
    # 하한 (허점 2, 2026-09-26): 같은 구조 태그(llama, qwen2 …)의 루트를 한 계보로 묶는다.
    # 부모를 선언하지 않은 파인튜닝이 독립 계보로 세어지는 문제를 보수적으로 막는다.
    # 같은 구조로 따로 학습한 모델도 합쳐지므로 독립 계보를 과소 추정한다 = 대체 가능성 하한.
    n_arch = np.full(eco.n_options, -1)
    ap_ = args.processed / "arch.parquet"
    if ap_.exists():
        a = pd.read_parquet(ap_)
        code = pd.Series(np.arange(eco.n), index=eco.ids)
        fam = np.arange(eco.n) + 1_000_000                    # 태그 없는 루트는 자기 자신
        hit = a["model_id"].isin(code.index)
        cats = pd.factorize(a.loc[hit, "arch"].astype(str))[0]
        fam[code.loc[a.loc[hit, "model_id"]].to_numpy()] = cats
        n_arch = lineage_counts(eco, mm, fam[rr])
    res = pd.DataFrame({"option": eco.option_names, "n_root_lineages": n_all,
                        "n_root_lineages_known": n_known, "n_root_lineages_arch": n_arch,
                        "n_providers": eco.providers})
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "substitutability.csv", index=False)
    s = res[res["n_root_lineages"] > 0]
    print(f"options: {eco.n_options:,}  single-lineage: {100 * (s['n_root_lineages'] == 1).mean():.2f}%")


if __name__ == "__main__":
    main()
