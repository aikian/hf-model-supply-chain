"""기반 모델 제거 시뮬레이션: Functional Option Loss (RQ2, RQ3).

정의
    기능 옵션 o = (pipeline_tag, language, license_class)
    M(G)       = 사용 가능한 모델이 1개 이상 제공하는 옵션 집합
    제거 S_k   = 예산 k개의 기반 모델. 전염(contagion) 의미론: S의 후손 전체가 사용 불가
                 (병합 모델은 부모 중 하나라도 제거되면 사용 불가)
    FunctionalLoss(k) = |M(G)| - |M(G \\ closure(S_k))|

전략
    targeted : descendants | descendant_authors | downloads | outdegree   (상위 k개)
               descendant_authors = 후손을 만든 서로 다른 계정 수 (대량 중복 업로드에 강건)
    null     : random          후보(자식 ≥1) 중 무작위 k개
               random_matched  표적 집합의 후손 수 분포(로그 구간)를 맞춘 무작위 k개
                               → "크기"를 통제하고도 표적 제거가 더 치명적인지 본다
    scenario : license:<class> 해당 라이선스 계열 모델 전체를 제거 (RQ3)

사용
    python removal_sim.py ../../02_data/processed/2026-09-25 --k 1 5 10 50 100 --seeds 1000
    ⚠ RR 원칙: Stage 1 승인 전에는 --sample(부분 그래프)로 파이프라인만 검증한다.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------- license classes
# 라이선스 분류는 license_map.csv 에 83종을 하나씩 명시한다 (원고 부록 표와 동일).
# 표에 없는 라이선스는 규칙식으로 추정하고 "other" 로 보수적으로 떨어뜨린다.
LICENSE_MAP = pd.read_csv(Path(__file__).with_name("license_map.csv")).set_index("license")
LICENSE_RULES = [
    ("noncommercial", r"(^|-)nc(-|$)|non-?commercial|research|academic"),
    ("no_derivatives", r"(^|-)nd(-|$)"),
    ("copyleft", r"^(a|l)?gpl|^cc-by-sa|^mpl|^epl|^eupl|^osl"),
    ("responsible_ai", r"openrail|^bigscience|^creativeml"),
    ("vendor_custom", r"llama|gemma|qwen|deepseek|mistral|falcon"),
    ("permissive", r"^apache|^mit$|^bsd|^cc-by-\d|^cc0|^unlicense|^afl|^zlib|^isc|^wtfpl|^artistic|^bsl-1"),
]
# 원고 정의: 상업 이용 가능 = permissive, copyleft, RAIL, vendor-custom (조건부 포함).
# no-derivatives 는 원본의 상업 이용은 허용하지만 파생물을 금지하므로 파생 옵션의 공급원이 될 수 없어 제외
# (2026-09-26 수정: 이전에는 포함되어 원고 정의와 달랐다. analysis_changelog.md #4)
COMMERCIAL_CLASSES = {"permissive", "copyleft", "responsible_ai", "vendor_custom"}


def license_class(lic):
    if not isinstance(lic, str) or not lic:
        return "unknown"
    lic = lic.lower()
    if lic.startswith("override:"):          # enrich_attributes.py 가 license_overrides.csv 로 바로잡은 값
        return lic.split(":", 1)[1]
    if lic in LICENSE_MAP.index:
        return LICENSE_MAP.at[lic, "class"]
    for name, pat in LICENSE_RULES:
        if re.search(pat, lic):
            return name
    return "other"


# ---------------------------------------------------------------- graph
class Ecosystem:
    def __init__(self, nodes, edges, drop_temporal_violations=True, provider=None,
                 option_def="full", other_as_unknown=False, strict_commercial=False):
        """provider: 기능 옵션 제공자로 셀 모델 마스크 (정제 tier). None이면 전체.
        제외된 모델도 그래프에는 남아 전파 경로가 된다."""
        self.ids = nodes["model_id"].to_numpy()
        author = nodes["author"] if "author" in nodes else nodes["model_id"].str.split("/").str[0]
        self.author_code = pd.factorize(author)[0]
        idx = pd.Series(np.arange(len(nodes)), index=nodes["model_id"])
        e = edges[edges["parent_in_snapshot"]]
        if "in_cycle" in e:
            e = e[~e["in_cycle"]]
        if drop_temporal_violations:
            e = e[e["temporal_ok"].astype("boolean").fillna(True).astype(bool)]
        p = idx.loc[e["parent_id"]].to_numpy()
        c = idx.loc[e["child_id"]].to_numpy()
        n = len(nodes)
        self.n = n
        self.children = sparse.csr_matrix((np.ones(len(p), dtype=np.int8), (p, c)), shape=(n, n))
        self.outdeg = np.asarray(self.children.sum(axis=1)).ravel()
        # v2 (2026-09-26, 모의 심사 반영): 충격 두 종류
        #   legal        : 라이선스 제약·취약점 상속 → 모든 후손에 전파 (기존 방식)
        #   availability : 저장소 삭제 → 기반 가중치가 없으면 못 쓰는 '어댑터' 엣지로만 전파.
        #                  파인튜닝·양자화·병합·미러는 가중치를 통째로 가지므로 살아남는다.
        #                  지워진 모델에 지워지지 않은 미러가 있으면 어댑터도 미러로 대체되어 살아남는다.
        rel = e["relation"].to_numpy() if "relation" in e else np.full(len(e), "finetune", dtype=object)
        hard = rel == "adapter"
        self.hard_children = sparse.csr_matrix((np.ones(int(hard.sum()), dtype=np.int8), (p[hard], c[hard])),
                                               shape=(n, n))
        mir = rel == "mirror"
        self.mirror_children = sparse.csr_matrix((np.ones(int(mir.sum()), dtype=np.int8), (p[mir], c[mir])),
                                                 shape=(n, n))
        self.provider = np.ones(n, dtype=bool) if provider is None else np.asarray(provider, dtype=bool)
        self.downloads = nodes["downloads_all"].fillna(nodes["downloads_30d"]).fillna(0).to_numpy() * self.provider

        # 모델-옵션 쌍 (언어가 여러 개면 옵션도 여러 개)
        lang = nodes["languages"].fillna("unk").str.split(",")
        lic_cls = nodes["license"].map(license_class).to_numpy()
        if other_as_unknown:                # 강건성: 'other/unspecified' 를 unknown 으로
            lic_cls = np.where(lic_cls == "other", "unknown", lic_cls)
        opt = pd.DataFrame({
            "m": np.arange(n),
            "task": nodes["pipeline_tag"].fillna("unk").to_numpy(),
            "lang": lang.to_numpy(),
            "lic": lic_cls,
        })[self.provider].explode("lang")
        self.license_class = lic_cls
        if option_def == "coarse":          # 강건성: 언어 태그가 드물어 태스크 × 라이선스만
            keys = opt["task"] + "|" + opt["lic"]
        else:
            keys = opt["task"] + "|" + opt["lang"] + "|" + opt["lic"]
        codes, self.option_names = pd.factorize(keys)
        self.pair_model = opt["m"].to_numpy()
        self.pair_option = codes
        self.n_options = len(self.option_names)
        self.providers = np.bincount(codes, minlength=self.n_options)
        opt_lic = pd.Series(self.option_names).str.rsplit("|", n=1).str[-1]
        commercial = {"permissive"} if strict_commercial else COMMERCIAL_CLASSES
        self.option_commercial = opt_lic.isin(commercial).to_numpy()
        self.option_class = opt_lic.to_numpy()
        # v2 언어 와일드카드 (option_def="full"): 언어 태그가 없는 모델(unk)은 같은 (태스크, 라이선스)의
        # 모든 언어 옵션을 대체할 수 있다고 본다. 언어 태그가 15%뿐이라, unk 를 별개 언어로 두면
        # 태그 없는 영어 모델이 'en' 옵션의 대체재가 되지 못한다 (모의 심사 M2).
        # option_def="strict" 는 v1 방식(unk 를 별개 값으로), "coarse" 는 태스크 × 라이선스.
        self.wildcard = option_def == "full"
        parts = pd.Series(self.option_names).str.split("|")
        if option_def == "coarse":
            self.option_is_unk = np.zeros(self.n_options, dtype=bool)
            self.option_group = np.arange(self.n_options)
        else:
            self.option_is_unk = (parts.str[1] == "unk").to_numpy()
            self.option_group = pd.factorize(parts.str[0] + "|" + parts.str[2])[0]
        self.n_groups = int(self.option_group.max()) + 1 if self.n_options else 0
        # 탐색적 '품질 하락' 지표 준비 (2026-09-26, 결과를 본 뒤 추가 — 판정에 쓰지 않음):
        # 옵션별 제공자를 다운로드 내림차순으로 정렬해 두고, 제거 후 남은 최선 제공자의 다운로드를 빠르게 찾는다.
        dl_pair = self.downloads[self.pair_model]
        order = np.lexsort((-dl_pair, self.pair_option))
        self._sorted_pair = order
        self._sorted_dl = dl_pair[order]
        starts = np.searchsorted(self.pair_option[order], np.arange(self.n_options))
        self._opt_start = starts
        self.best_before_spec = np.where(self.providers > 0, self._sorted_dl[np.minimum(starts, len(order) - 1)], 0.0)
        unk_opt = np.full(self.n_groups, -1)
        unk_opt[self.option_group[self.option_is_unk]] = np.flatnonzero(self.option_is_unk)
        self._group_unk_opt = unk_opt
        self.best_before = self._with_wildcard(self.best_before_spec)

    def _with_wildcard(self, best_spec):
        """언어 와일드카드: 언어가 있는 옵션의 최선 제공자 = max(그 옵션, 같은 그룹의 언어 미상 옵션).
        언어 미상 옵션 자신은 그룹 안 모든 옵션의 최선."""
        if not self.wildcard:
            return best_spec
        g = self.option_group
        u = self._group_unk_opt[g]
        unk_best = np.where(u >= 0, best_spec[np.maximum(u, 0)], 0.0)
        grp_max = np.zeros(self.n_groups)
        np.maximum.at(grp_max, g, best_spec)
        return np.where(self.option_is_unk, grp_max[g], np.maximum(best_spec, unk_best))

    def degradation(self, removed_mask):
        """남은 최선 제공자의 다운로드가 제거 전 최선의 몇 %인가 (탐색적)."""
        rem_sorted = removed_mask[self.pair_model[self._sorted_pair]]
        big = len(rem_sorted)
        idx = np.where(rem_sorted, big, np.arange(big))
        first = np.minimum.reduceat(idx, np.minimum(self._opt_start, big - 1)) if big else np.zeros(0, int)
        first = np.where(self.providers > 0, first, big)
        # 옵션의 구간 끝을 넘으면 남은 제공자가 없는 것
        end = np.append(self._opt_start[1:], big)
        ok = first < end
        best_spec = np.where(ok, self._sorted_dl[np.minimum(first, big - 1)], 0.0)
        after = self._with_wildcard(best_spec)
        valid = self.best_before > 0
        ratio = np.where(valid, after / np.where(valid, self.best_before, 1), np.nan)
        opt_dl = np.bincount(self.pair_option, weights=self.downloads[self.pair_model], minlength=self.n_options)
        deg = 1 - ratio
        w = opt_dl * valid
        return {
            "pct_options_degraded90": 100 * np.nansum(ratio < 0.1) / max(valid.sum(), 1),
            "demand_weighted_degradation": float(np.nansum(w * deg) / max(w.sum(), 1)),
        }

    def reach(self, seeds, graph=None):
        """seeds와 그 후손 전체의 인덱스 배열. 프런티어 확장이라 도달한 노드 수에만 비례한다
        (scipy BFS는 호출마다 길이 n 배열을 만들어 후손이 적은 모델에도 O(n) 비용이 든다).
        graph: 따라갈 엣지 (기본 전체 계보, availability 충격은 hard_children)."""
        graph = self.children if graph is None else graph
        seen = np.zeros(self.n, dtype=bool) if not hasattr(self, "_seen") else self._seen
        self._seen = seen
        frontier = np.unique(np.asarray(seeds, dtype=np.int64))
        if not len(frontier):
            return frontier
        seen[frontier] = True
        out = [frontier]
        ip, ix = graph.indptr, graph.indices
        while len(frontier):
            starts, ends = ip[frontier], ip[frontier + 1]
            lens = ends - starts
            if lens.sum() == 0:
                break
            # 각 프런티어 노드의 자식 구간을 한 번에 모은다
            offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
            nxt = ix[np.arange(lens.sum()) + offs]
            nxt = np.unique(nxt[~seen[nxt]])
            seen[nxt] = True
            out.append(nxt)
            frontier = nxt
        res = np.concatenate(out)
        seen[res] = False                     # 재사용 버퍼 초기화 (O(도달 수))
        return res

    def closure(self, seeds, semantics="legal"):
        """제거 집합 seeds 가 사용 불가로 만드는 모델 마스크.
        legal: seeds + 모든 후손. availability: seeds + (미러가 남지 않은 seed 의) 어댑터 후손."""
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        mask = np.zeros(self.n, dtype=bool)
        if semantics == "legal":
            mask[self.reach(seeds)] = True
            return mask
        mask[seeds] = True
        # 미러가 하나라도 제거 집합 밖에 남아 있으면 그 seed 의 어댑터는 미러로 대체된다
        # (2026-09-27: seed 마다 돌던 파이썬 반복을 벡터 연산으로 바꿈. 결과는 같다 — test_closure_vectorized)
        substituted = self._mirror_substituted(seeds, mask)
        hard_seeds = seeds[~substituted]
        mask[self.reach(hard_seeds, graph=self.hard_children)] = True
        return mask

    def _mirror_substituted(self, seeds, mask):
        """seed 마다: 미러 자식 중 mask 밖에 남은 것이 하나라도 있는가."""
        if not len(seeds):
            return np.zeros(0, bool)
        ip, ix = self.mirror_children.indptr, self.mirror_children.indices
        starts, lens = ip[seeds], ip[seeds + 1] - ip[seeds]
        if lens.sum() == 0:
            return np.zeros(len(seeds), bool)
        offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        kids = ix[np.arange(lens.sum()) + offs]
        seg = np.repeat(np.arange(len(seeds)), lens)
        return np.bincount(seg, weights=(~mask[kids]).astype(float), minlength=len(seeds)) > 0

    def _mirror_substituted_reference(self, seeds, mask):
        """벡터화 이전의 원래 구현 (검증용)."""
        ip, ix = self.mirror_children.indptr, self.mirror_children.indices
        return np.array([bool((~mask[ix[ip[s]:ip[s + 1]]]).any()) for s in seeds], dtype=bool) \
            if len(seeds) else np.zeros(0, bool)

    def descendants_count(self, candidates):
        """(후손 모델 수, 후손을 만든 서로 다른 계정 수). 자기 계정은 제외.
        계정 수는 같은 계정의 대량 업로드(하이퍼파라미터 탐색, 채굴)에 흔들리지 않는다.
        2026-09-26 수정: 기능 옵션 제공자(provider, 정제 tier)인 후손만 센다. 이전에는 정제에서 걸러 낸
        봇 업로드까지 세어서 Qwen1.5-0.5B(후손 32,595 중 제공자 331) 같은 모델이 최상위 표적이 됐다."""
        n_models, n_authors = [], []
        for s in candidates:
            d = self.reach([s])
            d = d[(d != s) & self.provider[d]]
            n_models.append(len(d))
            a = np.unique(self.author_code[d])
            n_authors.append(len(a) - int(self.author_code[s] in a))
        return np.array(n_models), np.array(n_authors)

    def loss(self, removed_mask, removed_class=None):
        """removed_class: 라이선스 전염 시나리오(RQ3)에서 지운 계열. 그 계열 자신의 옵션을 뺀
        '부수 피해(collateral)' 손실을 따로 낸다. 지운 계열의 옵션이 사라지는 건 정의상 당연하므로
        (동어반복), H3는 다른 라이선스를 가진 하위 옵션의 손실로 판정한다 (2026-09-26 수정)."""
        gone = np.bincount(self.pair_option[removed_mask[self.pair_model]], minlength=self.n_options)
        rem = self.providers - gone
        if self.wildcard:
            g = self.option_group
            unk_rem = np.bincount(g[self.option_is_unk], weights=rem[self.option_is_unk], minlength=self.n_groups)
            all_rem = np.bincount(g, weights=rem, minlength=self.n_groups)
            lost_known = ~self.option_is_unk & (rem == 0) & (unk_rem[g] == 0)
            lost_unk = self.option_is_unk & (all_rem[g] == 0)
            lost = (self.providers > 0) & (lost_known | lost_unk)
        else:
            lost = (self.providers > 0) & (rem == 0)
        opt_dl = np.bincount(self.pair_option, weights=self.downloads[self.pair_model],
                             minlength=self.n_options)
        other = self.option_class != removed_class if removed_class else np.ones(self.n_options, bool)
        extra = {
            "pct_collateral_options_lost": 100 * (lost & other).sum() / max(other.sum(), 1),
            "pct_collateral_commercial_lost": 100 * (lost & other & self.option_commercial).sum()
                                              / max((other & self.option_commercial).sum(), 1),
        }
        return {
            "options_total": int(self.n_options),
            "options_lost": int(lost.sum()),
            "pct_options_lost": 100 * lost.sum() / self.n_options,
            "pct_option_demand_lost": 100 * opt_dl[lost].sum() / max(opt_dl.sum(), 1),
            "pct_commercial_options_lost": 100 * (lost & self.option_commercial).sum()
                                           / max(self.option_commercial.sum(), 1),
            "models_lost": int((removed_mask & self.provider).sum()),     # 제공자만 (2026-09-26 수정)
            "models_lost_incl_flagged": int(removed_mask.sum()),
            "pct_downloads_lost": 100 * self.downloads[removed_mask].sum() / max(self.downloads.sum(), 1),
            **self.degradation(removed_mask),
            **extra,
        }


# ---------------------------------------------------------------- strategies (v2)
SEMANTICS = ("legal", "availability")
LICENSE_SCENARIOS = ["noncommercial", "no_derivatives", "vendor_custom", "responsible_ai", "copyleft"]


def single_closure_sizes(eco, cand, semantics, cand_desc):
    """후보 하나만 지웠을 때 사용 불가가 되는 제공자 수 (규모 맞춤 대조군의 누적합용)."""
    if semantics == "legal":
        return cand_desc + eco.provider[cand].astype(int)
    return np.array([int((eco.closure([c], "availability") & eco.provider).sum()) for c in cand])


def union_matched(rng, eco, cand, dsize, target_n, semantics, tol=0.95):
    """제거되는 제공자 수(합집합)가 표적 전략과 같아질 때까지 무작위 후보를 더한다 (모의 심사 M3).
    누적합으로 필요한 개수를 먼저 추정하고, 겹침 때문에 모자라면 15%씩 늘린다."""
    perm = rng.permutation(len(cand))
    cs = np.cumsum(dsize[perm])
    m = int(np.searchsorted(cs, target_n)) + 1
    while True:
        m = min(m, len(cand))
        mask = eco.closure(cand[perm[:m]], semantics)
        got = int((mask & eco.provider).sum())
        if got >= tol * target_n or m >= len(cand):
            return mask, got, m
        m = int(m * 1.15) + 1


def greedy_sequence(eco, pool, kmax, semantics):
    """탐욕적 최대 손실: 매 단계 기능 옵션 손실을 가장 많이 늘리는 후보를 고른다 (lazy 평가).
    손실 함수가 부분 모듈 함수가 아니어서 최적 보장은 없다. '공격자가 도달할 수 있는 수준'의 근사."""
    import heapq
    base = np.zeros(eco.n, dtype=bool)
    cur = 0.0
    single = {int(c): eco.closure([c], semantics) for c in pool}
    heap = [(-eco.loss(single[c])["options_lost"], c, 0) for c in single]
    heapq.heapify(heap)
    chosen, step = [], 0
    while heap and len(chosen) < kmax:
        neg, c, stamp = heapq.heappop(heap)
        if stamp == step:                        # 최신 이득이면 채택
            chosen.append(c)
            base |= single[c]
            cur = eco.loss(base)["options_lost"]
            step += 1
            continue
        gain = eco.loss(base | single[c])["options_lost"] - cur
        heapq.heappush(heap, (-gain, c, step))
    return chosen


def run(eco, ks, seeds, out_dir, semantics_list=SEMANTICS, greedy_pool=2000):
    cand = np.flatnonzero((eco.outdeg > 0) & eco.provider)
    print(f"candidates (models with ≥1 child): {len(cand):,}", flush=True)
    cand_desc, cand_auth = eco.descendants_count(cand)
    rank = {
        "descendants": cand[np.argsort(-cand_desc, kind="stable")],
        "descendant_authors": cand[np.argsort(-cand_auth, kind="stable")],
        "downloads": cand[np.argsort(-eco.downloads[cand], kind="stable")],
        "outdegree": cand[np.argsort(-eco.outdeg[cand], kind="stable")],
    }
    pool = np.unique(np.concatenate([rank["descendants"][:greedy_pool], rank["descendant_authors"][:greedy_pool]]))
    # 단계별 중간 저장: Colab 세션이 끊겨도 끝난 (충격, k) 묶음은 다시 계산하지 않는다.
    # 난수는 (seed, k, 전략) 로 고정되므로 이어서 계산해도 한 번에 계산한 결과와 같다.
    part = out_dir / f"_partial_s{seeds}"
    part.mkdir(parents=True, exist_ok=True)
    rows = []
    for sem in semantics_list:
        todo = [k for k in ks if not (part / f"{sem}_k{k}.csv").exists()]
        for k in ks:
            if k not in todo:
                rows.extend(pd.read_csv(part / f"{sem}_k{k}.csv").to_dict("records"))
                print(f"[{sem}] k={k} done (resumed)", flush=True)
        if not todo:
            print(f"[{sem}] greedy done (resumed)", flush=True)
            continue
        dsize = single_closure_sizes(eco, cand, sem, cand_desc)
        gpath = part / f"{sem}_greedy.json"
        if gpath.exists():
            greedy = json.loads(gpath.read_text())
        else:
            greedy = [int(g) for g in greedy_sequence(eco, pool, max(ks), sem)]
            gpath.write_text(json.dumps(greedy))
        print(f"[{sem}] greedy done", flush=True)
        for k in todo:
            block_start = len(rows)
            targets = {name: order[:k] for name, order in rank.items()}
            targets["greedy"] = np.array(greedy[:k])
            for name, t in targets.items():
                mask = eco.closure(t, sem)
                tn = int((mask & eco.provider).sum())
                rows.append({"semantics": sem, "k": k, "strategy": name, "seed": None, "matched_to": None,
                             "removed_providers": tn, **eco.loss(mask)})
                if name == "greedy":
                    continue
                for s in range(seeds):             # 규모 맞춤 대조군 (전략마다 따로)
                    rng = np.random.default_rng([s, k, len(name)])
                    mm, got, m = union_matched(rng, eco, cand, dsize, tn, sem)
                    rows.append({"semantics": sem, "k": k, "strategy": "random_matched", "seed": s,
                                 "matched_to": name, "removed_providers": got, "n_seeds_drawn": m, **eco.loss(mm)})
            for s in range(seeds):                 # 균등 무작위 (참고용 기준선)
                rng = np.random.default_rng([s, k])
                rnd = rng.choice(cand, size=min(k, len(cand)), replace=False)
                mask = eco.closure(rnd, sem)
                rows.append({"semantics": sem, "k": k, "strategy": "random", "seed": s, "matched_to": None,
                             "removed_providers": int((mask & eco.provider).sum()), **eco.loss(mask)})
            tmp = part / f"{sem}_k{k}.csv.tmp"
            pd.DataFrame(rows[block_start:]).to_csv(tmp, index=False)
            tmp.replace(part / f"{sem}_k{k}.csv")       # 다 쓴 뒤에 이름을 바꿔 반쯤 쓴 파일이 남지 않게
            print(f"[{sem}] k={k} done", flush=True)

    # RQ3: 라이선스 의무를 계보를 따라 적용 = 법적 충격 (해당 계열 모델과 모든 후손)
    for cls in LICENSE_SCENARIOS:
        seeds_mask = (eco.license_class == cls) & eco.provider
        mask = eco.closure(np.flatnonzero(seeds_mask), "legal")
        rows.append({"semantics": "legal", "k": int(seeds_mask.sum()), "strategy": f"license:{cls}", "seed": None,
                     "matched_to": None, "removed_providers": int((mask & eco.provider).sum()),
                     "pct_models_in_class": 100 * seeds_mask.sum() / max(eco.provider.sum(), 1),
                     **eco.loss(mask, removed_class=cls)})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "removal_results.csv", index=False)
    agg = (df.groupby(["semantics", "k", "strategy", "matched_to"], dropna=False)
             [["pct_options_lost", "removed_providers"]].agg(["mean", "std"]).round(3))
    agg.to_csv(out_dir / "removal_summary.csv")
    return df


def estimate_runtime(eco, ks, seeds, n_variants=9, probe_candidates=2000, probe_sets=5):
    """전체 실행 시간 추정. 손실 값은 계산만 하고 버린다 (RR: 결과를 보지 않음)."""
    cand = np.flatnonzero((eco.outdeg > 0) & eco.provider)
    probe = cand[:probe_candidates]
    t = time.perf_counter(); eco.descendants_count(probe); t_desc = (time.perf_counter() - t) / len(probe)
    rng = np.random.default_rng(0)
    per_k = {}
    for k in ks:
        t = time.perf_counter()
        for _ in range(probe_sets):
            eco.loss(eco.closure(rng.choice(cand, size=min(k, len(cand)), replace=False)))
        per_k[k] = (time.perf_counter() - t) / probe_sets
    t_cand = t_desc * len(cand)
    t_null = sum(per_k.values()) * seeds * 2            # uniform + matched
    one = t_cand + t_null
    rep = {"candidates": int(len(cand)), "sec_per_descendant_bfs": round(t_desc, 5),
           "sec_per_removal_eval": {int(k): round(v, 4) for k, v in per_k.items()},
           "est_minutes_candidates": round(t_cand / 60, 1), "est_minutes_nulls": round(t_null / 60, 1),
           "est_hours_one_run": round(one / 3600, 2), "n_variants": n_variants,
           "est_hours_all_variants": round(one * n_variants / 3600, 1)}
    print(json.dumps(rep, indent=1))
    return rep


def add_common_args(ap):
    ap.add_argument("processed", type=Path)
    ap.add_argument("--tier", choices=["T0", "T1", "T2"], default="T1",
                    help="옵션 제공자 정제 수준 (clean_models.py). 주 분석 T1")
    ap.add_argument("--sample", type=float, default=None,
                    help="파일럿: 계정의 이 비율만 (계정 단위로 추출해 계보를 보존)")
    # 강건성 점검 (사전 등록)
    ap.add_argument("--declared-only", action="store_true", help="추론 엣지 제외")
    ap.add_argument("--no-inherit", action="store_true", help="양자화·미러 속성 상속 끄기")
    ap.add_argument("--no-license-overrides", action="store_true",
                    help="'other' 라이선스 재분류(license_overrides.csv) 끄기")
    ap.add_argument("--option", choices=["full", "strict", "coarse"], default="full",
                    help="full = 태스크×언어×라이선스 (언어 없음 = 와일드카드), strict = v1 (언어 없음을 별개 값), "
                         "coarse = 태스크×라이선스")
    ap.add_argument("--exclude-quantized", action="store_true", help="양자화 모델을 제공자에서 제외")
    ap.add_argument("--other-as-unknown", action="store_true")
    ap.add_argument("--strict-commercial", action="store_true", help="상업 이용 = permissive 만")
    ap.add_argument("--keep-temporal-violations", action="store_true")
    ap.add_argument("--drop-rule", nargs="+", default=[],
                    choices=["f_bot", "f_course", "f_test", "f_boilerplate", "f_empty"],
                    help="이 정제 규칙을 끈다 (그 규칙에만 걸린 모델을 제공자로 되돌림). "
                         "사전에 정한 결정 규칙: 검증 정밀도 80%% 미만 규칙은 끄고 주 분석을 다시 돌린다")
    ap.add_argument("--out", type=Path, default=None)


def load_inputs(args):
    """nodes, edges, provider mask, 결과 폴더 태그. 두 시뮬레이션 스크립트가 공유한다."""
    P = args.processed
    nodes = pd.read_parquet(P / "nodes.parquet")
    attr = P / "attributes.parquet"
    if attr.exists() and not args.no_inherit:           # enrich_attributes.py 결과로 교체
        a = pd.read_parquet(attr, columns=["model_id", "task", "languages", "license", "license_no_override"])
        a["license"] = a.pop("license_no_override") if args.no_license_overrides else a["license"]
        a = a.drop(columns=["license_no_override"], errors="ignore")
        nodes = nodes.drop(columns=["pipeline_tag", "languages", "license"]).merge(a, on="model_id", how="left")
        nodes = nodes.rename(columns={"task": "pipeline_tag"})
    ea = P / "edges_all.parquet"
    edges = pd.read_parquet(ea if ea.exists() else P / "edges.parquet")
    if args.declared_only and "source" in edges:
        edges = edges[edges["source"].eq("declared")]
    drop = getattr(args, "drop_rule", [])
    if args.tier != "T0":
        rules = ["f_bot", "f_boilerplate", "f_course", "f_test", "f_empty"]
        flags = pd.read_parquet(P / "model_flags.parquet", columns=["model_id", f"in_{args.tier}", *rules,
                                                                    "f_zero_downloads"])
        if drop:                                          # T1 = 남은 규칙 어느 것에도 안 걸림 (clean_models.py 와 같은 정의)
            keep = ~flags[[r for r in rules if r not in drop]].any(axis=1)
            flags[f"in_{args.tier}"] = keep & (~flags["f_zero_downloads"] if args.tier == "T2" else True)
            print(f"drop rule(s) {drop}: providers {int(keep.sum()):,} (before: {int(flags.shape[0] - flags[rules].any(axis=1).sum()):,})")
        nodes = nodes.merge(flags[["model_id", f"in_{args.tier}"]], on="model_id", how="left")
    if args.sample:
        authors = nodes["author"].drop_duplicates().sample(frac=args.sample, random_state=0)
        nodes = nodes[nodes["author"].isin(authors)].reset_index(drop=True)
        keep = set(nodes["model_id"])
        edges = edges[edges["child_id"].isin(keep)].copy()
        edges["parent_in_snapshot"] &= edges["parent_id"].isin(keep)

    provider = (np.ones(len(nodes), dtype=bool) if args.tier == "T0"
                else nodes[f"in_{args.tier}"].fillna(False).to_numpy(dtype=bool))
    if args.exclude_quantized:
        quant = set(edges.loc[edges["relation"].eq("quantized"), "child_id"])
        provider &= ~nodes["model_id"].isin(quant).to_numpy()

    variant = [f"sample{args.sample}" if args.sample else "full", args.tier]
    variant += [flag for flag, on in [("declared", args.declared_only), ("noinherit", args.no_inherit),
                                      ("noov", args.no_license_overrides),
                                      ("coarse", args.option == "coarse"), ("strictlang", args.option == "strict"), ("noquant", args.exclude_quantized),
                                      ("other2unk", args.other_as_unknown),
                                      ("strictcom", args.strict_commercial),
                                      ("keeptv", args.keep_temporal_violations)] if on]
    variant += [f"no{r.removeprefix('f_')}" for r in drop]
    tag = "_".join(variant)
    print(f"nodes {len(nodes):,}  edges {len(edges):,}  providers {provider.sum():,}  ({tag})")
    eco = Ecosystem(nodes, edges, drop_temporal_violations=not args.keep_temporal_violations,
                    provider=provider, option_def=args.option, other_as_unknown=args.other_as_unknown,
                    strict_commercial=args.strict_commercial)
    out = args.out or Path(__file__).resolve().parents[2] / "04_results" / "tables" / f"{P.name}_{tag}"
    return eco, out


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    ap.add_argument("--seeds", type=int, default=1000,
                    help="null 반복 수. 1,000 미만이면 Holm 보정 후 유의 판정이 불가능할 수 있다")
    ap.add_argument("--estimate-runtime", action="store_true",
                    help="실행 시간만 추정하고 결과는 저장·출력하지 않음 (RR 안전)")
    args = ap.parse_args()
    eco, out = load_inputs(args)
    if args.estimate_runtime:
        rep = estimate_runtime(eco, args.k, args.seeds)
        est = Path(__file__).resolve().parents[2] / "04_results" / "tables" / f"runtime_estimate_{args.processed.name}.json"
        est.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return
    run(eco, args.k, args.seeds, out)
    (out / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
