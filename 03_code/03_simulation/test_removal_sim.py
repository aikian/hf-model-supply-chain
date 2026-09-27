"""합성 그래프로 removal_sim 검증.  python -m pytest 03_simulation -q  또는  python test_removal_sim.py"""
import numpy as np
import pandas as pd

from removal_sim import Ecosystem, license_class


def _toy_frames():
    #   A ─┬─ A1 ─ A11          B ── B1        C (고립)
    #      └─ A2 ─┐
    #   B ────────┴─ M (merge: A2 + B)
    nodes = pd.DataFrame({
        "model_id": ["A", "A1", "A11", "A2", "B", "B1", "M", "C"],
        "pipeline_tag": ["tg", "tg", "tc", "tg", "tg", "asr", "tg", "tg"],
        "languages": ["en", "en", "ko", "en", "en", "en", "en,ko", "en"],
        "license": ["apache-2.0", "mit", "mit", "cc-by-nc-4.0", "llama3", "apache-2.0", "mit", "apache-2.0"],
        "downloads_all": [100, 10, 5, 3, 50, 2, 1, 7],
        "downloads_30d": [0] * 8,
    })
    E = [("A", "A1"), ("A1", "A11"), ("A", "A2"), ("A2", "M"), ("B", "M"), ("B", "B1")]
    edges = pd.DataFrame(E, columns=["parent_id", "child_id"])
    edges["parent_in_snapshot"] = True
    edges["temporal_ok"] = True
    return nodes, edges


def toy():
    return Ecosystem(*_toy_frames())


def test_license_class():
    assert license_class("apache-2.0") == "permissive"
    assert license_class("cc-by-nc-4.0") == "noncommercial"
    assert license_class("gpl-3.0") == "copyleft"
    assert license_class("llama3.1") == "vendor_custom"
    assert license_class("openrail++") == "responsible_ai"
    assert license_class(None) == "unknown"
    assert license_class("other") == "other"             # 자체 라이선스는 회사 라이선스와 구분
    assert license_class("cc-by-nd-4.0") == "no_derivatives"
    assert license_class("apple-amlr") == "noncommercial"
    assert license_class("ncsa") == "permissive"
    assert license_class("some-new-nc-license") == "noncommercial"   # 표에 없으면 규칙식
    assert license_class("override:noncommercial") == "noncommercial"


def test_closure_merge_contagion():
    eco = toy()
    i = {m: k for k, m in enumerate(eco.ids)}
    removed = eco.closure([i["B"]])
    assert set(eco.ids[removed]) == {"B", "B1", "M"}      # 병합 모델 M은 B 제거로 오염
    removed = eco.closure([i["A"]])
    assert set(eco.ids[removed]) == {"A", "A1", "A11", "A2", "M"}


def test_loss():
    eco = toy()
    i = {m: k for k, m in enumerate(eco.ids)}
    none = eco.loss(np.zeros(eco.n, dtype=bool))
    assert none["options_lost"] == 0
    r = eco.loss(eco.closure([i["B"]]))
    # B 계열에만 있던 옵션: tg|en|vendor_custom(B), asr|en|permissive(B1). M의 tg|ko|permissive는 A11(tc) 아님 → 손실
    lost = {"tg|en|vendor_custom", "asr|en|permissive", "tg|ko|permissive"}
    assert r["options_lost"] == len(lost)
    # tg|en|permissive는 A1, C가 대체 → 손실 아님
    assert r["models_lost"] == 3




def test_descendant_authors():
    nodes, edges = _toy_frames()
    nodes["author"] = ["a", "a", "x", "y", "b", "b", "z", "c"]     # A 계열 후손: A1(a) A11(x) A2(y) M(z)
    eco = Ecosystem(nodes, edges)
    i = {m: k for k, m in enumerate(eco.ids)}
    nm, na = eco.descendants_count([i["A"], i["B"]])
    assert list(nm) == [4, 2]
    assert list(na) == [3, 1]          # A: x,y,z (자기 계정 a 제외) / B: z (B1은 같은 계정 b)


def test_provider_descendants_and_collateral():
    # 제공자가 아닌 후손은 표적 순위에 세지 않는다
    nodes, edges = _toy_frames()
    nodes["author"] = ["a", "a", "x", "y", "b", "b", "z", "c"]
    base = Ecosystem(nodes, edges)
    i = {m: k for k, m in enumerate(base.ids)}
    prov = np.ones(base.n, dtype=bool); prov[[i["A1"], i["A11"]]] = False
    eco = Ecosystem(nodes, edges, provider=prov)
    nm, _ = eco.descendants_count([i["A"]])
    assert list(nm) == [2]                                   # A2, M 만 (A1, A11 제외)
    # 부수 피해: 비상업(A2) 제거 시 비상업 옵션 자신은 부수 피해에서 제외
    r = base.loss(base.closure([i["A2"]]), removed_class="noncommercial")
    # 사라진 옵션: tg|en|noncommercial(A2 자신), tg|ko|permissive(M) → 부수 피해는 M 쪽 1개
    assert r["options_lost"] == 2
    other = base.option_class != "noncommercial"
    assert abs(r["pct_collateral_options_lost"] - 100 * 1 / other.sum()) < 1e-9


def test_provider_mask():
    # A2(비상업)를 제공자에서 빼도 그래프에는 남아 M으로 전파된다
    base = toy()
    i = {m: k for k, m in enumerate(base.ids)}
    prov = np.ones(base.n, dtype=bool); prov[i["A2"]] = False
    eco = Ecosystem(*_toy_frames(), provider=prov)
    assert "tg|en|noncommercial" not in set(eco.option_names)     # A2의 옵션은 세지 않음
    assert set(eco.ids[eco.closure([i["A"]])]) >= {"A2", "M"}      # 전파 경로는 유지


def _v2_frames():
    #  B ─adapter→ L1 ─adapter→ L2        B ─finetune→ F        B ─mirror→ Bm
    #  D ─adapter→ L3   (D 는 미러 없음)
    nodes = pd.DataFrame({
        "model_id": ["B", "L1", "L2", "F", "Bm", "D", "L3", "U"],
        "pipeline_tag": ["tg"] * 8,
        "languages": ["en", "en", "en", "en", "en", "ko", "ko", None],
        "license": ["mit"] * 8,
        "downloads_all": [1] * 8, "downloads_30d": [0] * 8,
    })
    E = [("B", "L1", "adapter"), ("L1", "L2", "adapter"), ("B", "F", "finetune"),
         ("B", "Bm", "mirror"), ("D", "L3", "adapter")]
    edges = pd.DataFrame(E, columns=["parent_id", "child_id", "relation"])
    edges["parent_in_snapshot"] = True
    edges["temporal_ok"] = True
    return nodes, edges


def test_availability_semantics_and_mirror():
    eco = Ecosystem(*_v2_frames())
    i = {m: k for k, m in enumerate(eco.ids)}
    # 법적 충격: 모든 후손
    assert set(eco.ids[eco.closure([i["B"]], "legal")]) == {"B", "L1", "L2", "F", "Bm"}
    # 가용성 충격: B 에 미러 Bm 이 남아 있으므로 어댑터도 살아남는다 → B 만
    assert set(eco.ids[eco.closure([i["B"]], "availability")]) == {"B"}
    # 미러까지 지우면 어댑터 사슬이 끊긴다 (파인튜닝 F 는 가중치를 가지므로 생존)
    assert set(eco.ids[eco.closure([i["B"], i["Bm"]], "availability")]) == {"B", "Bm", "L1", "L2"}
    # 미러 없는 D: 어댑터 L3 도 사용 불가
    assert set(eco.ids[eco.closure([i["D"]], "availability")]) == {"D", "L3"}


def test_language_wildcard():
    nodes, edges = _v2_frames()
    eco = Ecosystem(nodes, edges)                      # option_def="full" (와일드카드)
    i = {m: k for k, m in enumerate(eco.ids)}
    # ko 옵션 제공자는 D, L3. 언어 없는 U 가 ko 를 대체 → D, L3 를 지워도 ko 옵션 유지
    r = eco.loss(eco.closure([i["D"]], "legal"))
    assert r["options_lost"] == 0
    strict = Ecosystem(nodes, edges, option_def="strict")
    assert strict.loss(strict.closure([i["D"]], "legal"))["options_lost"] == 1   # v1 방식이면 ko 옵션 손실
    # U 까지 지우면 ko 손실, unk 옵션은 en 제공자가 남아 있으므로 유지
    m = eco.closure([i["D"]], "legal"); m[i["U"]] = True
    assert eco.loss(m)["options_lost"] == 1


def test_union_matched_reaches_size():
    from removal_sim import union_matched, single_closure_sizes
    eco = Ecosystem(*_v2_frames())
    cand = np.flatnonzero(eco.outdeg > 0)
    desc, _ = eco.descendants_count(cand)
    dsize = single_closure_sizes(eco, cand, "legal", desc)
    mask, got, m = union_matched(np.random.default_rng(0), eco, cand, dsize, target_n=4, semantics="legal")
    assert got >= 0.95 * 4


def test_degradation():
    eco = toy()
    i = {m: k for k, m in enumerate(eco.ids)}
    o = list(eco.option_names).index("tg|en|permissive")
    assert eco.best_before[o] == 100                       # A(100), A1(10), C(7) 중 최선 = A
    m = eco.closure([i["A"]], "legal")                     # A, A1, A11, A2, M 제거 → C(7)만 남음
    d = eco.degradation(m)
    # 옵션별 비율을 직접 계산해 비교
    none = eco.degradation(np.zeros(eco.n, bool))
    assert none["pct_options_degraded90"] == 0 and abs(none["demand_weighted_degradation"]) < 1e-12
    assert d["pct_options_degraded90"] > 0
    # tg|en|permissive 는 100 → 7 (7%) 로 '크게 약해짐'에 포함되어야 한다
    rem_sorted = m[eco.pair_model[eco._sorted_pair]]
    s, e = eco._opt_start[o], (eco._opt_start[o + 1] if o + 1 < eco.n_options else len(rem_sorted))
    left = eco._sorted_dl[s:e][~rem_sorted[s:e]]
    assert left.max() == 7


if __name__ == "__main__":
    test_license_class(); test_closure_merge_contagion(); test_loss(); test_descendant_authors(); test_provider_descendants_and_collateral(); test_provider_mask(); test_availability_semantics_and_mirror(); test_language_wildcard(); test_union_matched_reaches_size(); test_degradation()
    print("all tests passed")
