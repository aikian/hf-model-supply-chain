"""Tests for removal_sim on synthetic graphs.  python -m pytest 03_simulation -q  or  python test_removal_sim.py"""
import numpy as np
import pandas as pd

from removal_sim import Ecosystem, license_class


def _toy_frames():
    #   A ─┬─ A1 ─ A11          B ── B1        C (isolated)
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
    assert license_class("other") == "other"             # custom licenses are kept apart from vendor licenses
    assert license_class("cc-by-nd-4.0") == "no_derivatives"
    assert license_class("apple-amlr") == "noncommercial"
    assert license_class("ncsa") == "permissive"
    assert license_class("some-new-nc-license") == "noncommercial"   # not in the table -> regex rules
    assert license_class("override:noncommercial") == "noncommercial"


def test_closure_merge_contagion():
    eco = toy()
    i = {m: k for k, m in enumerate(eco.ids)}
    removed = eco.closure([i["B"]])
    assert set(eco.ids[removed]) == {"B", "B1", "M"}      # merge model M is contaminated by removing B
    removed = eco.closure([i["A"]])
    assert set(eco.ids[removed]) == {"A", "A1", "A11", "A2", "M"}


def test_loss():
    eco = toy()
    i = {m: k for k, m in enumerate(eco.ids)}
    none = eco.loss(np.zeros(eco.n, dtype=bool))
    assert none["options_lost"] == 0
    r = eco.loss(eco.closure([i["B"]]))
    # options only in the B lineage: tg|en|vendor_custom(B), asr|en|permissive(B1). M's tg|ko|permissive is not covered by A11(tc) -> lost
    lost = {"tg|en|vendor_custom", "asr|en|permissive", "tg|ko|permissive"}
    assert r["options_lost"] == len(lost)
    # tg|en|permissive is substituted by A1 and C -> not lost
    assert r["models_lost"] == 3




def test_descendant_authors():
    nodes, edges = _toy_frames()
    nodes["author"] = ["a", "a", "x", "y", "b", "b", "z", "c"]     # descendants of A: A1(a) A11(x) A2(y) M(z)
    eco = Ecosystem(nodes, edges)
    i = {m: k for k, m in enumerate(eco.ids)}
    nm, na = eco.descendants_count([i["A"], i["B"]])
    assert list(nm) == [4, 2]
    assert list(na) == [3, 1]          # A: x,y,z (own account a excluded) / B: z (B1 is the same account b)


def test_provider_descendants_and_collateral():
    # non-provider descendants do not count toward target ranking
    nodes, edges = _toy_frames()
    nodes["author"] = ["a", "a", "x", "y", "b", "b", "z", "c"]
    base = Ecosystem(nodes, edges)
    i = {m: k for k, m in enumerate(base.ids)}
    prov = np.ones(base.n, dtype=bool); prov[[i["A1"], i["A11"]]] = False
    eco = Ecosystem(nodes, edges, provider=prov)
    nm, _ = eco.descendants_count([i["A"]])
    assert list(nm) == [2]                                   # A2 and M only (A1, A11 excluded)
    # collateral: when the noncommercial model A2 is removed, the noncommercial option itself is excluded from collateral
    r = base.loss(base.closure([i["A2"]]), removed_class="noncommercial")
    # lost options: tg|en|noncommercial (A2 itself), tg|ko|permissive (M) -> collateral is the single M-side option
    assert r["options_lost"] == 2
    other = base.option_class != "noncommercial"
    assert abs(r["pct_collateral_options_lost"] - 100 * 1 / other.sum()) < 1e-9


def test_provider_mask():
    # A2 (noncommercial) removed from providers still stays in the graph and propagates to M
    base = toy()
    i = {m: k for k, m in enumerate(base.ids)}
    prov = np.ones(base.n, dtype=bool); prov[i["A2"]] = False
    eco = Ecosystem(*_toy_frames(), provider=prov)
    assert "tg|en|noncommercial" not in set(eco.option_names)     # A2's option is not counted
    assert set(eco.ids[eco.closure([i["A"]])]) >= {"A2", "M"}      # propagation path is kept


def _v2_frames():
    #  B ─adapter→ L1 ─adapter→ L2        B ─finetune→ F        B ─mirror→ Bm
    #  D ─adapter→ L3   (D has no mirror)
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
    # legal shock: all descendants
    assert set(eco.ids[eco.closure([i["B"]], "legal")]) == {"B", "L1", "L2", "F", "Bm"}
    # availability shock: mirror Bm of B survives, so the adapters survive too -> B only
    assert set(eco.ids[eco.closure([i["B"]], "availability")]) == {"B"}
    # removing the mirror as well breaks the adapter chain (fine-tune F carries weights, so it survives)
    assert set(eco.ids[eco.closure([i["B"], i["Bm"]], "availability")]) == {"B", "Bm", "L1", "L2"}
    # D has no mirror: adapter L3 becomes unusable too
    assert set(eco.ids[eco.closure([i["D"]], "availability")]) == {"D", "L3"}


def test_language_wildcard():
    nodes, edges = _v2_frames()
    eco = Ecosystem(nodes, edges)                      # option_def="full" (wildcard)
    i = {m: k for k, m in enumerate(eco.ids)}
    # ko option providers are D and L3. Untagged U substitutes for ko -> the ko option survives removing D and L3
    r = eco.loss(eco.closure([i["D"]], "legal"))
    assert r["options_lost"] == 0
    strict = Ecosystem(nodes, edges, option_def="strict")
    assert strict.loss(strict.closure([i["D"]], "legal"))["options_lost"] == 1   # under the v1 method the ko option is lost
    # removing U as well loses ko; the unk option survives because en providers remain
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
    assert eco.best_before[o] == 100                       # best of A(100), A1(10), C(7) = A
    m = eco.closure([i["A"]], "legal")                     # removes A, A1, A11, A2, M -> only C(7) remains
    d = eco.degradation(m)
    # compare against per-option ratios computed directly
    none = eco.degradation(np.zeros(eco.n, bool))
    assert none["pct_options_degraded90"] == 0 and abs(none["demand_weighted_degradation"]) < 1e-12
    assert d["pct_options_degraded90"] > 0
    # tg|en|permissive goes 100 -> 7 (7%), so it must count as 'degraded by 90% or more'
    rem_sorted = m[eco.pair_model[eco._sorted_pair]]
    s, e = eco._opt_start[o], (eco._opt_start[o + 1] if o + 1 < eco.n_options else len(rem_sorted))
    left = eco._sorted_dl[s:e][~rem_sorted[s:e]]
    assert left.max() == 7


if __name__ == "__main__":
    test_license_class(); test_closure_merge_contagion(); test_loss(); test_descendant_authors(); test_provider_descendants_and_collateral(); test_provider_mask(); test_availability_semantics_and_mirror(); test_language_wildcard(); test_union_matched_reaches_size(); test_degradation()
    print("all tests passed")
