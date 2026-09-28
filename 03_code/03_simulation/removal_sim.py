"""Base-model removal simulation: Functional Option Loss (RQ2, RQ3).

Definitions
    functional option o = (pipeline_tag, language, license_class)
    M(G)       = set of options provided by at least one usable model
    removal S_k = budget of k base models. Contagion semantics: every descendant of S becomes unusable
                 (a merge model is unusable if any of its parents is removed)
    FunctionalLoss(k) = |M(G)| - |M(G \\ closure(S_k))|

Strategies
    targeted : descendants | descendant_authors | downloads | outdegree   (top k)
               descendant_authors = number of distinct accounts that created descendants (resists mass duplicate uploads)
    null     : random          k random picks from the candidates (>= 1 child)
               random_matched  k random picks matching the descendant-count distribution (log bins) of the target set
                               -> tests whether targeted removal is still more damaging after controlling for "size"
    scenario : license:<class> remove every model of that license class (RQ3)

Usage
    python removal_sim.py ../../02_data/processed/2026-09-25 --k 1 5 10 50 100 --seeds 1000
    RR rule: before Stage 1 approval, only validate the pipeline with --sample (subgraph).
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
# License classes: license_map.csv lists all 83 licenses explicitly (same as the appendix table in the manuscript).
# Licenses not in the table are guessed by regex rules and otherwise fall back conservatively to "other".
LICENSE_MAP = pd.read_csv(Path(__file__).with_name("license_map.csv")).set_index("license")
LICENSE_RULES = [
    ("noncommercial", r"(^|-)nc(-|$)|non-?commercial|research|academic"),
    ("no_derivatives", r"(^|-)nd(-|$)"),
    ("copyleft", r"^(a|l)?gpl|^cc-by-sa|^mpl|^epl|^eupl|^osl"),
    ("responsible_ai", r"openrail|^bigscience|^creativeml"),
    ("vendor_custom", r"llama|gemma|qwen|deepseek|mistral|falcon"),
    ("permissive", r"^apache|^mit$|^bsd|^cc-by-\d|^cc0|^unlicense|^afl|^zlib|^isc|^wtfpl|^artistic|^bsl-1"),
]
# Manuscript definition: commercial use allowed = permissive, copyleft, RAIL, vendor-custom (conditional ones included).
# no-derivatives allows commercial use of the original but forbids derivatives, so it cannot supply derived options; excluded
# (fixed 2026-09-26: it was included before, which disagreed with the manuscript definition. analysis_changelog.md #4)
COMMERCIAL_CLASSES = {"permissive", "copyleft", "responsible_ai", "vendor_custom"}


def license_class(lic):
    if not isinstance(lic, str) or not lic:
        return "unknown"
    lic = lic.lower()
    if lic.startswith("override:"):          # value corrected by enrich_attributes.py via license_overrides.csv
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
        """provider: mask of models counted as functional-option providers (cleaning tier). None = all models.
        Excluded models stay in the graph and still act as propagation paths."""
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
        # v2 (2026-09-26, after mock review): two shock types
        #   legal        : license constraints / inherited vulnerabilities -> propagate to all descendants (original method)
        #   availability : repository deletion -> propagate only over 'adapter' edges, which are unusable without the base weights.
        #                  Fine-tunes, quantizations, merges and mirrors carry the full weights, so they survive.
        #                  If a deleted model has a surviving mirror, its adapters fall back to the mirror and survive too.
        rel = e["relation"].to_numpy() if "relation" in e else np.full(len(e), "finetune", dtype=object)
        hard = rel == "adapter"
        self.hard_children = sparse.csr_matrix((np.ones(int(hard.sum()), dtype=np.int8), (p[hard], c[hard])),
                                               shape=(n, n))
        mir = rel == "mirror"
        self.mirror_children = sparse.csr_matrix((np.ones(int(mir.sum()), dtype=np.int8), (p[mir], c[mir])),
                                                 shape=(n, n))
        self.provider = np.ones(n, dtype=bool) if provider is None else np.asarray(provider, dtype=bool)
        self.downloads = nodes["downloads_all"].fillna(nodes["downloads_30d"]).fillna(0).to_numpy() * self.provider

        # model-option pairs (a model with several languages yields several options)
        lang = nodes["languages"].fillna("unk").str.split(",")
        lic_cls = nodes["license"].map(license_class).to_numpy()
        if other_as_unknown:                # sensitivity check: map 'other/unspecified' to unknown
            lic_cls = np.where(lic_cls == "other", "unknown", lic_cls)
        opt = pd.DataFrame({
            "m": np.arange(n),
            "task": nodes["pipeline_tag"].fillna("unk").to_numpy(),
            "lang": lang.to_numpy(),
            "lic": lic_cls,
        })[self.provider].explode("lang")
        self.license_class = lic_cls
        if option_def == "coarse":          # sensitivity check: language tags are sparse, so task x license only
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
        # v2 language wildcard (option_def="full"): a model with no language tag (unk) is treated as a substitute for
        # every language option of the same (task, license). Only 15% of models carry a language tag, so treating unk as
        # a separate language would stop untagged English models from substituting for 'en' options (mock review M2).
        # option_def="strict" is the v1 method (unk as a separate value); "coarse" is task x license.
        self.wildcard = option_def == "full"
        parts = pd.Series(self.option_names).str.split("|")
        if option_def == "coarse":
            self.option_is_unk = np.zeros(self.n_options, dtype=bool)
            self.option_group = np.arange(self.n_options)
        else:
            self.option_is_unk = (parts.str[1] == "unk").to_numpy()
            self.option_group = pd.factorize(parts.str[0] + "|" + parts.str[2])[0]
        self.n_groups = int(self.option_group.max()) + 1 if self.n_options else 0
        # Exploratory 'quality degradation' metric (added 2026-09-26 after seeing results; not used for verdicts):
        # sort each option's providers by downloads descending so the best remaining provider after removal is found quickly.
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
        """Language wildcard: best provider of a language-specific option = max(that option, the unk option of the same group).
        The unk option itself takes the best over every option in its group."""
        if not self.wildcard:
            return best_spec
        g = self.option_group
        u = self._group_unk_opt[g]
        unk_best = np.where(u >= 0, best_spec[np.maximum(u, 0)], 0.0)
        grp_max = np.zeros(self.n_groups)
        np.maximum.at(grp_max, g, best_spec)
        return np.where(self.option_is_unk, grp_max[g], np.maximum(best_spec, unk_best))

    def degradation(self, removed_mask):
        """Downloads of the best remaining provider as a share of the best provider before removal (exploratory)."""
        rem_sorted = removed_mask[self.pair_model[self._sorted_pair]]
        big = len(rem_sorted)
        idx = np.where(rem_sorted, big, np.arange(big))
        first = np.minimum.reduceat(idx, np.minimum(self._opt_start, big - 1)) if big else np.zeros(0, int)
        first = np.where(self.providers > 0, first, big)
        # past the end of the option's segment means no provider is left
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
        """Index array of seeds plus all their descendants. Frontier expansion, so cost scales only with the number of reached nodes
        (scipy BFS allocates a length-n array per call, costing O(n) even for models with few descendants).
        graph: edges to follow (default: full lineage; the availability shock uses hard_children)."""
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
            # gather the child segments of all frontier nodes at once
            offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
            nxt = ix[np.arange(lens.sum()) + offs]
            nxt = np.unique(nxt[~seen[nxt]])
            seen[nxt] = True
            out.append(nxt)
            frontier = nxt
        res = np.concatenate(out)
        seen[res] = False                     # reset the reusable buffer (O(reached nodes))
        return res

    def closure(self, seeds, semantics="legal"):
        """Mask of models made unusable by removing seeds.
        legal: seeds + all descendants. availability: seeds + adapter descendants of seeds that have no surviving mirror."""
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        mask = np.zeros(self.n, dtype=bool)
        if semantics == "legal":
            mask[self.reach(seeds)] = True
            return mask
        mask[seeds] = True
        # if at least one mirror survives outside the removal set, that seed's adapters fall back to the mirror
        # (2026-09-27: per-seed Python loop replaced by vector ops. Same result; see test_closure_vectorized)
        substituted = self._mirror_substituted(seeds, mask)
        hard_seeds = seeds[~substituted]
        mask[self.reach(hard_seeds, graph=self.hard_children)] = True
        return mask

    def _mirror_substituted(self, seeds, mask):
        """Per seed: does at least one mirror child survive outside mask?"""
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
        """Original implementation before vectorization (kept for verification)."""
        ip, ix = self.mirror_children.indptr, self.mirror_children.indices
        return np.array([bool((~mask[ix[ip[s]:ip[s + 1]]]).any()) for s in seeds], dtype=bool) \
            if len(seeds) else np.zeros(0, bool)

    def descendants_count(self, candidates):
        """(number of descendant models, number of distinct accounts that created them). Own account excluded.
        The account count is not skewed by mass uploads from one account (hyperparameter sweeps, mining).
        Fixed 2026-09-26: count only descendants that are providers (cleaning tier). Before, bot uploads removed by
        cleaning were counted too, so models like Qwen1.5-0.5B (32,595 descendants, 331 providers) became top targets."""
        n_models, n_authors = [], []
        for s in candidates:
            d = self.reach([s])
            d = d[(d != s) & self.provider[d]]
            n_models.append(len(d))
            a = np.unique(self.author_code[d])
            n_authors.append(len(a) - int(self.author_code[s] in a))
        return np.array(n_models), np.array(n_authors)

    def loss(self, removed_mask, removed_class=None):
        """removed_class: the license class removed in the license-contagion scenario (RQ3). The loss excluding that
        class's own options is reported separately as 'collateral'. Losing the removed class's own options holds by
        definition (a tautology), so H3 is judged on lost downstream options that carry a different license (fixed 2026-09-26)."""
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
            "models_lost": int((removed_mask & self.provider).sum()),     # providers only (fixed 2026-09-26)
            "models_lost_incl_flagged": int(removed_mask.sum()),
            "pct_downloads_lost": 100 * self.downloads[removed_mask].sum() / max(self.downloads.sum(), 1),
            **self.degradation(removed_mask),
            **extra,
        }


# ---------------------------------------------------------------- strategies (v2)
SEMANTICS = ("legal", "availability")
LICENSE_SCENARIOS = ["noncommercial", "no_derivatives", "vendor_custom", "responsible_ai", "copyleft"]


def single_closure_sizes(eco, cand, semantics, cand_desc):
    """Number of providers made unusable by removing a single candidate (for the cumulative sum in the size-matched null)."""
    if semantics == "legal":
        return cand_desc + eco.provider[cand].astype(int)
    return np.array([int((eco.closure([c], "availability") & eco.provider).sum()) for c in cand])


def union_matched(rng, eco, cand, dsize, target_n, semantics, tol=0.95):
    """Add random candidates until the number of removed providers (union) matches the targeted strategy (mock review M3).
    The count needed is first estimated from the cumulative sum; if overlap leaves it short, grow it by 15% per step."""
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
    """Greedy maximal loss: at each step pick the candidate that raises functional option loss the most (lazy evaluation).
    The loss function is not submodular, so there is no optimality guarantee. An approximation of 'what an attacker could reach'."""
    import heapq
    base = np.zeros(eco.n, dtype=bool)
    cur = 0.0
    single = {int(c): eco.closure([c], semantics) for c in pool}
    heap = [(-eco.loss(single[c])["options_lost"], c, 0) for c in single]
    heapq.heapify(heap)
    chosen, step = [], 0
    while heap and len(chosen) < kmax:
        neg, c, stamp = heapq.heappop(heap)
        if stamp == step:                        # gain is up to date: accept
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
    # Save after each step: if the Colab session drops, finished (shock, k) blocks are not recomputed.
    # Random streams are keyed by (seed, k, strategy), so a resumed run equals a single uninterrupted run.
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
                for s in range(seeds):             # size-matched null (separately per strategy)
                    rng = np.random.default_rng([s, k, len(name)])
                    mm, got, m = union_matched(rng, eco, cand, dsize, tn, sem)
                    rows.append({"semantics": sem, "k": k, "strategy": "random_matched", "seed": s,
                                 "matched_to": name, "removed_providers": got, "n_seeds_drawn": m, **eco.loss(mm)})
            for s in range(seeds):                 # uniform random (reference baseline)
                rng = np.random.default_rng([s, k])
                rnd = rng.choice(cand, size=min(k, len(cand)), replace=False)
                mask = eco.closure(rnd, sem)
                rows.append({"semantics": sem, "k": k, "strategy": "random", "seed": s, "matched_to": None,
                             "removed_providers": int((mask & eco.provider).sum()), **eco.loss(mask)})
            tmp = part / f"{sem}_k{k}.csv.tmp"
            pd.DataFrame(rows[block_start:]).to_csv(tmp, index=False)
            tmp.replace(part / f"{sem}_k{k}.csv")       # rename after writing so a half-written file is never left behind
            print(f"[{sem}] k={k} done", flush=True)

    # RQ3: apply license obligations along the lineage = legal shock (models of that class plus all descendants)
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
    """Estimate the total run time. Loss values are computed and discarded (RR: results are not inspected)."""
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
                    help="Provider cleaning tier (clean_models.py). Main analysis: T1")
    ap.add_argument("--sample", type=float, default=None,
                    help="Pilot: keep only this fraction of accounts (sampled per account to preserve lineages)")
    # sensitivity checks (pre-registered)
    ap.add_argument("--declared-only", action="store_true", help="Exclude inferred edges")
    ap.add_argument("--no-inherit", action="store_true", help="Disable attribute inheritance for quantized and mirror models")
    ap.add_argument("--no-license-overrides", action="store_true",
                    help="Disable reclassification of 'other' licenses (license_overrides.csv)")
    ap.add_argument("--option", choices=["full", "strict", "coarse"], default="full",
                    help="full = task x language x license (missing language = wildcard), strict = v1 (missing language as a separate value), "
                         "coarse = task x license")
    ap.add_argument("--exclude-quantized", action="store_true", help="Exclude quantized models from providers")
    ap.add_argument("--other-as-unknown", action="store_true")
    ap.add_argument("--strict-commercial", action="store_true", help="Commercial use = permissive only")
    ap.add_argument("--keep-temporal-violations", action="store_true")
    ap.add_argument("--drop-rule", nargs="+", default=[],
                    choices=["f_bot", "f_course", "f_test", "f_boilerplate", "f_empty"],
                    help="Disable these cleaning rules (models flagged only by them become providers again). "
                         "Pre-set decision rule: a rule with validated precision below 80%% is disabled and the main analysis is rerun")
    ap.add_argument("--out", type=Path, default=None)


def load_inputs(args):
    """nodes, edges, provider mask, and the result-folder tag. Shared by both simulation scripts."""
    P = args.processed
    nodes = pd.read_parquet(P / "nodes.parquet")
    attr = P / "attributes.parquet"
    if attr.exists() and not args.no_inherit:           # replace with the enrich_attributes.py output
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
        if drop:                                          # T1 = flagged by none of the remaining rules (same definition as clean_models.py)
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
                    help="Number of null draws. Below 1,000, significance after Holm correction may be unreachable")
    ap.add_argument("--estimate-runtime", action="store_true",
                    help="Only estimate the run time; results are neither saved nor printed (RR-safe)")
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
