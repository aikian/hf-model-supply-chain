"""Model-list cleaning: attach exclusion flags and an analysis tier to each model instead of deleting it.

Input   02_data/processed/<snap>/nodes.parquet
Output  02_data/processed/<snap>/model_flags.parquet   model_id + f_* flags + tier
        04_results/tables/cleaning_report_<snap>.md    counts per flag, tier sizes, examples

Tiers (pre-registered: main analysis T1, sensitivity T0/T2)
    T0  all models
    T1  excludes automated uploads, course assignments, test models and empty shells
    T2  T1 with all-time downloads >= 1

Excluded models stay in the graph (lineage propagation paths are preserved). The simulation
only stops counting them as functional-option "providers".

Usage
    python clean_models.py ../../02_data/processed/2026-09-25
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Automated uploads (mining / distributed-training rewards, competition submissions): names are generated mechanically
BOT_PATTERNS = {
    "timestamp_suffix": r"[-_]1[67]\d{8}$",          # FLock etc.: Qwen-Qwen1.5-0.5B-1725623935
    "blockassist": r"blockassist",                    # Gensyn BlockAssist
    "gensyn_swarm": r"gensyn-swarm|rl-swarm",         # Gensyn RL Swarm
    "uuid_name": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "hex_hash_name": r"^[0-9a-f]{16,}$",
    "omega_miner": r"^omega_[a-z0-9]{5}$",            # Bittensor family
    "cuid_name": r"^c(?=[a-z0-9]*\d[a-z0-9]*\d[a-z0-9]*\d)[a-z0-9]{24}(?:_|$)",  # auto IDs from LoRA training services (3+ digits)
}
# accounts confirmed as automated submitters by account rather than by name (found in the 2026-09-25 review)
BOT_AUTHORS = {"gradients-io-tournaments"}
# assignment submissions from the HF Deep RL course and similar
COURSE_PATTERN = (r"lunarlander|huggy|snowballtarget|pyramids|soccertwos|cartpole|pixelcopter|"
                  r"spaceinvaders|pandareach|^ppo-|^dqn-|^a2c-|^q-frozenlake|^q-taxi|^taxi-v3|^reinforce-")
# libraries for course / game RL environments among RL-tagged models (LLMs trained with RL via transformers/peft are not excluded)
RL_COURSE_LIBS = {"stable-baselines3", "ml-agents", "sample-factory", "cleanrl", "hivex", "rl-algo-impls", "reinforce", "skrl"}
TEST_PATTERN = r"(?:^|[-_.])(?:test|tmp|temp|dummy|debug|placeholder)(?:[-_.\d]|$)"   # 'demo' left out: the review found many legitimate models
BOILERPLATE_MIN_AUTHORS = 50   # a name used identically by 50+ distinct authors = tutorial default
BOILERPLATE_MAX_CHILDREN = 10  # a model with this many children is a supplier, not a tutorial artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()
    snap = args.processed.name

    n = pd.read_parquet(args.processed / "nodes.parquet")
    name = n["model_id"].str.split("/", n=1).str[1].str.lower()
    f = pd.DataFrame({"model_id": n["model_id"]})

    for k, pat in BOT_PATTERNS.items():
        f[f"bot_{k}"] = name.str.contains(pat, regex=True)
    f["bot_author"] = n["author"].isin(BOT_AUTHORS)
    f["f_bot"] = f[[c for c in f if c.startswith("bot_")]].any(axis=1)

    authors_per_name = n.groupby(name)["author"].nunique()
    edges0 = pd.read_parquet(args.processed / "edges.parquet", columns=["child_id", "relation"])
    quantized = n["model_id"].isin(edges0.loc[edges0["relation"].eq("quantized"), "child_id"])
    # quantized releases conventionally keep the original name, so they are exempt from the shared-name rule
    # Fixed 2026-09-26: the shared-name rule was catching famous originals (FLUX.1-dev, Llama-3.1-8B-Instruct, ...)
    # and their mirrors. The following are not tutorial artifacts and are exempt:
    #   (a) the earliest-uploaded model of a shared name (the original)
    #   (b) models linked to the original by an inferred mirror edge (redistributions: infer_edges.py)
    #   (c) models with at least BOILERPLATE_MAX_CHILDREN direct children (suppliers that others build on)
    first = n.assign(nm=name).sort_values("created_at").drop_duplicates("nm")["model_id"]
    is_first = n["model_id"].isin(first)
    ea = args.processed / "edges_all.parquet"
    if ea.exists():
        eall = pd.read_parquet(ea, columns=["parent_id", "child_id", "source"])
        is_mirror = n["model_id"].isin(eall.loc[eall["source"].eq("inferred_mirror"), "child_id"])
        n_children = n["model_id"].map(eall.groupby("parent_id").size()).fillna(0)
    else:
        is_mirror = pd.Series(False, index=n.index)
        n_children = pd.Series(0, index=n.index)
    shared = name.map(authors_per_name).ge(BOILERPLATE_MIN_AUTHORS)
    f["f_boilerplate"] = (shared & ~quantized & ~is_first & ~is_mirror
                          & n_children.lt(BOILERPLATE_MAX_CHILDREN).to_numpy())
    rl = n["pipeline_tag"].eq("reinforcement-learning")
    f["f_course"] = (name.str.contains(COURSE_PATTERN, regex=True)
                     | (rl & (n["library_name"].isin(RL_COURSE_LIBS) | n["library_name"].isna())))
    f["f_test"] = name.str.contains(TEST_PATTERN, regex=True)
    # models whose parent was recovered by inferred edges (infer_edges.py) do not count as metadata-less
    ea = args.processed / "edges_all.parquet"
    has_parent = n["n_parents"].gt(0)
    if ea.exists():
        has_parent |= n["model_id"].isin(pd.read_parquet(ea, columns=["child_id"])["child_id"])
    f["f_empty"] = (n["pipeline_tag"].isna() & n["library_name"].isna()
                    & n["license"].isna() & ~has_parent)
    f["f_zero_downloads"] = n["downloads_all"].fillna(0).eq(0)

    excl = f[["f_bot", "f_boilerplate", "f_course", "f_test", "f_empty"]].any(axis=1)
    f["tier"] = "T0"
    f.loc[~excl, "tier"] = "T1"
    f.loc[~excl & ~f["f_zero_downloads"], "tier"] = "T2"
    f["in_T1"] = ~excl
    f["in_T2"] = ~excl & ~f["f_zero_downloads"]
    f.to_parquet(args.processed / "model_flags.parquet", index=False)

    # ------------------------------------------------ report
    N = len(f)
    edges = pd.read_parquet(args.processed / "edges.parquet")
    excluded_ids = set(f.loc[excl, "model_id"])
    excl_with_children = edges.loc[edges["parent_id"].isin(excluded_ids), "parent_id"].nunique()

    def row(label, mask):
        ex = n.loc[mask, "model_id"].sample(min(2, int(mask.sum())), random_state=0).tolist()
        return f"| {label} | {int(mask.sum()):,} | {100*mask.mean():.2f}% | {', '.join(f'`{e}`' for e in ex)} |"

    lines = [f"# Model-list cleaning report ({snap})", "",
             f"Total models: **{N:,}**. Flags can overlap.", "",
             "## By exclusion reason", "", "| Reason | Models | Share | Examples |", "|---|---:|---:|---|"]
    for k in list(BOT_PATTERNS) + ["author"]:
        lines.append(row(f"automated upload: {k}", f[f"bot_{k}"]))
    lines += [row("**automated uploads, total**", f["f_bot"]),
              row(f"tutorial default name (shared by {BOILERPLATE_MIN_AUTHORS}+ authors)", f["f_boilerplate"]),
              row("course assignment (Deep RL etc.)", f["f_course"]),
              row("test / temporary name", f["f_test"]),
              row("no metadata (no task, library, license or parent)", f["f_empty"]),
              row("**T1 exclusions, total**", excl),
              row("zero all-time downloads (additionally excluded in T2)", f["f_zero_downloads"] & ~excl),
              "", "## Tier sizes", "", "| Tier | Models | Share | Lineage edges (by child) |", "|---|---:|---:|---:|"]
    for t, mask in [("T0 all", pd.Series(True, index=f.index)), ("T1 main analysis", f["in_T1"]), ("T2 used models", f["in_T2"])]:
        ids = set(f.loc[mask, "model_id"])
        lines.append(f"| {t} | {int(mask.sum()):,} | {100*mask.mean():.1f}% | {int(edges['child_id'].isin(ids).sum()):,} |")
    lines += ["", f"- Excluded models that have children: **{excl_with_children:,}**. They stay in the graph as propagation paths.",
              "- Mass-quantization accounts (mradermacher, TheBloke, RichardErkhov, ...) are **legitimate distributors** and are not excluded.",
              "- Pattern rules and thresholds are at the top of this script. Regenerate this report after changing them."]
    top_parent = (edges.assign(excluded=edges["child_id"].isin(excluded_ids))
                  .groupby("parent_id")["excluded"].agg(["size", "mean"])
                  .sort_values("size", ascending=False).head(10))
    lines += ["", "## Base models with many children: before and after cleaning", "", "| Base model | Children (T0) | Excluded share | Children (T1) |", "|---|---:|---:|---:|"]
    for pid, r in top_parent.iterrows():
        lines.append(f"| `{pid}` | {int(r['size']):,} | {100*r['mean']:.0f}% | {int(r['size']*(1-r['mean'])):,} |")

    out = ROOT / "04_results" / "tables" / f"cleaning_report_{snap}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
