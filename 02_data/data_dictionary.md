# Data dictionary — Hugging Face model lineage snapshot 2026-09-25

Files in this record. The `raw/` files are unmodified dumps of the Hugging Face Hub API. Everything else was produced from them by the scripts in `03_code/` of the replication package (GitHub `aikian/hf-model-supply-chain`, archived on Zenodo under 10.5281/zenodo.22997474). SHA-256 hashes of every file are in `MANIFEST_2026-09-25.json`.

Model metadata remains subject to the Hugging Face Hub terms of service. The derived tables are released under CC BY 4.0.

## Raw files

| File | Content |
|---|---|
| `hf_models_2026-09-25.jsonl.gz` | One JSON object per public model, as returned by the Hub API (`/api/models`, paginated by creation time). Keys: `id`, `author`, `createdAt`, `lastModified`, `pipeline_tag`, `library_name`, `tags` (includes `license:*`, `language:*`, `base_model:*`, `dataset:*` tags), `downloads` (30 days), `downloadsAllTime`, `likes`, `gated`, `private`, `_id`. |
| `license_names_2026-09-25.jsonl` | Model-card license names fetched for prominent model families whose tag is `other`; used to reclassify those families. |
| `missing_parents_api_2026-09-25.jsonl` | Hub API lookups for the 405 declared parents that were missing from the snapshot and had at least five children. |

## Processed tables

All tables are keyed by `model_id` (`owner/name`). Lists of tags are stored as strings.

`nodes.parquet` — one row per model (3,094,856 rows)

| Column | Meaning |
|---|---|
| `model_id`, `author` | Repository ID and uploading account |
| `created_at` | Upload time (UTC). `created_at_legacy` is true for models carrying the Hub's bulk-migration timestamp (2022-03-02), which is treated as unknown in the temporal checks |
| `last_modified` | Last modification time as reported by the API |
| `pipeline_tag`, `library_name` | Task and library tags |
| `license`, `n_licenses` | Declared license tag and the number of license tags |
| `languages` | Declared language tags |
| `downloads_30d`, `downloads_all`, `likes`, `gated` | Popularity and gating status at collection time |
| `n_parents`, `n_datasets` | Number of declared `base_model` and `dataset` tags |

`edges.parquet` — declared lineage edges (903,939 rows)

| Column | Meaning |
|---|---|
| `parent_id`, `child_id` | Normalized IDs; the child declares the parent as a base model |
| `relation` | `finetune`, `adapter`, `quantized`, `merge`, or `unspecified` |
| `parent_id_declared` | The parent ID as written in the child's tag, before normalization |
| `parent_in_snapshot` | Whether the parent exists in the snapshot |
| `temporal_ok` | Whether the parent was uploaded before the child (false for the 0.42% of dated edges that violate this) |

`edges_all.parquet` — the analysis graph: declared edges plus reconnected and inferred edges (951,033 rows)

| Column | Meaning |
|---|---|
| (columns of `edges.parquet`) | As above |
| `source` | `declared`, `inferred_mirror` (undeclared re-upload with the exact name of a popular model), `inferred_name` (parent encoded in a `<author>_-_<model>` repository name), or `inferred_quant` (quantized model whose name without the quantization suffix matches exactly one model) |
| `parent_resolution` | For declared parents that were missing and reconnected: `renamed`, `alias` (old organization ID), or `local_path`; empty otherwise |
| `in_cycle` | Whether the edge lies on one of the 12 directed cycles, which the analysis excludes |

`attributes.parquet` — task, languages, and license after inheritance and overrides

| Column | Meaning |
|---|---|
| `task`, `languages`, `license` | Values used by the analysis. Missing values on mirrors and quantized models are inherited from their single parent; `other`-tagged licenses of prominent families are reclassified from the model card |
| `inherited_task`, `inherited_languages`, `inherited_license` | Whether the value was inherited from the parent |
| `license_no_override` | License before the model-card reclassification (used by the "no overrides" robustness variant) |

`model_flags.parquet` — cleaning flags and analysis tiers

| Column | Meaning |
|---|---|
| `bot_timestamp_suffix`, `bot_blockassist`, `bot_gensyn_swarm`, `bot_uuid_name`, `bot_hex_hash_name`, `bot_omega_miner`, `bot_cuid_name`, `bot_author` | Sub-rules of the automated-upload rule |
| `f_bot` | Automated upload (any sub-rule) |
| `f_boilerplate` | Shared boilerplate name used by at least 50 authors, with the exemptions described in the paper |
| `f_course` | Course-assignment naming pattern |
| `f_test` | Test-repository naming pattern. This rule was dropped from the main analysis after validation (precision 64%); T1-main = T1-original plus the models excluded only by this rule |
| `f_empty` | Model lacking all of task, library, license, and parent |
| `f_zero_downloads` | No all-time downloads |
| `tier` | Highest tier the model belongs to: `T0` (excluded from T1-original), `T1` (in T1-original but not T2-original), `T2` (in T2-original) |
| `in_T1`, `in_T2` | Membership in T1-original and T2-original |

The cleaning flags are name- and metadata-based heuristics used to decide which models count as providers of a functional option. They are not judgments about the accounts or people who uploaded the models (validated precision per rule: 64–100%, see the paper). Do not use them to single out individual users.

`arch.parquet` — `model_id`, `arch`: architecture tag from the model configuration, available for about 36% of models; used only for the architecture-grouped sensitivity check of RQ1.

`dataset_edges.parquet` — `dataset_id`, `model_id`: declared training-dataset tags.

`missing_parents_resolved.csv` — `parent_id`, `children`, `status`, `resolved_id`, `in_snapshot`: the resolution of each declared parent missing from the snapshot (renamed, alias, local path, or unavailable) and the ID it was reconnected to.

`summary.json`, `inference_report.json`, `enrich_report.json` — counts reported in the paper: graph statistics, edges added by each inference rule, and attributes filled by inheritance.

## Reproduction

See `03_code/README.md` in the replication package. The pipeline runs `01_collect` (crawl), `02_graph` (graph construction, cleaning, inference, enrichment), `03_simulation` (removal experiments), and `04_analysis` (tables and figures) in that order; the processed tables in this record are the output of `02_graph`.
