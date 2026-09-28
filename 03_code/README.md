# Pipeline

```bash
pip install -r requirements.txt

# 1) Collection: full snapshot. If interrupted, the same command resumes
python 01_collect/collect_hf_models.py --snapshot 2026-09-25
#    pilot: --max-pages 3  (1,000 models per page)

# 2) Graph: node/edge parquet + summary.json
python 02_graph/build_graph.py ../02_data/raw/hf_models_2026-09-25.jsonl.gz

# 2b) Inferred edges (mirrors, parent in the name, quantization suffix) + cycle marking -> edges_all.parquet
python 02_graph/infer_edges.py ../02_data/processed/2026-09-25

# 2b') Model-card lookup for reclassifying 'other' licenses (result -> 03_simulation/license_overrides.csv; the class mapping is done by hand after reading the license text)
python 02_graph/fetch_license_names.py ../02_data/processed/2026-09-25 --out ../02_data/raw/license_names_2026-09-25.jsonl

# 2c) Attribute inheritance (+ apply license_overrides.csv): fill the empty task/language/license of quantizations and mirrors from the parent -> attributes.parquet
python 02_graph/enrich_attributes.py ../02_data/processed/2026-09-25

# 3) Cleaning: exclusion flags + tier (T0 all / T1 main analysis / T2 used models) -> 04_results/tables/cleaning_report_*.md
python 02_graph/clean_models.py ../02_data/processed/2026-09-25

# 4) Simulation (--sample runs a quick functional check on a subsample)
python 03_simulation/removal_sim.py ../02_data/processed/2026-09-25 --tier T1 --sample 0.05
python 03_simulation/substitutability.py ../02_data/processed/2026-09-25 --tier T1 --sample 0.05

#    robustness variants: --tier T0|T2 --declared-only --no-inherit --option coarse
#                 --exclude-quantized --other-as-unknown --strict-commercial --keep-temporal-violations
#                 --no-license-overrides

# 4b) Hypothesis decisions. Self-check: --selftest
python 04_analysis/hypothesis_tests.py ../04_results/tables/2026-09-25_full_main_notest

# 5) Pilot figures/tables
python 04_analysis/pilot_figures.py ../02_data/processed/2026-09-25
```

The full set of runs used in the paper (main analysis, T1-original, and the robustness variants) is driven by `run_all_experiments.py`; see the repository README.

## Collection
- `GET https://huggingface.co/api/models?sort=createdAt&direction=1&limit=1000&expand[]=...`
  with cursor pagination (`Link: rel="next"`). Because results are sorted by creation date ascending, models uploaded during the crawl cause neither gaps nor duplicates.
- The anonymous rate limit is 500 requests per 5 minutes. Setting the `HF_TOKEN` environment variable raises the limit.
- On a rate-limit response, the crawler waits for `t=` (seconds until reset) from the `RateLimit` header and retries.

## Lineage edge parsing (tag-based)
| Tag | Interpretation |
|---|---|
| `base_model:<rel>:<parent>` | rel ∈ {finetune, adapter, quantized, merge} |
| `base_model:<parent>` (no type) | `unspecified` unless a typed tag for the same parent exists |
| `license:<id>` | declared license (the first one if several; the count is in `n_licenses`) |
| `dataset:<id>` | training-dataset edge |
| two-letter ISO 639-1 tag | language (**three-letter codes are dropped; a limitation**) |

- `temporal_ok` = parent createdAt ≤ child createdAt (computed only when the parent is in the snapshot)
- `parent_in_snapshot = False`: the parent was deleted or made private. This is itself an analysis point: a supplier that has already disappeared.

## Known limitations (discussed in the paper's threats to validity)
- `base_model` is self-reported and often missing (prior work: only 15–26% of models declare it).
- Parsing is tag-based, so lineage stated only in the model-card text is missed.
- `downloads` is the last-30-day count; `downloadsAllTime` is the cumulative count.
