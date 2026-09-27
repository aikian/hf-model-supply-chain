# Lineage Diversity and Metadata-Defined Option Survival in the Hugging Face Model Supply Chain — replication package

Code and derived results for a study of counterfactual upstream-model removal on the Hugging Face model
lineage graph (snapshot **2026-09-25**, 3,094,856 models, 951,033 lineage edges).

> The raw snapshot (~184 MB) and processed graph tables exceed GitHub's file
> limit and are archived as a Zenodo dataset: [10.5281/zenodo.22998359](https://doi.org/10.5281/zenodo.22998359).
> This code is archived on Zenodo under [10.5281/zenodo.22997474](https://doi.org/10.5281/zenodo.22997474) (all versions).
> SHA-256 hashes of the data files are in `02_data/MANIFEST_2026-09-25.json`.

## Layout
| Path | Content |
|---|---|
| `03_code/01_collect/` | Hub metadata crawler (cursor pagination, resumable) |
| `03_code/02_graph/` | Lineage graph, parent reconnection, inferred edges, attribute inheritance, cleaning rules, rule tests |
| `03_code/03_simulation/` | Removal simulation (RQ2/RQ3), lineage counts (RQ1), initial-plan rerun, matched-null sensitivity, license map |
| `03_code/04_analysis/` | Hypothesis tests, result audit, detectable effects, paper and supplement tables, figures |
| `03_code/run_all_experiments.py` | Main analysis, original T1, and ten robustness variants (resumable) |
| `04_results/tables/` | All result tables; `_superseded_*` hold outputs of earlier runs |
| `04_results/figures/` | Figures (PDF/PNG) |
| `02_data/` | Data dictionary, SHA-256 manifest, labeling codebook |

## Reproduce
From `03_code`, with `P=../02_data/processed/2026-09-25`:
```bash
pip install -r requirements.txt
python run_all_experiments.py $P                                       # revised analyses (12 runs)
python 03_simulation/planned_analysis.py $P [--tier T0]                # initial planned analysis
python 03_simulation/null_sensitivity.py $P --setting tol90|tol99|nearest
python 03_simulation/substitutability.py $P --option strict --drop-rule f_test --out <dir>   # T1-main strict
python 04_analysis/audit_results.py                                    # checks all outputs
python 04_analysis/make_paper_tables.py && python 04_analysis/make_supplement.py
```
Runs checkpoint after every (shock, k) block and resume where they stopped. Random seeds are fixed per
(run, k, strategy), so resumed and uninterrupted runs give identical results.

**Resources.** One run needs about 3 GB of memory. On a laptop with 8 GB of memory or a free Colab instance
(2 cores, 12 GB), each variant took 1–3.5 hours; the main analysis with 1,000 null runs took 3.4 hours.

## Notes
- Cleaning flags are heuristics for deciding which models count as providers, not judgments about uploaders.
  Example model IDs of individual accounts in the cleaning report are pseudonymized.
- License classes come from self-declared tags; they are not legal assessments.

## License
Code: MIT (see LICENSE). Derived tables and results: CC BY 4.0. Hugging Face metadata remains subject to
the Hub's terms of service.
