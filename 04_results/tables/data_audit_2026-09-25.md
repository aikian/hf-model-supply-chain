# Data audit report (snapshot 2026-09-25)

Auditor: Claude (AI assistant, in a data-analyst role), 2026-09-25. Sample labels: `02_data/validation/dev_rule_development/cleaning_validation_labeled.csv`

> **Caution: the figures in Section 1 are not reported in the paper.** The labeler was an AI, and the rules were revised on the same sample,
> so the precision estimates are optimistically biased. They are for rule development only. The validation reported in the paper follows the separate protocol described in the paper's data section (see `02_data/validation/codebook.md`).

## 1. Precision of the cleaning rules (rule-development sample of 300, AI labels)

Criterion: anything that is **not** an independently usable model supplier (bot, assignment, test, tutorial artifact) = noise (1)

| Stratum | Sample | Noise | Precision | 95% CI (Wilson) | False-positive types |
|---|---:|---:|---:|---|---|
| Automated upload (f_bot) | 40 | 40 | **100%** | 91.2–100 | none |
| Course assignment (f_course) | 40 | 37 | **92.5%** | 80.1–97.4 | official cleanrl benchmarks, research RL |
| Test name (f_test) | 40 | 35 | **87.5%** | 73.9–94.5 | "test" in research experiment names (DomainBed etc.), genuine fine-tunes made for demos |
| Boilerplate card (f_boilerplate) | 40 | 35 | **87.5%** | 73.9–94.5 | **mirrors** (re-uploads of the original), genuine models with generic names |
| No metadata (f_empty) | 40 | 20 | 50% | 35.2–64.8 | genuine models without tags (RichardErkhov quantizations etc.) |
| **Retained (T1)** | 100 | 8 | residual noise **8%** | 4.1–15.0 | automatic IDs from a LoRA service, Bittensor mining, `test.2` missed |

- **f_empty is not a noise rule but an "unanalyzable" rule.** Without task and license, no functional option can be formed. The paper reports it separately from noise removal.
- Changes made after the audit:
  - Added patterns found in the retained sample: cuid automatic IDs (2,716), `gradients-io-tournaments` (1,565), `test.N`
  - Removed `demo` from the test rule (too many false positives)
  - Restricted the course rule to libraries used for RL assignments (protects LLMs trained with RL)
- **Remaining limitation:** Bittensor mining accounts (`tomaszki` etc.) cannot be separated by name rules. Residual noise in T1 is estimated at about 5–8%.

## 2. Structural integrity checks and fixes

| # | Problem | Size | Fix | Where |
|---|---|---:|---|---|
| 1 | Model declares itself as its parent (self-loop) | 2,044 | removed | build_graph.py |
| 2 | Parent ID case mismatch → misclassified as "missing parent" | 2,343 | normalized to the snapshot spelling | build_graph.py |
| 3 | Legacy batch date 2022-03-02 (not the actual upload time) | 29,579 models | temporal check for these edges set to NA | build_graph.py |
| 4 | Cycles (strongly connected components) | 12, with 29 edges | excluded from the analysis (`in_cycle`) | infer_edges.py |
| 5 | **Mirrors**: copies that reuse a popular model's name with no parent declared → spurious independent lineages | 11,264 | original inferred as parent (relation=mirror) | infer_edges.py |
| 6 | **Parent only in the name** (RichardErkhov `author_-_model`) | 22,962 | parent recovered | infer_edges.py |
| 7 | Quantization name with no parent declared | 14,231 | parent recovered only when the name without the suffix matches **exactly one** model | infer_edges.py |

- Edges after the fixes: 903,939 declared + 48,457 inferred = **952,396**. 48,457 models gained a parent. Only 34 inferred edges (0.07%) are temporal reversals, so the inference is largely sound.
- 1.89% of edges have a parent that is not in the snapshot: parents that were deleted or made private, e.g. `runwayml/stable-diffusion-v1-5` (3,370 edges). **These are supply interruptions that have already happened** and can serve as motivating examples in the paper. (Later correction: `runwayml/stable-diffusion-v1-5` was not simply removed; the Hub redirects the ID to the re-upload `stable-diffusion-v1-5/stable-diffusion-v1-5`, and such edges were reconnected in a later pipeline step. See the paper's data section.)
- Of the edges for which the temporal check is possible, 99.58% have the parent uploaded first, consistent with the 99.73% reported by Horwitz et al.

## 3. Field quality

| Field | Status | Impact |
|---|---|---|
| License | 36% declared; 83 distinct values; `other` 118,553 | `other` was classed as vendor_custom but is ambiguous → re-analyzed as unknown in a robustness check |
| Language | only 14.7% tagged | most options have language `unk`. Robustness check with the coarse definition (task × license) |
| Task (pipeline_tag) | only 34% present | quantizations and mirrors could **inherit** the parent's task (not applied; proposed) |
| Downloads | no missing or negative values; max 3.9 billion (all-MiniLM-L6-v2) | heavy tail → report on a log scale |
| Dates | no future dates; legacy batch date 29,579 | fixed |

## 4. Before and after cleaning (a case where cleaning changes the conclusion)

Before cleaning, the base models ranked 2nd, 3rd, and 4th by number of children were Qwen1.5-0.5B, Qwen1.5-1.8B, and gemma-2b; 96–99% of their children were mining-bot uploads. After cleaning, Qwen1.5-0.5B's children drop from 32,534 to 270. **Without cleaning, the ranking of "core base models" is distorted by bots.** The comparison is tabulated in `cleaning_effect_2026-09-25.csv`.

## 5. Follow-ups proposed at the time of the audit
1. Let quantizations and mirrors inherit the parent's task and license → fewer "no metadata" models (implemented later in `03_code/02_graph/enrich_attributes.py`)
2. Compile a list of Bittensor mining accounts (account-level rule)
3. Validation for the paper: freeze the rules → **new sample** → two independent raters → precision + Cohen's κ (carried out with two language-model labelers; see `02_data/validation/codebook.md` and the paper)
