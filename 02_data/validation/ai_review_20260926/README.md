# AI labeling record — 2026-09-26

As requested, the 300 rows of `../paper_validation_to_label.csv` were labeled and the labels were saved to the same CSV.
The labeler is **an AI (OpenAI GPT-6 Astra, run in the Codex agent environment; notes are tagged `AI(Codex)`)**. This is not, and does not replace, an independent human labeling by the author.
This CSV must not be reported as human labels, whether as evidence of human–LLM agreement or of human validation.
If an independent human labeling is needed, use the blank form in the backup without looking at the AI results.

## Method

- Applied `../codebook.md` and the criteria stated in the author's request.
- `paper_validation_llm_labels.csv` and `paper_validation_key.csv` were not opened. Which cleaning rule flagged each model was not looked up either.
- Evidence came from the public Hugging Face API (model info and file list), the raw README, and the uploader's model list and profile. This was not a human visual inspection in a browser.
- The model-info API returned HTTP 200 for all 300 models. Model lists of the 284 unique accounts were also fetched. Account lists were capped at 100 models (50 on retry), so accounts were not inspected exhaustively. Account model counts in `note` are the public totals shown on the profile.
- README responses: 238 × 200, 60 × 404, 2 × 401. A missing or access-restricted README alone was not taken to mean that the model was deleted or undecidable; the file list and account evidence were applied together.
- Ambiguous cases were supplemented with the organization description, config files, file sizes, and public commits. For the Kate LoRA, only the safetensors header was read with an HTTP Range request instead of the whole file, to check the base model, trigger, and tensor layout.
- Weights were not run, and model performance and full-file integrity were not verified. This is a repository classification based on file presence, configs, descriptions, and account context.
- The collection scripts only gathered evidence. Final labels were assigned row by row after reviewing the evidence and recorded in `decisions.tsv`. Reasoning for borderline cases is in `note`.

## Results

| Label | Count |
|---|---:|
| 0 genuine | 116 |
| 1 artifact | 170 |
| 9 undecidable | 14 |
| Total | 300 |

Artifact types: A 56, B 73, C 5, D 36.

## Files and checks

- `paper_validation_to_label.before_ai.csv`: backup of the blank input before labeling.
- `decisions.tsv`: per-row decision ledger for the 300 rows. Row numbers start at 1, excluding the header.
- `evidence.jsonl`: model API and README responses with fetch timestamps.
- `accounts.jsonl`, `accounts_retry.jsonl`: account evidence and the retried results of failed requests.
- `extra.jsonl`, `kate_weight_header.json`: additional evidence for borderline cases.
- `evidence_index.csv`: per-model link table of public evidence URLs, labeler, and label.
- `verification.json`: check results and the SHA-256 of the final CSV.

Checked: 300 rows; unique model IDs; original row order and URLs preserved; required column order; no empty labels; values in {0, 1, 9}; a type in A/B/C/D whenever the label is 1; the AI tag in every note; UTF-8 BOM; and that the CSV re-reads cleanly.
Agreement with the existing comparison LLM labels and the precision of the cleaning rules were not computed here.
