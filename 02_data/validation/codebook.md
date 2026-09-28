# Labeling codebook (validation for the paper)

Labels are assigned by **the author alone**; reliability is measured as κ against an **independent LLM labeler** given the same codebook (decision of 2026-09-26: no self-relabeling after two weeks). The final labels are the author's. Labeling is based on the model page (huggingface.co/<id>) and the file list.
**The labeler is not shown which rule flagged each model.**

## Labeling question
> Is this repository an **independent model supplier** that someone else could take and use?

| Value | Meaning |
|---|---|
| **1 = artifact** | Falls under one of types A–D below |
| **0 = genuine (an actual supplier)** | A model with a purpose. Quality and popularity do not matter |
| **9 = undecidable** | Page is empty or deleted. Excluded from the analysis; only the count is reported |

## Artifact types
- **A. Automated upload:** a repository produced mechanically by a reward, mining, or competition system
  - the name contains a timestamp, UUID, or random ID
  - the same account holds a large number of near-identical repositories
- **B. Course assignment:** an assignment submission for a public course or tutorial (e.g., `ppo-LunarLander-v2` from the Deep RL course, `bert-finetuned-ner` from the HF course)
- **C. Tutorial default name:** a repository that kept the default output name of tutorial code as is (`my_awesome_model`, `results`, `test-trainer`)
- **D. Test / temporary:** upload trials, debugging, temporary repositories

## Borderline cases
| Case | Label | Reason |
|---|---|---|
| Checkpoints or hyperparameter-search outputs of a research experiment (even with mechanical names) | 0 | Actual models built for research |
| A mirror that re-uploads the original unchanged | 0 | Still a supplier. Mirror status is handled by a separate rule (inferred edges) |
| Bulk quantization distributors (mradermacher, TheBloke, etc.) | 0 | Distributions that are actually used |
| A course assignment, but with a thorough model card and evaluation | 1 | Type B takes precedence |
| "demo" in the name, but an actual fine-tuning result | 0 | |
| Empty model card, but weights and config files are in order | 0 | Only the metadata is missing |

## Supplementary rules (added 2026-09-26)
Seven ambiguities that surfaced during the LLM labeling were made explicit **before the author's labeling**. The manuscript states this.

| # | Case | Label |
|---|---|---|
| S1 | Repository is empty (only `.gitattributes`, or an empty README) but **the type is evident from the name or account pattern** (bulk UUID accounts, `test`, `my_awesome_*`, course assignment names, etc.) | **1** + the corresponding type. **9** if the type cannot be determined |
| S2 | Name contains "test", but it is a fine-tune with a clear purpose or a pre-release of a named model family (e.g., `xls-r-hi-test`, `…-Fusion-test0`) | **0** |
| S2' | Name contains "test" and it is an ordinary trial run (e.g., `…-ft-test3`, `Qwen2-0.5B-GRPO-test`) | **1-D** |
| S3 | Name is mechanical (hash, timestamp, job ID), but **the model card or account shows a research context** | **0** (research checkpoint) |
| S3' | Name is mechanical and there are **traces of a reward, mining, or competition system** (Gensyn, Bittensor, subnet, competition accounts, etc.) | **1-A** |
| S4 | Course assignment name (a name assigned by the HF course, Deep RL course, etc.) | **1-B**. If it is the **default output name of tutorial code** (`results`, `test-trainer`, `lora_model`), **1-C**. If both apply, B |
| S5 | Repository that is not a model (code only, images only, tokenizer only) | **1-D** |
| S6 | Repository of a bulk quantization distributor **without weights**: **0** if it has a README, **9** if completely empty | 0 / 9 |
| S7 | Personal-archive repository that re-uploads someone else's model or checkpoint as is (re-posted civitai LoRAs, zip collections of voice models, etc.) | **0** (treated like a mirror) |

## Record fields
`model_id, rater, label(1/0/9), type(A/B/C/D/-), note`
