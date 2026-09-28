# Reclassification of `license:other` (snapshot 2026-09-25)

Descriptive statistics only. No removal simulation, hypothesis test, or functional-option-loss figures were computed for this report (at the time, the study was planned as a Registered Report and those analyses were held back until after Stage 1).

## 1. Procedure

- **Target selection** (`03_code/02_graph/fetch_license_names.py`): among models whose effective license (`attributes.parquet`, inheritance applied) is `other`,
  (a) the **315 models** with 50 or more descendants (`eco.reach`, excluding the model itself, all tiers), plus
  (b) the top 100 `other` models by `downloads_all`. The union of the two sets has **368 models**
  (268 by the descendant criterion only, 53 by the download criterion only, 47 by both). Of the 368, 364 carry the HF tag `license:other` themselves;
  4 have no tag and inherited `other` from a parent. 354 of the 368 root models are in T1.
- **Collection**: `cardData.license_name` / `cardData.license_link` were read from `GET https://huggingface.co/api/models/<id>?expand[]=cardData`
  (0.7 s between requests, backoff on 429). **All 368 requests returned HTTP 200; 0 failures.**
  The raw responses are in `02_data/raw/license_names_2026-09-25.jsonl`.
- **Classification**: for each of the 69 distinct `license_name` values (including "no name"), the linked license text (HF LICENSE file, vendor page, PDF)
  was downloaded and its commercial-use and derivative-work clauses were checked. For the 3 whose text could not be retrieved (ghost-open-llms, bria-rmbg-1.4, the nemotron page),
  the model card or vendor page was checked separately on the web. Class criteria follow `license_map.csv` and the rules in `removal_sim.py`
  (llama/gemma/qwen/deepseek/falcon → vendor_custom, etc.).
  Models without a `license_name` were left as `other` rather than inferred from the base model.
- Output: `03_code/03_simulation/license_overrides.csv` (368 model rows, with `evidence_url` and `note`).

## 2. Reclassification result (number of models)

| New class | Models |
|---|---:|
| vendor_custom | 161 |
| noncommercial | 55 |
| permissive | 19 |
| copyleft | 9 |
| responsible_ai | 2 |
| **Reclassified, total** | **246** |
| Kept as other (no license_name in cardData) | 122 |
| Total | 368 |

## 3. Descendants under the reclassified roots (T1)

Descendants are the union of the roots' `eco.reach` minus the roots themselves. Because roots are nested, the per-class rows do not add up to the total.
"Current class" is the class assigned to the descendant model by the existing pipeline (before the overrides).

| Root class | Roots (in T1) | Descendants, all | Descendants, T1 | T1 descendants currently `other` | currently `unknown` |
|---|---:|---:|---:|---:|---:|
| All reclassified | 246 (235) | 159,629 | 81,871 | 43,770 | 18,248 |
| noncommercial | 55 (50) | 122,311 | 53,323 | 29,466 | 10,835 |
| vendor_custom | 161 (156) | 31,275 | 22,535 | 10,049 | 7,009 |
| copyleft | 9 (9) | 4,808 | 4,787 | 3,364 | 429 |
| permissive | 19 (18) | 1,495 | 1,430 | 900 | 102 |
| responsible_ai | 2 (2) | 281 | 281 | 245 | 5 |
| (reference) kept as other | 122 (119) | 14,515 | 12,132 | 3,060 | 5,388 |

The `tongyi-qianwen-research` roots (Qwen1.5-0.5B/1.8B/4B) have 64,326 descendants, but only 919 of them are in T1.
Most descendants under these roots are models removed by the T1 cleaning.

## 4. Top 10 reclassified models (by descendants)

| Model | license_name | Previous → new class | Descendants | Descendants, T1 |
|---|---|---|---:|---:|
| black-forest-labs/FLUX.1-dev | flux-1-dev-non-commercial-license | other → noncommercial | 43,688 | 39,585 |
| Qwen/Qwen1.5-0.5B | tongyi-qianwen-research | other → noncommercial | 32,595 | 331 |
| Qwen/Qwen1.5-1.8B | tongyi-qianwen-research | other → noncommercial | 30,671 | 304 |
| Qwen/Qwen2.5-3B | qwen-research | other → noncommercial | 7,950 | 6,736 |
| Qwen/Qwen1.5-7B | tongyi-qianwen | other → vendor_custom | 6,412 | 158 |
| Qwen/Qwen2.5-3B-Instruct | qwen-research | other → noncommercial | 5,751 | 5,181 |
| OnomaAIResearch/Illustrious-xl-early-release-v0 | fair-ai-public-license-1.0-sd | other → copyleft | 3,466 | 3,457 |
| NousResearch/Meta-Llama-3-8B | llama3 | other → vendor_custom | 2,765 | 1,886 |
| NousResearch/Meta-Llama-3-8B-Instruct | llama3 | other → vendor_custom | 2,011 | 1,795 |
| VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct | llama3 | other → vendor_custom | 1,723 | 1,531 |

Among the roots kept as `other`, the ones with the most descendants are Llama-3-family merges and fine-tunes
(e.g., Locutusque/llama-3-neural-chat-v1-8b 1,939; TheDrummer/Rocinante-12B-v1 1,930; mlabonne/OrpoLlama-3-8B 1,796).
Their cards have no license_name, so no class was inferred.

## 5. Classification and evidence by license_name

The "Models" column is the number of inspected models declaring that name. "Descendants, T1" is the size of the union of T1 descendants under those roots.

| license_name | Class | Models | Descendants, T1 | Evidence (summary) |
|---|---|---:|---:|---|
| flux-1-dev-non-commercial-license | noncommercial | 6 | 40,190 | FLUX.1 [dev] Non-Commercial License v1.1.1: non-commercial, non-production use only ([LICENSE.md](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)) |
| qwen-research | noncommercial | 7 | 7,555 | Qwen RESEARCH LICENSE: "FOR NON-COMMERCIAL PURPOSES ONLY" ([link](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE)) |
| fair-ai-public-license-1.0-sd | copyleft | 8 | 4,546 | FAIPL-1.0-SD: same-license and source-availability obligations for derivatives and network use, plus a list of prohibited uses ([link](https://freedevproject.org/faipl-1.0-sd/)) |
| llama3 | vendor_custom | 30 | 2,981 | Meta Llama 3 Community License (same as `llama3` in license_map) |
| lfm1.0 | vendor_custom | 19 | 2,775 | LFM Open License v1.0: commercial use allowed only below USD 10M annual revenue |
| krea-2-community-license | vendor_custom | 3 | 2,544 | Krea 2 Community License: commercial use allowed only below USD 1M annual revenue (PDF) |
| mrl | noncommercial | 13 | 1,911 | Mistral AI Research License: research purposes only ([link](https://mistral.ai/licenses/MRL-0.1.md)) |
| qwen | vendor_custom | 10 | 1,768 | Qwen LICENSE AGREEMENT: commercial use allowed; a separate license is required above 100M MAU |
| deepseek / deepseek-license | vendor_custom | 10 / 4 | 1,690 / 917 | DeepSeek License v1.0: royalty-free grant including commercial use, with OpenRAIL-style use restrictions (vendor_custom, as in the existing rules) |
| nvidia-open-model-license | vendor_custom | 13 | 1,317 | "Models are commercially usable. You are free to create and distribute Derivative Models" |
| eva-llama3.3 | vendor_custom | 4 | 1,308 | Model card: Llama 3.3 license terms plus one added clause prohibiting use by a specific company |
| health-ai-developer-foundations | vendor_custom | 6 | 1,237 | Google HAI-DEF Terms (MedGemma): use, modification, and distribution allowed; use restrictions pass on to derivatives |
| tongyi-qianwen-research | noncommercial | 6 | 919 | Tongyi Qianwen RESEARCH LICENSE: non-commercial use only |
| nvidia-nemotron-open-model-license | vendor_custom | 6 | 817 | Commercial use and derivatives allowed; NOTICE attribution required |
| flux-non-commercial-license | noncommercial | 4 | 738 | FLUX Non-Commercial License v2.1 (FLUX.2 [dev]/klein 9B) |
| stabilityai-ai-community | vendor_custom | 3 | 712 | Stability AI Community License: commercial use allowed below USD 1M annual revenue |
| modified-mit | permissive | 9 | 684 | Kimi K2.x / MiniMax M2.x: MIT terms plus only an obligation to display the model name in large-scale or commercial products |
| falcon-llm-license | vendor_custom | 7 | 581 | TII Falcon License (Apache 2.0-based, with an AUP) |
| exaone | noncommercial | 3 | 562 | EXAONE AI Model License 1.1-NC: research purposes only, commercial use prohibited |
| ltx-2-community-license-agreement | vendor_custom | 4 | 550 | LTX-2 Community License: companies with annual revenue of USD 10M or more need a paid license |
| qwen-community-1.0 | vendor_custom | 1 | 536 | Qwen Community License 1.0: MaaS and AI-assistant businesses need a separate license |
| tongyi-qianwen | vendor_custom | 7 | 536 | Tongyi Qianwen LICENSE AGREEMENT: a separate license is required above 100M MAU |
| llama-3 | vendor_custom | 2 | 514 | Links to the Llama 3 license |
| minimax-h3-community-license-agreement | vendor_custom | 2 | 412 | Excludes the EU, UK, South Korea, and US; approval required above USD 20M annual revenue |
| openmdw-1.1 / openmdw1.1-license | permissive | 4 / 1 | 336 / 55 | OpenMDW-1.1 (permissive in license_map) |
| stabilityai-nc-research-community | noncommercial | 1 | 327 | Stability AI Non-Commercial Research Community License (SD3-medium-diffusers) |
| circlestone-labs-non-commercial-license | noncommercial | 1 | 318 | CircleStone Labs Non-Commercial License v1.2 |
| apache-2.0 / apache-license-2.0 | permissive | 2 / 1 | 285 / 2 | The linked file is the Apache 2.0 text |
| sdxl-license | responsible_ai | 2 | 281 | The linked file is CreativeML Open RAIL++-M |
| faipl-1.0 | copyleft | 1 | 241 | Fair AI Public License 1.0 |
| other (MiniMax-M2.7) | noncommercial | 1 | 209 | The linked LICENSE is the MiniMax NON-COMMERCIAL LICENSE (commercial use requires prior written approval) |
| tencent-hunyuan-community | vendor_custom | 3 | 200 | Excludes the EU, UK, and South Korea; a separate license is required above 100M MAU |
| llama4 | vendor_custom | 2 | 184 | Llama 4 Community License |
| sai-nc-community | vendor_custom | 1 | 168 | The name says NC, but the linked sdxl-turbo LICENSE.md is currently the Stability AI Community License (2024-07). **Classified by the current text** |
| mnpl | noncommercial | 2 | 166 | Mistral AI Non-Production License |
| ghost-open-llms | vendor_custom | 2 | 160 | Model card: Ghost Open LLMs + Llama 3 license; commercial use is free, but confirmation is requested (license page unreachable) |
| glm-5.3 | vendor_custom | 1 | 154 | MIT-style license; MaaS operators with revenue above USD 10B need a security review |
| dinov3-license | vendor_custom | 3 | 139 | Meta DINOv3 License: royalty-free grant, no non-commercial clause |
| kimi-k3 | vendor_custom | 1 | 116 | MIT-style license; MaaS operators with revenue above USD 20M need a separate agreement |
| microsoft-research-license | noncommercial | 3 | 112 | MSR License Terms: non-commercial research use only. TheBloke/phi-2-GGUF also declares this name, although the original phi-2 later changed to MIT |
| nvidia-open-model-agreement | vendor_custom | 2 | 105 | "Works are commercially usable" |
| yandexgpt-5-lite-8b | vendor_custom | 1 | 102 | Commercial use allowed; above 10M output tokens per month, an agreement with Yandex is required |
| minimax-community | vendor_custom | 1 | 99 | MiniMax M3: notification required for commercial use; approval required above USD 20M annual revenue |
| ltx-2.x-community-license-agreement | vendor_custom | 1 | 97 | Companies with annual revenue of USD 10M or more need a paid license |
| swift-open-license-1.0 | vendor_custom | 1 | 95 | Commercial use allowed only below USD 1M revenue |
| coqui-public-model-license | noncommercial | 1 | 81 | CPML 1.0: "allows only non-commercial use" |
| katanemo-research | noncommercial | 1 | 79 | Commercial use requires a separate license |
| gemma-terms-of-use | vendor_custom | 1 | 71 | Gemma Terms (same as `gemma` in license_map) |
| ideogram-4-non-commercial | noncommercial | 1 | 70 | Ideogram Non-Commercial Model Agreement |
| yi-34b | permissive | 1 | 67 | The linked Yi-34B LICENSE is currently Apache 2.0 (relicensed). **Classified by the current text** |
| apollo-v0.1-4b-thinking | noncommercial | 1 | 66 | "strictly for Non-Commercial Purposes" |
| inf | vendor_custom | 1 | 66 | INF license: allows even selling the model and derivatives; revocable |
| fish-audio-research-license | noncommercial | 1 | 63 | Commercial use requires a separate license |
| glm-4 | vendor_custom | 1 | 61 | Free commercial use after registration |
| hyperclovax-seed | vendor_custom | 2 | 61 | A separate license is required above 10M MAU or for competing services |
| qwen3.8-max | vendor_custom | 1 | 56 | MaaS operators with revenue above USD 50M need a separate license |
| falcon-mamba-7b-license | vendor_custom | 1 | 47 | Falcon Mamba TII License |
| livekit-model-license | vendor_custom | 1 | 15 | Usable only together with the LiveKit Agents framework |
| playground-v2dot5-community | vendor_custom | 1 | 10 | Image-generation businesses with more than 1M monthly unique users need a license |
| bria-rmbg-2.0 | noncommercial | 1 | 8 | Links to CC BY-NC 4.0 |
| stable-video-diffusion-community | vendor_custom | 1 | 8 | Stability AI Community License |
| tongyi-qianwen-license-agreement | vendor_custom | 1 | 7 | Tongyi Qianwen LICENSE AGREEMENT |
| bria-rmbg-1.4 | noncommercial | 1 | 5 | Model card: non-commercial CC license; commercial use requires an agreement (license page 404) |
| public-domain | permissive | 1 | 1 | US Government work, public-domain notice |
| nvidia-license | noncommercial | 1 | 0 | The linked NVIDIA card: non-commercial and academic research use only |
| (none) | kept as other | 122 | 12,132 | No license_name/license_link in cardData |

## 6. Caveats

- **Name and text disagree**: for `sai-nc-community` (sdxl-turbo) and `yi-34b`, the declared name differs from the currently linked text.
  Both were classified by the text as of the lookup date (2026-09-26), with a remark in the note column.
  Conversely, `microsoft-research-license` (phi-2-GGUF) was classified by the name declared on the card.
- **What vendor_custom means**: vendor_custom denotes vendor licenses that allow commercial use but attach conditions (revenue or MAU thresholds, regional exclusions, registration or notification).
  `modified-mit` only adds an attribution obligation, so it was classified as permissive. In the strict-commercial variant, this boundary can change results.
- **Scope of the overrides**: the overrides apply only to the 368 roots that were looked up. How licenses propagate to descendants (the inheritance rule)
  is decided in the pipeline integration step. This report only describes descendant counts and assumes no propagation.
