# license:other 재분류 보고서 (task B11, 스냅샷 2026-09-25)

기술 통계만 담았다. 제거 시뮬레이션, 가설 검정, 기능 옵션 손실 수치는 계산하지 않았다 (Registered Report 제약).

## 1. 절차

- **대상 선정** (`03_code/02_graph/fetch_license_names.py`): 유효 라이선스(`attributes.parquet`, 상속 반영)가 `other`인 모델 가운데
  (a) 후손 수(`eco.reach`, 자기 자신 제외, 모든 tier)가 50 이상인 모델 **315개**, 그리고
  (b) `other` 모델 중 `downloads_all` 상위 100개를 더했다. 두 집합의 합집합은 **368개**이다
  (후손 조건만 268, 다운로드 조건만 53, 둘 다 47). 368개 중 364개는 HF 태그 자체가 `license:other`이고,
  4개는 태그가 없어 부모에게서 `other`를 상속받았다. 루트 모델 368개 중 354개가 T1이다.
- **수집**: `GET https://huggingface.co/api/models/<id>?expand[]=cardData` 에서 `cardData.license_name` /
  `cardData.license_link`를 읽었다 (요청 간격 0.7초, 429 응답 시 백오프). **368건 모두 HTTP 200이었고 실패는 0건이다.**
  원본 응답은 `02_data/raw/license_names_2026-09-25.jsonl`에 있다.
- **분류**: 서로 다른 `license_name` 69종(이름 없음 포함)마다 연결된 라이선스 원문(HF LICENSE 파일, 벤더 페이지, PDF)을
  내려받아 상업적 이용 조항과 파생물 조항을 확인했다. 원문을 받지 못한 3종(ghost-open-llms, bria-rmbg-1.4, nemotron 페이지)은
  모델 카드나 벤더 페이지를 웹에서 따로 확인했다. 클래스 기준은 `license_map.csv`와 `removal_sim.py`의 규칙
  (llama/gemma/qwen/deepseek/falcon → vendor_custom 등)에 맞췄다.
  `license_name`이 없는 모델은 base model에서 추정하지 않고 `other`로 두었다.
- 결과: `03_code/03_simulation/license_overrides.csv` (모델 368행, `evidence_url`과 `note` 포함).

## 2. 재분류 결과 (모델 수)

| 새 클래스 | 모델 수 |
|---|---:|
| vendor_custom | 161 |
| noncommercial | 55 |
| permissive | 19 |
| copyleft | 9 |
| responsible_ai | 2 |
| **재분류 합계** | **246** |
| other 유지 (cardData에 license_name 없음) | 122 |
| 합계 | 368 |

## 3. 재분류된 루트 아래의 후손 (T1)

후손은 루트들의 `eco.reach` 합집합에서 루트 자신을 뺀 것이다. 루트가 중첩되어 있으므로 클래스별 합은 전체와 일치하지 않는다.
"현재 클래스"는 기존 파이프라인(오버라이드 적용 전)에서 후손 모델에 매겨진 클래스이다.

| 루트 클래스 | 루트 수 (T1) | 후손 전체 | 후손 T1 | 후손 T1 중 현재 `other` | 현재 `unknown` |
|---|---:|---:|---:|---:|---:|
| 재분류 전체 | 246 (235) | 159,629 | 81,871 | 43,770 | 18,248 |
| noncommercial | 55 (50) | 122,311 | 53,323 | 29,466 | 10,835 |
| vendor_custom | 161 (156) | 31,275 | 22,535 | 10,049 | 7,009 |
| copyleft | 9 (9) | 4,808 | 4,787 | 3,364 | 429 |
| permissive | 19 (18) | 1,495 | 1,430 | 900 | 102 |
| responsible_ai | 2 (2) | 281 | 281 | 245 | 5 |
| (참고) other 유지 | 122 (119) | 14,515 | 12,132 | 3,060 | 5,388 |

`tongyi-qianwen-research` 루트(Qwen1.5-0.5B/1.8B/4B)는 후손이 64,326개이지만 그중 T1은 919개뿐이다.
이 루트 아래의 후손 대부분은 T1 정제에서 빠지는 모델이다.

## 4. 재분류 상위 10개 (후손 수 기준)

| 모델 | license_name | 이전 → 새 클래스 | 후손 | 후손 T1 |
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

`other`로 남은 루트 중 후손이 가장 많은 것은 Llama-3 계열 병합·파인튜닝 모델이다
(예: Locutusque/llama-3-neural-chat-v1-8b 1,939개, TheDrummer/Rocinante-12B-v1 1,930개, mlabonne/OrpoLlama-3-8B 1,796개).
이 모델들은 카드에 license_name이 없으므로 추정하지 않았다.

## 5. license_name별 분류와 근거

"모델" 열은 해당 이름을 쓰는 점검 대상 모델 수이다. "후손 T1" 열은 그 루트들 아래 T1 후손의 합집합 크기이다.

| license_name | 클래스 | 모델 | 후손 T1 | 근거 (요지) |
|---|---|---:|---:|---|
| flux-1-dev-non-commercial-license | noncommercial | 6 | 40,190 | FLUX.1 [dev] Non-Commercial License v1.1.1: 비상업·비프로덕션 용도만 허용 ([LICENSE.md](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)) |
| qwen-research | noncommercial | 7 | 7,555 | Qwen RESEARCH LICENSE: "FOR NON-COMMERCIAL PURPOSES ONLY" ([link](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE)) |
| fair-ai-public-license-1.0-sd | copyleft | 8 | 4,546 | FAIPL-1.0-SD: 파생물·네트워크 제공 시 동일 라이선스와 소스 제공 의무, 금지 용도 목록 ([link](https://freedevproject.org/faipl-1.0-sd/)) |
| llama3 | vendor_custom | 30 | 2,981 | Meta Llama 3 Community License (license_map `llama3`와 같음) |
| lfm1.0 | vendor_custom | 19 | 2,775 | LFM Open License v1.0: 연매출 1천만 달러 미만일 때만 상업 이용 허용 |
| krea-2-community-license | vendor_custom | 3 | 2,544 | Krea 2 Community License: 연매출 100만 달러 미만일 때만 상업 이용 허용 (PDF) |
| mrl | noncommercial | 13 | 1,911 | Mistral AI Research License: 연구 목적만 허용 ([link](https://mistral.ai/licenses/MRL-0.1.md)) |
| qwen | vendor_custom | 10 | 1,768 | Qwen LICENSE AGREEMENT: 상업 이용 허용, MAU 1억 초과 시 별도 라이선스 필요 |
| deepseek / deepseek-license | vendor_custom | 10 / 4 | 1,690 / 917 | DeepSeek License v1.0: 상업 이용을 포함한 무상 허여, OpenRAIL식 용도 제한 (기존 규칙과 같게 vendor_custom) |
| nvidia-open-model-license | vendor_custom | 13 | 1,317 | "Models are commercially usable. You are free to create and distribute Derivative Models" |
| eva-llama3.3 | vendor_custom | 4 | 1,308 | 모델 카드: Llama 3.3 라이선스 조건에 특정 회사 사용 금지 조항 하나 추가 |
| health-ai-developer-foundations | vendor_custom | 6 | 1,237 | Google HAI-DEF Terms (MedGemma): 사용·수정·배포 허용, 파생물에 용도 제한 승계 |
| tongyi-qianwen-research | noncommercial | 6 | 919 | Tongyi Qianwen RESEARCH LICENSE: 비상업 용도만 허용 |
| nvidia-nemotron-open-model-license | vendor_custom | 6 | 817 | 상업 이용과 파생물 허용, NOTICE 표기 의무 |
| flux-non-commercial-license | noncommercial | 4 | 738 | FLUX Non-Commercial License v2.1 (FLUX.2 [dev]/klein 9B) |
| stabilityai-ai-community | vendor_custom | 3 | 712 | Stability AI Community License: 연매출 100만 달러 미만 상업 이용 허용 |
| modified-mit | permissive | 9 | 684 | Kimi K2.x / MiniMax M2.x: MIT 조건에 대규모·상업 제품의 모델명 표시 의무만 추가 |
| falcon-llm-license | vendor_custom | 7 | 581 | TII Falcon License (Apache 2.0 기반, AUP 포함) |
| exaone | noncommercial | 3 | 562 | EXAONE AI Model License 1.1-NC: 연구 목적만 허용, 상업 이용 금지 |
| ltx-2-community-license-agreement | vendor_custom | 4 | 550 | LTX-2 Community License: 연매출 1천만 달러 이상 기업은 유료 라이선스 필요 |
| qwen-community-1.0 | vendor_custom | 1 | 536 | Qwen Community License 1.0: MaaS·AI 어시스턴트 사업은 별도 라이선스 필요 |
| tongyi-qianwen | vendor_custom | 7 | 536 | Tongyi Qianwen LICENSE AGREEMENT: MAU 1억 초과 시 별도 라이선스 필요 |
| llama-3 | vendor_custom | 2 | 514 | Llama 3 라이선스로 연결 |
| minimax-h3-community-license-agreement | vendor_custom | 2 | 412 | EU·영국·한국·미국 제외, 연매출 2천만 달러 초과 시 승인 필요 |
| openmdw-1.1 / openmdw1.1-license | permissive | 4 / 1 | 336 / 55 | OpenMDW-1.1 (license_map에서 permissive) |
| stabilityai-nc-research-community | noncommercial | 1 | 327 | Stability AI Non-Commercial Research Community License (SD3-medium-diffusers) |
| circlestone-labs-non-commercial-license | noncommercial | 1 | 318 | CircleStone Labs Non-Commercial License v1.2 |
| apache-2.0 / apache-license-2.0 | permissive | 2 / 1 | 285 / 2 | 연결된 파일이 Apache 2.0 원문 |
| sdxl-license | responsible_ai | 2 | 281 | 연결된 파일이 CreativeML Open RAIL++-M |
| faipl-1.0 | copyleft | 1 | 241 | Fair AI Public License 1.0 |
| other (MiniMax-M2.7) | noncommercial | 1 | 209 | 연결된 LICENSE가 MiniMax NON-COMMERCIAL LICENSE (상업 이용은 사전 서면 승인 필요) |
| tencent-hunyuan-community | vendor_custom | 3 | 200 | EU·영국·한국 제외, MAU 1억 초과 시 별도 라이선스 필요 |
| llama4 | vendor_custom | 2 | 184 | Llama 4 Community License |
| sai-nc-community | vendor_custom | 1 | 168 | 이름은 NC이지만 연결된 sdxl-turbo LICENSE.md가 현재 Stability AI Community License(2024-07)임. **현재 원문 기준으로 분류** |
| mnpl | noncommercial | 2 | 166 | Mistral AI Non-Production License |
| ghost-open-llms | vendor_custom | 2 | 160 | 모델 카드: Ghost Open LLMs + Llama 3 라이선스, 상업 이용은 무료이나 확인 요청 (라이선스 페이지 접속 불가) |
| glm-5.3 | vendor_custom | 1 | 154 | MIT식 라이선스, 매출 100억 달러 초과 MaaS 사업자는 보안 심사 필요 |
| dinov3-license | vendor_custom | 3 | 139 | Meta DINOv3 License: 무상 허여, 비상업 조항 없음 |
| kimi-k3 | vendor_custom | 1 | 116 | MIT식 라이선스, 매출 2천만 달러 초과 MaaS 사업자는 별도 계약 필요 |
| microsoft-research-license | noncommercial | 3 | 112 | MSR License Terms: 비상업 연구 목적만 허용. TheBloke/phi-2-GGUF도 이 이름을 선언했으나 원본 phi-2는 이후 MIT로 바뀜 |
| nvidia-open-model-agreement | vendor_custom | 2 | 105 | "Works are commercially usable" |
| yandexgpt-5-lite-8b | vendor_custom | 1 | 102 | 상업 이용 허용, 월 출력 토큰 1천만 개 초과 시 Yandex와 협의 필요 |
| minimax-community | vendor_custom | 1 | 99 | MiniMax M3: 상업 이용 시 통지 필요, 연매출 2천만 달러 초과 시 승인 필요 |
| ltx-2.x-community-license-agreement | vendor_custom | 1 | 97 | 연매출 1천만 달러 이상 기업은 유료 라이선스 필요 |
| swift-open-license-1.0 | vendor_custom | 1 | 95 | 매출 100만 달러 미만일 때만 상업 이용 허용 |
| coqui-public-model-license | noncommercial | 1 | 81 | CPML 1.0: "allows only non-commercial use" |
| katanemo-research | noncommercial | 1 | 79 | 상업 이용은 별도 라이선스 필요 |
| gemma-terms-of-use | vendor_custom | 1 | 71 | Gemma Terms (license_map `gemma`와 같음) |
| ideogram-4-non-commercial | noncommercial | 1 | 70 | Ideogram Non-Commercial Model Agreement |
| yi-34b | permissive | 1 | 67 | 연결된 Yi-34B LICENSE가 현재 Apache 2.0 (재라이선스됨). **현재 원문 기준으로 분류** |
| apollo-v0.1-4b-thinking | noncommercial | 1 | 66 | "strictly for Non-Commercial Purposes" |
| inf | vendor_custom | 1 | 66 | INF 라이선스: 모델·파생물 판매까지 허용, 철회 가능 |
| fish-audio-research-license | noncommercial | 1 | 63 | 상업 이용은 별도 라이선스 필요 |
| glm-4 | vendor_custom | 1 | 61 | 등록 후 무료 상업 이용 가능 |
| hyperclovax-seed | vendor_custom | 2 | 61 | MAU 1천만 초과 또는 경쟁 서비스는 별도 라이선스 필요 |
| qwen3.8-max | vendor_custom | 1 | 56 | 매출 5천만 달러 초과 MaaS 사업자는 별도 라이선스 필요 |
| falcon-mamba-7b-license | vendor_custom | 1 | 47 | Falcon Mamba TII License |
| livekit-model-license | vendor_custom | 1 | 15 | LiveKit Agents 프레임워크와 함께 쓸 때만 이용 가능 |
| playground-v2dot5-community | vendor_custom | 1 | 10 | 월 순사용자 100만 초과 이미지 생성 사업은 라이선스 필요 |
| bria-rmbg-2.0 | noncommercial | 1 | 8 | CC BY-NC 4.0 링크 |
| stable-video-diffusion-community | vendor_custom | 1 | 8 | Stability AI Community License |
| tongyi-qianwen-license-agreement | vendor_custom | 1 | 7 | Tongyi Qianwen LICENSE AGREEMENT |
| bria-rmbg-1.4 | noncommercial | 1 | 5 | 모델 카드: 비상업 CC 라이선스, 상업 이용은 계약 필요 (라이선스 페이지 404) |
| public-domain | permissive | 1 | 1 | 미국 정부 저작물 공공 영역 고지 |
| nvidia-license | noncommercial | 1 | 0 | 연결된 NVIDIA 카드: 비상업·학술 연구 목적만 허용 |
| (없음) | other 유지 | 122 | 12,132 | cardData에 license_name/license_link 없음 |

## 6. 주의점

- **이름과 원문이 어긋나는 경우**: `sai-nc-community`(sdxl-turbo)와 `yi-34b`는 선언된 이름과 현재 연결된 원문이 다르다.
  두 경우 모두 조회일(2026-09-26) 기준 원문으로 분류했고 note 열에 적어 두었다.
  반대로 `microsoft-research-license`(phi-2-GGUF)는 카드에 선언된 이름대로 분류했다.
- **vendor_custom 판정**: vendor_custom은 상업 이용이 가능하지만 조건(매출·MAU 기준, 지역 제외, 등록·통지)이 붙은 벤더 라이선스를 뜻한다.
  `modified-mit`는 표시 의무만 추가되므로 permissive로 분류했다. strict-commercial 변형에서는 이 경계가 결과를 바꿀 수 있다.
- **오버라이드 범위**: 오버라이드는 조회한 368개 루트에만 적용된다. 후손의 라이선스를 어떻게 전파할지(상속 규칙)는
  파이프라인 연결 단계에서 정할 일이다. 이 보고서는 후손 수를 기술할 뿐 전파를 가정하지 않는다.
