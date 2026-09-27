# 데이터 사전

규칙: `raw/`는 API 덤프 원본만 두고 **수정 금지**. 모든 가공은 `03_code/`의 스크립트로 해서 `processed/`에 저장한다.
파일명에 스냅샷 날짜를 붙인다. 예: `hf_models_2026-10-01.parquet`

## raw/: HF 모델 메타데이터
| 필드 | 출처 | 설명 |
|---|---|---|
| model_id | HF API | `org/name` |
| author | HF API | 업로드 조직/사용자 |
| created_at | HF API | 업로드 시각 (temporal DAG용) |
| pipeline_tag | HF API | 태스크 |
| languages | model card | 언어 태그 |
| license | model card | 선언 라이선스 |
| base_model | model card | 부모 모델 (여러 개일 수 있음) |
| base_model_relation | model card | finetune / merge / quantized / adapter |
| downloads, likes | HF API | 인기도 (통제 변수) |

## processed/
| 파일 | 내용 |
|---|---|
| nodes.parquet | model_id + 속성 |
| edges.parquet | parent_id, child_id, relation, temporal_ok |
| dataset_edges.parquet | dataset_id → model_id |
| model_flags.parquet | 정제 flag (f_bot, f_boilerplate, f_course, f_test, f_empty, f_zero_downloads) + tier, in_T1, in_T2 |
| summary.json | 기초 통계 |

> **Note on flags / 정제 flag 주의.** The cleaning flags are name- and metadata-based heuristics used to
> decide which models count as functional-option providers. They are not judgments about the accounts or
> people who uploaded the models (validated precision per rule: 64–100%, see the paper). Please do not use
> them to single out individual users. / flag 는 제공자 판정용 휴리스틱이며 업로더에 대한 평가가 아니다.

## validation/
| 파일 | 내용 |
|---|---|
| dev_rule_development/ | 규칙 개발용 표본 300개 (AI 판정). **논문 근거로 쓰지 않음** |
| (예정) paper_validation_sample.csv | 규칙 동결 후 새로 뽑는 논문용 표본. 저자 2인 독립 판정 |

## external_mgmt/ (2차 논문)
Crunchbase, GitHub 릴리즈 등. 1차 논문에서는 쓰지 않는다.
