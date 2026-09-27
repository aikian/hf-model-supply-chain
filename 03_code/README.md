# 파이프라인

```bash
pip install -r requirements.txt

# 1) 수집: 전수 스냅샷. 중단되면 같은 명령으로 재개된다
python 01_collect/collect_hf_models.py --snapshot 2026-09-25
#    파일럿: --max-pages 3  (페이지당 1,000개)

# 2) 그래프: 노드/엣지 parquet + summary.json
python 02_graph/build_graph.py ../02_data/raw/hf_models_2026-09-25.jsonl.gz

# 2b) 추론 엣지 (미러, 이름 속 부모, 양자화 접미사) + 순환 표시 → edges_all.parquet
python 02_graph/infer_edges.py ../02_data/processed/2026-09-25

# 2b') 'other' 라이선스 재분류용 모델 카드 조회 (결과 → 03_simulation/license_overrides.csv, 매핑은 원문 확인 후 수작업)
python 02_graph/fetch_license_names.py ../02_data/processed/2026-09-25 --out ../02_data/raw/license_names_2026-09-25.jsonl

# 2c) 속성 상속 (+ license_overrides.csv 적용): 양자화·미러의 빈 태스크/언어/라이선스를 부모 값으로 → attributes.parquet
python 02_graph/enrich_attributes.py ../02_data/processed/2026-09-25

# 3) 정제: 제외 flag + tier (T0 전체 / T1 주 분석 / T2 사용된 모델) → 04_results/tables/cleaning_report_*.md
python 02_graph/clean_models.py ../02_data/processed/2026-09-25

# 4) 시뮬레이션 (RR: Stage 1 승인 전에는 --sample 로 작동 확인만)
python 03_simulation/removal_sim.py ../02_data/processed/2026-09-25 --tier T1 --sample 0.05
python 03_simulation/substitutability.py ../02_data/processed/2026-09-25 --tier T1 --sample 0.05

#    강건성 변형: --tier T0|T2 --declared-only --no-inherit --option coarse
#                 --exclude-quantized --other-as-unknown --strict-commercial --keep-temporal-violations
#                 --no-license-overrides

# 4b) 가설 판정 (사전 등록 기준). 자체 점검: --selftest
python 04_analysis/hypothesis_tests.py ../04_results/tables/2026-09-25_full_T1

# 5) 파일럿 그림/표
python 04_analysis/pilot_figures.py ../02_data/processed/2026-09-25
```

## 수집 방식
- `GET https://huggingface.co/api/models?sort=createdAt&direction=1&limit=1000&expand[]=...`
  cursor 페이지네이션 (`Link: rel="next"`). 생성일 오름차순이라 수집 중 새 모델이 올라와도 누락이나 중복이 없다.
- 익명 rate limit은 5분에 500요청. `HF_TOKEN` 환경변수를 설정하면 한도가 늘어난다.
- `RateLimit` 헤더의 `t=`(리셋까지 남은 초)만큼 기다렸다가 재시도한다.

## 계보 엣지 파싱 (태그 기반)
| 태그 | 해석 |
|---|---|
| `base_model:<rel>:<parent>` | rel ∈ {finetune, adapter, quantized, merge} |
| `base_model:<parent>` (타입 없음) | 같은 부모의 타입 태그가 없으면 `unspecified` |
| `license:<id>` | 선언 라이선스 (여러 개면 첫 번째. `n_licenses`에 개수) |
| `dataset:<id>` | 학습 데이터셋 엣지 |
| ISO 639-1 두 글자 태그 | 언어 (**3글자 코드는 제외. 한계**) |

- `temporal_ok` = 부모 createdAt ≤ 자식 createdAt (부모가 스냅샷에 있을 때만 계산)
- `parent_in_snapshot = False`: 삭제됐거나 비공개인 부모. 이 자체가 "이미 사라진 공급원"이라는 분석 포인트가 된다.

## 알려진 한계 (논문 Threats to Validity에 반영)
- `base_model`은 자가 신고라 누락이 많다 (선행 연구 기준 15–26%만 선언).
- 태그 기반이라 모델 카드 본문에만 적힌 계보는 놓친다.
- `downloads`는 최근 30일 값이고 `downloadsAllTime`은 누적 값이다.
