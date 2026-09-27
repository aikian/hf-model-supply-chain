# 모델 목록 정제 보고서 (2026-09-25)

전체 모델 **3,094,856개**. flag는 겹칠 수 있다.

## 제외 사유별

| 사유 | 모델 수 | 비율 | 예시 |
|---|---:|---:|---|
| 자동 업로드: timestamp_suffix | 203,967 | 6.59% | `user-0905e0/blockassist-bc-tangled_slithering_alligator_1760593195`, `user-4f36a6/google-gemma-7b-1726420121` |
| 자동 업로드: blockassist | 99,051 | 3.20% | `user-13dfc7/blockassist-bc-sprightly_knobby_tiger_1756467332`, `user-5e5e39/blockassist-bc-mottled_foraging_ape_1755356352` |
| 자동 업로드: gensyn_swarm | 11,227 | 0.36% | `user-eb16ae/Qwen2.5-0.5B-Instruct-Gensyn-Swarm-fluffy_leaping_mole`, `user-474896/Qwen3-0.6B-Gensyn-Swarm-agile_gentle_prawn` |
| 자동 업로드: uuid_name | 107,084 | 3.46% | `user-14be10/53ae3e5c-8a9d-4454-af25-5033049d2df7`, `user-21de9d/66a0e9bf-80b5-4101-b669-c63d6410135d` |
| 자동 업로드: hex_hash_name | 359 | 0.01% | `user-4f5027/139263fea5da1ce75`, `user-4a717e/65646543241564345` |
| 자동 업로드: omega_miner | 12,156 | 0.39% | `user-ddd405/omega_hc944`, `user-f76750/omega_2f30y` |
| 자동 업로드: cuid_name | 2,716 | 0.09% | `user-0c7e4d/cmbngcrn3021sekg0z7nswiiv_cmbomn07v0430ekg0a1zk9cm7`, `user-0c7e4d/cmbbnm79s08h585uuatqmooyn_cmbbnwhcx08j185uuoyz54ivv` |
| 자동 업로드: author | 1,565 | 0.05% | `gradients-io-tournaments/tournament-tourn_79c20b9b66ecda90_20260803-27840c46-dd63-4bfb-abd2-8b4ba556f84a-5EFLCMFD`, `gradients-io-tournaments/tournament-tourn_1682452b0289f70b_20260914-d01a245b-6b80-4abe-87fb-cc9d1afdadbf-5EUHojrM` |
| **자동 업로드 합계** | 346,202 | 11.19% | `user-d92a65/ed4c42a0-cd2d-437f-b272-4ce08fdedc75`, `user-b16380/16b9955f-1a26-4817-a328-ff55907c040a` |
| 튜토리얼 기본 이름 (작성자 50명 이상 공유) | 233,641 | 7.55% | `user-35e512/sd-naruto-model`, `user-0ad552/distilgpt2-finetuned-wikitext2` |
| 강의 과제 (Deep RL 등) | 78,594 | 2.54% | `user-03e413/rl_course_vizdoom_health_gathering_supreme`, `user-f1228b/ppo-LunarLander-v2` |
| 테스트/임시 이름 | 62,145 | 2.01% | `user-f83e55/XCoder-Unit-Test-Model`, `user-594d51/test` |
| 메타데이터 없음 (태스크·라이브러리·라이선스·부모 모두 없음) | 1,096,073 | 35.42% | `user-1b1aff/blockassist`, `user-8a23da/ec6a5666-d80a-48bd-aa67-573bf4e67a53` |
| **T1 제외 합계** | 1,463,256 | 47.28% | `user-cdc991/blockassist-bc-armored_stealthy_elephant_1761459840`, `user-3e0eee/combine_parquet_r1gui_org_grpo_qwen2_5_vl_3b_h20_step_350` |
| 누적 다운로드 0 (T2에서 추가 제외) | 427,991 | 13.83% | `user-04ce26/Yi-34B-Chat-a0.1-b0.1-L3-l1-e2`, `user-950b07/vvg` |

## Tier 규모

| Tier | 모델 수 | 비율 | 계보 엣지 (자식 기준) |
|---|---:|---:|---:|
| T0 전체 | 3,094,856 | 100.0% | 903,939 |
| T1 주 분석 | 1,631,600 | 52.7% | 697,691 |
| T2 사용된 모델 | 1,203,609 | 38.9% | 614,112 |

- 제외된 모델 중 자식이 있는 모델: **6,532개**. 그래프에는 남겨서 전파 경로로 쓴다.
- 대량 양자화 계정(mradermacher, TheBloke, RichardErkhov 등)은 **정상 배포자**라 제외하지 않는다.
- 패턴 규칙과 임계값은 이 스크립트 상단에 있다. 바꾸면 이 보고서를 다시 생성할 것.

## 자식이 많은 기반 모델: 정제 전후

| 기반 모델 | 자식 (T0) | 제외 비율 | 자식 (T1) |
|---|---:|---:|---:|
| `black-forest-labs/FLUX.1-dev` | 43,388 | 9% | 39,357 |
| `Qwen/Qwen1.5-0.5B` | 32,534 | 99% | 270 |
| `Qwen/Qwen1.5-1.8B` | 30,626 | 99% | 258 |
| `google/gemma-2b` | 24,035 | 96% | 946 |
| `distilbert/distilbert-base-uncased` | 12,960 | 51% | 6,306 |
| `stabilityai/stable-diffusion-xl-base-1.0` | 10,956 | 8% | 10,041 |
| `google/gemma-7b` | 9,586 | 94% | 588 |
| `Qwen/Qwen3-4B-Instruct-2507` | 8,280 | 9% | 7,564 |
| `lerobot/smolvla_base` | 7,981 | 17% | 6,605 |
| `google-bert/bert-base-uncased` | 7,151 | 12% | 6,297 |