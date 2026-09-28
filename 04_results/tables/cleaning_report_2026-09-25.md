# Model cleaning report (2026-09-25)

**3,094,856** models in total. Flags can overlap.

## Exclusions by reason

| Reason | Models | Share | Examples |
|---|---:|---:|---|
| Automated upload: timestamp_suffix | 203,967 | 6.59% | `user-0905e0/blockassist-bc-tangled_slithering_alligator_1760593195`, `user-4f36a6/google-gemma-7b-1726420121` |
| Automated upload: blockassist | 99,051 | 3.20% | `user-13dfc7/blockassist-bc-sprightly_knobby_tiger_1756467332`, `user-5e5e39/blockassist-bc-mottled_foraging_ape_1755356352` |
| Automated upload: gensyn_swarm | 11,227 | 0.36% | `user-eb16ae/Qwen2.5-0.5B-Instruct-Gensyn-Swarm-fluffy_leaping_mole`, `user-474896/Qwen3-0.6B-Gensyn-Swarm-agile_gentle_prawn` |
| Automated upload: uuid_name | 107,084 | 3.46% | `user-14be10/53ae3e5c-8a9d-4454-af25-5033049d2df7`, `user-21de9d/66a0e9bf-80b5-4101-b669-c63d6410135d` |
| Automated upload: hex_hash_name | 359 | 0.01% | `user-4f5027/139263fea5da1ce75`, `user-4a717e/65646543241564345` |
| Automated upload: omega_miner | 12,156 | 0.39% | `user-ddd405/omega_hc944`, `user-f76750/omega_2f30y` |
| Automated upload: cuid_name | 2,716 | 0.09% | `user-0c7e4d/cmbngcrn3021sekg0z7nswiiv_cmbomn07v0430ekg0a1zk9cm7`, `user-0c7e4d/cmbbnm79s08h585uuatqmooyn_cmbbnwhcx08j185uuoyz54ivv` |
| Automated upload: author | 1,565 | 0.05% | `gradients-io-tournaments/tournament-tourn_79c20b9b66ecda90_20260803-27840c46-dd63-4bfb-abd2-8b4ba556f84a-5EFLCMFD`, `gradients-io-tournaments/tournament-tourn_1682452b0289f70b_20260914-d01a245b-6b80-4abe-87fb-cc9d1afdadbf-5EUHojrM` |
| **Automated upload, total (f_bot)** | 346,202 | 11.19% | `user-d92a65/ed4c42a0-cd2d-437f-b272-4ce08fdedc75`, `user-b16380/16b9955f-1a26-4817-a328-ff55907c040a` |
| Boilerplate card (f_boilerplate): tutorial default name shared by 50 or more authors | 233,641 | 7.55% | `user-35e512/sd-naruto-model`, `user-0ad552/distilgpt2-finetuned-wikitext2` |
| Course assignment (f_course): Deep RL course etc. | 78,594 | 2.54% | `user-03e413/rl_course_vizdoom_health_gathering_supreme`, `user-f1228b/ppo-LunarLander-v2` |
| Test name (f_test): test or temporary names | 62,145 | 2.01% | `user-f83e55/XCoder-Unit-Test-Model`, `user-594d51/test` |
| No metadata (f_empty): none of task, library, license, parent | 1,096,073 | 35.42% | `user-1b1aff/blockassist`, `user-8a23da/ec6a5666-d80a-48bd-aa67-573bf4e67a53` |
| **Excluded from T1, total** | 1,463,256 | 47.28% | `user-cdc991/blockassist-bc-armored_stealthy_elephant_1761459840`, `user-3e0eee/combine_parquet_r1gui_org_grpo_qwen2_5_vl_3b_h20_step_350` |
| Cumulative downloads 0 (additionally excluded from T2) | 427,991 | 13.83% | `user-04ce26/Yi-34B-Chat-a0.1-b0.1-L3-l1-e2`, `user-950b07/vvg` |

## Tier sizes

| Tier | Models | Share | Lineage edges (by child) |
|---|---:|---:|---:|
| T0 all | 3,094,856 | 100.0% | 903,939 |
| T1 main analysis (T1-original in the paper) | 1,631,600 | 52.7% | 697,691 |
| T2 used models (T2-original in the paper) | 1,203,609 | 38.9% | 614,112 |

- Excluded models that have children: **6,532**. They stay in the graph as propagation paths.
- Bulk quantization accounts (mradermacher, TheBloke, RichardErkhov, etc.) are **legitimate distributors** and are not excluded.
- The pattern rules and thresholds are at the top of `03_code/02_graph/clean_models.py`, which generates this report. Regenerate the report after changing them.

## Base models with the most children: before and after cleaning

| Base model | Children (T0) | Excluded share | Children (T1) |
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
