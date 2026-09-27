"""모델 목록 정제: 모델을 지우지 않고 제외 사유 flag와 분석 tier를 붙인다.

입력  02_data/processed/<snap>/nodes.parquet
출력  02_data/processed/<snap>/model_flags.parquet   model_id + f_* flag + tier
      04_results/tables/cleaning_report_<snap>.md    flag별 개수, tier별 규모, 예시

Tier (사전 등록: 주 분석 T1, 강건성 T0/T2)
    T0  전체
    T1  자동 업로드, 강의 과제, 테스트, 빈 껍데기 제외
    T2  T1 ∩ 누적 다운로드 ≥ 1

제외된 모델도 그래프에는 남는다 (계보 전파 경로 보존). 시뮬레이션에서는
기능 옵션 "제공자"로만 세지 않는다.

사용
    python clean_models.py ../../02_data/processed/2026-09-25
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# 자동 업로드 (채굴·분산학습 보상, 대회 제출물): 이름 규칙이 기계적으로 생성됨
BOT_PATTERNS = {
    "timestamp_suffix": r"[-_]1[67]\d{8}$",          # FLock 등: Qwen-Qwen1.5-0.5B-1725623935
    "blockassist": r"blockassist",                    # Gensyn BlockAssist
    "gensyn_swarm": r"gensyn-swarm|rl-swarm",         # Gensyn RL Swarm
    "uuid_name": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "hex_hash_name": r"^[0-9a-f]{16,}$",
    "omega_miner": r"^omega_[a-z0-9]{5}$",            # Bittensor 계열
    "cuid_name": r"^c(?=[a-z0-9]*\d[a-z0-9]*\d[a-z0-9]*\d)[a-z0-9]{24}(?:_|$)",  # LoRA 학습 서비스 자동 ID (숫자 3개 이상)
}
# 이름 대신 계정 단위로 확인된 자동 제출 계정 (2026-09-25 검수에서 발견)
BOT_AUTHORS = {"gradients-io-tournaments"}
# HF Deep RL 강의 등 과제 제출물
COURSE_PATTERN = (r"lunarlander|huggy|snowballtarget|pyramids|soccertwos|cartpole|pixelcopter|"
                  r"spaceinvaders|pandareach|^ppo-|^dqn-|^a2c-|^q-frozenlake|^q-taxi|^taxi-v3|^reinforce-")
# RL 태스크 중 강의·게임 환경용 라이브러리 (RL로 학습한 LLM = transformers/peft 는 제외 대상 아님)
RL_COURSE_LIBS = {"stable-baselines3", "ml-agents", "sample-factory", "cleanrl", "hivex", "rl-algo-impls", "reinforce", "skrl"}
TEST_PATTERN = r"(?:^|[-_.])(?:test|tmp|temp|dummy|debug|placeholder)(?:[-_.\d]|$)"   # demo는 검수 결과 정상 모델이 많아 제외
BOILERPLATE_MIN_AUTHORS = 50   # 서로 다른 작성자 50명 이상이 똑같이 쓰는 이름 = 튜토리얼 기본값
BOILERPLATE_MAX_CHILDREN = 10  # 자식이 이만큼 있으면 튜토리얼 산출물이 아니라 공급원으로 본다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()
    snap = args.processed.name

    n = pd.read_parquet(args.processed / "nodes.parquet")
    name = n["model_id"].str.split("/", n=1).str[1].str.lower()
    f = pd.DataFrame({"model_id": n["model_id"]})

    for k, pat in BOT_PATTERNS.items():
        f[f"bot_{k}"] = name.str.contains(pat, regex=True)
    f["bot_author"] = n["author"].isin(BOT_AUTHORS)
    f["f_bot"] = f[[c for c in f if c.startswith("bot_")]].any(axis=1)

    authors_per_name = n.groupby(name)["author"].nunique()
    edges0 = pd.read_parquet(args.processed / "edges.parquet", columns=["child_id", "relation"])
    quantized = n["model_id"].isin(edges0.loc[edges0["relation"].eq("quantized"), "child_id"])
    # 양자화 배포본은 원본 이름을 그대로 쓰는 게 관례라 공유 이름 규칙에서 제외
    # 2026-09-26 수정: 공유 이름 규칙이 원본 유명 모델(FLUX.1-dev, Llama-3.1-8B-Instruct 등)과
    # 그 미러까지 잡고 있었다. 다음은 튜토리얼 산출물이 아니므로 제외한다.
    #   (a) 같은 이름 중 가장 먼저 올라온 모델 (원본)
    #   (b) 미러 추론 엣지로 원본에 연결된 모델 (재배포본: infer_edges.py)
    #   (c) 직계 자식이 BOILERPLATE_MAX_CHILDREN 개 이상인 모델 (다른 사람들이 가져다 쓰는 공급원)
    first = n.assign(nm=name).sort_values("created_at").drop_duplicates("nm")["model_id"]
    is_first = n["model_id"].isin(first)
    ea = args.processed / "edges_all.parquet"
    if ea.exists():
        eall = pd.read_parquet(ea, columns=["parent_id", "child_id", "source"])
        is_mirror = n["model_id"].isin(eall.loc[eall["source"].eq("inferred_mirror"), "child_id"])
        n_children = n["model_id"].map(eall.groupby("parent_id").size()).fillna(0)
    else:
        is_mirror = pd.Series(False, index=n.index)
        n_children = pd.Series(0, index=n.index)
    shared = name.map(authors_per_name).ge(BOILERPLATE_MIN_AUTHORS)
    f["f_boilerplate"] = (shared & ~quantized & ~is_first & ~is_mirror
                          & n_children.lt(BOILERPLATE_MAX_CHILDREN).to_numpy())
    rl = n["pipeline_tag"].eq("reinforcement-learning")
    f["f_course"] = (name.str.contains(COURSE_PATTERN, regex=True)
                     | (rl & (n["library_name"].isin(RL_COURSE_LIBS) | n["library_name"].isna())))
    f["f_test"] = name.str.contains(TEST_PATTERN, regex=True)
    # 추론 엣지(infer_edges.py)로 부모가 복원된 모델은 메타데이터 없음으로 보지 않는다
    ea = args.processed / "edges_all.parquet"
    has_parent = n["n_parents"].gt(0)
    if ea.exists():
        has_parent |= n["model_id"].isin(pd.read_parquet(ea, columns=["child_id"])["child_id"])
    f["f_empty"] = (n["pipeline_tag"].isna() & n["library_name"].isna()
                    & n["license"].isna() & ~has_parent)
    f["f_zero_downloads"] = n["downloads_all"].fillna(0).eq(0)

    excl = f[["f_bot", "f_boilerplate", "f_course", "f_test", "f_empty"]].any(axis=1)
    f["tier"] = "T0"
    f.loc[~excl, "tier"] = "T1"
    f.loc[~excl & ~f["f_zero_downloads"], "tier"] = "T2"
    f["in_T1"] = ~excl
    f["in_T2"] = ~excl & ~f["f_zero_downloads"]
    f.to_parquet(args.processed / "model_flags.parquet", index=False)

    # ------------------------------------------------ report
    N = len(f)
    edges = pd.read_parquet(args.processed / "edges.parquet")
    excluded_ids = set(f.loc[excl, "model_id"])
    excl_with_children = edges.loc[edges["parent_id"].isin(excluded_ids), "parent_id"].nunique()

    def row(label, mask):
        ex = n.loc[mask, "model_id"].sample(min(2, int(mask.sum())), random_state=0).tolist()
        return f"| {label} | {int(mask.sum()):,} | {100*mask.mean():.2f}% | {', '.join(f'`{e}`' for e in ex)} |"

    lines = [f"# 모델 목록 정제 보고서 ({snap})", "",
             f"전체 모델 **{N:,}개**. flag는 겹칠 수 있다.", "",
             "## 제외 사유별", "", "| 사유 | 모델 수 | 비율 | 예시 |", "|---|---:|---:|---|"]
    for k in list(BOT_PATTERNS) + ["author"]:
        lines.append(row(f"자동 업로드: {k}", f[f"bot_{k}"]))
    lines += [row("**자동 업로드 합계**", f["f_bot"]),
              row(f"튜토리얼 기본 이름 (작성자 {BOILERPLATE_MIN_AUTHORS}명 이상 공유)", f["f_boilerplate"]),
              row("강의 과제 (Deep RL 등)", f["f_course"]),
              row("테스트/임시 이름", f["f_test"]),
              row("메타데이터 없음 (태스크·라이브러리·라이선스·부모 모두 없음)", f["f_empty"]),
              row("**T1 제외 합계**", excl),
              row("누적 다운로드 0 (T2에서 추가 제외)", f["f_zero_downloads"] & ~excl),
              "", "## Tier 규모", "", "| Tier | 모델 수 | 비율 | 계보 엣지 (자식 기준) |", "|---|---:|---:|---:|"]
    for t, mask in [("T0 전체", pd.Series(True, index=f.index)), ("T1 주 분석", f["in_T1"]), ("T2 사용된 모델", f["in_T2"])]:
        ids = set(f.loc[mask, "model_id"])
        lines.append(f"| {t} | {int(mask.sum()):,} | {100*mask.mean():.1f}% | {int(edges['child_id'].isin(ids).sum()):,} |")
    lines += ["", f"- 제외된 모델 중 자식이 있는 모델: **{excl_with_children:,}개**. 그래프에는 남겨서 전파 경로로 쓴다.",
              "- 대량 양자화 계정(mradermacher, TheBloke, RichardErkhov 등)은 **정상 배포자**라 제외하지 않는다.",
              "- 패턴 규칙과 임계값은 이 스크립트 상단에 있다. 바꾸면 이 보고서를 다시 생성할 것."]
    top_parent = (edges.assign(excluded=edges["child_id"].isin(excluded_ids))
                  .groupby("parent_id")["excluded"].agg(["size", "mean"])
                  .sort_values("size", ascending=False).head(10))
    lines += ["", "## 자식이 많은 기반 모델: 정제 전후", "", "| 기반 모델 | 자식 (T0) | 제외 비율 | 자식 (T1) |", "|---|---:|---:|---:|"]
    for pid, r in top_parent.iterrows():
        lines.append(f"| `{pid}` | {int(r['size']):,} | {100*r['mean']:.0f}% | {int(r['size']*(1-r['mean'])):,} |")

    out = ROOT / "04_results" / "tables" / f"cleaning_report_{snap}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
