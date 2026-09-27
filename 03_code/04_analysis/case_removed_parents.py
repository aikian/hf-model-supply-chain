"""사례 분석: 이미 사라진 상위 모델 (자연 실험).

선언된 부모가 스냅샷에 없는 경우 = 삭제·비공개·이름 변경된 상위 공급원.
각 사라진 부모에 대해 다음을 기술한다 (가설 검정이 아닌 기술 통계):
  - 선언된 자식 수, 그중 T1 자식 수
  - 자식 업로드 시점 분포: 마지막 자식 업로드일 (부모가 사라진 뒤에도 파생이 계속됐는가?)
  - 대체 공급원: 같은 모델 이름(조직 제외)을 가진 스냅샷 내 모델 = 재배포본/이전 저장소 후보.
    그 후보의 생성일, 다운로드, 자식 수 → 원본이 사라진 뒤 생태계가 어디로 옮겨 갔는가
  - 사라진 부모의 자식들이 제공하던 기능 옵션(태스크 × 라이선스 계열) 중, 같은 옵션을
    사라진 계보 밖의 모델이 여전히 제공하는 비율 (대체 가능성의 실제 사례)

주의: HF API 로는 삭제 시점을 알 수 없다. "사라진 뒤"는 스냅샷 시점 기준이다.

출력: 04_results/tables/case_removed_parents_<snap>.csv, case_removed_parents_<snap>.md

사용
    python case_removed_parents.py ../../02_data/processed/2026-09-25 --min-children 20
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code" / "03_simulation"))
from removal_sim import license_class  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--min-children", type=int, default=20)
    args = ap.parse_args()
    P, snap = args.processed, args.processed.name

    n = pd.read_parquet(P / "nodes.parquet", columns=["model_id", "created_at", "downloads_all", "pipeline_tag"])
    attr = pd.read_parquet(P / "attributes.parquet", columns=["model_id", "task", "license"])
    flags = pd.read_parquet(P / "model_flags.parquet", columns=["model_id", "in_T1"])
    e = pd.read_parquet(P / "edges_all.parquet", columns=["parent_id", "child_id", "relation", "parent_in_snapshot", "source"])
    n = n.merge(attr, on="model_id").merge(flags, on="model_id")
    n["opt"] = n["task"].fillna("unk") + "|" + n["license"].map(license_class)
    n["name"] = n["model_id"].str.split("/", n=1).str[1].str.lower()
    idx = n.set_index("model_id")
    kids = e[e["parent_in_snapshot"]].groupby("parent_id").size()

    missing = e[~e["parent_in_snapshot"] & e["source"].eq("declared")]
    big = missing.groupby("parent_id").size()
    big = big[big >= args.min_children].sort_values(ascending=False)

    # T1 제공자가 제공하는 옵션별 제공자 집합 (대체 가능성 판단용)
    t1 = n[n["in_T1"]]
    rows = []
    for pid, n_decl in big.items():
        ch = missing.loc[missing["parent_id"] == pid, "child_id"]
        c = idx.loc[idx.index.intersection(ch)]
        c1 = c[c["in_T1"]]
        pname = pid.split("/", 1)[1].lower()
        alt = n[(n["name"] == pname) & (n["model_id"] != pid)]
        alt_best = alt.sort_values("downloads_all", ascending=False).head(1)
        # 사라진 부모 계열이 제공하던 옵션 중, 이 자식들 밖의 T1 모델이 여전히 제공하는 비율
        opts = set(c1["opt"])
        outside = t1[~t1["model_id"].isin(c1.index)]
        still = sum(1 for o in opts if (outside["opt"] == o).any()) if opts else 0
        rows.append({
            "missing_parent": pid,
            "declared_children": int(n_decl),
            "t1_children": int(len(c1)),
            "first_child": str(c["created_at"].min())[:10],
            "last_child": str(c["created_at"].max())[:10],
            "children_downloads": int(c["downloads_all"].fillna(0).sum()),
            "same_name_copies": int(len(alt)),
            "top_copy": alt_best["model_id"].iat[0] if len(alt_best) else "",
            "top_copy_created": str(alt_best["created_at"].iat[0])[:10] if len(alt_best) else "",
            "top_copy_children": int(kids.get(alt_best["model_id"].iat[0], 0)) if len(alt_best) else 0,
            "options_of_children": len(opts),
            "pct_options_still_provided_elsewhere": round(100 * still / len(opts), 1) if opts else None,
        })
    d = pd.DataFrame(rows)
    out = ROOT / "04_results" / "tables"
    d.to_csv(out / f"case_removed_parents_{snap}.csv", index=False)
    md = [f"# 사라진 상위 모델 사례 ({snap}, 선언된 자식 {args.min_children}개 이상)", "",
          f"사라진 부모 {len(d)}개, 이들을 선언한 자식 {int(d['declared_children'].sum()):,}개 "
          f"(T1 {int(d['t1_children'].sum()):,}개).", "", d.head(30).to_markdown(index=False)]
    (out / f"case_removed_parents_{snap}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:3]))
    print(d.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
