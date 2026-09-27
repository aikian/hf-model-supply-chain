"""raw 스냅샷(jsonl.gz) -> 노드/엣지 테이블 + 기초 통계.

출력 (02_data/processed/<snapshot>/):
    nodes.parquet          모델 1행: 속성, 라이선스, 언어, 부모 수
    edges.parquet          parent -> child, relation, temporal_ok, parent_in_snapshot
    dataset_edges.parquet  dataset -> model
    summary.json           파일럿 표에 쓸 기초 통계

사용:
    python build_graph.py ../../02_data/raw/hf_models_2026-09-25.jsonl.gz
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELATIONS = {"finetune", "adapter", "quantized", "merge"}

# ISO 639-1. 3글자 언어 코드는 태그만으로 기술 태그와 구분이 어려워 제외 (한계로 명시)
ISO639_1 = set("""aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs
cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht
hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg
li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa
pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti
tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu""".split())


def parse(rec):
    parents = {}          # parent_id -> relation
    untyped = set()
    licenses, languages, datasets = [], [], []
    for t in rec.get("tags") or []:
        if t.startswith("base_model:"):
            rest = t[len("base_model:"):]
            head, _, tail = rest.partition(":")
            if head in RELATIONS and tail:
                parents[tail] = head
            else:
                untyped.add(rest)
        elif t.startswith("license:"):
            licenses.append(t[len("license:"):])
        elif t.startswith("dataset:"):
            datasets.append(t[len("dataset:"):])
        elif t in ISO639_1:
            languages.append(t)
    for p in untyped:
        parents.setdefault(p, "unspecified")
    return parents, licenses, languages, datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    snap = args.raw.name.removeprefix("hf_models_").removesuffix(".jsonl.gz")
    out = args.out_dir or ROOT / "02_data" / "processed" / snap
    out.mkdir(parents=True, exist_ok=True)

    nodes, edges, dedges, seen = [], [], [], set()
    with gzip.open(args.raw, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mid = r["id"]
            if mid in seen:      # 재개 시 경계 페이지 중복 방지
                continue
            seen.add(mid)
            parents, lic, lang, ds = parse(r)
            nodes.append({
                "model_id": mid,
                "author": r.get("author") or mid.split("/")[0],
                "created_at": r.get("createdAt"),
                "last_modified": r.get("lastModified"),
                "pipeline_tag": r.get("pipeline_tag"),
                "library_name": r.get("library_name"),
                "license": lic[0] if lic else None,
                "n_licenses": len(lic),
                "languages": ",".join(sorted(set(lang))) or None,
                "downloads_30d": r.get("downloads"),
                "downloads_all": r.get("downloadsAllTime"),
                "likes": r.get("likes"),
                "gated": str(r.get("gated") or False),   # False | "auto" | "manual"
                "n_parents": len(parents),
                "n_datasets": len(ds),
            })
            edges += [{"parent_id": p, "child_id": mid, "relation": rel}
                      for p, rel in parents.items()]
            dedges += [{"dataset_id": d, "model_id": mid} for d in ds]

    N = pd.DataFrame(nodes)
    N["created_at"] = pd.to_datetime(N["created_at"], utc=True)
    E = pd.DataFrame(edges, columns=["parent_id", "child_id", "relation"])
    D = pd.DataFrame(dedges, columns=["dataset_id", "model_id"])

    # 부모 ID 정규화: HF ID는 대소문자를 구분하지 않으므로 스냅샷의 실제 표기로 맞춘다
    canon = pd.Series(N["model_id"].to_numpy(), index=N["model_id"].str.lower())
    canon = canon[~canon.index.duplicated()]
    E["parent_id_declared"] = E["parent_id"]
    E["parent_id"] = E["parent_id"].str.lower().map(canon).fillna(E["parent_id"])
    E = E.drop_duplicates(["parent_id", "child_id"])
    # 자기 자신을 부모로 선언한 엣지 제거 (메타데이터 오류)
    n_self = int((E["parent_id"] == E["child_id"]).sum())
    E = E[E["parent_id"] != E["child_id"]].reset_index(drop=True)

    created = N.set_index("model_id")["created_at"]
    # 2022-03-02는 HF가 이전 저장소를 일괄 이관한 날짜라 실제 업로드 시각이 아니다
    legacy = created.dt.strftime("%Y-%m-%d").eq("2022-03-02")
    N["created_at_legacy"] = N["model_id"].map(legacy).to_numpy()
    E["parent_in_snapshot"] = E["parent_id"].isin(created.index)
    pc = E["parent_id"].map(created)
    cc = E["child_id"].map(created)
    known_time = E["parent_in_snapshot"] & ~E["parent_id"].map(legacy).fillna(False) \
        & ~E["child_id"].map(legacy).fillna(False)
    E["temporal_ok"] = (pc <= cc).where(known_time)

    N.to_parquet(out / "nodes.parquet", index=False)
    E.to_parquet(out / "edges.parquet", index=False)
    D.to_parquet(out / "dataset_edges.parquet", index=False)

    indeg = E.groupby("child_id").size()
    outdeg = E.groupby("parent_id").size()
    known = E[E["parent_in_snapshot"]]
    summary = {
        "snapshot": snap,
        "n_models": len(N),
        "n_edges": len(E),
        "n_self_loops_removed": n_self,
        "n_parent_id_case_fixed": int((E["parent_id"] != E["parent_id_declared"]).sum()),
        "n_legacy_timestamp_models": int(legacy.sum()),
        "n_dataset_edges": len(D),
        "pct_models_with_parent": round(100 * (N["n_parents"] > 0).mean(), 2),
        "pct_models_with_license": round(100 * N["license"].notna().mean(), 2),
        "pct_models_with_language": round(100 * N["languages"].notna().mean(), 2),
        "pct_models_with_dataset": round(100 * (N["n_datasets"] > 0).mean(), 2),
        "relation_counts": E["relation"].value_counts().to_dict(),
        "multi_parent_models": int((indeg > 1).sum()),
        "pct_edges_parent_in_snapshot": round(100 * E["parent_in_snapshot"].mean(), 2) if len(E) else None,
        "pct_temporal_ok": round(100 * known["temporal_ok"].dropna().astype(bool).mean(), 2) if known["temporal_ok"].notna().any() else None,
        "n_edges_time_unknown": int(E["temporal_ok"].isna().sum()),
        "top10_parents_by_children": outdeg.sort_values(ascending=False).head(10).to_dict(),
        "top10_licenses": Counter(N["license"].dropna()).most_common(10),
        "top10_pipeline_tags": Counter(N["pipeline_tag"].dropna()).most_common(10),
        "created_at_range": [str(N["created_at"].min()), str(N["created_at"].max())],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
