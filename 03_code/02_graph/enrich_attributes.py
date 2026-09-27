"""속성 상속: 양자화·미러 모델의 빈 태스크/언어/라이선스를 부모 값으로 채운다.

근거: 양자화 모델과 미러는 부모와 가중치가 같아 기능(태스크, 언어)이 같고,
      라이선스상으로도 부모 라이선스의 적용을 받는다.
      파인튜닝·어댑터·병합은 기능이나 라이선스가 바뀔 수 있으므로 상속하지 않는다.
규칙: 자식 값이 **비어 있을 때만** 채운다 (선언값은 덮어쓰지 않는다).
      양자화의 양자화, 미러의 양자화처럼 여러 단계면 반복해서 전파한다.

입력  nodes.parquet, edges_all.parquet
출력  attributes.parquet   model_id, task, languages, license (상속 반영) + inherited_* 표시
      enrich_report.json

사용
    python enrich_attributes.py ../../02_data/processed/2026-09-25
"""
import argparse
import json
from pathlib import Path

import pandas as pd

INHERIT_RELATIONS = {"quantized", "mirror"}
FIELDS = {"task": "pipeline_tag", "languages": "languages", "license": "license"}
MAX_DEPTH = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()

    n = pd.read_parquet(args.processed / "nodes.parquet", columns=["model_id", *FIELDS.values()])
    e = pd.read_parquet(args.processed / "edges_all.parquet",
                        columns=["parent_id", "child_id", "relation", "parent_in_snapshot", "in_cycle"])
    e = e[e["relation"].isin(INHERIT_RELATIONS) & e["parent_in_snapshot"] & ~e["in_cycle"]]
    # 부모가 여럿인 양자화/미러는 모호 → 상속하지 않음
    e = e[~e["child_id"].duplicated(keep=False)]

    base = n.rename(columns={v: k for k, v in FIELDS.items()}).set_index("model_id")
    before = {k: int(base[k].isna().sum()) for k in FIELDS}

    def inherit(a):
        a = a.copy()
        for k in FIELDS:
            a[f"inherited_{k}"] = False
        rounds = 0
        for rounds in range(1, MAX_DEPTH + 1):
            changed = 0
            for k in FIELDS:
                pv = e["parent_id"].map(a[k])
                cv = e["child_id"].map(a[k])
                fill = cv.isna() & pv.notna()
                if fill.any():
                    ids = e.loc[fill, "child_id"].to_numpy()
                    a.loc[ids, k] = pv[fill].to_numpy()
                    a.loc[ids, f"inherited_{k}"] = True
                    changed += int(fill.sum())
            if not changed:
                break
        return a, rounds

    # 라이선스 재분류 (license_overrides.csv, fetch_license_names.py로 모델 카드의 license_name 확인):
    # 'other' 로 태그된 큰 루트의 실제 라이선스 계열. "override:<class>" 로 표기해 license_class()가 그대로 읽는다.
    # 상속보다 먼저 적용하므로, 라이선스가 비어 부모 값을 받던 양자화·미러도 바로잡힌 값을 받는다.
    ov_path = Path(__file__).resolve().parents[1] / "03_simulation" / "license_overrides.csv"
    a_raw, _ = inherit(base)
    b = base.copy()
    n_ov = 0
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        ov = ov[(ov["class"] != "other") & ov["model_id"].isin(b.index)]
        b.loc[ov["model_id"].to_numpy(), "license"] = ("override:" + ov["class"]).to_numpy()
        n_ov = len(ov)
    a, rounds = inherit(b)
    a["license_no_override"] = a_raw["license"]
    a.reset_index().to_parquet(args.processed / "attributes.parquet", index=False)
    rep = {
        "inherit_relations": sorted(INHERIT_RELATIONS),
        "eligible_edges": int(len(e)),
        "rounds": rounds,
        "license_overrides_applied": n_ov,
        "models_with_override_license": int(a["license"].astype(str).str.startswith("override:").sum()),
        "missing_before": before,
        "missing_after": {k: int(a[k].isna().sum()) for k in FIELDS},
        "filled": {k: int(a[f"inherited_{k}"].sum()) for k in FIELDS},
    }
    (args.processed / "enrich_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
