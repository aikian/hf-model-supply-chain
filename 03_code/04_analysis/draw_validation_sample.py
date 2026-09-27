"""논문용 정제 검증 표본 추출 (원고 §Data Cleaning and Its Validation). ⚠ Stage 2에서, 규칙 동결 후 실행한다.

- 제외 규칙별 50개 (f_bot, f_boilerplate, f_course, f_test) + 유지(T1) 100개
- 규칙 개발 때 본 모델(dev_rule_development/*.csv)은 후보에서 뺀다
- 한 모델이 여러 층에 걸리면 먼저 뽑힌 층에만 둔다
- 판정자에게 층과 flag를 숨긴 판정용 파일, 그리고 층 정보가 담긴 키 파일을 따로 만든다

출력 (02_data/validation/)
    paper_validation_to_label.csv   model_id, url, label, type, note  (판정자용, 순서 섞음)
    paper_validation_key.csv        model_id, stratum                  (판정 끝난 뒤에만 연다)
    paper_validation_meta.json      시드, 규칙 파일 해시, 층별 개수

사용
    python draw_validation_sample.py ../../02_data/processed/2026-09-25 --seed 20270205
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "02_data" / "validation"
STRATA = {"f_bot": 50, "f_boilerplate": 50, "f_course": 50, "f_test": 50, "kept_T1": 100}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--seed", type=int, required=True, help="사전에 정해 원고에 적을 시드")
    args = ap.parse_args()

    out_label = VAL / "paper_validation_to_label.csv"
    if out_label.exists():
        raise SystemExit(f"이미 있음: {out_label} (재추출하면 검증이 무효가 된다)")

    f = pd.read_parquet(args.processed / "model_flags.parquet")
    seen = set()
    for csv in (VAL / "dev_rule_development").glob("*.csv"):
        d = pd.read_csv(csv)
        if "model_id" in d:
            seen |= set(d["model_id"])
    f = f[~f["model_id"].isin(seen)]

    taken, parts = set(), []
    for i, (stratum, size) in enumerate(STRATA.items()):
        pool = f[f["in_T1"]] if stratum == "kept_T1" else f[f[stratum]]
        pool = pool[~pool["model_id"].isin(taken)]
        s = pool.sample(n=size, random_state=args.seed + i)
        taken |= set(s["model_id"])
        parts.append(pd.DataFrame({"model_id": s["model_id"], "stratum": stratum}))
    key = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    lab = key[["model_id"]].assign(url="https://huggingface.co/" + key["model_id"],
                                   label="", type="", note="")
    lab.to_csv(out_label, index=False, encoding="utf-8-sig")
    key.to_csv(VAL / "paper_validation_key.csv", index=False, encoding="utf-8-sig")
    meta = {"seed": args.seed, "excluded_dev_models": len(seen),
            "strata": key["stratum"].value_counts().to_dict(),
            "rules_sha256": sha256(ROOT / "03_code" / "02_graph" / "clean_models.py"),
            "flags_sha256": sha256(args.processed / "model_flags.parquet")}
    (VAL / "paper_validation_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
