"""정제 규칙 검증: 판정자 간 일치도와 층(stratum)별 규칙 정확도.

입력 (02_data/validation/)
    paper_validation_to_label.csv    판정자 A (2026-09-26: AI(Codex) — 사람 판정 아님)
    paper_validation_llm_labels.csv  판정자 B (Claude Opus 5.5)
    paper_validation_key.csv         모델별 층 (kept_T1 / f_bot / f_course / f_boilerplate / f_test)
라벨: 0 실제 공급원, 1 산출물, 9 판단 불가 (정확도 계산에서 제외)
층별 정확도: 걸린 층은 라벨 1 비율(정밀도), kept_T1 은 라벨 0 비율.

사용: python validation_agreement.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

V = Path(__file__).resolve().parents[2] / "02_data" / "validation"


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    a = pd.read_csv(V / "paper_validation_to_label.csv", encoding="utf-8-sig")[["model_id", "label"]]
    b = pd.read_csv(V / "paper_validation_llm_labels.csv", encoding="utf-8-sig")[["model_id", "label"]]
    key = pd.read_csv(V / "paper_validation_key.csv", encoding="utf-8-sig")
    d = key.merge(a.rename(columns={"label": "A"}), on="model_id").merge(b.rename(columns={"label": "B"}), on="model_id")
    assert len(d) == 300, len(d)
    out = {"n": len(d)}
    both = d[(d.A != 9) & (d.B != 9)]
    out["agreement"] = {"n_both_decidable": len(both), "percent_agree": round(100 * (both.A == both.B).mean(), 1),
                        "kappa_binary": round(kappa(both.A, both.B), 3),
                        "kappa_3cat_all": round(kappa(d.A, d.B), 3),
                        "confusion_A_rows_B_cols": pd.crosstab(d.A, d.B).to_dict()}
    rows = []
    for s, g in d.groupby("stratum"):
        want = 0 if s == "kept_T1" else 1
        r = {"stratum": s, "n": len(g)}
        for who in ["A", "B"]:
            x = g[g[who] != 9]
            k = int((x[who] == want).sum())
            lo, hi = wilson(k, len(x))
            r[f"{who}_correct"] = f"{k}/{len(x)}"
            r[f"{who}_pct"] = round(100 * k / len(x), 1)
            r[f"{who}_ci95"] = f"{100 * lo:.0f}-{100 * hi:.0f}"
        cons = g[(g.A == g.B) & (g.A != 9)]
        r["consensus_correct"] = f"{int((cons.A == want).sum())}/{len(cons)}"
        rows.append(r)
    t = pd.DataFrame(rows)
    out["by_stratum"] = t.to_dict("records")
    dis = d[(d.A != d.B)]
    dis[["model_id", "stratum", "A", "B"]].to_csv(V / "disagreements_for_author.csv", index=False, encoding="utf-8-sig")
    out["n_disagreements"] = len(dis)
    (V / "validation_agreement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out["agreement"], indent=1, default=str))
    print(t.to_string(index=False))
    print(f"disagreements: {len(dis)} → disagreements_for_author.csv")


if __name__ == "__main__":
    main()
