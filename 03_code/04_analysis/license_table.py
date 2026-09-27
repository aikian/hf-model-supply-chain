"""라이선스 분류 요약표 (LaTeX) → 04_results/tables/license_classes_<snap>.tex

분석에 쓰는 값으로 센다: T1 모델, 속성 상속과 'other' 재분류(license_overrides.csv)를 반영한
attributes.parquet 의 license.

사용
    python license_table.py ../../02_data/processed/2026-09-25
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code" / "03_simulation"))
from removal_sim import LICENSE_MAP, license_class  # noqa: E402

ORDER = ["permissive", "copyleft", "responsible_ai", "vendor_custom",
         "noncommercial", "no_derivatives", "other", "unknown"]
NAMES = {"permissive": "Permissive", "copyleft": "Copyleft", "responsible_ai": "Responsible-AI (RAIL)",
         "vendor_custom": "Vendor-custom", "noncommercial": "Non-commercial",
         "no_derivatives": "No-derivatives", "other": "Other / unspecified", "unknown": "Unknown / none"}
COMMERCIAL = {"permissive": "yes", "copyleft": "yes", "responsible_ai": "yes (use restr.)",
              "vendor_custom": "conditional", "noncommercial": "no", "no_derivatives": "no derivatives",
              "other": "unknown", "unknown": "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()
    snap = args.processed.name

    attr = pd.read_parquet(args.processed / "attributes.parquet", columns=["model_id", "license"])
    t1 = pd.read_parquet(args.processed / "model_flags.parquet", columns=["model_id", "in_T1"])
    lic = attr.merge(t1, on="model_id")
    lic = lic.loc[lic["in_T1"], "license"]
    cls = lic.map(license_class)
    n_by_class = cls.value_counts()
    cnt = lic.value_counts()
    ov = lic[lic.astype(str).str.startswith("override:")].map(license_class).value_counts()
    ids_by_class = LICENSE_MAP.reset_index().groupby("class")["license"].apply(list)

    rows = []
    for c in ORDER:
        ids = sorted(ids_by_class.get(c, []), key=lambda x: -cnt.get(x, 0))
        tot = int(n_by_class.get(c, 0))
        # 단 폭에 맞추려고 라이선스 ID 예시 열은 뺀다 (전체 매핑은 보충 자료)
        rows.append(f"{NAMES[c]} & {len(ids)} & {tot:,} ({100 * tot / len(lic):.1f}\\%) & {COMMERCIAL[c]} \\\\")

    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{License classes of T1-original models, after inheritance and after reclassifying large "
        r"\texttt{other}-licensed families from their model cards. IDs is the number of Hub license "
        r"identifiers mapped to the class. The full mapping of all "
        + str(len(LICENSE_MAP)) + r" identifiers is in the supplementary material.}",
        r"\label{tab:licenses}", r"\begin{tabular}{lrrl}", r"\toprule",
        r"Class & IDs & T1-original models & Commercial use \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out = ROOT / "04_results" / "tables" / f"license_classes_{snap}.tex"
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print("\nT1 models reclassified from 'other' (incl. inherited):", ov.to_dict(), "total", int(ov.sum()))
    print("class shares %:", (100 * n_by_class / len(lic)).round(2).to_dict())


if __name__ == "__main__":
    main()
