"""재현 저장소로 내보내기: 공개해도 되는 파일만 로컬 git 저장소로 복사한다.

Google Drive 폴더 안에 .git 을 두면 동기화 충돌로 저장소가 깨질 수 있어서 저장소는 로컬에 둔다.
    python export_repo.py            # 기본 대상: C:\\Users\\donggyu\\repos\\hf-model-supply-chain

포함: 코드, 규칙표, 결과 표·그림, 데이터 사전, 기준서, MANIFEST
제외: 00_admin(개인 계획), 01_literature(논문 PDF 저작권), 원본·가공 데이터(100MB 초과 → Zenodo),
      규칙 개발용 AI 판정 표본, 원고(투고 전), 캐시
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "03_code",
    "04_results/tables",
    "04_results/figures",
    "02_data/data_dictionary.md",
    "02_data/MANIFEST_2026-09-25.json",
    "02_data/validation/codebook.md",
    # 판정 절차 (원고: "The codebook and both labeling protocols are in the replication package")
    "02_data/validation/ai_review_20260926/README.md",
    "02_data/validation/paper_validation_llm_meta.json",
    "02_data/validation/paper_validation_rater_a_meta.json",
    "02_data/validation/validation_agreement.json",
    "02_data/processed/2026-09-25/summary.json",
    "02_data/processed/2026-09-25/inference_report.json",
    "02_data/processed/2026-09-25/enrich_report.json",
]
EXCLUDE_PARTS = {"__pycache__", "_pilot_smoke_not_results", "_pages", "dev_rule_development"}
EXCLUDE_SUFFIX = {".pyc", ".parquet", ".gz", ".log", ".pdfrender.png"}
MAX_MB = 50
# 개인 계정 가명화: 정제 보고서는 개인 계정의 모델을 '자동 업로드'·'테스트' 등의 예시로 든다.
# 공개본에서는 예시 ID 의 계정을 해시 가명으로 바꾼다 (조직·회사 계정은 그대로).
PSEUDONYMIZE = ["04_results/tables/cleaning_report_*.md"]
ORG_ACCOUNTS = {"black-forest-labs", "Qwen", "google", "distilbert", "stabilityai", "lerobot", "google-bert",
                "meta-llama", "mistralai", "microsoft", "openai", "deepseek-ai", "nvidia", "unsloth",
                "runwayml", "stable-diffusion-v1-5", "facebook",
                "gradients-io-tournaments"}                 # 대회 운영 조직의 봇 계정 (정제 규칙 BOT_AUTHORS 에 이미 공개)


def pseudonymize(text: str) -> str:
    import hashlib
    import re

    def sub(m):
        author, model = m.group(1), m.group(2)
        if author in ORG_ACCOUNTS:
            return m.group(0)
        alias = "user-" + hashlib.sha1(author.encode()).hexdigest()[:6]
        return f"`{alias}/{model.replace(author, alias)}`"
    return re.sub(r"`([A-Za-z0-9][\w.\-]*)/([\w.\-]+)`", sub, text)


def keep(f: Path) -> bool:
    rel = f.relative_to(ROOT)
    if EXCLUDE_PARTS & set(rel.parts):
        return False
    if any(p.startswith("_partial_s") for p in rel.parts):     # 이어서 계산용 중간 저장 (결과 CSV 와 중복)
        return False
    if f.suffix in EXCLUDE_SUFFIX or f.name.endswith("_pdfrender.png") or f.name == ".gitkeep":
        return False
    return f.stat().st_size < MAX_MB * 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path.home() / "repos" / "hf-model-supply-chain")
    args = ap.parse_args()
    n = 0
    for item in INCLUDE:
        src = ROOT / item
        files = [src] if src.is_file() else [f for f in src.rglob("*") if f.is_file()]
        for f in files:
            if not keep(f):
                continue
            out = args.dest / f.relative_to(ROOT)
            out.parent.mkdir(parents=True, exist_ok=True)
            if any(f.match(str(ROOT / p)) or f.relative_to(ROOT).match(p) for p in PSEUDONYMIZE):
                out.write_text(pseudonymize(f.read_text(encoding="utf-8")), encoding="utf-8")
            else:
                shutil.copy2(f, out)
            n += 1
    print(f"exported {n} files -> {args.dest}")


if __name__ == "__main__":
    main()
