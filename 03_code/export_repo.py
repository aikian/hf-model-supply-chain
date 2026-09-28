"""Export to the replication repository: copy only the files that may be published into a local git repository.

A .git inside a Google Drive folder can be corrupted by sync conflicts, so the repository lives on the local disk.
    python export_repo.py            # default target: C:\\Users\\donggyu\\repos\\hf-model-supply-chain

Included: code, rule tables, result tables and figures, data dictionary, codebook, MANIFEST
Excluded: 00_admin (personal planning), 01_literature (copyrighted paper PDFs), raw and processed data (over 100 MB -> Zenodo),
      AI-labeled samples used for rule development, manuscript (before submission), caches
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
    # labeling protocols (manuscript: "The codebook and both labeling protocols are in the replication package")
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
# Pseudonymize personal accounts: the cleaning report cites models of personal accounts as examples of 'automated upload', 'test', etc.
# In the public copy the account part of each example ID is replaced by a hash alias (organization and company accounts unchanged).
PSEUDONYMIZE = ["04_results/tables/cleaning_report_*.md"]
ORG_ACCOUNTS = {"black-forest-labs", "Qwen", "google", "distilbert", "stabilityai", "lerobot", "google-bert",
                "meta-llama", "mistralai", "microsoft", "openai", "deepseek-ai", "nvidia", "unsloth",
                "runwayml", "stable-diffusion-v1-5", "facebook",
                "gradients-io-tournaments"}                 # bot account of a competition organizer (already public in the BOT_AUTHORS cleaning rule)


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
    if any(p.startswith("_partial_s") for p in rel.parts):     # intermediate checkpoints for resuming (redundant with the result CSVs)
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
