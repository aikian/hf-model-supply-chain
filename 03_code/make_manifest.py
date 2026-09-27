"""재현 패키지용 SHA-256 목록: 원본 스냅샷, 가공 데이터, 코드 전체.

출력: 02_data/MANIFEST_2026-09-25.json  (파일별 bytes, sha256)
사용: python make_manifest.py
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = "2026-09-25"
PATTERNS = [f"02_data/raw/hf_models_{SNAP}.jsonl.gz", f"02_data/raw/license_names_{SNAP}.jsonl",
            f"02_data/raw/missing_parents_api_{SNAP}.jsonl", f"02_data/processed/{SNAP}/*",
            "03_code/**/*.py", "03_code/**/*.csv", "03_code/requirements.txt", "03_code/*.ipynb"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = {}
    for pat in PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file() and "__pycache__" not in p.parts:
                files[p.relative_to(ROOT).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    out = ROOT / "02_data" / f"MANIFEST_{SNAP}.json"
    out.write_text(json.dumps({"created": dt.datetime.now().isoformat(timespec="seconds"), "snapshot": SNAP,
                               "files": files}, indent=1), encoding="utf-8")
    print(f"{len(files)} files -> {out}")


if __name__ == "__main__":
    main()
