"""명령 하나를 실행하고 출력을 로그에 한 줄씩 저장한다 (Colab 드라이브는 파일을 닫아야 동기화되므로 줄마다 열고 닫는다).

사용: python run_logged.py <로그 이름> <명령 ...>
      예) python run_logged.py null_tol90_colab python 03_simulation/null_sensitivity.py ../02_data/processed/2026-09-25 --setting tol90
로그: 04_results/logs/2026-09-25_<로그 이름>.log
"""
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE.parent / "04_results" / "logs" / f"2026-09-25_{sys.argv[1]}.log"


def write(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s)


def main():
    cmd = sys.argv[2:]
    write(f"\n[{dt.datetime.now():%H:%M:%S}] $ {' '.join(cmd)}\n")
    p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    for line in p.stdout:
        write(line)
        if "done" in line or "verified" in line or "Error" in line or "Traceback" in line:
            print(f"[{dt.datetime.now():%H:%M:%S}] {sys.argv[1]}: {line.strip()}", flush=True)
    rc = p.wait()
    write(f"[{dt.datetime.now():%H:%M:%S}] exit {rc}\n")
    print(f"{sys.argv[1]} exit {rc}", flush=True)


if __name__ == "__main__":
    main()
