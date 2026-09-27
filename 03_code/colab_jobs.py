"""Colab 분담분 (2026-09-27 심사 대응): 대조군 기준 민감도 tol90, tol99 를 동시에 돌린다.

Colab 칸에서:  !python colab_jobs.py
끊겨도 다시 실행하면 저장된 단계부터 이어서 계산한다. 진행은 칸 출력과 드라이브 로그에 한 줄씩 남는다.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = "../02_data/processed/2026-09-25"
JOBS = [("null_tol90_colab", ["03_simulation/null_sensitivity.py", P, "--setting", "tol90"]),
        ("null_tol99_colab", ["03_simulation/null_sensitivity.py", P, "--setting", "tol99"])]


def main():
    procs = []
    for log, cmd in JOBS:
        print(f"start {log}", flush=True)
        procs.append(subprocess.Popen([sys.executable, "run_logged.py", log, sys.executable, *cmd], cwd=HERE))
    codes = [p.wait() for p in procs]
    print("all done" if not any(codes) else f"exit codes {codes}", flush=True)


if __name__ == "__main__":
    main()
