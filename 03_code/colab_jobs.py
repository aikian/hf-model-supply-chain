"""Colab share of the 2026-09-27 review response: run the matched-null criterion sensitivities tol90 and tol99 at the same time.

In a Colab cell:  !python colab_jobs.py
If the session drops, rerun and it resumes from the saved step. Progress goes line by line to the cell output and the Drive log.
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
