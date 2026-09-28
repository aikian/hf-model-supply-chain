"""Run one command and append its output to a log line by line (the Colab Drive mount syncs only on close, so the file is opened and closed per line).

Usage: python run_logged.py <log name> <command ...>
      e.g. python run_logged.py null_tol90_colab python 03_simulation/null_sensitivity.py ../02_data/processed/2026-09-25 --setting tol90
Log: 04_results/logs/2026-09-25_<log name>.log
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
