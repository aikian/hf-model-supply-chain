"""Run all pre-specified experiments in order (no parallelism by default: the laptop has 7.7 GB of memory).

Per variant: removal_sim.py -> substitutability.py -> hypothesis_tests.py
Results: 04_results/tables/<snap>_full_<variant>/  (removal_results.csv, substitutability.csv,
      h2_tests.csv, h3_tests.csv, hypothesis_verdicts.json)
Log: 04_results/logs/run_all_<snap>.log

Usage
    python run_all_experiments.py ../02_data/processed/2026-09-25
    python run_all_experiments.py ../02_data/processed/2026-09-25 --only main     # main analysis only
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# (name, extra args) — same order as the robustness list (a)-(h) in the manuscript's Analysis Plan
VARIANTS = [
    ("main", []),
    ("main_notest", ["--drop-rule", "f_test"]),                   # pre-registered rule: test rule precision 64% < 80% (validated 2026-09-26)
    ("tierT0", ["--tier", "T0"]),                                   # (a)
    ("tierT2", ["--tier", "T2"]),                                   # (a)
    ("declared", ["--declared-only", "--no-inherit"]),              # (b)
    ("keeptv", ["--keep-temporal-violations"]),                     # (c)
    ("noquant", ["--exclude-quantized"]),                           # (d)
    ("coarse", ["--option", "coarse"]),                             # (e)
    ("strictlang", ["--option", "strict"]),                         # (e') missing language as a distinct value (v1, lower bound on substitutability)
    ("other2unk", ["--other-as-unknown"]),                          # (f)
    ("strictcom", ["--strict-commercial"]),                         # (g)
    ("noov", ["--no-license-overrides"]),                           # (h)
]


def run_variant(name, extra, processed, seeds, logdir, host):
    """One variant: removal_sim -> substitutability -> hypothesis_tests. One log file per variant and host
    (the laptop and Colab write to the same Drive folder at the same time, so they must not share a file)."""
    import os
    snap = processed.name
    out = ROOT / "04_results" / "tables" / f"{snap}_full_{name}"
    if (out / "hypothesis_verdicts.json").exists():      # skip finished variants (resume after an interruption)
        return f"skip {name} (done)"
    logpath = logdir / f"{snap}_{name}_{host}.log"

    def write(s):
        # open and close per line: the Colab Drive mount syncs only on close, so writing through an open handle
        # shows nothing in the laptop's progress view until the run ends
        with open(logpath, "a", encoding="utf-8") as f:
            f.write(s)

    def run(cmd):
        t0 = dt.datetime.now()
        write(f"\n[{t0:%H:%M:%S}] $ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            write(line)
            if "done" in line:                            # also echo step-completion lines to the cell output
                print(f"[{dt.datetime.now():%H:%M:%S}] {name}: {line.strip()}", flush=True)
        rc = p.wait()
        write(f"[{dt.datetime.now():%H:%M:%S}] exit {rc} ({(dt.datetime.now() - t0).seconds}s)\n")
        if rc:
            raise RuntimeError(f"{name} failed: {' '.join(map(str, cmd))} (see {logpath.name})")

    t0 = dt.datetime.now()
    run([sys.executable, "03_simulation/removal_sim.py", processed, "--seeds", str(seeds), "--out", out, *extra])
    run([sys.executable, "03_simulation/substitutability.py", processed, "--out", out, *extra])
    run([sys.executable, "04_analysis/hypothesis_tests.py", out])
    return f"{name} done ({(dt.datetime.now() - t0).seconds // 60} min)"


def main():
    import socket
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=1000, help="number of runs for the main analysis")
    ap.add_argument("--seeds-variants", type=int, default=500,
                    help="number of runs for robustness variants (400 is the minimum for Holm significance in a family of 20)")
    ap.add_argument("--workers", type=int, default=1,
                    help="variants to run at once. One variant needs about 3 GB of memory -> laptop (7.7 GB) 1, Colab (12 GB) 2")
    ap.add_argument("--host", default=socket.gethostname().split(".")[0][:20], help="host label used in log file names")
    args = ap.parse_args()
    processed = args.processed.resolve()
    logdir = ROOT / "04_results" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    todo = [(n, e) for n, e in VARIANTS if not args.only or n in args.only]
    print(f"[{dt.datetime.now():%H:%M:%S}] host={args.host} workers={args.workers} variants={[n for n, _ in todo]}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_variant, n, e, processed, args.seeds if n.startswith("main") else args.seeds_variants,
                          logdir, args.host): n for n, e in todo}
        for f in as_completed(futs):
            try:
                print(f"[{dt.datetime.now():%H:%M:%S}] {f.result()}", flush=True)
            except Exception as err:                      # keep going when one variant fails
                print(f"[{dt.datetime.now():%H:%M:%S}] FAILED {err}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
