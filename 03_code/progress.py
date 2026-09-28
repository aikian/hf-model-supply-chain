"""Experiment progress view: progress of every variant on the laptop and on Colab in one screen (reads the logs synced through Drive).

    python progress.py            # print once
    python progress.py --watch    # refresh every 10 s (Ctrl+C to quit)

Steps: 2 shock types x (greedy strategy + 5 k) = 12, plus substitutability and verdicts = 14 steps.
Remaining time is estimated from the mean time per step so far (larger k take longer, so the real time may be somewhat longer).
"""
import argparse
import datetime as dt
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "04_results" / "logs"
TABLES = ROOT / "04_results" / "tables"
SNAP = "2026-09-25"
PLAN = [("main", "laptop"), ("main_notest", "laptop"), ("tierT2", "laptop"), ("declared", "laptop"), ("tierT0", "laptop"),
        ("keeptv", "colab"), ("noquant", "colab"), ("coarse", "colab"), ("strictlang", "colab"),
        ("other2unk", "colab"), ("strictcom", "colab"), ("noov", "colab"),
        # 2026-09-27 review response: initial planned analysis, matched-null criterion sensitivity
        ("planned", "laptop"), ("null_tol90", "colab"), ("null_tol99", "colab"), ("null_nearest", "colab"), ("planned_T0", "colab")]
STEPS = 14
STEP_RE = re.compile(r"\[(legal|availability)\] (greedy done|k=\d+ done)")


def read_log(name, host):
    """Return (log text, start time or None). For main, read only the last main run from the old shared log."""
    if name == "main" and not (LOGS / f"{SNAP}_main_{host}.log").exists():
        p = LOGS / f"run_all_{SNAP}.log"
        if not p.exists():
            return "", None
        txt = p.read_text(encoding="utf-8", errors="replace")
        i = txt.rfind("2026-09-25_full_main\n")
        i = txt.rfind("\n[", 0, i) if i >= 0 else -1
        seg = txt[i:] if i >= 0 else ""
        m = re.search(r"\[(\d\d:\d\d:\d\d)\] \$", seg)
        return seg, m.group(1) if m else None
    p = LOGS / f"{SNAP}_{name}_{host}.log"
    if not p.exists():
        return "", None
    txt = p.read_text(encoding="utf-8", errors="replace")
    # the log is appended across runs, so look only from the last removal_sim start (ignores errors of interrupted earlier runs)
    starts = [m.start() for m in re.finditer(r"\[\d\d:\d\d:\d\d\] \$ \S+ 03_simulation/removal_sim\.py", txt)]
    if starts:
        txt = txt[starts[-1]:]
    m = re.search(r"\[(\d\d:\d\d:\d\d)\] \$", txt)
    return txt, m.group(1) if m else None


def status(name, host, now):
    done = (TABLES / f"{SNAP}_full_{name}" / "hypothesis_verdicts.json").exists()
    txt, start = read_log(name, host)
    steps = len(STEP_RE.findall(txt))
    if "substitutability.py" in txt:
        steps = max(steps, 12) + 1
    if "hypothesis_tests.py" in txt:
        steps += 1
    failed = bool(re.search(r"exit [1-9]\d*|Traceback|MemoryError", txt))
    if done:
        steps = STEPS
    last = STEP_RE.findall(txt)
    last = f"{last[-1][0]} {last[-1][1]}" if last else ("loading data" if start else "")
    elapsed = eta = ""
    if start:
        t0 = dt.datetime.combine(now.date(), dt.time.fromisoformat(start))
        # Colab logs are stamped in UTC; shift to Korean time (+9)
        if host == "colab":
            t0 += dt.timedelta(hours=9)
        if t0 > now + dt.timedelta(minutes=5):
            t0 -= dt.timedelta(days=1)
        el = (now - t0).total_seconds() / 60
        elapsed = f"{el:5.0f}m"
        if 0 < steps < STEPS and not done:
            eta = f"~{el / steps * (STEPS - steps):4.0f}m"
    # a manual stop or dropped session (KeyboardInterrupt) is not a failure: the next run resumes from the saved step
    interrupted = failed and "KeyboardInterrupt" in txt and not re.search(r"MemoryError|Error:", txt)
    state = "DONE" if done else "INT " if interrupted else "FAIL" if failed else "run " if start else "wait"
    return state, steps, elapsed, eta, last


def render():
    now = dt.datetime.now()
    lines = [f"HF supply chain experiment progress   {now:%Y-%m-%d %H:%M:%S}   (refreshes every 10 s, Ctrl+C to quit)", ""]
    lines.append(f"{'variant':12s} {'host':7s} {'state':5s} {'progress':24s} {'time':>6s} {'ETA':>7s}  last step")
    lines.append("-" * 92)
    tot = 0
    for name, host in PLAN:
        st, steps, el, eta, last = status(name, host, now)
        tot += steps
        bar = "█" * round(16 * steps / STEPS) + "·" * (16 - round(16 * steps / STEPS))
        lines.append(f"{name:12s} {host:7s} {st:5s} {bar} {steps:2d}/{STEPS} {el:>6s} {eta:>7s}  {last}")
    lines.append("-" * 92)
    pct = 100 * tot / (STEPS * len(PLAN))
    lines.append(f"overall {pct:5.1f}%   done {sum(status(n, h, now)[0] == 'DONE' for n, h in PLAN)}/{len(PLAN)}")
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        lines.append(f"laptop memory: free {m.avail / 2**30:.1f} GB / {m.total / 2**30:.1f} GB (used {m.load}%)")
    except Exception:
        pass
    lines.append("Colab rows may lag a few minutes because of Drive sync.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    if not args.watch:
        print(render())
        return
    while True:
        out = render()
        os.system("cls" if os.name == "nt" else "clear")
        print(out, flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
