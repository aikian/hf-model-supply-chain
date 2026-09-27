"""실험 진행 창: 노트북·Colab 의 모든 변형 진행률을 한 화면에 (드라이브로 동기화된 로그를 읽는다).

    python progress.py            # 한 번 출력
    python progress.py --watch    # 10초마다 새로 고침 (Ctrl+C 로 종료)

단계: 충격 2종 × (탐욕 전략 + k 5개) = 12 + 대체 가능성 + 판정 = 14 단계.
남은 시간은 지금까지 단계당 평균으로 추정한다 (큰 k 가 더 오래 걸려서 실제로는 조금 더 길 수 있다).
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
        # 2026-09-27 심사 대응: 최초 계획 분석, 대조군 기준 민감도
        ("planned", "laptop"), ("null_tol90", "colab"), ("null_tol99", "colab"), ("null_nearest", "colab"), ("planned_T0", "colab")]
STEPS = 14
STEP_RE = re.compile(r"\[(legal|availability)\] (greedy done|k=\d+ done)")


def read_log(name, host):
    """(로그 텍스트, 시작 시각 또는 None). main 은 예전 방식의 공용 로그에서 마지막 main 실행 부분만 읽는다."""
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
    # 같은 로그에 이어 쓰므로, 마지막으로 removal_sim 을 시작한 부분부터만 본다 (중단된 이전 실행의 오류 무시)
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
        # Colab 로그는 UTC 로 찍힌다 → 한국 시간(+9)으로 맞춘다
        if host == "colab":
            t0 += dt.timedelta(hours=9)
        if t0 > now + dt.timedelta(minutes=5):
            t0 -= dt.timedelta(days=1)
        el = (now - t0).total_seconds() / 60
        elapsed = f"{el:5.0f}m"
        if 0 < steps < STEPS and not done:
            eta = f"~{el / steps * (STEPS - steps):4.0f}m"
    # 수동 중단·세션 끊김(KeyboardInterrupt)은 실패가 아니다: 다음 실행 때 저장된 단계부터 이어서 계산한다
    interrupted = failed and "KeyboardInterrupt" in txt and not re.search(r"MemoryError|Error:", txt)
    state = "DONE" if done else "INT " if interrupted else "FAIL" if failed else "run " if start else "wait"
    return state, steps, elapsed, eta, last


def render():
    now = dt.datetime.now()
    lines = [f"HF 공급망 실험 진행도   {now:%Y-%m-%d %H:%M:%S}   (10초마다 새로 고침, Ctrl+C 종료)", ""]
    lines.append(f"{'변형':12s} {'장소':7s} {'상태':5s} {'진행':24s} {'경과':>6s} {'남은':>7s}  마지막 단계")
    lines.append("-" * 92)
    tot = 0
    for name, host in PLAN:
        st, steps, el, eta, last = status(name, host, now)
        tot += steps
        bar = "█" * round(16 * steps / STEPS) + "·" * (16 - round(16 * steps / STEPS))
        lines.append(f"{name:12s} {host:7s} {st:5s} {bar} {steps:2d}/{STEPS} {el:>6s} {eta:>7s}  {last}")
    lines.append("-" * 92)
    pct = 100 * tot / (STEPS * len(PLAN))
    lines.append(f"전체 {pct:5.1f}%   완료 {sum(status(n, h, now)[0] == 'DONE' for n, h in PLAN)}/{len(PLAN)}")
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        lines.append(f"노트북 메모리: 여유 {m.avail / 2**30:.1f} GB / {m.total / 2**30:.1f} GB (사용률 {m.load}%)")
    except Exception:
        pass
    lines.append("Colab 줄은 드라이브 동기화 때문에 몇 분 늦게 반영될 수 있어요.")
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
