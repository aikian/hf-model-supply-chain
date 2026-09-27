"""사전에 정한 모든 실험을 순서대로 실행한다 (메모리 7.7GB 환경이라 병렬 실행하지 않음).

각 변형마다: removal_sim.py → substitutability.py → hypothesis_tests.py
결과: 04_results/tables/<snap>_full_<variant>/  (removal_results.csv, substitutability.csv,
      h2_tests.csv, h3_tests.csv, hypothesis_verdicts.json)
로그: 04_results/logs/run_all_<snap>.log

사용
    python run_all_experiments.py ../02_data/processed/2026-09-25
    python run_all_experiments.py ../02_data/processed/2026-09-25 --only main     # 주 분석만
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# (이름, 추가 인자) — 원고의 Analysis Plan 강건성 목록 (a)–(h)와 같은 순서
VARIANTS = [
    ("main", []),
    ("main_notest", ["--drop-rule", "f_test"]),                   # 사전 결정 규칙: test 규칙 정밀도 64% < 80% (2026-09-26 검증)
    ("tierT0", ["--tier", "T0"]),                                   # (a)
    ("tierT2", ["--tier", "T2"]),                                   # (a)
    ("declared", ["--declared-only", "--no-inherit"]),              # (b)
    ("keeptv", ["--keep-temporal-violations"]),                     # (c)
    ("noquant", ["--exclude-quantized"]),                           # (d)
    ("coarse", ["--option", "coarse"]),                             # (e)
    ("strictlang", ["--option", "strict"]),                         # (e') 언어 없음을 별개 값 (v1, 대체 가능성 하한)
    ("other2unk", ["--other-as-unknown"]),                          # (f)
    ("strictcom", ["--strict-commercial"]),                         # (g)
    ("noov", ["--no-license-overrides"]),                           # (h)
]


def run_variant(name, extra, processed, seeds, logdir, host):
    """변형 하나: removal_sim → substitutability → hypothesis_tests. 로그는 변형·장소별 파일
    (노트북과 Colab 이 같은 드라이브 폴더에 동시에 쓰므로 한 파일을 같이 쓰지 않는다)."""
    import os
    snap = processed.name
    out = ROOT / "04_results" / "tables" / f"{snap}_full_{name}"
    if (out / "hypothesis_verdicts.json").exists():      # 끝난 변형은 건너뛴다 (중단 후 재개)
        return f"skip {name} (done)"
    logpath = logdir / f"{snap}_{name}_{host}.log"

    def write(s):
        # 줄마다 열고 닫는다: Colab 의 드라이브 마운트는 파일을 닫아야 동기화하므로, 열어 둔 채 쓰면
        # 실행이 끝날 때까지 노트북 쪽 진행 창에 아무것도 보이지 않는다
        with open(logpath, "a", encoding="utf-8") as f:
            f.write(s)

    def run(cmd):
        t0 = dt.datetime.now()
        write(f"\n[{t0:%H:%M:%S}] $ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            write(line)
            if "done" in line:                            # 단계 완료 줄은 셀 출력에도 보여 준다
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
    ap.add_argument("--seeds", type=int, default=1000, help="주 분석 반복 수")
    ap.add_argument("--seeds-variants", type=int, default=500,
                    help="강건성 변형 반복 수 (묶음 20에서 Holm 유의가 가능한 최소는 400)")
    ap.add_argument("--workers", type=int, default=1,
                    help="동시에 돌릴 변형 수. 변형 하나가 메모리 약 3GB → 노트북(7.7GB) 1, Colab(12GB) 2")
    ap.add_argument("--host", default=socket.gethostname().split(".")[0][:20], help="로그 파일 이름용 장소 표시")
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
            except Exception as err:                      # 한 변형이 실패해도 나머지는 계속
                print(f"[{dt.datetime.now():%H:%M:%S}] FAILED {err}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
