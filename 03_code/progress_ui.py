"""실험 모니터 (연구용 대시보드). 노트북과 Colab 작업을 구역으로 나눠 보여 준다.

    python progress_ui.py          # 5초마다 새로 고침, Ctrl+C 종료

데이터: 04_results/logs 의 로그 (Colab 로그는 구글 드라이브 동기화로 들어온다) + 완료 표시 파일.
단계: 법적 충격(탐욕 + k 5개) │ 가용성 충격(탐욕 + k 5개) │ RQ1 대체 가능성 │ 가설 판정 = 14 단계.
"""
import ctypes
import datetime as dt
import re
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from progress import PLAN, SNAP, STEPS, STEP_RE, TABLES, read_log, status

DESC = {
    "main": "Main analysis (T1, wildcard, 1,000 runs)",
    "tierT0": "No cleaning (T0)",
    "tierT2": "Cleaned + downloaded (T2)",
    "declared": "Declared edges only, no inheritance",
    "keeptv": "Keep temporally inverted edges",
    "noquant": "Exclude quantized models",
    "coarse": "Options = task × license",
    "strictlang": "Unknown language ≠ wildcard (v1)",
    "planned": "Initial plan, corrected code (H2a/H2b, δ)",
    "null_tol90": "Matched null stops at 90%",
    "null_tol99": "Matched null stops at 99%",
    "null_nearest": "Matched null nearest to target",
    "planned_T0": "Initial plan on T0 (all models)",
    "other2unk": "'other' license → unknown",
    "strictcom": "Commercial = permissive only",
    "noov": "No model-card license reclassification",
}
PIPE = [("legal", s) for s in ["greedy", "k=1", "k=5", "k=10", "k=50", "k=100"]] + \
       [("availability", s) for s in ["greedy", "k=1", "k=5", "k=10", "k=50", "k=100"]] + \
       [("rq1", "subst"), ("test", "verdict")]
HOSTS = [("laptop", "💻  Laptop", "1 worker · 7.7 GB RAM · sequential", "cyan"),
         ("colab", "☁   Google Colab", "2 workers · ~12 GB RAM · parallel", "magenta")]


def steps_done(name, host):
    txt, _ = read_log(name, host)
    done = {(sem, st.replace(" done", "")) for sem, st in STEP_RE.findall(txt)}
    if "substitutability.py" in txt and re.search(r"hypothesis_tests\.py", txt):
        done.add(("rq1", "subst"))
    if (TABLES / f"{SNAP}_full_{name}" / "hypothesis_verdicts.json").exists():
        done |= set(PIPE)
    return done


def pipeline_text(name, host, state):
    done = steps_done(name, host)
    t = Text()
    nxt = next((p for p in PIPE if p not in done), None)
    for i, p in enumerate(PIPE):
        if i in (6, 12):
            t.append(" │ ", style="grey42")
        if p in done:
            t.append("■", style="bold green")
        elif p == nxt and state == "run ":
            t.append("■", style="bold yellow blink")
        else:
            t.append("□", style="grey35")
    return t


def badge(state):
    return {"DONE": Text(" DONE ", style="bold black on green"),
            "run ": Text(" RUN  ", style="bold black on yellow"),
            "wait": Text(" WAIT ", style="grey62 on grey19"),
            "FAIL": Text(" FAIL ", style="bold white on red"),
            "INT ": Text(" STOP ", style="bold black on grey70")}[state]


def host_panel(host, title, sub, color, now):
    tab = Table(box=box.SIMPLE_HEAD, expand=True, header_style=f"bold {color}", pad_edge=False)
    tab.add_column("Variant", style="bold", no_wrap=True, min_width=10)
    tab.add_column("Tests", style="grey70", no_wrap=True, min_width=20, max_width=36)
    tab.add_column("Status", no_wrap=True, width=6)
    tab.add_column("Pipeline", no_wrap=True, min_width=23)
    tab.add_column("Step", justify="right", no_wrap=True, width=5)
    tab.add_column("Elapsed", justify="right", no_wrap=True, width=7)
    tab.add_column("ETA", justify="right", no_wrap=True, width=6)
    tab.add_column("Last event", style="grey62", no_wrap=True, min_width=18, max_width=24)
    n_done = n_all = 0
    for name, h in PLAN:
        if h != host:
            continue
        st, steps, el, eta, last = status(name, host, now)
        n_done += steps
        n_all += STEPS
        tab.add_row(name, DESC.get(name, ""), badge(st), pipeline_text(name, host, st),
                    f"{steps}/{STEPS}", el.strip() or "–", eta.strip() or ("✓" if st == "DONE" else "–"),
                    last or "–")
    bar = ProgressBar(total=max(n_all, 1), completed=n_done, width=60, complete_style=color, finished_style="green")
    head = Table.grid(expand=True)
    head.add_column(); head.add_column(justify="right")
    head.add_row(Text.assemble((sub, "grey62"), ("     pipeline: legal G·1·5·10·50·100 │ availability G·1·5·10·50·100 │ RQ1·H", "grey42")),
                 Text(f"{100 * n_done / n_all:5.1f}%", style=f"bold {color}", justify="right"))
    head.add_row(bar, "")
    return Panel(Group(head, tab), title=f"[bold {color}]{title}[/]", border_style=color, padding=(0, 1))


def memory():
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
    m = MS(); m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.load, m.avail / 2**30, m.total / 2**30


def render():
    now = dt.datetime.now()
    states = [status(n, h, now) for n, h in PLAN]
    tot = sum(s[1] for s in states)
    done = sum(s[0] == "DONE" for s in states)
    running = sum(s[0] == "run " for s in states)
    failed = sum(s[0] == "FAIL" for s in states)

    title = Text.assemble(("HF MODEL SUPPLY CHAIN", "bold white"), ("  ·  Functional Option Loss  ·  ", "grey62"),
                          ("experiment monitor", "bold #7aa2f7"))
    meta = Text.assemble(("snapshot ", "grey50"), (SNAP, "white"), ("   design ", "grey50"), ("v2", "white"),
                         ("   variants ", "grey50"), (f"{len(PLAN)}", "white"),
                         ("   done ", "grey50"), (f"{done}", "bold green"), ("   running ", "grey50"),
                         (f"{running}", "bold yellow"), ("   failed ", "grey50"),
                         (f"{failed}", "bold red" if failed else "white"),
                         ("   ", ""), (f"{now:%Y-%m-%d %H:%M:%S}", "grey70"))
    overall = ProgressBar(total=STEPS * len(PLAN), completed=tot, width=100, complete_style="#7aa2f7",
                          finished_style="green")
    header = Panel(Group(title, meta, overall, Text(f"overall {100 * tot / (STEPS * len(PLAN)):.1f}%  "
                                                    f"({tot}/{STEPS * len(PLAN)} steps)", style="#7aa2f7")),
                   box=box.HEAVY, border_style="#7aa2f7", padding=(0, 1))

    load, avail, total = memory()
    mcol = "green" if load < 75 else "yellow" if load < 90 else "red"
    foot = Table.grid(expand=True)
    foot.add_column(); foot.add_column(justify="right")
    foot.add_row(
        Group(Text.assemble(("Laptop memory ", "grey62"), (f"{load}% used", f"bold {mcol}"),
                            (f"   {avail:.1f} / {total:.1f} GB free", "grey62")),
              ProgressBar(total=100, completed=load, width=40, complete_style=mcol)),
        Text("■ done  ■ running  □ pending     Colab rows lag a few minutes (Drive sync)\n"
             "Ctrl+C to close · refreshes every 5 s", style="grey50", justify="right"))
    return Group(header, host_panel(*HOSTS[0], now), host_panel(*HOSTS[1], now), Panel(foot, border_style="grey35"))


def main():
    console = Console()
    with Live(render(), console=console, refresh_per_second=1, screen=True) as live:
        while True:
            time.sleep(5)
            live.update(render())


if __name__ == "__main__":
    main()
