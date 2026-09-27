"""스냅샷에 없는 '선언된 부모'가 실제로 삭제된 것인지 분류하고, 이름이 바뀐 경우 실제 ID 로 잇는다.

분류
    renamed   : HF API 가 200 을 돌려주고 id 가 다름 (저장소 이름 변경·조직 이전 → HF 가 옛 이름을 새 이름으로 연결)
    alias     : 조직 없는 옛 ID (예: roberta-base) 가 API 에서 조직 포함 ID 로 연결됨 (renamed 의 한 경우)
    local_path: 올린 사람의 로컬 경로 (/cache/models/Org--Name, models--Org--Name) → Org/Name 으로 해석, 스냅샷에 있으면 연결
    same      : API 가 같은 id 로 200 (스냅샷 이후 생성·공개 전환 등) → 스냅샷 시점에는 없던 것으로 둔다
    unavailable: 401/403/404 → 삭제 또는 비공개. HF 는 비공개와 없음을 구분해 주지 않는다

입력  edges_all.parquet, nodes.parquet
출력  02_data/processed/<snap>/missing_parents_resolved.csv
      02_data/raw/missing_parents_api_<snap>.jsonl (API 응답 기록)

사용
    python resolve_missing_parents.py ../../02_data/processed/2026-09-25 --min-children 5
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
LOCAL = [re.compile(r"(?:^|/)models--([^/]+?)--([^/]+?)(?:/|$)"), re.compile(r"(?:^|/)([A-Za-z0-9_.\-]+)--([A-Za-z0-9_.\-]+)/?$")]


def local_path_id(pid):
    if "/" not in pid.strip("/") or pid.startswith(("/", "~", ".")) or "--" in pid:
        for rx in LOCAL:
            m = rx.search(pid)
            if m:
                return f"{m.group(1)}/{m.group(2)}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    ap.add_argument("--min-children", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.65)
    args = ap.parse_args()
    P, snap = args.processed, args.processed.name
    e = pd.read_parquet(P / "edges_all.parquet", columns=["parent_id", "child_id", "parent_in_snapshot"])
    ids = set(pd.read_parquet(P / "nodes.parquet", columns=["model_id"])["model_id"])
    low = {m.lower(): m for m in ids}
    miss = e[~e["parent_in_snapshot"]].groupby("parent_id").size().sort_values(ascending=False)

    rows = []
    log = open(ROOT / "02_data" / "raw" / f"missing_parents_api_{snap}.jsonl", "a", encoding="utf-8")
    s = requests.Session()
    for pid, n in miss.items():
        row = {"parent_id": pid, "children": int(n), "status": None, "resolved_id": None, "in_snapshot": False}
        lp = local_path_id(pid)
        if lp:
            hit = low.get(lp.lower())
            row.update(status="local_path", resolved_id=hit or lp, in_snapshot=bool(hit))
        elif n >= args.min_children:
            url = "https://huggingface.co/api/models/" + quote(pid, safe="/")
            for attempt in range(5):
                try:
                    r = s.get(url, timeout=30, allow_redirects=True)
                except requests.RequestException:
                    time.sleep(2 ** attempt); continue
                if r.status_code == 429:
                    m = re.search(r"t=(\d+)", r.headers.get("RateLimit", ""))
                    time.sleep(int(m.group(1)) + 1 if m else 60); continue
                break
            rec = {"parent_id": pid, "http": r.status_code}
            if r.status_code == 200:
                rid = r.json().get("id") or r.json().get("modelId")
                rec["api_id"] = rid
                if rid and rid.lower() != pid.lower():
                    hit = low.get(rid.lower())
                    row.update(status="alias" if "/" not in pid else "renamed", resolved_id=hit or rid,
                               in_snapshot=bool(hit))
                else:
                    row.update(status="same")
            else:
                row.update(status="unavailable")
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            time.sleep(args.sleep)
        else:
            row.update(status="not_checked")
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(P / "missing_parents_resolved.csv", index=False)
    chk = d[d["status"] != "not_checked"]
    print("by status (parents):", chk["status"].value_counts().to_dict())
    print("by status (edges):", chk.groupby("status")["children"].sum().to_dict())
    print("edges reconnectable (resolved id in snapshot):", int(d.loc[d["in_snapshot"], "children"].sum()),
          "of", int(d["children"].sum()))
    print(d[d["status"].isin(["renamed", "alias", "local_path"])].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
