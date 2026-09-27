"""Zenodo 데이터 레코드 업로드 (02_data/zenodo_upload.md 의 파일·메타데이터를 API 로 올린다).

토큰: ~/.zenodo_token (한 줄) 또는 환경변수 ZENODO_TOKEN. 필요한 권한: deposit:write, deposit:actions.
상태: 02_data/zenodo_state.json 에 deposition id 를 저장해 두므로 중단 후 다시 실행하면 이어서 올린다.
기본은 초안(draft)까지만 만들고 DOI 를 예약한다. --publish 를 주면 공개한다 (공개 후 삭제 불가).

사용: python zenodo_upload.py            # 초안 생성 + 파일 업로드 + MD5 검증
      python zenodo_upload.py --publish  # 검증 후 공개
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data"
STATE = DATA / "zenodo_state.json"
API = "https://zenodo.org/api"
SOFTWARE_CONCEPT_DOI = "10.5281/zenodo.22997474"
GITHUB = "https://github.com/aikian/hf-model-supply-chain"

FILES = [
    DATA / "raw" / "hf_models_2026-09-25.jsonl.gz",
    DATA / "raw" / "license_names_2026-09-25.jsonl",
    DATA / "raw" / "missing_parents_api_2026-09-25.jsonl",
    *[DATA / "processed" / "2026-09-25" / f"{n}.parquet"
      for n in ["nodes", "edges", "edges_all", "attributes", "model_flags", "arch", "dataset_edges"]],
    DATA / "processed" / "2026-09-25" / "missing_parents_resolved.csv",
    *sorted((DATA / "processed" / "2026-09-25").glob("*.json")),
    DATA / "MANIFEST_2026-09-25.json",
    DATA / "data_dictionary.md",
]

METADATA = {
    "upload_type": "dataset",
    "title": ("Hugging Face model lineage snapshot (2026-09-25) for \"Lineage Diversity and "
              "Metadata-Defined Option Survival in the Hugging Face Model Supply Chain\""),
    "creators": [{"name": "An, Donggyu", "affiliation": "Yeungnam University", "orcid": "0009-0004-9977-4168"}],
    "description": (
        "<p>Metadata snapshot of all 3,094,856 public models on the Hugging Face Hub, collected on 2026-09-25 "
        "through the public Hub API, with the reconstructed lineage graph (951,033 edges: declared, reconnected, "
        "and inferred), cleaning flags, analysis tiers, and the derived task/language/license attributes used in "
        "the paper. Column descriptions are in <code>data_dictionary.md</code>; SHA-256 hashes of every file are in "
        "<code>MANIFEST_2026-09-25.json</code>.</p>"
        "<p>Cleaning flags are heuristics used to decide which models count as providers of a functional option; "
        "they are not judgments about uploaders. Model metadata remains subject to the Hugging Face Hub terms of "
        "service; the derived tables are released under CC BY 4.0.</p>"
        f"<p>Code and results: <a href=\"{GITHUB}\">{GITHUB}</a> "
        f"(archived as software at <a href=\"https://doi.org/{SOFTWARE_CONCEPT_DOI}\">{SOFTWARE_CONCEPT_DOI}</a>).</p>"
    ),
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": ["Hugging Face", "pre-trained models", "software supply chain", "model lineage",
                 "mining software repositories"],
    "related_identifiers": [
        {"identifier": SOFTWARE_CONCEPT_DOI, "relation": "isSupplementedBy", "resource_type": "software"},
        {"identifier": GITHUB, "relation": "isSupplementedBy", "resource_type": "software"},
    ],
    "version": "2026-09-25",
    "language": "eng",
}


def token():
    t = os.environ.get("ZENODO_TOKEN") or (Path.home() / ".zenodo_token").read_text(encoding="utf-8").strip()
    if not t:
        sys.exit("no token: put it in ~/.zenodo_token or ZENODO_TOKEN")
    return t


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def req(method, url, *, retries=4, **kw):
    kw.setdefault("timeout", (60, 1800))
    for i in range(retries):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code < 500:
                return r
            print(f"  {method} {url.split('/api/')[-1]} -> {r.status_code}, retry {i + 1}", flush=True)
        except requests.RequestException as e:
            print(f"  {method} failed ({type(e).__name__}), retry {i + 1}", flush=True)
        time.sleep(15 * (i + 1))
    sys.exit(f"gave up: {method} {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--parallel", type=int, default=6, help="동시에 올릴 파일(또는 조각) 수")
    ap.add_argument("--only", default=None, help="이 이름의 파일 하나만 올린다 (다른 파일은 건드리지 않음)")
    ap.add_argument("--multipart", type=int, default=0,
                    help="이 크기(MB)보다 큰 파일은 InvenioRDM 멀티파트로 조각내 올린다 (0 = 끔). 큰 PUT 이 502 로 죽을 때 사용")
    args = ap.parse_args()
    global FILES
    if args.only:
        FILES = [p for p in FILES if p.name in args.only.split(",")]
        print("only:", [p.name for p in FILES])
    missing = [p for p in FILES if not p.exists()]
    if missing:
        sys.exit("missing files: " + ", ".join(str(m) for m in missing))
    names = [p.name for p in FILES]
    assert len(names) == len(set(names)), "duplicate file names"
    params = {"access_token": token()}

    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if state.get("id") and args.only:
        # --only 모드: 브라우저가 올리는 중인 pending 항목이 있으면 deposit API 가 500 을 내므로 RDM 파일 API 만 쓴다
        dep = {"id": state["id"], "links": {}, "metadata": {}}
        print("resuming deposition", dep["id"], "(only mode, RDM files API)")
    elif state.get("id"):
        r = req("GET", f"{API}/deposit/depositions/{state['id']}", params=params)
        if r.status_code != 200:
            sys.exit(f"cannot load deposition {state['id']}: {r.status_code} {r.text[:300]}")
        dep = r.json()
        print("resuming deposition", dep["id"], "state", dep.get("state"))
    else:
        r = req("POST", f"{API}/deposit/depositions", params=params, json={})
        if r.status_code != 201:
            sys.exit(f"create failed: {r.status_code} {r.text[:500]}")
        dep = r.json()
        state = {"id": dep["id"]}
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print("created deposition", dep["id"])

    if dep.get("submitted"):
        print("already published:", dep.get("doi"))
        return

    # 메타데이터 (예약 DOI 포함)
    if not args.only:
        r = req("PUT", f"{API}/deposit/depositions/{dep['id']}", params=params,
                json={"metadata": {**METADATA, "prereserve_doi": True}})
        if r.status_code != 200:
            sys.exit(f"metadata failed: {r.status_code} {r.text[:800]}")
        dep = r.json()
        doi = dep["metadata"].get("prereserve_doi", {}).get("doi") or dep.get("doi")
        state["reserved_doi"] = doi
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print("reserved DOI:", doi)

    # 파일 업로드 (이미 있고 MD5 가 같으면 건너뜀). 원격 목록은 RDM 파일 API (pending 항목이 있어도 동작)
    bucket = dep["links"].get("bucket")
    HB0 = {"Authorization": f"Bearer {params['access_token']}"}
    r = requests.get(f"{API}/records/{dep['id']}/draft/files", headers=HB0, timeout=120)
    remote = {e["key"]: (e.get("checksum") or "").replace("md5:", "") for e in r.json().get("entries", [])
              if e.get("status") == "completed"} if r.status_code == 200 else {}
    total = sum(p.stat().st_size for p in FILES)
    todo = []
    for p in FILES:
        local = md5(p)
        if remote.get(p.name) == local:
            print(f"  skip (already uploaded) {p.name}")
            continue
        if p.name in remote:  # 손상된 업로드는 지우고 다시
            requests.delete(f"{API}/records/{dep['id']}/draft/files/{p.name}", headers=HB0, timeout=120)
        todo.append((p, local))
    todo.sort(key=lambda x: -x[0].stat().st_size)  # 큰 파일부터

    def upload_one(item):
        p, local = item
        size = p.stat().st_size
        if bucket is None:  # --only 모드: deposit API 를 못 쓰므로 RDM 단일 파일 업로드 (init -> content PUT -> commit)
            RDM = f"{API}/records/{dep['id']}/draft/files"
            for attempt in range(5):
                print(f"  uploading {p.name} ({size / 1048576:.1f} MB) via RDM" + (f", attempt {attempt + 1}" if attempt else "") + " ...", flush=True)
                t = time.time()
                try:
                    if requests.get(f"{RDM}/{p.name}", headers=HB0, timeout=120).status_code == 200:
                        requests.delete(f"{RDM}/{p.name}", headers=HB0, timeout=120)
                    r = requests.post(RDM, headers=HB0, timeout=120, json=[{"key": p.name}])
                    if r.status_code != 201:
                        print(f"    {p.name}: init HTTP {r.status_code} {r.text[:100]}", flush=True); time.sleep(30); continue
                    with open(p, "rb") as fh:
                        r = requests.put(f"{RDM}/{p.name}/content", headers={**HB0, "Content-Type": "application/octet-stream"},
                                         data=fh, timeout=(60, 3600))
                    if r.status_code not in (200, 201):
                        print(f"    {p.name}: content HTTP {r.status_code} {r.text[:100]}", flush=True); time.sleep(30 * (attempt + 1)); continue
                    r = requests.post(f"{RDM}/{p.name}/commit", headers=HB0, timeout=900)
                    if r.status_code not in (200, 201):
                        print(f"    {p.name}: commit HTTP {r.status_code} {r.text[:100]}", flush=True); time.sleep(30); continue
                except requests.RequestException as e:
                    print(f"    {p.name}: {type(e).__name__}, retrying", flush=True); time.sleep(30 * (attempt + 1)); continue
                got = (requests.get(f"{RDM}/{p.name}", headers=HB0, timeout=120).json().get("checksum") or "").replace("md5:", "")
                if got != local:
                    print(f"    {p.name}: MD5 mismatch ({got[:8]}), re-uploading", flush=True); continue
                print(f"    ok {p.name}, md5 verified, {size / 1048576 / max(time.time() - t, 1e-6):.2f} MB/s", flush=True)
                return None
            return f"gave up on {p.name} after 5 attempts"
        for attempt in range(5):
            print(f"  uploading {p.name} ({size / 1048576:.1f} MB)" + (f", attempt {attempt + 1}" if attempt else "") + " ...",
                  flush=True)
            t = time.time()
            try:
                with open(p, "rb") as fh:  # 재시도마다 파일을 처음부터 다시 읽는다 (같은 핸들을 재사용하면 빈 본문이 간다)
                    r = requests.put(f"{bucket}/{p.name}", params=params, data=fh, timeout=(60, 1800))
            except requests.RequestException as e:
                print(f"    {p.name}: {type(e).__name__}, retrying", flush=True)
                time.sleep(20 * (attempt + 1))
                continue
            if r.status_code >= 500:
                print(f"    {p.name}: HTTP {r.status_code}, retrying", flush=True)
                time.sleep(20 * (attempt + 1))
                continue
            if r.status_code not in (200, 201):
                return f"upload failed for {p.name}: {r.status_code} {r.text[:500]}"
            got = r.json()["checksum"].replace("md5:", "")
            if got != local:
                print(f"    {p.name}: MD5 mismatch (remote {got[:8]}), re-uploading", flush=True)
                continue
            print(f"    ok {p.name}, md5 verified, {size / 1048576 / max(time.time() - t, 1e-6):.2f} MB/s", flush=True)
            return None
        return f"gave up on {p.name} after 5 attempts"

    # InvenioRDM 멀티파트: 큰 파일을 조각으로 나눠 병렬 PUT 후 commit. 조각 하나가 실패해도 그 조각만 다시 보낸다.
    RDM = f"{API}/records/{dep['id']}/draft/files"
    HB = {"Authorization": f"Bearer {params['access_token']}"}

    def multipart_upload(p, local):
        part = args.multipart * 1024 * 1024
        size = p.stat().st_size
        nparts = -(-size // part)
        if requests.get(f"{RDM}/{p.name}", headers=HB, timeout=120).status_code == 200:  # 남은 pending 항목 제거
            requests.delete(f"{RDM}/{p.name}", headers=HB, timeout=120)
        entry = None
        for attempt in range(3):
            r = requests.post(RDM, headers=HB, timeout=300, json=[{"key": p.name, "size": size,
                              "transfer": {"type": "M", "parts": nparts, "part_size": part}}])
            if r.status_code == 201:
                entry = r.json()["entries"][0]
                break
            r2 = requests.get(f"{RDM}/{p.name}", headers=HB, timeout=120)  # 게이트웨이가 끊겨도 항목은 생겼을 수 있다
            if r2.status_code == 200 and r2.json().get("links", {}).get("parts"):
                entry = r2.json()
                break
            print(f"    {p.name}: multipart init HTTP {r.status_code}, retrying", flush=True)
            time.sleep(20)
        if entry is None:
            return f"{p.name}: multipart init failed"
        if not entry.get("links", {}).get("parts"):  # 목록 응답에는 조각 링크가 없을 수 있다: 항목을 직접 조회
            r = requests.get(f"{RDM}/{p.name}", headers=HB, timeout=120)
            entry = r.json() if r.status_code == 200 else entry
        links = entry.get("links", {}).get("parts")
        if not links:
            return f"{p.name}: no part links (status {entry.get('status')}, transfer {entry.get('transfer')}, links {list(entry.get('links', {}))})"
        print(f"  uploading {p.name} ({size / 1048576:.1f} MB) in {nparts} parts of {args.multipart} MB ...", flush=True)
        t0 = time.time()

        def put_part(i):
            with open(p, "rb") as fh:
                fh.seek(i * part)
                data = fh.read(part)
            msg = ""
            for a in range(6):
                t = time.time()
                try:
                    r = requests.put(links[i]["url"], headers={**HB, "Content-Type": "application/octet-stream"},
                                     data=data, timeout=(60, 900))
                    if r.status_code in (200, 201, 204):
                        print(f"    part {i + 1}/{nparts} ok ({len(data) / 1048576 / max(time.time() - t, 1e-6):.1f} MB/s)",
                              flush=True)
                        return None
                    msg = f"HTTP {r.status_code}"
                except requests.RequestException as e:
                    msg = type(e).__name__
                print(f"    part {i + 1}/{nparts}: {msg}, retry {a + 1}", flush=True)
                time.sleep(15 * (a + 1))
            return f"{p.name} part {i + 1}: {msg}"

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            errs = [e for e in ex.map(put_part, range(nparts)) if e]
        if errs:
            return "; ".join(errs[:3])
        r = requests.post(entry["links"]["commit"], headers=HB, timeout=900)
        if r.status_code not in (200, 201):
            return f"{p.name}: commit failed {r.status_code}: {r.text[:200]}"
        got = requests.get(f"{RDM}/{p.name}", headers=HB, timeout=120).json().get("checksum")
        if got != "md5:" + local:
            return f"{p.name}: checksum after commit {got} != local md5:{local}"
        print(f"    ok {p.name}, md5 verified, {size / 1048576 / max(time.time() - t0, 1e-6):.2f} MB/s overall", flush=True)
        return None

    from concurrent.futures import ThreadPoolExecutor
    errors = []
    if args.multipart:
        big = [x for x in todo if x[0].stat().st_size > args.multipart * 1024 * 1024]
        todo = [x for x in todo if x not in big]
        for p, local in big:  # 큰 파일은 하나씩, 조각은 병렬
            e = multipart_upload(p, local)
            if e:
                errors.append(e)
    # Zenodo 까지의 TCP 스트림 하나가 느려서 (약 50 KB/s) 작은 파일 여러 개를 동시에 올린다
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        errors += [e for e in ex.map(upload_one, todo) if e]
    if errors:
        sys.exit("\n".join(errors))

    # 최종 검증: 원격 파일 목록 = 로컬 목록, MD5 일치
    r = requests.get(f"{API}/records/{dep['id']}/draft/files", headers=HB0, timeout=120)
    remote = {e["key"]: (e.get("checksum") or "").replace("md5:", "") for e in r.json().get("entries", [])
              if e.get("status") == "completed"}
    bad = [p.name for p in FILES if remote.get(p.name) != md5(p)]
    extra = [] if args.only else sorted(set(remote) - set(names))
    if bad or extra:
        sys.exit(f"verification failed: mismatched {bad}, unexpected {extra}")
    print(f"verified {len(FILES)} files, {total / 1048576:.0f} MB. draft: https://zenodo.org/uploads/{dep['id']}")

    if args.publish and not args.only:
        r = req("POST", f"{API}/deposit/depositions/{dep['id']}/actions/publish", params=params)
        if r.status_code != 202:
            sys.exit(f"publish failed: {r.status_code} {r.text[:800]}")
        pub = r.json()
        state.update({"doi": pub.get("doi"), "conceptdoi": pub.get("conceptdoi"), "html": pub["links"].get("html")})
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print("PUBLISHED:", pub.get("doi"), "| all versions:", pub.get("conceptdoi"), "|", pub["links"].get("html"))
    else:
        print("not published (draft). Run with --publish after checking the draft page.")


if __name__ == "__main__":
    main()
