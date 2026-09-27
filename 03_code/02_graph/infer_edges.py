"""선언되지 않은 계보 엣지 추론 + 순환 제거.

입력  nodes.parquet, edges.parquet (build_graph.py 출력)
출력  edges_all.parquet   선언 엣지 + 추론 엣지. 열: source ∈ {declared, inferred_mirror, inferred_name}
                          in_cycle = True인 엣지는 분석에서 제외
      inference_report.json

추론 규칙 (보수적: 이름이 스냅샷 안에서 유일하게 대응될 때만)
    inferred_name    RichardErkhov 양식 "<author>_-_<model>[-gguf|-4bits|-8bits|...]"
                     → <author>/<model>
    inferred_mirror  부모를 선언하지 않았고, 이름(작성자 제외)이 자식 20개 이상인
                     기반 모델의 이름과 정확히 같으며, 그 기반 모델보다 나중에 올라온 모델
                     → 그 기반 모델을 부모로 (relation = "mirror")
    inferred_quant   양자화 접미사(-GGUF, -AWQ, -GPTQ, -EXL2, -MLX, -Nbit, -bnb-4bit)가 붙었고
                     부모 선언이 없으며, 접미사를 뗀 이름이 스냅샷에서 정확히 1개 모델에만 대응
                     → 그 모델을 부모로 (relation = "quantized")

사용
    python infer_edges.py ../../02_data/processed/2026-09-25
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

MIRROR_MIN_CHILDREN = 20
QUANT_SUFFIX = re.compile(
    r"[-_.](?:i1-)?(?:gguf|awq|gptq|exl2|mlx(?:-\d+bits?)?|\d+(?:\.\d+)?-?bits?|\d+bpw|bnb-4bit|"
    r"q\d(?:_k)?(?:_[msl])?|fp8|int[48])(?:[-_].*)?$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()

    n = pd.read_parquet(args.processed / "nodes.parquet",
                        columns=["model_id", "author", "created_at", "created_at_legacy", "n_parents"])
    e = pd.read_parquet(args.processed / "edges.parquet")
    e["source"] = "declared"
    # 2026-09-26: 스냅샷에 없던 선언 부모 중 이름 변경·옛 ID·로컬 경로로 확인된 것을 실제 ID 로 다시 잇는다
    # (resolve_missing_parents.py). 이전에는 이 자식들이 부모와 끊겨 독립 계보로 세어졌다.
    e["parent_resolution"] = None
    rp = args.processed / "missing_parents_resolved.csv"
    if rp.exists():
        rs = pd.read_csv(rp)
        rs = rs[rs["in_snapshot"].astype(bool) & rs["status"].isin(["renamed", "alias", "local_path"])]
        mp = rs.set_index("parent_id")["resolved_id"]
        st = rs.set_index("parent_id")["status"]
        hit = ~e["parent_in_snapshot"] & e["parent_id"].isin(mp.index)
        e.loc[hit, "parent_resolution"] = e.loc[hit, "parent_id"].map(st)
        e.loc[hit, "parent_id_declared"] = e.loc[hit, "parent_id"]
        e.loc[hit, "parent_id"] = e.loc[hit, "parent_id"].map(mp)
        e.loc[hit, "parent_in_snapshot"] = True
        e = e[e["parent_id"] != e["child_id"]].drop_duplicates(["parent_id", "child_id"])
        created0 = n.set_index("model_id")["created_at"]
        legacy0 = n.set_index("model_id")["created_at_legacy"]
        # 다시 이은 엣지는 시간 판정을 하지 않는다 (NA): 연결된 저장소의 생성 시각은 원본의 업로드 시각이 아니다.
        # 예) runwayml/stable-diffusion-v1-5 → stable-diffusion-v1-5/stable-diffusion-v1-5 는 원본이 사라진 뒤
        # 만든 재배포 저장소라 생성 시각이 늦다. 시간 판정을 하면 2,540개 엣지가 '시간 역전'으로 빠져 재연결이 무효가 된다.
        h = e["parent_resolution"].notna()
        e.loc[h, "temporal_ok"] = pd.NA
        print(f"reconnected {int(h.sum()):,} declared edges via name resolution "
              f"({e.loc[h, 'parent_resolution'].value_counts().to_dict()})")

    name = n["model_id"].str.split("/", n=1).str[1]
    low_id = n["model_id"].str.lower()
    id_by_low = pd.Series(n["model_id"].to_numpy(), index=low_id)
    id_by_low = id_by_low[~id_by_low.index.duplicated()]
    created = n.set_index("model_id")["created_at"]
    orphan = n["n_parents"].eq(0)

    # name_low -> 모델 목록 (유일성 판정용)
    name_low = name.str.lower()
    name_count = name_low.value_counts()

    inferred = []

    # 1) RichardErkhov 양식
    m = n["author"].eq("RichardErkhov") & orphan
    parts = name[m].str.extract(r"^(?P<a>.+?)_-_(?P<b>.+)$")
    base = parts["b"].str.replace(QUANT_SUFFIX, "", regex=True)
    cand = (parts["a"] + "/" + base).str.lower().map(id_by_low)
    ok = cand.notna()
    inferred.append(pd.DataFrame({"parent_id": cand[ok].to_numpy(), "child_id": n.loc[m, "model_id"][ok].to_numpy(),
                                  "relation": "quantized", "source": "inferred_name"}))

    # 2) 미러
    outdeg = e[e["parent_in_snapshot"]].groupby("parent_id").size()
    popular = outdeg[outdeg >= MIRROR_MIN_CHILDREN].index
    pop = pd.DataFrame({"parent_id": popular, "nl": pd.Series(popular).str.split("/", n=1).str[1].str.lower()})
    pop = pop[pop["nl"].map(pop["nl"].value_counts()).eq(1)]          # 같은 이름의 인기 모델이 여럿이면 모호 → 제외
    pmap = pop.set_index("nl")["parent_id"]
    m = orphan & name_low.isin(pmap.index) & ~n["model_id"].isin(popular)
    par = name_low[m].map(pmap)
    later = n.loc[m, "created_at"].to_numpy() >= par.map(created).to_numpy()
    ch = n.loc[m, "model_id"][later]
    inferred.append(pd.DataFrame({"parent_id": par[later].to_numpy(), "child_id": ch.to_numpy(),
                                  "relation": "mirror", "source": "inferred_mirror"}))

    # 3) 양자화 접미사
    done = set(pd.concat(inferred)["child_id"])
    has_q = name.str.contains(QUANT_SUFFIX, regex=True)
    m = orphan & has_q & ~n["model_id"].isin(done)
    stripped = name[m].str.replace(QUANT_SUFFIX, "", regex=True).str.lower()
    uniq = stripped.map(name_count).eq(1)                              # 스냅샷에서 이름이 유일
    nl_to_id = pd.Series(n["model_id"].to_numpy(), index=name_low)
    nl_to_id = nl_to_id[~nl_to_id.index.duplicated(keep=False)]
    par = stripped[uniq].map(nl_to_id)
    ok = par.notna() & (par != n.loc[m, "model_id"][uniq])
    ch = n.loc[m, "model_id"][uniq][ok]
    par = par[ok]
    later = ch.map(created).to_numpy() >= par.map(created).to_numpy()
    inferred.append(pd.DataFrame({"parent_id": par[later].to_numpy(), "child_id": ch[later].to_numpy(),
                                  "relation": "quantized", "source": "inferred_quant"}))

    inf = pd.concat(inferred, ignore_index=True).drop_duplicates(["parent_id", "child_id"])
    inf = inf[inf["parent_id"] != inf["child_id"]]
    inf["parent_in_snapshot"] = True
    pc, cc = inf["parent_id"].map(created), inf["child_id"].map(created)
    inf["temporal_ok"] = (pc <= cc)
    allE = pd.concat([e, inf], ignore_index=True).drop_duplicates(["parent_id", "child_id"])

    # 4) 순환: 강연결요소(크기 ≥ 2) 안의 엣지는 분석에서 제외
    idx = pd.Series(np.arange(len(n)), index=n["model_id"])
    inside = allE[allE["parent_in_snapshot"]]
    A = sparse.csr_matrix((np.ones(len(inside)), (idx[inside["parent_id"]].to_numpy(),
                                                  idx[inside["child_id"]].to_numpy())), shape=(len(n), len(n)))
    _, lab = connected_components(A, directed=True, connection="strong")
    big = np.bincount(lab) > 1
    allE["in_cycle"] = False
    cs = allE.index[allE["parent_in_snapshot"]]
    lp = lab[idx[allE.loc[cs, "parent_id"]].to_numpy()]
    lc = lab[idx[allE.loc[cs, "child_id"]].to_numpy()]
    allE.loc[cs, "in_cycle"] = (lp == lc) & big[lp]

    allE.to_parquet(args.processed / "edges_all.parquet", index=False)
    rep = {
        "declared_edges": int(len(e)),
        "declared_edges_reconnected_by_name_resolution": int(e["parent_resolution"].notna().sum()),
        "inferred_edges": inf["source"].value_counts().to_dict(),
        "inferred_temporal_violations": int((~inf["temporal_ok"]).sum()),
        "models_gaining_parent": int(inf["child_id"].nunique()),
        "cycle_components": int(big.sum()),
        "edges_in_cycles": int(allE["in_cycle"].sum()),
        "total_edges": int(len(allE)),
        "examples": {s: g[["parent_id", "child_id"]].head(5).values.tolist() for s, g in inf.groupby("source")},
    }
    (args.processed / "inference_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
