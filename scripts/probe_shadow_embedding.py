#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dead-end probe: does bge-large-zh reach what base-zh cannot?

The last unclosed hole in the falsification matrix: every embedding swap tried
was multilingual (m3, e5, mMiniLM) and lost Chinese ground; a *stronger Chinese*
embedding was never tested.  The questions probed here are the accumulated
dead ends -- realworld-52 rank-0s, Final v4 stable out-of-pool, Stage-2 no-term
misses -- i.e. exactly the deep-paraphrase shapes where the embedding space
itself is the suspect.

Dense-only, depth 60, both models, plus a regression bed of questions base-zh
already serves.  No production index change; the landing shape on GO would be
an auxiliary channel over the shadow collection (rank fusion, distances folded
back to base-zh space), not a model swap -- per the user's explicit constraint.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHADOW = "kb_shadow_large_zh_v1"
LARGE = "BAAI/bge-large-zh-v1.5"
DEPTH = 60


def main() -> None:
    os.environ.setdefault("OFFERCLAW_EMBED_MAX_SEQ", "512")
    os.environ.setdefault("OFFERCLAW_TORCH_DEVICE", "cpu")
    import chromadb
    from rag_tools import _local_embed, get_collection_name, get_embeddings_batch

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    main_col = client.get_collection(get_collection_name())
    shadow_col = client.get_collection(SHADOW)
    assert shadow_col.count() == main_col.count(), "影子集合块数与主库不一致"

    # --- dead-end questions ---------------------------------------------
    cases = []
    bench = json.loads((ROOT / "docs/rag_eval/final_v4/bench52_qualitydefault.json"
                        ).read_text(encoding="utf-8"))
    heldout = {i["id"]: i for i in json.loads(
        (ROOT / "tests/rag_bench_paraphrase_set.json").read_text(encoding="utf-8"))["items"]}
    for row in bench["rows"]:
        if row.get("rank") == 0 and row["id"] in heldout:
            cases.append({"id": f"rw-{row['id']}", "q": heldout[row["id"]]["q"],
                          "gold_kind": "source_substring",
                          "gold": heldout[row["id"]]["expect_sources"], "bed": "dead"})
    v4 = {i["query_id"]: i for i in json.loads(
        (ROOT / "docs/rag_eval/final_v4/final_v4.json").read_text(encoding="utf-8"))["items"]}
    for qid in ("fv4-a00-natural", "fv4-a16-natural", "fv4-a26-implicit_oral",
                "fv4-a22-standard", "fv4-a28-standard", "fv4-a02-standard",
                "fv4-a15-long_noisy"):
        item = v4[qid]
        cases.append({"id": qid, "q": item["question"], "gold_kind": "chunk",
                      "gold": [t["chunk_id"] for t in item["relevant_targets"]], "bed": "dead"})
    s2 = json.loads((ROOT / "docs/rag_eval/next_stage/STAGE2_PILOT_RESULTS.json"
                     ).read_text(encoding="utf-8"))["rows"]["noterm"]
    anchors = json.loads((ROOT / "docs/rag_eval/next_stage/STAGE2_ANCHOR_CANDIDATES.json"
                          ).read_text(encoding="utf-8"))["anchors"]
    scope: dict = {}
    exec((ROOT / "docs/rag_eval/next_stage/stage2_noterm.py").read_text(encoding="utf-8"), scope)
    noterm_q = {f"s2-{i:02d}": (aidx, q) for i, (aidx, q) in enumerate(scope["Q"], 1)}
    for row in s2:
        aidx, q = noterm_q[row["query_id"]]
        bed = "dead" if row["rank_prod"] == 0 else "regression"
        cases.append({"id": row["query_id"], "q": q, "gold_kind": "chunk",
                      "gold": [anchors[aidx]["chunk_id"]], "bed": bed})

    # --- probe both spaces ----------------------------------------------
    def rank_in(col, embed_fn, case) -> int:
        emb = embed_fn([case["q"]])
        got = col.query(query_embeddings=emb, n_results=DEPTH,
                        include=["metadatas"])
        ids = got["ids"][0]
        metas = got["metadatas"][0]
        for position, (chunk_id, meta) in enumerate(zip(ids, metas), 1):
            if case["gold_kind"] == "chunk" and chunk_id in set(case["gold"]):
                return position
            if case["gold_kind"] == "source_substring":
                source = ((meta or {}).get("source") or "").lower()
                if any(g.lower() in source for g in case["gold"]):
                    return position
        return 0

    base_embed = lambda texts: get_embeddings_batch(texts)
    large_embed = lambda texts: _local_embed(texts, LARGE)
    rows = []
    for index, case in enumerate(cases, 1):
        row = {"id": case["id"], "bed": case["bed"],
               "rank_base": rank_in(main_col, base_embed, case),
               "rank_large": rank_in(shadow_col, large_embed, case)}
        rows.append(row)
        print(f"[emb] {index}/{len(cases)} {case['id']:22s} {case['bed']:10s} "
              f"base={row['rank_base']:>2} large={row['rank_large']:>2}", flush=True)

    POOL = 28
    dead = [r for r in rows if r["bed"] == "dead"]
    reg = [r for r in rows if r["bed"] == "regression"]
    rescued = [r["id"] for r in dead
               if not (0 < r["rank_base"] <= POOL) and 0 < r["rank_large"] <= POOL]
    lost = [r["id"] for r in reg
            if 0 < r["rank_base"] <= POOL and not (0 < r["rank_large"] <= POOL)]
    summary = {"dead_total": len(dead), "rescued_within_pool": rescued,
               "regression_total": len(reg), "regression_lost": lost,
               "dead_base_reach": sum(1 for r in dead if 0 < r["rank_base"] <= POOL),
               "dead_large_reach": sum(1 for r in dead if 0 < r["rank_large"] <= POOL)}
    out = ROOT / "docs/rag_eval/next_stage/SHADOW_EMB_PROBE.json"
    out.write_text(json.dumps({"schema_version": "shadow-embedding-probe-v1",
                               "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[emb] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
