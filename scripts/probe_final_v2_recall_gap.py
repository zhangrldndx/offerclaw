#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Size the remaining recall gap on Final v2, before building anything.

On Final v2 the binding constraint is candidate formation, not ranking: the best
arm reaches 60 of 80 golds and converts 67% of what it reaches.  So the question
worth answering first is not "which reranking change helps" but "is the missing
gold reachable by any query text at all".

The variant that matters is the fourth one.  Production feeds a *single* dense
query, and HyDE replaces it with "question + hypothetical answer" -- so a
fabricated answer does not merely add candidates, it moves the whole query.  On
Final v2 that cost one genuine false accept (a question about benchmark data the
corpus does not contain).  ``union`` simulates retrieving with the original query
*and* the HyDE query and merging: it keeps the original query's precision while
still buying HyDE's reach, and it is measurable here without writing it first.

Judge-free and reranker-free by construction: this measures channels, not
ranking.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPTH = 60


def _rank(ids: list[str], gold: set[str]) -> int:
    for position, chunk_id in enumerate(ids, start=1):
        if chunk_id in gold:
            return position
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="一次已完成的臂产出，用来找池外的题")
    parser.add_argument("--dataset", default="docs/rag_eval/final_v2/final_v2.json")
    parser.add_argument("--pool", type=int, default=28, help="判定「进池」的深度")
    parser.add_argument("--output", default="docs/rag_eval/final_v2/RECALL_GAP.json")
    args = parser.parse_args()

    rows = json.loads(Path(args.run).expanduser().read_text(encoding="utf-8"))[
        "runs"][0]["positive"]["rows"]
    questions = {i["query_id"]: i["question"]
                 for i in json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["items"]}
    targets = [r for r in rows if not r["reranked_rank"]]
    print(f"[gap] 池外的题: {len(targets)}/{len(rows)}", flush=True)

    import chromadb
    from rag_bm25 import bm25_search
    from rag_retrieval_trace import stable_chunk_id
    from rag_tools import get_collection_name, get_embeddings_batch

    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name())

    def dense_ids(text: str) -> list[str]:
        got = collection.query(query_embeddings=get_embeddings_batch([text]),
                               n_results=DEPTH, include=["documents", "metadatas"])
        return [i or stable_chunk_id(d, m) for i, d, m in
                zip(got["ids"][0], got["documents"][0], got["metadatas"][0])]

    def bm25_ids(text: str) -> list[str]:
        return [stable_chunk_id(doc, meta)
                for doc, meta, _s in (bm25_search(text, DEPTH) or [])]

    os.environ["RAG_HYDE"] = "1"
    os.environ["RAG_QUERY_REWRITE"] = "1"
    from rag_hyde import hyde_expand, rewrite_query

    results = []
    for index, row in enumerate(targets, start=1):
        query_id = row["query_id"]
        gold = set(row["grade3_chunk_ids"])
        question = questions[query_id]
        hyde = hyde_expand(question, enabled=True)
        variants = {"original": question, "rewrite": rewrite_query(question, enabled=True),
                    "hyde": hyde}
        entry = {"query_id": query_id, "query_style": row["query_style"],
                 "dense": {}, "bm25": {}}
        dense_lists = {}
        for name, text in variants.items():
            dense_lists[name] = dense_ids(text)
            entry["dense"][name] = _rank(dense_lists[name], gold)
            entry["bm25"][name] = _rank(bm25_ids(text), gold)
        # Interleave the two dense result lists rather than concatenating: a
        # union that simply appends would put every HyDE candidate behind every
        # original one and hide whether HyDE reaches the gold early.
        merged, seen = [], set()
        for pair in zip(dense_lists["original"], dense_lists["hyde"]):
            for chunk_id in pair:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    merged.append(chunk_id)
        entry["dense"]["union"] = _rank(merged, gold)
        results.append(entry)
        print(f"[gap] {index}/{len(targets)} {query_id} dense={entry['dense']} "
              f"bm25={entry['bm25']}", flush=True)

    def reachable(entry, pool):
        return any(0 < entry[channel][name] <= pool
                   for channel in ("dense", "bm25") for name in entry[channel])

    summary = {}
    for name in ("original", "rewrite", "hyde", "union"):
        hit = sum(1 for e in results
                  if (0 < e["dense"].get(name, 0) <= args.pool)
                  or (0 < e["bm25"].get(name, 0) <= args.pool))
        summary[name] = hit
    payload = {"schema_version": "final-v2-recall-gap-v1", "depth": DEPTH,
               "pool": args.pool, "n_out_of_pool": len(results),
               "reachable_within_pool": summary, "rows": results}
    out = ROOT / args.output
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[gap] 池外 {len(results)} 题中，各变体能送进 depth<= {args.pool} 的题数:")
    for name, hit in summary.items():
        print(f"    {name:9s} {hit}")
    print(f"[gap] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
