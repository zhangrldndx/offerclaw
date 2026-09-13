#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask one question about the ten Dev-New golds no channel reaches.

R@5 already equals the candidate-pool ceiling, so every remaining Dev-New miss
is a *recall* failure, not a ranking one.  Widening the pool was measured and
rejected (pool44 buys one gold at reranked rank 37).  What is left is the query
side, and the cheap decisive question is not "does end-to-end R@1 move" but
"does this query text put the gold into the channel at all".

Answering that needs no judge, no reranker and no full evaluation run: embed the
variant, ask the frozen collection, and read off the gold's rank.  A variant
that cannot reach the gold here cannot help downstream either, whatever an
end-to-end A/B would show, so this probe is run *before* anything expensive.

It also measures a rewrite on the BM25 side, which production does not do today
(``rewrite_query`` only feeds the dense embedding).  If lexical rewriting is
what reaches these golds, that is a code change worth making rather than a knob
worth flipping.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # scripts/ is sys.path[0], the repo is not
DEPTH = 60


def _rank(ids: list[str], gold: set[str]) -> int:
    for position, chunk_id in enumerate(ids, start=1):
        if chunk_id in gold:
            return position
    return 0


def _dense_ids(text: str, collection, depth: int = DEPTH) -> list[str]:
    from rag_retrieval_trace import stable_chunk_id
    from rag_tools import get_embeddings_batch

    emb = get_embeddings_batch([text])
    got = collection.query(query_embeddings=emb, n_results=depth,
                           include=["documents", "metadatas"])
    docs = (got.get("documents") or [[]])[0]
    metas = (got.get("metadatas") or [[]])[0]
    ids = (got.get("ids") or [[]])[0]
    return [i or stable_chunk_id(d, m) for i, d, m in zip(ids, docs, metas)]


def _bm25_ids(text: str, depth: int = DEPTH) -> list[str]:
    from rag_bm25 import bm25_search

    from rag_retrieval_trace import stable_chunk_id

    # bm25_search yields (document, metadata, score); the chunk id is derived
    # the same way the evaluator derives it, so ranks are comparable.
    return [stable_chunk_id(document, meta)
            for document, meta, _score in (bm25_search(text, depth) or [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="docs/rag_eval/colloquial/live/v2adev_pool28_d12_gate_v2.json")
    parser.add_argument("--cases", default="docs/rag_eval/colloquial/rag_colloquial_v2a_dev_with_negatives.json")
    parser.add_argument("--output", default="docs/rag_eval/colloquial/live/unreachable_probe.json")
    args = parser.parse_args()

    rows = json.loads((ROOT / args.run).read_text(encoding="utf-8"))["runs"][0]["positive"]["rows"]
    questions = {item["query_id"]: item["question"]
                 for item in json.loads((ROOT / args.cases).read_text(encoding="utf-8"))["items"]}
    targets = [r for r in rows if not r["reranked_rank"]]
    print(f"[probe] 池外 gold 的题: {len(targets)}", flush=True)

    import chromadb
    from rag_tools import get_collection_name

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection(get_collection_name())

    os.environ["RAG_QUERY_REWRITE"] = "1"
    os.environ["RAG_HYDE"] = "1"
    from rag_hyde import hyde_expand, rewrite_query

    results = []
    for index, row in enumerate(targets, start=1):
        query_id = row["query_id"]
        gold = set(row["grade3_chunk_ids"])
        question = questions[query_id]
        variants = {"original": question,
                    "rewrite": rewrite_query(question),
                    "hyde": hyde_expand(question)}
        entry = {"query_id": query_id, "query_style": row["query_style"],
                 "rewrite_text": variants["rewrite"][len(question):].strip(),
                 "dense": {}, "bm25": {}}
        for name, text in variants.items():
            entry["dense"][name] = _rank(_dense_ids(text, collection), gold)
            entry["bm25"][name] = _rank(_bm25_ids(text), gold)
        results.append(entry)
        print(f"[probe] {index}/{len(targets)} {query_id} "
              f"dense {entry['dense']} bm25 {entry['bm25']}", flush=True)

    payload = {"schema_version": "unreachable-gold-probe-v1", "depth": DEPTH,
               "n": len(results), "rows": results}
    out = ROOT / args.output
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
