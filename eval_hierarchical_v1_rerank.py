#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final one-shot rerank of the frozen v1 hierarchical candidate pools.

This evaluator does not retrieve, gate, rewrite, or tune anything.  It reads
the already-frozen pool20 membership, scores the body text with production
``bge-reranker-base``, and compares Top-1 with the immutable A0 artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parent
V1 = ROOT / "docs" / "rag_eval" / "r1_upgrade" / "hierarchical_candidate_result.json"
A0 = ROOT / "logs" / "rag_eval" / "reranker_ab_3348" / "A0_current_mps.json"
EVAL_SET = ROOT / "tests" / "rag_bench_paraphrase_set.json"
OUTPUT = ROOT / "docs" / "rag_eval" / "r1_upgrade" / "hierarchical_v1_rerank_result.json"
MODEL = "BAAI/bge-reranker-base"
P95_LIMIT_MS = 11836.0


def _hit_source(source: str, expected_sources: list[str]) -> bool:
    value = str(source or "").casefold()
    return any(str(expected).casefold() in value for expected in expected_sources)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def _comparison(candidate: dict[str, bool], baseline: dict[str, bool]) -> dict:
    wins = sorted(key for key, hit in candidate.items() if hit and not baseline[key])
    losses = sorted(key for key, hit in candidate.items() if not hit and baseline[key])
    return {
        "baseline_hits": sum(baseline.values()),
        "candidate_hits": sum(candidate.values()),
        "n": len(candidate),
        "delta_hits": sum(candidate.values()) - sum(baseline.values()),
        "wins": wins,
        "losses": losses,
    }


def main() -> int:
    import chromadb

    from rag_rerank import _load_reranker, rerank
    from rag_tools import get_collection_name, index_fingerprint

    v1 = json.loads(V1.read_text(encoding="utf-8"))
    a0 = json.loads(A0.read_text(encoding="utf-8"))
    items_payload = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    items = items_payload["items"]
    questions = {str(item["id"]): item for item in items}
    v1_rows = {str(row["id"]): row for row in v1["rows"]}
    a0_rows = {
        str(row["id"]): row
        for set_result in a0["sets"] if set_result["set"] == "heldout52"
        for row in set_result["runs"][0]["rows"]
    }
    expected_ids = set(questions)
    if set(v1_rows) != expected_ids or set(a0_rows) != expected_ids:
        raise ValueError("v1/A0/set query membership drifted")
    if a0["profile"] != {
        "name": "A0", "pool_size": 20,
        "reranker_model": MODEL, "reranker_use_breadcrumb": False,
        "chunker_version": "2026-08-10", "enable_hyde": False,
        "enable_query_rewrite": False, "enable_doc2query": False,
        "enable_quota": False, "enable_alias": False,
        "enable_rerank_bridge": False,
    }:
        raise ValueError("A0 profile contract drifted")

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection(get_collection_name())
    current_index = index_fingerprint(collection=collection, cache_ttl=0)
    if current_index["collection_content_hash"] != a0["index"]["collection_content_hash"]:
        raise ValueError("live collection no longer matches A0")

    os.environ["RAG_RERANK"] = "1"
    os.environ["RAG_RERANK_MODEL"] = MODEL
    os.environ.pop("RAG_RERANK_MAX_SEQ", None)
    load_started = time.perf_counter()
    model = _load_reranker(MODEL)
    model_load_ms = (time.perf_counter() - load_started) * 1000
    if model is None:
        raise RuntimeError("production reranker failed to load")

    rows = []
    for index, item in enumerate(items, start=1):
        query_id = str(item["id"])
        print(f"[v1-base-body] {index}/52 {query_id}", flush=True)
        frozen = v1_rows[query_id]
        pool_ids = [str(row["chunk_id"]) for row in frozen["final_pool"]]
        if len(pool_ids) != 20 or len(set(pool_ids)) != 20:
            raise ValueError(f"invalid frozen pool20 for {query_id}")
        snapshot = collection.get(ids=pool_ids, include=["documents", "metadatas"])
        found = {
            str(chunk_id): (document, dict(meta or {}))
            for chunk_id, document, meta in zip(
                snapshot["ids"], snapshot["documents"], snapshot["metadatas"]
            )
        }
        if set(found) != set(pool_ids):
            raise ValueError(f"frozen v1 chunks are missing for {query_id}")
        docs = []
        metas = []
        for chunk_id in pool_ids:
            document, meta = found[chunk_id]
            meta["_eval_chunk_id"] = chunk_id
            docs.append(document)
            metas.append(meta)
        runtime = {}
        started = time.perf_counter()
        _docs, ranked_metas, _dists, scores = rerank(
            str(item["q"]), docs, metas, [0.0] * 20, 20,
            model_name=MODEL, runtime_diag=runtime,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if runtime != {
            "requested": MODEL, "actual": MODEL, "status": "ok", "scored_count": 20,
        } or len(scores) != 20:
            raise RuntimeError(f"incomplete reranker scoring for {query_id}: {runtime}")
        ranked_ids = [str(meta["_eval_chunk_id"]) for meta in ranked_metas]
        top1_id = ranked_ids[0]
        top1_source = str(ranked_metas[0].get("source") or "")
        direct_targets = set(frozen["direct_target_chunk_ids"])
        baseline = a0_rows[query_id]
        rows.append({
            "id": query_id,
            "top1_chunk_id": top1_id,
            "top1_source": top1_source,
            "file_r1": _hit_source(top1_source, list(item["expect_sources"])),
            "direct_targets": sorted(direct_targets),
            "direct_r1": bool(direct_targets and top1_id in direct_targets),
            "a0_file_r1": int(baseline["rank"]) == 1,
            "a0_direct_r1": bool(direct_targets and int(baseline["direct_rank"]) == 1),
            "latency_ms": round(latency_ms, 3),
            "top5_chunk_ids": ranked_ids[:5],
            "top5_scores": scores[:5],
        })

    file_candidate = {row["id"]: row["file_r1"] for row in rows}
    file_baseline = {row["id"]: row["a0_file_r1"] for row in rows}
    covered = [row for row in rows if row["direct_targets"]]
    direct_candidate = {row["id"]: row["direct_r1"] for row in covered}
    direct_baseline = {row["id"]: row["a0_direct_r1"] for row in covered}
    if sum(file_baseline.values()) != 25 or sum(direct_baseline.values()) != 21:
        raise ValueError("A0 R@1 baseline did not reproduce from frozen artifact")
    latencies = [row["latency_ms"] for row in rows]
    p50 = round(_percentile(latencies, 0.50), 3)
    p95 = round(_percentile(latencies, 0.95), 3)
    payload = {
        "schema_version": "hierarchical-v1-rerank-final-v1",
        "experiment": "single_final_full_heldout52",
        "configuration": {
            "candidate_pool": str(V1), "pool_size": 20,
            "reranker": MODEL, "input": "body_only",
            "device": os.environ.get("OFFERCLAW_TORCH_DEVICE") or "auto",
            "gate_run": False, "other_sets_run": False,
            "parameters_changed": False, "production_switched": False,
        },
        "index": current_index,
        "model_load_ms_excluded_from_query_latency": round(model_load_ms, 3),
        "file_level_r1": _comparison(file_candidate, file_baseline),
        "direct_r1": _comparison(direct_candidate, direct_baseline),
        "latency": {
            "scope": "reranker_wall_only_frozen_candidates",
            "p50_ms": p50, "p95_ms": p95,
            "mean_ms": round(statistics.fmean(latencies), 3),
            "limit_ms": P95_LIMIT_MS,
            "p95_within_limit": p95 <= P95_LIMIT_MS,
        },
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
