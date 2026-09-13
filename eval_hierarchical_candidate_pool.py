#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One preregistered, qrels-blind source→section candidate experiment.

Candidate generation sees only the question and collection-derived metadata.
Qrels are loaded *after* all candidate traces have been built and are used only
for scoring.  This script never calls the cross encoder and cannot alter the
production retrieval profile.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SET = ROOT / "tests" / "rag_bench_paraphrase_set.json"
DEFAULT_QRELS = (
    ROOT / "docs" / "rag_eval" / "qrels" /
    "rag_bench_paraphrase_adjudicated.json"
)
DEFAULT_A0 = ROOT / "logs" / "rag_eval" / "reranker_ab_3348" / "A0_current_mps.json"


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"evaluation set has no items: {path}")
    return items


def _cosine_scores(matrix, vector):
    import numpy as np

    rows = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(vector, dtype=np.float32)
    if rows.ndim != 2 or query.ndim != 1 or rows.shape[1] != query.shape[0]:
        raise ValueError("catalog and query embeddings have incompatible dimensions")
    row_norms = np.linalg.norm(rows, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominator = row_norms * query_norm
    return np.divide(
        rows @ query,
        denominator,
        out=np.zeros(rows.shape[0], dtype=np.float32),
        where=denominator > 0,
    ).tolist()


def _source_where(sources: list[str]) -> dict[str, Any]:
    from eval_candidate_pools import _reference_where

    if not sources:
        raise ValueError("hierarchical lookup requires at least one selected source")
    base = _reference_where()
    clauses = list(base.get("$and") or [base])
    clauses.append({"source": {"$in": list(sources)}})
    return {"$and": clauses}


def _reference_snapshot(collection) -> tuple[list[str], list[dict]]:
    from eval_candidate_pools import _allowed_reference_meta

    snapshot = collection.get(include=["metadatas"])
    ids = [str(value) for value in (snapshot.get("ids") or [])]
    metas = list(snapshot.get("metadatas") or [])
    if len(ids) != len(metas):
        raise ValueError("collection returned misaligned IDs and metadata")
    selected = [
        (chunk_id, meta)
        for chunk_id, meta in zip(ids, metas)
        if isinstance(meta, dict) and _allowed_reference_meta(meta)
    ]
    if not selected:
        raise ValueError("reference metadata snapshot is empty")
    return [item[0] for item in selected], [item[1] for item in selected]


def build_candidate_traces(
    items: list[dict[str, Any]],
    collection,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Generate every pool without reading qrels or expected source labels."""

    from rank_bm25 import BM25Okapi
    from eval_candidate_pools import read_candidate_channels
    from rag_bm25 import _tokenize
    from rag_candidate_pool import select_candidate_pool
    from rag_hierarchical_pool import (
        PREREGISTERED_HIERARCHY_CONFIG,
        RestrictedCandidate,
        build_source_section_catalog,
        rank_source_catalog,
        select_hierarchical_pool,
    )
    from rag_tools import get_embeddings_batch

    started = time.perf_counter()
    _chunk_ids, reference_metas = _reference_snapshot(collection)
    catalog = build_source_section_catalog(reference_metas)
    if not catalog:
        raise ValueError("source/title catalog is empty")
    questions = [str(item["q"]) for item in items]
    combined_vectors = get_embeddings_batch([
        *(entry.text for entry in catalog),
        *questions,
    ])
    catalog_vectors = combined_vectors[:len(catalog)]
    query_vectors = combined_vectors[len(catalog):]
    if len(query_vectors) != len(items):
        raise ValueError("embedding batch did not preserve question alignment")
    catalog_bm25 = BM25Okapi([_tokenize(entry.text) for entry in catalog])
    catalog_ms = (time.perf_counter() - started) * 1000

    channel_started = time.perf_counter()
    channels = read_candidate_channels(
        items,
        collection,
        channel_depth=20,
        query_embeddings=query_vectors,
    )
    global_channel_ms = (time.perf_counter() - channel_started) * 1000

    traces: dict[str, dict[str, Any]] = {}
    restricted_total_ms = 0.0
    for item, query_vector in zip(items, query_vectors):
        query_id = str(item["id"])
        dense_scores = _cosine_scores(catalog_vectors, query_vector)
        lexical_scores = catalog_bm25.get_scores(_tokenize(str(item["q"]))).tolist()
        ranked_sources = rank_source_catalog(
            catalog,
            dense_scores,
            lexical_scores,
            source_limit=PREREGISTERED_HIERARCHY_CONFIG["source_limit"],
            rrf_k=PREREGISTERED_HIERARCHY_CONFIG["catalog_rrf_k"],
        )
        selected_sources = [source.source for source in ranked_sources]

        restricted_started = time.perf_counter()
        restricted_result = collection.query(
            query_embeddings=[query_vector],
            n_results=PREREGISTERED_HIERARCHY_CONFIG["restricted_dense_depth"],
            where=_source_where(selected_sources),
            include=["metadatas", "distances"],
        )
        restricted_total_ms += (time.perf_counter() - restricted_started) * 1000
        ids = (restricted_result.get("ids") or [[]])[0]
        metas = (restricted_result.get("metadatas") or [[]])[0]
        distances = (restricted_result.get("distances") or [[]])[0]
        if not (len(ids) == len(metas) == len(distances)):
            raise ValueError(f"restricted lookup returned misaligned rows for {query_id}")
        restricted = [
            RestrictedCandidate(
                chunk_id=str(chunk_id),
                source=str((meta or {}).get("source") or ""),
                title=str((meta or {}).get("title") or ""),
                dense_rank=rank,
                distance=float(distance),
            )
            for rank, (chunk_id, meta, distance) in enumerate(
                zip(ids, metas, distances), start=1,
            )
        ]

        dense, bm25 = channels[query_id]
        baseline = select_candidate_pool(
            dense, bm25, strategy="baseline_rrf20", pool_size=20,
        )
        result = select_hierarchical_pool(
            baseline,
            restricted,
            ranked_sources,
            pool_size=PREREGISTERED_HIERARCHY_CONFIG["pool_size"],
            protected_global_slots=(
                PREREGISTERED_HIERARCHY_CONFIG["protected_global_slots"]
            ),
            hierarchical_slots=(
                PREREGISTERED_HIERARCHY_CONFIG["hierarchical_slots"]
            ),
            rrf_k=PREREGISTERED_HIERARCHY_CONFIG["candidate_rrf_k"],
        )
        traces[query_id] = {
            "baseline_pool_chunk_ids": [candidate.chunk_id for candidate in baseline],
            "baseline_pool_sources": [candidate.source for candidate in baseline],
            "selected_sources": [{
                "source": source.source,
                "rank": source.rank,
                "best_catalog_id": source.best_catalog_id,
                "best_title": source.best_title,
                "best_rrf_score": round(source.best_rrf_score, 10),
            } for source in ranked_sources],
            "restricted_candidates": [{
                "chunk_id": candidate.chunk_id,
                "source": candidate.source,
                "title": candidate.title,
                "dense_rank": candidate.dense_rank,
                "distance": candidate.distance,
            } for candidate in restricted],
            "ranked_hierarchical_candidates": [{
                "chunk_id": candidate.chunk_id,
                "source": candidate.source,
                "title": candidate.title,
                "dense_rank": candidate.dense_rank,
                "source_rank": candidate.source_rank,
                "section_rank": candidate.section_rank,
                "hierarchy_score": round(candidate.hierarchy_score, 10),
            } for candidate in result.ranked_restricted],
            "hierarchy_added_ids": list(result.hierarchy_added_ids),
            "protected_global_ids": list(result.protected_global_ids),
            "backfilled_global_ids": list(result.backfilled_global_ids),
            "final_pool": [{
                "chunk_id": candidate.chunk_id,
                "source": candidate.source,
                "origin": candidate.origin,
                "global_rank": candidate.global_rank,
                "hierarchy_rank": candidate.hierarchy_rank,
            } for candidate in result.pool],
        }
    return traces, {
        "catalog_entries": len(catalog),
        "catalog_sources": len({entry.source for entry in catalog}),
        "catalog_embedding_ms": round(catalog_ms, 3),
        "global_channel_ms": round(global_channel_ms, 3),
        "restricted_dense_ms": round(restricted_total_ms, 3),
        "cross_encoder_calls": 0,
        "qrels_visible_during_candidate_generation": False,
        "expected_sources_visible_during_candidate_generation": False,
    }


def _hit_source(sources: list[str], expected_sources: list[str]) -> bool:
    expected = [str(value).casefold() for value in expected_sources]
    return any(
        any(label in str(source).casefold() for label in expected)
        for source in sources
    )


def _metric(hits: dict[str, bool], baseline: dict[str, bool]) -> dict[str, Any]:
    rescued = sorted(
        query_id for query_id, hit in hits.items()
        if hit and not baseline.get(query_id, False)
    )
    lost = sorted(
        query_id for query_id, hit in hits.items()
        if not hit and baseline.get(query_id, False)
    )
    count = sum(hits.values())
    return {
        "hits": count,
        "n": len(hits),
        "candidate_recall": round(count / len(hits), 6) if hits else None,
        "delta_hits_vs_baseline": count - sum(baseline.values()),
        "rescued_query_ids": rescued,
        "lost_query_ids": lost,
    }


def score_candidate_traces(
    *,
    items: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    scoped_qrels,
    direct_target_sources: dict[str, set[str]],
    a0_rows: dict[str, dict[str, Any]],
    expected_baseline_direct_hits: int = 34,
    expected_baseline_file_hits: int = 42,
) -> dict[str, Any]:
    """Apply qrels only after candidate generation and enforce GO gates."""

    baseline_direct: dict[str, bool] = {}
    candidate_direct: dict[str, bool] = {}
    baseline_file: dict[str, bool] = {}
    candidate_file: dict[str, bool] = {}
    stage1_source: dict[str, bool] = {}
    rows = []
    exact_reproduction = []
    reproduction_mismatches = []
    for item in items:
        query_id = str(item["id"])
        trace = traces[query_id]
        baseline_ids = list(trace["baseline_pool_chunk_ids"])
        final_ids = [row["chunk_id"] for row in trace["final_pool"]]
        final_sources = [row["source"] for row in trace["final_pool"]]
        direct_ids = set(scoped_qrels.strict_child_ids.get(query_id, set()))
        if direct_ids:
            baseline_direct[query_id] = bool(set(baseline_ids) & direct_ids)
            candidate_direct[query_id] = bool(set(final_ids) & direct_ids)
            selected_source_keys = {
                str(row["source"]).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                for row in trace["selected_sources"]
            }
            target_source_keys = {
                str(source).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                for source in direct_target_sources.get(query_id, set())
            }
            stage1_source[query_id] = bool(
                selected_source_keys & target_source_keys
            )
        baseline_file[query_id] = _hit_source(
            list(trace["baseline_pool_sources"]), item.get("expect_sources", []),
        )
        candidate_file[query_id] = _hit_source(
            final_sources, item.get("expect_sources", []),
        )
        expected_ids = list(a0_rows.get(query_id, {}).get("fusion_chunk_ids") or [])
        exact = (baseline_ids == expected_ids) if expected_ids else None
        if exact is not None:
            exact_reproduction.append(exact)
            if exact is False:
                reproduction_mismatches.append(query_id)
        rows.append({
            "id": query_id,
            "direct_target_chunk_ids": sorted(direct_ids),
            "baseline_direct_hit": baseline_direct.get(query_id),
            "candidate_direct_hit": candidate_direct.get(query_id),
            "stage1_target_source_hit": stage1_source.get(query_id),
            "baseline_file_hit": baseline_file[query_id],
            "candidate_file_hit": candidate_file[query_id],
            "baseline_a0_exact_order": exact,
            **trace,
        })

    baseline_direct_metric = _metric(baseline_direct, baseline_direct)
    candidate_direct_metric = _metric(candidate_direct, baseline_direct)
    baseline_file_metric = _metric(baseline_file, baseline_file)
    candidate_file_metric = _metric(candidate_file, baseline_file)
    stage1_hits = sum(stage1_source.values())
    stage1_source_metric = {
        "hits": stage1_hits,
        "n": len(stage1_source),
        "candidate_recall": (
            round(stage1_hits / len(stage1_source), 6)
            if stage1_source else None
        ),
        "hit_query_ids": sorted(
            query_id for query_id, hit in stage1_source.items() if hit
        ),
        "missed_query_ids": sorted(
            query_id for query_id, hit in stage1_source.items() if not hit
        ),
    }
    if baseline_direct_metric["n"] != 50:
        raise ValueError(
            f"production qrels coverage drifted: {baseline_direct_metric['n']} != 50"
        )
    if baseline_direct_metric["hits"] != expected_baseline_direct_hits:
        raise ValueError(
            "production direct Candidate@20 did not reproduce: "
            f"{baseline_direct_metric['hits']} != {expected_baseline_direct_hits}"
        )
    if baseline_file_metric["n"] != 52:
        raise ValueError(f"file-level set drifted: {baseline_file_metric['n']} != 52")
    if baseline_file_metric["hits"] != expected_baseline_file_hits:
        raise ValueError(
            "production file Candidate@20 did not reproduce: "
            f"{baseline_file_metric['hits']} != {expected_baseline_file_hits}"
        )
    if exact_reproduction and not all(exact_reproduction):
        raise ValueError(
            "candidate reader no longer reproduces A0 stable chunk order: "
            f"{reproduction_mismatches}"
        )

    checks = {
        "direct_candidate_at_least_37_of_50": (
            candidate_direct_metric["hits"] >= 37
        ),
        "direct_losses_at_most_1": (
            len(candidate_direct_metric["lost_query_ids"]) <= 1
        ),
        "file_candidate_at_least_42_of_52": (
            candidate_file_metric["hits"] >= 42
        ),
        "pool_size_exactly_20": all(
            len(trace["final_pool"]) == 20 for trace in traces.values()
        ),
        "a0_order_reproduced": bool(exact_reproduction) and all(exact_reproduction),
    }
    return {
        "baseline": {
            "direct_qrels_candidate_recall": baseline_direct_metric,
            "file_level_candidate_recall": baseline_file_metric,
        },
        "hierarchical": {
            "stage1_direct_source_recall": stage1_source_metric,
            "direct_qrels_candidate_recall": candidate_direct_metric,
            "file_level_candidate_recall": candidate_file_metric,
        },
        "go_no_go": {
            "decision": "GO" if all(checks.values()) else "STOP",
            "checks": checks,
            "production_default_changed": False,
            "final_r1_claimed": False,
        },
        "rows": rows,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb
    from eval_candidate_pools import _a0_rows, _load_scoped_qrels
    from rag_hierarchical_pool import PREREGISTERED_HIERARCHY_CONFIG
    from rag_tools import get_collection_name, index_fingerprint

    items = _load_items(args.set)
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection(get_collection_name())

    # Critical isolation boundary: all candidate membership is frozen before
    # the evaluator opens qrels or historical expected-source labels.
    candidate_inputs = [
        {"id": str(item["id"]), "q": str(item["q"])}
        for item in items
    ]
    traces, candidate_trace = build_candidate_traces(candidate_inputs, collection)

    query_ids = [str(item["id"]) for item in items]
    scoped_qrels, qrels_payload = _load_scoped_qrels(
        args.qrels, query_ids, collection, client=client,
    )
    direct_target_sources = {
        str(item["query_id"]): {
            str(target["source"])
            for target in item["relevant_targets"]
            if target.get("relevance") == "direct"
            and target.get("evidence_scope", "child") == "child"
        }
        for item in qrels_payload["items"]
    }
    a0_rows = _a0_rows(args.a0)
    scored = score_candidate_traces(
        items=items,
        traces=traces,
        scoped_qrels=scoped_qrels,
        direct_target_sources=direct_target_sources,
        a0_rows=a0_rows,
        expected_baseline_direct_hits=args.expected_baseline_direct_hits,
        expected_baseline_file_hits=args.expected_baseline_file_hits,
    )
    return {
        "schema_version": "hierarchical-candidate-experiment-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "algorithm": {
            "name": "source_section_catalog_rrf_then_restricted_dense",
            "configuration": dict(PREREGISTERED_HIERARCHY_CONFIG),
            "configuration_cli_overridable": False,
            "catalog_inputs": ["metadata.source", "metadata.title"],
            "catalog_fusion": "dense_title_rank_plus_BM25_title_rank_RRF",
            "restricted_lookup": "dense_within_selected_sources",
            "combination": "protect_global16_add_hierarchy4_then_global_backfill",
            "qrels_used_for_parameters": False,
            "query_rewrite": False,
            "alias": False,
            "quota": False,
            "global_recall_n_expanded": False,
        },
        "set": str(args.set),
        "qrels": {
            "path": str(args.qrels),
            "reviewer_id": qrels_payload["reviewer_id"],
            "loaded_after_candidate_generation": True,
        },
        "a0": str(args.a0),
        "index": index_fingerprint(collection=collection, cache_ttl=0),
        "candidate_trace": candidate_trace,
        **scored,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--a0", type=Path, default=DEFAULT_A0)
    parser.add_argument("--expected-baseline-direct-hits", type=int, default=34)
    parser.add_argument("--expected-baseline-file-hits", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = evaluate(args)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
