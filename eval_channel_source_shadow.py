#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the single preregistered channel-source candidate shadow.

The candidate generator is deliberately label blind: it accepts only stable
query IDs and question text, freezes all three sets, and only then does the
evaluation entrypoint load file labels, qrels, or the historical A0 artifact.
The resulting pool is an isolated diagnostic and cannot select a production
retrieval profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SET_PATHS = {
    "heldout52": ROOT / "tests" / "rag_bench_paraphrase_set.json",
    "zh_final90": ROOT / "tests" / "zh_final_set.json",
    "rag_bench100": ROOT / "tests" / "rag_bench_set.json",
}
DEFAULT_QRELS = (
    ROOT / "docs" / "rag_eval" / "qrels" /
    "rag_bench_paraphrase_adjudicated.json"
)
DEFAULT_A0 = (
    ROOT / "logs" / "rag_eval" / "reranker_ab_3348" /
    "A0_current_mps.json"
)
DEFAULT_OUTPUT = (
    ROOT / "docs" / "rag_eval" / "r1_upgrade" /
    "channel_source_shadow_result.json"
)


def _items_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"evaluation set has no items: {path}")
    return items


def load_query_inputs(paths: dict[str, Path]) -> list[dict[str, str]]:
    """Load only candidate-visible fields, namespaced across evaluation sets."""

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for set_name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in _items_from_payload(payload, path):
            query_id = str(item["id"])
            trace_id = f"{set_name}::{query_id}"
            if trace_id in seen:
                raise ValueError(f"duplicate namespaced query ID: {trace_id}")
            seen.add(trace_id)
            output.append({
                "id": trace_id,
                "q": str(item["q"]),
            })
    return output


def load_labeled_sets(paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Load labels only after candidate membership has been frozen."""

    return {
        set_name: _items_from_payload(
            json.loads(path.read_text(encoding="utf-8")), path,
        )
        for set_name, path in paths.items()
    }


def embed_query_inputs_by_namespace(items, embedder):
    """Embed each frozen set as its own stable batch, preserving input order.

    The production A0 artifact embedded heldout52 as one 52-row batch.  Mixing
    unrelated 90/100-set rows into that tensor can introduce last-bit model
    batching drift at a dense tie.  Namespaced batching reproduces the frozen
    evaluation unit without exposing labels or changing candidate semantics.
    """

    groups: dict[str, list[tuple[int, str]]] = {}
    for index, item in enumerate(items):
        trace_id = str(item["id"])
        if "::" not in trace_id:
            raise ValueError(f"candidate input is not namespaced: {trace_id}")
        namespace = trace_id.split("::", 1)[0]
        groups.setdefault(namespace, []).append((index, str(item["q"])))
    output: list[Any] = [None] * len(items)
    for rows in groups.values():
        vectors = embedder([question for _index, question in rows])
        if len(vectors) != len(rows):
            raise ValueError("namespace embedding batch did not preserve alignment")
        for (index, _question), vector in zip(rows, vectors):
            output[index] = vector
    if any(vector is None for vector in output):
        raise AssertionError("namespace embedding left an unfilled query vector")
    return output


def _reference_embedding_snapshot(collection):
    import numpy as np

    from eval_candidate_pools import _allowed_reference_meta

    snapshot = collection.get(include=["embeddings", "metadatas"])
    ids = [str(value) for value in (snapshot.get("ids") or [])]
    metas = list(snapshot.get("metadatas") or [])
    raw_embeddings = snapshot.get("embeddings")
    if raw_embeddings is None:
        raise ValueError("collection snapshot did not include stored embeddings")
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("collection embeddings must be a two-dimensional matrix")
    if len(ids) != len(metas) or len(ids) != embeddings.shape[0]:
        raise ValueError("collection IDs, metadata and embeddings are misaligned")

    records = []
    seen: dict[str, str] = {}
    for offset, (chunk_id, meta) in enumerate(zip(ids, metas)):
        if not isinstance(meta, dict) or not _allowed_reference_meta(meta):
            continue
        source = str(meta.get("source") or "").strip()
        if not source:
            raise ValueError(f"reference chunk has no source lineage: {chunk_id}")
        previous = seen.get(chunk_id)
        if previous is not None and previous != source:
            raise ValueError(f"duplicate chunk ID has conflicting source: {chunk_id}")
        if previous is not None:
            raise ValueError(f"duplicate stable chunk ID in collection: {chunk_id}")
        seen[chunk_id] = source
        records.append((chunk_id, source, offset))
    if not records:
        raise ValueError("reference embedding snapshot is empty")

    # Collection get order is not an identity contract.  A stable ID sort makes
    # both exact-score ties and trace generation deterministic.
    records.sort(key=lambda row: row[0])
    selected_offsets = [row[2] for row in records]
    matrix = embeddings[selected_offsets]
    row_norms = np.linalg.norm(matrix, axis=1)
    return (
        [row[0] for row in records],
        [row[1] for row in records],
        matrix,
        row_norms,
    )


def _source_key(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _source_distribution(sources: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(_source_key(source) for source in sources).items()))


def build_candidate_traces(
    items: list[dict[str, str]],
    collection,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Freeze label-blind baseline and shadow pools for every query."""

    import numpy as np

    from eval_candidate_pools import read_candidate_channels
    from rag_candidate_pool import select_candidate_pool
    from rag_channel_source_shadow import (
        ExactSourceCandidate,
        SHADOW_CONFIG,
        select_channel_source_pool,
        source_shortlist_from_global,
    )
    from rag_tools import get_embeddings_batch

    started = time.perf_counter()
    chunk_ids, sources, matrix, row_norms = _reference_embedding_snapshot(collection)
    source_rows: dict[str, list[int]] = {}
    for index, source in enumerate(sources):
        source_rows.setdefault(_source_key(source), []).append(index)
    snapshot_ms = (time.perf_counter() - started) * 1000

    embedding_started = time.perf_counter()
    query_vectors = embed_query_inputs_by_namespace(items, get_embeddings_batch)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    if len(query_vectors) != len(items):
        raise ValueError("query embedding batch did not preserve alignment")

    channel_started = time.perf_counter()
    channels = read_candidate_channels(
        items,
        collection,
        channel_depth=SHADOW_CONFIG["global_dense_depth"],
        query_embeddings=query_vectors,
    )
    channel_ms = (time.perf_counter() - channel_started) * 1000

    exact_total_ms = 0.0
    traces: dict[str, dict[str, Any]] = {}
    for item, raw_vector in zip(items, query_vectors):
        trace_id = str(item["id"])
        dense, bm25 = channels[trace_id]
        baseline = select_candidate_pool(
            dense,
            bm25,
            strategy="baseline_rrf20",
            pool_size=SHADOW_CONFIG["pool_size"],
        )
        shortlist = source_shortlist_from_global(
            baseline,
            protected_global_slots=SHADOW_CONFIG["protected_global_slots"],
        )
        selected_rows = sorted({
            row
            for source in shortlist
            for row in source_rows.get(_source_key(source), [])
        })
        if not selected_rows:
            raise ValueError(f"source shortlist has no exact-dense rows: {trace_id}")

        exact_started = time.perf_counter()
        query = np.asarray(raw_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != matrix.shape[1]:
            raise ValueError("query and collection embedding dimensions differ")
        query_norm = float(np.linalg.norm(query))
        selected_norms = row_norms[selected_rows] * query_norm
        selected_matrix = matrix[selected_rows]
        scores = np.divide(
            selected_matrix @ query,
            selected_norms,
            out=np.zeros(len(selected_rows), dtype=np.float32),
            where=selected_norms > 0,
        )
        exact = [
            ExactSourceCandidate(
                chunk_id=chunk_ids[row],
                source=sources[row],
                cosine_score=float(score),
            )
            for row, score in zip(selected_rows, scores)
        ]
        result = select_channel_source_pool(
            baseline,
            exact,
            pool_size=SHADOW_CONFIG["pool_size"],
            protected_global_slots=SHADOW_CONFIG["protected_global_slots"],
            exact_unseen_slots=SHADOW_CONFIG["exact_unseen_slots"],
        )
        exact_total_ms += (time.perf_counter() - exact_started) * 1000

        traces[trace_id] = {
            "baseline_pool_chunk_ids": [candidate.chunk_id for candidate in baseline],
            "baseline_pool_sources": [candidate.source for candidate in baseline],
            "source_shortlist": list(result.source_shortlist),
            "source_shortlist_chunk_count": len(selected_rows),
            "ranked_exact_top20": [
                {
                    "chunk_id": candidate.chunk_id,
                    "source": candidate.source,
                    "cosine_score": round(candidate.cosine_score, 9),
                }
                for candidate in result.ranked_exact[:20]
            ],
            "exact_added_ids": list(result.exact_added_ids),
            "backfilled_global_ids": list(result.backfilled_global_ids),
            "final_pool": [
                {
                    "chunk_id": candidate.chunk_id,
                    "source": candidate.source,
                    "origin": candidate.origin,
                    "global_rank": candidate.global_rank,
                    "exact_rank": candidate.exact_rank,
                    "cosine_score": candidate.cosine_score,
                }
                for candidate in result.pool
            ],
            "baseline_source_distribution": _source_distribution(
                candidate.source for candidate in baseline
            ),
            "candidate_source_distribution": _source_distribution(
                candidate.source for candidate in result.pool
            ),
        }

    return traces, {
        "queries": len(items),
        "reference_chunks": len(chunk_ids),
        "embedding_dimension": int(matrix.shape[1]),
        "snapshot_ms": round(snapshot_ms, 3),
        "query_embedding_ms": round(embedding_ms, 3),
        "global_channel_ms": round(channel_ms, 3),
        "source_exact_dense_ms": round(exact_total_ms, 3),
        "cross_encoder_calls": 0,
        "qrels_visible_during_candidate_generation": False,
        "expected_sources_visible_during_candidate_generation": False,
    }


def _hit_source(sources: Iterable[str], expected_sources: Iterable[str]) -> bool:
    expected = [str(value).casefold() for value in expected_sources]
    return any(
        any(label in str(source).casefold() for label in expected)
        for source in sources
    )


def _metric(candidate: dict[str, bool], baseline: dict[str, bool]) -> dict[str, Any]:
    rescued = sorted(
        query_id for query_id, hit in candidate.items()
        if hit and not baseline.get(query_id, False)
    )
    lost = sorted(
        query_id for query_id, hit in candidate.items()
        if not hit and baseline.get(query_id, False)
    )
    hits = sum(candidate.values())
    return {
        "hits": hits,
        "n": len(candidate),
        "candidate_recall": round(hits / len(candidate), 6) if candidate else None,
        "delta_hits_vs_baseline": hits - sum(baseline.values()),
        "rescued_query_ids": rescued,
        "lost_query_ids": lost,
    }


def score_file_level_sets(
    labeled_sets: dict[str, list[dict[str, Any]]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for set_name, items in labeled_sets.items():
        baseline_hits: dict[str, bool] = {}
        candidate_hits: dict[str, bool] = {}
        rows = []
        for item in items:
            query_id = str(item["id"])
            trace = traces[f"{set_name}::{query_id}"]
            expected = list(item.get("expect_sources") or [])
            baseline_hit = _hit_source(trace["baseline_pool_sources"], expected)
            candidate_sources = [row["source"] for row in trace["final_pool"]]
            candidate_hit = _hit_source(candidate_sources, expected)
            baseline_hits[query_id] = baseline_hit
            candidate_hits[query_id] = candidate_hit
            rows.append({
                "id": query_id,
                "baseline_hit": baseline_hit,
                "candidate_hit": candidate_hit,
                "pool_size": len(trace["final_pool"]),
                "source_shortlist_count": len(trace["source_shortlist"]),
                "source_shortlist_chunk_count": trace["source_shortlist_chunk_count"],
                "exact_added_ids": list(trace["exact_added_ids"]),
            })
        reports[set_name] = {
            "claim_scope": "file_level_source_label_only_not_direct_chunk_or_final_r1",
            "baseline": _metric(baseline_hits, baseline_hits),
            "candidate": _metric(candidate_hits, baseline_hits),
            "rows": rows,
        }
    return reports


def score_heldout_dev_direct(
    *,
    heldout_items: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    scoped_qrels,
    a0_rows: dict[str, dict[str, Any]],
    expected_baseline_hits: int = 34,
) -> dict[str, Any]:
    """Score design-informed direct qrels as DEV ONLY, never for promotion."""

    baseline_hits: dict[str, bool] = {}
    candidate_hits: dict[str, bool] = {}
    exact_order: dict[str, bool] = {}
    rows = []
    for item in heldout_items:
        query_id = str(item["id"])
        direct_ids = set(scoped_qrels.strict_child_ids.get(query_id, set()))
        trace = traces[f"heldout52::{query_id}"]
        baseline_ids = list(trace["baseline_pool_chunk_ids"])
        candidate_ids = [row["chunk_id"] for row in trace["final_pool"]]
        if direct_ids:
            baseline_hits[query_id] = bool(set(baseline_ids) & direct_ids)
            candidate_hits[query_id] = bool(set(candidate_ids) & direct_ids)
        expected_ids = list(a0_rows.get(query_id, {}).get("fusion_chunk_ids") or [])
        if expected_ids:
            exact_order[query_id] = baseline_ids == expected_ids
        rows.append({
            "id": query_id,
            "direct_target_chunk_ids": sorted(direct_ids),
            "baseline_hit": baseline_hits.get(query_id),
            "candidate_hit": candidate_hits.get(query_id),
            "a0_exact_chunk_id_order": exact_order.get(query_id),
        })
    if len(baseline_hits) != 50:
        raise ValueError(f"direct qrels coverage drifted: {len(baseline_hits)} != 50")
    if sum(baseline_hits.values()) != expected_baseline_hits:
        raise ValueError(
            "production direct Candidate@20 did not reproduce: "
            f"{sum(baseline_hits.values())} != {expected_baseline_hits}"
        )
    if len(exact_order) != 52 or not all(exact_order.values()):
        mismatches = sorted(key for key, value in exact_order.items() if not value)
        raise ValueError(
            "candidate reader no longer reproduces all 52 A0 stable ID orders: "
            f"checked={len(exact_order)}, mismatches={mismatches}"
        )
    return {
        "claim_scope": "DEV_ONLY_DESIGN_INFORMED_NOT_PROMOTION_EVIDENCE",
        "baseline": _metric(baseline_hits, baseline_hits),
        "candidate": _metric(candidate_hits, baseline_hits),
        "a0_stable_id_order": {
            "checked": len(exact_order),
            "exact_matches": sum(exact_order.values()),
            "all_exact": all(exact_order.values()),
        },
        "rows": rows,
    }


def shadow_signal(file_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = ("zh_final90", "rag_bench100")
    missing = [name for name in required if name not in file_reports]
    if missing:
        raise ValueError(f"shadow signal missing frozen sets: {missing}")
    no_regression = all(
        file_reports[name]["candidate"]["hits"]
        >= file_reports[name]["baseline"]["hits"]
        for name in required
    )
    wins = sum(
        len(file_reports[name]["candidate"]["rescued_query_ids"])
        for name in required
    )
    losses = sum(
        len(file_reports[name]["candidate"]["lost_query_ids"])
        for name in required
    )
    warrants = no_regression and wins > losses
    return {
        "decision": "WARRANTS_NEW_BLIND_EVAL" if warrants else "STOP",
        "checks": {
            "both_untouched_file_sets_non_regressing": no_regression,
            "combined_file_wins_gt_losses": wins > losses,
        },
        "combined_file_wins": wins,
        "combined_file_losses": losses,
        "promotion": "BLOCKED_PENDING_NEW_DIRECT_QRELS_BLIND_OR_REAL_SHADOW_SET",
        "production_default_changed": False,
        "final_r1_claimed": False,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb

    from eval_candidate_pools import _a0_rows, _load_scoped_qrels
    from rag_channel_source_shadow import SHADOW_CONFIG
    from rag_tools import get_collection_name, index_fingerprint

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection(get_collection_name())

    # Isolation boundary: only namespaced IDs and question text exist in this
    # object.  All 242 memberships are frozen before labels/qrels/A0 are read.
    query_inputs = load_query_inputs(SET_PATHS)
    traces, candidate_trace = build_candidate_traces(query_inputs, collection)

    labeled_sets = load_labeled_sets(SET_PATHS)
    file_reports = score_file_level_sets(labeled_sets, traces)
    heldout_items = labeled_sets["heldout52"]
    scoped_qrels, qrels_payload = _load_scoped_qrels(
        args.qrels,
        [str(item["id"]) for item in heldout_items],
        collection,
        client=client,
    )
    direct_dev = score_heldout_dev_direct(
        heldout_items=heldout_items,
        traces=traces,
        scoped_qrels=scoped_qrels,
        a0_rows=_a0_rows(args.a0),
    )
    signal = shadow_signal(file_reports)
    return {
        "schema_version": "channel-source-candidate-shadow-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "algorithm": {
            "name": "protected_global16_source_exact_dense_add4",
            "configuration": dict(SHADOW_CONFIG),
            "configuration_cli_overridable": False,
            "source_shortlist": "all_unique_sources_in_protected_global16",
            "source_local_retrieval": "exact_cosine_over_all_stored_C0_chunk_embeddings",
            "combination": "protect_global16_add_top4_unseen_then_global_backfill",
            "query_rewrite": False,
            "alias": False,
            "quota": False,
            "global_recall_n_expanded": False,
            "production_path_changed": False,
        },
        "isolation": {
            "candidate_generation_completed_before_labels_or_qrels_loaded": True,
            "candidate_visible_fields": ["namespaced_id", "question_text"],
            "all_sets_frozen_together": True,
        },
        "sets": {name: str(path) for name, path in SET_PATHS.items()},
        "qrels": {
            "path": str(args.qrels),
            "reviewer_id": qrels_payload["reviewer_id"],
            "loaded_after_all_candidate_traces": True,
            "heldout_claim_scope": "DEV_ONLY_DESIGN_INFORMED",
        },
        "a0": str(args.a0),
        "index": index_fingerprint(collection=collection, cache_ttl=0),
        "candidate_trace": candidate_trace,
        "file_level_sets": file_reports,
        "heldout52_direct_qrels_dev_only": direct_dev,
        "shadow_signal": signal,
        "promotion": {
            "decision": "BLOCKED",
            "reason": "heldout_direct_qrels_informed_design_and_untouched_sets_have_only_file_labels",
            "required_next_evidence": "new_direct_qrels_blind_or_real_shadow_set",
            "production_default_changed": False,
            "final_r1_claimed": False,
        },
        "traces": traces,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--a0", type=Path, default=DEFAULT_A0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = evaluate(args)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
