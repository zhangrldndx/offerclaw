#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Stage-C candidate-pool alternatives without a cross encoder.

This evaluator intentionally owns a *read-only candidate path*: one batched
dense query plus BM25 retrieval with the exact production ``reference_kb``
filters.  It never imports or monkeypatches the reranker.  The only metric that
can establish eligibility here is reviewer-approved **direct chunk candidate
recall**.  Source-title recall is reported as a historical diagnostic and
final R@1 is explicitly out of scope until the same pool is reranked.

Example::

    .venv/bin/python eval_candidate_pools.py \
      --qrels docs/rag_eval/qrels/rag_bench_paraphrase_adjudicated.json \
      --a0 /tmp/offerclaw_reranker_A0_current_mps.json \
      --output docs/rag_eval/r1_upgrade/stage_c_candidate_pools.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SET = ROOT / "tests" / "rag_bench_paraphrase_set.json"
DEFAULT_QRELS = ROOT / "docs" / "rag_eval" / "qrels" / "rag_bench_paraphrase_adjudicated.json"
STRATEGIES = (
    "baseline_rrf20",
    "source_cap4",
    "retain_channel_exclusives",
)


@dataclass(frozen=True)
class ScopedDirectQrels:
    """Direct relevance separated by the evidence unit the index can prove.

    ``strict_child_ids`` are direct gold chunks in the evaluated collection.
    ``parent_origin_ids`` are immutable reviewed parent IDs whose full evidence
    crosses a child boundary.  ``parent_trigger_child_ids`` are *not* promoted
    to direct gold: they only prove that retrieval could trigger deterministic
    parent expansion.
    """

    strict_child_ids: dict[str, set[str]]
    parent_origin_ids: dict[str, set[str]]
    parent_trigger_child_ids: dict[str, set[str]]
    production_collection: str = ""

    def combined_trigger_ids(self, query_id: str) -> set[str]:
        return set(self.strict_child_ids.get(query_id, set())) | set(
            self.parent_trigger_child_ids.get(query_id, set())
        )


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"evaluation set has no items: {path}")
    return items


def _collection_ids_and_metas(collection) -> tuple[list[str], list[dict]]:
    snapshot = collection.get(include=["metadatas"])
    ids = [str(value) for value in (snapshot.get("ids") or [])]
    metas = list(snapshot.get("metadatas") or [])
    if len(ids) != len(metas) or any(not isinstance(meta, dict) for meta in metas):
        raise ValueError("collection returned misaligned or invalid child metadata")
    return ids, metas


def _load_scoped_qrels(
    path: Path,
    query_ids: list[str],
    collection,
    *,
    client=None,
) -> tuple[ScopedDirectQrels, dict]:
    """Load child/parent qrels and fail closed on experiment lineage drift."""

    from rag_qrels import load_qrels_overlay, validate_qrels_against_collection

    payload = load_qrels_overlay(path, expected_query_ids=query_ids)
    if hasattr(collection, "count") and int(payload["index"]["count"]) != int(collection.count()):
        raise ValueError(
            "qrels index count mismatch: "
            f"overlay={payload['index']['count']} live={collection.count()}"
        )

    child_ids, child_metas = _collection_ids_and_metas(collection)
    production_values = {
        str(meta.get("production_collection") or "").strip()
        for meta in child_metas if str(meta.get("production_collection") or "").strip()
    }
    parent_targets_exist = any(
        target.get("relevance") == "direct"
        and target.get("evidence_scope") == "parent_expand_required"
        for item in payload["items"] for target in item["relevant_targets"]
    )
    parent_collection = None
    production_collection = ""
    if production_values:
        if len(production_values) != 1:
            raise ValueError(
                f"experiment children reference multiple production collections: {sorted(production_values)}"
            )
        if any(not str(meta.get("production_collection") or "").strip() for meta in child_metas):
            raise ValueError("experiment collection mixes children with and without production_collection")
        if client is None:
            raise ValueError("experiment qrels validation requires a Chroma client for parent loading")
        production_collection = next(iter(production_values))
        try:
            parent_collection = client.get_collection(production_collection)
        except Exception as exc:
            raise ValueError(
                f"cannot load reviewed production parent collection {production_collection!r}"
            ) from exc
    elif parent_targets_exist:
        raise ValueError(
            "parent_expand_required qrels need child metadata.production_collection"
        )

    # For an experiment collection this verifies the immutable parent hash,
    # source and full evidence in addition to child lineage.  C0 overlays have
    # no production_collection metadata and retain the historical child-only
    # validation path.
    validate_qrels_against_collection(
        payload,
        collection,
        parent_collection=parent_collection,
    )

    strict: dict[str, set[str]] = {}
    origins: dict[str, set[str]] = {}
    for item in payload["items"]:
        query_id = str(item["query_id"])
        strict[query_id] = {
            str(target["chunk_id"])
            for target in item["relevant_targets"]
            if target["relevance"] == "direct"
            and target.get("evidence_scope", "child") == "child"
        }
        origins[query_id] = {
            str(target["origin_chunk_id"])
            for target in item["relevant_targets"]
            if target["relevance"] == "direct"
            and target.get("evidence_scope") == "parent_expand_required"
        }

    children_by_origin: dict[str, set[str]] = {}
    for child_id, meta in zip(child_ids, child_metas):
        origin = str(meta.get("origin_chunk_id") or "").strip()
        if origin:
            children_by_origin.setdefault(origin, set()).add(child_id)
    parent_triggers = {
        query_id: set().union(*[
            children_by_origin.get(origin, set()) for origin in query_origins
        ]) if query_origins else set()
        for query_id, query_origins in origins.items()
    }
    missing_origins = sorted({
        origin
        for query_origins in origins.values() for origin in query_origins
        if not children_by_origin.get(origin)
    })
    if missing_origins:
        raise ValueError(
            f"reviewed parents have no triggering experiment children: {missing_origins}"
        )
    return ScopedDirectQrels(
        strict_child_ids=strict,
        parent_origin_ids=origins,
        parent_trigger_child_ids=parent_triggers,
        production_collection=production_collection,
    ), payload


def _load_qrels(
    path: Path,
    query_ids: list[str],
    collection,
    *,
    client=None,
) -> tuple[dict[str, set[str]], dict]:
    """Backward-compatible combined trigger view used by C0 callers."""

    scoped, payload = _load_scoped_qrels(
        path, query_ids, collection, client=client,
    )
    combined = {
        query_id: scoped.combined_trigger_ids(query_id)
        for query_id in query_ids
    }
    return combined, payload


def _a0_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for set_result in payload.get("sets", []):
        if set_result.get("set") == "heldout52" and set_result.get("runs"):
            return {
                row["id"]: row
                for row in set_result["runs"][0].get("rows", [])
            }
    raise ValueError(f"A0 artifact has no heldout52 rows: {path}")


def _reference_where() -> dict[str, Any]:
    """Mirror ``retrieve_with_trace(reference_kb)`` metadata policy."""

    from rag_source_policy import PERSONAL_VECTOR_SOURCE_TYPES, INTERNAL_SOURCE_TYPES

    return {
        "$and": [
            {
                "source_type": {
                    "$nin": sorted(PERSONAL_VECTOR_SOURCE_TYPES | INTERNAL_SOURCE_TYPES)
                }
            },
            {"owner_scope": {"$in": ["curated"]}},
        ]
    }


def _allowed_reference_meta(meta: dict | None) -> bool:
    from rag_source_policy import PERSONAL_VECTOR_SOURCE_TYPES, INTERNAL_SOURCE_TYPES

    meta = meta or {}
    return (
        str(meta.get("source_type") or "")
        not in (PERSONAL_VECTOR_SOURCE_TYPES | INTERNAL_SOURCE_TYPES)
        and str(meta.get("owner_scope") or "") == "curated"
    )


def read_candidate_channels(
    items: list[dict[str, Any]],
    collection,
    *,
    channel_depth: int = 20,
    query_embeddings: list[list[float]] | None = None,
) -> dict[str, tuple[list, list]]:
    """Read production-equivalent dense/BM25 channels, with no reranker.

    The dense model receives all questions in one batch.  Production requests
    ``4 * depth`` BM25 rows before applying metadata filters, so this evaluator
    does the same and truncates the allowed results back to ``channel_depth``.
    """

    from rag_bm25 import bm25_search
    from rag_candidate_pool import ChannelCandidate
    from rag_retrieval_trace import stable_chunk_id
    from rag_tools import get_embeddings_batch

    questions = [str(item["q"]) for item in items]
    embeddings = query_embeddings
    if embeddings is None:
        embeddings = get_embeddings_batch(questions)
    if len(embeddings) != len(items):
        raise ValueError("query_embeddings must align with candidate items")
    dense_result = collection.query(
        query_embeddings=embeddings,
        n_results=channel_depth,
        where=_reference_where(),
        include=["documents", "metadatas", "distances"],
    )
    ids_rows = dense_result.get("ids") or [[] for _ in items]
    docs_rows = dense_result.get("documents") or [[] for _ in items]
    metas_rows = dense_result.get("metadatas") or [[] for _ in items]
    dists_rows = dense_result.get("distances") or [[] for _ in items]

    output = {}
    for index, item in enumerate(items):
        dense = [
            ChannelCandidate(
                chunk_id=str(chunk_id),
                source=str((meta or {}).get("source") or ""),
                rank=rank,
                distance=float(distance),
            )
            for rank, (chunk_id, _document, meta, distance) in enumerate(
                zip(ids_rows[index], docs_rows[index], metas_rows[index], dists_rows[index]),
                start=1,
            )
        ]
        raw_bm25 = bm25_search(str(item["q"]), channel_depth * 4)
        allowed_bm25 = [hit for hit in raw_bm25 if _allowed_reference_meta(hit[1])][
            :channel_depth
        ]
        bm25 = [
            ChannelCandidate(
                chunk_id=stable_chunk_id(document, meta),
                source=str((meta or {}).get("source") or ""),
                rank=rank,
                score=float(score),
            )
            for rank, (document, meta, score) in enumerate(allowed_bm25, start=1)
        ]
        output[str(item["id"])] = (dense, bm25)
    return output


def _source_hit(pool, expected_sources: list[str]) -> bool:
    expected = [str(source).casefold() for source in expected_sources]
    return any(
        any(label in candidate.source.casefold() for label in expected)
        for candidate in pool
    )


def _scoped_candidate_metrics(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    baseline_hits: dict[str, bool],
) -> dict[str, Any]:
    contracts = {
        "strict_child": ("strict_child_target_ids", "strict_child_hit"),
        "parent_expandable": ("parent_origin_ids", "parent_expandable_hit"),
        "combined": ("combined_evaluable", "combined_hit"),
    }
    target_key, hit_key = contracts[scope]
    covered = [row for row in rows if row.get(target_key)]
    hits = {str(row["id"]): bool(row.get(hit_key)) for row in covered}
    rescued = sorted(
        query_id for query_id, hit in hits.items()
        if hit and not baseline_hits.get(query_id, False)
    )
    lost = sorted(
        query_id for query_id, hit in hits.items()
        if not hit and baseline_hits.get(query_id, False)
    )
    hit_count = sum(hits.values())
    output = {
        "n": len(covered),
        "hits": hit_count,
        "candidate_recall": round(hit_count / len(covered), 6) if covered else None,
        "delta_hits_vs_baseline": hit_count - sum(
            bool(baseline_hits.get(str(row["id"]), False)) for row in covered
        ),
        "rescued_query_ids": rescued,
        "lost_query_ids": lost,
    }
    if scope == "parent_expandable":
        output.update({
            "parent_expansion_applied": False,
            "claim_scope": (
                "candidate_can_trigger_parent_expansion_not_strict_child_relevance"
            ),
        })
    return output


def _file_level_candidate_report(
    strategy: str,
    items: list[dict[str, Any]],
    channels: dict[str, tuple[list, list]],
    baseline_hits: dict[str, bool],
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    """Report immutable historical source-label Candidate@20 separately.

    This metric intentionally includes unsupported chunk-qrels questions: it
    reproduces the original 52-question file contract and must never be mixed
    with strict/parent relevance.  In ``baseline-misses`` mode it describes
    only that explicitly selected subset and therefore cannot claim 52-set
    coverage.
    """

    from rag_candidate_pool import select_candidate_pool, source_distribution

    rows = []
    for item in items:
        query_id = str(item["id"])
        dense, bm25 = channels[query_id]
        pool = select_candidate_pool(
            dense,
            bm25,
            strategy=strategy,
            pool_size=20,
            max_per_source=4,
            exclusive_per_channel=4,
        )
        hit = _source_hit(pool, item.get("expect_sources", []))
        rows.append({
            "id": query_id,
            "expect_sources": list(item.get("expect_sources", [])),
            "candidate_hit": hit,
            "pool_size": len(pool),
            "pool_chunk_ids": [candidate.chunk_id for candidate in pool],
            "source_distribution": source_distribution(pool),
        })
    hits = {row["id"]: bool(row["candidate_hit"]) for row in rows}
    hit_count = sum(hits.values())
    rescued = sorted(
        query_id for query_id, hit in hits.items()
        if hit and not baseline_hits.get(query_id, False)
    )
    lost = sorted(
        query_id for query_id, hit in hits.items()
        if not hit and baseline_hits.get(query_id, False)
    )
    return {
        "n": len(rows),
        "hits": hit_count,
        "candidate_recall": round(hit_count / len(rows), 6) if rows else None,
        "delta_hits_vs_baseline": hit_count - sum(
            bool(baseline_hits.get(row["id"], False)) for row in rows
        ),
        "rescued_query_ids": rescued,
        "lost_query_ids": lost,
        "evaluation_scope": evaluation_scope,
        "full_immutable_set": evaluation_scope == "all" and len(rows) == 52,
        "claim_scope": "historical_file_source_label_only_not_chunk_relevance_or_final_r1",
        "rows": rows,
    }


def _strategy_report(
    strategy: str,
    items: list[dict[str, Any]],
    channels: dict[str, tuple[list, list]],
    scoped_qrels: ScopedDirectQrels,
    baseline_scope_hits: dict[str, dict[str, bool]],
    *,
    file_level_items: list[dict[str, Any]] | None = None,
    baseline_file_hits: dict[str, bool] | None = None,
    evaluation_scope: str = "all",
) -> dict[str, Any]:
    from rag_candidate_pool import select_candidate_pool, source_distribution

    rows = []
    aggregate_sources: Counter[str] = Counter()
    for item in items:
        query_id = str(item["id"])
        strict_ids = set(scoped_qrels.strict_child_ids.get(query_id, set()))
        parent_origins = set(scoped_qrels.parent_origin_ids.get(query_id, set()))
        parent_trigger_ids = set(
            scoped_qrels.parent_trigger_child_ids.get(query_id, set())
        )
        if not strict_ids and not parent_origins:
            continue
        dense, bm25 = channels[query_id]
        pool = select_candidate_pool(
            dense,
            bm25,
            strategy=strategy,
            pool_size=20,
            max_per_source=4,
            exclusive_per_channel=4,
        )
        pool_ids = [candidate.chunk_id for candidate in pool]
        pool_id_set = set(pool_ids)
        strict_hit = bool(pool_id_set & strict_ids)
        parent_expandable_hit = bool(pool_id_set & parent_trigger_ids)
        combined_hit = strict_hit or parent_expandable_hit
        distribution = source_distribution(pool)
        aggregate_sources.update(distribution)
        rows.append({
            "id": query_id,
            # Backward-compatible name remains strict-only. Parent trigger
            # children are deliberately not promoted to direct child gold.
            "direct_target_chunk_ids": sorted(strict_ids),
            "direct_candidate_hit": strict_hit,
            "strict_child_target_ids": sorted(strict_ids),
            "strict_child_hit": strict_hit,
            "parent_origin_ids": sorted(parent_origins),
            "parent_trigger_child_ids": sorted(parent_trigger_ids),
            "parent_expandable_hit": parent_expandable_hit,
            "parent_expansion_applied": False,
            "combined_evaluable": True,
            "combined_hit": combined_hit,
            "source_label_candidate_hit": _source_hit(pool, item.get("expect_sources", [])),
            "pool_size": len(pool),
            "pool_chunk_ids": pool_ids,
            "source_distribution": distribution,
            "channel_class_distribution": dict(sorted(Counter(
                candidate.channel_class for candidate in pool
            ).items())),
        })

    strict_metrics = _scoped_candidate_metrics(
        rows,
        scope="strict_child",
        baseline_hits=baseline_scope_hits.get("strict_child", {}),
    )
    parent_metrics = _scoped_candidate_metrics(
        rows,
        scope="parent_expandable",
        baseline_hits=baseline_scope_hits.get("parent_expandable", {}),
    )
    combined_metrics = _scoped_candidate_metrics(
        rows,
        scope="combined",
        baseline_hits=baseline_scope_hits.get("combined", {}),
    )
    file_level = _file_level_candidate_report(
        strategy,
        list(file_level_items if file_level_items is not None else items),
        channels,
        baseline_file_hits or {},
        evaluation_scope=evaluation_scope,
    )
    return {
        "strategy": strategy,
        "configuration": {
            "pool_size": 20,
            "rrf_k": 60,
            "max_per_source": 4 if strategy == "source_cap4" else None,
            "exclusive_per_channel": (
                4 if strategy == "retain_channel_exclusives" else None
            ),
        },
        "qrels_scopes": {
            "strict_child": strict_metrics,
            "parent_expandable": parent_metrics,
            "combined": combined_metrics,
        },
        # C0/report consumers used this key before evidence scopes existed.
        # It now aliases combined candidate-or-parent-trigger coverage and says
        # so explicitly; strict child metrics live above.
        "direct_qrels": {
            **combined_metrics,
            "scope": "combined_strict_or_parent_expandable",
        },
        "file_level_candidate_recall": file_level,
        # Compatibility summary; detailed per-query rows live in the explicit
        # file-level block above.
        "historical_source_label_diagnostic": {
            key: value for key, value in file_level.items() if key != "rows"
        },
        "aggregate_pool_sources": dict(sorted(aggregate_sources.items())),
        "pool_size": {
            "min": min((row["pool_size"] for row in rows), default=0),
            "max": max((row["pool_size"] for row in rows), default=0),
            "mean": (
                round(sum(row["pool_size"] for row in rows) / len(rows), 3)
                if rows else 0.0
            ),
        },
        "eligible_for_reranker_evaluation": (
            strategy != "baseline_rrf20"
            and int(combined_metrics["delta_hits_vs_baseline"]) > 0
        ),
        "rows": rows,
    }


def _select_scope_items(
    items: list[dict[str, Any]],
    scoped_qrels: ScopedDirectQrels,
    a0_rows: dict[str, dict[str, Any]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(qrels_items, file_level_items)`` for one CLI scope.

    The historical ``baseline-misses`` contract remains qrels-supported direct
    candidate misses only.  It does not silently expand to unsupported items
    merely because the full-file diagnostic was added.
    """

    supported = [
        item for item in items
        if scoped_qrels.strict_child_ids.get(str(item["id"]))
        or scoped_qrels.parent_origin_ids.get(str(item["id"]))
    ]
    if scope == "all":
        return supported, list(items)
    if scope != "baseline-misses":
        raise ValueError(f"unknown evaluation scope: {scope}")
    if not a0_rows:
        raise ValueError("--scope baseline-misses requires --a0")
    misses = [
        item for item in supported
        if not a0_rows.get(str(item["id"]), {}).get("direct_candidate_rank")
    ]
    return misses, list(misses)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb
    from rag_candidate_pool import select_candidate_pool
    from rag_tools import get_collection_name, index_fingerprint

    items = _load_items(args.set)
    query_ids = [str(item["id"]) for item in items]
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection(get_collection_name())
    scoped_qrels, qrels_payload = _load_scoped_qrels(
        args.qrels, query_ids, collection, client=client,
    )
    a0_rows = _a0_rows(args.a0)

    supported_items, file_level_items = _select_scope_items(
        items, scoped_qrels, a0_rows, scope=args.scope,
    )
    # ``all`` reads all 52 exactly once.  qrels scoring below remains restricted
    # to the 50 supported questions, while file-level scoring includes all 52.
    channel_ids = {
        str(item["id"]) for item in supported_items + file_level_items
    }
    channel_items = [item for item in items if str(item["id"]) in channel_ids]
    channels = read_candidate_channels(channel_items, collection, channel_depth=20)
    baseline_scope_hits: dict[str, dict[str, bool]] = {
        "strict_child": {},
        "parent_expandable": {},
        "combined": {},
    }
    baseline_file_hits: dict[str, bool] = {}
    supported_ids = {str(item["id"]) for item in supported_items}
    file_level_ids = {str(item["id"]) for item in file_level_items}
    reproduction_rows = []
    for item in channel_items:
        query_id = str(item["id"])
        dense, bm25 = channels[query_id]
        pool = select_candidate_pool(
            dense, bm25, strategy="baseline_rrf20", pool_size=20,
        )
        pool_ids = [candidate.chunk_id for candidate in pool]
        pool_id_set = set(pool_ids)
        if query_id in supported_ids:
            strict_hit = bool(
                pool_id_set & scoped_qrels.strict_child_ids.get(query_id, set())
            )
            parent_hit = bool(
                pool_id_set & scoped_qrels.parent_trigger_child_ids.get(query_id, set())
            )
            baseline_scope_hits["strict_child"][query_id] = strict_hit
            baseline_scope_hits["parent_expandable"][query_id] = parent_hit
            baseline_scope_hits["combined"][query_id] = strict_hit or parent_hit
        if query_id in file_level_ids:
            baseline_file_hits[query_id] = _source_hit(
                pool, item.get("expect_sources", []),
            )
        expected_ids = list(a0_rows.get(query_id, {}).get("fusion_chunk_ids") or [])
        reproduction_rows.append({
            "id": query_id,
            "exact_chunk_id_order_match": (pool_ids == expected_ids) if expected_ids else None,
            "candidate_only_chunk_ids": pool_ids,
            "a0_fusion_chunk_ids": expected_ids,
        })

    reports = [
        _strategy_report(
            strategy,
            supported_items,
            channels,
            scoped_qrels,
            baseline_scope_hits,
            file_level_items=file_level_items,
            baseline_file_hits=baseline_file_hits,
            evaluation_scope=args.scope,
        )
        for strategy in STRATEGIES
    ]
    exact_checks = [
        row["exact_chunk_id_order_match"] for row in reproduction_rows
        if row["exact_chunk_id_order_match"] is not None
    ]
    return {
        "schema_version": "candidate-pool-stage-c-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_scope": args.scope,
        "claims": {
            "eligible_metric": "combined_strict_or_parent_expandable_candidate_recall",
            "strict_child_metric": "reviewer_approved_child_ids_only",
            "parent_expandable_metric": "trigger_only_expansion_not_applied",
            "source_level_candidate_recall": "diagnostic_only",
            "final_production_r1": "not_evaluated_requires_cross_encoder",
            "production_default_changed": False,
        },
        "set": str(args.set),
        "qrels": {
            "path": str(args.qrels),
            "reviewer_id": qrels_payload["reviewer_id"],
            "supported_queries": len(supported_items),
            "file_level_queries": len(file_level_items),
            "strict_child_queries": sum(
                bool(scoped_qrels.strict_child_ids.get(str(item["id"])))
                for item in supported_items
            ),
            "parent_expand_required_queries": sum(
                bool(scoped_qrels.parent_origin_ids.get(str(item["id"])))
                for item in supported_items
            ),
            "production_collection": scoped_qrels.production_collection,
            "parent_expansion_applied": False,
        },
        "index": index_fingerprint(collection=collection, cache_ttl=0),
        "candidate_reader": {
            "dense_depth": 20,
            "bm25_prefilter_depth": 80,
            "bm25_postfilter_depth": 20,
            "reference_kb_metadata_policy": _reference_where(),
            "cross_encoder_calls": 0,
        },
        "baseline_reproduction_against_a0": {
            "artifact": str(args.a0) if args.a0 else "",
            "checked": len(exact_checks),
            "exact_matches": sum(value is True for value in exact_checks),
            "all_exact": bool(exact_checks) and all(exact_checks),
            "mismatch_query_ids": [
                row["id"] for row in reproduction_rows
                if row["exact_chunk_id_order_match"] is False
            ],
            "rows": reproduction_rows,
        },
        "strategies": reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--a0", type=Path)
    parser.add_argument("--scope", choices=("all", "baseline-misses"), default="all")
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
