# -*- coding: utf-8 -*-
"""Production-equivalent, process-isolated reranker A/B evaluation.

Each arm runs in a fresh child process.  This is intentional: on an 8 GB Mac,
keeping both bge-reranker-base and bge-reranker-v2-m3 in the module cache can
exhaust unified memory and also makes it too easy to accidentally reuse the
wrong model.  The parent launches children strictly serially and only compares
their JSON results after each process exits.

Examples::

    # Fast decision path: A0 versus the low-cost base+breadcrumb B0 control.
    .venv/bin/python eval_reranker_profiles.py --profiles auto --sets heldout52

    # Full release evidence after a candidate wins the 52-question set.
    .venv/bin/python eval_reranker_profiles.py \
        --profiles A0,B0 --sets heldout52,bench100,final90 --repeat 3 --gate
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
SET_ALIASES = {
    "heldout52": ROOT / "tests" / "rag_bench_paraphrase_set.json",
    "bench100": ROOT / "tests" / "rag_bench_set.json",
    "final90": ROOT / "tests" / "zh_final_set.json",
}
REFERENCE_PLAN = {
    "decision": "answer",
    "routes": [{"source": "reference_kb", "operation": "search"}],
}
KNOWN_EARLY_REJECTIONS = {
    "A1": {
        "status": "early_reject_latency",
        "profile": "bge-reranker-v2-m3, pool20, body-only",
        "current_index_count": 3348,
        "observations": {
            "mps": {"queries_observed": 6, "latency_range_ms": [60000, 90000]},
            "cpu": {"queries_observed": 1, "latency_ms": 40479.476},
        },
        "historical_quality_reference": {
            "index_count": 3346,
            "r1_hits": 24,
            "n": 52,
            "advisory_only": True,
        },
        "reason": "observed latency is already far above A0 p95 +10%; full arm stopped",
    },
}


def arm_profile(arm: str):
    """Return a fully explicit profile; no experimental env switch can leak in."""

    from rag_retrieval_trace import RetrievalProfile
    from rag_tools import CHUNKER_VERSION

    specs = {
        "A0": (20, "BAAI/bge-reranker-base", False),
        "A1": (20, "BAAI/bge-reranker-v2-m3", False),
        "A2": (10, "BAAI/bge-reranker-v2-m3", False),
        "A3": (20, "BAAI/bge-reranker-v2-m3", True),
        "B0": (20, "BAAI/bge-reranker-base", True, "baseline_rrf20"),
        "C1": (20, "BAAI/bge-reranker-base", False, "source_cap4"),
        "C2": (20, "BAAI/bge-reranker-base", True, "source_cap4"),
        # 2026-08-28 recall arms.  D0 is the answerability-judge candidate pool
        # (28) with today's reranker; D1 adds HyDE, which on Dev-New recovered
        # 6 of the 10 golds no channel reaches (R@1 64->68).  Both exist so the
        # held-out 52 can test the *same* configuration Dev-New was tuned on —
        # a confirmation run against a different pool size proves nothing.
        "D0": (28, "BAAI/bge-reranker-base", False),
        "D1": (28, "BAAI/bge-reranker-base", False),
        # D2 runs HyDE as a second dense channel instead of a replacement query,
        # so the evidence gate keeps seeing question->document distances.
        "D2": (28, "BAAI/bge-reranker-base", False),
        # D3 = D2 + a lexical channel over the same HyDE text.
        "D3": (28, "BAAI/bge-reranker-base", False),
    }
    hyde_arms = {"D1"}
    hyde_channel_arms = {"D2", "D3"}
    hyde_bm25_channel_arms = {"D3"}
    if arm in {"H1", "H2"}:
        model_path = Path(os.environ.get("RAG_HARD_NEGATIVE_MODEL", "")).expanduser()
        if not str(model_path) or not model_path.is_dir():
            raise ValueError(f"{arm} requires an existing RAG_HARD_NEGATIVE_MODEL directory")
        if not (model_path / "DEVELOPMENT_ONLY").exists():
            raise ValueError(f"{arm} evaluator only accepts an audited development model")
        specs[arm] = (20, str(model_path.resolve()), True, "baseline_rrf20")
    if arm not in specs:
        raise ValueError(f"unknown arm: {arm}")
    # The first four arms predate Stage C and retain the production pool.
    if len(specs[arm]) == 3:
        pool, model, breadcrumb = specs[arm]
        pool_strategy = "baseline_rrf20"
    else:
        pool, model, breadcrumb, pool_strategy = specs[arm]
    effective_chunker_version = (
        os.environ.get("RAG_EVAL_CHUNKER_VERSION", "").strip() or CHUNKER_VERSION
    )
    return RetrievalProfile(
        name=arm,
        pool_size=pool,
        reranker_model=model,
        reranker_use_breadcrumb=breadcrumb,
        candidate_pool_strategy=pool_strategy,
        chunker_version=effective_chunker_version,
        enable_hyde=arm in hyde_arms,
        enable_hyde_channel=arm in hyde_channel_arms,
        enable_hyde_bm25_channel=arm in hyde_bm25_channel_arms,
        enable_query_rewrite=False,
        enable_doc2query=False,
        enable_quota=False,
        enable_alias=False,
        enable_rerank_bridge=False,
    )


def _set_path(name_or_path: str) -> Path:
    path = SET_ALIASES.get(name_or_path, Path(name_or_path))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_items(name_or_path: str) -> tuple[str, list[dict[str, Any]]]:
    path = _set_path(name_or_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} has no evaluation items")
    for item in items:
        if not item.get("id") or not item.get("q") or not item.get("expect_sources"):
            raise ValueError(f"{path} contains an invalid retrieval row")
    return (name_or_path if name_or_path in SET_ALIASES else str(path), items)


def _hit_rank(sources: list[str], expected: list[str]) -> int:
    expected_lower = [str(value).lower() for value in expected]
    for index, source in enumerate(sources, start=1):
        source_lower = str(source or "").lower()
        if any(value in source_lower for value in expected_lower):
            return index
    return 0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ranks = [int(row["rank"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    candidate_hits = sum(bool(row["candidate_rank"]) for row in rows)
    effective_hits = sum(bool(row["effective_rank1"]) for row in rows)
    return {
        "n": n,
        "r1_hits": sum(rank == 1 for rank in ranks),
        "recall@1": round(sum(rank == 1 for rank in ranks) / n, 6),
        "recall@3": round(sum(0 < rank <= 3 for rank in ranks) / n, 6),
        "recall@5": round(sum(0 < rank <= 5 for rank in ranks) / n, 6),
        "mrr": round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / n, 6),
        "candidate_hits": candidate_hits,
        "candidate_recall": round(candidate_hits / n, 6),
        "effective_r1_hits": effective_hits,
        "effective_recall@1": round(effective_hits / n, 6),
        "p50_ms": round(_percentile(latencies, 0.50), 3),
        "p95_ms": round(_percentile(latencies, 0.95), 3),
        "mean_ms": round(statistics.fmean(latencies), 3),
    }


def _direct_qrels_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score reviewer-approved direct chunks; supporting chunks never count."""

    covered = [row for row in rows if row.get("direct_target_chunk_ids")]
    if not covered:
        return {"n": 0, "coverage": 0}
    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        ranks = [int(row.get("direct_rank") or 0) for row in subset]
        candidate_hits = sum(bool(row.get("direct_candidate_rank")) for row in subset)
        effective_hits = sum(bool(row.get("direct_effective_rank1")) for row in subset)
        n = len(subset)
        return {
            "n": n,
            "coverage": n,
            "r1_hits": sum(rank == 1 for rank in ranks),
            "recall@1": round(sum(rank == 1 for rank in ranks) / n, 6),
            "recall@3": round(sum(0 < rank <= 3 for rank in ranks) / n, 6),
            "recall@5": round(sum(0 < rank <= 5 for rank in ranks) / n, 6),
            "mrr": round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / n, 6),
            "candidate_hits": candidate_hits,
            "candidate_recall": round(candidate_hits / n, 6),
            "effective_r1_hits": effective_hits,
            "effective_recall@1": round(effective_hits / n, 6),
        }

    result = aggregate(covered)
    result["by_review_outcome"] = {
        outcome: aggregate([
            row for row in covered if row.get("direct_qrels_outcome") == outcome
        ])
        for outcome in ("accepted", "partial")
        if any(row.get("direct_qrels_outcome") == outcome for row in covered)
    }
    return result


def _load_direct_qrels(
    path: str | Path | None,
    expected_query_ids: list[str],
    *,
    collection=None,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, Any]]:
    if not path:
        return {}, {}, {}
    from rag_qrels import load_qrels_overlay, validate_qrels_against_collection

    payload = load_qrels_overlay(path, expected_query_ids=expected_query_ids)
    if collection is None:
        import chromadb
        from rag_tools import get_collection_name

        collection = chromadb.PersistentClient(
            path=str(ROOT / "chroma_db")
        ).get_collection(get_collection_name())
    validate_qrels_against_collection(payload, collection)
    if payload["index"]["count"] != collection.count():
        raise ValueError(
            "qrels index count mismatch: "
            f"overlay={payload['index']['count']} live={collection.count()}"
        )
    direct: dict[str, set[str]] = {}
    outcomes: dict[str, str] = {}
    for item in payload["items"]:
        outcomes[item["query_id"]] = item["review_outcome"]
        direct[item["query_id"]] = {
            target["chunk_id"] for target in item["relevant_targets"]
            if target["relevance"] == "direct"
            and target.get("evidence_scope", "child") == "child"
        }
    parent_expand_required = {
        item["query_id"]: [
            target["origin_chunk_id"] for target in item["relevant_targets"]
            if target["relevance"] == "direct"
            and target.get("evidence_scope") == "parent_expand_required"
        ]
        for item in payload["items"]
    }
    parent_expand_required = {
        query_id: origins for query_id, origins in parent_expand_required.items() if origins
    }
    return direct, outcomes, {
        "schema_version": payload["schema_version"],
        "reviewer_id": payload["reviewer_id"],
        "base_set": payload["base_set"],
        "path": str(Path(path)),
        "validation": "schema_and_live_collection_validated",
        "index": dict(payload["index"]),
        "strict_child_direct_only": True,
        "parent_expand_required_query_ids": sorted(parent_expand_required),
        "parent_expand_required_origins": parent_expand_required,
    }


def _evaluate_set(
    set_name: str,
    items: list[dict[str, Any]],
    profile,
    repeat: int,
    expected_index_count: int,
    direct_qrels: dict[str, set[str]] | None = None,
    qrels_outcomes: dict[str, str] | None = None,
    checkpoint_callback=None,
    early_reject_ms: float | None = None,
) -> dict[str, Any]:
    from rag_gate import retrieve_with_trace

    runs = []
    for run_index in range(repeat):
        rows = []
        for item_index, item in enumerate(items, start=1):
            print(
                f"[{profile.name}] {set_name} run {run_index + 1}/{repeat} "
                f"{item_index}/{len(items)} {item['id']}",
                file=sys.stderr,
                flush=True,
            )
            trace = retrieve_with_trace(
                item["q"], REFERENCE_PLAN, profile, top_k=5,
            )
            actual_count = trace.index_metadata.get("collection_count")
            if actual_count != expected_index_count:
                raise RuntimeError(
                    f"index changed during evaluation: expected {expected_index_count}, "
                    f"got {actual_count}"
                )
            _assert_reranker_complete(trace)
            final_sources = [candidate.source for candidate in trace.final_candidates]
            fusion_sources = [candidate.source for candidate in trace.fusion_candidates]
            rank = _hit_rank(final_sources, item["expect_sources"])
            candidate_rank = _hit_rank(fusion_sources, item["expect_sources"])
            direct_ids = set((direct_qrels or {}).get(item["id"], set()))
            final_chunk_ids = [candidate.chunk_id for candidate in trace.final_candidates]
            fusion_chunk_ids = [candidate.chunk_id for candidate in trace.fusion_candidates]
            dense_candidates = list(getattr(trace, "dense_candidates", []))
            bm25_candidates = list(getattr(trace, "bm25_candidates", []))
            dense_chunk_ids = [candidate.chunk_id for candidate in dense_candidates]
            bm25_chunk_ids = [candidate.chunk_id for candidate in bm25_candidates]
            reranked_chunk_ids = [
                candidate.chunk_id for candidate in trace.reranked_candidates
            ]
            reranked_scores = [
                candidate.rerank_score for candidate in trace.reranked_candidates
            ]
            direct_rank = next(
                (index for index, chunk_id in enumerate(final_chunk_ids, start=1)
                 if chunk_id in direct_ids),
                0,
            )
            direct_candidate_rank = next(
                (index for index, chunk_id in enumerate(fusion_chunk_ids, start=1)
                 if chunk_id in direct_ids),
                0,
            )
            rows.append({
                "id": item["id"],
                "domain": item.get("domain", ""),
                "hard": bool(item.get("hard")),
                "question": item["q"],
                "expect_sources": list(item["expect_sources"]),
                "rank": rank,
                "candidate_rank": candidate_rank,
                "gate_decision": bool(trace.gate_decision),
                "effective_rank1": bool(rank == 1 and trace.gate_decision),
                "direct_target_chunk_ids": sorted(direct_ids),
                "direct_qrels_outcome": (qrels_outcomes or {}).get(item["id"], ""),
                "direct_rank": direct_rank,
                "direct_candidate_rank": direct_candidate_rank,
                "direct_effective_rank1": bool(direct_rank == 1 and trace.gate_decision),
                "top1_source": final_sources[0] if final_sources else "",
                "final_sources": final_sources,
                "final_chunk_ids": final_chunk_ids,
                "dense_candidates": [
                    candidate.to_dict() if hasattr(candidate, "to_dict") else {
                        "chunk_id": candidate.chunk_id,
                        "source": candidate.source,
                    }
                    for candidate in dense_candidates
                ],
                "bm25_candidates": [
                    candidate.to_dict() if hasattr(candidate, "to_dict") else {
                        "chunk_id": candidate.chunk_id,
                        "source": candidate.source,
                    }
                    for candidate in bm25_candidates
                ],
                "fusion_candidates": [
                    candidate.to_dict() if hasattr(candidate, "to_dict") else {
                        "chunk_id": candidate.chunk_id,
                        "source": candidate.source,
                    }
                    for candidate in trace.fusion_candidates
                ],
                "dense_chunk_ids": dense_chunk_ids,
                "bm25_chunk_ids": bm25_chunk_ids,
                "fusion_chunk_ids": fusion_chunk_ids,
                "reranked_candidate_ids": reranked_chunk_ids,
                "reranked_scores": reranked_scores,
                "gate_candidate_chunk_id": trace.gate_candidate_chunk_id,
                "final_top1_chunk_id": trace.final_top1_chunk_id,
                "gate_alignment": trace.gate_alignment,
                "reranker_requested": trace.reranker_requested,
                "reranker_actual": trace.reranker_actual,
                "reranker_status": trace.reranker_status,
                "reranker_scored_count": trace.reranker_scored_count,
                "rerank_margin": trace.gate_features.get("rerank_margin"),
                "rerank_top": trace.gate_features.get("rerank_top"),
                "gate_features": dict(trace.gate_features),
                "latency_ms": float(trace.latency_by_stage.get("total", 0.0)),
                "latency_by_stage": dict(trace.latency_by_stage),
                "retrieval_profile": (
                    trace.retrieval_profile.to_dict()
                    if hasattr(getattr(trace, "retrieval_profile", None), "to_dict")
                    else {}
                ),
                "index_fingerprint": getattr(trace, "index_fingerprint", ""),
            })
            current_run = {
                "run": run_index + 1,
                "metrics": _metrics(rows),
                "direct_qrels_metrics": _direct_qrels_metrics(rows),
                "rows": rows,
            }
            if checkpoint_callback:
                checkpoint_callback({
                    "set": set_name,
                    "status": "running",
                    "runs": runs + [current_run],
                    "top1_stability": None,
                })
            if early_reject_ms and rows[-1]["latency_ms"] > early_reject_ms:
                current_run["early_reject"] = {
                    "reason": "latency_budget_exceeded",
                    "threshold_ms": early_reject_ms,
                    "observed_ms": rows[-1]["latency_ms"],
                    "at_query_id": item["id"],
                }
                runs.append(current_run)
                return {
                    "set": set_name,
                    "status": "early_reject_latency",
                    "runs": runs,
                    "top1_stability": None,
                }
        runs.append({
            "run": run_index + 1,
            "metrics": _metrics(rows),
            "direct_qrels_metrics": _direct_qrels_metrics(rows),
            "rows": rows,
        })
    top1_sources = {
        row["id"]: [] for row in runs[0]["rows"]
    }
    for run in runs:
        for row in run["rows"]:
            top1_sources[row["id"]].append(row["top1_source"])
    stable = sum(len(set(values)) == 1 for values in top1_sources.values())
    return {
        "set": set_name,
        "status": "completed",
        "runs": runs,
        "top1_stability": round(stable / len(top1_sources), 6),
    }


def _evaluate_gates(
    profile,
    expected_index_count: int,
    *,
    checkpoint_callback=None,
) -> dict[str, Any]:
    from rag_gate import retrieve_with_trace

    bench = json.loads(SET_ALIASES["bench100"].read_text(encoding="utf-8"))
    adversarial = json.loads(
        (ROOT / "tests" / "rag_gate_adversarial_negatives.json").read_text(encoding="utf-8")
    )
    groups = {
        "simple_negative": [(f"simple-{i + 1}", q, False)
                            for i, q in enumerate(bench["gate_negatives"])],
        "adversarial_negative": [(item["id"], item["q"], False)
                                 for item in adversarial],
        "positive": [(f"positive-{i + 1}", q, True)
                     for i, q in enumerate(bench["gate_positives"])],
    }
    output = {}
    for group_name, items in groups.items():
        rows = []
        for item_id, question, expected in items:
            print(f"[{profile.name}] gate {group_name} {item_id}", file=sys.stderr, flush=True)
            trace = retrieve_with_trace(question, REFERENCE_PLAN, profile, top_k=5)
            if trace.index_metadata.get("collection_count") != expected_index_count:
                raise RuntimeError("index changed during gate evaluation")
            _assert_reranker_complete(trace)
            actual = bool(trace.gate_decision)
            rows.append({
                "id": item_id,
                "expected_in_kb": expected,
                "actual_in_kb": actual,
                "correct": actual == expected,
                "rerank_top": trace.gate_features.get("rerank_top"),
                "rerank_margin": trace.gate_features.get("rerank_margin"),
                "gate_features": dict(trace.gate_features),
                "final_chunk_ids": [
                    candidate.chunk_id for candidate in trace.final_candidates
                ],
                "fusion_chunk_ids": [
                    candidate.chunk_id for candidate in trace.fusion_candidates
                ],
                "reranked_candidate_ids": [
                    candidate.chunk_id for candidate in trace.reranked_candidates
                ],
                "reranked_scores": [
                    candidate.rerank_score for candidate in trace.reranked_candidates
                ],
                "gate_candidate_chunk_id": trace.gate_candidate_chunk_id,
                "final_top1_chunk_id": trace.final_top1_chunk_id,
                "gate_alignment": trace.gate_alignment,
                "reranker_requested": trace.reranker_requested,
                "reranker_actual": trace.reranker_actual,
                "reranker_status": trace.reranker_status,
                "reranker_scored_count": trace.reranker_scored_count,
            })
            if checkpoint_callback:
                checkpoint_callback({
                    **output,
                    group_name: {
                        "correct": sum(row["correct"] for row in rows),
                        "total": len(rows),
                        "expected_total": len(items),
                        "status": "running",
                        "rows": rows,
                    },
                })
        correct = sum(row["correct"] for row in rows)
        output[group_name] = {
            "correct": correct,
            "total": len(rows),
            "score": f"{correct}/{len(rows)}",
            "rows": rows,
        }
    return output


def _assert_reranker_complete(trace) -> None:
    """Fail closed instead of treating a silent reranker fallback as an arm."""

    fusion = list(trace.fusion_candidates)
    reranked = list(trace.reranked_candidates)
    scored = sum(candidate.rerank_score is not None for candidate in reranked)
    status = str(getattr(trace, "reranker_status", "") or "")
    requested = str(getattr(trace, "reranker_requested", "") or "")
    actual = str(getattr(trace, "reranker_actual", "") or "")
    scored_count = int(getattr(trace, "reranker_scored_count", 0) or 0)
    if (
        status != "ok"
        or not requested
        or actual != requested
        or scored_count != len(fusion)
        or len(reranked) != len(fusion)
        or scored != len(reranked)
    ):
        raise RuntimeError(
            "candidate reranker scoring incomplete: "
            f"requested={requested!r} actual={actual!r} status={status!r} "
            f"fusion={len(fusion)} reranked={len(reranked)} "
            f"runtime_scored={scored_count} persisted_scores={scored}"
        )


def _validate_index(expected_count: int) -> dict[str, Any]:
    from rag_tools import index_fingerprint

    fingerprint = index_fingerprint(cache_ttl=0)
    if fingerprint.get("collection_count") != expected_count:
        raise RuntimeError(
            f"refusing to run on a different index: expected {expected_count}, "
            f"got {fingerprint.get('collection_count')}"
        )
    return fingerprint


def run_arm(
    arm: str,
    sets: list[str],
    repeat: int,
    expected_index_count: int,
    include_gates: bool,
    qrels_overlay: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    early_reject_ms: float | None = None,
) -> dict[str, Any]:
    from rag_retrieval_trace import bind_profile_fingerprint

    profile = arm_profile(arm)
    raw_index = _validate_index(expected_index_count)
    results = []
    qrels_meta: dict[str, Any] = {}
    heldout_direct_qrels: dict[str, set[str]] = {}
    heldout_qrels_outcomes: dict[str, str] = {}
    if qrels_overlay and "heldout52" not in sets:
        raise ValueError("--qrels-overlay requires the heldout52 set")
    if qrels_overlay:
        # Fail closed before loading either the embedding model or reranker.
        _heldout_name, heldout_items = _load_items("heldout52")
        heldout_direct_qrels, heldout_qrels_outcomes, qrels_meta = _load_direct_qrels(
            qrels_overlay, [item["id"] for item in heldout_items],
        )

    def persist_checkpoint(
        *,
        status: str,
        sets_snapshot: list[dict[str, Any]],
        gates_snapshot: dict[str, Any],
    ) -> None:
        if not checkpoint_path:
            return
        _write_json(Path(checkpoint_path), {
            "schema_version": "reranker-profile-ab-v1",
            "status": status,
            "arm": arm,
            "profile": asdict(profile),
            "index": bind_profile_fingerprint(raw_index, profile),
            "sets": sets_snapshot,
            "qrels_overlay": qrels_meta,
            "gates": gates_snapshot,
        })

    for set_value in sets:
        set_name, items = _load_items(set_value)
        direct_qrels = {}
        qrels_outcomes = {}
        if qrels_overlay and set_name == "heldout52":
            direct_qrels = heldout_direct_qrels
            qrels_outcomes = heldout_qrels_outcomes

        def save_checkpoint(partial_set):
            persist_checkpoint(
                status="running",
                sets_snapshot=results + [partial_set],
                gates_snapshot={},
            )

        evaluated = _evaluate_set(
            set_name, items, profile, repeat, expected_index_count,
            direct_qrels, qrels_outcomes, save_checkpoint, early_reject_ms,
        )
        results.append(evaluated)
        if evaluated.get("status") == "early_reject_latency":
            early_result = {
                "schema_version": "reranker-profile-ab-v1",
                "status": "early_reject_latency",
                "arm": arm,
                "profile": asdict(profile),
                "index": bind_profile_fingerprint(raw_index, profile),
                "sets": results,
                "qrels_overlay": qrels_meta,
                "gates": {},
            }
            persist_checkpoint(
                status="early_reject_latency",
                sets_snapshot=results,
                gates_snapshot={},
            )
            return early_result
    first_trace_index = None
    if results and results[0]["runs"] and results[0]["runs"][0]["rows"]:
        # The profile-aware fingerprint is identical for every trace in one arm.
        first_trace_index = bind_profile_fingerprint(raw_index, profile)
    gates = {}
    if include_gates:
        gates = _evaluate_gates(
            profile,
            expected_index_count,
            checkpoint_callback=lambda partial: persist_checkpoint(
                status="running",
                sets_snapshot=results,
                gates_snapshot=partial,
            ),
        )
    completed = {
        "schema_version": "reranker-profile-ab-v1",
        "status": "completed",
        "arm": arm,
        "profile": asdict(profile),
        "index": first_trace_index or raw_index,
        "sets": results,
        "qrels_overlay": qrels_meta,
        "gates": gates,
    }
    persist_checkpoint(
        status="completed",
        sets_snapshot=results,
        gates_snapshot=gates,
    )
    return completed


def _first_run(arm_result: dict[str, Any], set_name: str = "heldout52") -> dict[str, Any]:
    selected = next(item for item in arm_result["sets"] if item["set"] == set_name)
    return selected["runs"][0]


def compare_arms(base: dict[str, Any], candidate: dict[str, Any], set_name: str) -> dict[str, Any]:
    base_run = _first_run(base, set_name)
    candidate_run = _first_run(candidate, set_name)
    base_rows = {row["id"]: row for row in base_run["rows"]}
    candidate_rows = {row["id"]: row for row in candidate_run["rows"]}
    wins, losses, changed = [], [], []
    direct_wins, direct_losses = [], []
    effective_wins, effective_losses = [], []
    for item_id in sorted(base_rows):
        before, after = base_rows[item_id], candidate_rows[item_id]
        record = {
            "id": item_id,
            "baseline_rank": before["rank"],
            "candidate_rank": after["rank"],
            "baseline_top1": before["top1_source"],
            "candidate_top1": after["top1_source"],
            "baseline_margin": before["rerank_margin"],
            "candidate_margin": after["rerank_margin"],
        }
        if before["rank"] != after["rank"] or before["top1_source"] != after["top1_source"]:
            changed.append(record)
        if before["rank"] != 1 and after["rank"] == 1:
            wins.append(record)
        elif before["rank"] == 1 and after["rank"] != 1:
            losses.append(record)
        if before.get("direct_target_chunk_ids") and after.get("direct_target_chunk_ids"):
            direct_record = {
                **record,
                "baseline_direct_rank": int(before.get("direct_rank") or 0),
                "candidate_direct_rank": int(after.get("direct_rank") or 0),
            }
            if before.get("direct_rank") != 1 and after.get("direct_rank") == 1:
                direct_wins.append(direct_record)
            elif before.get("direct_rank") == 1 and after.get("direct_rank") != 1:
                direct_losses.append(direct_record)
        if not before.get("effective_rank1") and after.get("effective_rank1"):
            effective_wins.append(record)
        elif before.get("effective_rank1") and not after.get("effective_rank1"):
            effective_losses.append(record)
    base_p95 = float(base_run["metrics"]["p95_ms"])
    candidate_p95 = float(candidate_run["metrics"]["p95_ms"])
    return {
        "set": set_name,
        "baseline_arm": base["arm"],
        "candidate_arm": candidate["arm"],
        "wins": wins,
        "losses": losses,
        "net_wins": len(wins) - len(losses),
        "direct_wins": direct_wins,
        "direct_losses": direct_losses,
        "direct_net_wins": len(direct_wins) - len(direct_losses),
        "effective_wins": effective_wins,
        "effective_losses": effective_losses,
        "effective_net_wins": len(effective_wins) - len(effective_losses),
        "changed": changed,
        "p95_ratio": round(candidate_p95 / base_p95, 6) if base_p95 else None,
    }


def _release_decision(
    candidate: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    heldout = _first_run(candidate, "heldout52")["metrics"]
    gates = candidate.get("gates") or {}
    available_sets = {item["set"]: item for item in candidate["sets"]}
    heldout_set = available_sets["heldout52"]
    qrels_meta = candidate.get("qrels_overlay") or {}
    checks = {
        "heldout_r1_at_least_29": heldout["r1_hits"] >= 29,
        "paired_losses_at_most_1": len(comparison["losses"]) <= 1,
        "latency_p95_within_10_percent": (
            comparison["p95_ratio"] is not None and comparison["p95_ratio"] <= 1.10
        ),
        "repeat_at_least_3": len(heldout_set.get("runs") or []) >= 3,
        "bench100_present": "bench100" in available_sets,
        "all_three_gate_sets_present": all(
            name in gates
            for name in ("simple_negative", "adversarial_negative", "positive")
        ),
        "qrels_overlay_validated": (
            qrels_meta.get("validation") == "schema_and_live_collection_validated"
        ),
    }
    checks["bench100_r1_at_least_85_percent"] = (
        "bench100" in available_sets
        and (
            _first_run(candidate, "bench100")["metrics"]["recall@1"] >= 0.85
        )
    )
    checks.update({
        "simple_negative_12_of_12": (
            gates.get("simple_negative", {}).get("correct") == 12
        ),
        "adversarial_negative_at_least_11_of_12": (
            gates.get("adversarial_negative", {}).get("correct", 0) >= 11
        ),
        "positive_12_of_12": gates.get("positive", {}).get("correct") == 12,
    })
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "provisional",
        "checks": checks,
        "passed": passed,
    }


def _child_command(args, arm: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child-arm", arm,
        "--sets", ",".join(args.sets),
        "--repeat", str(args.repeat),
        "--expected-index-count", str(args.expected_index_count),
        "--device", args.device,
        "--output", str(output),
    ]
    if args.gate:
        command.append("--gate")
    if args.qrels_overlay:
        command.extend(["--qrels-overlay", str(args.qrels_overlay)])
    if args.early_reject_ms is not None:
        command.extend(["--early-reject-ms", str(args.early_reject_ms)])
    return command


def _run_child(args, arm: str, output: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env.update({
        "OFFERCLAW_TORCH_DEVICE": args.device,
        "RAG_RERANK": "1",
        "RAG_HYDE": "0",
        "RAG_QUERY_REWRITE": "0",
        "RAG_DOC2QUERY": "0",
        "RAG_EN_QUOTA": "0",
        "RAG_CONCEPT_ALIAS": "0",
        "RAG_RERANK_BRIDGE": "0",
        "RAG_CRAG": "0",
        "RAG_PAPER_ROUTE": "0",
        "RAG_RANK_FUSION": "off",
        "RAG_RESERVE_SLOT": "",
        "RAG_EN_TOP1_MARGIN": "0",
    })
    subprocess.run(_child_command(args, arm, output), cwd=ROOT, env=env, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="auto",
                        help="auto (A0+B0) or comma-separated A0,A1,A2,A3,B0,C1,C2,H1,H2")
    parser.add_argument("--sets", default="heldout52",
                        help="comma-separated aliases heldout52,bench100,final90 or JSON paths")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--expected-index-count", type=int, default=3348)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qrels-overlay", type=Path,
                        help="optional adjudicated rag-qrels-overlay-v1 for heldout52")
    parser.add_argument("--early-reject-ms", type=float,
                        help="child arm stops after the first query above this latency")
    parser.add_argument("--_child-arm", choices=("A0", "A1", "A2", "A3", "B0", "C1", "C2", "D0", "D1", "D2", "D3", "H1", "H2"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.sets = [value.strip() for value in args.sets.split(",") if value.strip()]
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def main() -> None:
    args = _parse_args()
    if args._child_arm:
        result = run_arm(
            args._child_arm, args.sets, args.repeat,
            args.expected_index_count, args.gate, args.qrels_overlay,
            args.output, args.early_reject_ms,
        )
        if not args.output:
            raise SystemExit("child mode requires --output")
        _write_json(args.output, result)
        return

    auto = args.profiles.strip().lower() == "auto"
    requested = (["A0", "B0"] if auto else
                 [value.strip().upper() for value in args.profiles.split(",") if value.strip()])
    if any(arm not in {"A0", "A1", "A2", "A3", "B0", "C1", "C2", "D0", "D1", "D2", "D3", "H1", "H2"} for arm in requested):
        raise SystemExit("--profiles only accepts auto or A0,A1,A2,A3,B0,C1,C2,H1,H2")
    if "A0" not in requested:
        requested.insert(0, "A0")

    final_output = args.output or (
        ROOT / "logs" / "rag_eval" /
        f"reranker_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    run_dir = final_output.with_suffix("").with_name(final_output.stem + "_arms")
    run_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    comparisons = []
    decisions = {}
    for arm in requested:
        print(f"\n=== serial arm {arm} ===", file=sys.stderr, flush=True)
        results[arm] = _run_child(args, arm, run_dir / f"{arm}.json")

    if auto:
        if "heldout52" not in args.sets:
            raise SystemExit("auto mode requires heldout52")
        comparisons.append(compare_arms(results["A0"], results["B0"], "heldout52"))
    else:
        for arm in requested:
            if arm != "A0":
                for set_name in args.sets:
                    comparisons.append(compare_arms(results["A0"], results[arm], set_name))

    for comparison in comparisons:
        if comparison["set"] == "heldout52":
            decisions[comparison["candidate_arm"]] = _release_decision(
                results[comparison["candidate_arm"]], comparison,
            )
    payload = {
        "schema_version": "reranker-profile-ab-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution": {
            "strictly_serial": True,
            "one_arm_per_process": True,
            "device": args.device,
            "repeat": args.repeat,
            "sets": args.sets,
            "expected_index_count": args.expected_index_count,
            "arm_artifact_dir": str(run_dir),
        },
        "arms": results,
        "comparisons": comparisons,
        "decisions": decisions,
        "early_rejections": KNOWN_EARLY_REJECTIONS,
    }
    _write_json(final_output, payload)
    print(f"\nSaved: {final_output}")
    for arm, result in results.items():
        for evaluated in result["sets"]:
            metrics = evaluated["runs"][0]["metrics"]
            print(
                f"{arm} {evaluated['set']}: R@1 {metrics['r1_hits']}/{metrics['n']} "
                f"Candidate {metrics['candidate_hits']}/{metrics['n']} "
                f"Effective {metrics['effective_r1_hits']}/{metrics['n']} "
                f"p95 {metrics['p95_ms']}ms"
            )
            direct_metrics = evaluated["runs"][0].get("direct_qrels_metrics", {})
            if direct_metrics.get("n"):
                print(
                    f"  direct chunk R@1 {direct_metrics['r1_hits']}/{direct_metrics['n']} "
                    f"Candidate {direct_metrics['candidate_hits']}/{direct_metrics['n']} "
                    f"Effective {direct_metrics['effective_r1_hits']}/{direct_metrics['n']}"
                )
    for arm, decision in decisions.items():
        print(f"{arm} release gate: {'PASS' if decision['passed'] else 'FAIL'} {decision['checks']}")


if __name__ == "__main__":
    main()
