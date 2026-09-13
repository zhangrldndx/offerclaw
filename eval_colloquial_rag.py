#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production-mirror evaluation for graded colloquial RAG cases.

Release evaluation is fail-closed: every row must be human-approved and a
blind input may only write results outside the repository.  ``--allow-draft``
exists solely for smoke/diagnostic runs and marks the artifact ineligible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval_metrics import (  # noqa: E402
    aggregate_negative_rows,
    aggregate_ranked_rows,
    channel_exclusivity,
    context_precision,
    duplicate_rate,
    ndcg,
    paired_bootstrap_ci,
    paired_wins_losses,
    rank_of,
    reciprocal_rank,
    requirements_covered,
    wilson_interval,
)
from rag_qrels_v2 import (  # noqa: E402
    index_contract_fingerprint,
    load_graded_qrels,
    validate_graded_qrels_against_collection,
)


REFERENCE_PLAN = {
    "decision": "answer",
    "routes": [{"source": "reference_kb", "operation": "search"}],
}


def _ids(candidates: list[Any]) -> list[str]:
    return [str(candidate.chunk_id) for candidate in candidates]


def _unique_union(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys([*left, *right]))


def _target_maps(item: dict[str, Any]) -> tuple[dict[str, int], dict[str, set[str]]]:
    grades: dict[str, int] = {}
    requirements: dict[str, set[str]] = defaultdict(set)
    for target in item["relevant_targets"]:
        chunk_id = str(target["chunk_id"])
        grades[chunk_id] = max(grades.get(chunk_id, 0), int(target["relevance_grade"]))
        requirements[chunk_id].update(target["supported_requirements"])
    return grades, dict(requirements)


def _stage_recalls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for stage in ("dense", "bm25", "union", "fusion", "reranked", "final"):
        stage_metrics = {}
        for k in (1, 3, 5, 10, 20):
            if stage == "union":
                hits = sum(
                    bool((
                        set((row.get("dense_chunk_ids") or [])[:k])
                        | set((row.get("bm25_chunk_ids") or [])[:k])
                    ) & set(row.get("grade3_chunk_ids") or []))
                    for row in rows
                )
            else:
                hits = sum(
                    bool(set((row.get(f"{stage}_chunk_ids") or [])[:k])
                         & set(row.get("grade3_chunk_ids") or []))
                    for row in rows
                )
            stage_metrics[f"recall@{k}"] = wilson_interval(hits, len(rows))
        output[stage] = stage_metrics
    return output


def _router_plan(question: str) -> tuple[dict[str, Any], float]:
    from rag_query_plan import plan_query

    started = time.perf_counter()
    plan = plan_query(question)
    if hasattr(plan, "to_dict"):
        plan = plan.to_dict()
    elif not isinstance(plan, dict):
        plan = dict(plan)
    return plan, (time.perf_counter() - started) * 1000


def _route_keys(plan: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(route.get("source") or "") for route in plan.get("routes", [])
        if route.get("source")
    ))


def _router_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or not any("planned_routes" in row for row in rows):
        return {"status": "not_run"}
    required_hits = sum("reference_kb" in row.get("planned_routes", []) for row in rows)
    exact = sum(row.get("planned_routes") == ["reference_kb"] for row in rows)
    # With one required source per row, source-set macro F1 is the mean of
    # per-query set F1 and remains interpretable when optional routes appear.
    f1_values = []
    for row in rows:
        predicted = set(row.get("planned_routes") or [])
        expected = {"reference_kb"}
        overlap = len(predicted & expected)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(expected)
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return {
        "required_route_recall": wilson_interval(required_hits, len(rows)),
        "exact_reference_only": wilson_interval(exact, len(rows)),
        "source_set_macro_f1": round(statistics.fmean(f1_values), 6),
        "mean_planner_ms": round(statistics.fmean(
            float(row.get("planner_ms") or 0.0) for row in rows
        ), 3),
    }


def _generation_metrics(path: Path | None, expected_ids: set[str]) -> dict[str, Any]:
    if path is None:
        return {"status": "not_run", "reason": "no generation score artifact supplied"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items") if isinstance(payload, dict) else payload
    by_id = {str(row["query_id"]): row for row in rows}
    if not expected_ids <= set(by_id):
        raise ValueError(
            f"generation score coverage missing {sorted(expected_ids - set(by_id))}"
        )
    keys = (
        "claim_faithfulness", "answer_correctness", "answer_relevance",
        "citation_precision", "citation_recall",
    )
    result = {"status": "completed", "n": len(expected_ids)}
    for key in keys:
        values = [float(by_id[query_id][key]) for query_id in expected_ids]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{key} scores must be in [0, 1]")
        result[key] = round(statistics.fmean(values), 6)
    return result


def _positive_row(item: dict[str, Any], trace: Any) -> dict[str, Any]:
    grades, requirements_by_id = _target_maps(item)
    grade3 = {chunk_id for chunk_id, grade in grades.items() if grade == 3}
    relevant = set(grades)
    required = set(item["answer_requirements"])
    dense_ids = _ids(trace.dense_candidates)
    bm25_ids = _ids(trace.bm25_candidates)
    union_ids = _unique_union(dense_ids, bm25_ids)
    fusion_ids = _ids(trace.fusion_candidates)
    reranked_ids = _ids(trace.reranked_candidates)
    final_ids = _ids(trace.final_candidates)
    final_rank = rank_of(final_ids, grade3)
    fusion_rank = rank_of(fusion_ids, grade3)
    dense_rank = rank_of(dense_ids, grade3)
    bm25_rank = rank_of(bm25_ids, grade3)
    union_ranks = [rank for rank in (dense_rank, bm25_rank) if rank]
    union_rank = min(union_ranks) if union_ranks else 0
    covered = requirements_covered(final_ids, requirements_by_id, required, 5)
    return {
        "query_id": item["query_id"],
        "anchor_id": item["anchor_id"],
        "domain": item.get("domain", ""),
        "query_style": item["query_style"],
        "phenomena": list(item["phenomena"]),
        "grade3_chunk_ids": sorted(grade3),
        "all_relevant_chunk_ids": sorted(relevant),
        "dense_chunk_ids": dense_ids,
        "bm25_chunk_ids": bm25_ids,
        "union_chunk_ids": union_ids,
        "fusion_chunk_ids": fusion_ids,
        "reranked_chunk_ids": reranked_ids,
        "final_chunk_ids": final_ids,
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "union_rank": union_rank,
        "fusion_rank": fusion_rank,
        "reranked_rank": rank_of(reranked_ids, grade3),
        "final_rank": final_rank,
        "reciprocal_rank_at_10": reciprocal_rank(final_ids, grade3, 10),
        "ndcg_at_5": ndcg(final_ids, grades, 5),
        "ndcg_at_10": ndcg(final_ids, grades, 10),
        "context_precision_at_5": context_precision(final_ids, relevant, 5),
        "requirement_coverage_at_5": len(covered) / len(required) if required else 0.0,
        "sufficient_evidence_at_5": bool(required and covered == required),
        "duplicate_rate_at_5": duplicate_rate(final_ids, 5),
        "gate_decision": bool(trace.gate_decision),
        "correct_top1_gate_pass": bool(final_rank == 1 and trace.gate_decision),
        "effective_evidence": bool(final_rank == 1 and trace.gate_decision),
        "gate_features": dict(trace.gate_features),
        "gate_candidate_chunk_id": trace.gate_candidate_chunk_id,
        "gate_alignment": trace.gate_alignment,
        "latency_ms": float(trace.latency_by_stage.get("total", 0.0)),
        "latency_by_stage": dict(trace.latency_by_stage),
        "reranker_requested": trace.reranker_requested,
        "reranker_actual": trace.reranker_actual,
        "reranker_status": trace.reranker_status,
        "retrieval_trace": trace.to_dict(),
    }


def _negative_row(item: dict[str, Any], trace: Any) -> dict[str, Any]:
    return {
        "query_id": item["query_id"],
        "phenomena": list(item["phenomena"]),
        "gate_decision": bool(trace.gate_decision),
        "final_chunk_ids": _ids(trace.final_candidates),
        "gate_features": dict(trace.gate_features),
        "latency_ms": float(trace.latency_by_stage.get("total", 0.0)),
        "latency_by_stage": dict(trace.latency_by_stage),
        "retrieval_trace": trace.to_dict(),
    }


def _evaluate_once(
    items: list[dict[str, Any]], *, profile: Any, run_index: int,
    route_mode: str,
) -> dict[str, Any]:
    from rag_gate import retrieve_with_trace

    positive_rows = []
    negative_rows = []
    for position, item in enumerate(items, start=1):
        print(
            f"[{profile.name}] run {run_index} {position}/{len(items)} "
            f"{item['query_id']}", file=sys.stderr, flush=True,
        )
        planned_routes = None
        planner_ms = None
        if route_mode == "production":
            plan, planner_ms = _router_plan(item["question"])
            planned_routes = _route_keys(plan)
        trace = retrieve_with_trace(
            item["question"], REFERENCE_PLAN, profile, top_k=5,
        )
        row = (_positive_row(item, trace) if item["case_kind"] == "positive"
               else _negative_row(item, trace))
        if planned_routes is not None:
            row["planned_routes"] = planned_routes
            row["planner_ms"] = planner_ms
        if item["case_kind"] == "positive":
            positive_rows.append(row)
        else:
            negative_rows.append(row)
    return {
        "run": run_index,
        "positive": {
            "metrics": {
                **aggregate_ranked_rows(positive_rows),
                "retriever_and_fusion": _stage_recalls(positive_rows),
                "channel_exclusivity": channel_exclusivity(positive_rows),
                "router": _router_metrics(positive_rows),
            },
            "rows": positive_rows,
        },
        "negative": {
            "metrics": aggregate_negative_rows(negative_rows),
            "rows": negative_rows,
        },
    }


def _stability(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        return {"runs": len(runs), "route_set_consistency": None,
                "top1_consistency": None}
    top1_by_id: dict[str, list[str]] = defaultdict(list)
    routes_by_id: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for run in runs:
        for row in run["positive"]["rows"]:
            top1_by_id[row["query_id"]].append(
                (row.get("final_chunk_ids") or [""])[0]
            )
            if "planned_routes" in row:
                routes_by_id[row["query_id"]].append(tuple(row["planned_routes"]))
    return {
        "runs": len(runs),
        "top1_consistency": round(sum(len(set(values)) == 1 for values in top1_by_id.values())
                                  / len(top1_by_id), 6) if top1_by_id else None,
        "route_set_consistency": round(sum(len(set(values)) == 1 for values in routes_by_id.values())
                                       / len(routes_by_id), 6) if routes_by_id else None,
    }


def _comparison(current: dict[str, Any], baseline_path: Path | None) -> dict[str, Any]:
    if baseline_path is None:
        return {"status": "not_run"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = baseline["runs"][0]["positive"]["rows"]
    current_rows = current["runs"][0]["positive"]["rows"]
    base_hit = {row["query_id"]: int(row.get("final_rank") or 0) == 1
                for row in baseline_rows}
    current_hit = {row["query_id"]: int(row.get("final_rank") or 0) == 1
                   for row in current_rows}
    common = sorted(set(base_hit) & set(current_hit))
    deltas = [float(current_hit[query_id]) - float(base_hit[query_id])
              for query_id in common]
    return {
        "status": "completed",
        "baseline_path": str(baseline_path),
        "strict_r1": paired_wins_losses(base_hit, current_hit),
        "paired_bootstrap": paired_bootstrap_ci(deltas),
    }


def main(args: argparse.Namespace) -> None:
    import chromadb
    from rag_tools import get_collection_name, index_fingerprint

    input_path = Path(args.cases).expanduser().resolve()
    payload = load_graded_qrels(input_path, require_approved=not args.allow_draft)
    splits = {item["split"] for item in payload["items"]}
    output_path = Path(args.output).expanduser().resolve()
    if "blind" in splits and (output_path == ROOT or ROOT in output_path.parents):
        raise SystemExit("blind evaluation output must stay outside the repository")
    if args.max_cases:
        payload["items"] = payload["items"][:args.max_cases]
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    validate_graded_qrels_against_collection(payload, collection)
    if collection.count() != payload["index"]["count"]:
        raise SystemExit(
            f"index count drift: qrels={payload['index']['count']} live={collection.count()}"
        )
    live_index = index_fingerprint(collection=collection)
    live_contract_fingerprint = index_contract_fingerprint(live_index)
    if live_contract_fingerprint != payload["index"].get("fingerprint"):
        raise SystemExit(
            "index fingerprint drift: "
            f"qrels={payload['index'].get('fingerprint')} "
            f"live={live_contract_fingerprint}"
        )
    from rag_colloquial_profiles import colloquial_profile
    reranker_model = None
    if args.reranker_model:
        candidate = Path(args.reranker_model).expanduser()
        reranker_model = str(candidate.resolve()) if candidate.exists() else args.reranker_model
    profile = colloquial_profile(args.arm, reranker_model=reranker_model)
    runs = [
        _evaluate_once(
            payload["items"], profile=profile, run_index=run_index,
            route_mode=args.route_mode,
        )
        for run_index in range(1, args.repeat + 1)
    ]
    positive_ids = {
        item["query_id"] for item in payload["items"]
        if item["case_kind"] == "positive"
    }
    artifact = {
        "schema_version": "rag-colloquial-evaluation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_eligible": not args.allow_draft,
        "draft_diagnostic_only": bool(args.allow_draft),
        "input": {
            "path": str(input_path),
            "dataset_id": payload["dataset_id"],
            "splits": sorted(splits),
            "rows": len(payload["items"]),
            "index": payload["index"],
        },
        "configuration": {
            "retrieval_arm": args.arm,
            "retrieval_profile": profile.to_dict(),
            "route_mode": args.route_mode,
            "repeat": args.repeat,
            "process_environment": {
                "RAG_RETRIEVAL_PROFILE": os.environ.get("RAG_RETRIEVAL_PROFILE", ""),
                "RAG_COLLECTION_NAME": os.environ.get("RAG_COLLECTION_NAME", ""),
                "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", ""),
            },
        },
        "runs": runs,
        "stability": _stability(runs),
        "generation": _generation_metrics(
            Path(args.generation_scores).expanduser().resolve()
            if args.generation_scores else None,
            positive_ids,
        ),
        "engineering": {
            "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_rss_platform_note": "KiB on Linux; bytes on macOS",
        },
    }
    artifact["comparison"] = _comparison(
        artifact,
        Path(args.compare_to).expanduser().resolve() if args.compare_to else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "release_eligible": artifact["release_eligible"],
        "positive": runs[0]["positive"]["metrics"],
        "negative": runs[0]["negative"]["metrics"],
        "stability": artifact["stability"],
    }, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cases", required=True)
    result.add_argument("--output", required=True)
    from rag_colloquial_profiles import ARMS
    result.add_argument("--arm", default="baseline", choices=tuple(ARMS))
    result.add_argument("--route-mode", default="oracle",
                        choices=("oracle", "production"))
    result.add_argument("--repeat", type=int, default=1)
    result.add_argument("--allow-draft", action="store_true")
    result.add_argument("--max-cases", type=int, default=0)
    result.add_argument("--generation-scores")
    result.add_argument("--compare-to")
    result.add_argument(
        "--reranker-model",
        help=(
            "offline-only model id or local development checkpoint; all other "
            "settings continue to come from --arm"
        ),
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.repeat < 1:
        raise SystemExit("--repeat must be positive")
    main(arguments)
