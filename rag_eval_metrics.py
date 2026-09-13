# -*- coding: utf-8 -*-
"""Pure metrics for graded retrieval, evidence gates, and paired experiments."""

from __future__ import annotations

from collections import Counter
import math
import random
import statistics
from typing import Any, Iterable, Sequence


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def wilson_interval(hits: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    """Return count, rate, and Wilson 95% interval for a binomial metric."""

    if n < 0 or hits < 0 or hits > n:
        raise ValueError("invalid binomial counts")
    if n == 0:
        return {"hits": 0, "n": 0, "rate": 0.0, "ci95": [0.0, 0.0]}
    p = hits / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return {
        "hits": hits,
        "n": n,
        "rate": round(p, 6),
        "ci95": [round(max(0.0, center - margin), 6),
                 round(min(1.0, center + margin), 6)],
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rank_of(candidate_ids: Sequence[str], relevant_ids: set[str]) -> int:
    return next(
        (rank for rank, chunk_id in enumerate(candidate_ids, start=1)
         if chunk_id in relevant_ids),
        0,
    )


def reciprocal_rank(candidate_ids: Sequence[str], relevant_ids: set[str], k: int = 10) -> float:
    rank = rank_of(list(candidate_ids)[:k], relevant_ids)
    return 1.0 / rank if rank else 0.0


def dcg(grades: Sequence[int], k: int) -> float:
    return sum(
        (2.0 ** int(grade) - 1.0) / math.log2(position + 2.0)
        for position, grade in enumerate(list(grades)[:k])
    )


def ndcg(candidate_ids: Sequence[str], grades_by_id: dict[str, int], k: int) -> float:
    observed = [int(grades_by_id.get(chunk_id, 0)) for chunk_id in candidate_ids[:k]]
    ideal = sorted((int(value) for value in grades_by_id.values()), reverse=True)[:k]
    ideal_score = dcg(ideal, k)
    return dcg(observed, k) / ideal_score if ideal_score else 0.0


def requirements_covered(
    candidate_ids: Sequence[str],
    requirements_by_id: dict[str, set[str]],
    required: set[str],
    k: int,
) -> set[str]:
    covered: set[str] = set()
    for chunk_id in candidate_ids[:k]:
        covered.update(requirements_by_id.get(chunk_id, set()))
    return covered & required


def context_precision(candidate_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    selected = list(candidate_ids)[:k]
    return safe_ratio(sum(chunk_id in relevant_ids for chunk_id in selected), len(selected))


def duplicate_rate(candidate_ids: Sequence[str], k: int) -> float:
    selected = list(candidate_ids)[:k]
    return 1.0 - safe_ratio(len(set(selected)), len(selected)) if selected else 0.0


def anchor_level_recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recall counted over anchors, not over query rows.

    Each anchor appears in the colloquial sets under three or four phrasings
    that share a gold chunk and a corpus neighbourhood, so those rows are not
    independent observations.  Reporting only per-query counts inflates every
    effect size by roughly the number of styles: the 2026-08-27 evidence-gate
    result read "+7 questions" and was in fact four distinct anchors, and a
    tie-break arm that appeared to break seven Dev80 questions had broken two.

    ``all`` counts an anchor only when every one of its phrasings succeeds --
    the honest reading of "this anchor is retrievable"; ``any`` counts it when
    at least one does.  Both are reported because the gap between them is the
    style sensitivity of that anchor.
    """
    by_anchor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_anchor.setdefault(str(row.get("anchor_id") or ""), []).append(row)
    out: dict[str, Any] = {"anchors": len(by_anchor)}
    for k in (1, 3, 5):
        hit = {anchor: [0 < int(row.get("final_rank") or 0) <= k for row in group]
               for anchor, group in by_anchor.items()}
        out[f"recall@{k}"] = {
            "any": sum(1 for flags in hit.values() if any(flags)),
            "all": sum(1 for flags in hit.values() if all(flags)),
            "n": len(by_anchor),
        }
    effective = {anchor: [bool(row.get("effective_evidence")) for row in group]
                 for anchor, group in by_anchor.items()}
    out["effective_evidence"] = {
        "any": sum(1 for flags in effective.values() if any(flags)),
        "all": sum(1 for flags in effective.values() if all(flags)),
        "n": len(by_anchor),
    }
    return out


def aggregate_ranked_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate positive per-query rows emitted by the production-mirror evaluator."""

    n = len(rows)
    rank_counts = {
        k: sum(0 < int(row.get("final_rank") or 0) <= k for row in rows)
        for k in (1, 3, 5)
    }
    candidate_hits = sum(bool(row.get("fusion_rank")) for row in rows)
    union_hits = sum(bool(row.get("union_rank")) for row in rows)
    correct_top1 = rank_counts[1]
    effective = sum(bool(row.get("effective_evidence")) for row in rows)
    gate_positive = sum(bool(row.get("correct_top1_gate_pass")) for row in rows)
    sufficient5 = sum(bool(row.get("sufficient_evidence_at_5")) for row in rows)
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    stages = sorted({
        stage for row in rows for stage in (row.get("latency_by_stage") or {})
    })
    stage_latency = {
        stage: {
            "p50_ms": round(percentile([
                float((row.get("latency_by_stage") or {}).get(stage, 0.0))
                for row in rows
            ], 0.50), 3),
            "p95_ms": round(percentile([
                float((row.get("latency_by_stage") or {}).get(stage, 0.0))
                for row in rows
            ], 0.95), 3),
            "p99_ms": round(percentile([
                float((row.get("latency_by_stage") or {}).get(stage, 0.0))
                for row in rows
            ], 0.99), 3),
        }
        for stage in stages
    }
    promoted = sum(
        int(row.get("fusion_rank") or 0) != 1
        and int(row.get("final_rank") or 0) == 1
        for row in rows
    )
    harmed = sum(
        int(row.get("fusion_rank") or 0) == 1
        and int(row.get("final_rank") or 0) != 1
        for row in rows
    )
    top3_preserved = sum(
        0 < int(row.get("fusion_rank") or 0) <= 3
        and 0 < int(row.get("final_rank") or 0) <= 3
        for row in rows
    )
    fusion_top3 = sum(0 < int(row.get("fusion_rank") or 0) <= 3 for row in rows)
    no_rerank_counts = {
        k: sum(0 < int(row.get("fusion_rank") or 0) <= k for row in rows)
        for k in (1, 3, 5)
    }
    return {
        "n": n,
        "anchor_level": anchor_level_recall(rows),
        "strict_ranking": {
            "recall@1": wilson_interval(rank_counts[1], n),
            "recall@3": wilson_interval(rank_counts[3], n),
            "recall@5": wilson_interval(rank_counts[5], n),
            "mrr@10": round(statistics.fmean(
                float(row.get("reciprocal_rank_at_10") or 0.0) for row in rows
            ), 6) if rows else 0.0,
            "ndcg@5": round(statistics.fmean(
                float(row.get("ndcg_at_5") or 0.0) for row in rows
            ), 6) if rows else 0.0,
            "ndcg@10": round(statistics.fmean(
                float(row.get("ndcg_at_10") or 0.0) for row in rows
            ), 6) if rows else 0.0,
        },
        "funnel": {
            "union_candidate": wilson_interval(union_hits, n),
            "rrf_candidate": wilson_interval(candidate_hits, n),
            "union_preservation": {
                **wilson_interval(candidate_hits, union_hits),
                "denominator": "queries_with_relevant_union_candidate",
            },
            "candidate_to_top1": {
                **wilson_interval(correct_top1, candidate_hits),
                "denominator": "queries_with_relevant_rrf_candidate",
            },
            "correct_top1_gate_pass": {
                **wilson_interval(gate_positive, correct_top1),
                "denominator": "queries_with_correct_final_top1",
            },
            "effective_evidence": wilson_interval(effective, n),
        },
        "reranker": {
            "no_rerank_diagnostic": {
                f"recall@{k}": wilson_interval(hits, n)
                for k, hits in no_rerank_counts.items()
            },
            "positive_promoted": promoted,
            "positive_harmed": harmed,
            "gain_minus_harm": promoted - harmed,
            "top3_preservation": wilson_interval(top3_preserved, fusion_top3),
        },
        "context": {
            "sufficient_evidence@5": wilson_interval(sufficient5, n),
            "mean_context_precision@5": round(statistics.fmean(
                float(row.get("context_precision_at_5") or 0.0) for row in rows
            ), 6) if rows else 0.0,
            "mean_requirement_coverage@5": round(statistics.fmean(
                float(row.get("requirement_coverage_at_5") or 0.0) for row in rows
            ), 6) if rows else 0.0,
            "mean_duplicate_rate@5": round(statistics.fmean(
                float(row.get("duplicate_rate_at_5") or 0.0) for row in rows
            ), 6) if rows else 0.0,
        },
        "latency": {
            "p50_ms": round(percentile(latencies, 0.50), 3),
            "p95_ms": round(percentile(latencies, 0.95), 3),
            "p99_ms": round(percentile(latencies, 0.99), 3),
            "by_stage": stage_latency,
        },
    }


def aggregate_negative_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    false_accepts = sum(bool(row.get("gate_decision")) for row in rows)
    return {
        "n": n,
        "false_accept": wilson_interval(false_accepts, n),
        "true_reject": wilson_interval(n - false_accepts, n),
        "false_accept_query_ids": [
            str(row.get("query_id")) for row in rows if row.get("gate_decision")
        ],
    }


def paired_wins_losses(
    baseline: dict[str, bool], candidate: dict[str, bool],
) -> dict[str, Any]:
    ids = sorted(set(baseline) & set(candidate))
    wins = [query_id for query_id in ids if not baseline[query_id] and candidate[query_id]]
    losses = [query_id for query_id in ids if baseline[query_id] and not candidate[query_id]]
    ties = len(ids) - len(wins) - len(losses)
    return {
        "n": len(ids),
        "wins": len(wins),
        "losses": len(losses),
        "ties": ties,
        "net": len(wins) - len(losses),
        "win_query_ids": wins,
        "loss_query_ids": losses,
        "mcnemar_exact_p": mcnemar_exact_p(len(wins), len(losses)),
    }


def mcnemar_exact_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2.0 ** discordant)
    return round(min(1.0, 2.0 * probability), 8)


def paired_bootstrap_ci(
    deltas: Iterable[float], *, samples: int = 10000, seed: int = 20260825,
) -> dict[str, float]:
    values = [float(value) for value in deltas]
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(max(1, samples))
    ]
    return {
        "mean": round(statistics.fmean(values), 6),
        "ci95_low": round(percentile(means, 0.025), 6),
        "ci95_high": round(percentile(means, 0.975), 6),
    }


def channel_exclusivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        dense = set(row.get("dense_chunk_ids") or [])
        sparse = set(row.get("bm25_chunk_ids") or [])
        relevant = set(row.get("grade3_chunk_ids") or [])
        counts["dense_only_hits"] += int(bool((dense - sparse) & relevant))
        counts["bm25_only_hits"] += int(bool((sparse - dense) & relevant))
        counts["shared_hits"] += int(bool((dense & sparse) & relevant))
    return dict(counts)
