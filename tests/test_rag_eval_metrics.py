from rag_eval_metrics import (
    aggregate_negative_rows,
    aggregate_ranked_rows,
    mcnemar_exact_p,
    ndcg,
    paired_bootstrap_ci,
    paired_wins_losses,
    requirements_covered,
    wilson_interval,
)


def test_wilson_reports_counts_and_non_degenerate_interval():
    result = wilson_interval(8, 10)
    assert result["hits"] == 8 and result["n"] == 10
    assert result["ci95"][0] < 0.8 < result["ci95"][1]


def test_ndcg_rewards_correct_grade_order():
    grades = {"direct": 3, "support": 2}
    assert ndcg(["direct", "support"], grades, 2) == 1.0
    assert ndcg(["support", "direct"], grades, 2) < 1.0


def test_requirement_coverage_can_require_multiple_chunks():
    mapping = {"a": {"mechanism"}, "b": {"boundary"}}
    assert requirements_covered(["a", "b"], mapping, {"mechanism", "boundary"}, 2) == {
        "mechanism", "boundary",
    }
    assert requirements_covered(["a", "b"], mapping, {"mechanism", "boundary"}, 1) == {
        "mechanism",
    }


def test_funnel_uses_stage_specific_denominators_and_no_rerank_control():
    rows = [{
        "final_rank": 1, "fusion_rank": 2, "union_rank": 1,
        "effective_evidence": True, "correct_top1_gate_pass": True,
        "sufficient_evidence_at_5": True, "context_precision_at_5": 0.4,
        "requirement_coverage_at_5": 1.0, "duplicate_rate_at_5": 0.0,
        "reciprocal_rank_at_10": 1.0, "ndcg_at_5": 1.0, "ndcg_at_10": 1.0,
        "latency_ms": 10, "latency_by_stage": {"dense": 2, "total": 10},
    }, {
        "final_rank": 0, "fusion_rank": 1, "union_rank": 1,
        "effective_evidence": False, "correct_top1_gate_pass": False,
        "sufficient_evidence_at_5": False, "context_precision_at_5": 0.0,
        "requirement_coverage_at_5": 0.0, "duplicate_rate_at_5": 0.0,
        "reciprocal_rank_at_10": 0.0, "ndcg_at_5": 0.0, "ndcg_at_10": 0.0,
        "latency_ms": 20, "latency_by_stage": {"dense": 3, "total": 20},
    }]
    result = aggregate_ranked_rows(rows)
    assert result["funnel"]["union_candidate"]["hits"] == 2
    assert result["funnel"]["candidate_to_top1"]["hits"] == 1
    assert result["reranker"]["positive_promoted"] == 1
    assert result["reranker"]["positive_harmed"] == 1
    assert result["reranker"]["no_rerank_diagnostic"]["recall@1"]["hits"] == 1


def test_negative_and_paired_statistics_are_fail_visible():
    negatives = aggregate_negative_rows([
        {"query_id": "n1", "gate_decision": False},
        {"query_id": "n2", "gate_decision": True},
    ])
    assert negatives["false_accept"]["hits"] == 1
    paired = paired_wins_losses(
        {"a": False, "b": True, "c": True},
        {"a": True, "b": False, "c": True},
    )
    assert paired["wins"] == paired["losses"] == 1
    assert mcnemar_exact_p(1, 1) == 1.0
    assert paired_bootstrap_ci([1, 1, 0], samples=100)["ci95_low"] >= 0

