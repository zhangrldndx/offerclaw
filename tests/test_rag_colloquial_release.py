from rag_colloquial_release import evaluate_release_artifact


def _artifact():
    metric = lambda hits, n=64: {"hits": hits, "n": n, "rate": hits / n}
    negative_metric = lambda hits: {"hits": hits, "n": 16, "rate": hits / 16}
    return {
        "release_eligible": True,
        "input": {"splits": ["blind"]},
        "generation": {
            "status": "completed", "claim_faithfulness": 0.96,
            "answer_correctness": 0.86, "answer_relevance": 0.9,
            "citation_precision": 0.96, "citation_recall": 0.91,
        },
        "runs": [{
            "positive": {"metrics": {
                "n": 64,
                "retriever_and_fusion": {"fusion": {"recall@20": metric(54)}},
                "funnel": {
                    "union_preservation": {"rate": 0.99},
                    "candidate_to_top1": {"rate": 0.76},
                    "correct_top1_gate_pass": {"rate": 0.91},
                    "effective_evidence": metric(38),
                },
                "strict_ranking": {
                    "recall@1": metric(42), "recall@3": metric(51),
                    "recall@5": metric(55), "mrr@10": 0.73, "ndcg@5": 0.81,
                },
                "context": {"sufficient_evidence@5": metric(54)},
            }},
            "negative": {
                "metrics": {"n": 16, "false_accept": negative_metric(1)},
                "rows": [{"phenomena": ["near_domain_missing"], "gate_decision": True}],
            },
        }],
    }


REGRESSION = {
    "bench100_r1_hits": 85, "bench100_r5_hits": 98, "bench100_mrr": 0.90,
    "final90_r1_hits": 70, "simple_negative_rejects": 12,
    "paper_grounded_correct": 30, "paper_wrong_accepts": 0,
}


def test_release_gate_requires_every_new_and_existing_guard():
    report = evaluate_release_artifact(_artifact(), regression=REGRESSION)
    assert report["decision"] == "GO"
    assert not report["failed"]


def test_release_gate_fails_closed_without_generation_or_regression():
    artifact = _artifact()
    artifact["generation"] = {"status": "not_run"}
    report = evaluate_release_artifact(artifact)
    assert report["decision"] == "NO_GO"
    assert {item["name"] for item in report["failed"]} >= {
        "generation_scores_present", "regression_report_present",
    }
