# -*- coding: utf-8 -*-
"""Fail-closed release gate for the sealed 64-positive/16-negative blind set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _value(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        current = current[key]
    return current


def evaluate_release_artifact(
    artifact: dict[str, Any], *, regression: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: float | int | bool, operator: str,
              threshold: float | int | bool) -> None:
        if operator == ">=":
            passed = observed >= threshold
        elif operator == "<=":
            passed = observed <= threshold
        elif operator == "==":
            passed = observed == threshold
        else:
            raise ValueError(f"unsupported gate operator: {operator}")
        checks.append({
            "name": name, "observed": observed, "operator": operator,
            "threshold": threshold, "passed": bool(passed),
        })

    check("release_eligible_artifact", bool(artifact.get("release_eligible")), "==", True)
    check("sealed_blind_input", artifact.get("input", {}).get("splits") == ["blind"], "==", True)
    run = artifact["runs"][0]
    positive = run["positive"]
    negative = run["negative"]
    metrics = positive["metrics"]
    check("blind_positive_count", metrics["n"], "==", 64)
    check("blind_negative_count", negative["metrics"]["n"], "==", 16)
    check("strict_candidate@20", _value(
        metrics, "retriever_and_fusion", "fusion", "recall@20", "hits"
    ), ">=", 54)
    check("union_preservation", _value(
        metrics, "funnel", "union_preservation", "rate"
    ), ">=", 0.98)
    for k, threshold in ((1, 42), (3, 51), (5, 55)):
        check(f"strict_r@{k}", _value(
            metrics, "strict_ranking", f"recall@{k}", "hits"
        ), ">=", threshold)
    check("mrr@10", _value(metrics, "strict_ranking", "mrr@10"), ">=", 0.72)
    check("ndcg@5", _value(metrics, "strict_ranking", "ndcg@5"), ">=", 0.80)
    check("candidate_to_top1", _value(
        metrics, "funnel", "candidate_to_top1", "rate"
    ), ">=", 0.75)
    check("sufficient_evidence@5", _value(
        metrics, "context", "sufficient_evidence@5", "hits"
    ), ">=", 54)
    check("gate_positive_pass", _value(
        metrics, "funnel", "correct_top1_gate_pass", "rate"
    ), ">=", 0.90)
    check("effective_evidence", _value(
        metrics, "funnel", "effective_evidence", "hits"
    ), ">=", 38)

    generation = artifact.get("generation") or {}
    check("generation_scores_present", generation.get("status") == "completed", "==", True)
    if generation.get("status") == "completed":
        check("claim_faithfulness", generation["claim_faithfulness"], ">=", 0.95)
        check("answer_correctness", generation["answer_correctness"], ">=", 0.85)
        check("citation_precision", generation["citation_precision"], ">=", 0.95)
        check("citation_recall", generation["citation_recall"], ">=", 0.90)

    negative_rows = negative["rows"]
    out_of_domain_false_accepts = sum(
        row.get("gate_decision")
        and "out_of_domain" in (row.get("phenomena") or [])
        for row in negative_rows
    )
    check("out_of_domain_false_accept", out_of_domain_false_accepts, "==", 0)
    check("blind_false_accept_total", _value(
        negative, "metrics", "false_accept", "hits"
    ), "<=", 2)

    # Existing same-distribution, Final90, old negatives and paper_quality
    # guards are independent artifacts.  Omitting them can never produce GO.
    check("regression_report_present", regression is not None, "==", True)
    if regression is not None:
        required = {
            "bench100_r1_hits": (85, ">="),
            "bench100_r5_hits": (98, ">="),
            "bench100_mrr": (0.90, ">="),
            "final90_r1_hits": (70, ">="),
            "simple_negative_rejects": (12, "=="),
            "paper_grounded_correct": (30, ">="),
            "paper_wrong_accepts": (0, "=="),
        }
        for key, (threshold, operator) in required.items():
            if key not in regression:
                checks.append({
                    "name": f"regression.{key}", "observed": None,
                    "operator": operator, "threshold": threshold, "passed": False,
                    "reason": "missing",
                })
            else:
                check(f"regression.{key}", regression[key], operator, threshold)

    failed = [item for item in checks if not item["passed"]]
    return {
        "decision": "GO" if not failed else "NO_GO",
        "checks": checks,
        "failed": failed,
    }


def load_and_check(
    artifact_path: str | Path, regression_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    regression = (
        json.loads(Path(regression_path).read_text(encoding="utf-8"))
        if regression_path else None
    )
    return evaluate_release_artifact(artifact, regression=regression)
