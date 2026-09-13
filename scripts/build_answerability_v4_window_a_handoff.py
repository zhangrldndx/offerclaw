#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the v4 candidate set and build the organic Window A protocol.

Only public artifacts are parsed.  ``private_audit.jsonl`` and
``generated_queries_private.jsonl`` are covered by their source manifest byte
count and SHA256, but their contents are never decoded or inspected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_answerability_shadow_seed import (  # noqa: E402
    _file_sha256,
    _git_state,
    _json_dump,
)


ANALYSIS_SCHEMA = "answerability-v4-candidate-freeze-v1"
PROTOCOL_SCHEMA = "answerability-organic-window-a-protocol-v1"
JUDGE_SCHEMA = "answerability-v4"
JUDGE_QUESTION_ROLE = "original_user_question"
RELEASE_STATUS = "candidate_discovery_only_not_blind"

REQUIRED_PACKAGE_FILES = frozenset({
    "route_outcomes.jsonl",
    "claude_agent_calibration.jsonl",
    "private_audit.jsonl",
    "summary.json",
    "CLAUDE_HANDOFF.md",
    "generated_queries_private.jsonl",
})
PRIVATE_UNPARSED_FILES = frozenset({
    "private_audit.jsonl", "generated_queries_private.jsonl",
})
PUBLIC_JSON_FILES = (
    "route_outcomes.jsonl", "claude_agent_calibration.jsonl", "summary.json",
)
OUTPUT_FILES = (
    "INPUT_AUDIT.md",
    "CANDIDATE_SET_FREEZE.md",
    "ORGANIC_WINDOW_A_PREREGISTRATION.md",
    "ORGANIC_WINDOW_A_PROTOCOL.json",
    "CLAUDE_HANDOFF.md",
)

PUBLIC_FORBIDDEN_KEYS = frozenset({
    "question", "retrieval_question", "document", "chunk_id",
    "heading_path", "candidates_private", "reason", "error",
})
QUESTION_FORMS = frozenset({"polar", "open"})
DIRECT_ANSWERS = frozenset({
    "proposition_true", "proposition_false", "unknown", "not_applicable",
})
PREMISE_STATUSES = frozenset({
    "supported", "refuted", "not_established", "none",
})
RELATIONS = frozenset({"entails", "contradicts", "not_established"})
ACTIONS = frozenset({"answer", "correct_premise", "abstain"})
ACCEPTED_INTERVENTIONS = frozenset({
    "block_unsupported", "correct_premise", "rerank_evidence",
})
ALLOWED_FEATURES = (
    "rerank_margin", "rerank_top", "high_scorers",
    "gate_score_minus_threshold",
)


RULES: dict[str, dict[str, Any]] = {
    "SAT": {
        "description": "legacy saturation diagnostic",
        "expression": (
            "rerank_margin < 0.01 AND rerank_top >= 0.99 "
            "AND high_scorers == 5"
        ),
        "conditions": [
            {"feature": "rerank_margin", "operator": "lt", "value": 0.01},
            {"feature": "rerank_top", "operator": "gte", "value": 0.99},
            {"feature": "high_scorers", "operator": "eq", "value": 5},
        ],
        "condition_count": 3,
        "selection_eligible": False,
    },
    "H1": {
        "description": "separated topical winner",
        "expression": "rerank_margin >= 0.01",
        "conditions": [
            {"feature": "rerank_margin", "operator": "gte", "value": 0.01},
        ],
        "condition_count": 1,
        "selection_eligible": True,
    },
    "H2": {
        "description": "not fully saturated and separated",
        "expression": "high_scorers <= 4 AND rerank_margin >= 0.01",
        "conditions": [
            {"feature": "high_scorers", "operator": "lte", "value": 4},
            {"feature": "rerank_margin", "operator": "gte", "value": 0.01},
        ],
        "condition_count": 2,
        "selection_eligible": True,
    },
    "H3": {
        "description": "limited gate surplus and non-extreme top score",
        "expression": (
            "gate_score_minus_threshold <= 0.14 AND rerank_top <= 0.99"
        ),
        "conditions": [
            {
                "feature": "gate_score_minus_threshold",
                "operator": "lte",
                "value": 0.14,
            },
            {"feature": "rerank_top", "operator": "lte", "value": 0.99},
        ],
        "condition_count": 2,
        "selection_eligible": True,
    },
}


class FreezeBuildError(RuntimeError):
    """Raised before publication when an input or protocol contract fails."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeBuildError(f"cannot parse public JSON {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FreezeBuildError(f"public JSON {path.name} is not an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FreezeBuildError(f"cannot read public JSONL {path.name}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FreezeBuildError(
                f"cannot parse public JSONL {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise FreezeBuildError(
                f"public JSONL row is not an object in {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def _relative_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.resolve().name


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _walk_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _walk_keys(child)}
    return set()


def _scan_public_payload(value: Any, *, artifact: str) -> None:
    forbidden = _walk_keys(value) & PUBLIC_FORBIDDEN_KEYS
    if forbidden:
        raise FreezeBuildError(
            f"public artifact {artifact} contains private keys: "
            + ", ".join(sorted(forbidden))
        )
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if re.search(r"(?:/Users/|/home/|[A-Za-z]:\\\\)", encoded):
        raise FreezeBuildError(f"public artifact {artifact} contains an absolute path")
    if "Traceback (most recent call last)" in encoded:
        raise FreezeBuildError(f"public artifact {artifact} contains a stack trace")


def _verify_manifest(package: Path) -> tuple[dict[str, Any], str]:
    manifest_path = package / "manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_PACKAGE_FILES:
        raise FreezeBuildError(
            f"{package.name}: manifest must declare exactly the six package artifacts"
        )
    for name in sorted(REQUIRED_PACKAGE_FILES):
        metadata = files.get(name)
        path = package / name
        if not isinstance(metadata, dict) or not path.is_file():
            raise FreezeBuildError(f"{package.name}: missing manifest artifact {name}")
        expected_bytes = metadata.get("bytes")
        expected_sha256 = metadata.get("sha256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise FreezeBuildError(f"{package.name}: invalid byte count for {name}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise FreezeBuildError(f"{package.name}: invalid SHA256 for {name}")
        if path.stat().st_size != expected_bytes:
            raise FreezeBuildError(f"{package.name}: byte count changed for {name}")
        # Private files are hashed as opaque bytes here; they are never parsed.
        if _file_sha256(path) != expected_sha256:
            raise FreezeBuildError(f"{package.name}: SHA256 changed for {name}")

    required_manifest_values = {
        "judge_schema": JUDGE_SCHEMA,
        "judge_question_role": JUDGE_QUESTION_ROLE,
        "judge_mode": "all",
        "judge_depth": 2,
        "traffic_origin": "agent_generated_reference_probe",
        "release_status": RELEASE_STATUS,
    }
    for field, expected in required_manifest_values.items():
        if manifest.get(field) != expected:
            raise FreezeBuildError(
                f"{package.name}: manifest {field} must be {expected!r}"
            )
    lineage = manifest.get("rejudge_lineage")
    if not isinstance(lineage, dict) or any((
        lineage.get("operation") != "judge_only_replay",
        lineage.get("planner_rerun") is not False,
        lineage.get("retrieval_rerun") is not False,
        lineage.get("source_judge_schema") != "answerability-v3",
        lineage.get("target_judge_schema") != JUDGE_SCHEMA,
        lineage.get("judge_question_role") != JUDGE_QUESTION_ROLE,
    )):
        raise FreezeBuildError(f"{package.name}: invalid v4 rejudge lineage")
    # `seed_lineage.reason` is a bounded provenance enum from the producer,
    # not free-form judge reasoning.  Exempt only that exact manifest location.
    manifest_for_scan = json.loads(json.dumps(manifest))
    if isinstance(manifest_for_scan.get("seed_lineage"), dict):
        manifest_for_scan["seed_lineage"].pop("reason", None)
    _scan_public_payload(
        manifest_for_scan, artifact=f"{package.name}/manifest.json",
    )
    return manifest, _file_sha256(manifest_path)


def _expected_relation(verdict: dict[str, Any]) -> str | None:
    grade = verdict.get("grade")
    question_form = verdict.get("question_form")
    direct_answer = verdict.get("direct_answer")
    premise_status = verdict.get("premise_status")
    if question_form == "polar":
        return {
            ("proposition_true", "supported"): "entails",
            ("proposition_false", "refuted"): "contradicts",
            ("unknown", "not_established"): "not_established",
        }.get((direct_answer, premise_status))
    if question_form != "open" or direct_answer != "not_applicable":
        return None
    if premise_status == "supported":
        return "entails"
    if premise_status == "refuted":
        return "contradicts"
    if premise_status == "not_established":
        return "not_established"
    if premise_status == "none":
        return "entails" if grade == 3 else "not_established"
    return None


def _expected_action(verdict: dict[str, Any]) -> str:
    if verdict["grade"] < 3:
        return "abstain"
    return {
        "entails": "answer",
        "contradicts": "correct_premise",
        "not_established": "abstain",
    }[verdict["relation"]]


def _expected_query_label(row: dict[str, Any]) -> tuple[str, str, int | None]:
    usable = [
        verdict for verdict in row["verdicts"]
        if verdict["grade"] >= 3
        and verdict["action"] in {"answer", "correct_premise"}
    ]
    selected = sorted(
        usable, key=lambda verdict: (-verdict["grade"], verdict["rank"]),
    )[0] if usable else None
    recommended = selected["action"] if selected else "abstain"
    selected_rank = selected["rank"] if selected else None
    if row["gate_decision"]:
        if recommended == "abstain":
            intervention = "block_unsupported"
        elif recommended == "correct_premise":
            intervention = "correct_premise"
        elif selected_rank != 1:
            intervention = "rerank_evidence"
        else:
            intervention = "no_change"
    elif recommended == "answer":
        intervention = "rescue_answer"
    elif recommended == "correct_premise":
        intervention = "rescue_correction"
    else:
        intervention = "no_change"
    return recommended, intervention, selected_rank


def _validate_calibration_row(row: dict[str, Any], *, package_name: str, line: int) -> None:
    prefix = f"{package_name}/claude_agent_calibration.jsonl:{line}"
    if row.get("distribution_eligible") is not False:
        raise FreezeBuildError(f"{prefix}: probe row is distribution eligible")
    required_values = {
        "traffic_origin": "agent_generated_reference_probe",
        "judge_schema": JUDGE_SCHEMA,
        "judge_question_role": JUDGE_QUESTION_ROLE,
        "judge_selected": True,
        "label_available": True,
        "judge_depth": 2,
    }
    for field, expected in required_values.items():
        if row.get(field) != expected:
            raise FreezeBuildError(f"{prefix}: {field} must be {expected!r}")
    features = row.get("features")
    if not isinstance(features, dict):
        raise FreezeBuildError(f"{prefix}: features is not an object")
    for feature in ALLOWED_FEATURES:
        value = features.get(feature)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FreezeBuildError(f"{prefix}: invalid feature {feature}")
    if isinstance(features["high_scorers"], float) and not features["high_scorers"].is_integer():
        raise FreezeBuildError(f"{prefix}: high_scorers is not integral")

    verdicts = row.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != 2:
        raise FreezeBuildError(f"{prefix}: expected exactly Top2 verdicts")
    if [verdict.get("rank") for verdict in verdicts] != [1, 2]:
        raise FreezeBuildError(f"{prefix}: verdict ranks are not exactly [1, 2]")
    for verdict in verdicts:
        if verdict.get("available") is not True:
            raise FreezeBuildError(f"{prefix}: judge unavailable at rank {verdict.get('rank')}")
        grade = verdict.get("grade")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade not in range(4):
            raise FreezeBuildError(f"{prefix}: invalid grade at rank {verdict.get('rank')}")
        if verdict.get("question_form") not in QUESTION_FORMS:
            raise FreezeBuildError(f"{prefix}: invalid question_form")
        if verdict.get("direct_answer") not in DIRECT_ANSWERS:
            raise FreezeBuildError(f"{prefix}: invalid direct_answer")
        if verdict.get("premise_status") not in PREMISE_STATUSES:
            raise FreezeBuildError(f"{prefix}: invalid premise_status")
        if verdict.get("relation") not in RELATIONS:
            raise FreezeBuildError(f"{prefix}: invalid relation")
        if verdict.get("action") not in ACTIONS:
            raise FreezeBuildError(f"{prefix}: invalid action")
        if verdict["relation"] != _expected_relation(verdict):
            raise FreezeBuildError(f"{prefix}: v4 enum/relation mismatch")
        if verdict["action"] != _expected_action(verdict):
            raise FreezeBuildError(f"{prefix}: verdict action mismatch")
        chunk_hash = verdict.get("chunk_id_sha256")
        if not isinstance(chunk_hash, str) or len(chunk_hash) != 64:
            raise FreezeBuildError(f"{prefix}: invalid chunk identity hash")

    recommended, intervention, selected_rank = _expected_query_label(row)
    if (
        row.get("recommended_action"),
        row.get("intervention_type"),
        row.get("selected_rank"),
    ) != (recommended, intervention, selected_rank):
        raise FreezeBuildError(f"{prefix}: query-level label is inconsistent")
    if row.get("intervention_needed") != (intervention != "no_change"):
        raise FreezeBuildError(f"{prefix}: intervention_needed is inconsistent")


def load_public_package(package: Path, *, kind: str) -> dict[str, Any]:
    package = package.expanduser().resolve()
    if not package.is_dir():
        raise FreezeBuildError(f"input package does not exist: {package}")
    manifest, manifest_sha256 = _verify_manifest(package)
    if kind == "mixed":
        if manifest.get("probe_profile") != "mixed":
            raise FreezeBuildError(f"{package.name}: expected mixed probe profile")
        if manifest.get("probe_lineage", {}).get("document_text_visible") is not False:
            raise FreezeBuildError(f"{package.name}: mixed probe was not metadata-only")
    elif kind == "evidence":
        lineage = manifest.get("probe_lineage", {})
        if manifest.get("probe_profile") != "evidence_mixed":
            raise FreezeBuildError(f"{package.name}: expected evidence_mixed profile")
        if lineage.get("evidence_seeded") is not True:
            raise FreezeBuildError(f"{package.name}: evidence probe lacks evidence_seeded lineage")
    else:
        raise FreezeBuildError(f"unknown package kind: {kind}")

    routes = _read_jsonl(package / "route_outcomes.jsonl")
    calibration = _read_jsonl(package / "claude_agent_calibration.jsonl")
    summary = _read_json(package / "summary.json")
    for artifact, value in (
        ("route_outcomes.jsonl", routes),
        ("claude_agent_calibration.jsonl", calibration),
        ("summary.json", summary),
    ):
        _scan_public_payload(value, artifact=f"{package.name}/{artifact}")
    handoff = (package / "CLAUDE_HANDOFF.md").read_text(encoding="utf-8")
    if re.search(r"(?:/Users/|/home/|[A-Za-z]:\\\\)", handoff):
        raise FreezeBuildError(f"{package.name}: public handoff contains an absolute path")
    if "Traceback (most recent call last)" in handoff:
        raise FreezeBuildError(f"{package.name}: public handoff contains a stack trace")

    route_ids: set[str] = set()
    for line, route in enumerate(routes, 1):
        traffic_id = route.get("traffic_id")
        if not isinstance(traffic_id, str) or not traffic_id or traffic_id in route_ids:
            raise FreezeBuildError(f"{package.name}: invalid/duplicate route traffic_id at {line}")
        route_ids.add(traffic_id)
        if route.get("distribution_eligible") is not False:
            raise FreezeBuildError(f"{package.name}: route row is distribution eligible")
    calibration_ids: set[str] = set()
    for line, row in enumerate(calibration, 1):
        _validate_calibration_row(row, package_name=package.name, line=line)
        traffic_id = row.get("traffic_id")
        if not isinstance(traffic_id, str) or traffic_id in calibration_ids:
            raise FreezeBuildError(f"{package.name}: invalid/duplicate calibration traffic_id")
        if traffic_id not in route_ids:
            raise FreezeBuildError(f"{package.name}: calibration row has no public route parent")
        calibration_ids.add(traffic_id)
    if summary.get("generated_queries") != len(routes):
        raise FreezeBuildError(f"{package.name}: summary generated count mismatch")
    if summary.get("reference_kb_queries") != len(calibration):
        raise FreezeBuildError(f"{package.name}: summary reference count mismatch")
    if summary.get("judge_run", {}).get("judge_unavailable") != 0:
        raise FreezeBuildError(f"{package.name}: summary reports judge unavailable")

    return {
        "kind": kind,
        "package": _relative_name(package),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "routes": routes,
        "calibration": calibration,
        "summary": summary,
        "private_files_parsed": False,
    }


def _common_lineage(packages: list[dict[str, Any]]) -> dict[str, str]:
    fields = (
        "index_fingerprint", "reranker_requested", "reranker_actual",
        "reranker_status", "retrieval_profile", "judge_model_requested",
    )
    lineage: dict[str, str] = {}
    all_rows = [row for package in packages for row in package["calibration"]]
    if not all_rows:
        raise FreezeBuildError("input packages contain no calibration rows")
    for field in fields:
        values = {str(row.get(field) or "") for row in all_rows}
        if len(values) != 1 or not next(iter(values)):
            raise FreezeBuildError(f"input packages do not share one non-empty {field}")
        lineage[field] = next(iter(values))
    return lineage


def _rule_matches(name: str, features: dict[str, Any]) -> bool:
    if name == "SAT":
        return (
            features["rerank_margin"] < 0.01
            and features["rerank_top"] >= 0.99
            and features["high_scorers"] == 5
        )
    if name == "H1":
        return features["rerank_margin"] >= 0.01
    if name == "H2":
        return (
            features["high_scorers"] <= 4
            and features["rerank_margin"] >= 0.01
        )
    if name == "H3":
        return (
            features["gate_score_minus_threshold"] <= 0.14
            and features["rerank_top"] <= 0.99
        )
    raise FreezeBuildError(f"unknown frozen rule {name}")


def evaluate_mixed_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        row for row in rows if row["gate_decision"] and row["label_available"]
    ]
    interventions = [
        row for row in accepted if row["intervention_type"] in ACCEPTED_INTERVENTIONS
    ]
    no_change = [row for row in accepted if row["intervention_type"] == "no_change"]
    if len(interventions) + len(no_change) != len(accepted):
        raise FreezeBuildError("mixed accepted path contains an unknown label")
    results: dict[str, Any] = {}
    for name in RULES:
        results[name] = {
            "triggered_intervention": sum(
                _rule_matches(name, row["features"]) for row in interventions
            ),
            "total_intervention": len(interventions),
            "triggered_no_change": sum(
                _rule_matches(name, row["features"]) for row in no_change
            ),
            "total_no_change": len(no_change),
        }
    return {
        "accepted_rows": len(accepted),
        "intervention_rows": len(interventions),
        "no_change_rows": len(no_change),
        "intervention_type_counts": dict(sorted(Counter(
            row["intervention_type"] for row in accepted
        ).items())),
        "rules": results,
    }


def evidence_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row.get("probe_type") == "evidence_positive"]
    wrong_relations = [
        row for row in rows if row.get("probe_type") == "evidence_wrong_relation"
    ]
    if not positives or not wrong_relations:
        raise FreezeBuildError("evidence package lacks required positive/wrong-relation strata")
    positive_answer = sum(row["recommended_action"] == "answer" for row in positives)
    wrong_corrections = sum(
        row["recommended_action"] == "correct_premise" for row in wrong_relations
    )
    strict_refutations = sum(any(
        verdict["grade"] == 3
        and verdict["relation"] == "contradicts"
        and verdict["question_form"] == "polar"
        and verdict["direct_answer"] == "proposition_false"
        and verdict["premise_status"] == "refuted"
        for verdict in row["verdicts"]
    ) for row in wrong_relations)
    rerank_evidence = sum(
        row["intervention_type"] == "rerank_evidence" for row in rows
    )
    passed = (
        positive_answer == len(positives)
        and wrong_corrections == len(wrong_relations)
        and strict_refutations == len(wrong_relations)
        and rerank_evidence >= 1
    )
    if not passed:
        raise FreezeBuildError("evidence-seeded v4 contract did not pass")
    return {
        "passed": True,
        "evidence_positive": len(positives),
        "positive_recommended_answer": positive_answer,
        "evidence_wrong_relation": len(wrong_relations),
        "wrong_relation_recommended_correction": wrong_corrections,
        "wrong_relation_strict_refutation": strict_refutations,
        "rerank_evidence_rows": rerank_evidence,
    }


def build_protocol(
    *, mixed_manifest_sha256: str, evidence_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": "answerability-organic-v4",
        "purpose": "select_one_frozen_candidate_for_blind_window_b",
        "input_manifest_sha256": {
            "mixed_v4": mixed_manifest_sha256,
            "evidence_v4": evidence_manifest_sha256,
        },
        "traffic": {
            "origin": "organic",
            "route_source": "reference_kb",
            "distribution_eligible": True,
            "unit": "request",
            "deduplicate": False,
        },
        "window_a": {
            "sampling_mode": "accepted_control",
            "eligible_requests": 200,
            "selection": "first_200_after_protocol_activation",
            "gate_accept": {
                "judge_probability": 1.0,
                "judge_depth": 2,
            },
            "gate_reject_control": {
                "judge_probability": 0.05,
                "judge_depth": 2,
                "sampling": "runtime_bernoulli_rng",
                "required_row_fields": [
                    "sampling_probability", "judge_selected",
                ],
            },
            "judge_schema": JUDGE_SCHEMA,
            "judge_question_role": JUDGE_QUESTION_ROLE,
            "answer_path_effect": "record_only",
        },
        "allowed_features": list(ALLOWED_FEATURES),
        "forbidden_predictor_fields": [
            "grade", "relation", "action", "question_form", "direct_answer",
            "premise_status", "recommended_action", "intervention_type",
            "judge latency", "question text", "document text",
        ],
        "labels": {
            "population": "gate_decision=true AND label_available=true",
            "accepted_intervention": sorted(ACCEPTED_INTERVENTIONS),
            "accepted_no_change": ["no_change"],
            "correct_premise": ["correct_premise"],
            "judge_unavailable": "unlabeled_and_reported_not_imputed",
        },
        "candidate_rules": {
            name: {
                "description": rule["description"],
                "expression": rule["expression"],
                "conditions": rule["conditions"],
                "condition_count": rule["condition_count"],
                "selection_eligible": rule["selection_eligible"],
            }
            for name, rule in RULES.items()
        },
        "minimum_class_counts": {
            "accepted_intervention": 20,
            "accepted_no_change": 20,
            "if_below_either": "underpowered_stop_without_rule_selection",
        },
        "feasibility_point_estimates": {
            "accepted_intervention_recall_gte": 0.70,
            "accepted_no_change_trigger_rate_lte": 0.25,
            "conditional_reference_call_rate_lte": 0.30,
            "conditional_reference_call_rate_formula": (
                "gate_true_rule_hits / 200 eligible_reference_requests"
            ),
            "gate_reject_control_cost_role": (
                "window_measurement_cost_reported_separately_not_candidate_call_rate"
            ),
            "correct_premise": {
                "when_count_gte": 10,
                "recall_gte": 0.80,
                "otherwise": "descriptive_only_not_a_feasibility_gate",
            },
        },
        "selection_algorithm": {
            "eligible_rules": ["H1", "H2", "H3"],
            "SAT_role": "diagnostic_only_never_selectable",
            "if_no_feasible_rule": "stop_no_selection",
            "lexicographic_order": [
                "lowest_conditional_reference_call_rate",
                "highest_accepted_intervention_recall",
                "fewest_conditions",
                "fixed_priority_H1_then_H2_then_H3",
            ],
            "threshold_search_or_rule_editing": "forbidden",
        },
        "window_b": {
            "eligible_requests": 200,
            "selection": "next_200_after_window_a",
            "overlap_with_window_a": "forbidden",
            "precondition": "freeze_exactly_one_rule_before_inspecting_window_b_labels",
            "purpose": "blind_release_validation_only",
            "post_label_rule_changes": "forbidden",
        },
        "excluded_sets": [
            "Dev80", "V1", "Dev-New", "final90",
            "reference_probe_mixed16_v3", "reference_probe_evidence12_v3",
            "reference_probe_mixed16_v4", "reference_probe_evidence12_v4",
            "all_agent_generated_or_stress_traffic",
        ],
}


def _candidate_table(candidate_eval: dict[str, Any]) -> str:
    lines = [
        "| Rule | intervention | no_change | selection role |",
        "|---|---:|---:|---|",
    ]
    for name, rule in RULES.items():
        counts = candidate_eval["rules"][name]
        role = "diagnostic only" if not rule["selection_eligible"] else "Window A candidate"
        lines.append(
            f"| {name} | {counts['triggered_intervention']} / "
            f"{counts['total_intervention']} | {counts['triggered_no_change']} / "
            f"{counts['total_no_change']} | {role} |"
        )
    return "\n".join(lines)


def _input_audit_markdown(
    mixed: dict[str, Any], evidence: dict[str, Any], common: dict[str, str],
    evidence_check: dict[str, Any],
) -> str:
    return f"""# INPUT_AUDIT — v4 candidate-set freeze

All required checks passed.

| Check | Result |
|---|---|
| Both manifests and all six declared artifacts match bytes + SHA256 | PASS |
| Public JSON contains no raw question/evidence/chunk/reason/error fields | PASS |
| Judge contract is `{JUDGE_SCHEMA}` + `{JUDGE_QUESTION_ROLE}` | PASS |
| Every calibration row has exactly two available, enum-consistent verdicts | PASS |
| Every route and calibration row has `distribution_eligible=false` | PASS |
| Both packages share one index/retriever/reranker lineage | PASS |
| Private/generated JSONL content was not parsed | PASS |

## Inputs

- mixed: `{mixed['package']}`; manifest `{mixed['manifest_sha256']}`;
  {len(mixed['routes'])} routes / {len(mixed['calibration'])} reference rows.
- evidence: `{evidence['package']}`; manifest `{evidence['manifest_sha256']}`;
  {len(evidence['routes'])} routes / {len(evidence['calibration'])} reference rows.

Opaque-only artifacts in both packages: `private_audit.jsonl` and
`generated_queries_private.jsonl`. Their bytes and SHA256 were verified against
the manifest, but their content was never decoded.

## Common retrieval lineage

- index fingerprint: `{common['index_fingerprint']}`
- reranker requested/actual: `{common['reranker_requested']}` /
  `{common['reranker_actual']}`
- reranker status and retrieval profile: `{common['reranker_status']}` /
  `{common['retrieval_profile']}`
- judge model: `{common['judge_model_requested']}`

## Evidence-only contract check

- positive probes answered: {evidence_check['positive_recommended_answer']} /
  {evidence_check['evidence_positive']}
- wrong-relation probes corrected with strict v4 refutation:
  {evidence_check['wrong_relation_strict_refutation']} /
  {evidence_check['evidence_wrong_relation']}
- Top2 identity changed the usable evidence on
  {evidence_check['rerank_evidence_rows']} row(s).

The evidence package is contract evidence only. It is not used for candidate
counts, threshold choice, prevalence, or call-rate estimation.
"""


def _candidate_freeze_markdown(candidate_eval: dict[str, Any]) -> str:
    return f"""# CANDIDATE_SET_FREEZE — thresholds fixed before organic Window A

This freezes a **candidate set**, not a production rule. The v3 labels and the
analysis fitted to them are stale and superseded by v4; they must not be mixed
into this count or used to restore an older threshold conclusion.

Fitting/diagnostic population here is limited to the metadata-only mixed v4
package's accepted path: `gate_decision=true AND label_available=true`.
It contains {candidate_eval['accepted_rows']} rows:
{candidate_eval['intervention_rows']} interventions and
{candidate_eval['no_change_rows']} `no_change` controls.

{_candidate_table(candidate_eval)}

Frozen definitions:

- SAT: `{RULES['SAT']['expression']}`. Legacy saturation diagnostic only;
  it can never be selected.
- H1: `{RULES['H1']['expression']}`.
- H2: `{RULES['H2']['expression']}`.
- H3: `{RULES['H3']['expression']}`.

Only `{', '.join(ALLOWED_FEATURES)}` may be used. No threshold search,
feature addition, conjunction change, label-driven rewrite, or v3 fallback is
allowed after Window A begins.

These synthetic counts do not estimate organic prevalence, recall, false-call
rate, or production fitness. Window A applies the already-frozen candidates to
new organic traffic and uses only the preregistered selection algorithm.
"""


def _preregistration_markdown(protocol: dict[str, Any], protocol_sha256: str) -> str:
    return f"""# ORGANIC_WINDOW_A_PREREGISTRATION

Machine-readable source of truth:
`ORGANIC_WINDOW_A_PROTOCOL.json` (SHA256 `{protocol_sha256}`). Production
capture must set protocol ID `answerability-organic-v4` and
`RAG_SHADOW_PROTOCOL_SHA256` to that exact value.

## Population and sampling

- Take the first **200** requests after activation satisfying all three:
  organic traffic, `reference_kb` route, `distribution_eligible=true`.
- Count requests, not unique question strings; record each request once.
- `gate_decision=true`: judge every request, exactly Top2.
- `gate_decision=false`: runtime Bernoulli 5% control, exactly Top2 when
  selected; every row records `sampling_probability` and `judge_selected`.
- Judge asynchronously in record-only mode. It must not alter the user answer.
- Judge `{JUDGE_QUESTION_ROLE}` under `{JUDGE_SCHEMA}`; the retrieval subquery
  may define evidence scope but may not replace the original proposition.

Window A's deliberately high all-accepted measurement cost and the 5% rejected
control are not the proposed deployment call rate. Candidate feasibility uses
the frozen counterfactual formula `gate_true_rule_hits / 200`. Report actual
measurement calls and rejected-control calls separately.

## Labels and minimum power

Accepted intervention is exactly
`block_unsupported | correct_premise | rerank_evidence`; the accepted control is
`no_change`. If either class has fewer than **20** labeled rows, declare
`underpowered` and select no rule.

Judge-unavailable rows are reported and left unlabeled; they are never imputed
as either class.

## Point-estimate feasibility gates

Each of H1/H2/H3 must simultaneously meet:

- accepted intervention recall >= 0.70;
- accepted no-change trigger rate <= 0.25;
- conditional reference call rate <= 0.30.

If at least 10 accepted `correct_premise` rows exist, its recall must also be
>= 0.80. Below 10, report it descriptively without making it a feasibility
gate.

SAT is always reported but cannot be selected.

## Frozen selection algorithm

If no rule is feasible, stop. Otherwise select exactly one feasible rule by:

1. lowest conditional reference call rate;
2. then highest accepted intervention recall;
3. then fewer conditions;
4. then fixed priority H1 > H2 > H3.

No new threshold, feature, rule, or tie-break may be introduced.

## Window B

Freeze the selected rule before reading any Window B labels. Window B is the
next 200 eligible organic reference requests, with zero overlap with Window A.
It is blind release validation only; a failure cannot be repaired by changing
the rule on B.

The excluded sets listed in the protocol—including Dev80, V1, Dev-New,
final90, all v3/v4 probes, agent traffic, and stress traffic—cannot enter A or B.
"""


def _claude_handoff(protocol_sha256: str) -> str:
    return f"""# CLAUDE_HANDOFF — two-stage audit; do not redesign the protocol

The candidate set and selection protocol are frozen. The authoritative machine
protocol is `ORGANIC_WINDOW_A_PROTOCOL.json`, SHA256 `{protocol_sha256}`.
Its exact protocol ID is `answerability-organic-v4`.

## Stage 1 — immediate task, before Window A opens

There is no organic Window A dataset yet. Audit only:

1. the two v4 public packages and their input manifest hashes;
2. the frozen SAT/H1/H2/H3 definitions and v4-only candidate counts;
3. the machine protocol against the human preregistration;
4. the runtime code/config wiring for protocol ID `answerability-organic-v4`,
   `RAG_SHADOW_PROTOCOL_SHA256`,
   `accepted_control`, all gate accepts, runtime-Bernoulli 5% rejected control,
   original-user-question v4 Top2 judging, and record-only behavior.

Return exactly `GO_OPEN_WINDOW_A` or `NO_GO_PROTOCOL`, with evidence. Do not
pretend the first 200 requests already exist.

## Stage 2 — future task, only after Window A reaches 200 eligible requests

Then, and only then:

1. verify the captured run used the exact protocol hash and the first 200
   eligible organic `reference_kb` requests;
2. verify all gate accepts and the recorded runtime-Bernoulli 5% gate-reject
   control were sent to the v4 Top2 judge, record-only;
3. compute SAT/H1/H2/H3 using their exact frozen conditions;
4. apply the minimum class counts, feasibility gates, and lexicographic
   selection algorithm exactly as written;
5. return `NO_GO_UNDERPOWERED`, `NO_GO_NO_FEASIBLE_RULE`, or one uniquely
   selected H1/H2/H3 rule for freezing before Window B.

At either stage, do not search thresholds, add features, rewrite a condition,
use evidence-probe rows for fitting, or use v3 labels. The v3 analysis is stale.
A Stage 2 selection only authorizes freezing one rule and collecting blind
Window B; it does not authorize production enablement.
"""


def build_handoff(
    mixed_package: Path, evidence_package: Path, output_dir: Path,
) -> dict[str, Any]:
    mixed_path = mixed_package.expanduser().resolve()
    evidence_path = evidence_package.expanduser().resolve()
    output_path = output_dir.expanduser().resolve()
    if mixed_path == evidence_path:
        raise FreezeBuildError("mixed and evidence inputs must be different packages")
    if output_path.exists():
        raise FreezeBuildError(f"output directory already exists: {output_path}")
    if output_path.is_relative_to(mixed_path) or output_path.is_relative_to(evidence_path):
        raise FreezeBuildError("output directory cannot be inside an input package")

    mixed = load_public_package(mixed_path, kind="mixed")
    evidence = load_public_package(evidence_path, kind="evidence")
    common = _common_lineage([mixed, evidence])
    candidate_eval = evaluate_mixed_candidates(mixed["calibration"])
    evidence_check = evidence_contract(evidence["calibration"])
    protocol = build_protocol(
        mixed_manifest_sha256=mixed["manifest_sha256"],
        evidence_manifest_sha256=evidence["manifest_sha256"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_path.name}.staging-", dir=output_path.parent,
    ))
    try:
        protocol_path = staging / "ORGANIC_WINDOW_A_PROTOCOL.json"
        _json_dump(protocol_path, protocol)
        protocol_sha256 = _file_sha256(protocol_path)
        texts = {
            "INPUT_AUDIT.md": _input_audit_markdown(
                mixed, evidence, common, evidence_check,
            ),
            "CANDIDATE_SET_FREEZE.md": _candidate_freeze_markdown(candidate_eval),
            "ORGANIC_WINDOW_A_PREREGISTRATION.md": _preregistration_markdown(
                protocol, protocol_sha256,
            ),
            "CLAUDE_HANDOFF.md": _claude_handoff(protocol_sha256),
        }
        for name, text in texts.items():
            (staging / name).write_text(text, encoding="utf-8")

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        input_record = {
            "mixed_v4": {
                "package": mixed["package"],
                "manifest_sha256": mixed["manifest_sha256"],
                "routes": len(mixed["routes"]),
                "reference_rows": len(mixed["calibration"]),
                "private_files_parsed": False,
            },
            "evidence_v4": {
                "package": evidence["package"],
                "manifest_sha256": evidence["manifest_sha256"],
                "routes": len(evidence["routes"]),
                "reference_rows": len(evidence["calibration"]),
                "private_files_parsed": False,
            },
        }
        analysis_manifest = {
            "schema_version": ANALYSIS_SCHEMA,
            "created_at": now,
            "conclusion": "candidate_set_frozen_for_organic_window_a",
            "production_rule_selected": False,
            "v3_analysis_status": "stale_superseded",
            "git": _git_state(),
            "builder_sha256": _file_sha256(Path(__file__).resolve()),
            "judge_schema": JUDGE_SCHEMA,
            "judge_question_role": JUDGE_QUESTION_ROLE,
            "input_checks": {
                "all_checks_passed": True,
                "manifest_six_file_hashes_match": True,
                "public_privacy_scan_passed": True,
                "exactly_top2_available": True,
                "v4_enum_relation_action_consistent": True,
                "distribution_eligible_all_false": True,
                "shared_index_reranker_lineage": True,
                "private_generated_content_parsed": False,
                "evidence_contract_passed": True,
            },
            "inputs": input_record,
            "common_lineage": common,
            "mixed_accepted_path": candidate_eval,
            "evidence_contract": evidence_check,
            "protocol_sha256": protocol_sha256,
            "runtime_binding": {
                "protocol_id": "answerability-organic-v4",
                "environment_variable": "RAG_SHADOW_PROTOCOL_SHA256",
                "required_value": protocol_sha256,
            },
            "outputs": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": _file_sha256(staging / name),
                }
                for name in OUTPUT_FILES
            },
        }
        manifest_path = staging / "analysis_manifest.json"
        _json_dump(manifest_path, analysis_manifest)
        if output_path.exists():
            raise FreezeBuildError(f"output directory appeared before publish: {output_path}")
        os.replace(staging, output_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_path),
        "analysis_manifest": str(output_path / "analysis_manifest.json"),
        "protocol": str(output_path / "ORGANIC_WINDOW_A_PROTOCOL.json"),
        "protocol_sha256": protocol_sha256,
        "candidate_counts": candidate_eval,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-package", type=Path, required=True)
    parser.add_argument("--evidence-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_handoff(
            args.mixed_package, args.evidence_package, args.output_dir,
        )
    except FreezeBuildError as exc:
        parser.exit(2, f"freeze refused: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
