#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Organic Window A and execute its preregistered Stage 2 decision.

The protocol JSON is the source of truth.  This script never reads raw query or
document text, never searches thresholds, and evaluates only the first frozen
number of eligible requests in file order.  It can be run repeatedly in status
mode while the window is collecting; final publication is atomic and refuses
to overwrite an existing result directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


WINDOW_SCHEMA = "shadow-answerability-v3"
JUDGE_SCHEMA = "answerability-v4"
JUDGE_QUESTION_ROLE = "original_user_question"
ALLOWED_ACTIONS = {"answer", "correct_premise", "abstain"}
INTERVENTIONS = {"block_unsupported", "correct_premise", "rerank_evidence"}
PRIVATE_KEYS = {
    "question", "retrieval_question", "document", "chunk_id", "reason",
    "error", "heading_path", "candidates_private",
}
ABSOLUTE_PATH = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\)")


class WindowAuditError(RuntimeError):
    """A frozen-window integrity or protocol contract failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowAuditError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WindowAuditError(f"{path} is not a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WindowAuditError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WindowAuditError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise WindowAuditError(f"non-object row at {path}:{line_number}")
        rows.append(row)
    return rows


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _walk_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _walk_keys(child)}
    return set()


def _privacy_check(row: dict[str, Any], line_number: int) -> None:
    forbidden = _walk_keys(row) & PRIVATE_KEYS
    if forbidden:
        raise WindowAuditError(
            f"row {line_number} contains private/free-text keys: "
            + ", ".join(sorted(forbidden))
        )
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if ABSOLUTE_PATH.search(encoded):
        raise WindowAuditError(f"row {line_number} contains an absolute path")
    if "Traceback (most recent call last)" in encoded:
        raise WindowAuditError(f"row {line_number} contains a stack trace")


def _number(value: Any, *, field: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WindowAuditError(f"row {line_number} has invalid {field}")
    number = float(value)
    if not math.isfinite(number):
        raise WindowAuditError(f"row {line_number} has non-finite {field}")
    return number


def _validate_protocol(protocol: dict[str, Any], protocol_sha256: str) -> None:
    traffic = protocol.get("traffic") or {}
    window = protocol.get("window_a") or {}
    if protocol.get("protocol_id") != "answerability-organic-v4":
        raise WindowAuditError("unexpected protocol_id")
    if traffic != {
        "deduplicate": False,
        "distribution_eligible": True,
        "origin": "organic",
        "route_source": "reference_kb",
        "unit": "request",
    }:
        raise WindowAuditError("traffic contract changed")
    required = {
        "sampling_mode": "accepted_control",
        "eligible_requests": 200,
        "selection": "first_200_after_protocol_activation",
        "judge_schema": JUDGE_SCHEMA,
        "judge_question_role": JUDGE_QUESTION_ROLE,
        "answer_path_effect": "record_only",
    }
    for field, expected in required.items():
        if window.get(field) != expected:
            raise WindowAuditError(f"window_a.{field} changed")
    if window.get("gate_accept") != {"judge_probability": 1.0, "judge_depth": 2}:
        raise WindowAuditError("gate-accept sampling contract changed")
    reject = window.get("gate_reject_control") or {}
    if reject.get("judge_probability") != 0.05 or reject.get("judge_depth") != 2:
        raise WindowAuditError("gate-reject sampling contract changed")
    if reject.get("sampling") != "runtime_bernoulli_rng":
        raise WindowAuditError("gate-reject RNG contract changed")
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256):
        raise WindowAuditError("invalid protocol SHA256")
    allowed = set(protocol.get("allowed_features") or [])
    if allowed != {
        "rerank_margin", "rerank_top", "high_scorers",
        "gate_score_minus_threshold",
    }:
        raise WindowAuditError("allowed feature set changed")
    for rule_id, rule in (protocol.get("candidate_rules") or {}).items():
        for condition in rule.get("conditions") or []:
            if condition.get("feature") not in allowed:
                raise WindowAuditError(f"rule {rule_id} uses a forbidden feature")


def _expected_relation(verdict: dict[str, Any]) -> str | None:
    grade = verdict.get("grade")
    form = verdict.get("question_form")
    direct = verdict.get("direct_answer")
    premise = verdict.get("premise_status")
    if form == "polar":
        return {
            ("proposition_true", "supported"): "entails",
            ("proposition_false", "refuted"): "contradicts",
            ("unknown", "not_established"): "not_established",
        }.get((direct, premise))
    if form != "open" or direct != "not_applicable":
        return None
    if premise == "supported":
        return "entails"
    if premise == "refuted":
        return "contradicts"
    if premise == "not_established":
        return "not_established"
    if premise == "none":
        return "entails" if grade == 3 else "not_established"
    return None


def _validate_verdict(container: dict[str, Any], *, line_number: int, rank: int) -> None:
    if container.get("rank") != rank:
        raise WindowAuditError(f"row {line_number} does not contain exact Top2 ranks")
    chunk_hash = container.get("chunk_id_sha256")
    if not isinstance(chunk_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", chunk_hash):
        raise WindowAuditError(f"row {line_number} has invalid chunk hash")
    _number(container.get("rerank_score"), field="rerank_score", line_number=line_number)
    available = container.get("available")
    if not isinstance(available, bool):
        raise WindowAuditError(f"row {line_number} has invalid availability")
    if not available:
        if container.get("verdict") is not None:
            raise WindowAuditError(f"row {line_number} unavailable verdict is not null")
        if container.get("action") != "abstain":
            raise WindowAuditError(f"row {line_number} unavailable action is not abstain")
        return
    verdict = container.get("verdict")
    if not isinstance(verdict, dict):
        raise WindowAuditError(f"row {line_number} available verdict is absent")
    grade = verdict.get("grade")
    if isinstance(grade, bool) or not isinstance(grade, int) or grade not in range(4):
        raise WindowAuditError(f"row {line_number} has invalid grade")
    relation = _expected_relation(verdict)
    if relation is None or verdict.get("relation") != relation:
        raise WindowAuditError(f"row {line_number} has inconsistent v4 relation")
    expected_action = "abstain" if grade < 3 else {
        "entails": "answer",
        "contradicts": "correct_premise",
        "not_established": "abstain",
    }[relation]
    if container.get("action") != expected_action:
        raise WindowAuditError(f"row {line_number} has inconsistent action")


def _validate_row(
    row: dict[str, Any], *, line_number: int, protocol: dict[str, Any],
    protocol_sha256: str,
) -> None:
    _privacy_check(row, line_number)
    expected = {
        "schema_version": WINDOW_SCHEMA,
        "sampling_mode": "accepted_control",
        "sampling_config_valid": True,
        "window_phase": "calibration_a",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "judge_question_role": JUDGE_QUESTION_ROLE,
        "judge_schema": JUDGE_SCHEMA,
        "route": "reference_kb",
        "traffic_origin": "organic",
        "distribution_eligible": True,
        "judge_depth": 2,
        "queue_status": "enqueued",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise WindowAuditError(
                f"row {line_number} has {field}={row.get(field)!r}, expected {value!r}"
            )
    if row.get("sampling_config_errors") != []:
        raise WindowAuditError(f"row {line_number} contains sampling config errors")
    for field in ("original_question_sha256", "retrieval_question_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise WindowAuditError(f"row {line_number} has invalid {field}")
    features = row.get("features")
    if not isinstance(features, dict):
        raise WindowAuditError(f"row {line_number} has no feature object")
    for feature in protocol.get("allowed_features") or []:
        _number(features.get(feature), field=feature, line_number=line_number)
    if bool(features.get("gate_decision")) != bool(row.get("gate_decision")):
        raise WindowAuditError(f"row {line_number} gate decision disagrees with features")

    gate = bool(row["gate_decision"])
    selected = row.get("judge_selected")
    if not isinstance(selected, bool):
        raise WindowAuditError(f"row {line_number} has invalid judge_selected")
    if gate:
        wanted = (True, 1.0, 1.0, "all_gate_accepts")
    else:
        weight = 20.0 if selected else None
        wanted = (selected, 0.05, weight, "random_gate_reject_control")
    actual = (
        selected, row.get("sampling_probability"), row.get("sampling_weight"),
        row.get("sampling_reason"),
    )
    if actual != wanted:
        raise WindowAuditError(f"row {line_number} violates sampling contract")

    candidate_count = row.get("candidate_count")
    verdicts = row.get("verdicts")
    if not isinstance(verdicts, list):
        raise WindowAuditError(f"row {line_number} verdicts is not a list")
    if selected:
        if candidate_count != 2 or len(verdicts) != 2:
            raise WindowAuditError(f"row {line_number} was not sent as exact Top2")
        for rank, verdict in enumerate(verdicts, 1):
            _validate_verdict(verdict, line_number=line_number, rank=rank)
        expected_available = all(verdict.get("available") is True for verdict in verdicts)
        if row.get("label_available") is not expected_available:
            raise WindowAuditError(f"row {line_number} label availability is inconsistent")
        if row.get("top2_complete") is not expected_available:
            raise WindowAuditError(f"row {line_number} Top2 completeness is inconsistent")
    else:
        if gate or verdicts or row.get("label_available") or row.get("top2_complete"):
            raise WindowAuditError(f"row {line_number} unselected row contains a label")


def _query_label(row: dict[str, Any]) -> tuple[str | None, int | None]:
    if not row.get("label_available"):
        return None, None
    usable = []
    for container in row["verdicts"]:
        verdict = container["verdict"]
        if verdict["grade"] >= 3 and container["action"] in {
            "answer", "correct_premise",
        }:
            usable.append((verdict["grade"], container["rank"], container["action"]))
    selected = sorted(usable, key=lambda item: (-item[0], item[1]))[0] if usable else None
    action = selected[2] if selected else "abstain"
    rank = selected[1] if selected else None
    if row["gate_decision"]:
        if action == "abstain":
            return "block_unsupported", rank
        if action == "correct_premise":
            return "correct_premise", rank
        if rank != 1:
            return "rerank_evidence", rank
        return "no_change", rank
    if action == "answer":
        return "rescue_answer", rank
    if action == "correct_premise":
        return "rescue_correction", rank
    return "no_change", rank


def _operator_match(value: float, operator: str, threshold: float) -> bool:
    return {
        "lt": value < threshold,
        "lte": value <= threshold,
        "eq": value == threshold,
        "gte": value >= threshold,
        "gt": value > threshold,
    }.get(operator, False)


def _rule_match(rule: dict[str, Any], features: dict[str, Any]) -> bool:
    conditions = rule.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise WindowAuditError("candidate rule has no conditions")
    for condition in conditions:
        if not isinstance(condition, dict):
            raise WindowAuditError("candidate condition is not an object")
        feature = condition.get("feature")
        operator = condition.get("operator")
        threshold = condition.get("value")
        value = features.get(feature)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WindowAuditError(f"missing numeric feature for rule: {feature}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise WindowAuditError(f"invalid threshold for rule: {feature}")
        if operator not in {"lt", "lte", "eq", "gte", "gt"}:
            raise WindowAuditError(f"invalid frozen operator: {operator}")
        if not _operator_match(float(value), operator, float(threshold)):
            return False
    return True


def _evaluate(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_accepted: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        label, _rank = _query_label(row)
        if row["gate_decision"] and label is not None:
            labeled_accepted.append((row, label))
    interventions = [item for item in labeled_accepted if item[1] in INTERVENTIONS]
    no_changes = [item for item in labeled_accepted if item[1] == "no_change"]
    corrections = [item for item in labeled_accepted if item[1] == "correct_premise"]
    gate_true = [row for row in rows if row["gate_decision"]]
    unavailable = sum(
        bool(row["judge_selected"] and not row["label_available"]) for row in rows
    )

    rules: dict[str, Any] = {}
    candidate_rules = protocol.get("candidate_rules") or {}
    for rule_id, rule in candidate_rules.items():
        hit_interventions = sum(_rule_match(rule, row["features"]) for row, _ in interventions)
        hit_no_changes = sum(_rule_match(rule, row["features"]) for row, _ in no_changes)
        hit_corrections = sum(_rule_match(rule, row["features"]) for row, _ in corrections)
        gate_true_hits = sum(_rule_match(rule, row["features"]) for row in gate_true)
        recall = hit_interventions / len(interventions) if interventions else None
        false_call = hit_no_changes / len(no_changes) if no_changes else None
        call_rate = gate_true_hits / len(rows)
        correction_recall = hit_corrections / len(corrections) if corrections else None
        rules[rule_id] = {
            "selection_eligible": bool(rule.get("selection_eligible")),
            "condition_count": int(rule.get("condition_count") or 0),
            "triggered_intervention": hit_interventions,
            "total_intervention": len(interventions),
            "accepted_intervention_recall": recall,
            "triggered_no_change": hit_no_changes,
            "total_no_change": len(no_changes),
            "accepted_no_change_trigger_rate": false_call,
            "gate_true_rule_hits": gate_true_hits,
            "conditional_reference_call_rate": call_rate,
            "triggered_correct_premise": hit_corrections,
            "total_correct_premise": len(corrections),
            "correct_premise_recall": correction_recall,
        }

    minimums = protocol["minimum_class_counts"]
    underpowered = (
        len(interventions) < minimums["accepted_intervention"]
        or len(no_changes) < minimums["accepted_no_change"]
    )
    selected_rule: str | None = None
    if underpowered:
        conclusion = "NO_GO_UNDERPOWERED"
    else:
        gates = protocol["feasibility_point_estimates"]
        feasible: list[str] = []
        for rule_id in protocol["selection_algorithm"]["eligible_rules"]:
            metrics = rules[rule_id]
            ok = (
                metrics["accepted_intervention_recall"] >= gates["accepted_intervention_recall_gte"]
                and metrics["accepted_no_change_trigger_rate"] <= gates["accepted_no_change_trigger_rate_lte"]
                and metrics["conditional_reference_call_rate"] <= gates["conditional_reference_call_rate_lte"]
            )
            correction_gate = gates["correct_premise"]
            if len(corrections) >= correction_gate["when_count_gte"]:
                ok = ok and metrics["correct_premise_recall"] >= correction_gate["recall_gte"]
            metrics["feasible"] = bool(ok)
            if ok:
                feasible.append(rule_id)
        if not feasible:
            conclusion = "NO_GO_NO_FEASIBLE_RULE"
        else:
            priority = {
                rule_id: index
                for index, rule_id in enumerate(
                    protocol["selection_algorithm"]["eligible_rules"]
                )
            }
            selected_rule = min(
                feasible,
                key=lambda rule_id: (
                    rules[rule_id]["conditional_reference_call_rate"],
                    -rules[rule_id]["accepted_intervention_recall"],
                    rules[rule_id]["condition_count"],
                    priority[rule_id],
                ),
            )
            conclusion = "SELECT_RULE"

    return {
        "conclusion": conclusion,
        "selected_rule": selected_rule,
        "window_rows": len(rows),
        "gate_true_rows": len(gate_true),
        "gate_false_rows": len(rows) - len(gate_true),
        "judge_unavailable_rows": unavailable,
        "labeled_accepted_rows": len(labeled_accepted),
        "accepted_intervention_rows": len(interventions),
        "accepted_no_change_rows": len(no_changes),
        "accepted_correct_premise_rows": len(corrections),
        "label_counts": dict(sorted(Counter(label for _, label in labeled_accepted).items())),
        "rules": rules,
    }


def analyze_window(
    *, protocol_path: Path, log_path: Path, drop_log_path: Path | None = None,
) -> dict[str, Any]:
    protocol_path = protocol_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    protocol = _read_json(protocol_path)
    protocol_sha256 = _sha256(protocol_path)
    _validate_protocol(protocol, protocol_sha256)
    all_rows = _read_jsonl(log_path)
    drops_path = (
        drop_log_path.expanduser().resolve()
        if drop_log_path is not None
        else log_path.with_name(f"{log_path.stem}.drops.jsonl")
    )
    drops = _read_jsonl(drops_path)
    for index, row in enumerate(all_rows, 1):
        _privacy_check(row, index)
    for index, row in enumerate(drops, 1):
        _privacy_check(row, index)
    target = int(protocol["window_a"]["eligible_requests"])
    used_rows = all_rows[:target]
    for index, row in enumerate(used_rows, 1):
        _validate_row(
            row, line_number=index, protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
    window_ids = {row.get("window_id") for row in used_rows}
    if len(window_ids) > 1:
        raise WindowAuditError("first window rows contain multiple window_id values")
    timestamps = [row.get("observed_at_ms") for row in used_rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in timestamps):
        raise WindowAuditError("window contains invalid observation timestamps")
    if timestamps != sorted(timestamps):
        raise WindowAuditError("window rows are not in observation order")
    lineage_fields = (
        "index_fingerprint", "retrieval_profile", "reranker_requested",
        "reranker_actual", "reranker_status",
    )
    common_lineage: dict[str, str] = {}
    for field in lineage_fields:
        values = {str(row.get(field) or "") for row in used_rows}
        if used_rows and (len(values) != 1 or not next(iter(values))):
            raise WindowAuditError(f"window does not have one non-empty {field}")
        common_lineage[field] = next(iter(values), "")

    base = {
        "schema_version": "answerability-organic-window-a-analysis-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "window_id": next(iter(window_ids), None),
        "eligible_target": target,
        "eligible_rows_total": len(all_rows),
        "eligible_rows_used": len(used_rows),
        "extra_rows_ignored": max(0, len(all_rows) - target),
        "queue_drop_rows": len(drops),
        "log_path": str(log_path),
        "drop_log_path": str(drops_path),
        "common_lineage": common_lineage,
    }
    if drops:
        return {**base, "conclusion": "NO_GO_WINDOW_INCOMPLETE"}
    if len(used_rows) < target:
        return {
            **base,
            "conclusion": "COLLECTING",
            "remaining": target - len(used_rows),
        }
    return {**base, **_evaluate(protocol, used_rows)}


def _safe_result(result: dict[str, Any]) -> dict[str, Any]:
    safe = dict(result)
    for key in ("log_path", "drop_log_path"):
        if key in safe:
            safe[key] = Path(str(safe[key])).name
    return safe


def _render_markdown(result: dict[str, Any]) -> str:
    def metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    lines = [
        "# Organic Window A — preregistered Stage 2 result",
        "",
        f"- conclusion: `{result['conclusion']}`",
        f"- protocol: `{result['protocol_id']}` / `{result['protocol_sha256']}`",
        f"- window: `{result.get('window_id')}`",
        f"- rows: {result['eligible_rows_used']} / {result['eligible_target']}",
        f"- queue drops: {result['queue_drop_rows']}",
    ]
    if result.get("selected_rule"):
        lines.append(f"- selected rule: `{result['selected_rule']}`")
    if result.get("rules"):
        lines.extend([
            "",
            "| Rule | intervention recall | no-change trigger | conditional call rate | feasible |",
            "|---|---:|---:|---:|---|",
        ])
        for rule_id, metrics in result["rules"].items():
            recall = metrics["accepted_intervention_recall"]
            false_call = metrics["accepted_no_change_trigger_rate"]
            call_rate = metrics["conditional_reference_call_rate"]
            lines.append(
                f"| {rule_id} | {metric(recall)} | {metric(false_call)} | "
                f"{metric(call_rate)} | {metrics.get('feasible', False)} |"
            )
    lines.extend([
        "",
        "This result applies the frozen JSON conditions and tie-break order. It",
        "does not authorize production enablement; a selected rule only authorizes",
        "freezing that rule before blind Window B.",
        "",
    ])
    return "\n".join(lines)


def publish_result(
    *, result: dict[str, Any], protocol: dict[str, Any], output_dir: Path,
) -> dict[str, Any]:
    if result["conclusion"] == "COLLECTING":
        raise WindowAuditError("cannot finalize before the window reaches its target")
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise WindowAuditError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        safe = _safe_result(result)
        audit_path = staging / "WINDOW_A_RESULT.json"
        audit_path.write_text(
            json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "WINDOW_A_RESULT.md").write_text(
            _render_markdown(safe), encoding="utf-8",
        )
        frozen_rule_name = None
        frozen_rule_sha256 = None
        if safe.get("selected_rule"):
            rule_id = safe["selected_rule"]
            frozen = {
                "schema_version": "answerability-frozen-trigger-rule-v1",
                "protocol_id": safe["protocol_id"],
                "protocol_sha256": safe["protocol_sha256"],
                "source_window_id": safe["window_id"],
                "rule_id": rule_id,
                "rule": protocol["candidate_rules"][rule_id],
                "window_a_metrics": safe["rules"][rule_id],
                "authorization": "collect_blind_window_b_only",
            }
            frozen_rule_name = "FROZEN_RULE.json"
            frozen_path = staging / frozen_rule_name
            frozen_path.write_text(
                json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            frozen_rule_sha256 = _sha256(frozen_path)
        files = {}
        for path in sorted(staging.iterdir()):
            files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        manifest = {
            "schema_version": "answerability-organic-window-a-result-manifest-v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "conclusion": safe["conclusion"],
            "selected_rule": safe.get("selected_rule"),
            "frozen_rule_file": frozen_rule_name,
            "frozen_rule_sha256": frozen_rule_sha256,
            "protocol_id": safe["protocol_id"],
            "protocol_sha256": safe["protocol_sha256"],
            "source_window_id": safe.get("window_id"),
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": str(output),
        "conclusion": result["conclusion"],
        "selected_rule": result.get("selected_rule"),
        "frozen_rule_sha256": frozen_rule_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--drop-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze_window(
            protocol_path=args.protocol,
            log_path=args.log,
            drop_log_path=args.drop_log,
        )
        if args.output_dir is not None:
            protocol = _read_json(args.protocol.expanduser().resolve())
            published = publish_result(
                result=result, protocol=protocol, output_dir=args.output_dir,
            )
            print(json.dumps(published, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(_safe_result(result), ensure_ascii=False, indent=2))
    except WindowAuditError as exc:
        parser.exit(2, f"window audit refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
