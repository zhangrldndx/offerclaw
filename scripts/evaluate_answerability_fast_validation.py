#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen answerability rules on blind agent and historical replay.

This is deliberately an offline release decision, not an estimator of future
organic-human prevalence.  Candidate expressions are loaded byte-for-byte from
the earlier frozen protocol; this script never searches or edits thresholds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package(
    package: Path, *, release_status: str, traffic_origin: str,
) -> dict[str, Any]:
    manifest_path = package / "manifest.json"
    manifest = _json(manifest_path)
    errors: list[str] = []
    if manifest.get("release_status") != release_status:
        errors.append("release_status")
    if manifest.get("traffic_origin") != traffic_origin:
        errors.append("traffic_origin")
    if manifest.get("judge_schema") != "answerability-v4":
        errors.append("judge_schema")
    if manifest.get("judge_question_role") != "original_user_question":
        errors.append("judge_question_role")
    for name, expected in (manifest.get("files") or {}).items():
        path = package / name
        if not path.is_file():
            errors.append(f"missing:{name}")
            continue
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"bytes:{name}")
        if _sha256(path) != expected.get("sha256"):
            errors.append(f"sha256:{name}")
    if errors:
        raise ValueError(f"invalid package {package}: {', '.join(errors)}")
    return {
        "path": str(package.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "random_seed": manifest.get("random_seed"),
        "files_verified": len(manifest.get("files") or {}),
    }


def rule_matches(features: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    operators = {
        "lt": lambda left, right: left < right,
        "lte": lambda left, right: left <= right,
        "gt": lambda left, right: left > right,
        "gte": lambda left, right: left >= right,
        "eq": lambda left, right: left == right,
    }
    for condition in conditions:
        feature = condition["feature"]
        if feature not in features or condition["operator"] not in operators:
            return False
        if not operators[condition["operator"]](features[feature], condition["value"]):
            return False
    return True


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)]


def _request_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject ambiguous duplicate captures instead of silently double-counting."""
    counts = Counter(row["traffic_id"] for row in rows)
    duplicates = [traffic_id for traffic_id, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"expected one reference capture per request; duplicates={len(duplicates)}")
    return rows


def evaluate(
    *, fast_protocol: dict[str, Any], frozen_protocol: dict[str, Any],
    agent_packages: list[Path], historical_package: Path,
) -> dict[str, Any]:
    agent_verified = [verify_package(
        path, release_status="agent_blind_rule_validation",
        traffic_origin="agent_generated_reference_probe",
    ) for path in agent_packages]
    historical_verified = verify_package(
        historical_package, release_status="historical_distribution_replay",
        traffic_origin="historical_replay",
    )
    agent_rows = _request_rows([
        row for package in agent_packages
        for row in _jsonl(package / "claude_agent_calibration.jsonl")
    ])
    historical_rows = _request_rows(_jsonl(
        historical_package / "claude_agent_calibration.jsonl"
    ))
    historical_routes = _jsonl(historical_package / "route_outcomes.jsonl")

    labels = fast_protocol["labels"]
    interventions = set(labels["accepted_intervention"])
    no_changes = set(labels["accepted_no_change"])
    accepted = [
        row for row in agent_rows
        if row.get("gate_decision") is True and row.get("label_available") is True
    ]
    intervention_rows = [row for row in accepted if row.get("intervention_type") in interventions]
    no_change_rows = [row for row in accepted if row.get("intervention_type") in no_changes]
    correct_premise_rows = [row for row in accepted if row.get("intervention_type") == "correct_premise"]

    minimums = fast_protocol["agent_blind"]["minimum_class_counts"]
    powered = (
        len(intervention_rows) >= minimums["accepted_intervention"]
        and len(no_change_rows) >= minimums["accepted_no_change"]
    )
    route_weights = {
        row["traffic_id"]: max(1, int(row.get("occurrence_count") or 1))
        for row in historical_routes
    }
    total_history_weight = sum(route_weights.values())
    reference_history_ids = {row["traffic_id"] for row in historical_rows}
    reference_history_weight = sum(route_weights[item] for item in reference_history_ids)
    thresholds = fast_protocol["feasibility"]
    rules: dict[str, Any] = {}
    for rule_id in frozen_protocol["selection_algorithm"]["eligible_rules"]:
        definition = frozen_protocol["candidate_rules"][rule_id]
        conditions = definition["conditions"]
        intervention_hits = sum(rule_matches(row["features"], conditions) for row in intervention_rows)
        no_change_hits = sum(rule_matches(row["features"], conditions) for row in no_change_rows)
        correct_premise_hits = sum(rule_matches(row["features"], conditions) for row in correct_premise_rows)
        historical_trigger_ids = {
            row["traffic_id"] for row in historical_rows
            if row.get("gate_decision") is True and rule_matches(row["features"], conditions)
        }
        historical_trigger_weight = sum(route_weights[item] for item in historical_trigger_ids)
        recall = intervention_hits / len(intervention_rows) if intervention_rows else None
        false_trigger = no_change_hits / len(no_change_rows) if no_change_rows else None
        historical_call_rate = historical_trigger_weight / total_history_weight if total_history_weight else None
        premise_recall = correct_premise_hits / len(correct_premise_rows) if correct_premise_rows else None
        premise_gate = (
            len(correct_premise_rows) < thresholds["correct_premise_gate_applies_when_count_gte"]
            or premise_recall >= thresholds["correct_premise_recall_gte"]
        )
        feasible = bool(
            powered and recall is not None and false_trigger is not None
            and recall >= thresholds["agent_blind_intervention_recall_gte"]
            and false_trigger <= thresholds["agent_blind_no_change_trigger_rate_lte"]
            and historical_call_rate is not None
            and historical_call_rate <= thresholds["historical_weighted_end_to_end_call_rate_lte"]
            and premise_gate
        )
        rules[rule_id] = {
            "expression": definition["expression"],
            "condition_count": definition["condition_count"],
            "agent_intervention_hits": intervention_hits,
            "agent_intervention_total": len(intervention_rows),
            "agent_intervention_recall": recall,
            "agent_intervention_recall_wilson95": wilson(intervention_hits, len(intervention_rows)),
            "agent_no_change_hits": no_change_hits,
            "agent_no_change_total": len(no_change_rows),
            "agent_no_change_trigger_rate": false_trigger,
            "agent_no_change_trigger_rate_wilson95": wilson(no_change_hits, len(no_change_rows)),
            "agent_correct_premise_hits": correct_premise_hits,
            "agent_correct_premise_total": len(correct_premise_rows),
            "historical_trigger_weight": historical_trigger_weight,
            "historical_total_weight": total_history_weight,
            "historical_weighted_end_to_end_call_rate": historical_call_rate,
            "feasible": feasible,
        }
    feasible_ids = [rule_id for rule_id, metrics in rules.items() if metrics["feasible"]]
    priority = {rule_id: index for index, rule_id in enumerate(
        frozen_protocol["selection_algorithm"]["eligible_rules"]
    )}
    selected = min(feasible_ids, key=lambda rule_id: (
        rules[rule_id]["historical_weighted_end_to_end_call_rate"],
        -rules[rule_id]["agent_intervention_recall"],
        rules[rule_id]["condition_count"], priority[rule_id],
    )) if feasible_ids else None
    conclusion = "SELECT_RULE" if selected else ("UNDERPOWERED" if not powered else "NO_RULE")
    return {
        "schema_version": "answerability-fast-validation-result-v1",
        "protocol_id": fast_protocol["protocol_id"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "conclusion": conclusion,
        "selected_rule": selected,
        "production_effect": "none",
        "scope_limit": fast_protocol["scope_limit"],
        "packages": {"agent_blind": agent_verified, "historical_replay": historical_verified},
        "agent_blind_counts": {
            "generated": sum(_json(path / "summary.json")["generated_queries"] for path in agent_packages),
            "reference": len(agent_rows),
            "accepted_labeled": len(accepted),
            "accepted_intervention": len(intervention_rows),
            "accepted_no_change": len(no_change_rows),
            "correct_premise": len(correct_premise_rows),
            "powered": powered,
        },
        "historical_replay_counts": {
            "unique_requests": len(historical_routes),
            "weighted_requests": total_history_weight,
            "reference_requests": len(reference_history_ids),
            "weighted_reference_requests": reference_history_weight,
            "weighted_reference_route_rate": (
                reference_history_weight / total_history_weight if total_history_weight else None
            ),
        },
        "rules": rules,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Answerability fast validation result", "",
        f"- Conclusion: **{result['conclusion']}**",
        f"- Selected rule: **{result['selected_rule'] or 'none'}**",
        "- Production effect: none (offline validation only)", "",
        "## Counts", "",
        f"- Agent blind: `{json.dumps(result['agent_blind_counts'], ensure_ascii=False)}`",
        f"- Historical replay: `{json.dumps(result['historical_replay_counts'], ensure_ascii=False)}`",
        "", "## Frozen rule metrics", "",
        "| Rule | Intervention recall | No-change trigger | Historical e2e call rate | Feasible |",
        "|---|---:|---:|---:|---|",
    ]
    for rule_id, row in result["rules"].items():
        lines.append(
            f"| {rule_id} | {row['agent_intervention_recall']:.3f} "
            f"({row['agent_intervention_hits']}/{row['agent_intervention_total']}) | "
            f"{row['agent_no_change_trigger_rate']:.3f} "
            f"({row['agent_no_change_hits']}/{row['agent_no_change_total']}) | "
            f"{row['historical_weighted_end_to_end_call_rate']:.3f} | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )
    lines += ["", "## Interpretation", "", result["scope_limit"] + ".", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-protocol", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--agent-package", type=Path, action="append", required=True)
    parser.add_argument("--historical-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        fast_protocol=_json(args.fast_protocol),
        frozen_protocol=_json(args.frozen_protocol),
        agent_packages=args.agent_package,
        historical_package=args.historical_package,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "result.json"
    md_path = args.output_dir / "RESULT.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"result": str(json_path), "conclusion": result["conclusion"], "selected_rule": result["selected_rule"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
