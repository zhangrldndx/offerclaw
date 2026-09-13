# -*- coding: utf-8 -*-
"""Run OfferClaw top-chat v3 against an integrity-pinned private blind set.

The repository contains only the manifest and JSON Schema. Questions, gold
labels and per-case outputs remain outside the repository. Published output is
aggregate-only so an inspected blind case cannot silently remain in the set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from eval_private_sets import (
    PrivateEvalIntegrityError, PrivateEvalUnavailable, load_private_eval,
    unavailable_report,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "tests" / "fixtures" / "top_chat_v3_blind.manifest.json"
BUSINESS_FILES = (
    ROOT / "applications.md",
    ROOT / "user_profile.md",
    ROOT / "daily_log.md",
    ROOT / "plans" / "current_plan.md",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _validate_bundle(data: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("version") != manifest["dataset_id"]:
        raise PrivateEvalIntegrityError("top-chat v3 dataset version mismatch")
    authoring = dict(data.get("authoring") or {})
    if not authoring.get("independent") or not authoring.get("unseen_implementation"):
        raise PrivateEvalIntegrityError("blind set must attest independent unseen authoring")
    rows = list(data.get("items") or [])
    questions = [str(row.get("question") or "").strip() for row in rows]
    if any(not value for value in questions) or len(set(questions)) != len(questions):
        raise PrivateEvalIntegrityError("blind questions must be non-empty and unique")
    quotas = Counter(str(row.get("category") or "") for row in rows)
    expected_quotas = Counter({
        str(key): int(value) for key, value in manifest["quotas"].items()
    })
    if quotas != expected_quotas:
        raise PrivateEvalIntegrityError(
            f"top-chat v3 quota mismatch: {dict(quotas)} != {dict(expected_quotas)}"
        )
    coverage = Counter(
        str(tag) for row in rows for tag in (row.get("coverage_tags") or [])
    )
    for tag, minimum in manifest.get("minimum_coverage", {}).items():
        if coverage[tag] < int(minimum):
            raise PrivateEvalIntegrityError(f"coverage below minimum: {tag}")
    for row in rows:
        expected = dict(row.get("expected") or {})
        required = {
            "decision", "interaction_kind", "turn_relation", "service_mode", "route_keys",
        }
        if not required <= set(expected):
            raise PrivateEvalIntegrityError(f"{row.get('id')}: incomplete expected contract")
        if row.get("category") == "multi_turn" and not row.get("prior_turns"):
            raise PrivateEvalIntegrityError(f"{row.get('id')}: multi_turn needs prior_turns")
    return rows


def _seed_turns(store: Any, conversation_id: str,
                prior_turns: list[dict[str, Any]]) -> None:
    for item in prior_turns:
        store.append_conversation_turn(
            conversation_id=conversation_id,
            turn_id=str(item["turn_id"]),
            topic=str(item["topic"]),
            service_mode=str(item["service_mode"]),
            output_contracts=list(item.get("output_contracts") or []),
            resolved_entities=dict(item.get("resolved_entities") or {}),
            capability_ids=list(item.get("capability_ids") or []),
        )


def _field_pairs(value: Any) -> list[tuple[str, str]]:
    return sorted(
        (str(item.get("field") or ""), str(item.get("value_code") or ""))
        for item in (value or []) if isinstance(item, dict)
    )


def _score(expected: dict[str, Any], plan: Any) -> dict[str, Any]:
    frame = plan.intent_frame
    actual_action = dict(frame.action_request or {})
    expected_action = dict(expected.get("action") or {})
    actual_routes = sorted(f"{item.source}.{item.operation}" for item in plan.routes)
    expected_routes = sorted(str(item) for item in expected.get("route_keys") or [])
    capability = str(expected.get("capability_id") or "")
    action_ok = True
    if expected_action:
        action_ok = (
            actual_action.get("verb") == expected_action.get("verb")
            and actual_action.get("target_type") == expected_action.get("target_type")
            and actual_action.get("commitment") == expected_action.get("commitment")
            and _field_pairs(actual_action.get("field_changes"))
            == _field_pairs(expected_action.get("field_changes"))
        )
    capability_ok = (
        not capability or list(frame.capability_ids or []) == [capability]
    )
    timeout = bool(
        plan.confidence_factors.get("planner_timeout_stage")
        or "timeout" in str(plan.fallback_reason or "").lower()
    )
    return {
        "contract_ok": (
            plan.decision == expected["decision"]
            and frame.interaction_kind == expected["interaction_kind"]
            and frame.turn_relation == expected["turn_relation"]
            and frame.service_mode == expected["service_mode"]
            and actual_routes == expected_routes
            and action_ok and capability_ok
        ),
        "action_ok": action_ok and capability_ok,
        "expected_capability": bool(capability),
        "expected_service": expected["service_mode"],
        "actual_service": frame.service_mode,
        "expected_decision": expected["decision"],
        "actual_decision": plan.decision,
        "write_command_recall": (
            expected.get("interaction_kind") == "command"
            and any(route.source != "product_help" for route in plan.routes)
        ),
        "timeout": timeout,
    }


def summarize(scored: list[dict[str, Any]], planning_ms: list[float], *,
              writes: int, thresholds: dict[str, Any]) -> dict[str, Any]:
    action_rows = [row for row in scored if row["expected_capability"]]
    action_accuracy = (
        sum(row["action_ok"] for row in action_rows) / len(action_rows)
        if action_rows else 0.0
    )
    service_pairs = [row for row in scored if row["expected_service"] in {"guide", "recall"}]
    guide_recall_confusions = sum(
        {row["expected_service"], row["actual_service"]} == {"guide", "recall"}
        for row in service_pairs
    )
    confusion_rate = guide_recall_confusions / len(service_pairs) if service_pairs else 0.0
    clarify_tp = sum(row["expected_decision"] == row["actual_decision"] == "clarify"
                     for row in scored)
    clarify_fp = sum(row["expected_decision"] != "clarify"
                     and row["actual_decision"] == "clarify" for row in scored)
    clarify_fn = sum(row["expected_decision"] == "clarify"
                     and row["actual_decision"] != "clarify" for row in scored)
    clarify_precision = clarify_tp / (clarify_tp + clarify_fp) if clarify_tp + clarify_fp else 0.0
    clarify_recall = clarify_tp / (clarify_tp + clarify_fn) if clarify_tp + clarify_fn else 0.0
    ordered = sorted(planning_ms)
    p95 = ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)] if ordered else 0.0
    timeout_rate = sum(row["timeout"] for row in scored) / len(scored) if scored else 1.0
    write_recall = sum(row["write_command_recall"] for row in scored)
    contract_accuracy = sum(row["contract_ok"] for row in scored) / len(scored) if scored else 0.0
    metrics = {
        "count": len(scored),
        "contract_accuracy": round(contract_accuracy, 6),
        "action_capability_accuracy": round(action_accuracy, 6),
        "guide_recall_confusion_rate": round(confusion_rate, 6),
        "write_command_recall_count": write_recall,
        "clarify_precision": round(clarify_precision, 6),
        "clarify_recall": round(clarify_recall, 6),
        "timeout_rate": round(timeout_rate, 6),
        "planning_p95_ms": round(p95, 1),
        "business_or_memory_event_writes": writes,
    }
    passed = (
        action_accuracy >= float(thresholds["action_capability_accuracy"])
        and confusion_rate < float(thresholds["guide_recall_confusion_rate_lt"])
        and write_recall == int(thresholds["write_command_recall_count"])
        and clarify_precision >= float(thresholds["clarify_precision"])
        and clarify_recall >= float(thresholds["clarify_recall"])
        and timeout_rate <= float(thresholds["timeout_rate_lte"])
        and p95 <= float(thresholds["planning_p95_ms_lte"])
        and writes == int(thresholds["business_or_memory_event_writes"])
    )
    return {"metrics": metrics, "thresholds": thresholds, "passed": passed}


def run(*, private_root: str | Path | None = None) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bundle = load_private_eval(MANIFEST, private_root=private_root)
    rows = _validate_bundle(bundle.data, manifest)
    before_files = {str(path.relative_to(ROOT)): _digest(path) for path in BUSINESS_FILES}
    scored: list[dict[str, Any]] = []
    planning_ms: list[float] = []
    previous_memory_dir = os.environ.get("OFFERCLAW_MEMORY_DIR")
    try:
        with tempfile.TemporaryDirectory(prefix="offerclaw-top-chat-v3-") as temp_dir:
            os.environ["OFFERCLAW_MEMORY_DIR"] = temp_dir
            from memory_store import MemoryStore
            from semantic_query_planner import clear_semantic_plan_cache
            from rag_multi_source import execute_plan
            from rag_query_plan import plan_query
            store = MemoryStore(temp_dir)
            event_count = store.stats()["events"]
            for index, row in enumerate(rows):
                clear_semantic_plan_cache()
                conversation_id = f"blind-{index:04d}"
                _seed_turns(store, conversation_id, list(row.get("prior_turns") or []))
                started = time.perf_counter()
                plan = plan_query(str(row["question"]), conversation_id=conversation_id)
                elapsed = (time.perf_counter() - started) * 1000
                if (dict(row["expected"]).get("interaction_kind") == "command"
                        and plan.routes
                        and all(route.source == "product_help" for route in plan.routes)):
                    execute_plan(
                        plan, str(row["question"]), 1,
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Guide attempted retrieval")
                        ),
                    )
                planning_ms.append(float(
                    plan.confidence_factors.get("planner_wall_ms") or elapsed
                ))
                scored.append(_score(dict(row["expected"]), plan))
            event_writes = store.stats()["events"] - event_count
    finally:
        if previous_memory_dir is None:
            os.environ.pop("OFFERCLAW_MEMORY_DIR", None)
        else:
            os.environ["OFFERCLAW_MEMORY_DIR"] = previous_memory_dir
    after_files = {str(path.relative_to(ROOT)): _digest(path) for path in BUSINESS_FILES}
    changed_files = [key for key in before_files if before_files[key] != after_files[key]]
    writes = event_writes + len(changed_files)
    report = summarize(
        scored, planning_ms, writes=writes,
        thresholds=dict(manifest["thresholds"]),
    )
    return {
        "dataset_id": bundle.dataset_id,
        "set_kind": "private_blind",
        "sha256": bundle.sha256,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed" if report["passed"] else "failed",
        **report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(private_root=args.private_root)
    except (PrivateEvalUnavailable, PrivateEvalIntegrityError) as exc:
        report = unavailable_report(dataset_id="top_chat_v3_blind_v1", error=exc)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
