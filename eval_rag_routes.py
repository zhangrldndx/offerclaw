# -*- coding: utf-8 -*-
"""OfferClaw 多意图路由离线评测。

用法：``.venv/bin/python eval_rag_routes.py``。默认执行规则规划器，不调用 LLM、
不读取个人数据、也不运行向量检索，因此适合 CI 和路由改动前后的快速回归。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


DEFAULT_SET = Path(__file__).resolve().parent / "tests" / "rag_route_eval_set.json"


def _f1(expected: set[str], predicted: set[str]) -> float:
    if not expected and not predicted:
        return 1.0
    tp = len(expected & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _expanded_cases(data: dict) -> list[dict]:
    if data.get("items"):
        return list(data["items"])
    wrappers = data.get("wrappers") or ["{q}"]
    cases = []
    for suite in data.get("suites") or []:
        for seed_n, seed in enumerate(suite.get("seeds") or [], start=1):
            for wrapper_n, wrapper in enumerate(wrappers, start=1):
                cases.append({
                    "id": f"{suite['id']}-{seed_n:02d}-{wrapper_n:02d}",
                    "category": suite["id"],
                    "question": str(wrapper).replace("{q}", seed["q"]),
                    "context": seed.get("context") or [],
                    "expected": seed.get("routes") or [],
                    "expected_objects": seed.get("objects") or [],
                    "expected_operations": seed.get("operations") or [],
                    "expected_service_mode": seed.get("service_mode") or "",
                    "expected_decision": seed.get("decision") or "answer",
                })
    return cases


def _planner_for_mode(mode: str):
    from rag_query_plan import plan_query, rule_plan_query
    mode = mode.upper()
    if mode == "A":
        return lambda question, context: rule_plan_query(question, context=context)
    os.environ["RAG_ROUTER_MODE"] = {
        "B": "hybrid", "C": "hybrid", "D": "llm_ceiling",
    }.get(mode, "legacy")
    os.environ["RAG_SCOPED_LLM_PLANNER"] = "0" if mode == "B" else "1"
    return lambda question, context: plan_query(question, context=context)


def evaluate(path: str | Path = DEFAULT_SET, *, mode: str = "A") -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = _expanded_cases(data)
    planner = _planner_for_mode(mode)
    failures = []
    f1s = []
    object_hits = operation_hits = 0
    object_cases = operation_cases = 0
    decision_hits = service_mode_hits = 0
    service_mode_cases = 0
    app_expected = app_hit = 0
    by_category: dict[str, list[float]] = {}
    latencies = []
    embedding_calls = llm_calls = 0
    for case in cases:
        expected = set(case.get("expected") or [])
        started = time.perf_counter()
        plan = planner(case.get("question", ""), case.get("context") or [])
        latencies.append((time.perf_counter() - started) * 1000)
        predicted = {f"{route.source}.{route.operation}" for route in plan.routes}
        score = _f1(expected, predicted)
        f1s.append(score)
        by_category.setdefault(case.get("category", "unknown"), []).append(score)
        for route in expected:
            if route.startswith("application_state."):
                app_expected += 1
                app_hit += int(route in predicted)
        expected_objects = set(case.get("expected_objects") or [])
        if expected_objects:
            object_cases += 1
            predicted_objects = set(plan.intent_frame.answer_objects or [])
            object_hits += int(expected_objects == predicted_objects)
        else:
            predicted_objects = set()
        expected_operations = set(case.get("expected_operations") or [])
        if expected_operations:
            operation_cases += 1
            predicted_operations = set(plan.intent_frame.operations or [])
            operation_hits += int(expected_operations == predicted_operations)
        else:
            predicted_operations = set()
        expected_decision = case.get("expected_decision") or "answer"
        decision_hits += int(plan.decision == expected_decision)
        expected_service_mode = str(case.get("expected_service_mode") or "")
        if expected_service_mode:
            service_mode_cases += 1
            service_mode_hits += int(plan.intent_frame.service_mode == expected_service_mode)
        embedding_calls += int(plan.confidence_factors.get("embedding_calls", 0) or 0)
        llm_calls += int(plan.confidence_factors.get("llm_calls", 0) or 0)
        if (expected != predicted or (expected_objects and expected_objects != predicted_objects)
                or (expected_operations and expected_operations != predicted_operations)
                or (expected_service_mode
                    and plan.intent_frame.service_mode != expected_service_mode)
                or plan.decision != expected_decision):
            failures.append({
                "id": case.get("id"), "category": case.get("category"),
                "question": case.get("question"),
                "expected": sorted(expected), "predicted": sorted(predicted),
                "expected_objects": sorted(expected_objects),
                "predicted_objects": sorted(predicted_objects),
                "expected_operations": sorted(expected_operations),
                "predicted_operations": sorted(predicted_operations),
                "expected_decision": expected_decision, "decision": plan.decision,
                "expected_service_mode": expected_service_mode,
                "service_mode": plan.intent_frame.service_mode,
                "f1": round(score, 4),
            })
    ordered_latency = sorted(latencies)
    p95_index = max(0, min(len(ordered_latency) - 1,
                           int(len(ordered_latency) * 0.95) - 1)) if ordered_latency else 0
    return {
        "set_version": data.get("version", "rag-route-eval-v1"),
        "mode": mode.upper(),
        "count": len(cases),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "application_state_recall": round(app_hit / app_expected, 4) if app_expected else 1.0,
        "exact_match": round((len(cases) - len(failures)) / len(cases), 4) if cases else 0.0,
        "answer_object_accuracy": round(object_hits / object_cases, 4) if object_cases else None,
        "operation_accuracy": round(operation_hits / operation_cases, 4) if operation_cases else None,
        "decision_accuracy": round(decision_hits / len(cases), 4) if cases else 0.0,
        "service_mode_accuracy": (
            round(service_mode_hits / service_mode_cases, 4) if service_mode_cases else None
        ),
        "planning_p95_ms": round(ordered_latency[p95_index], 1) if ordered_latency else 0.0,
        "embedding_calls": embedding_calls,
        "llm_calls": llm_calls,
        "by_category": {
            name: round(sum(scores) / len(scores), 4) for name, scores in by_category.items()
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OfferClaw RAG route evaluation")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--show-failures", type=int, default=20)
    parser.add_argument("--mode", choices=["A", "B", "C", "D"], default="A",
                        help="A=规则；B=规则+语义；C=按需裁决；D=全量LLM质量上限")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    os.environ["RAG_LLM_PLANNER"] = "0"
    report = evaluate(args.set, mode=args.mode)
    display = {k: v for k, v in report.items() if k != "failures"}
    display["failure_count"] = len(report["failures"])
    display["failure_samples"] = report["failures"][:args.show_failures]
    print(json.dumps(display, ensure_ascii=False, indent=2))
    if args.no_gate:
        return 0
    expected_count = 120
    try:
        raw = json.loads(Path(args.set).read_text(encoding="utf-8"))
        if raw.get("expected_count") is not None:
            expected_count = int(raw["expected_count"])
        elif raw.get("suites"):
            expected_count = sum(int(s.get("expected_count", 0)) for s in raw["suites"])
    except Exception:
        pass
    objects_ok = report["answer_object_accuracy"] is None or report["answer_object_accuracy"] >= 0.97
    operations_ok = report["operation_accuracy"] is None or report["operation_accuracy"] >= 0.97
    services_ok = report["service_mode_accuracy"] is None or report["service_mode_accuracy"] >= 0.97
    return 0 if (report["count"] == expected_count and report["macro_f1"] >= 0.95
                 and report["application_state_recall"] == 1.0
                 and objects_ok and operations_ok and services_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
