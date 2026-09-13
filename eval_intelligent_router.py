# -*- coding: utf-8 -*-
"""Evaluate OfferClaw routing on development or private blind sets.

The fixture is a regression set, not a claim of open-world/blind
generalisation. Intelligent mode may make one structured model call (and one
format-only repair) per non-hard question, so this command is intentionally
explicit rather than part of the default offline pytest run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from eval_private_sets import (PrivateEvalIntegrityError, PrivateEvalUnavailable,
                               load_private_eval, sha256_file, unavailable_report)


BASE = Path(__file__).resolve().parent
FIXTURES = BASE / "tests" / "fixtures"
MANIFEST = FIXTURES / "intelligent_router_frozen_v1.json"
SURFACE_MANIFEST = FIXTURES / "router_surface_invariance_v1.manifest.json"
BLIND_MANIFEST = FIXTURES / "router_blind_v1.manifest.json"
HUMAN_ACCEPTANCE_SET = (
    BASE / "docs/rag_eval/human_acceptance/router_human_frozen_v1.json"
)
HUMAN_ACCEPTANCE_MANIFEST = HUMAN_ACCEPTANCE_SET.with_suffix(".manifest.json")
GOLD_ONLY_ROUTES = {"general_fallback.answer"}

SET_ALIASES = {
    "reviewed_regression": "router_reviewed_regression_v1",
    "surface_invariance": "router_surface_invariance_v1",
    "blind": "router_blind_v1",
    "human_acceptance": "router_human_frozen_v1",
}


def _normalise_pairs(value: Any) -> list[str]:
    """Normalise optional ``route -> value`` annotations for future fixtures."""
    if isinstance(value, dict):
        pairs: list[str] = []
        for key, item in value.items():
            if isinstance(item, list):
                pairs.extend(f"{key}={nested}" for nested in item)
            else:
                pairs.append(f"{key}={item}")
        return sorted(set(pairs))
    if not isinstance(value, list):
        return []
    pairs: list[str] = []
    for item in value:
        if isinstance(item, dict):
            route = str(item.get("route") or item.get("route_key") or "")
            role = str(item.get("role") or item.get("depends_on") or "")
            if route and role:
                pairs.append(f"{route}={role}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append(f"{item[0]}={item[1]}")
        elif isinstance(item, str) and item:
            if "<-" in item:
                consumer, producer = (part.strip() for part in item.split("<-", 1))
                pairs.append(f"{consumer}={producer.split('.', 1)[0]}")
            else:
                pairs.append(item)
    return sorted(set(pairs))


def _row_from_item(item: dict[str, Any]) -> dict[str, Any]:
    expected = list(item.get("expected") or item.get("required_routes") or [])
    return {
        "category": str(item.get("category") or ""),
        "coverage_tags": list(item.get("coverage_tags") or []),
        "question": item["question"],
        "context": list(item.get("context") or []),
        "expected": expected,
        "required_routes": list(item.get("required_routes") or expected),
        "allowed_optional_routes": list(item.get("allowed_optional_routes") or []),
        "forbidden_routes": list(item.get("forbidden_routes") or []),
        "acceptable_plans": list(item.get("acceptable_plans") or []),
        "closed_world": bool(item.get(
            "closed_world", ("expected" in item or "required_routes" in item)
        )),
        "service_mode": item.get("expected_service_mode", item.get("service_mode", "")),
        "expected_objects": list(
            item.get("expected_objects") or item.get("answer_objects") or []
        ),
        "expected_operations": list(
            item.get("expected_operations") or item.get("operations") or []
        ),
        "expected_decision": item.get("expected_decision", item.get("decision", "answer")),
        "expected_source_roles": _normalise_pairs(
            item.get("expected_source_roles") or item.get("source_roles")
        ),
        "expected_dependencies": _normalise_pairs(
            item.get("expected_dependencies") or item.get("depends_on")
        ),
        "required_entities": dict(item.get("required_entities") or {}),
        "id": item["id"],
    }


def _reviewed_rows() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        data = json.loads((FIXTURES / source["path"]).read_text(encoding="utf-8"))
        selector = source["selector"]
        if selector == "all_queries":
            default_operation = {
                "application_state": "list_current",
                "application_jd": "search_bound_jd",
                "reference_kb": "search",
            }
            selected = [{
                "question": question,
                "context": [],
                "expected": ([f"{pair[0]}.{pair[1]}" for pair in family.get("must_pairs", [])]
                             or [f"{source_name}.{default_operation[source_name]}"
                                 for source_name in family.get("must_sources", [])]),
                "service_mode": "",
                "expected_objects": [],
                "expected_operations": [],
                "expected_decision": "answer",
                "expected_source_roles": [],
                "expected_dependencies": [],
                "required_entities": {},
                "id": f"{family['id']}:{index}",
            } for family in data["families"]
              for index, question in enumerate(family["queries"])]
        elif selector == "all_items":
            selected = [_row_from_item(item) for item in data["items"]]
        else:
            selected = []
            for index, raw_question in enumerate(data[selector]):
                item = raw_question if isinstance(raw_question, dict) else {}
                selected.append({
                    "question": item.get("q", "") if item else raw_question,
                    "context": list(item.get("context") or []),
                    "expected": list(source["expected"]),
                    "service_mode": source.get("service_mode", ""),
                    "expected_objects": list(source.get("expected_objects") or []),
                    "expected_operations": list(source.get("expected_operations") or []),
                    "expected_decision": source.get("expected_decision", "answer"),
                    "expected_source_roles": _normalise_pairs(
                        source.get("expected_source_roles")
                    ),
                    "expected_dependencies": _normalise_pairs(
                        source.get("expected_dependencies")
                    ),
                    "required_entities": dict(source.get("required_entities") or {}),
                    "id": f"{Path(source['path']).stem}:{selector}:{index}",
                })
        rows.extend(selected[:int(source["count"])])
    for row in rows:
        row.setdefault("required_routes", list(row.get("expected") or []))
        row.setdefault("allowed_optional_routes", [])
        row.setdefault("forbidden_routes", [])
        row.setdefault("acceptable_plans", [])
        row.setdefault("closed_world", True)
        row.setdefault("required_entities", {})
    expected_count = int(manifest["expected_count"])
    if len(rows) != expected_count:
        raise RuntimeError(f"regression-set count mismatch: {len(rows)} != {expected_count}")
    questions = [row["question"] for row in rows]
    if len(set(questions)) != len(questions):
        duplicates = [q for q, count in Counter(questions).items() if count > 1]
        raise RuntimeError(f"regression set has duplicate questions: {duplicates[:5]}")
    return rows


def _surface_rows() -> list[dict[str, Any]]:
    manifest = json.loads(SURFACE_MANIFEST.read_text(encoding="utf-8"))
    fixture = FIXTURES / str(manifest["dataset_file"])
    actual_hash = sha256_file(fixture)
    if actual_hash != manifest["sha256"]:
        raise RuntimeError(
            f"surface-invariance SHA-256 mismatch: {actual_hash} != {manifest['sha256']}"
        )
    data = json.loads(fixture.read_text(encoding="utf-8"))
    wrappers = list(data.get("wrappers") or ["{q}"])
    rows: list[dict[str, Any]] = []
    for suite in data.get("suites") or []:
        for seed_n, seed in enumerate(suite.get("seeds") or [], start=1):
            for wrapper_n, wrapper in enumerate(wrappers, start=1):
                rows.append(_row_from_item({
                    "id": f"{suite['id']}-{seed_n:02d}-{wrapper_n:02d}",
                    "category": str(suite["id"]),
                    "question": str(wrapper).replace("{q}", str(seed["q"])),
                    "context": seed.get("context") or [],
                    "expected": seed.get("routes") or [],
                    "expected_objects": seed.get("objects") or [],
                    "expected_operations": seed.get("operations") or [],
                    "expected_service_mode": seed.get("service_mode") or "",
                    "expected_decision": seed.get("decision") or "answer",
                }))
    if len(rows) != int(manifest["expected_count"]):
        raise RuntimeError(
            f"surface-invariance count mismatch: {len(rows)} != {manifest['expected_count']}"
        )
    return rows


def _blind_rows(private_root: str | Path | None = None) -> tuple[list[dict[str, Any]], str]:
    bundle = load_private_eval(BLIND_MANIFEST, private_root=private_root)
    rows = [_row_from_item(item) for item in bundle.data["items"]]
    for row in rows:
        if not row["required_routes"] and row["expected_decision"] != "clarify":
            raise PrivateEvalIntegrityError(
                "answer cases require at least one required route"
            )
        if set(row["required_routes"]) & set(row["forbidden_routes"]):
            raise PrivateEvalIntegrityError(
                "a route cannot be both required and forbidden"
            )
    questions = [row["question"] for row in rows]
    if len(set(questions)) != len(questions):
        raise PrivateEvalIntegrityError("private router set contains duplicate questions")
    manifest = json.loads(BLIND_MANIFEST.read_text(encoding="utf-8"))
    expected_quotas = {str(key): int(value) for key, value in manifest["quotas"].items()}
    actual_quotas = Counter(row.get("category") or "" for row in rows)
    if dict(actual_quotas) != expected_quotas:
        raise PrivateEvalIntegrityError(
            f"private router quota mismatch: {dict(actual_quotas)} != {expected_quotas}"
        )
    if sum(row["expected_decision"] == "clarify" for row in rows) != expected_quotas["clarify"]:
        raise PrivateEvalIntegrityError("clarify decision count does not match its quota")
    coverage = Counter(
        tag for row in rows for tag in row.get("coverage_tags") or []
    )
    for tag, minimum in manifest.get("minimum_coverage", {}).items():
        if coverage[tag] < int(minimum):
            raise PrivateEvalIntegrityError(
                f"private router coverage is below minimum for {tag}"
            )
    return rows, bundle.sha256


def _human_acceptance_rows() -> tuple[list[dict[str, Any]], str]:
    """Load the user-reviewed set and preserve prior user turns as context."""
    if not HUMAN_ACCEPTANCE_SET.exists() or not HUMAN_ACCEPTANCE_MANIFEST.exists():
        raise PrivateEvalUnavailable("human acceptance frozen set or manifest is missing")
    manifest = json.loads(HUMAN_ACCEPTANCE_MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256_file(HUMAN_ACCEPTANCE_SET)
    expected_hash = str(manifest.get("sha256") or "")
    if not expected_hash or actual_hash != expected_hash:
        raise PrivateEvalIntegrityError(
            f"human acceptance SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )
    payload = json.loads(HUMAN_ACCEPTANCE_SET.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise PrivateEvalIntegrityError("human acceptance set must have status=frozen")
    if payload.get("version") != manifest.get("version"):
        raise PrivateEvalIntegrityError("human acceptance version does not match manifest")
    scored_fields = set(payload.get("scored_fields") or [])

    rows: list[dict[str, Any]] = []
    for case in payload.get("cases") or []:
        if case.get("review_status") != "approved":
            raise PrivateEvalIntegrityError(
                f"{case.get('case_id')}: frozen set contains an unapproved case"
            )
        prior_questions: list[str] = []
        for turn in case.get("turns") or []:
            gold = dict(turn.get("gold") or {})
            rows.append(_row_from_item({
                "id": f"{case['case_id']}:T{turn['turn']}",
                "category": case.get("primary_category") or "",
                "coverage_tags": case.get("coverage_tags") or [],
                "question": turn.get("question") or "",
                "context": prior_questions[-3:],
                "required_routes": gold.get("required_routes") or [],
                "allowed_optional_routes": gold.get("allowed_optional_routes") or [],
                "forbidden_routes": gold.get("forbidden_routes") or [],
                "acceptable_plans": gold.get("acceptable_plans") or [],
                "closed_world": gold.get("closed_world", True),
                "service_mode": gold.get("service_mode") or "",
                "answer_objects": (
                    gold.get("answer_objects") or []
                    if "answer_objects" in scored_fields else []
                ),
                "operations": (
                    gold.get("operations") or []
                    if "operations" in scored_fields else []
                ),
                "decision": gold.get("decision") or "answer",
                "source_roles": (
                    gold.get("source_roles") or []
                    if "source_roles" in scored_fields else []
                ),
                "depends_on": (
                    gold.get("depends_on") or []
                    if "depends_on" in scored_fields else []
                ),
                "required_entities": gold.get("required_entities") or {},
            }))
            prior_questions.append(str(turn.get("question") or ""))

    if len(payload.get("cases") or []) != int(manifest.get("case_count") or -1):
        raise PrivateEvalIntegrityError("human acceptance case count does not match manifest")
    if len(rows) != int(manifest.get("turn_count") or -1):
        raise PrivateEvalIntegrityError("human acceptance turn count does not match manifest")
    if not rows or any(not row["question"] for row in rows):
        raise PrivateEvalIntegrityError("human acceptance set contains an empty question")
    return rows, actual_hash


def _rows(set_name: str = "reviewed_regression", *,
          private_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Compatibility loader; defaults to the reviewed 240-question set."""
    if set_name == "reviewed_regression":
        return _reviewed_rows()
    if set_name == "surface_invariance":
        return _surface_rows()
    if set_name == "blind":
        return _blind_rows(private_root)[0]
    if set_name == "human_acceptance":
        return _human_acceptance_rows()[0]
    raise ValueError(f"unknown evaluation set: {set_name}")


def _validate_gold_registry(rows: list[dict[str, Any]],
                            registered_routes: set[str]) -> None:
    unknown: set[str] = set()
    for row in rows:
        route_keys = set(row.get("required_routes") or [])
        route_keys.update(row.get("allowed_optional_routes") or [])
        route_keys.update(row.get("forbidden_routes") or [])
        for option in row.get("acceptable_plans") or []:
            if isinstance(option, list):
                route_keys.update(str(value) for value in option)
            elif isinstance(option, dict):
                route_keys.update(option.get("required_routes") or option.get("routes") or [])
                route_keys.update(option.get("allowed_optional_routes") or [])
        unknown.update(route_keys - registered_routes)
    if unknown:
        raise PrivateEvalIntegrityError(
            "gold contract references unregistered routes: " + ", ".join(sorted(unknown))
        )


def _f1(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    precision = len(expected & actual) / len(actual)
    recall = len(expected & actual) / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _gold_options(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand shorthand/alternative gold plans into comparable contracts."""
    global_required = set(row.get("required_routes") or row.get("expected") or [])
    global_optional = set(row.get("allowed_optional_routes") or [])
    global_roles = list(row.get("expected_source_roles") or [])
    global_dependencies = list(row.get("expected_dependencies") or [])
    raw_options = list(row.get("acceptable_plans") or [])
    if not raw_options:
        return [{
            "required": global_required,
            "optional": global_optional,
            "closed_world": bool(row.get("closed_world", True)),
            "roles": global_roles,
            "dependencies": global_dependencies,
        }]
    options: list[dict[str, Any]] = []
    for raw in raw_options:
        if isinstance(raw, list):
            required = set(str(value) for value in raw)
            options.append({
                "required": required,
                "optional": set(),
                "closed_world": True,
                "roles": global_roles,
                "dependencies": global_dependencies,
            })
            continue
        if not isinstance(raw, dict):
            continue
        required = global_required | set(raw.get("required_routes") or raw.get("routes") or [])
        optional = global_optional | set(raw.get("allowed_optional_routes") or [])
        options.append({
            "required": required,
            "optional": optional,
            "closed_world": bool(raw.get("closed_world", True)),
            "roles": _normalise_pairs(raw.get("source_roles")) or global_roles,
            "dependencies": _normalise_pairs(raw.get("depends_on")) or global_dependencies,
        })
    return options or [{
        "required": global_required,
        "optional": global_optional,
        "closed_world": bool(row.get("closed_world", True)),
        "roles": global_roles,
        "dependencies": global_dependencies,
    }]


def _gold_score(row: dict[str, Any], actual_routes: set[str], *,
                actual_roles: list[str], actual_dependencies: list[str]) -> dict[str, Any]:
    forbidden = set(row.get("forbidden_routes") or [])
    candidates: list[dict[str, Any]] = []
    for index, option in enumerate(_gold_options(row)):
        required = set(option["required"])
        optional = set(option["optional"])
        missing = required - actual_routes
        forbidden_hits = forbidden & actual_routes
        allowed = required | optional
        unexpected = (actual_routes - allowed) if option["closed_world"] else set()
        required_recall = len(required & actual_routes) / len(required) if required else 1.0
        precision = (
            len(actual_routes - forbidden_hits - unexpected) / len(actual_routes)
            if actual_routes else (1.0 if not required else 0.0)
        )
        route_f1 = (
            2 * precision * required_recall / (precision + required_recall)
            if precision + required_recall else 0.0
        )
        roles_ok = not option["roles"] or option["roles"] == actual_roles
        dependencies_ok = (
            not option["dependencies"] or option["dependencies"] == actual_dependencies
        )
        candidates.append({
            "option_index": index,
            "route_f1": route_f1,
            "required_recall": required_recall,
            "missing": sorted(missing),
            "forbidden_hits": sorted(forbidden_hits),
            "unexpected": sorted(unexpected),
            "roles_ok": roles_ok,
            "dependencies_ok": dependencies_ok,
            "matched": not missing and not forbidden_hits and not unexpected
                       and roles_ok and dependencies_ok,
            "required_routes": sorted(required),
            "allowed_routes": sorted(required | optional),
            "closed_world": bool(option["closed_world"]),
        })
    return max(
        candidates,
        key=lambda item: (item["matched"], item["route_f1"], item["required_recall"]),
    )


def _entity_value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        actual_values = actual if isinstance(actual, list) else [actual]
        return all(any(_entity_value_matches(value, candidate) for candidate in actual_values)
                   for value in expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _entity_value_matches(value, actual[key])
            for key, value in expected.items()
        )
    return str(expected).strip().lower() == str(actual).strip().lower()


def _required_entities_ok(required: dict[str, Any], plan: Any) -> bool:
    if not required:
        return True
    actual = dict(getattr(plan, "entities", {}) or {})
    actual.update(dict(getattr(plan.intent_frame, "filters", {}) or {}))
    return all(key in actual and _entity_value_matches(value, actual[key])
               for key, value in required.items())


def _factor_metric(factors: dict[str, Any], key: str, default: Any = 0) -> Any:
    """Read current or nested shadow/observability planner metrics."""
    if key in factors:
        return factors[key]
    for nested_key in ("planner_metrics", "shadow_factors", "semantic_meta"):
        nested = factors.get(nested_key)
        if isinstance(nested, dict):
            value = _factor_metric(nested, key, None)
            if value is not None:
                return value
    return default


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * .95) - 1)]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _dag_valid(routes: list[Any]) -> tuple[bool, list[str]]:
    """Validate the executable source-level dependency graph."""
    errors: list[str] = []
    sources = {route.source for route in routes}
    graph: dict[str, set[str]] = {source: set() for source in sources}
    for route in routes:
        for dependency in route.depends_on:
            if dependency not in sources:
                errors.append(f"missing_dependency:{route.source}->{dependency}")
            elif dependency == route.source:
                errors.append(f"self_dependency:{route.source}")
            else:
                graph.setdefault(route.source, set()).add(dependency)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            errors.append(f"dependency_cycle:{node}")
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, set()):
            visit(dependency)
        visiting.discard(node)
        visited.add(node)

    for source in graph:
        visit(source)
    return not errors, list(dict.fromkeys(errors))


def _plan_structure(plan: Any, registered_routes: set[str], source_roles: set[str]) -> dict[str, Any]:
    route_keys = [f"{route.source}.{route.operation}" for route in plan.routes]
    errors: list[str] = []
    if plan.decision not in {"answer", "clarify", "fallback"}:
        errors.append(f"invalid_decision:{plan.decision}")
    if plan.decision == "answer" and not plan.routes:
        errors.append("answer_without_routes")
    if plan.decision == "clarify" and not str(plan.clarification).strip():
        errors.append("clarify_without_question")
    if not plan.planner_engine or not plan.planner_version or not plan.schema_version:
        errors.append("missing_planner_schema_provenance")
    unknown = sorted(set(route_keys) - registered_routes)
    if unknown:
        errors.append("unknown_routes:" + ",".join(unknown))
    invalid_roles = sorted({route.source_role for route in plan.routes} - source_roles)
    if invalid_roles:
        errors.append("invalid_source_roles:" + ",".join(invalid_roles))
    if plan.decision == "answer" and plan.routes and not any(
            route.source_role == "answer_source" for route in plan.routes):
        errors.append("missing_answer_source")
    dag_ok, dag_errors = _dag_valid(plan.routes)
    errors.extend(dag_errors)
    try:
        json.dumps(plan.to_dict(), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"not_json_serialisable:{type(exc).__name__}")
    return {
        "schema_valid": not errors,
        "source_roles_valid": not invalid_roles,
        "dag_valid": dag_ok,
        "errors": list(dict.fromkeys(errors)),
    }


def _ratio(hits: int, cases: int) -> float | None:
    return round(hits / cases, 6) if cases else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set", choices=sorted(SET_ALIASES), default="reviewed_regression",
        help="Development, human-reviewed acceptance, or repository-external blind set",
    )
    parser.add_argument("--private-root", default="",
                        help="Override OFFERCLAW_PRIVATE_EVAL_ROOT for blind evaluation")
    parser.add_argument(
        "--mode", choices=["rule_baseline", "intelligent"], default="intelligent",
        help="Production semantic v3 or the explicit offline rule baseline",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="",
                        help="Comma-separated case IDs/turn IDs for a focused development rerun")
    parser.add_argument("--pace-ms", type=float, default=0.0,
                        help="Pause between model calls without including the pause in latency")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-results", action="store_true",
                        help="Include every development row, not only failures")
    parser.add_argument("--no-gate", action="store_true",
                        help="Report metrics without returning non-zero on threshold failure")
    parser.add_argument("--min-source-f1", type=float, default=.95)
    parser.add_argument("--min-route-f1", type=float, default=.95)
    parser.add_argument("--min-service-accuracy", type=float, default=.97)
    parser.add_argument("--min-object-accuracy", type=float, default=.97)
    parser.add_argument("--min-operation-accuracy", type=float, default=.97)
    parser.add_argument("--min-decision-accuracy", type=float, default=.99)
    parser.add_argument("--min-schema-valid-rate", type=float, default=.995)
    parser.add_argument("--min-structural-valid-rate", type=float, default=1.0)
    parser.add_argument("--max-llm-calls-per-question", type=int, default=2)
    parser.add_argument("--max-non-hard-p95-ms", type=float, default=5000.0)
    parser.add_argument("--min-required-route-recall", type=float, default=.98)
    parser.add_argument("--min-clarify-precision", type=float, default=.95)
    parser.add_argument("--min-clarify-recall", type=float, default=.95)
    parser.add_argument("--max-timeout-rate", type=float, default=.01)
    parser.add_argument("--max-fallback-rate", type=float, default=.03)
    args = parser.parse_args()
    from rag_query_plan import plan_query, rule_plan_query
    planner = rule_plan_query if args.mode == "rule_baseline" else plan_query
    from rag_route_registry import READ_SOURCE_REGISTRY, SOURCE_ROLES

    registered_routes = {item.key for item in READ_SOURCE_REGISTRY}
    private_hash = ""
    try:
        if args.set == "blind":
            rows, private_hash = _blind_rows(args.private_root or None)
        elif args.set == "human_acceptance":
            rows, private_hash = _human_acceptance_rows()
        else:
            rows = _rows(args.set)
    except (PrivateEvalUnavailable, PrivateEvalIntegrityError) as exc:
        report = unavailable_report(dataset_id=SET_ALIASES[args.set], error=exc)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
        return 2
    if args.set == "blind" and (args.limit or args.ids or args.include_results):
        parser.error("row selection and case output are not allowed for blind evaluation")
    if args.ids:
        selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
        rows = [row for row in rows if row["id"] in selected_ids]
        missing_ids = selected_ids - {row["id"] for row in rows}
        if missing_ids:
            parser.error("unknown --ids: " + ",".join(sorted(missing_ids)))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error("evaluation set is empty")
    try:
        _validate_gold_registry(rows, registered_routes | GOLD_ONLY_ROUTES)
    except PrivateEvalIntegrityError as exc:
        if args.set != "blind":
            raise
        report = unavailable_report(dataset_id=SET_ALIASES[args.set], error=exc)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
        return 2

    results: list[dict[str, Any]] = []
    application_expected = application_hit = 0
    service_hits = service_cases = 0
    object_hits = object_cases = 0
    operation_hits = operation_cases = 0
    decision_hits = 0
    role_gold_hits = role_gold_cases = 0
    dependency_gold_hits = dependency_gold_cases = 0
    required_route_expected = required_route_hit = 0
    forbidden_route_violations = 0
    acceptable_plan_hits = 0
    required_entity_cases = required_entity_hits = 0
    clarify_expected = clarify_predicted = clarify_true_positive = 0
    timeout_count = fallback_count = circuit_rejected_count = 0
    prompt_tokens_total = completion_tokens_total = 0
    llm_calls_total = 0
    llm_budget_violations = 0
    hard_call_violations = 0
    non_hard_latencies: list[float] = []
    all_latencies: list[float] = []
    queue_latencies: list[float] = []
    provider_latencies: list[float] = []
    planner_wall_latencies: list[float] = []
    planner_cache_hits = 0
    structured_output_modes: Counter[str] = Counter()
    circuit_states: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        if row_index and args.pace_ms > 0:
            time.sleep(args.pace_ms / 1000.0)
        started = time.perf_counter()
        plan = planner(row["question"], context=row.get("context") or [])
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        all_latencies.append(latency_ms)
        actual_routes = {f"{route.source}.{route.operation}" for route in plan.routes}
        expected_routes = set(row["expected"])
        actual_sources = {route.source for route in plan.routes}
        required_routes = set(row.get("required_routes") or expected_routes)
        actual_roles = sorted(
            f"{route.source}.{route.operation}={route.source_role}" for route in plan.routes
        )
        actual_dependencies = sorted(
            f"{route.source}.{route.operation}={dependency}"
            for route in plan.routes for dependency in route.depends_on
        )
        gold = _gold_score(
            row, actual_routes, actual_roles=actual_roles,
            actual_dependencies=actual_dependencies,
        )
        effective_required_routes = set(gold["required_routes"])
        expected_sources = {
            route.split(".", 1)[0] for route in effective_required_routes
        }
        route_f1 = gold["route_f1"]
        allowed_sources = {
            route.split(".", 1)[0] for route in gold["allowed_routes"]
        }
        source_recall = (
            len(expected_sources & actual_sources) / len(expected_sources)
            if expected_sources else 1.0
        )
        source_precision = 1.0 if not gold["closed_world"] else (
            len(actual_sources & allowed_sources) / len(actual_sources)
            if actual_sources else (1.0 if not expected_sources else 0.0)
        )
        source_f1 = (
            2 * source_precision * source_recall / (source_precision + source_recall)
            if source_precision + source_recall else 0.0
        )

        required_route_expected += len(effective_required_routes)
        required_route_hit += len(effective_required_routes & actual_routes)
        forbidden_route_violations += len(gold["forbidden_hits"])
        acceptable_plan_hits += int(gold["matched"])

        for route_key in effective_required_routes:
            if route_key.startswith("application_state."):
                application_expected += 1
                application_hit += int(route_key in actual_routes)

        expected_service = str(row.get("service_mode") or "")
        service_ok: bool | None = None
        if expected_service:
            service_cases += 1
            service_ok = plan.intent_frame.service_mode == expected_service
            service_hits += int(service_ok)

        expected_objects = set(row.get("expected_objects") or [])
        predicted_objects = set(plan.intent_frame.answer_objects or [])
        object_ok: bool | None = None
        if expected_objects:
            object_cases += 1
            object_ok = expected_objects == predicted_objects
            object_hits += int(object_ok)

        expected_operations = set(row.get("expected_operations") or [])
        predicted_operations = set(plan.intent_frame.operations or [])
        operation_ok: bool | None = None
        if expected_operations:
            operation_cases += 1
            operation_ok = expected_operations == predicted_operations
            operation_hits += int(operation_ok)

        expected_decision = str(row.get("expected_decision") or "answer")
        decision_ok = plan.decision == expected_decision
        decision_hits += int(decision_ok)
        expected_is_clarify = expected_decision == "clarify"
        predicted_is_clarify = plan.decision == "clarify"
        clarify_expected += int(expected_is_clarify)
        clarify_predicted += int(predicted_is_clarify)
        clarify_true_positive += int(expected_is_clarify and predicted_is_clarify)

        expected_roles = list(row.get("expected_source_roles") or [])
        role_gold_ok: bool | None = None
        if expected_roles:
            role_gold_cases += 1
            role_gold_ok = expected_roles == actual_roles
            role_gold_hits += int(role_gold_ok)

        expected_dependencies = list(row.get("expected_dependencies") or [])
        dependency_gold_ok: bool | None = None
        if expected_dependencies:
            dependency_gold_cases += 1
            dependency_gold_ok = expected_dependencies == actual_dependencies
            dependency_gold_hits += int(dependency_gold_ok)

        structure = _plan_structure(plan, registered_routes, set(SOURCE_ROLES))
        factors = dict(plan.confidence_factors or {})
        llm_calls = int(_factor_metric(factors, "llm_calls", 0) or 0)
        llm_calls_total += llm_calls
        if llm_calls > args.max_llm_calls_per_question:
            llm_budget_violations += 1
        is_hard = plan.planner_engine == "hard_rule"
        if is_hard and llm_calls:
            hard_call_violations += 1
        if not is_hard:
            non_hard_latencies.append(latency_ms)

        fallback_reason = str(plan.fallback_reason or "")
        timeout_stage = str(_factor_metric(factors, "planner_timeout_stage", "") or "")
        llm_status = str(_factor_metric(factors, "llm_status", "") or "")
        circuit_state = str(
            _factor_metric(factors, "planner_circuit_state", "") or ""
        )
        circuit_rejected = bool(
            circuit_state == "open" or fallback_reason == "circuit_open"
        )
        timed_out = bool(not circuit_rejected and (
            timeout_stage or "timeout" in fallback_reason.lower()
            or "timeout" in llm_status.lower()
        ))
        used_fallback = bool(
            fallback_reason or "fallback" in str(plan.planner_engine).lower()
        )
        timeout_count += int(timed_out)
        circuit_rejected_count += int(circuit_rejected)
        fallback_count += int(used_fallback)
        prompt_tokens = int(_factor_metric(factors, "prompt_tokens", 0) or 0)
        completion_tokens = int(_factor_metric(factors, "completion_tokens", 0) or 0)
        prompt_tokens_total += prompt_tokens
        completion_tokens_total += completion_tokens
        for metric_key, destination in (
            ("planner_queue_ms", queue_latencies),
            ("planner_provider_ms", provider_latencies),
            ("planner_wall_ms", planner_wall_latencies),
        ):
            value = _factor_metric(factors, metric_key, None)
            if value is not None:
                try:
                    destination.append(float(value))
                except (TypeError, ValueError):
                    pass
        planner_cache_hits += int(bool(_factor_metric(factors, "planner_cache_hit", False)))
        structured_output_modes[str(
            _factor_metric(factors, "structured_output_mode", "<unknown>") or "<unknown>"
        )] += 1
        circuit_states[str(
            _factor_metric(factors, "planner_circuit_state", "<unknown>") or "<unknown>"
        )] += 1
        entities_ok = _required_entities_ok(
            dict(row.get("required_entities") or {}), plan
        )
        if row.get("required_entities"):
            required_entity_cases += 1
            required_entity_hits += int(entities_ok)

        failed = bool(
            not gold["matched"] or source_f1 < 1.0 or service_ok is False
            or object_ok is False or operation_ok is False or not decision_ok
            or role_gold_ok is False or dependency_gold_ok is False
            or not entities_ok or not structure["schema_valid"]
        )
        results.append({
            **row,
            "actual": sorted(actual_routes),
            "actual_sources": sorted(actual_sources),
            "route_f1": route_f1,
            "required_route_recall": gold["required_recall"],
            "gold_plan_match": gold["matched"],
            "gold_option_index": gold["option_index"],
            "missing_required_routes": gold["missing"],
            "forbidden_route_hits": gold["forbidden_hits"],
            "unexpected_routes": gold["unexpected"],
            "source_f1": source_f1,
            "service_ok": service_ok,
            "actual_service_mode": plan.intent_frame.service_mode,
            "predicted_objects": sorted(predicted_objects),
            "object_ok": object_ok,
            "predicted_operations": sorted(predicted_operations),
            "operation_ok": operation_ok,
            "decision": plan.decision,
            "decision_ok": decision_ok,
            "actual_source_roles": actual_roles,
            "source_role_gold_ok": role_gold_ok,
            "actual_dependencies": actual_dependencies,
            "dependency_gold_ok": dependency_gold_ok,
            "required_entities_ok": entities_ok,
            "planner_engine": plan.planner_engine,
            "planner_version": plan.planner_version,
            "schema_version": plan.schema_version,
            "route_model": plan.route_model,
            "repair_used": plan.repair_used,
            "fallback_reason": plan.fallback_reason,
            "timed_out": timed_out,
            "circuit_rejected": circuit_rejected,
            "planner_timeout_stage": timeout_stage,
            "planner_circuit_state": circuit_state,
            "used_fallback": used_fallback,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_calls": llm_calls,
            "structure": structure,
            "latency_ms": latency_ms,
            "failed": failed,
        })

    source_macro = sum(row["source_f1"] for row in results) / len(results)
    route_macro = sum(row["route_f1"] for row in results) / len(results)
    schema_rate = sum(row["structure"]["schema_valid"] for row in results) / len(results)
    roles_rate = sum(row["structure"]["source_roles_valid"] for row in results) / len(results)
    dag_rate = sum(row["structure"]["dag_valid"] for row in results) / len(results)
    app_recall = application_hit / application_expected if application_expected else 1.0
    service_accuracy = _ratio(service_hits, service_cases)
    object_accuracy = _ratio(object_hits, object_cases)
    operation_accuracy = _ratio(operation_hits, operation_cases)
    decision_accuracy = decision_hits / len(results)
    role_gold_accuracy = _ratio(role_gold_hits, role_gold_cases)
    dependency_gold_accuracy = _ratio(dependency_gold_hits, dependency_gold_cases)
    non_hard_p95 = _p95(non_hard_latencies)
    required_route_recall = (
        required_route_hit / required_route_expected if required_route_expected else 1.0
    )
    clarify_precision = (
        clarify_true_positive / clarify_predicted if clarify_predicted else
        (1.0 if not clarify_expected else 0.0)
    )
    clarify_recall = (
        clarify_true_positive / clarify_expected if clarify_expected else 1.0
    )
    timeout_rate = timeout_count / len(results)
    circuit_rejected_rate = circuit_rejected_count / len(results)
    fallback_rate = fallback_count / len(results)

    gates = {
        "source_set_macro_f1": source_macro >= args.min_source_f1,
        "route_set_macro_f1": route_macro >= args.min_route_f1,
        "required_route_micro_recall": (
            required_route_recall >= args.min_required_route_recall
        ),
        "forbidden_route_violations": forbidden_route_violations == 0,
        "required_entity_accuracy": required_entity_hits == required_entity_cases,
        "application_state_recall": app_recall == 1.0,
        "service_mode_accuracy": service_accuracy is None
                                 or service_accuracy >= args.min_service_accuracy,
        "answer_object_accuracy": object_accuracy is None
                                  or object_accuracy >= args.min_object_accuracy,
        "operation_accuracy": operation_accuracy is None
                              or operation_accuracy >= args.min_operation_accuracy,
        "decision_accuracy": decision_accuracy >= args.min_decision_accuracy,
        "clarify_precision": clarify_precision >= args.min_clarify_precision,
        "clarify_recall": clarify_recall >= args.min_clarify_recall,
        "schema_valid_rate": schema_rate >= args.min_schema_valid_rate,
        "source_role_structural_valid_rate": roles_rate >= args.min_structural_valid_rate,
        "dag_structural_valid_rate": dag_rate >= args.min_structural_valid_rate,
        "llm_call_budget": llm_budget_violations == 0 and hard_call_violations == 0,
        "non_hard_p95_ms": non_hard_p95 <= args.max_non_hard_p95_ms,
        "timeout_rate": timeout_rate <= args.max_timeout_rate,
        "fallback_rate": fallback_rate <= args.max_fallback_rate,
    }
    if role_gold_accuracy is not None:
        gates["source_role_gold_accuracy"] = role_gold_accuracy >= .97
    if dependency_gold_accuracy is not None:
        gates["dependency_gold_accuracy"] = dependency_gold_accuracy >= .97

    report = {
        "set_version": SET_ALIASES[args.set],
        "set_kind": {
            "reviewed_regression": "reviewed_regression_dev",
            "surface_invariance": "surface_invariance_dev",
            "blind": "private_blind",
            "human_acceptance": "user_reviewed_acceptance",
        }[args.set],
        "private_dataset_sha256": private_hash or None,
        "mode": args.mode,
        "pace_ms": args.pace_ms,
        "count": len(results),
        "source_set_macro_f1": round(source_macro, 6),
        "route_set_macro_f1": round(route_macro, 6),
        "required_route_micro_recall": round(required_route_recall, 6),
        "forbidden_route_violation_count": forbidden_route_violations,
        "acceptable_plan_match_rate": round(acceptable_plan_hits / len(results), 6),
        "required_entity_accuracy": (
            round(required_entity_hits / required_entity_cases, 6)
            if required_entity_cases else None
        ),
        "exact_route_match": round(sum(row["route_f1"] == 1.0 for row in results)
                                   / len(results), 6),
        "application_state_recall": round(app_recall, 6),
        "service_mode_accuracy": service_accuracy,
        "service_mode_labeled_count": service_cases,
        "answer_object_accuracy": object_accuracy,
        "answer_object_labeled_count": object_cases,
        "operation_accuracy": operation_accuracy,
        "operation_labeled_count": operation_cases,
        "decision_accuracy": round(decision_accuracy, 6),
        "clarify_precision": round(clarify_precision, 6),
        "clarify_recall": round(clarify_recall, 6),
        "clarify_expected_count": clarify_expected,
        "clarify_predicted_count": clarify_predicted,
        "source_role_structural_valid_rate": round(roles_rate, 6),
        "dag_structural_valid_rate": round(dag_rate, 6),
        "source_role_gold_accuracy": role_gold_accuracy,
        "dependency_gold_accuracy": dependency_gold_accuracy,
        "schema_valid_rate": round(schema_rate, 6),
        "repair_count": sum(row["repair_used"] for row in results),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_rate, 6),
        "timeout_count": timeout_count,
        "timeout_rate": round(timeout_rate, 6),
        "circuit_rejected_count": circuit_rejected_count,
        "circuit_rejected_rate": round(circuit_rejected_rate, 6),
        "prompt_tokens_total": prompt_tokens_total,
        "prompt_tokens_mean": round(prompt_tokens_total / len(results), 2),
        "completion_tokens_total": completion_tokens_total,
        "completion_tokens_mean": round(completion_tokens_total / len(results), 2),
        "llm_calls_total": llm_calls_total,
        "llm_calls_max": max(row["llm_calls"] for row in results),
        "llm_budget_violations": llm_budget_violations,
        "hard_route_call_violations": hard_call_violations,
        "non_hard_p95_ms": round(non_hard_p95, 1),
        "latency_ms": {
            "p50": round(_percentile(all_latencies, .50), 1),
            "p95": round(_percentile(all_latencies, .95), 1),
            "p99": round(_percentile(all_latencies, .99), 1),
            "max": round(max(all_latencies), 1),
        },
        "planner_observability": {
            "queue_p95_ms": round(_percentile(queue_latencies, .95), 1),
            "provider_p95_ms": round(_percentile(provider_latencies, .95), 1),
            "wall_p95_ms": round(_percentile(planner_wall_latencies, .95), 1),
            "cache_hit_count": planner_cache_hits,
            "structured_output_modes": dict(structured_output_modes),
            "circuit_states": dict(circuit_states),
        },
        "planner_engines": dict(Counter(row["planner_engine"] for row in results)),
        "planner_versions": dict(Counter(row["planner_version"] for row in results)),
        "schema_versions": dict(Counter(row["schema_version"] for row in results)),
        "route_models": dict(Counter(row["route_model"] or "<none>" for row in results)),
        "gates": gates,
        "passed": all(gates.values()),
        "failure_count": sum(row["failed"] for row in results),
    }
    # Blind results are aggregate-only by contract. Development reports retain
    # case-level diagnostics so failures can be fixed without exposing a
    # private corpus.
    if args.set != "blind":
        report["failures"] = [row for row in results if row["failed"]]
        if args.include_results:
            report["results"] = results
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.no_gate:
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
