#!/usr/bin/env python3
"""Validate and render the human-reviewed open-language router case set.

This script deliberately does not call the UI, planner, retriever, or an LLM.
It is the review/freeze gate before a test run is allowed to exist.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_route_registry import READ_SOURCE_REGISTRY, SERVICE_MODES  # noqa: E402


DEFAULT_CASES = ROOT / "docs/rag_eval/human_acceptance/router_human_candidate_v1.json"
DEFAULT_REVIEW = ROOT / "docs/rag_eval/human_acceptance/CANDIDATE_CASES_V1_REVIEW.md"
REVIEW_STATUSES = {"pending", "approved", "revise", "rejected"}
DECISIONS = {"answer", "clarify"}
SPECIAL_ROUTES = {"general_fallback.answer"}


def _as_list(value: Any, field: str, location: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{location}: {field} must be a list")
        return []
    return value


def validate(payload: dict[str, Any], *, require_frozen: bool = False) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    cases = _as_list(payload.get("cases"), "cases", "root", errors)
    expected = payload.get("expected_count")
    if expected != len(cases):
        errors.append(f"root: expected_count={expected!r}, actual={len(cases)}")

    if require_frozen and payload.get("status") != "frozen":
        errors.append("root: require-frozen needs status=frozen")

    known_routes = {route.key for route in READ_SOURCE_REGISTRY} | SPECIAL_ROUTES
    ids: set[str] = set()
    categories: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    multi_turn = 0
    total_turns = 0

    for index, case in enumerate(cases, start=1):
        location = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{location}: must be an object")
            continue
        case_id = str(case.get("case_id") or "")
        location = case_id or location
        if not case_id:
            errors.append(f"{location}: missing case_id")
        elif case_id in ids:
            errors.append(f"{location}: duplicate case_id")
        ids.add(case_id)

        review_status = case.get("review_status")
        if review_status not in REVIEW_STATUSES:
            errors.append(f"{location}: invalid review_status={review_status!r}")
        if require_frozen and review_status != "approved":
            errors.append(f"{location}: frozen set contains review_status={review_status!r}")

        category = str(case.get("primary_category") or "")
        if category not in SERVICE_MODES and category != "clarify":
            errors.append(f"{location}: invalid primary_category={category!r}")
        categories[category] += 1

        for field in ("scenario", "review_focus"):
            if not str(case.get(field) or "").strip():
                errors.append(f"{location}: missing {field}")
        if not _as_list(case.get("ui_entries"), "ui_entries", location, errors):
            errors.append(f"{location}: ui_entries cannot be empty")
        if not _as_list(case.get("expected_evidence"), "expected_evidence", location, errors):
            errors.append(f"{location}: expected_evidence cannot be empty")
        tags = _as_list(case.get("coverage_tags"), "coverage_tags", location, errors)
        tag_counts.update(str(tag) for tag in tags)

        turns = _as_list(case.get("turns"), "turns", location, errors)
        if not turns:
            errors.append(f"{location}: turns cannot be empty")
            continue
        if len(turns) > 1:
            multi_turn += 1
        total_turns += len(turns)

        for turn_index, turn in enumerate(turns, start=1):
            turn_location = f"{location}/turn{turn_index}"
            if turn.get("turn") != turn_index:
                errors.append(f"{turn_location}: non-sequential turn number")
            if not str(turn.get("question") or "").strip():
                errors.append(f"{turn_location}: missing question")
            gold = turn.get("gold")
            if not isinstance(gold, dict):
                errors.append(f"{turn_location}: missing gold object")
                continue
            service_mode = gold.get("service_mode")
            decision = gold.get("decision")
            if service_mode not in SERVICE_MODES:
                errors.append(f"{turn_location}: invalid service_mode={service_mode!r}")
            if decision not in DECISIONS:
                errors.append(f"{turn_location}: invalid decision={decision!r}")
            decision_counts[str(decision)] += 1

            route_sets: dict[str, set[str]] = {}
            for field in ("required_routes", "allowed_optional_routes", "forbidden_routes"):
                routes = _as_list(gold.get(field), field, turn_location, errors)
                route_sets[field] = {str(route) for route in routes}
                for route in routes:
                    route_counts[str(route)] += 1
                    if route not in known_routes:
                        errors.append(f"{turn_location}: unknown route {route!r} in {field}")

            required = route_sets["required_routes"]
            optional = route_sets["allowed_optional_routes"]
            forbidden = route_sets["forbidden_routes"]
            if required & optional or required & forbidden or optional & forbidden:
                errors.append(f"{turn_location}: route sets overlap")
            if decision == "clarify" and required:
                errors.append(f"{turn_location}: clarify turn cannot have required_routes")
            if decision == "answer" and not required:
                errors.append(f"{turn_location}: answer turn needs at least one required route")

            available = required | optional
            for dep_index, dependency in enumerate(gold.get("depends_on", []), start=1):
                if isinstance(dependency, str) and "<-" in dependency:
                    consumer, producer = (part.strip() for part in dependency.split("<-", 1))
                elif isinstance(dependency, list) and len(dependency) == 2:
                    producer, consumer = map(str, dependency)
                else:
                    errors.append(
                        f"{turn_location}: depends_on[{dep_index}] must be "
                        "[producer, consumer] or consumer<-producer"
                    )
                    continue
                if producer not in available or consumer not in available:
                    errors.append(
                        f"{turn_location}: dependency {producer!r}->{consumer!r} must reference selected routes"
                    )

    contrast_pairs = _as_list(payload.get("contrast_pairs"), "contrast_pairs", "root", errors)
    pair_ids: set[str] = set()
    for pair_index, pair in enumerate(contrast_pairs, start=1):
        location = f"contrast_pair[{pair_index}]"
        if not isinstance(pair, dict):
            errors.append(f"{location}: must be an object")
            continue
        pair_id = str(pair.get("pair_id") or "")
        if not pair_id or pair_id in pair_ids:
            errors.append(f"{location}: pair_id must be present and unique")
        pair_ids.add(pair_id)
        case_ids = _as_list(pair.get("case_ids"), "case_ids", location, errors)
        if len(case_ids) != 2 or len(set(case_ids)) != 2:
            errors.append(f"{location}: case_ids must contain two different cases")
        for case_id in case_ids:
            if case_id not in ids:
                errors.append(f"{location}: unknown case_id={case_id!r}")
        if not str(pair.get("contrast") or "").strip():
            errors.append(f"{location}: missing contrast")

    quota_groups = payload.get("quota_case_groups") or {}
    if not isinstance(quota_groups, dict):
        errors.append("root: quota_case_groups must be an object")
        quota_groups = {}
    group_counts: dict[str, int] = {}
    for group, case_ids in quota_groups.items():
        case_ids = _as_list(case_ids, group, "quota_case_groups", errors)
        unique_case_ids = set(map(str, case_ids))
        group_counts[str(group)] = len(unique_case_ids)
        for case_id in unique_case_ids:
            if case_id not in ids:
                errors.append(f"quota_case_groups/{group}: unknown case_id={case_id!r}")

    quota_actual = {
        "min_cases": len(cases),
        "min_multi_turn_cases": multi_turn,
        "min_contrast_pairs": len(contrast_pairs),
        "min_multi_source_dag_cases": group_counts.get("multi_source_dag", 0),
        "min_personal_vs_general_cases": group_counts.get("personal_vs_general", 0),
        "min_missing_evidence_cases": group_counts.get("missing_evidence", 0),
        "min_negation_conflict_stale_cases": group_counts.get("negation_conflict_stale", 0),
        "min_cold_warm_latency_cases": group_counts.get("cold_warm_latency", 0),
    }
    quotas = payload.get("review_quotas") or {}
    if not isinstance(quotas, dict):
        errors.append("root: review_quotas must be an object")
        quotas = {}
    for quota, actual in quota_actual.items():
        minimum = quotas.get(quota)
        if not isinstance(minimum, int):
            errors.append(f"review_quotas: missing integer {quota}")
        elif actual < minimum:
            errors.append(f"review_quotas: {quota} needs {minimum}, actual={actual}")

    stats = {
        "case_count": len(cases),
        "turn_count": total_turns,
        "multi_turn_case_count": multi_turn,
        "category_counts": dict(sorted(categories.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "coverage_tag_counts": dict(tag_counts.most_common()),
        "route_counts": dict(route_counts.most_common()),
        "contrast_pair_count": len(contrast_pairs),
        "quota_group_counts": dict(sorted(group_counts.items())),
        "quota_actual": quota_actual,
    }
    return errors, stats


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_review(payload: dict[str, Any], stats: dict[str, Any]) -> str:
    lines = [
        "# OfferClaw 顶部问答候选 Case 人工审核表 V1",
        "",
        "> 当前状态：候选审核稿。此文档只用于确认问题和金标；尚未运行 UI、路由器或检索评测。",
        "",
        "## 审核方式",
        "",
        "请逐项判断问题是否自然、场景是否真实、预期答案来源是否符合你的使用习惯。可以直接回复：",
        "",
        "```text",
        "G01 保留",
        "A04 修改：把问题改成……；预期必须同时参考……",
        "C03 删除：现实中我不会这样问",
        "新增：……",
        "```",
        "",
        "审核口径：`保留` → approved；`修改` → revise；`删除` → rejected。只有全部处理完并冻结版本后，才进入自动运行与结果对照阶段。",
        "",
        "## 当前覆盖概览",
        "",
        f"- 候选 case：{stats['case_count']} 个，共 {stats['turn_count']} 个真实提问 turn。",
        f"- 多轮 case：{stats['multi_turn_case_count']} 组。",
        f"- 同关键词/不同答案中心对照：{stats['contrast_pair_count']} 对。",
        "- 分类：" + "；".join(f"{key} {value}" for key, value in stats["category_counts"].items()) + "。",
        "- 交叉配额：" + "；".join(
            f"{key} {value}" for key, value in stats["quota_group_counts"].items()
        ) + "。",
        "- 每题均预标注 UI 入口、服务模式、回答/澄清决策、必需/可选/禁止路由、预期证据和人工关注点。",
        "",
        "## 候选题",
        "",
    ]

    category_order = ("guide", "recall", "explain", "advise", "diagnose", "clarify")
    category_titles = {
        "guide": "Guide：OfferClaw 使用指导",
        "recall": "Recall：个人事实与历史索引",
        "explain": "Explain：资料与专业知识解释",
        "advise": "Advise：多源个性化建议",
        "diagnose": "Diagnose：无结果与运行状态诊断",
        "clarify": "Clarify：应先澄清而非猜测",
    }
    for category in category_order:
        rows = [case for case in payload["cases"] if case["primary_category"] == category]
        if not rows:
            continue
        lines.extend([
            f"### {category_titles[category]}",
            "",
            "| ID | 问题/多轮对话 | 预期模式与路由 | UI 入口 | 审核重点 | 你的结论 |",
            "|---|---|---|---|---|---|",
        ])
        for case in rows:
            questions = "<br>".join(
                f"T{turn['turn']}：{turn['question']}" for turn in case["turns"]
            )
            gold_parts = []
            for turn in case["turns"]:
                gold = turn["gold"]
                routes = ", ".join(gold.get("required_routes", [])) or "先澄清，不执行检索"
                gold_parts.append(
                    f"T{turn['turn']} {gold['service_mode']}/{gold['decision']} → {routes}"
                )
            lines.append(
                "| " + " | ".join([
                    _cell(case["case_id"]),
                    _cell(questions),
                    _cell("<br>".join(gold_parts)),
                    _cell("、".join(case["ui_entries"])),
                    _cell(case["review_focus"]),
                    "□保留 □修改 □删除",
                ]) + " |"
            )
        lines.append("")

    lines.extend([
        "## 冻结前检查",
        "",
        "- [ ] 所有 case 已标记为保留、修改或删除。",
        "- [ ] 修改后的问法仍是自然用户语言，不包含路由名或实现暗示。",
        "- [ ] 个人本地事实、已批准个人记忆、通用资料、论文和模型常识的边界符合实际偏好。",
        "- [ ] 多来源题的必需来源和禁止来源已经人工确认。",
        "- [ ] 新增真实问题已经补齐同样的金标字段。",
        "- [ ] 冻结文件生成新版本号和内容哈希；旧版本不覆盖。",
        "",
        "冻结后才执行：UI 提问 → 保存原始 SSE/meta 与回答 → 自动比对路由 → 人工评价答案 → 聚合问题类型 → 决定是否修改系统。",
        "",
    ])
    return "\n".join(lines)


def freeze_payload(payload: dict[str, Any], *, approve_pending: bool, version: str) -> dict[str, Any]:
    frozen = copy.deepcopy(payload)
    approved_cases: list[dict[str, Any]] = []
    for case in frozen["cases"]:
        status = case.get("review_status")
        if status == "rejected":
            continue
        if status == "revise":
            raise ValueError(f"{case.get('case_id')}: unresolved review_status=revise")
        if status == "pending":
            if not approve_pending:
                raise ValueError(
                    f"{case.get('case_id')}: pending review; pass --approve-pending only after explicit user approval"
                )
            case["review_status"] = "approved"
        approved_cases.append(case)
    frozen["cases"] = approved_cases
    frozen["expected_count"] = len(approved_cases)
    frozen["source_version"] = payload.get("version")
    frozen["version"] = version
    frozen["status"] = "frozen"
    frozen["frozen_at"] = date.today().isoformat()
    frozen["instructions"] = "Immutable reviewed gold set. Do not edit after execution; create a new version instead."
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="?", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--render-review", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--freeze-output", type=Path)
    parser.add_argument("--frozen-version", default="router_human_frozen_v1")
    parser.add_argument("--approve-pending", action="store_true")
    args = parser.parse_args()

    with args.cases.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    errors, stats = validate(payload, require_frozen=args.require_frozen)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        return 1

    print(json.dumps({"status": "valid", **stats}, ensure_ascii=False, indent=2))
    if args.render_review:
        args.render_review.parent.mkdir(parents=True, exist_ok=True)
        args.render_review.write_text(render_review(payload, stats), encoding="utf-8")
        print(f"review_markdown={args.render_review}")
    if args.freeze_output:
        try:
            frozen = freeze_payload(
                payload,
                approve_pending=args.approve_pending,
                version=args.frozen_version,
            )
        except ValueError as exc:
            print(f"FREEZE_BLOCKED: {exc}")
            return 1
        frozen_errors, frozen_stats = validate(frozen, require_frozen=True)
        if frozen_errors:
            print("FREEZE_INVALID")
            for error in frozen_errors:
                print(f"- {error}")
            return 1
        encoded = (json.dumps(frozen, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        args.freeze_output.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_output.write_bytes(encoded)
        manifest_path = args.freeze_output.with_suffix(".manifest.json")
        manifest = {
            "version": frozen["version"],
            "source_version": frozen.get("source_version"),
            "frozen_at": frozen["frozen_at"],
            "sha256": digest,
            "case_count": frozen_stats["case_count"],
            "turn_count": frozen_stats["turn_count"],
            "multi_turn_case_count": frozen_stats["multi_turn_case_count"],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"frozen_cases={args.freeze_output}")
        print(f"freeze_manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
