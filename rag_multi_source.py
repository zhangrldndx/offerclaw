# -*- coding: utf-8 -*-
"""OfferClaw 多数据源 RAG 执行与证据契约。

本模块只执行 ``rag_query_plan.QueryPlan`` 中的白名单路由。结构化投递事实直接
渲染；向量检索仅用于个人语义记忆和策展资料。所有送入合成模型的文本在这里完成
最小必要裁剪与联系方式脱敏。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import datetime as dt
import os
import re
import time
from typing import Any, Callable

from rag_query_plan import QueryPlan
from rag_source_policy import (
    CURATED_SOURCE_TYPES,
    INTERNAL_SOURCE_TYPES,
    PERSONAL_VECTOR_SOURCE_TYPES,
    evidence_allowed,
    normalized_source_type,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PERSONAL_SOURCE_TYPES = set(PERSONAL_VECTOR_SOURCE_TYPES)
REFERENCE_SOURCE_TYPES = set(CURATED_SOURCE_TYPES)
REFERENCE_EXCLUDES = set(PERSONAL_VECTOR_SOURCE_TYPES | INTERNAL_SOURCE_TYPES)


@dataclass
class EvidenceItem:
    source_type: str
    source_id: str
    updated_at: str
    authority: str
    matched_by: str
    text: str
    route: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionResult:
    deterministic_answer: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    source_status: dict[str, dict] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    coverage: dict[str, int] = field(default_factory=dict)
    freshness: str = ""
    in_kb: bool = False
    failed_routes: list[str] = field(default_factory=list)
    missing_personal_routes: list[str] = field(default_factory=list)
    resolved_entities: dict[str, Any] = field(default_factory=dict)
    retrieval_traces: dict[str, dict] = field(default_factory=dict)
    retrieval_profile: str = ""
    index_fingerprint: str = ""
    effective_hit: bool = False
    # 门的结构化动作(answer/correct_premise),取自 reference_kb 向量路的
    # 检索章(rag_gate._finish 盖的)。个人结构化路无此语义,保持默认。
    answer_action: str = "answer"


def redact_private_text(text: str) -> str:
    """删除合成回答不需要的联系方式、账号与密钥形态；保留企业/岗位/日期。"""
    out = str(text or "")
    patterns = [
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[已脱敏邮箱]"),
        (r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[已脱敏手机号]"),
        (r"(?i)(微信|wechat|wx|qq|钉钉|联系人|hr联系方式)\s*[:：=]?\s*[A-Za-z0-9_.-]{4,}",
         r"\1：[已脱敏账号]"),
        (r"(?i)(api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*[^\s,;，；]{8,}",
         r"\1=[已脱敏密钥]"),
        (r"(?<!\d)\d{17}[0-9Xx](?!\d)", "[已脱敏证件号]"),
    ]
    for pattern, repl in patterns:
        out = re.sub(pattern, repl, out)
    return out


def _status(result: ExecutionResult, route: str, status: str, count: int,
            started: float, *, error: str = "") -> None:
    item = {"status": status, "count": count,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    if error:
        item["error"] = error[:200]
    result.source_status[route] = item
    if status == "error":
        result.failed_routes.append(route)


def _profile_plan_evidence(operations: set[str]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    now = dt.date.today().isoformat()

    if "get_plan" in operations:
        try:
            from plan_gen import summarize_plan_for_automation
            data = summarize_plan_for_automation()
            if data.get("has_plan"):
                week = data.get("current_week") or {}
                text = (
                    f"计划文件：{data.get('plan_file', '')}\n周期：{data.get('period', '')}\n"
                    f"当前周：第{week.get('n', '—')}周｜主题：{week.get('theme', '—')}｜"
                    f"交付：{week.get('deliverable', '—')}\n"
                    f"今日任务：{'；'.join(data.get('today_tasks') or []) or '未记录'}"
                )
                items.append(EvidenceItem("plan", data.get("plan_file", "plans"), now,
                                          "live_structured", "deterministic", text[:2200],
                                          "profile_plan"))
        except Exception:
            pass

    if "get_gaps" in operations:
        try:
            from application_jd_store import plan_gaps_text
            text = plan_gaps_text(max_chars=2200)
            if text:
                items.append(EvidenceItem("application_gap", "投递管理·活动JD目标", now, "live_structured",
                                          "deterministic", text, "profile_plan"))
        except Exception:
            pass

    if "get_recent_log" in operations:
        try:
            from summary_tool import extract_recent_blocks
            path = os.path.join(BASE_DIR, "daily_log.md")
            with open(path, encoding="utf-8") as f:
                text = extract_recent_blocks(f.read(), days=7)
            if text:
                items.append(EvidenceItem("log", "daily_log.md", now, "live_structured",
                                          "deterministic", text[:2200], "profile_plan"))
        except Exception:
            pass

    if "get_profile" in operations:
        try:
            path = os.path.join(BASE_DIR, "user_profile.md")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text:
                items.append(EvidenceItem("profile", "user_profile.md", now,
                                          "live_structured", "deterministic", text[:1800],
                                          "profile_plan"))
        except Exception:
            pass
    return items


def _match_snapshot_text(target: dict, match: dict) -> str:
    """把已确认匹配产物裁成项目适配所需的最小、可溯源证据。"""
    lines = [
        f"企业：{target.get('company') or '未记录'}｜岗位：{target.get('position') or '未记录'}",
        f"application_id：{target.get('application_id') or '未记录'}",
        f"jd_version_id：{target.get('jd_version_id') or '未记录'}",
        f"match_id：{match.get('match_id') or target.get('match_id') or '未记录'}",
        f"匹配结论：{match.get('status') or '未记录'}｜方向：{match.get('direction') or '未记录'}",
        "能力缺口：",
    ]
    gap_count = 0
    for category, values in (match.get("gap_list") or {}).items():
        for value in values or []:
            lines.append(f"- {category}：{value}")
            gap_count += 1
    if not gap_count:
        lines.append("- 当前匹配快照未记录明确缺口。")
    languages = (((match.get("requirement_analysis") or {})
                  .get("programming_languages") or {}).get("languages") or [])
    if languages:
        lines.append("JD 编程语言候选：" + "、".join(map(str, languages)))
    suggestions = match.get("suggestions") or []
    if suggestions:
        lines.append("已记录建议：" + "；".join(map(str, suggestions[:3])))
    return "\n".join(lines)[:2600]


def _direct_experience_evidence(targets: list[dict] | None = None,
                                limit: int = 3, *, query: str = "") -> list[EvidenceItem]:
    try:
        from applications_store import list_experiences
        records = list_experiences()
    except Exception:
        return []
    if targets is not None:
        target_pairs = {
            (str(t.get("company", "")).strip(), str(t.get("position", "")).strip())
            for t in targets if t.get("company")
        }
        records = [
            r for r in records
            if any(
                r.get("company") == company
                and (not position or not r.get("position") or r.get("position") == position)
                for company, position in target_pairs
            )
        ]
    elif query:
        # A direct experience lookup does not need an application-state route
        # solely to resolve a company already named by the user. Resolve that
        # entity against the local first-party records; if none match, keep the
        # original set rather than inventing an empty strict filter.
        try:
            from applications_store import _company_aliases, _normalized_entity_text
            query_key = _normalized_entity_text(query)
            matched = [
                record for record in records
                if any(alias and alias in query_key
                       for alias in _company_aliases(record.get("company", "")))
            ]
            if matched:
                records = matched
        except Exception:
            pass
    out = []
    for record in records[:limit]:
        text = (f"企业：{record.get('company')}｜岗位：{record.get('position')}｜"
                f"阶段：{record.get('stage')}｜日期：{record.get('date')}\n"
                f"经验总结：{record.get('summary') or '未记录'}")
        out.append(EvidenceItem(
            "experience", record.get("path", "experience_posts"), record.get("date", ""),
            "personal_first_party", "structured_filter", text[:1400],
            "application_experience",
            {"company": record.get("company", ""), "position": record.get("position", "")},
        ))
    return out


def _selected_application_targets(facts: dict) -> list[dict]:
    """只选择本题状态操作返回的实体，供后续依赖路由解析“该企业/这些岗位”。"""
    ops = set(facts.get("operations") or [])
    keys = []
    if "list_pending_submission" in ops:
        keys.append("pending_submission")
    if "list_failed" in ops:
        keys.append("failed")
    if "list_ever_applied" in ops:
        keys.append("ever_applied")
    if "list_current" in ops:
        keys.append("current")
    if not keys and "get_next_actions" in ops:
        keys.append("next_actions")
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for key in keys:
        for row in facts.get(key) or []:
            pair = (str(row.get("company", "")), str(row.get("position", "")))
            identity = ("id", str(row.get("application_id"))) if row.get("application_id") else pair
            if not pair[0] or identity in seen:
                continue
            seen.add(identity)
            item = {"company": pair[0], "position": pair[1],
                    "status": str(row.get("status", ""))}
            for field in ("application_id", "jd_id", "jd_version_id", "match_id"):
                if row.get(field):
                    item[field] = str(row[field])
            out.append(item)
    return out


def _route_query(routes: list, fallback: str, targets: list[dict]) -> str:
    subqueries = list(dict.fromkeys(
        str(route.subquery or "").strip() for route in routes
        if str(route.subquery or "").strip()
    ))
    query = "；".join(subqueries) or fallback
    if any("application_state" in route.depends_on for route in routes):
        if targets:
            context = "；".join(
                f"企业：{item['company']}，岗位：{item.get('position') or '未记录'}，"
                f"状态：{item.get('status') or '未记录'}"
                for item in targets
            )
            query += f"\n已解析的目标投递实体：{context}"
        else:
            query += "\n实时投递结果没有解析出目标企业或岗位，不得猜测具体企业。"
    return query[:1400]


def _retrieval_route_query(source: str, routes: list, fallback: str,
                           targets: list[dict]) -> str:
    """Separate semantic meaning from metadata wording for KB search.

    Prefer the planner's retrieval-ready subquery; use the compact topic only
    when no subquery was supplied. Explicit dates are removed because
    :func:`_reference_source_filter` enforces them as source metadata.
    """
    if source in {"reference_kb", "paper_kb"}:
        topics = list(dict.fromkeys(
            str(route.filters.get(key) or "").strip()
            for route in routes for key in ("topic", "technical_topic", "skill")
            if str(route.filters.get(key) or "").strip()
        ))
        semantic_queries: list[str] = []
        for route in routes:
            query = str(route.subquery or "").strip()
            for key in ("date", "date_from", "date_to"):
                value = str((route.filters or {}).get(key) or "").strip()
                if value:
                    query = query.replace(value, "").replace(value.replace("-", "/"), "")
            query = re.sub(r"\s+", " ", query).strip(" ，,；;：:")
            if query and query not in semantic_queries:
                semantic_queries.append(query)
        if not semantic_queries:
            semantic_queries.extend(topics)
        if (targets and any(
                "application_state" in route.depends_on for route in routes)):
            semantic_queries.append("；".join(
                f"企业：{item['company']}，岗位：{item.get('position') or '未记录'}，"
                f"状态：{item.get('status') or '未记录'}"
                for item in targets
            ))
        if semantic_queries:
            return "；".join(semantic_queries)[:1400]
    return _route_query(routes, fallback, targets)


def _reference_source_filter(routes: list) -> set[str] | None:
    """Compile explicit knowledge-batch dates into exact indexed sources.

    A date in this route identifies the authoritative import batch; it is not
    a semantic topic. Source filenames follow ``YYYY-MM-DD_<slug>.md``, so
    exact dates and closed ranges can be enforced through Chroma metadata.
    When the requested approved batch is absent, return a non-existent source
    sentinel instead of silently accepting a similar document from another
    date.
    """
    ranges: list[tuple[str, str]] = []
    for route in routes:
        filters = route.filters or {}
        exact = str(filters.get("date") or "").strip().replace("/", "-")
        start = str(filters.get("date_from") or "").strip().replace("/", "-")
        end = str(filters.get("date_to") or "").strip().replace("/", "-")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", exact):
            ranges.append((exact, exact))
        elif (re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
              and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end)):
            ranges.append((min(start, end), max(start, end)))
    if not ranges:
        return None

    sources: set[str] = set()
    knowledge_root = os.path.join(BASE_DIR, "knowledge_base")
    if os.path.isdir(knowledge_root):
        for _root, dirs, filenames in os.walk(knowledge_root):
            dirs[:] = [name for name in dirs if not name.startswith("_")]
            for filename in filenames:
                match = re.match(r"^(\d{4}-\d{2}-\d{2})_.*\.md$", filename)
                if match and any(start <= match.group(1) <= end for start, end in ranges):
                    sources.add(filename)
    if sources:
        return sources
    label = "_".join(f"{start}_{end}" for start, end in ranges)
    return {f"__no_approved_kb_source_for_{label}__"}


def _source_date(source: str) -> str:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_", os.path.basename(source))
    return match.group(1) if match else ""


def _experience_item_matches_targets(item: EvidenceItem, targets: list[dict]) -> bool:
    if not targets:
        return False
    haystack = f"{item.source_id}\n{item.text}\n{item.metadata.get('title', '')}".lower()
    return any(str(target.get("company", "")).lower() in haystack for target in targets)


def _direct_resume_evidence(limit: int = 3) -> list[EvidenceItem]:
    try:
        from resume_project import extract_project_blocks, load_materials
        mats = load_materials()
    except Exception:
        return []
    items: list[EvidenceItem] = []
    for guidance in mats.get("guidance", [])[:2]:
        items.append(EvidenceItem(
            "resume_rule", guidance.get("name", "resume guidance"), "",
            "personal_confirmed", "direct_file", guidance.get("content", "")[:1600],
            "resume_rules", {"content_role": "rule"},
        ))
    for example in mats.get("examples", [])[:2]:
        blocks = extract_project_blocks(example.get("content", ""), max_chars=1800)
        text = "\n\n".join(blocks) or example.get("content", "")[:1600]
        if text:
            items.append(EvidenceItem(
                "resume_rule", example.get("name", "resume example"), "",
                "personal_confirmed", "direct_file", text[:1800],
                "resume_rules", {"content_role": "example"},
            ))
    return items[:limit]


def _metadata_for_chunk(hit: dict, chunk: str) -> dict:
    for doc, meta in zip(hit.get("docs") or [], hit.get("metas") or []):
        if doc == chunk:
            return meta or {}
    return {}


def _evidence_from_hit(route: str, hit: dict, authority: str) -> list[EvidenceItem]:
    if not hit or not hit.get("in_kb"):
        return []
    items: list[EvidenceItem] = []
    for chunk in hit.get("chunks") or []:
        meta = _metadata_for_chunk(hit, chunk)
        source = str(meta.get("source") or "?")
        source_type = normalized_source_type(str(meta.get("source_type") or route), source)
        if not evidence_allowed(route, source_type, str(meta.get("owner_scope") or ""), source):
            continue
        items.append(EvidenceItem(
            source_type=source_type,
            source_id=source,
            updated_at=str(
                meta.get("crawl_date") or meta.get("updated_at")
                or _source_date(source)
            ),
            authority=authority,
            matched_by=str(hit.get("matched_by") or "hybrid"),
            text=str(chunk)[:1800],
            route=route,
            metadata={k: meta[k] for k in ("title", "owner_scope", "source_type") if k in meta},
        ))
    return items


def _dedupe_and_redact(items: list[EvidenceItem], max_items: int = 12,
                       max_chars: int = 10000) -> list[EvidenceItem]:
    seen: set[tuple[str, str]] = set()
    seen_experience_sources: set[str] = set()
    out: list[EvidenceItem] = []
    used = 0
    for item in items:
        item.text = redact_private_text(item.text)
        canonical_source = os.path.basename(item.source_id)
        if item.source_type == "experience" and canonical_source in seen_experience_sources:
            # 同一亲历文件可能同时被结构化读取和向量召回；结构化证据先加入，后续
            # 同源向量块不再重复注入模型，避免答案把一份经历标成两条独立证据。
            continue
        key = (item.source_id, re.sub(r"\s+", "", item.text)[:160])
        if key in seen or not item.text.strip():
            continue
        room = max_chars - used
        if room <= 0 or len(out) >= max_items:
            break
        item.text = item.text[:room]
        used += len(item.text)
        seen.add(key)
        if item.source_type == "experience":
            seen_experience_sources.add(canonical_source)
        out.append(item)
    return out


def execute_plan(plan: QueryPlan, question: str, top_k: int,
                 retrieve: Callable[..., dict],
                 retrieve_papers: Callable[[str, int], dict | None] | None = None) -> ExecutionResult:
    """执行多源计划。向量/论文路由并行，个人结构化来源先在本地确定性读取。"""
    result = ExecutionResult()
    by_source: dict[str, list] = {}
    for route in plan.routes:
        by_source.setdefault(route.source, []).append(route)

    if "product_help" in by_source:
        started = time.perf_counter()
        try:
            from product_help import (CARD_CAPABILITY_REGISTRY_VERSION,
                                      card_guidance, render_product_help)
            topics: list[str] = []
            capability_ids: list[str] = []
            action_request: dict[str, Any] = {}
            action = "locate"
            for route in by_source["product_help"]:
                topics.extend(str(item) for item in (route.filters.get("topics") or []))
                capability_ids.extend(str(item) for item in (
                    route.filters.get("capability_ids") or []
                ))
                action_request.update(route.filters.get("action_request") or {})
                action = str(route.filters.get("action") or action)
            topics = list(dict.fromkeys(topic for topic in topics if topic))
            capability_ids = list(dict.fromkeys(
                value for value in capability_ids if value
            ))
            result.deterministic_answer = render_product_help(
                topics, action, capability_ids=capability_ids,
                action_request=action_request or None,
            )
            result.sources.append("OfferClaw 功能说明（本地）")
            result.coverage["product_help"] = 1
            result.resolved_entities["capability_topics"] = topics
            result.resolved_entities["capability_ids"] = capability_ids
            result.resolved_entities["requested_action"] = action
            result.resolved_entities["action_request"] = action_request
            result.resolved_entities["card_capability_registry_version"] = (
                CARD_CAPABILITY_REGISTRY_VERSION
            )
            result.resolved_entities["card_guidance"] = card_guidance(
                topics, capability_ids=capability_ids,
            )
            _status(result, "product_help", "ok", 1, started)
        except Exception as exc:
            _status(result, "product_help", "error", 0, started, error=str(exc))

    if "system_diagnostics" in by_source:
        started = time.perf_counter()
        try:
            from system_diagnostics import (DIAGNOSTIC_REGISTRY_VERSION,
                                            render_diagnostics)
            topics: list[str] = []
            for route in by_source["system_diagnostics"]:
                topics.extend(str(item) for item in (route.filters.get("topics") or []))
            topics = list(dict.fromkeys(topic for topic in topics if topic)) or ["routing"]
            result.deterministic_answer = render_diagnostics(topics)
            result.sources.append("OfferClaw 只读诊断（本地）")
            result.coverage["system_diagnostics"] = 1
            result.resolved_entities["diagnostic_topics"] = topics
            result.resolved_entities["diagnostic_registry_version"] = DIAGNOSTIC_REGISTRY_VERSION
            _status(result, "system_diagnostics", "ok", 1, started)
        except Exception as exc:
            _status(result, "system_diagnostics", "error", 0, started, error=str(exc))

    application_targets: list[dict] = []
    if "application_state" in by_source:
        started = time.perf_counter()
        try:
            from applications_store import render_application_facts, select_application_facts
            operations = [r.operation for r in by_source["application_state"]]
            state_filters: dict[str, Any] = {}
            for route in by_source["application_state"]:
                state_filters.update(route.filters or {})
            # Model-authored scalar filters are optional hints. The original
            # question remains the local, privacy-safe entity resolver so a
            # named company cannot accidentally broaden to every application.
            state_filters.setdefault("entity_query", question)
            facts = select_application_facts(operations, filters=state_filters)
            result.deterministic_answer = render_application_facts(facts)
            application_targets = _selected_application_targets(facts)
            result.resolved_entities = {
                "applications": [dict(item) for item in application_targets],
                "companies": list(dict.fromkeys(
                    item["company"] for item in application_targets if item.get("company")
                )),
                "positions": list(dict.fromkeys(
                    item["position"] for item in application_targets if item.get("position")
                )),
            }
            application_ids = [item.get("application_id", "")
                               for item in application_targets if item.get("application_id")]
            if application_ids:
                result.resolved_entities["application_ids"] = application_ids
            if any(route.filters.get("intent") == "application_project_fit"
                   for route in by_source["application_state"]):
                result.resolved_entities["relationship_edges"] = []
            result.sources.append("applications.md（实时）")
            result.freshness = facts.get("freshness", "")
            _status(result, "application_state", "ok", facts.get("total", 0), started)
        except Exception as exc:
            _status(result, "application_state", "error", 0, started, error=str(exc))

    if "profile_plan" in by_source:
        started = time.perf_counter()
        ops = {r.operation for r in by_source["profile_plan"]}
        items = _profile_plan_evidence(ops)
        result.evidence.extend(items)
        _status(result, "profile_plan", "ok" if items else "no_evidence", len(items), started)

    if "reflection_memory" in by_source:
        started = time.perf_counter()
        try:
            from reflection_memory import search_memory
            routes = by_source["reflection_memory"]
            items: list[EvidenceItem] = []
            for route in routes:
                rows = search_memory(route.subquery or question, route.operation, limit=5)
                for row in rows:
                    status = row.get("source_status", "valid")
                    label = ("⚠️ 孤立复盘：找不到对应日期的原始执行日志，不作为画像升级依据。\n"
                             if status == "orphaned" else "")
                    source_type = row.get("source_type", "reflection_summary")
                    authority = ("live_execution_fact" if source_type == "daily_execution"
                                 else "personal_reflection_inference")
                    text_limit = 7000 if source_type == "reflection_overview" else 2600
                    items.append(EvidenceItem(
                        source_type, row.get("path") or row.get("id", ""),
                        row.get("date_to", ""), authority, row.get("matched_by", "lexical"),
                        label + row.get("text", "")[:text_limit], "reflection_memory",
                        {"reflection_id": row.get("id", ""),
                         "date_from": row.get("date_from", ""),
                         "date_to": row.get("date_to", ""),
                         "source_status": status,
                         "source_log_ids": (row.get("metadata") or {}).get("source_log_ids", []),
                         "covered_dates": (row.get("metadata") or {}).get("covered_dates", []),
                         "excluded_orphaned_reflections": (
                             row.get("metadata") or {}).get("excluded_orphaned_reflections", 0)},
                    ))
            result.evidence.extend(items)
            _status(result, "reflection_memory", "ok" if items else "no_evidence", len(items), started)
        except Exception as exc:
            _status(result, "reflection_memory", "error", 0, started, error=str(exc))

    if "application_jd" in by_source:
        started = time.perf_counter()
        try:
            from application_jd_store import (compare_versions, load_match_snapshot,
                                              load_snapshot, plan_targets,
                                              search_bound_jds)
            jd_routes = by_source["application_jd"]
            operations = {route.operation for route in jd_routes}
            explicit_versions: list[str] = []
            for route in jd_routes:
                stable_id = str(route.filters.get("stable_id") or "")
                if stable_id.startswith("jdv_"):
                    explicit_versions.append(stable_id)
                explicit_versions.extend(
                    str(value) for value in (route.filters.get("jd_version_ids") or [])
                    if str(value).startswith("jdv_")
                )
            explicit_versions = list(dict.fromkeys(explicit_versions))
            requires_application_binding = any(
                "application_state" in route.depends_on for route in jd_routes
            )
            precise_operations = {
                "get_bound_jd", "get_match_snapshot", "compare_versions",
            }
            requires_precise_target = bool(operations & precise_operations)
            # 依赖式查询目标为空时禁止回退到全部计划 JD，否则会把另一条投递的
            # JD/缺口错当成当前企业的证据。
            if explicit_versions:
                bound_targets = []
                for version_id in explicit_versions:
                    snapshot = load_snapshot(version_id)
                    if snapshot:
                        bound_targets.append({
                            "application_id": snapshot.get("application_id", ""),
                            "jd_id": snapshot.get("jd_id", ""),
                            "jd_version_id": version_id,
                            "company": snapshot.get("company", ""),
                            "position": snapshot.get("position", ""),
                        })
            elif requires_precise_target:
                # Exact JD/match/version operations are not allowed to silently
                # broaden to every plan target. A missing application binding is
                # an evidence gap, not permission to guess another application.
                bound_targets = application_targets
            else:
                bound_targets = (application_targets if requires_application_binding
                                 else application_targets or plan_targets()["included"])
            query = _route_query(by_source["application_jd"], question, bound_targets)
            rows = search_bound_jds(bound_targets, query=query) if bound_targets else []
            items = [EvidenceItem(
                "application_jd", row.get("path", row.get("jd_version_id", "")),
                row.get("captured_at", ""), "application_bound_jd", "deterministic_id",
                (f"企业：{row.get('company')}｜岗位：{row.get('position')}｜"
                 f"JD版本：{row.get('jd_version_id')}｜来源：{row.get('source_url') or '未记录'}\n"
                 f"{row.get('text', '')}"), "application_jd",
                {"application_id": row.get("application_id", ""),
                 "jd_version_id": row.get("jd_version_id", ""),
                 "company": row.get("company", ""), "position": row.get("position", "")},
            ) for row in rows]
            if "get_match_snapshot" in operations:
                for target in bound_targets:
                    if not target.get("jd_id") or not target.get("jd_version_id"):
                        continue
                    match = load_match_snapshot(
                        target.get("jd_id", ""), target.get("jd_version_id", ""),
                        target.get("match_id", ""),
                    )
                    if not match:
                        continue
                    items.append(EvidenceItem(
                        "application_match", match.get("match_id", ""),
                        match.get("generated_at", ""), "confirmed_match_snapshot",
                        "deterministic_id", _match_snapshot_text(target, match),
                        "application_jd",
                        {"application_id": target.get("application_id", ""),
                         "jd_version_id": target.get("jd_version_id", ""),
                         "match_id": match.get("match_id", ""),
                         "company": target.get("company", ""),
                         "position": target.get("position", ""),
                         "evidence_kind": "match_snapshot"},
                    ))
            result.evidence.extend(items)
            if any(route.operation == "compare_versions" for route in by_source["application_jd"]):
                for target in bound_targets:
                    if not target.get("jd_id"):
                        continue
                    diff = compare_versions(target["jd_id"])
                    if diff.get("status") == "ok":
                        result.evidence.append(EvidenceItem(
                            "application_jd_diff", target.get("jd_id", ""), "",
                            "application_bound_jd", "deterministic_diff",
                            f"{target.get('company')}｜{target.get('position')}\n{diff.get('diff', '')}",
                            "application_jd", {"application_id": target.get("application_id", "")},
                        ))
            for target in bound_targets:
                app_id = target.get("application_id", "")
                version_id = target.get("jd_version_id", "")
                match_id = target.get("match_id", "")
                if app_id and version_id:
                    result.resolved_entities.setdefault("relationship_edges", []).append({
                        "from": app_id, "relation": "bound_to", "to": version_id,
                    })
                if version_id and match_id:
                    result.resolved_entities.setdefault("relationship_edges", []).append({
                        "from": version_id, "relation": "evaluated_by", "to": match_id,
                    })
            result.resolved_entities.setdefault("jd_version_ids", []).extend(
                target.get("jd_version_id", "") for target in bound_targets
                if target.get("jd_version_id")
            )
            result.resolved_entities["jd_version_ids"] = list(dict.fromkeys(
                result.resolved_entities.get("jd_version_ids") or []
            ))
            _status(result, "application_jd", "ok" if items else "no_evidence", len(items), started)
        except Exception as exc:
            _status(result, "application_jd", "error", 0, started, error=str(exc))

    experience_is_bound = False
    if "application_experience" in by_source:
        started = time.perf_counter()
        experience_is_bound = any(
            "application_state" in route.depends_on
            for route in by_source["application_experience"]
        )
        direct = _direct_experience_evidence(
            application_targets if experience_is_bound else None,
            query=question,
        )
        result.evidence.extend(direct)
        _status(result, "application_experience:direct",
                "ok" if direct else "no_evidence", len(direct), started)

    if "resume_rules" in by_source:
        started = time.perf_counter()
        direct = _direct_resume_evidence()
        result.evidence.extend(direct)
        _status(result, "resume_rules:direct", "ok" if direct else "no_evidence",
                len(direct), started)

    project_operations = {
        route.operation for route in by_source.get("project_memory", [])
    }
    if project_operations & {"list_catalog", "list_approved", "rank_for_application"}:
        started = time.perf_counter()
        try:
            from project_memory import (list_project_catalog, render_project_evidence,
                                        render_project_record)
            filters: dict[str, Any] = {}
            for route in by_source["project_memory"]:
                if route.operation in {"list_catalog", "list_approved"}:
                    filters.update(route.filters)
            fit_has_jd_evidence = any(
                item.route == "application_jd"
                and item.source_type in {"application_jd", "application_match"}
                for item in result.evidence
            )
            fit_ready = bool(application_targets and fit_has_jd_evidence)
            records = ([] if "rank_for_application" in project_operations and not fit_ready
                       else list_project_catalog(
                           open_source_only=bool(filters.get("open_source_only")),
                           approved_only="list_approved" in project_operations,
                           resume_confirmed_only=bool(filters.get("resume_confirmed_only")),
                       ))
            for record in records:
                result.evidence.append(EvidenceItem(
                    "project_context",
                    str(record.get("source_path") or record.get("project_id")),
                    str(record.get("updated_at") or ""),
                    "personal_confirmed",
                    "deterministic_catalog",
                    (render_project_evidence(record)
                     if "rank_for_application" in project_operations
                     else render_project_record(record)),
                    "project_memory",
                    {**dict(record),
                     "evidence_kind": ("project_fit_candidate"
                                       if "rank_for_application" in project_operations
                                       else "project_catalog")},
                ))
                if "rank_for_application" in project_operations:
                    for target in application_targets:
                        if target.get("application_id") and record.get("project_id"):
                            result.resolved_entities.setdefault("relationship_edges", []).append({
                                "from": target["application_id"],
                                "relation": "compare_project_candidate",
                                "to": record["project_id"],
                            })
            _status(
                result, "project_memory", "ok" if records else "no_evidence",
                len(records), started,
                **({"error": "missing_application_or_jd_binding"}
                   if "rank_for_application" in project_operations and not fit_ready else {}),
            )
            if records:
                result.in_kb = True
        except Exception as exc:
            _status(result, "project_memory", "error", 0, started, error=str(exc))

    route_queries = {
        source: _retrieval_route_query(source, routes, question, application_targets)
        for source, routes in by_source.items()
    }
    if "rank_for_application" in project_operations:
        relation_evidence = "\n".join(
            item.text for item in result.evidence
            if item.route == "application_jd"
            and item.source_type in {"application_jd", "application_match"}
        )
        route_queries["project_memory"] = (
            route_queries.get("project_memory", question)
            + "\n用于项目适配比较的目标 JD 与匹配依据：\n"
            + relation_evidence[:3000]
        )[:4200]
    tasks: dict[str, Callable[[], dict | None]] = {}
    if ("application_experience" in by_source
            and (not experience_is_bound or application_targets)):
        company_filter = {
            item["company"] for item in application_targets if item.get("company")
        } if experience_is_bound else set()
        tasks["application_experience"] = lambda: retrieve(
            route_queries["application_experience"], 3,
            source_types={"experience"},
            metadata_filters={
                **({"company": company_filter} if company_filter else {}),
                "owner_scope": {"personal"},
            },
            allow_paper_route=False,
            _answerability_shadow_route="application_experience")
    if ("project_memory" in by_source
            and project_operations & {"search", "rank_for_application"}):
        tasks["project_memory"] = lambda: retrieve(
            route_queries["project_memory"],
            5 if "rank_for_application" in project_operations else 3,
            source_types={"project_context"},
            metadata_filters={"owner_scope": {"personal"}}, allow_paper_route=False,
            _answerability_shadow_route="project_memory")
    if "resume_rules" in by_source:
        tasks["resume_rules"] = lambda: retrieve(
            route_queries["resume_rules"], 2,
            source_types={"resume_rule", "resume"},
            metadata_filters={"owner_scope": {"personal"}}, allow_paper_route=False,
            _answerability_shadow_route="resume_rules")
    if "reference_kb" in by_source:
        reference_sources = _reference_source_filter(by_source["reference_kb"])
        tasks["reference_kb"] = lambda: retrieve(
            route_queries["reference_kb"], min(5, max(1, top_k)),
            exclude_source_types=REFERENCE_EXCLUDES,
            metadata_filters={
                "owner_scope": {"curated"},
                **({"source": reference_sources} if reference_sources else {}),
            },
            allow_paper_route=False,
            _answerability_shadow_route="reference_kb",
            _answerability_shadow_question=question,
            # 判据判定对象 = 用户原问题:错误前提住在原问题里,subquery 是
            # 中性化改写,拿它判 relation 永远判不出 correct_premise
            judgement_question=question)
    if "paper_kb" in by_source and retrieve_papers:
        tasks["paper_kb"] = lambda: retrieve_papers(route_queries["paper_kb"], 3)

    authorities = {
        "application_experience": "personal_first_party",
        "project_memory": "personal_confirmed",
        "resume_rules": "personal_confirmed",
        "reference_kb": "curated_knowledge",
        "paper_kb": "paper",
    }
    hits: dict[str, dict | None] = {}
    started_by_route = {name: time.perf_counter() for name in tasks}
    if tasks:
        parallel_setting = os.environ.get("RAG_PARALLEL_ROUTES", "").strip()
        if parallel_setting:
            parallel_routes = parallel_setting == "1"
        else:
            # 本地 SentenceTransformer（尤其 macOS MPS）在多线程并发 encode 时可能
            # 发生原生层崩溃。远程 embedding 默认并行；本地 provider 默认串行，仍可
            # 通过 RAG_PARALLEL_ROUTES=1 显式压测覆盖。
            try:
                from rag_tools import get_embedding_config
                parallel_routes = get_embedding_config().get("provider") != "local"
            except Exception:
                parallel_routes = False
        if parallel_routes and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(tasks)),
                                    thread_name_prefix="rag-route") as pool:
                future_map = {pool.submit(fn): name for name, fn in tasks.items()}
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        hits[name] = future.result()
                    except Exception as exc:
                        hits[name] = None
                        _status(result, name, "error", 0, started_by_route[name], error=str(exc))
        else:
            for name, fn in tasks.items():
                try:
                    hits[name] = fn()
                except Exception as exc:
                    hits[name] = None
                    _status(result, name, "error", 0, started_by_route[name], error=str(exc))

    for name, hit in hits.items():
        if hit and hit.get("retrieval_trace"):
            result.retrieval_traces[name] = hit["retrieval_trace"]
            result.retrieval_profile = (
                result.retrieval_profile or str(hit.get("retrieval_profile") or "")
            )
            result.index_fingerprint = (
                result.index_fingerprint or str(hit.get("index_fingerprint") or "")
            )
            result.effective_hit = bool(
                result.effective_hit or hit.get("effective_hit", False)
            )
        items = _evidence_from_hit(name, hit or {}, authorities[name])
        if name == "application_experience" and any(
            "application_state" in route.depends_on
            for route in by_source.get("application_experience", [])
        ):
            # 依赖式经验检索必须与第一阶段解析出的企业一致；不能因为语义相似把
            # 另一家公司的面经当成当前公司的“注意事项”。
            items = [
                item for item in items
                if _experience_item_matches_targets(item, application_targets)
            ]
        result.evidence.extend(items)
        if name not in result.source_status:
            _status(result, name, "ok" if items else "no_evidence", len(items),
                    started_by_route[name])
        if items and hit and hit.get("in_kb"):
            result.in_kb = True
            if name == "reference_kb":
                # 收章:动作随证据一起交给下游合成(P0 交接契约的 multi 段)
                result.answer_action = str(hit.get("answer_action") or "answer")

    result.evidence = _dedupe_and_redact(result.evidence)
    for item in result.evidence:
        result.coverage[item.route] = result.coverage.get(item.route, 0) + 1
        if item.source_id not in result.sources:
            result.sources.append(item.source_id)
        if item.updated_at and item.updated_at > result.freshness:
            result.freshness = item.updated_at
    if result.deterministic_answer and "application_state" in by_source:
        result.coverage["application_state"] = 1
    personal_sources = {
        route.source for route in plan.routes
        if route.source in {
            "application_state", "application_experience", "application_jd", "profile_plan",
            "reflection_memory", "project_memory", "resume_rules",
        }
    }
    result.missing_personal_routes = sorted(
        source for source in personal_sources if result.coverage.get(source, 0) == 0
    )
    return result


def multi_source_messages(question: str, plan: QueryPlan, execution: ExecutionResult,
                          *, stance: str = "") -> list[dict]:
    """构造最小必要的综合回答上下文；确定性投递事实在外层直接输出。

    ``stance`` 是门动作对应的合同文本(如纠偏规则),由调用方(rag_gate)按
    ``execution.answer_action`` 注入——合同文本的唯一源留在 rag_gate,
    这里只负责把它放进 system 内容,不自造第二份措辞。"""
    role_by_source = {route.source: route.source_role for route in plan.routes}
    evidence_text = []
    for i, item in enumerate(execution.evidence, start=1):
        evidence_text.append(
            f"[证据{i}｜{item.route}｜{item.source_type}｜{item.source_id}｜"
            f"来源角色={role_by_source.get(item.route, 'supporting_context')}｜"
            f"权威={item.authority}｜更新={item.updated_at or '未记录'}]\n{item.text}"
        )
    missing = "、".join(execution.failed_routes) or "无"
    missing_personal = "、".join(execution.missing_personal_routes) or "无"
    sections = " → ".join(plan.answer_sections)
    deterministic = execution.deterministic_answer or "（本题没有结构化投递事实段）"
    entity_bound_advice = any(
        route.filters.get("intent") == "application_advice"
        and "application_state" in route.depends_on
        for route in plan.routes
    )
    resolved_label = "；".join(
        f"{item.get('company')}｜{item.get('position') or '岗位未记录'}"
        for item in execution.resolved_entities.get("applications", [])
        if item.get("company")
    )
    entity_instruction = (
        "本题含依赖式投递建议：必须先以确定性事实中的企业和岗位为对象，再按企业/岗位"
        "分别组织注意事项；个人经验与参考资料要分开标注。若目标实体为空或某企业没有"
        "对应证据，必须明确说未记录，不能用其他企业经验替代。"
        f"已解析目标为：{resolved_label or '空'}。各小节标题必须原样写出已解析的企业和岗位名，"
        "不得使用“上述企业”“该岗位”等代词替代。\n"
        if entity_bound_advice else ""
    )
    project_fit_workflow = any(
        route.operation == "rank_for_application" for route in plan.routes
    )
    project_fit_instruction = (
        "本题是关系感知的项目适配比较。必须严格沿已解析的投递 → 活动 JD → 已确认匹配快照"
        " → 已确认个人项目证据作答。先写目标企业、岗位、application_id、JD 版本；然后只从"
        "证据中的项目候选选一个主推荐，可给一个备选；逐条说明它覆盖了哪些 JD 要求、哪些能力"
        "仍只是弱证据；最后原样区分匹配快照中的硬门槛/技能/经历缺口。若目标投递、活动 JD、"
        "匹配快照或项目证据任一缺失，必须指出缺失环节，不得凭通用知识推荐项目。不得把项目"
        "未记录的技术、指标或成果写成已实现。"
        f"已解析目标为：{resolved_label or '空'}。\n"
        if project_fit_workflow else ""
    )
    reference_requested = any(route.source == "reference_kb" for route in plan.routes)
    reference_status = execution.source_status.get("reference_kb", {}).get("status", "")
    general_knowledge_instruction = ""
    if reference_requested and reference_status in {"no_evidence", "unavailable", "error"}:
        general_knowledge_instruction = (
            "本题明确要求概念/技术解释，但策展资料路由没有通过证据门。你可以在完成个人事实段后，"
            "用模型通用知识补足这一诉求；该部分标题必须是“⚠️ 通用知识补充（未经知识库验证）”，"
            "且不得给通用知识伪造证据编号或来源。个人事实仍只能来自已提供证据。\n"
        )
    system = (
        "你是 OfferClaw 的多源个人求职助手。除下述显式通用知识降级外，回答只能使用"
        "下面提供的结构化事实与证据；"
        "证据不足时明确说未记录，不得编造企业、岗位、状态、日期、官网链接或个人经历。\n"
        "冲突优先级：用户确认的当前画像与实时执行事实 > 投递绑定的 JD 版本 > 用户亲历 > "
        "复盘推断 > 个人项目/简历规则 > 策展资料 > 论文。\n"
        "外层会原样展示“确定性投递事实”，你不要重复或改写其中字段。"
        "回答‘做过什么’以每日执行事实为准；回答‘当前掌握什么’以用户确认画像为准。"
        "若只有练习/复盘证据而画像未确认，必须写‘已有实践证据，但用户尚未确认掌握’。"
        "来源角色为 filter_source 的证据只能用于筛选对象，不能成为回答主体；"
        "validation_source 只能用于核对结论；supporting_context 必须与 answer_source 分开标注。"
        "孤立复盘不得用于断言能力。能力缺口必须区分“记录事实”和“基于经验的推断”；概念知识标注证据编号；"
        "建议必须能追溯到个人事实或参考资料。\n"
        f"没有找到个人证据的路由：{missing_personal}。这些来源必须明确写“未找到个人记录”，"
        "不得用通用知识替代或暗示用户做过相关事情。\n"
        f"{general_knowledge_instruction}"
        f"{entity_instruction}"
        f"{project_fit_instruction}"
        f"建议结构：{sections}。失败数据源：{missing}。\n\n"
        + (stance + "\n\n" if stance else "")
        + "========== 确定性投递事实（只读，不得改写） ==========\n"
        f"{deterministic}\n\n========== 其余证据 ==========\n"
        + ("\n\n".join(evidence_text) if evidence_text else "（没有检索到其余可靠证据）")
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


def product_help_messages(question: str, plan: QueryPlan,
                          execution: ExecutionResult) -> list[dict]:
    """Build a question-focused Guide prompt from registered product facts.

    The route model chooses a finite capability.  This second, read-only step
    answers the user's concrete doubt instead of dumping the entire card copy.
    All implementation claims must remain grounded in ``product_help.py``.
    """
    topics = list(execution.resolved_entities.get("capability_topics") or [])
    guidance = execution.deterministic_answer or "（没有可用的注册功能事实）"
    system = (
        "你是 OfferClaw 的只读产品使用向导。只能依据下面的注册功能事实回答，不能臆测"
        "接口、文件、数据来源、自动化行为或尚未开放的功能。\n"
        "先直接回答用户真正问的具体疑点；若一句中有多个问号或多个疑点，逐项回答。"
        "不要原样输出整张功能卡片，不要罗列与问题无关的所有输入、步骤、限制和边界。"
        "用户比较两个入口时，必须分别说明产物粒度、实际处理链、审查/审批/保存方式，"
        "并直接回答它们是否重复、是否会触发相同 Agent；不要把共享输入误写成相同功能。"
        "当用户纠正先前的产品说明时，先明确承认需要更正，再用注册事实替换旧说法；"
        "不要继续复述已失效的入口。注册事实已经给出处理链时，要明确作答，不能说"
        "‘当前说明未记录’。比较多 Agent 协作时，必须点出准确按钮，并区分独立 Agent、"
        "硬校验器与普通单次模型生成。"
        "用户问字段/下拉框来源时，必须说明数据从哪个本地文件或 API 视图产生、选择后"
        "写到哪里、会不会反向修改来源；事实未提供就明确说当前说明未记录。"
        "回答保持简洁，通常使用一段结论加 2—5 个要点；顶部问答始终只读。\n"
        f"已选能力：{'、'.join(topics) or '未标注'}。\n"
        "========== 注册功能事实 ==========" + "\n" + guidance
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


def evidence_fallback(execution: ExecutionResult) -> str:
    """无合成模型时的可审计降级输出。"""
    parts = []
    if execution.deterministic_answer:
        parts.append(execution.deterministic_answer)
    if execution.evidence:
        parts.append("### 检索证据\n" + "\n\n".join(
            f"- [{item.route}] {item.source_id}：{item.text[:400]}"
            for item in execution.evidence
        ))
    if execution.failed_routes:
        parts.append("### 未完成的数据源\n- " + "\n- ".join(execution.failed_routes))
    if execution.missing_personal_routes:
        parts.append("### 未找到个人记录\n- " + "\n- ".join(execution.missing_personal_routes))
    return "\n\n".join(parts) or "没有找到可用的个人记录或知识库证据。"
