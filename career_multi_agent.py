# -*- coding: utf-8 -*-
"""Selective Plan/Resume author and Critic workflows for OfferClaw.

Only four graph roles call a model: Plan Agent, Plan Critic Agent, Resume Agent
and Resume Critic Agent. Context assembly, hard validation, review routing and
artifact persistence remain ordinary workflow tools. No approved user file is
written before the explicit approval interrupt resumes.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from structured_llm import call_structured


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, ".offerclaw")
DEFAULT_DB_PATH = os.path.join(RUNTIME_DIR, "career_threads.sqlite3")
RESUME_DRAFT_DIR = os.path.join(BASE_DIR, "resume_drafts")
SCOPE_TTL_SECONDS = 30 * 60
ACTIVE_PLAN_STATUSES = {"准备投递", "已投递", "等待反馈", "面试中"}
TERMINAL_STATUSES = {"不投递", "已拒绝", "主动放弃", "已 Offer"}
PLAN_DRAFT_DIR = os.path.join(RUNTIME_DIR, "unreviewed_plan_drafts")


class _ScopeResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_application_ids: list[str] = Field(default_factory=list, max_length=100)
    exclude_application_ids: list[str] = Field(default_factory=list, max_length=100)
    priority_application_ids: list[str] = Field(default_factory=list, max_length=100)
    reason: str = Field(default="", max_length=300)


def _merge_records(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Replay-safe reducer for metrics and artifact revision history."""
    out: list[dict] = []
    seen: set[str] = set()
    for item in [*(left or []), *(right or [])]:
        value = dict(item)
        identity = str(value.get("call_id") or value.get("revision_id") or "")
        if not identity:
            identity = _stable_id("record_", value, 24)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(value)
    return out


class AgentState(TypedDict, total=False):
    thread_id: str
    agent_task: Literal["portfolio_plan", "resume"]
    workflow_version: str
    scope_snapshot: dict
    application_id: str
    jd_version_id: str
    resume_scope: Literal["full_resume", "project_section"]
    project_repo_url: str
    project_text: str
    project_name: str
    stage_project_memory: bool
    resume_source_text: str
    start_date: str
    end_date: str
    revision_note: str
    user_requirements: list[dict]
    change_request: str
    remember_preference: bool
    agent_budget: dict
    agent_context: dict
    agent_artifact: dict
    artifact_spec: dict
    artifact_revision: int
    artifact_diff: dict
    hard_validation: dict
    review_contract: dict
    review_report: dict
    previous_review_report: dict
    review_mode: Literal["generate", "review_existing", "manual_edit"]
    review_round: int
    auto_revision_count: int
    context_fresh: bool
    commit_mode: Literal["approved", "unreviewed"]
    pending_approval: dict
    approval_decision: str
    edited_by_user: bool
    agent_metrics: Annotated[list[dict], _merge_records]
    revision_history: Annotated[list[dict], _merge_records]
    fatal_error: str


_scope_cache: dict[str, dict] = {}
_scope_lock = threading.RLock()
_runtime_lock = threading.RLock()
_graph = None
_checkpointer = None
_sqlite_connection = None


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _token(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _stable_id(prefix: str, value: Any, size: int = 16) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:size]


def _json_object(text: str) -> dict:
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON 对象")
    value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型返回值不是 JSON 对象")
    return value


def _llm_config() -> tuple[str, str]:
    from day1_api_starter import get_llm_config

    cfg = get_llm_config() or {}
    return str(cfg.get("api_key") or ""), str(cfg.get("api_key_env") or "LLM_API_KEY")


def _call_llm(messages: list[dict], *, max_tokens: int) -> str:
    from plan_gen import call_llm_plain

    key, env_name = _llm_config()
    if not key:
        raise RuntimeError(f"{env_name} 未配置")
    return call_llm_plain(messages, key, max_tokens=max_tokens)


def _safe_profile() -> dict:
    """Return only fields needed for planning/writing; contacts never leave local code."""
    from profile_loader import load_profile

    profile = load_profile()
    allowed = {
        "姓名", "学校", "毕业时间", "学历", "专业", "所在地", "可接受地域", "方向优先级", "目标岗位类型",
        "行业偏好", "明确不做", "工作性质偏好", "熟练技能", "会用技能",
        "技能证据", "项目技能", "项目数量", "实习数量", "英语自评",
        "工作日每日可投入小时", "周末每日可投入小时", "每周可投入小时",
    }
    return {key: value for key, value in profile.items() if key in allowed}


def _redact_text(value: str) -> str:
    """Remove contact/account data that is irrelevant to planning and review."""
    text = str(value or "")
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱已脱敏]", text)
    text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[手机号已脱敏]", text)
    text = re.sub(r"(?i)(微信|wechat|QQ|账号)\s*[:：]?\s*[A-Za-z0-9_.-]{5,}", r"\1：[账号已脱敏]", text)
    return text


def _candidate_rows() -> list[dict]:
    from application_jd_store import active_jd_identity, snapshot_summary
    from applications_store import application_fact_views

    rows: list[dict] = []
    for app in application_fact_views():
        summary = snapshot_summary(
            app.get("jd_id", ""), app.get("jd_version_id", ""), app.get("match_id", ""),
        )
        match = summary.get("match") or {}
        ready = bool(summary.get("linked") and match.get("match_id"))
        rows.append({
            **app,
            "jd": {key: value for key, value in summary.items() if key != "match"},
            "match": match,
            "ready": ready,
            "match_stale": bool(summary.get("match_stale")),
            "gap_count": sum(len(v or []) for v in (match.get("gap_list") or {}).values()),
        })
    priority = {"high": 0, "medium": 1, "low": 2}
    stage = {"面试中": 0, "等待反馈": 1, "已投递": 2, "准备投递": 3, "已评估": 4}
    representatives: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (
        priority.get(item.get("plan_priority", "medium"), 1),
        stage.get(item.get("status", ""), 9), item.get("date", ""),
        item.get("application_id", ""),
    )):
        identity = active_jd_identity(row)
        if not identity:
            continue
        representative = representatives.get(identity)
        if representative:
            row["duplicate_of"] = representative.get("application_id", "")
        else:
            representatives[identity] = row
    return rows


def _candidate_revision(row: dict) -> str:
    return _stable_id("rev_", {
        "application_id": row.get("application_id"),
        "date": row.get("date"),
        "status": row.get("status"),
        "include_in_plan": row.get("include_in_plan"),
        "plan_priority": row.get("plan_priority"),
        "jd_version_id": row.get("jd_version_id"),
        "match_id": row.get("match_id"),
        "content_hash": (row.get("jd") or {}).get("content_hash"),
        "profile_fingerprint": __import__("application_jd_store").profile_fingerprint(),
    })


def _public_candidate(row: dict, *, reason: str = "") -> dict:
    return {
        "application_id": row.get("application_id", ""),
        "company": row.get("company", ""),
        "position": row.get("position", ""),
        "status": row.get("status", ""),
        "date": row.get("date", ""),
        "plan_priority": row.get("plan_priority", "medium"),
        "include_in_plan": bool(row.get("include_in_plan")),
        "jd_linked": bool((row.get("jd") or {}).get("linked")),
        "jd_version_id": row.get("jd_version_id", ""),
        "match_id": row.get("match_id", ""),
        "match_stale": bool(row.get("match_stale")),
        "gap_count": int(row.get("gap_count") or 0),
        "next_action": row.get("next_action", ""),
        "duplicate_of": row.get("duplicate_of", ""),
        "reason": reason,
    }


def _mentioned_ids(instruction: str, rows: list[dict]) -> set[str]:
    text = instruction.lower()
    ids: set[str] = set()
    direction_terms = {
        "ai": ("ai", "人工智能", "大模型"),
        "agent": ("agent", "智能体"),
        "rag": ("rag", "检索增强"),
        "算法": ("算法",),
        "后端": ("后端", "backend"),
        "产品": ("产品",),
    }
    requested_terms = [alts for key, alts in direction_terms.items() if key in text]
    def company_mentioned(value: str) -> bool:
        company = (value or "").strip().lower()
        if not company:
            return False
        aliases = {company}
        # Chinese application rows often store a business-unit suffix while users
        # naturally say only the brand ("华为" vs "华为计算产品线").
        if re.match(r"^[\u4e00-\u9fff]{2,}", company):
            aliases.add(company[:2])
        for suffix in ("计算产品线", "产品线", "科技", "集团", "公司"):
            if company.endswith(suffix) and len(company) > len(suffix) + 1:
                aliases.add(company[:-len(suffix)])
        return any(alias and alias in text for alias in aliases)

    named_company_ids = {row["application_id"] for row in rows
                         if company_mentioned(str(row.get("company", "")))}
    for row in rows:
        haystack = " ".join([
            str(row.get("company", "")), str(row.get("position", "")),
            str((row.get("match") or {}).get("direction", "")),
        ]).lower()
        position = str(row.get("position", "")).lower()
        direction_match = (not requested_terms or any(
            any(term in haystack for term in alts) for alts in requested_terms))
        if named_company_ids and row["application_id"] in named_company_ids and direction_match:
            ids.add(row["application_id"])
        elif not named_company_ids and position and len(position) >= 3 and position in text:
            ids.add(row["application_id"])
        elif not named_company_ids and requested_terms and direction_match:
            ids.add(row["application_id"])
    return ids


def _rule_scope(instruction: str, rows: list[dict], default_ids: set[str]) -> tuple[set[str], dict, list[str], bool]:
    text = (instruction or "").strip()
    if not text:
        return set(default_ids), {}, [], False

    selected = set(default_ids)
    reasons: dict[str, str] = {}
    warnings: list[str] = []
    mentioned = _mentioned_ids(text, rows)
    exclusive = bool(re.search(r"(?:只|仅|这次先|先考虑|范围是)", text))
    if mentioned:
        if exclusive:
            selected = set(mentioned)
        else:
            selected |= mentioned
        for app_id in mentioned:
            reasons[app_id] = "命中范围描述中的企业、岗位或方向"

    include_statuses = {s for s in ACTIVE_PLAN_STATUSES | {"已评估"} if s in text}
    if include_statuses and not re.search(r"(?:不要|不考虑|排除)[^。；，,]*" +
                                          "(?:" + "|".join(map(re.escape, include_statuses)) + ")", text):
        status_ids = {r["application_id"] for r in rows if r.get("status") in include_statuses}
        selected = status_ids if exclusive and not mentioned else selected | status_ids

    # Exclusion clauses are intentionally local: never treat a standalone "不" as a global negation.
    for clause in re.findall(r"(?:不要|不考虑|排除)([^。；，,]+)", text):
        for row in rows:
            hay = f"{row.get('company','')} {row.get('position','')} {row.get('status','')}".lower()
            if any(token and token.lower() in hay for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9+#.-]*|[\u4e00-\u9fff]{2,}", clause)):
                selected.discard(row["application_id"])
                reasons[row["application_id"]] = "被范围描述显式排除"

    priority_ids: list[str] = []
    if "优先" in text:
        before = text.split("优先", 1)[1]
        priority_ids = sorted(_mentioned_ids(before, rows))
        if not priority_ids:
            warnings.append("检测到“优先”表达，但无法确定对应投递；建议检查范围预览")

    # Relative date only applies to recorded future action/status dates.  Missing deadlines stay visible.
    relative = re.search(r"(\d{1,2}|一|两|二|三|四)\s*周内", text)
    ambiguous = False
    if relative:
        cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4}
        weeks = cn.get(relative.group(1), int(relative.group(1)) if relative.group(1).isdigit() else 2)
        today, limit = _dt.date.today(), _dt.date.today() + _dt.timedelta(days=weeks * 7)
        dated: set[str] = set()
        for row in rows:
            try:
                day = _dt.date.fromisoformat(str(row.get("date") or ""))
            except ValueError:
                continue
            if today <= day <= limit:
                dated.add(row["application_id"])
        if dated:
            selected &= dated
        else:
            warnings.append("没有记录可用于该相对时间范围的未来日期，未按时间静默排除投递")
            ambiguous = True

    if not mentioned and not include_statuses and re.search(r"(?:相关|合适|重点|为主|近期)", text):
        ambiguous = True
    return selected, {"reasons": reasons, "priority_application_ids": priority_ids}, warnings, ambiguous


def _llm_resolve_scope(instruction: str, rows: list[dict], current_ids: set[str]) -> tuple[set[str], list[str], str]:
    candidates = [_public_candidate(row) for row in rows]
    messages = [
        {"role": "system", "content": (
            "你是 OfferClaw 的计划范围解析器。只把用户描述映射到候选 application_id，"
            "不得补充候选列表外的 ID，不生成计划。只返回 JSON："
            '{"include_application_ids":[],"exclude_application_ids":[],"priority_application_ids":[],"reason":""}'
        )},
        {"role": "user", "content": json.dumps({
            "instruction": instruction,
            "current_rule_selection": sorted(current_ids),
            "candidates": candidates,
        }, ensure_ascii=False)},
    ]
    def scope_caller(messages_arg, max_tokens, _temperature, _model):
        return _call_llm(messages_arg, max_tokens=max_tokens)

    resolved, meta = call_structured(
        _ScopeResolution, messages, max_tokens=800, timeout_seconds=5,
        repair=True, caller=scope_caller,
    )
    if resolved is None:
        raise ValueError("范围模型结构无效：" + ",".join(meta.errors))
    data = resolved.model_dump(mode="json")
    allowed = {row["application_id"] for row in rows}
    include = {str(x) for x in data.get("include_application_ids") or [] if str(x) in allowed}
    exclude = {str(x) for x in data.get("exclude_application_ids") or [] if str(x) in allowed}
    priority = [str(x) for x in data.get("priority_application_ids") or [] if str(x) in allowed]
    selected = (include or set(current_ids)) - exclude
    return selected, priority, str(data.get("reason") or "")[:300]


def preview_portfolio_scope(*, instruction: str = "", application_ids: list[str] | None = None,
                            include_profile_goals: bool = True,
                            allow_llm: bool | None = None) -> dict:
    """Build and cache a read-only portfolio scope preview."""
    rows = _candidate_rows()
    by_id = {row["application_id"]: row for row in rows}
    explicit = {str(x) for x in (application_ids or []) if str(x)}
    warnings: list[str] = []
    default_ids = {
        row["application_id"] for row in rows
        if (row.get("ready") and not row.get("duplicate_of") and row.get("include_in_plan")
            and row.get("status") not in TERMINAL_STATUSES)
    }
    if explicit:
        selected = explicit & set(by_id)
        missing = explicit - set(by_id)
        if missing:
            warnings.append("部分 application_id 不存在：" + "、".join(sorted(missing)))
        rule_meta, ambiguous = {"reasons": {}, "priority_application_ids": []}, False
    else:
        selected, rule_meta, more_warnings, ambiguous = _rule_scope(instruction, rows, default_ids)
        warnings.extend(more_warnings)

    resolver_mode = "rule"
    scope_llm_calls = 0
    llm_reason = ""
    if allow_llm is None:
        allow_llm = os.environ.get("CAREER_AGENT_SCOPE_LLM", "1") == "1"
    if ambiguous and allow_llm:
        try:
            selected, priorities, llm_reason = _llm_resolve_scope(instruction, rows, selected)
            rule_meta["priority_application_ids"] = priorities
            resolver_mode = "llm"
            scope_llm_calls = 1
        except Exception as exc:
            resolver_mode = "fallback"
            warnings.append(f"范围模型不可用，已回退规则结果：{str(exc)[:160]}")

    duplicate_mapping = {
        row["application_id"]: row["duplicate_of"] for row in rows
        if row.get("duplicate_of")
    }
    if duplicate_mapping:
        warnings.append(
            f"已折叠 {len(duplicate_mapping)} 条同公司、同岗位、同一 JD 内容的重复保存；"
            "每组只保留一条代表记录进入计划"
        )
    redirected = {app_id for app_id in selected if app_id in duplicate_mapping}
    if redirected:
        selected = {
            duplicate_mapping.get(app_id, app_id) for app_id in selected
        }
        for app_id in redirected:
            representative_id = duplicate_mapping[app_id]
            reason = (rule_meta.get("reasons") or {}).get(app_id)
            if reason:
                rule_meta.setdefault("reasons", {}).setdefault(representative_id, reason)
        warnings.append(
            f"已将 {len(redirected)} 条同公司、同岗位、同一 JD 内容的重复保存"
            "折叠为对应代表记录"
        )
    priorities = rule_meta.get("priority_application_ids") or []
    rule_meta["priority_application_ids"] = list(dict.fromkeys(
        duplicate_mapping.get(app_id, app_id) for app_id in priorities
    ))

    included_rows: list[dict] = []
    excluded_rows: list[tuple[dict, str]] = []
    for row in rows:
        app_id = row["application_id"]
        if row.get("duplicate_of"):
            excluded_rows.append((
                row,
                "同公司、同岗位、同一 JD 内容的重复保存；已保留一条代表记录",
            ))
            continue
        if app_id in selected:
            if not (row.get("jd") or {}).get("linked"):
                selected.discard(app_id)
                excluded_rows.append((row, "JD 未关联"))
                continue
            if not (row.get("match") or {}).get("match_id"):
                selected.discard(app_id)
                excluded_rows.append((row, "匹配快照缺失"))
                continue
            if row.get("match_stale"):
                warnings.append(f"{row.get('company')}·{row.get('position')} 的匹配可能过期，缺口不进入学习任务")
            if row.get("status") in TERMINAL_STATUSES:
                warnings.append(f"{row.get('company')}·{row.get('position')} 为终态，仅因手工范围而纳入")
            included_rows.append(row)
        else:
            reason = (rule_meta.get("reasons") or {}).get(app_id)
            if not reason:
                if not (row.get("jd") or {}).get("linked"):
                    reason = "JD 未关联"
                elif not (row.get("match") or {}).get("match_id"):
                    reason = "匹配快照缺失"
                elif row.get("status") in TERMINAL_STATUSES:
                    reason = "终态投递默认排除"
                elif not row.get("include_in_plan"):
                    reason = "未纳入学习计划"
                else:
                    reason = "不在本次描述范围"
            excluded_rows.append((row, reason))

    profile_fp = __import__("application_jd_store").profile_fingerprint()
    frozen = [{"application_id": row["application_id"],
               "revision": _candidate_revision(row)} for row in included_rows]
    scope_hash = _stable_id("psh_", {
        "instruction": instruction.strip(), "targets": frozen,
        "include_profile_goals": bool(include_profile_goals),
        "profile_fingerprint": profile_fp,
    })
    snapshot_id = _stable_id("scope_", {"scope_hash": scope_hash, "created_nonce": uuid.uuid4().hex})
    snapshot = {
        "scope_snapshot_id": snapshot_id,
        "instruction": instruction.strip(),
        "application_ids": [row["application_id"] for row in included_rows],
        "jd_version_ids": [row.get("jd_version_id", "") for row in included_rows],
        "match_ids": [row.get("match_id", "") for row in included_rows],
        "profile_fingerprint": profile_fp,
        "include_profile_goals": bool(include_profile_goals),
        "priority_application_ids": rule_meta.get("priority_application_ids") or [],
        "resolver_mode": resolver_mode,
        "resolver_reason": llm_reason,
        "created_at": _now(),
        "expires_at_epoch": time.time() + SCOPE_TTL_SECONDS,
        "scope_hash": scope_hash,
        "target_revisions": frozen,
        "targets": included_rows,
        "warnings": list(dict.fromkeys(warnings)),
        "scope_llm_calls": scope_llm_calls,
    }
    with _scope_lock:
        _scope_cache[snapshot_id] = snapshot
        now = time.time()
        for key in list(_scope_cache):
            if _scope_cache[key].get("expires_at_epoch", 0) < now:
                _scope_cache.pop(key, None)
    return {
        "scope_snapshot_id": snapshot_id,
        "instruction": snapshot["instruction"],
        "resolver_mode": resolver_mode,
        "resolver_reason": llm_reason,
        "included": [_public_candidate(row, reason=(rule_meta.get("reasons") or {}).get(
            row["application_id"], "已纳入计划并具备活动 JD/匹配快照")) for row in included_rows],
        "excluded": [_public_candidate(row, reason=reason) for row, reason in excluded_rows],
        "warnings": snapshot["warnings"],
        "include_profile_goals": bool(include_profile_goals),
        "scope_hash": scope_hash,
        "estimated_llm_calls": {"scope": scope_llm_calls, "generation": 1,
                                "total": scope_llm_calls + 1},
    }


def get_scope_snapshot(scope_snapshot_id: str, *, validate_freshness: bool = True) -> dict:
    with _scope_lock:
        snapshot = _scope_cache.get(scope_snapshot_id)
    if not snapshot:
        raise KeyError("范围快照不存在或已过期，请重新预览")
    if snapshot.get("expires_at_epoch", 0) < time.time():
        with _scope_lock:
            _scope_cache.pop(scope_snapshot_id, None)
        raise KeyError("范围快照已过期，请重新预览")
    if validate_freshness:
        live = {row["application_id"]: row for row in _candidate_rows()}
        for frozen in snapshot.get("target_revisions") or []:
            row = live.get(frozen["application_id"])
            if not row or _candidate_revision(row) != frozen["revision"]:
                raise RuntimeError("投递、JD、匹配或画像已变化，请重新确认范围")
    return snapshot


def _recent_text(path: str, max_chars: int) -> str:
    try:
        with open(os.path.join(BASE_DIR, path), encoding="utf-8") as handle:
            return handle.read()[-max_chars:]
    except OSError:
        return ""


def _number(value: Any, default: float) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else default


def _plan_capacity(profile: dict, memory_context: dict) -> dict[str, float]:
    """Resolve configured capacity without letting the model invent availability."""
    weekday = _number(profile.get("工作日每日可投入小时"), 3.0)
    weekend = _number(profile.get("周末每日可投入小时"), 5.0)
    for item in memory_context.get("semantic") or []:
        if item.get("memory_key") != "execution_capacity":
            continue
        value = item.get("value") or {}
        if isinstance(value, dict):
            weekday = _number(value.get("weekday_hours"), weekday)
            weekend = _number(value.get("weekend_hours"), weekend)
    return {"weekday": min(24.0, max(0.5, weekday)),
            "weekend": min(24.0, max(0.5, weekend))}


def _memory_revision(memory_context: dict) -> list[tuple[str, Any]]:
    return sorted((str(item.get("memory_id") or ""), item.get("version"))
                  for item in memory_context.get("semantic") or [])


def _portfolio_context(snapshot: dict) -> dict:
    from application_jd_store import load_snapshot, plan_gap_items
    from applications_store import application_fact_views, list_experiences
    from plan_gen import load_latest_plan, retrieve_learning_resources

    targets = snapshot.get("targets") or []
    stale_ids = {row["application_id"] for row in targets if row.get("match_stale")}
    gaps = []
    for item in plan_gap_items(targets):
        gap_id = _stable_id("gap_", {"text": item["text"], "sources": item["sources"]})
        kind = "hard_constraint" if not item["learnable"] else (
            "shared_gap" if item["count"] > 1 else "role_specific_gap")
        if any(src.get("application_id") in stale_ids for src in item["sources"]):
            kind = "stale_gap"
        gaps.append({**item, "gap_id": gap_id, "kind": kind,
                     "learnable": bool(item["learnable"] and kind != "stale_gap")})

    profile = _safe_profile()
    profile_goals = []
    if snapshot.get("include_profile_goals"):
        for value in profile.get("方向优先级") or []:
            profile_goals.append({"gap_id": _stable_id("pg_", value),
                                  "text": f"持续强化个人目标方向：{value}",
                                  "kind": "profile_general", "learnable": True})

    gap_text = "\n".join(f"- {g['text']}" for g in gaps if g["learnable"])
    resources = retrieve_learning_resources(gap_text, top_files=6) if gap_text else []
    for resource in resources:
        resource["resource_id"] = _stable_id("res_", {
            "source": resource.get("source"), "title": resource.get("title")})

    jds = []
    per_jd_chars = max(500, min(1800, 24000 // max(1, len(targets))))
    truncation_warnings = []
    for target in targets:
        snap = load_snapshot(target.get("jd_version_id", ""), jd_id=target.get("jd_id", ""))
        full_jd = _redact_text(str(snap.get("jd_text") or ""))
        if len(full_jd) > per_jd_chars:
            truncation_warnings.append(
                f"{target.get('company')}·{target.get('position')} 的 JD 原文按上下文预算截取；稳定 ID 与缺口关系未省略")
        jds.append({
            "application_id": target.get("application_id"),
            "company": target.get("company"), "position": target.get("position"),
            "status": target.get("status"), "date": target.get("date"),
            "plan_priority": target.get("plan_priority"),
            "next_action": target.get("next_action"),
            "jd_version_id": target.get("jd_version_id"),
            "jd_excerpt": full_jd[:per_jd_chars],
        })

    terminal_by_id = {row["application_id"]: row for row in application_fact_views()
                      if row.get("status") in {"已拒绝", "主动放弃"}}
    experiences = []
    for exp in list_experiences():
        app_id = exp.get("application_id")
        if app_id in terminal_by_id:
            item = {key: exp.get(key) for key in (
                "application_id", "company", "position", "stage", "date", "summary")}
            item["summary"] = _redact_text(str(item.get("summary") or ""))[:1200]
            experiences.append(item)
        if len(experiences) >= 5:
            break

    previous = load_latest_plan()
    from memory_layers import build_memory_context
    memory_context = build_memory_context(
        purpose="advice",
        context={
            "task_type": "portfolio_plan",
            "application_ids": [row.get("application_id") for row in targets],
            "directions": [row.get("position") for row in targets],
            "skills": [gap.get("text") for gap in gaps if gap.get("learnable")],
        },
    )
    capacity = _plan_capacity(profile, memory_context)
    result = {
        "scope": {key: snapshot.get(key) for key in (
            "scope_snapshot_id", "instruction", "application_ids", "jd_version_ids",
            "priority_application_ids", "include_profile_goals", "scope_hash")},
        "applications": jds,
        "gaps": gaps,
        "profile_goals": profile_goals,
        "profile": profile,
        "resources": resources,
        "historical_experiences": experiences,
        "recent_log": _redact_text(_recent_text("daily_log.md", 3500)),
        "previous_plan": (previous or {}).get("content", "")[:2500],
        "memory_context": memory_context,
        "capacity": capacity,
        "truncation_warnings": truncation_warnings,
    }
    result["_context_fingerprint"] = _stable_id("ctx_", {
        "scope_hash": snapshot.get("scope_hash"),
        "profile": snapshot.get("profile_fingerprint"),
        "memory": _memory_revision(memory_context),
        "capacity": capacity,
    }, 32)
    return result


def _plan_messages(context: dict, state: AgentState) -> list[dict]:
    from review_protocol import PlanSpec

    schema = PlanSpec.model_json_schema()
    return [
        {"role": "system", "content": (
            "你是 OfferClaw Portfolio Plan Agent。你面对的是一组已经由代码冻结的投递/JD，"
            "不得自行增加或排除目标。优先处理高优先级与临近面试，再合并共同缺口；"
            "高优先级岗位的特有致命缺口必须单列。硬门槛只能放 decision_warnings，不能生成学习任务。"
            "schedule 必须逐日连续覆盖整个周期，每个具体 task_id 只排一次；每天最多2个核心任务和1个可选任务。"
            "任务依赖写入 dependency_ids，且依赖任务必须排在当前任务之前。"
            "投入不得超过 context.capacity，并且每周至少保留15%机动时间。"
            "你只负责生成或修改计划，无权评价计划是否通过。修改时保留不受影响的稳定 task_id。"
            "所有 ID 必须逐字取自输入，只返回一个 JSON 对象，不要 Markdown，不要代码围栏。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}"
        )},
        {"role": "user", "content": json.dumps({
            "start_date": state.get("start_date"), "end_date": state.get("end_date"),
            "user_requirements": state.get("user_requirements") or [],
            "current_plan_spec": state.get("artifact_spec") or None,
            "critic_review": state.get("review_report") or None,
            "context": context,
        }, ensure_ascii=False)},
    ]


def _validate_plan(data: dict, context: dict) -> tuple[dict, list[str]]:
    del context  # Source/reference validity belongs to the hard validator.
    from review_protocol import PlanSpec

    try:
        parsed = PlanSpec.model_validate(data)
    except Exception as exc:
        return data, [f"PlanSpec 无法解析：{str(exc)[:500]}"]
    return parsed.model_dump(mode="json"), []


def _render_plan(data: dict, context: dict, state: AgentState) -> str:
    lines = ["# OfferClaw 求职组合学习计划", "",
             f"> 计划范围：`{context['scope']['scope_snapshot_id']}`",
             f"> 计划周期：{state.get('start_date')} → {state.get('end_date')}", "",
             "## 组合策略", "", data.get("portfolio_summary") or "—", ""]
    if data.get("priority_decisions"):
        lines += ["## 投递优先级判断", ""]
        apps = {x["application_id"]: x for x in context["applications"]}
        for item in data["priority_decisions"]:
            application_id = str(item.get("application_id") or "")
            app = apps.get(application_id) or {}
            label = (f"{app.get('company')}｜{app.get('position')}" if app
                     else f"未知投递 `{application_id or 'missing'}`")
            lines.append(f"- **{label}**：{item.get('reason','—')}")
        lines.append("")

    task_map = {}
    for title, field in (("共同缺口任务", "common_gap_tasks"),
                         ("岗位特有任务", "role_specific_tasks")):
        lines += [f"## {title}", ""]
        tasks = data.get(field) or []
        if not tasks:
            lines.append("- 本范围暂无该类任务。")
        for task in tasks:
            task_map[task["task_id"]] = task
            lines += [f"### {task['title']}",
                      f"- 目标：{task.get('objective','—')}",
                      f"- 时间：{task.get('time_window','—')} · 预计 {task.get('estimated_hours','—')} 小时",
                      f"- 交付物：{task.get('deliverable','—')}",
                      f"- 溯源：gap={','.join(task.get('gap_ids') or [])} · "
                      f"application={','.join(task.get('application_ids') or []) or '画像通用'} · "
                      f"JD={','.join(task.get('jd_version_ids') or []) or '画像通用'}", ""]

    lines += ["## 各投递非学习型动作", ""]
    apps = {x["application_id"]: x for x in context["applications"]}
    for action in data.get("application_actions") or []:
        app = apps.get(action.get("application_id"), {})
        lines.append(
            f"- **{app.get('company','')}｜{app.get('position','')}**：{action.get('action','—')}"
            f"（{action.get('time_window','—')}；交付物：{action.get('deliverable','—')}）")
    if data.get("decision_warnings"):
        lines += ["", "## 投递决策与硬门槛提醒", ""]
        for warning in data["decision_warnings"]:
            lines.append(f"- {warning.get('text','—')}")

    week_names = "一二三四五六日"
    lines += ["", "## 每日执行安排", ""]
    for index, day in enumerate(data.get("schedule") or [], start=1):
        try:
            date = _dt.date.fromisoformat(str(day.get("date") or ""))
            label = f"{date.month:02d}-{date.day:02d} 周{week_names[date.weekday()]}"
        except ValueError:
            label = str(day.get("date") or "日期无效")
        core_ids = [str(value) for value in day.get("core_task_ids") or []]
        optional_ids = [str(value) for value in day.get("optional_task_ids") or []]
        lines += [f"### D{index}（{label}）",
                  f"- **今日主线标签**：{day.get('focus') or '恢复与机动'}",
                  "- **核心任务**："]
        if core_ids:
            for number, task_id in enumerate(core_ids, start=1):
                task = task_map.get(task_id, {})
                lines.append(
                    f"  {number}. {task.get('title') or task_id}（预计 {task.get('estimated_hours', 0):g}h；"
                    f"验收：{task.get('deliverable') or '完成记录'}） <!-- task_id: {task_id} -->")
        else:
            lines.append("  - 无核心任务，保留恢复与机动时间。")
        lines.append("- **可选任务**：")
        if optional_ids:
            for task_id in optional_ids:
                task = task_map.get(task_id, {})
                lines.append(
                    f"  - {task.get('title') or task_id}（预计 {task.get('estimated_hours', 0):g}h）"
                    f" <!-- task_id: {task_id} -->")
        else:
            lines.append("  - 无")
        date_obj = None
        try:
            date_obj = _dt.date.fromisoformat(str(day.get("date") or ""))
        except ValueError:
            pass
        cap_key = "weekend" if date_obj and date_obj.weekday() >= 5 else "weekday"
        daily_cap = float((context.get("capacity") or {}).get(cap_key, 3.0))
        lines += [f"- **预计投入**：{float(day.get('planned_hours') or 0):g}h / 当日上限 {daily_cap:g}h", ""]

    lines += ["", "---", "", "## 确定性溯源附录", ""]
    for app in context["applications"]:
        lines.append(f"- {app['company']}｜{app['position']}：`{app['application_id']}` → `{app['jd_version_id']}`")
    for gap in context["gaps"]:
        sources = "、".join(f"{x['application_id']}→{x['jd_version_id']}" for x in gap["sources"])
        lines.append(f"- `{gap['gap_id']}` {gap['text']}：{sources}")
    for resource in context["resources"]:
        lines.append(f"- `{resource['resource_id']}` {resource.get('title') or resource.get('source')}：`{resource.get('source')}`")
    return "\n".join(lines).rstrip() + "\n"


def _fallback_plan(context: dict, message: str) -> str:
    lines = ["# OfferClaw 求职组合计划（确定性降级草稿）", "",
             f"> Agent 生成失败：{message}", "", "## 当前目标", ""]
    for app in context.get("applications") or []:
        lines.append(f"- {app['company']}｜{app['position']}｜{app['status']}｜JD `{app['jd_version_id']}`")
    lines += ["", "## 已确认的可学习缺口", ""]
    for gap in context.get("gaps") or []:
        if gap.get("learnable"):
            lines.append(f"- {gap['text']}（`{gap['gap_id']}`）")
    lines += ["", "> 该草稿只用于展示错误和事实范围，不能批准保存；修复模型配置后重新生成。"]
    return "\n".join(lines) + "\n"


def _artifact_id(kind: str, content: str, input_hash: str) -> str:
    return _stable_id("artifact_", {"kind": kind, "content": content, "input_hash": input_hash})


def _metric(node: str, started: float, **extra) -> dict:
    return {"call_id": _token("metric"), "node": node,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "at": _now(), **extra}


def _reserve_llm_call(state: AgentState, role: str) -> dict:
    budget = dict(state.get("agent_budget") or {})
    used = int(budget.get("used_llm_calls", 0))
    maximum = int(budget.get("max_llm_calls", 4))
    if used >= maximum:
        raise RuntimeError(f"LLM 调用预算已耗尽，无法运行 {role}")
    budget["used_llm_calls"] = used + 1
    budget["last_reserved_for"] = role
    return budget


def _revision_event(kind: str, revision: int, before: str, after: str,
                    source: str) -> dict:
    from review_protocol import artifact_diff

    diff = artifact_diff(before, after)
    return {"revision_id": _token("revision"), "kind": kind,
            "artifact_revision": revision, "source": source,
            "before_hash": hashlib.sha256(before.encode("utf-8")).hexdigest(),
            "after_hash": hashlib.sha256(after.encode("utf-8")).hexdigest(),
            "diff": diff, "at": _now()}


def _plan_author_agent_node(state: AgentState) -> dict:
    started = time.perf_counter()
    context = state.get("agent_context") or _portfolio_context(state["scope_snapshot"])
    before = str((state.get("agent_artifact") or {}).get("content_md") or "")
    revising = bool(before and state.get("review_report"))
    budget = dict(state.get("agent_budget") or {})
    reserved = False
    try:
        budget = _reserve_llm_call(state, "portfolio_plan_agent")
        reserved = True
        raw = _call_llm(_plan_messages(context, state), max_tokens=5000)
        data, errors = _validate_plan(_json_object(raw), context)
        if errors:
            raise ValueError("；".join(errors[:6]))
        from review_protocol import normalize_plan_schedule

        data = normalize_plan_schedule(data, state.get("start_date", ""),
                                       state.get("end_date", ""))
        if revising and state.get("artifact_spec"):
            old_ids = {str(item.get("task_id"))
                       for field in ("common_gap_tasks", "role_specific_tasks")
                       for item in (state["artifact_spec"].get(field) or [])
                       if item.get("task_id")}
            new_ids = {str(item.get("task_id"))
                       for field in ("common_gap_tasks", "role_specific_tasks")
                       for item in (data.get(field) or []) if item.get("task_id")}
            if old_ids and not (old_ids & new_ids):
                raise ValueError("Plan Agent 修改时丢失了全部稳定 task_id")
        content = _render_plan(data, context, state)
        revision = int(state.get("artifact_revision") or 0) + 1
        input_hash = _stable_id("input_", {"scope": state["scope_snapshot"]["scope_hash"],
                                             "dates": [state.get("start_date"), state.get("end_date")],
                                             "requirements": state.get("user_requirements") or []})
        artifact = {
            "artifact_id": _artifact_id("portfolio_plan", content, input_hash),
            "kind": "portfolio_plan", "status": "reviewing", "content_md": content,
            "scope_snapshot_id": state["scope_snapshot"]["scope_snapshot_id"],
            "input_hash": input_hash,
            "artifact_revision": revision,
            "source_refs": [{"source_type": "application_jd",
                             "application_id": app["application_id"],
                             "jd_version_id": app["jd_version_id"]}
                            for app in context["applications"]],
            "validation": {"author": "ok", "errors": [], "task_count": len(
                data.get("common_gap_tasks", [])) + len(data.get("role_specific_tasks", []))},
            "created_at": _now(), "saved_path": "",
        }
        diff = _revision_event("portfolio_plan", revision, before, content,
                               "critic_revision" if revising else "initial_generation")
        if revising and diff["before_hash"] == diff["after_hash"]:
            artifact["status"] = "needs_revision"
            artifact["validation"]["no_progress"] = True
    except Exception as exc:
        content = _fallback_plan(context, str(exc)[:300])
        input_hash = _stable_id("input_", state["scope_snapshot"].get("scope_hash"))
        artifact = {
            "artifact_id": _artifact_id("portfolio_plan", content, input_hash),
            "kind": "portfolio_plan", "status": "generation_error", "content_md": content,
            "scope_snapshot_id": state["scope_snapshot"]["scope_snapshot_id"],
            "input_hash": input_hash, "source_refs": [],
            "artifact_revision": int(state.get("artifact_revision") or 0),
            "validation": {"author": "error", "valid": False, "errors": [str(exc)[:500]]},
            "created_at": _now(), "saved_path": "",
        }
        return {"agent_context": context, "agent_artifact": artifact,
                "agent_budget": budget,
                "agent_metrics": [_metric("portfolio_plan_agent", started,
                                           llm_calls=1 if reserved else 0,
                                           error=True)]}
    return {"agent_context": context, "agent_artifact": artifact,
            "artifact_spec": data, "artifact_revision": revision,
            "artifact_diff": diff["diff"], "agent_budget": budget,
            "auto_revision_count": int(state.get("auto_revision_count") or 0) + (1 if revising else 0),
            "revision_history": [diff],
            "agent_metrics": [_metric("portfolio_plan_agent", started, llm_calls=1,
                                      phase="revision" if revising else "generation")]}


# Compatibility for integrations that imported the old private node name.
_plan_agent_node = _plan_author_agent_node


def _resume_context(application_id: str, jd_version_id: str, *,
                    resume_scope: str = "full_resume", project_repo_url: str = "",
                    project_text: str = "", project_name: str = "",
                    resume_source_text: str = "") -> dict:
    """Freeze evidence for either a full resume or one project section."""
    if resume_scope not in {"full_resume", "project_section"}:
        raise ValueError("resume_scope 必须是 full_resume 或 project_section")

    app: dict[str, Any] = {}
    snap: dict[str, Any] = {"jd_text": "", "content_hash": ""}
    match: dict[str, Any] = {}
    if application_id:
        from application_jd_store import load_match_snapshot, load_snapshot
        from applications_store import get_application

        app = get_application(application_id) or {}
        if not app:
            raise KeyError("找不到 application_id")
        active_version = str(app.get("jd_version_id") or "")
        if not active_version:
            raise ValueError("该投递未关联活动 JD")
        if jd_version_id and jd_version_id != active_version:
            raise RuntimeError("请求的 JD 版本不是该投递的活动版本")
        jd_version_id = active_version
        snap = load_snapshot(jd_version_id, jd_id=app.get("jd_id", "")) or {}
        if not snap:
            raise ValueError("活动 JD 文件不存在")
        match = load_match_snapshot(
            app.get("jd_id", ""), jd_version_id, app.get("match_id", "")) or {}
        if not match:
            raise ValueError("该投递缺少已确认匹配快照")
    elif resume_scope == "full_resume":
        raise ValueError("完整简历必须选择一条已绑定活动 JD 的投递")

    from jd_parser import analyze_jd
    jd_text = str(snap.get("jd_text") or "")
    jd_analysis = (analyze_jd(jd_text).model_dump(mode="json")
                   if jd_text else {"keywords": [], "requirements": []})

    project_sources: list[dict[str, Any]] = []
    source_origin = "OfferClaw 项目事实摘要"
    if project_repo_url or str(project_text or "").strip():
        from resume_project import gather_material

        gathered = gather_material(repo_url=project_repo_url, text=project_text)
        if gathered.get("status") != "ok":
            raise ValueError(gathered.get("error") or "项目素材获取失败")
        project_evidence = str(gathered.get("material") or "")
        source_origin = str(gathered.get("origin") or project_repo_url or "用户项目材料")
        project_sources.append({
            "evidence_ref": "project:provided",
            "source_type": "provided_project_source",
            "title": str(project_name or source_origin),
            "content": project_evidence,
            "source_path": source_origin,
            "approved_at": "current_request",
        })
    else:
        if resume_scope == "project_section":
            raise ValueError("生成项目经历前必须提供仓库地址、项目文件或项目介绍")
        project_evidence = ""

    if resume_scope == "full_resume":
        try:
            from resume_project import load_approved_project_evidence

            approved_sources = load_approved_project_evidence(
                base_dir=BASE_DIR, resume_draft_dir=RESUME_DRAFT_DIR,
            )
        except Exception:
            approved_sources = []
        known_hashes = {
            hashlib.sha256(str(item.get("content") or "").encode("utf-8")).hexdigest()
            for item in project_sources
        }
        for item in approved_sources:
            digest = str(item.get("content_hash") or hashlib.sha256(
                str(item.get("content") or "").encode("utf-8")).hexdigest())
            if digest not in known_hashes:
                project_sources.append(item)
                known_hashes.add(digest)

    if not project_sources:
        try:
            from resume_builder import _build_project_facts
            project_evidence = _build_project_facts()
        except Exception:
            project_evidence = ""
        if project_evidence:
            project_sources.append({
                "evidence_ref": "project:offerclaw",
                "source_type": "repository_project_facts",
                "title": "OfferClaw 项目事实摘要",
                "content": project_evidence,
                "source_path": "PROJECT_STATUS.md + docs/project_one_pager.md",
                "approved_at": "repository_source",
            })

    redacted_sources = []
    for item in project_sources:
        content = _redact_text(str(item.get("content") or ""))[:8000]
        if content:
            redacted_sources.append({**item, "content": content})
    project_sources = redacted_sources
    project_evidence = "\n\n".join(
        f"===== {item.get('title') or item.get('evidence_ref')} =====\n{item['content']}"
        for item in project_sources
    )[:24000]
    source_origin = "; ".join(
        str(item.get("source_path") or item.get("title") or "")
        for item in project_sources
    )[:1000]

    from memory_layers import build_memory_context
    memory_context = build_memory_context(
        purpose="advice",
        context={"task_type": "resume", "application_id": application_id,
                 "position": app.get("position", ""),
                 "direction": match.get("direction", "")},
    )
    profile = _safe_profile()
    result = {
        "resume_scope": resume_scope,
        "project_name": str(project_name or "").strip(),
        "project_origin": source_origin,
        "application": app, "jd": snap, "match": match,
        "jd_analysis": jd_analysis, "profile": profile,
        "project_evidence": project_evidence,
        "project_sources": project_sources,
        "resume_source": _redact_text(resume_source_text)[:16000],
        "memory_context": memory_context,
    }
    profile_fingerprint = __import__("application_jd_store").profile_fingerprint()
    result["_context_fingerprint"] = _stable_id("ctx_", {
        "resume_scope": resume_scope,
        "application_id": app.get("application_id", ""),
        "application_status": app.get("status", ""),
        "jd_version_id": app.get("jd_version_id", ""),
        "jd_hash": snap.get("content_hash", ""),
        "profile": profile_fingerprint,
        "project": hashlib.sha256(result["project_evidence"].encode("utf-8")).hexdigest(),
        "resume_source": hashlib.sha256(result["resume_source"].encode("utf-8")).hexdigest(),
        "memory": _memory_revision(memory_context),
    }, 32)
    return result


def _resume_context_from_state(state: AgentState, *, live_jd: bool = False) -> dict:
    application_id = state.get("application_id", "")
    jd_version_id = "" if live_jd else state.get("jd_version_id", "")
    is_legacy_default = (
        state.get("resume_scope", "full_resume") == "full_resume"
        and not state.get("project_repo_url")
        and not state.get("project_text")
        and not state.get("project_name")
        and not state.get("resume_source_text")
    )
    if is_legacy_default:
        return _resume_context(application_id, jd_version_id)
    return _resume_context(
        application_id, jd_version_id,
        resume_scope=state.get("resume_scope", "full_resume"),
        project_repo_url=state.get("project_repo_url", ""),
        project_text=state.get("project_text", ""),
        project_name=state.get("project_name", ""),
        resume_source_text=state.get("resume_source_text", ""),
    )


def _resume_author_agent_node(state: AgentState) -> dict:
    started = time.perf_counter()
    context = state.get("agent_context") or _resume_context_from_state(state)
    before = str((state.get("agent_artifact") or {}).get("content_md") or "")
    previous_spec = state.get("artifact_spec") or None
    revising = bool(before and state.get("review_report"))
    budget = dict(state.get("agent_budget") or {})
    reserved = False
    try:
        from resume_builder import (build_resume_agent_messages,
                                    build_resume_markdown,
                                    parse_resume_agent_output)
        from review_protocol import render_resume

        base = ("# 项目经历\n\n## 项目经历\n"
                if context.get("resume_scope") == "project_section" else
                str(context.get("resume_source") or "").strip() or
                build_resume_markdown(context["jd"]["jd_text"], skip_llm=True)["resume_md"])
        budget = _reserve_llm_call(state, "resume_author_agent")
        reserved = True
        raw = _call_llm(build_resume_agent_messages(
            context, base_resume_md=base,
            requirements=state.get("user_requirements") or [],
            current_spec=previous_spec if revising else None,
            review_report=state.get("review_report") if revising else None,
        ), max_tokens=5000)
        spec = parse_resume_agent_output(raw, context=context,
                                         previous_spec=previous_spec if revising else None)
        content = render_resume(spec)
        revision = int(state.get("artifact_revision") or 0) + 1
        input_hash = _stable_id("input_", {
            "resume_scope": context.get("resume_scope"),
            "application_id": (context.get("application") or {}).get("application_id", ""),
            "jd_version_id": (context.get("application") or {}).get("jd_version_id", ""),
            "project_evidence": hashlib.sha256(
                str(context.get("project_evidence") or "").encode("utf-8")).hexdigest(),
            "profile": __import__("application_jd_store").profile_fingerprint(),
            "requirements": state.get("user_requirements") or [],
        })
        artifact_kind = ("resume_project" if context.get("resume_scope") == "project_section"
                         else "resume")
        artifact = {
            "artifact_id": _artifact_id(artifact_kind, content, input_hash), "kind": artifact_kind,
            "resume_scope": context.get("resume_scope", "full_resume"),
            "status": "reviewing", "content_md": content, "scope_snapshot_id": "",
            "input_hash": input_hash,
            "artifact_revision": revision,
            "source_refs": ([
                {"source_type": "application_jd",
                 "application_id": context["application"]["application_id"],
                 "jd_version_id": context["application"]["jd_version_id"]},
            ] if context.get("application") else []) + [
                {"source_type": "profile", "source_id": "user_profile.md",
                 "updated_at": __import__("application_jd_store").profile_fingerprint()},
            ] + [{
                "source_type": item.get("source_type", "project_evidence"),
                "source_id": item.get("evidence_ref", ""),
                "source_path": item.get("source_path", ""),
                "approved_at": item.get("approved_at", ""),
            } for item in (context.get("project_sources") or [])],
            "validation": {"author": "ok"}, "created_at": _now(), "saved_path": "",
        }
        diff = _revision_event(artifact_kind, revision, before, content,
                               "critic_revision" if revising else "initial_generation")
        if revising and diff["before_hash"] == diff["after_hash"]:
            artifact["status"] = "needs_revision"
            artifact["validation"]["no_progress"] = True
    except Exception as exc:
        input_hash = _stable_id("input_", {"application_id": state.get("application_id"),
                                             "jd_version_id": state.get("jd_version_id")})
        artifact = {
            "artifact_id": _artifact_id("resume", str(exc), input_hash),
            "kind": ("resume_project" if state.get("resume_scope") == "project_section"
                     else "resume"),
            "resume_scope": state.get("resume_scope", "full_resume"),
            "status": "generation_error", "content_md": "",
            "input_hash": input_hash, "source_refs": [],
            "artifact_revision": int(state.get("artifact_revision") or 0),
            "validation": {"author": "error", "errors": [str(exc)[:500]]},
            "created_at": _now(), "saved_path": "",
        }
        return {"agent_context": context, "agent_artifact": artifact,
                "agent_budget": budget,
                "agent_metrics": [_metric("resume_author_agent", started,
                                           llm_calls=1 if reserved else 0,
                                           error=True)]}
    return {"agent_context": context, "agent_artifact": artifact,
            "artifact_spec": spec, "artifact_revision": revision,
            "artifact_diff": diff["diff"], "agent_budget": budget,
            "auto_revision_count": int(state.get("auto_revision_count") or 0) + (1 if revising else 0),
            "revision_history": [diff],
            "agent_metrics": [_metric("resume_author_agent", started, llm_calls=1,
                                      phase="revision" if revising else "generation")]}


_resume_writer_node = _resume_author_agent_node


def _plan_hard_validator_node(state: AgentState) -> dict:
    started = time.perf_counter()
    artifact = dict(state["agent_artifact"])
    if artifact.get("status") == "generation_error":
        return {"agent_artifact": artifact}
    from review_protocol import hard_validate_plan

    report = hard_validate_plan(
        state.get("artifact_spec") or {}, state.get("agent_context") or {},
        state.get("start_date", ""), state.get("end_date", ""),
        capacity=(state.get("agent_context") or {}).get("capacity") or {},
        requirements=state.get("user_requirements") or [],
    )
    validation = dict(artifact.get("validation") or {})
    validation["hard_validator"] = report
    artifact["validation"] = validation
    artifact["status"] = "reviewing"
    return {"agent_artifact": artifact, "hard_validation": report,
            "agent_metrics": [_metric("plan_hard_validator", started, llm_calls=0)]}


def _resume_hard_validator_node(state: AgentState) -> dict:
    started = time.perf_counter()
    artifact = dict(state["agent_artifact"])
    if artifact.get("status") == "generation_error":
        return {"agent_artifact": artifact}
    from resume_critic import critic_report
    from review_protocol import hard_validate_resume

    context = state["agent_context"]
    jd_keywords = (context.get("jd_analysis") or {}).get("keywords")
    deterministic = critic_report(
        artifact["content_md"], jd_text=str((context.get("jd") or {}).get("jd_text") or ""),
        profile=context["profile"], material=context.get("project_evidence", ""),
        jd_keywords=jd_keywords, use_llm=False,
    )
    report = hard_validate_resume(
        state.get("artifact_spec") or {}, artifact["content_md"], context,
        deterministic_report=deterministic,
        requirements=state.get("user_requirements") or [],
    )
    validation = dict(artifact.get("validation") or {})
    validation["hard_validator"] = report
    validation["deterministic_resume_checks"] = deterministic
    artifact["validation"] = validation
    artifact["status"] = "reviewing"
    return {"agent_artifact": artifact, "hard_validation": report,
            "agent_metrics": [_metric("resume_hard_validator", started, llm_calls=0)]}


def _critic_unavailable(state: AgentState, artifact: dict, role: str,
                        started: float, budget: dict, exc: Exception,
                        *, reserved: bool, contract: dict) -> dict:
    report = {
        "review_id": _token("review"),
        "artifact_revision": int(state.get("artifact_revision") or 0),
        "rubric_version": str(contract.get("rubric_version") or ""),
        "review_contract": contract,
        "verdict": "blocked", "findings": [], "unmet_requirement_ids": [],
        "evidence_conflicts": [], "revision_brief": [],
        "clarification_questions": [],
        "summary": f"Critic 暂不可用：{str(exc)[:300]}",
        "review_status": "unavailable",
        "final_review": int(state.get("review_round") or 0) > 0,
    }
    artifact["status"] = "review_unavailable"
    validation = dict(artifact.get("validation") or {})
    validation["critic"] = report
    artifact["validation"] = validation
    return {"agent_artifact": artifact, "review_report": report,
            "review_contract": contract,
            "agent_budget": budget,
            "agent_metrics": [_metric(role, started,
                                      llm_calls=1 if reserved else 0, error=True)]}


def _plan_critic_agent_node(state: AgentState) -> dict:
    started = time.perf_counter()
    artifact = dict(state["agent_artifact"])
    if artifact.get("status") == "generation_error":
        return {"agent_artifact": artifact}
    from plan_critic import run_plan_critic_agent
    from review_protocol import default_review_contract

    previous = state.get("review_report") or state.get("previous_review_report") or {}
    final_review = int(state.get("review_round") or 0) > 0
    contract = state.get("review_contract") or default_review_contract(
        "portfolio_plan", state.get("user_requirements") or [],
        previous_id=str((state.get("review_contract") or {}).get("contract_id") or ""),
    )
    budget = dict(state.get("agent_budget") or {})
    reserved = False
    try:
        budget = _reserve_llm_call(state, "plan_critic_agent")
        reserved = True
        report = run_plan_critic_agent(
            plan_spec=state.get("artifact_spec") or {},
            plan_md=artifact.get("content_md", ""), context=state["agent_context"],
            hard_report=state.get("hard_validation") or {}, review_contract=contract,
            artifact_revision=int(state.get("artifact_revision") or 0),
            caller=_call_llm, previous_report=previous or None,
            final_review=final_review,
        )
    except Exception as exc:
        return _critic_unavailable(state, artifact, "plan_critic_agent", started,
                                   budget, exc, reserved=reserved, contract=contract)
    artifact["status"] = {"pass": "ready_for_approval",
                          "pass_with_warnings": "ready_for_approval",
                          "revise": "needs_revision", "blocked": "blocked"}[report["verdict"]]
    validation = dict(artifact.get("validation") or {})
    validation["critic"] = report
    artifact["validation"] = validation
    return {"agent_artifact": artifact, "review_contract": report["review_contract"],
            "previous_review_report": previous, "review_report": report,
            "review_round": int(state.get("review_round") or 0) + 1,
            "agent_budget": budget,
            "agent_metrics": [_metric("plan_critic_agent", started, llm_calls=1,
                                      phase="final" if final_review else "initial")]}


def _resume_critic_agent_node(state: AgentState) -> dict:
    started = time.perf_counter()
    artifact = dict(state["agent_artifact"])
    if artifact.get("status") == "generation_error":
        return {"agent_artifact": artifact}
    from resume_critic import run_resume_critic_agent
    from review_protocol import default_review_contract

    previous = state.get("review_report") or state.get("previous_review_report") or {}
    final_review = int(state.get("review_round") or 0) > 0
    review_kind = ("resume_project" if state.get("resume_scope") == "project_section"
                   else "resume")
    contract = state.get("review_contract") or default_review_contract(
        review_kind, state.get("user_requirements") or [],
        previous_id=str((state.get("review_contract") or {}).get("contract_id") or ""),
    )
    budget = dict(state.get("agent_budget") or {})
    reserved = False
    try:
        budget = _reserve_llm_call(state, "resume_critic_agent")
        reserved = True
        report = run_resume_critic_agent(
            resume_spec=state.get("artifact_spec") or {},
            resume_md=artifact.get("content_md", ""), context=state["agent_context"],
            hard_report=state.get("hard_validation") or {}, review_contract=contract,
            artifact_revision=int(state.get("artifact_revision") or 0),
            caller=_call_llm, previous_report=previous or None,
            final_review=final_review,
        )
    except Exception as exc:
        return _critic_unavailable(state, artifact, "resume_critic_agent", started,
                                   budget, exc, reserved=reserved, contract=contract)
    artifact["status"] = {"pass": "ready_for_approval",
                          "pass_with_warnings": "ready_for_approval",
                          "revise": "needs_revision", "blocked": "blocked"}[report["verdict"]]
    validation = dict(artifact.get("validation") or {})
    validation["critic"] = report
    artifact["validation"] = validation
    return {"agent_artifact": artifact, "review_contract": report["review_contract"],
            "previous_review_report": previous, "review_report": report,
            "review_round": int(state.get("review_round") or 0) + 1,
            "agent_budget": budget,
            "agent_metrics": [_metric("resume_critic_agent", started, llm_calls=1,
                                      phase="final" if final_review else "initial")]}


_resume_critic_node = _resume_critic_agent_node


def _load_context_node(state: AgentState) -> dict:
    started = time.perf_counter()
    try:
        context = None
        if state.get("agent_task") == "portfolio_plan":
            snapshot = state.get("scope_snapshot") or {}
            if not snapshot.get("scope_snapshot_id"):
                raise ValueError("缺少已确认范围快照")
            context = _portfolio_context(snapshot)
        elif state.get("agent_task") == "resume":
            context = _resume_context_from_state(state)
        else:
            raise ValueError("不支持的 Agent 任务")
        patch = {"agent_metrics": [_metric("load_agent_context", started)]}
        if context is not None:
            patch["agent_context"] = context
            patch["context_fresh"] = True
        return patch
    except Exception as exc:
        artifact = {
            "artifact_id": _stable_id("artifact_", str(exc)),
            "kind": state.get("agent_task", ""), "status": "generation_error", "content_md": "",
            "input_hash": "", "source_refs": [],
            "validation": {"valid": False, "errors": [str(exc)[:500]]},
            "created_at": _now(), "saved_path": "",
        }
        return {"fatal_error": str(exc), "agent_artifact": artifact,
                "agent_metrics": [_metric("load_agent_context", started, error=True)]}


def _remember_review_preference(state: AgentState, requirement: dict) -> None:
    """Persist an explicitly opted-in review preference outside Critic ownership."""
    from memory_layers import record_business_event
    from memory_store import MemoryStore

    target_id = str((state.get("agent_context") or {}).get("memory_context", {}).get(
        "target_context_id") or "global")
    event = record_business_event(
        "conversation_message",
        {"role": "user", "content": f"我希望以后生成时遵守：{requirement['text']}",
         "state": "completed", "message_id": requirement["requirement_id"],
         "source_refs": []},
        actor="user", source="agent_review_preference",
        operation_id=f"remember-review:{requirement['requirement_id']}",
        entity_type="review_preference", entity_id=requirement["requirement_id"],
        target_context_id=target_id,
    )
    MemoryStore().upsert_semantic(
        f"review_preference:{requirement['requirement_id']}",
        {"statement": requirement["text"], "artifact_kind": (
            "resume_project" if state.get("resume_scope") == "project_section"
            else state.get("agent_task"))},
        memory_type="confirmed_preference", certainty="explicit", confidence=1.0,
        target_context_id=target_id, evidence_ids=[event["event_id"]],
    )


def _approval_options(artifact: dict) -> list[str]:
    status = artifact.get("status")
    if status == "generation_error":
        return ["reject"]
    if status == "ready_for_approval":
        return ["approve", "request_changes", "manual_edit", "reject"]
    return ["request_changes", "manual_edit", "save_unreviewed_draft", "reject"]


def _approval_node(state: AgentState) -> dict:
    artifact = state.get("agent_artifact") or {}
    legacy = state.get("workflow_version", "v1") != "v2"
    options = (["edit", "reject"] if artifact.get("status") == "rejected"
               else ["approve", "edit", "reject"]) if legacy else _approval_options(artifact)
    review = interrupt({
        "kind": "artifact_approval", "thread_id": state.get("thread_id"),
        "artifact_id": artifact.get("artifact_id"), "artifact_kind": artifact.get("kind"),
        "artifact_status": artifact.get("status"), "options": options,
        "validation": artifact.get("validation") or {},
        "review_report": state.get("review_report") or {},
        "review_contract": state.get("review_contract") or {},
        "artifact_diff": state.get("artifact_diff") or {},
        "message": "优先提出修改要求；完整手动编辑属于高级操作，并会重新执行硬校验和 Critic。",
    })
    review = review if isinstance(review, dict) else {"decision": str(review)}
    decision = str(review.get("decision") or "").lower()
    if legacy:
        if decision not in {"approve", "edit", "reject"}:
            return {"approval_decision": "invalid"}
        if decision == "approve" and artifact.get("status") == "rejected":
            return {"approval_decision": "invalid"}
        if decision == "edit":
            edited = str(review.get("edited_content_md") or "").strip()
            if len(edited) < 80:
                return {"approval_decision": "invalid"}
            artifact = dict(artifact)
            artifact["content_md"] = edited + "\n"
            artifact["artifact_id"] = _artifact_id(
                artifact.get("kind", ""), edited, artifact.get("input_hash", ""))
            artifact["status"] = "draft"
            artifact["saved_path"] = ""
            return {"approval_decision": "legacy_edit", "edited_by_user": True,
                    "agent_artifact": artifact}
        if decision == "reject":
            artifact = dict(artifact)
            artifact["status"] = "rejected"
            return {"approval_decision": "reject", "agent_artifact": artifact}
        return {"approval_decision": "approve", "commit_mode": "approved"}
    if decision == "edit":
        decision = "manual_edit"
    if decision not in {"approve", "request_changes", "manual_edit",
                        "save_unreviewed_draft", "reject"}:
        return {"approval_decision": "invalid"}
    if decision not in options and not (decision == "manual_edit" and "edit" in options):
        return {"approval_decision": "invalid"}
    if decision == "approve":
        return {"approval_decision": "approve", "commit_mode": "approved"}
    if decision == "save_unreviewed_draft":
        return {"approval_decision": decision, "commit_mode": "unreviewed"}
    if decision == "request_changes":
        from review_protocol import make_requirement

        change = str(review.get("change_request") or "").strip()
        if not change:
            return {"approval_decision": "invalid"}
        requirement = make_requirement(
            change, source="user_change",
            persistent=bool(review.get("remember_preference")),
        )
        requirements = [dict(item) for item in state.get("user_requirements") or []
                        if item.get("requirement_id") != requirement["requirement_id"]]
        requirements.append(requirement)
        if requirement["persistent"]:
            _remember_review_preference(state, requirement)
        artifact = dict(artifact)
        artifact["status"] = "reviewing"
        return {"approval_decision": decision, "change_request": change,
                "remember_preference": requirement["persistent"],
                "user_requirements": requirements, "agent_artifact": artifact,
                "agent_budget": {"max_llm_calls": 4, "used_llm_calls": 0,
                                 "cycle": int((state.get("agent_budget") or {}).get("cycle", 1)) + 1,
                                 "started_at": _now()},
                "review_mode": "review_existing", "review_contract": {},
                "previous_review_report": state.get("review_report") or {},
                "review_report": {}, "review_round": 0, "auto_revision_count": 0,
                "hard_validation": {}}
    if decision == "manual_edit":
        edited = str(review.get("edited_content_md") or "").strip()
        minimum_chars = 40 if artifact.get("kind") == "resume_project" else 80
        if len(edited) < minimum_chars:
            return {"approval_decision": "invalid"}
        from review_protocol import (artifact_diff, plan_spec_from_markdown,
                                     resume_spec_from_markdown)

        if artifact.get("kind") in {"resume", "resume_project"}:
            context = state.get("agent_context") or {}
            app = context.get("application") or {}
            spec = resume_spec_from_markdown(
                edited, application_id=str(app.get("application_id") or ""),
                jd_version_id=str(app.get("jd_version_id") or ""),
                artifact_scope=state.get("resume_scope", "full_resume"),
                previous_spec=state.get("artifact_spec") or None,
            )
        else:
            spec = plan_spec_from_markdown(edited, state.get("artifact_spec") or {})
        artifact = dict(artifact)
        artifact["content_md"] = edited + "\n"
        artifact["artifact_id"] = _artifact_id(artifact.get("kind", ""), edited,
                                                artifact.get("input_hash", ""))
        revision = int(state.get("artifact_revision") or 0) + 1
        artifact["artifact_revision"] = revision
        artifact["status"] = "reviewing"
        artifact["saved_path"] = ""
        event = _revision_event(artifact.get("kind", ""), revision,
                                str((state.get("agent_artifact") or {}).get("content_md") or ""),
                                artifact["content_md"], "manual_edit")
        return {"approval_decision": decision, "edited_by_user": True,
                "agent_artifact": artifact, "artifact_spec": spec,
                "artifact_revision": revision, "artifact_diff": artifact_diff(
                    str((state.get("agent_artifact") or {}).get("content_md") or ""), edited),
                "agent_budget": {"max_llm_calls": 4, "used_llm_calls": 0,
                                 "cycle": int((state.get("agent_budget") or {}).get("cycle", 1)) + 1,
                                 "started_at": _now()},
                "revision_history": [event], "review_mode": "manual_edit",
                "review_contract": state.get("review_contract") or {},
                "previous_review_report": state.get("review_report") or {},
                "review_report": {}, "review_round": 1, "auto_revision_count": 1,
                "hard_validation": {}}
    if decision == "reject":
        artifact = dict(artifact)
        validation = dict(artifact.get("validation") or {})
        validation["user_rejected"] = True
        artifact["validation"] = validation
        artifact["status"] = "rejected"
        return {"approval_decision": "reject", "agent_artifact": artifact}
    return {"approval_decision": "invalid"}


def _validate_edited_node(state: AgentState) -> dict:
    artifact = dict(state["agent_artifact"])
    if artifact.get("kind") in {"resume", "resume_project"}:
        from resume_critic import critic_report

        context = state["agent_context"]
        report = critic_report(artifact["content_md"],
                               jd_text=str((context.get("jd") or {}).get("jd_text") or ""),
                               profile=context["profile"], material=context.get("project_evidence", ""),
                               jd_keywords=(context.get("jd_analysis") or {}).get("keywords"),
                               use_llm=False)
        artifact["validation"] = {"edited": True, "critic": report,
                                  "semantic_status": "not_recalled_after_user_edit"}
        artifact["status"] = {"pass": "draft", "needs_fix": "needs_fix",
                              "reject": "rejected"}[report["verdict"]]
    else:
        valid = len(artifact.get("content_md", "").strip()) >= 80
        artifact["validation"] = {"edited": True, "valid": valid,
                                  "errors": [] if valid else ["计划内容过短"]}
        artifact["status"] = "draft" if valid else "rejected"
    return {"agent_artifact": artifact, "approval_decision": ""}


def _portfolio_trace(content: str, snapshot: dict) -> str:
    marker = "OFFERCLAW_PORTFOLIO_SCOPE="
    if marker in content:
        return content
    public = {key: snapshot.get(key) for key in (
        "scope_snapshot_id", "application_ids", "jd_version_ids", "match_ids",
        "profile_fingerprint", "scope_hash", "created_at")}
    encoded = json.dumps(public, ensure_ascii=False, separators=(",", ":"))
    return content.rstrip() + f"\n\n<!-- {marker}{encoded} -->\n"


def _freshness_check_node(state: AgentState) -> dict:
    started = time.perf_counter()
    artifact = dict(state["agent_artifact"])
    if state.get("commit_mode") == "unreviewed":
        return {"context_fresh": True,
                "agent_metrics": [_metric("context_freshness_check", started, skipped=True)]}
    if state.get("agent_task") == "portfolio_plan":
        live = {row["application_id"]: row for row in _candidate_rows()}
        stale = []
        for frozen in (state.get("scope_snapshot") or {}).get("target_revisions") or []:
            row = live.get(frozen["application_id"])
            if not row or _candidate_revision(row) != frozen["revision"]:
                stale.append(frozen["application_id"])
        if stale:
            validation = dict(artifact.get("validation") or {})
            validation["context_stale"] = {
                "reason": "投递、JD、匹配或画像已变化，请重新确认计划范围",
                "application_ids": stale,
            }
            artifact["validation"] = validation
            artifact["status"] = "blocked"
            return {"agent_artifact": artifact, "context_fresh": False,
                    "approval_decision": "",
                    "agent_metrics": [_metric("context_freshness_check", started,
                                              stale=True)]}
        return {"context_fresh": True,
                "agent_metrics": [_metric("context_freshness_check", started)]}
    try:
        # Load the currently active JD. Passing the frozen version here would
        # turn a legitimate JD switch into a 500 before we can invalidate the
        # old review.
        live_context = _resume_context_from_state(state, live_jd=True)
    except Exception as exc:
        validation = dict(artifact.get("validation") or {})
        validation["context_stale"] = {
            "reason": "投递、活动 JD、匹配快照或画像已变化，当前上下文无法重新装载",
            "detail": str(exc)[:300],
        }
        artifact["validation"] = validation
        artifact["status"] = "blocked"
        return {"agent_artifact": artifact, "context_fresh": False,
                "approval_decision": "",
                "agent_metrics": [_metric("context_freshness_check", started,
                                          stale=True, error=True)]}
    old_fingerprint = str((state.get("agent_context") or {}).get("_context_fingerprint") or "")
    if live_context.get("_context_fingerprint") != old_fingerprint:
        artifact["status"] = "reviewing"
        return {"agent_context": live_context, "agent_artifact": artifact,
                "jd_version_id": str((live_context.get("application") or {}).get(
                    "jd_version_id") or ""),
                "context_fresh": False, "approval_decision": "",
                "review_mode": "review_existing", "review_contract": {},
                "previous_review_report": state.get("review_report") or {},
                "review_report": {}, "review_round": 0, "auto_revision_count": 0,
                "hard_validation": {},
                "agent_budget": {"max_llm_calls": 4, "used_llm_calls": 0,
                                 "cycle": int((state.get("agent_budget") or {}).get("cycle", 1)) + 1,
                                 "started_at": _now()},
                "agent_metrics": [_metric("context_freshness_check", started,
                                          stale=True)]}
    return {"context_fresh": True,
            "agent_metrics": [_metric("context_freshness_check", started)]}


def _commit_node(state: AgentState) -> dict:
    started = time.perf_counter()
    artifact = dict(state["agent_artifact"])
    if artifact.get("saved_path"):
        return {"agent_artifact": artifact}
    unreviewed = state.get("commit_mode") == "unreviewed"
    if artifact.get("kind") == "portfolio_plan" and not unreviewed:
        from plan_gen import save_plan

        content = _portfolio_trace(artifact["content_md"], state["scope_snapshot"])
        path = save_plan(content, edited_by_user=bool(state.get("edited_by_user")))
        artifact["content_md"] = content
    elif artifact.get("kind") == "portfolio_plan":
        from io_utils import atomic_write_text

        os.makedirs(PLAN_DRAFT_DIR, exist_ok=True)
        path = os.path.join(PLAN_DRAFT_DIR, f"plan_{artifact['artifact_id']}.md")
        if not os.path.exists(path):
            atomic_write_text(path, artifact["content_md"])
    elif artifact.get("kind") in {"resume", "resume_project"}:
        if artifact.get("kind") == "resume_project":
            project_key = re.sub(
                r"[^A-Za-z0-9_-]+", "_", state.get("project_name", "") or "project")
            project_key = project_key.strip("_") or _stable_id(
                "project_", (state.get("agent_context") or {}).get("project_evidence", ""), 12)
            directory = os.path.join(RESUME_DRAFT_DIR, "projects", project_key)
            filename_prefix = "project_section"
        else:
            app_id = re.sub(r"[^A-Za-z0-9_-]+", "_", state.get("application_id", "unknown"))
            directory = os.path.join(RESUME_DRAFT_DIR, app_id)
            filename_prefix = "resume"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{filename_prefix}_{artifact['artifact_id']}.md")
        if not os.path.exists(path):
            from memory_layers import EpisodicMemory
            from memory_transactions import write_text_with_memory
            epi = EpisodicMemory()
            snapshot = epi.store.put_snapshot(
                artifact["content_md"], media_type="text/markdown",
                source_path=os.path.relpath(path, BASE_DIR),
            )
            write_text_with_memory(
                path, artifact["content_md"], event_kind="resume_artifact_changed",
                event_payload={"action": "generated" if unreviewed else "approved",
                               "snapshot_id": snapshot["snapshot_id"],
                               "content_hash": snapshot["content_hash"],
                               "application_id": state.get("application_id", ""),
                               "saved_path": os.path.relpath(path, BASE_DIR)},
                event_options={"actor": "user", "source": "agent_approval",
                               "entity_type": "resume_artifact",
                               "entity_id": artifact["artifact_id"]},
                operation_id=f"resume-{'draft' if unreviewed else 'approve'}:{artifact['artifact_id']}",
            )
        if not unreviewed and state.get("stage_project_memory"):
            from knowledge_crawler import stage_personal_memory

            context = state.get("agent_context") or {}
            candidate = stage_personal_memory(
                str(context.get("project_evidence") or ""),
                title=state.get("project_name") or "个人项目材料",
                kind="project_context",
                source_url=state.get("project_repo_url") or
                f"({context.get('project_origin') or '用户项目材料'})",
            )
            validation = dict(artifact.get("validation") or {})
            validation["project_source_candidate"] = candidate
            artifact["validation"] = validation
    else:
        raise ValueError("未知产物类型")
    artifact["status"] = "saved_unreviewed" if unreviewed else "approved"
    artifact["saved_path"] = path
    return {"agent_artifact": artifact,
            "agent_metrics": [_metric("commit_artifact", started,
                                      mode="unreviewed" if unreviewed else "approved")]}


def _route_after_load(state: AgentState) -> str:
    if state.get("fatal_error"):
        return "end"
    return ("plan_review_workflow" if state.get("agent_task") == "portfolio_plan"
            else "resume_review_workflow")


def _route_after_artifact(state: AgentState) -> str:
    return "end" if (state.get("agent_artifact") or {}).get("status") == "generation_error" else "approval"


def _review_entry_route(state: AgentState) -> str:
    return "validate" if state.get("review_mode") in {"review_existing", "manual_edit"} else "author"


def _route_after_critic(state: AgentState) -> str:
    artifact = state.get("agent_artifact") or {}
    budget = state.get("agent_budget") or {}
    can_revise = (
        artifact.get("status") == "needs_revision"
        and state.get("review_mode") != "manual_edit"
        and int(state.get("auto_revision_count") or 0) < 1
        and int(budget.get("used_llm_calls", 0)) + 2 <= int(budget.get("max_llm_calls", 4))
        and not (artifact.get("validation") or {}).get("no_progress")
    )
    return "author" if can_revise else "end"


def _route_after_approval(state: AgentState) -> str:
    decision = state.get("approval_decision")
    if state.get("workflow_version", "v1") != "v2" and decision == "approve":
        return "commit"
    if decision in {"approve", "save_unreviewed_draft"}:
        return "freshness_check"
    if decision == "legacy_edit":
        return "validate_edit"
    if decision in {"request_changes", "manual_edit"}:
        return ("plan_review_workflow" if state.get("agent_task") == "portfolio_plan"
                else "resume_review_workflow")
    if decision == "reject":
        return "end"
    return "approval"


def _route_after_freshness(state: AgentState) -> str:
    if state.get("context_fresh"):
        return "commit"
    if state.get("agent_task") == "resume" and (state.get("agent_artifact") or {}).get("status") == "reviewing":
        return "resume_review_workflow"
    return "approval"


def _review_entry_node(_state: AgentState) -> dict:
    return {}


def build_agent_task_graph(*, checkpointer=None):
    """Compile v2 author/validator/critic review workflows.

    Only the author and critic nodes are model-driven agents. Entry, validator,
    approval, freshness and commit nodes are workflow tools.
    """
    plan_builder = StateGraph(AgentState)
    plan_builder.add_node("plan_review_entry", _review_entry_node)
    plan_builder.add_node("portfolio_plan_agent", _plan_author_agent_node)
    plan_builder.add_node("plan_hard_validator", _plan_hard_validator_node)
    plan_builder.add_node("plan_critic_agent", _plan_critic_agent_node)
    plan_builder.add_edge(START, "plan_review_entry")
    plan_builder.add_conditional_edges(
        "plan_review_entry", _review_entry_route,
        {"author": "portfolio_plan_agent", "validate": "plan_hard_validator"})
    plan_builder.add_edge("portfolio_plan_agent", "plan_hard_validator")
    plan_builder.add_edge("plan_hard_validator", "plan_critic_agent")
    plan_builder.add_conditional_edges(
        "plan_critic_agent", _route_after_critic,
        {"author": "portfolio_plan_agent", "end": END})
    plan_subgraph = plan_builder.compile()

    resume_builder = StateGraph(AgentState)
    resume_builder.add_node("resume_review_entry", _review_entry_node)
    resume_builder.add_node("resume_author_agent", _resume_author_agent_node)
    resume_builder.add_node("resume_hard_validator", _resume_hard_validator_node)
    resume_builder.add_node("resume_critic_agent", _resume_critic_agent_node)
    resume_builder.add_edge(START, "resume_review_entry")
    resume_builder.add_conditional_edges(
        "resume_review_entry", _review_entry_route,
        {"author": "resume_author_agent", "validate": "resume_hard_validator"})
    resume_builder.add_edge("resume_author_agent", "resume_hard_validator")
    resume_builder.add_edge("resume_hard_validator", "resume_critic_agent")
    resume_builder.add_conditional_edges(
        "resume_critic_agent", _route_after_critic,
        {"author": "resume_author_agent", "end": END})
    resume_subgraph = resume_builder.compile()

    builder = StateGraph(AgentState)
    builder.add_node("load_context", _load_context_node)
    builder.add_node("plan_review_workflow", plan_subgraph)
    builder.add_node("resume_review_workflow", resume_subgraph)
    builder.add_node("approval", _approval_node)
    builder.add_node("validate_edit", _validate_edited_node)
    builder.add_node("freshness_check", _freshness_check_node)
    builder.add_node("commit", _commit_node)
    builder.add_edge(START, "load_context")
    builder.add_conditional_edges("load_context", _route_after_load,
                                  {"plan_review_workflow": "plan_review_workflow",
                                   "resume_review_workflow": "resume_review_workflow", "end": END})
    builder.add_conditional_edges("plan_review_workflow", _route_after_artifact,
                                  {"approval": "approval", "end": END})
    builder.add_conditional_edges("resume_review_workflow", _route_after_artifact,
                                  {"approval": "approval", "end": END})
    builder.add_conditional_edges("approval", _route_after_approval,
                                  {"approval": "approval",
                                   "plan_review_workflow": "plan_review_workflow",
                                   "resume_review_workflow": "resume_review_workflow",
                                   "freshness_check": "freshness_check",
                                   "validate_edit": "validate_edit",
                                   "commit": "commit", "end": END})
    builder.add_conditional_edges("freshness_check", _route_after_freshness,
                                  {"commit": "commit", "resume_review_workflow": "resume_review_workflow",
                                   "approval": "approval"})
    builder.add_edge("validate_edit", "approval")
    builder.add_edge("commit", END)
    return builder.compile(checkpointer=checkpointer)


def agent_runtime_health() -> dict:
    """Report whether the durable Agent checkpoint backend is importable."""
    dependency = "langgraph-checkpoint-sqlite==3.1.1"
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "status": "unavailable",
            "checkpoint": "sqlite",
            "dependency": dependency,
            "detail": str(exc),
        }
    return {
        "status": "ready",
        "checkpoint": "sqlite",
        "dependency": dependency,
        "persistence": "restart_safe",
    }


def _get_graph():
    global _graph, _checkpointer, _sqlite_connection
    with _runtime_lock:
        if _graph is not None:
            return _graph
        runtime = agent_runtime_health()
        if runtime["status"] != "ready":
            raise RuntimeError(
                "Agent 持久化组件不可用；请在启动 OfferClaw 的同一 Python 环境安装 "
                f"{runtime['dependency']}。原始错误：{runtime.get('detail') or 'unknown'}"
            )
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        db_path = os.environ.get("CAREER_AGENT_DB_PATH", DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        _sqlite_connection = sqlite3.connect(db_path, check_same_thread=False)
        _checkpointer = SqliteSaver(_sqlite_connection)
        _checkpointer.setup()
        _graph = build_agent_task_graph(checkpointer=_checkpointer)
        return _graph


def _interrupt_value(snapshot) -> dict:
    for task in snapshot.tasks or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
    return {}


def get_agent_flow(thread_id: str) -> dict:
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    with _runtime_lock:
        snapshot = graph.get_state(config)
    values = dict(snapshot.values or {})
    if not values:
        raise KeyError("Agent 线程不存在")
    artifact = dict(values.get("agent_artifact") or {})
    artifact["artifact_revision"] = int(values.get("artifact_revision") or
                                         artifact.get("artifact_revision") or 0)
    artifact["artifact_diff"] = values.get("artifact_diff") or {}
    interrupt_payload = _interrupt_value(snapshot)
    waiting = bool(interrupt_payload)
    return {
        "thread_id": thread_id,
        "task": values.get("agent_task"),
        "workflow_version": values.get("workflow_version", "v1"),
        "status": "waiting_approval" if waiting else (
            "error" if artifact.get("status") == "generation_error" else
            "completed" if not snapshot.next else "running"),
        "artifact": artifact,
        "interrupt": interrupt_payload,
        "artifact_revision": int(values.get("artifact_revision") or 0),
        "artifact_diff": values.get("artifact_diff") or {},
        "review_contract": values.get("review_contract") or {},
        "review_report": values.get("review_report") or {},
        "requirements": values.get("user_requirements") or [],
        "budget": values.get("agent_budget") or {},
        "metrics": values.get("agent_metrics") or [],
        "revision_history": values.get("revision_history") or [],
        "scope_snapshot_id": (values.get("scope_snapshot") or {}).get("scope_snapshot_id", ""),
        "resume_scope": values.get("resume_scope", "full_resume"),
        "next": list(snapshot.next or ()),
    }


def start_agent_flow(*, task: str, scope_snapshot_id: str = "", application_id: str = "",
                     jd_version_id: str = "", start_date: str = "", end_date: str = "",
                     revision_note: str = "", resume_scope: str = "full_resume",
                     project_repo_url: str = "", project_text: str = "",
                     project_name: str = "", stage_project_memory: bool = False,
                     resume_source_text: str = "") -> dict:
    if task not in {"portfolio_plan", "resume"}:
        raise ValueError("task 必须是 portfolio_plan 或 resume")
    if task == "resume" and resume_scope not in {"full_resume", "project_section"}:
        raise ValueError("resume_scope 必须是 full_resume 或 project_section")
    scope = get_scope_snapshot(scope_snapshot_id) if task == "portfolio_plan" else {}
    if task == "portfolio_plan":
        from review_protocol import normalized_period

        period_start, period_end = normalized_period(start_date, end_date)
        start_date, end_date = period_start.isoformat(), period_end.isoformat()
    requirements = []
    if str(revision_note or "").strip():
        from review_protocol import make_requirement

        requirements.append(make_requirement(revision_note, source="start"))
    thread_id = _token("agent")
    initial: AgentState = {
        "thread_id": thread_id, "agent_task": task, "workflow_version": "v2",
        "scope_snapshot": scope, "application_id": application_id,
        "jd_version_id": jd_version_id, "resume_scope": resume_scope,
        "project_repo_url": project_repo_url, "project_text": project_text,
        "project_name": project_name,
        "stage_project_memory": bool(stage_project_memory), "start_date": start_date,
        "resume_source_text": resume_source_text,
        "end_date": end_date, "revision_note": revision_note,
        "user_requirements": requirements,
        "agent_budget": {"max_llm_calls": 4, "used_llm_calls": 0,
                         "cycle": 1, "started_at": _now()},
        "agent_metrics": [], "approval_decision": "", "edited_by_user": False,
        "revision_history": [], "artifact_revision": 0,
        "review_mode": "generate", "review_round": 0,
        "auto_revision_count": 0, "review_contract": {}, "review_report": {},
        "hard_validation": {}, "context_fresh": True,
    }
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    with _runtime_lock:
        graph.invoke(initial, config=config)
    return get_agent_flow(thread_id)


def continue_agent_flow(thread_id: str, *, decision: str,
                        edited_content_md: str = "", change_request: str = "",
                        remember_preference: bool = False,
                        artifact_revision: int | None = None) -> dict:
    current = get_agent_flow(thread_id)
    if current["status"] == "completed" and current.get("artifact", {}).get("saved_path"):
        return current
    if not current.get("interrupt"):
        raise RuntimeError("该 Agent 线程当前不在审批等待状态")
    if artifact_revision is not None and artifact_revision != current.get("artifact_revision"):
        raise RuntimeError("草稿版本已变化，请刷新后再提交操作")
    payload = {"decision": decision, "edited_content_md": edited_content_md,
               "change_request": change_request,
               "remember_preference": bool(remember_preference)}
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    with _runtime_lock:
        graph.invoke(Command(resume=payload), config=config)
    return get_agent_flow(thread_id)


def resume_agent_flow(thread_id: str, *, decision: str,
                      edited_content_md: str = "", change_request: str = "",
                      remember_preference: bool = False,
                      artifact_revision: int | None = None) -> dict:
    """Compatibility alias for v1 callers; both artifact types use one workflow."""
    return continue_agent_flow(
        thread_id, decision=decision, edited_content_md=edited_content_md,
        change_request=change_request, remember_preference=remember_preference,
        artifact_revision=artifact_revision,
    )


def reset_agent_runtime_for_tests() -> None:
    """Close global SQLite state so tests can switch CAREER_AGENT_DB_PATH safely."""
    global _graph, _checkpointer, _sqlite_connection
    with _runtime_lock:
        if _sqlite_connection is not None:
            _sqlite_connection.close()
        _graph = _checkpointer = _sqlite_connection = None
    with _scope_lock:
        _scope_cache.clear()
