# -*- coding: utf-8 -*-
"""career_flow.py — OfferClaw CareerFlow Orchestrator (产品级 Agent 化指导 §5)

把 ``profile_loader / match_job / career_agent / resume_builder`` 串成
一个 LangGraph 主流程，使 OfferClaw 不再是"功能函数集合"而是
具有真实编排意义的求职 Agent。

设计原则
========
1. **状态驱动**：所有节点只读 ``CareerState``，输出 patch；不写文件。
2. **可解释**：每个节点都把"做了什么 / 为什么 / 来源"写进 ``trace`` 列表。
3. **写入需确认**（§5.5）：任何写入意图都附加到 ``requires_confirmation``
   并标 ``confirm_required=True``，由用户在 UI 上点确认才落盘。
4. **LLM 可插拔**：默认 ``skip_llm=True``，简历节点只产骨架；
   设为 False 时调 ``resume_builder.build_resume_for_jd`` 走 LLM。
   这让测试 / doctor / verify_pipeline 全程不依赖 KEY。
5. **不引入新依赖**：复用项目已有的 ``langgraph``。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import operator
import os
import re
import sys
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from profile_loader import load_profile  # noqa: E402


_log = logging.getLogger(__name__)


# =====================================================
# 状态结构
# =====================================================

MatchStatusCode = Literal["suitable", "stretch", "not_recommended", "unknown"]
FailPolicy = Literal["critical", "optional"]


class TraceEntry(TypedDict, total=False):
    node: str
    action: str
    source: str
    ts: str


class ErrorEntry(TypedDict, total=False):
    node: str
    message: str
    fatal: bool


class ConfirmationProposal(TypedDict, total=False):
    id: str
    kind: str
    target_file: str
    suggested_patch: str
    reason: str
    confirm_required: bool
    status: str


class RouteDecision(TypedDict, total=False):
    router: str
    code: str
    label: str
    ts: str


class MatchReportState(TypedDict, total=False):
    status: str
    status_code: MatchStatusCode
    direction: str
    summary: str
    gap_list: dict[str, Any]
    suggestions: list[Any]
    requirement_analysis: dict[str, Any]


class BudgetState(TypedDict, total=False):
    max_steps: Optional[int]
    max_wall_s: Optional[float]
    steps: int
    started: float
    reason: str


class OperationStatus(TypedDict, total=False):
    ok: bool
    skipped: bool
    node: str
    path: str
    error: str


def _merge_confirmation_records(
        left: list[ConfirmationProposal] | None,
        right: list[ConfirmationProposal] | None) -> list[ConfirmationProposal]:
    """按稳定 proposal id 合并待确认操作。

    LangGraph 在 replay/并行分支汇合时会调用该 reducer。无 id 的旧数据
    保持原顺序，但新产生的 proposal 全部带 id，因此可防止重放重复。
    """
    out: list[ConfirmationProposal] = []
    seen: set[str] = set()
    for item in [*(left or []), *(right or [])]:
        proposal = dict(item)
        proposal_id = str(proposal.get("id") or "")
        if proposal_id and proposal_id in seen:
            continue
        if proposal_id:
            seen.add(proposal_id)
        out.append(proposal)
    return out


class CareerState(TypedDict, total=False):
    # 输入
    jd_text: str
    jd_title: str
    skip_llm: bool

    # 中间结果
    jd_valid: bool
    jd_analysis: dict[str, Any]  # 单次证据化分析；match/resume/critic 全链复用
    semantic_alignment: dict[str, Any]  # JD 要求到画像证据的受约束语义对应
    profile: dict[str, Any]
    match_report: MatchReportState
    gaps: dict[str, Any]                  # gap_list 副本（方便单独取用）
    plan_outline: list[dict[str, Any]]    # 4 周计划骨架（标题 + 每周缺口对齐）
    today_advice: dict[str, Any]          # career_agent.get_today_advice()
    resume_skeleton: dict[str, Any]       # 简历段元信息（不写文件）
    application_suggestion: dict[str, Any]  # 是否建议加入 applications.md

    # [多-agent 升级] 新增 channel（docs/MULTI_AGENT_UPGRADE.md）——P1 仅声明占位，接线在 P2/P3
    jd_struct: dict[str, Any]   # legacy checkpoint 只读兼容；新运行不再写入
    plan_md: str                # 学习规划 agent 的 LLM 计划正文（skip_llm=False 才产）
    critic_report: dict[str, Any]  # 简历 Critic 独立审查结果（覆盖/编造标记/verdict）

    # 元信息
    trace: Annotated[list[TraceEntry], operator.add]
    requires_confirmation: Annotated[
        list[ConfirmationProposal], _merge_confirmation_records]
    errors: Annotated[list[ErrorEntry], operator.add]
    route_taken: str            # 仅 routed graph 写入，linear graph 留空
    route_history: Annotated[list[RouteDecision], operator.add]
    _route_code: str
    fatal_error: bool
    fatal_reason: str

    # [L3] 循环级预算 + checkpoint（非业务语义，须声明为 channel 才能跨节点传递）
    _budget: BudgetState        # {max_steps, max_wall_s, steps, started, reason}
    _run_id: str                # checkpoint / resume 用
    _run_date: str              # replay 时稳定提案日期
    checkpoint_status: OperationStatus
    memory_status: OperationStatus


# =====================================================
# 节点实现
# =====================================================

def _trace_event(node: str, action: str, source: str = "") -> TraceEntry:
    return {
        "node": node, "action": action, "source": source,
        "ts": datetime.datetime.now().strftime("%H:%M:%S"),
    }


def _error_event(node: str, msg: str, *, fatal: bool = False) -> ErrorEntry:
    return {"node": node, "message": msg, "fatal": fatal}


_APPEND_CHANNELS = {"trace", "errors", "route_history"}


def _merge_patch_for_snapshot(state: dict, patch: dict) -> dict:
    """在 guard 内预览 LangGraph 应用 patch 后的状态，仅用于 checkpoint。"""
    merged = dict(state)
    for key, value in patch.items():
        if key in _APPEND_CHANNELS:
            merged[key] = [*(merged.get(key) or []), *(value or [])]
        elif key == "requires_confirmation":
            merged[key] = _merge_confirmation_records(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(_canonical_json(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _confirmation_proposal(*, kind: str, target_file: str,
                           suggested_patch: str, reason: str,
                           identity: Any) -> ConfirmationProposal:
    return {
        "id": _stable_id("proposal", kind, target_file, identity),
        "kind": kind,
        "target_file": target_file,
        "suggested_patch": suggested_patch,
        "reason": reason,
        "confirm_required": True,
        "status": "pending",
    }


def _match_status_code(report: dict | None) -> MatchStatusCode:
    from domain_status import match_status_code
    report = report or {}
    return match_status_code(report.get("status_code") or report.get("status")).value


def node_guard(fn, *, fail_policy: FailPolicy = "optional"):
    """节点 guard：节点只返回 patch，并区分关键失败与可降级失败。

    ``critical`` 失败会写 ``fatal_error``，路由器必须终止；``optional``
    失败仅留下 errors/trace，允许使用已有证据继续。
    """
    import functools

    @functools.wraps(fn)
    def _wrapped(state):
        name = fn.__name__.replace("_node", "")
        budget = dict(state.get("_budget") or {})
        if _budget_exhausted(budget):
            patch: dict[str, Any] = {
                "trace": [_trace_event(
                    name, "skipped_budget_exhausted", budget.get("reason", ""))]
            }
            if state.get("_budget") is not None:
                patch["_budget"] = budget
            return patch
        try:
            patch = dict(fn(state) or {})
        except Exception as exc:  # 节点异常转结构化错误
            fatal = fail_policy == "critical"
            message = f"节点异常: {exc}"
            patch = {
                "errors": [_error_event(name, message, fatal=fatal)],
                "trace": [_trace_event(name, "node_failed", str(exc)[:80])],
            }
            if fatal:
                patch.update({"fatal_error": True, "fatal_reason": message})
        if state.get("_budget") is not None:
            _budget_tick(budget, name)
            patch["_budget"] = budget

        checkpoint = _checkpoint(
            state.get("_run_id"), name, _merge_patch_for_snapshot(state, patch))
        if state.get("_run_id"):
            patch["checkpoint_status"] = checkpoint
            if not checkpoint.get("ok"):
                message = checkpoint.get("error") or "checkpoint 写入失败"
                patch.setdefault("errors", []).append(
                    _error_event("checkpoint", message, fatal=False))
                patch.setdefault("trace", []).append(
                    _trace_event("checkpoint", "write_failed", message[:80]))
        return patch

    return _wrapped


# =====================================================
# [L3] 外循环预算 + checkpoint / resume
# =====================================================
# 动机：CareerFlow 此前 graph.invoke 无总预算（长任务/多 JD 无法限本）、崩溃无法恢复（state 仅在内存）。
# 这里加：① 循环级预算（step/wall，超限优雅降级跳过下游）；② 每节点 checkpoint 落盘 + resume 续跑。

CHECKPOINT_DIR = os.environ.get(
    "OFFERCLAW_CHECKPOINT_DIR", os.path.join(BASE_DIR, ".offerclaw", "checkpoints"))


def make_budget(max_steps: int = None, max_wall_s: float = None) -> dict:
    """构造循环级预算上下文。任一维度为 None 表示不限。"""
    import time
    return {"max_steps": max_steps, "max_wall_s": max_wall_s,
            "steps": 0, "started": time.monotonic(), "reason": ""}


def _budget_exhausted(b) -> bool:
    if not b:
        return False
    if b.get("max_steps") is not None and int(b.get("steps", 0)) >= int(b["max_steps"]):
        b["reason"] = f"max_steps({b['max_steps']}) 到顶"
        return True
    if b.get("max_wall_s") is not None:
        import time
        if (time.monotonic() - float(b.get("started", 0))) >= float(b["max_wall_s"]):
            b["reason"] = f"max_wall_s({b['max_wall_s']}s) 到顶"
            return True
    return False


def _budget_tick(b, node: str) -> None:
    if b:
        b["steps"] = int(b.get("steps", 0)) + 1


def _checkpoint(run_id, node: str, state) -> OperationStatus:
    """[L3] 原子写 state 快照，并返回可观测结果。"""
    if not run_id:
        return {"ok": True, "skipped": True, "node": node}
    try:
        from io_utils import atomic_write_json
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        snap = {"run_id": run_id, "node": node, "state": dict(state)}
        node_path = os.path.join(CHECKPOINT_DIR, f"{run_id}__{node}.json")
        latest_path = os.path.join(CHECKPOINT_DIR, f"{run_id}__latest.json")
        atomic_write_json(node_path, snap)
        atomic_write_json(latest_path, snap)
        return {"ok": True, "skipped": False, "node": node, "path": latest_path}
    except Exception as exc:
        _log.warning("CareerFlow checkpoint failed: run_id=%s node=%s error=%s",
                     run_id, node, exc)
        return {"ok": False, "skipped": False, "node": node, "error": str(exc)}


def load_checkpoint(run_id: str):
    """读最近一次 checkpoint。无则 None。"""
    import json
    path = os.path.join(CHECKPOINT_DIR, f"{run_id}__latest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _log.warning("CareerFlow checkpoint read failed: run_id=%s error=%s",
                     run_id, exc)
        return None


def _learn_from_flow(final: dict) -> OperationStatus:
    """[L4] 写 episodic/SOP；best-effort 但不再静默吞错。"""
    try:
        from memory_layers import (EpisodicMemory, ProceduralMemory,
                                    record_career_flow_run, distill_procedural_sops)
        report = (final or {}).get("match_report") or {}
        if not report.get("direction"):
            return {"ok": True, "skipped": True, "node": "memory"}
        epi = EpisodicMemory()
        record_career_flow_run(epi, jd_title=(final or {}).get("jd_title", ""),
                               status=report.get("status", ""),
                               status_code=_match_status_code(report),
                               direction=report.get("direction", ""))
        distill_procedural_sops(epi, ProceduralMemory())   # 达 min_support 次「适合」即沉淀 SOP
        return {"ok": True, "skipped": False, "node": "memory"}
    except Exception as exc:
        _log.warning("CareerFlow memory write failed: %s", exc)
        return {"ok": False, "skipped": False, "node": "memory", "error": str(exc)}


def profile_node(state: CareerState) -> dict:
    """读真实 user_profile.md（已经在第一阶段去 DEMO_PROFILE 化）。"""
    if state.get("profile"):
        return {}
    p = load_profile(include_evidence=not state.get("skip_llm", True))
    return {
        "profile": p,
        "trace": [_trace_event(
            "profile", f"loaded {len(p)-1} fields", source=p.get("_source", ""))],
    }


def job_input_node(state: CareerState) -> dict:
    """规范化 JD 输入。当前只做 trim + 长度兜底；URL/抽取留给 job_discovery。"""
    if "jd_valid" in state:
        return {}
    jd = (state.get("jd_text") or "").strip()
    if len(jd) < 30:
        message = f"JD 文本过短（{len(jd)} chars），已停止匹配"
        return {
            "jd_text": jd,
            "jd_title": state.get("jd_title") or "未命名 JD",
            "jd_valid": False,
            "errors": [_error_event("job_input", message, fatal=False)],
            "trace": [_trace_event("job_input", "invalid_jd_too_short", "user")],
        }

    return {
        "jd_text": jd,
        "jd_title": state.get("jd_title") or "未命名 JD",
        "jd_valid": True,
        "trace": [_trace_event("job_input", f"jd_chars={len(jd)}", source="user")],
    }


def jd_analyze_node(state: CareerState) -> dict:
    """Generate/read one JDAnalysis and share it with every downstream node."""
    jd = (state.get("jd_text") or "").strip()
    if not jd:
        raise ValueError("jd_text empty")
    from jd_parser import (JD_ANALYSIS_SCHEMA_VERSION, analyze_jd,
                           legacy_to_analysis)
    existing = state.get("jd_analysis") or {}
    expected_hash = hashlib.sha256(jd.encode("utf-8")).hexdigest()
    if (
        isinstance(existing, dict)
        and existing.get("input_hash") == expected_hash
        and existing.get("schema_version") == JD_ANALYSIS_SCHEMA_VERSION
    ):
        return {}
    if state.get("jd_struct"):
        analysis = legacy_to_analysis(jd, state.get("jd_struct"))
    else:
        mode = "deterministic" if state.get("skip_llm", True) else os.environ.get(
            "JD_ANALYZER_MODE", "shadow")
        analysis = analyze_jd(jd, mode=mode)
    data = analysis.model_dump(mode="json")
    patch = {
        "jd_text": jd,
        "jd_analysis": data,
        "trace": [_trace_event(
            "jd_analyze",
            f"source={analysis.source} req={len(analysis.requirements)} kw={len(analysis.keywords)}",
            source=f"jd_parser:{analysis.schema_version}")],
    }
    if existing:
        # A checkpoint may outlive either the JD text or the JDAnalysis schema.
        # Clear derived values so their normal idempotency guards recompute.
        patch.update({
            "semantic_alignment": {}, "match_report": {}, "gaps": {}, "plan_outline": [],
            "today_advice": {}, "resume_skeleton": {}, "critic_report": {},
            "application_suggestion": {},
        })
    return patch


def match_node(state: CareerState) -> dict:
    """调用现有 ``match_job.run_match``，吐出三档结论 + 缺口。"""
    if state.get("match_report"):            # [L3] resume 幂等：已算过不重算（避免续跑重复/不一致）
        return {}
    from match_job import format_report, run_match
    profile = state.get("profile") or load_profile(
        include_evidence=not state.get("skip_llm", True)
    )
    jd = state.get("jd_text", "")
    if not jd:
        raise ValueError("jd_text empty")
    semantic_alignment = state.get("semantic_alignment") or {}
    if not state.get("skip_llm", True) and not semantic_alignment:
        from semantic_matcher import align_requirements
        semantic_alignment = align_requirements(
            profile, state.get("jd_analysis") or {}, enabled=True,
        ).model_dump(mode="json")
    report = run_match(
        profile, jd, jd_title=state.get("jd_title", "未命名 JD"),
        jd_analysis=state.get("jd_analysis") or {},
        semantic_alignment=semantic_alignment,
    )
    match_report: MatchReportState = {
        "status": report.conclusion,
        "status_code": _match_status_code({"status": report.conclusion}),
        "direction": report.direction,
        "summary": format_report(report),
        "gap_list": report.gap_list or {},
        "suggestions": report.suggestions or [],
        "requirement_analysis": report.requirement_analysis or {},
    }
    semantic_status = str(semantic_alignment.get("status") or "not_requested")
    return {
        "semantic_alignment": semantic_alignment,
        "match_report": match_report,
        "gaps": report.gap_list or {},
        "trace": [_trace_event(
            "match", f"conclusion={report.conclusion} semantic={semantic_status}",
            source="match_job.run_match")],
    }


def gap_node(state: CareerState) -> dict:
    """从 match_report 抽缺口，按致命度归档到三类（硬门槛 / 技能 / 经历）。"""
    if isinstance(state.get("gaps"), dict) and "total" in state.get("gaps", {}):
        return {}
    if not state.get("match_report"):
        raise ValueError("match_report missing")
    gaps = state.get("gaps") or {}
    summary = {
        "硬门槛缺口": list(gaps.get("硬门槛缺口", [])),
        "技能缺口": list(gaps.get("技能缺口", [])),
        "经历缺口": list(gaps.get("经历缺口", [])),
    }
    summary["total"] = sum(len(v) for v in summary.values() if isinstance(v, list))
    return {
        "gaps": summary,
        "trace": [_trace_event(
            "gap", f"total_gaps={summary['total']}", source="match_report.gap_list")],
    }


def plan_node(state: CareerState) -> dict:
    """4 周计划骨架（不调 LLM）。每周对齐一类缺口，给出可校验的产出物。

    full LLM 计划仍走 ``plan_gen.py``；这里只先给 stepper 用的轮廓。
    """
    if state.get("plan_outline"):            # [L3] resume 幂等
        return {}
    gaps = state.get("gaps") or {}
    skill_gaps = gaps.get("技能缺口", [])
    exp_gaps = gaps.get("经历缺口", [])
    hard_gaps = gaps.get("硬门槛缺口", [])

    weeks = [
        {
            "week": 1,
            "focus": "硬门槛 / 高致命缺口收口",
            "items": (hard_gaps[:2] + skill_gaps[:2]) or ["确认 JD 关键技能 + 准备简历草稿"],
            "deliverable": "一份可投简历草稿 + 缺口表",
        },
        {
            "week": 2,
            "focus": "技能补强",
            "items": skill_gaps[:3] or ["巩固 JD 核心技术栈，产出 1 个最小 demo"],
            "deliverable": "1 个最小 demo / 1 篇学习笔记",
        },
        {
            "week": 3,
            "focus": "经历补强",
            "items": exp_gaps[:3] or ["把现有项目向 JD 方向收口（写一段定向项目描述）"],
            "deliverable": "1 个项目段（resume_builder 输出）",
        },
        {
            "week": 4,
            "focus": "投递与面试准备",
            "items": ["简历定稿", "面试故事整理", "准备 1 次模拟面"],
            "deliverable": "applications.md 至少 +1 条已投递",
        },
    ]
    return {
        "plan_outline": weeks,
        "trace": [_trace_event(
            "plan", f"4 weeks · skill_gaps={len(skill_gaps)} exp_gaps={len(exp_gaps)}",
            source="gaps")],
    }


def today_node(state: CareerState) -> dict:
    """复用 career_agent.get_today_advice，再叠加"本次 JD 是否优先今天处理"。

    设计：如果本次 CareerFlow 跑出来的 match_report.status 含"适合"，
    则把本次 JD 顶到 headline，避免 Stepper 上"今天该做什么"与本次 JD 无关。
    """
    if state.get("today_advice"):
        return {}
    try:
        from career_agent import get_today_advice
        adv = get_today_advice()
        report = state.get("match_report") or {}
        jd_title = state.get("jd_title", "")
        if jd_title and _match_status_code(report) == "suitable":
            adv = dict(adv)
            adv["headline"] = f"【本次 JD · {jd_title}】结论={report.get('status')}，建议今天定稿简历并投出"
            adv["next_action"] = (
                f"1) 按缺口清单做最后补强（{state.get('gaps', {}).get('total', 0)} 项）；"
                f"2) 在 /ui/console 确认 patch，写回 applications.md；"
                f"3) 复用 /api/resume/markdown 出一版 JD 定制简历草稿。"
            )
            adv["source"] = "career_flow.today_node (this-JD override)"
        return {
            "today_advice": adv,
            "trace": [_trace_event(
                "today", adv.get("headline", "")[:60],
                source=adv.get("source", "career_agent"))],
        }
    except Exception as e:  # pragma: no cover
        return {
            "today_advice": {},
            "errors": [_error_event("today", str(e), fatal=False)],
            "trace": [_trace_event("today", "advice_unavailable", str(e)[:80])],
        }


def resume_node(state: CareerState) -> dict:
    """简历节点：默认产骨架（含命中关键词），skip_llm=False 时才调 LLM。"""
    if state.get("resume_skeleton"):         # [L3] resume 幂等
        return {}
    skip_llm = state.get("skip_llm", True)
    profile = state.get("profile") or {}
    jd = state.get("jd_text", "")
    keyword_items = list((state.get("jd_analysis") or {}).get("keywords") or [])
    keywords = [
        str(item.get("canonical_name") or "") for item in keyword_items
        if isinstance(item, dict) and str(item.get("canonical_name") or "").strip()
    ]
    skeleton = {
        "mode": "skeleton" if skip_llm else "llm",
        "keywords_hit": keywords,
        "must_emphasize": [
            k for k in keywords
            if any(k.lower() in s.lower() for s in
                   profile.get("熟练技能", []) + profile.get("会用技能", []))
        ],
        "headline": f"{profile.get('学历', '')} · {profile.get('专业', '')} · "
                    f"目标方向 {(profile.get('方向优先级') or ['未指定'])[0]}",
    }
    errors: list[ErrorEntry] = []
    if not skip_llm and jd:
        try:
            from resume_builder import build_resume_for_jd
            out = build_resume_for_jd(jd, jd_analysis=state.get("jd_analysis") or {})
            skeleton["llm_md"] = out.get("resume_md", "")
        except Exception as e:
            errors.append(_error_event("resume", f"LLM 生成失败：{e}", fatal=False))
            skeleton["llm_md"] = ""
    patch: dict[str, Any] = {
        "resume_skeleton": skeleton,
        "trace": [_trace_event(
            "resume", f"mode={skeleton['mode']} keywords={len(keywords)}",
            source="resume_builder" if not skip_llm else "deterministic")],
    }
    if errors:
        patch["errors"] = errors
    return patch


def _critic_input_hash(state: CareerState, md: str) -> str:
    return _stable_id(
        "critic_input",
        md,
        state.get("jd_text", ""),
        state.get("profile") or {},
        (state.get("jd_analysis") or {}).get("keywords") or [],
        not state.get("skip_llm", True),
    )


def critic_node(state: CareerState) -> dict:
    """[多-agent ④] 独立审查简历 agent 产出（产-查分离）。

    只在 resume 真出了 LLM md 时审,否则**透传**（默认 skip_llm 走骨架 → 零行为改变）。verdict 由
    ``resume_critic`` 代码聚合;LLM 语义腿仅 skip_llm=False 才开（默认关）。未过时只**打标 + 挂
    requires_confirmation**,不代改简历（守"写入需确认"/不代编码）。
    """
    skeleton = state.get("resume_skeleton") or {}
    md = skeleton.get("llm_md", "")
    if skeleton.get("mode") != "llm" or not md:
        return {}                                 # 无 LLM md 可审 → 无 patch
    input_hash = _critic_input_hash(state, md)
    if (state.get("critic_report") or {}).get("_input_hash") == input_hash:
        return {}
    from resume_critic import critic_report
    report = critic_report(
        md, jd_text=state.get("jd_text", ""),
        profile=state.get("profile") or {},
        jd_keywords=(state.get("jd_analysis") or {}).get("keywords"),
        use_llm=not state.get("skip_llm", True))
    report = dict(report)
    report["_input_hash"] = input_hash
    confirmations: list[ConfirmationProposal] = []
    if report["verdict"] != "pass":
        detail = (f"编造 {len(report['fabrication_flags'])} 处" if report["fabrication_flags"]
                  else f"关键词覆盖 {report['keyword_coverage']['coverage']:.0%} 偏低")
        confirmations.append(_confirmation_proposal(
            kind="resume_critic_review",
            target_file="(简历草稿)",
            suggested_patch=f"Critic 判定 {report['verdict']}：{detail}",
            reason="简历独立审查未通过，建议改写后再用",
            identity=input_hash,
        ))
    patch: dict[str, Any] = {
        "critic_report": report,
        "trace": [_trace_event(
            "critic",
            f"verdict={report['verdict']} cov={report['keyword_coverage']['coverage']:.2f} "
            f"fab={len(report['fabrication_flags'])}", source="resume_critic")],
    }
    if confirmations:
        patch["requires_confirmation"] = confirmations
    return patch


def application_suggest_node(state: CareerState) -> dict:
    """根据匹配结论给出"是否加入 applications.md"建议；只产 patch，不写。"""
    report = state.get("match_report") or {}
    status = report.get("status", "")
    status_code = _match_status_code(report)
    title = state.get("jd_title", "未命名 JD")
    identity = {
        "title": title,
        "status_code": status_code,
        "direction": report.get("direction", ""),
        "jd_hash": _stable_id("jd", state.get("jd_text", "")),
    }
    input_id = _stable_id("application_suggestion", identity)
    if (state.get("application_suggestion") or {}).get("_input_id") == input_id:
        return {}

    if status_code == "suitable":
        suggest = "建议加入 applications.md（状态=待评估 / 投递准备）"
    elif status_code in {"stretch", "not_recommended"}:
        suggest = "暂不建议加入 applications.md（先修硬门槛或换方向）"
    else:
        suggest = "匹配状态无法识别，未生成投递建议"

    proposal = _confirmation_proposal(
        kind="application_create",
        target_file="applications.md",
        suggested_patch=(
            f"\n| {state.get('_run_date') or datetime.date.today().isoformat()} | {title} | "
            f"（公司）| 待评估 | CareerFlow 自动建议 |"
        ),
        reason=f"匹配结论 = {status}",
        identity=identity,
    )
    suggestion = {
        "decision": suggest,
        "status": status,
        "status_code": status_code,
        "reason": proposal["reason"],
        "patch": proposal,
        "_input_id": input_id,
    }
    patch: dict[str, Any] = {
        "application_suggestion": suggestion,
        "trace": [_trace_event(
            "application_suggest", suggest, source="match_report.status_code")],
    }
    if status_code == "suitable":
        patch["requires_confirmation"] = [proposal]
    return patch


# =====================================================
# Graph 构建
# =====================================================

def build_graph():
    g = StateGraph(CareerState)
    g.add_node("profile", node_guard(profile_node, fail_policy="critical"))
    g.add_node("job_input", node_guard(job_input_node, fail_policy="critical"))
    g.add_node("jd_analyze", node_guard(jd_analyze_node, fail_policy="critical"))
    g.add_node("match", node_guard(match_node, fail_policy="critical"))
    g.add_node("gap", node_guard(gap_node, fail_policy="critical"))
    g.add_node("plan", node_guard(plan_node, fail_policy="optional"))
    g.add_node("today", node_guard(today_node, fail_policy="optional"))
    g.add_node("resume", node_guard(resume_node, fail_policy="optional"))
    g.add_node("critic", node_guard(critic_node, fail_policy="optional"))
    g.add_node("application_suggest",
               node_guard(application_suggest_node, fail_policy="optional"))

    g.set_entry_point("profile")
    g.add_conditional_edges(
        "profile", _route_after_critical, {"ok": "job_input", "error": END})
    g.add_conditional_edges(
        "job_input", _route_after_job_input,
        {"ok": "jd_analyze", "too_short": END, "error": END})
    g.add_conditional_edges(
        "jd_analyze", _route_after_critical, {"ok": "match", "error": END})
    g.add_conditional_edges(
        "match", _route_after_known_match, {"ok": "gap", "error": END})
    g.add_conditional_edges(
        "gap", _route_after_critical, {"ok": "plan", "error": END})
    g.add_edge("plan", "today")
    g.add_edge("today", "resume")
    g.add_edge("resume", "critic")
    g.add_edge("critic", "application_suggest")
    g.add_edge("application_suggest", END)
    return g.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def _finalize_memory(out: dict) -> dict:
    """统一收尾记忆写入，把成功/跳过/失败明确放回 state。"""
    prior = out.get("memory_status") or {}
    if prior.get("ok") and not prior.get("skipped"):
        return out
    budget_reason = (out.get("_budget") or {}).get("reason")
    if out.get("fatal_error") or budget_reason:
        reason = out.get("fatal_reason") or budget_reason or "workflow incomplete"
        status: OperationStatus = {
            "ok": True, "skipped": True, "node": "memory", "error": reason}
    else:
        status = _learn_from_flow(out)
    action = "write_ok" if status.get("ok") and not status.get("skipped") else (
        "skipped" if status.get("ok") else "write_failed")
    patch: dict[str, Any] = {
        "memory_status": status,
        "trace": [_trace_event("memory", action, status.get("error", ""))],
    }
    if not status.get("ok"):
        patch["errors"] = [_error_event(
            "memory", status.get("error") or "memory write failed", fatal=False)]
    return _merge_patch_for_snapshot(out, patch)


# =====================================================
# 顶层入口
# =====================================================

def run_career_flow(jd_text: str, *, jd_title: str = "未命名 JD",
                    skip_llm: bool = True) -> dict:
    """对一份 JD 跑完整 CareerFlow，返回最终 state（dict）。"""
    state: CareerState = {
        "jd_text": jd_text,
        "jd_title": jd_title,
        "skip_llm": skip_llm,
        "trace": [],
        "requires_confirmation": [],
        "errors": [],
        "route_history": [],
        "_run_date": datetime.date.today().isoformat(),
    }
    final = get_graph().invoke(state)
    return _finalize_memory(dict(final))


# =====================================================
# Agentic Graph：基于 state 的条件路由（V4 §2）
# =====================================================
#
# 让 LangGraph 不再是"流程画板"。三档结论触发三条不同的后续路径，
# job_input 检测出 JD 过短直接终结。三个 router 函数与节点解耦，
# 每个分支都打 ``route_taken`` 入 trace 供 observability 回放。
#
#     profile → job_input → [router_jd]
#                              ├ "too_short" → END
#                              └ "ok" → match → gap → [router_match]
#                                                          ├ "suitable" → plan → today → resume → application_suggest → END
#                                                          ├ "stretch"  → plan → today → END
#                                                          └ "not_recommended" → application_suggest → END

ROUTE_LABELS = {
    "too_short": "jd_too_short:skip_all",
    "suitable": "suitable:full_path",
    "stretch": "stretch:plan_today_only",
    "not_recommended": "not_recommended:gap_only",
    "error": "error:stop",
    "ok": "ok:continue",
}


def _route_after_critical(state: CareerState) -> str:
    return "error" if state.get("fatal_error") else "ok"


def _route_after_known_match(state: CareerState) -> str:
    if state.get("fatal_error"):
        return "error"
    return "error" if _match_status_code(state.get("match_report")) == "unknown" else "ok"


def _route_after_job_input(state: CareerState) -> str:
    """JD 过短直接收尾——避免下游 match/gap 输出垃圾结论。"""
    if state.get("fatal_error"):
        return "error"
    if state.get("jd_valid") is False:
        return "too_short"
    return "ok"


def _route_after_gap(state: CareerState) -> str:
    """根据机器可读 status_code 决定下半段路径。未知值显式停止。"""
    if state.get("fatal_error"):
        return "error"
    code = _match_status_code(state.get("match_report"))
    return code if code != "unknown" else "error"


def _route_after_today(state: CareerState) -> str:
    return _route_after_gap(state)  # stretch 在 today 后结束；suitable 继续 resume


def _mark_route(state: CareerState, label: str, *, router: str) -> dict:
    """把分支选择写入 state 供 trace/UI 用。仅 router 调用，节点函数不直接调。"""
    route_taken = ROUTE_LABELS.get(label, label)
    return {
        "_route_code": label,
        "route_taken": route_taken,
        "route_history": [{
            "router": router,
            "code": label,
            "label": route_taken,
            "ts": datetime.datetime.now().strftime("%H:%M:%S"),
        }],
        "trace": [_trace_event(
            "router", f"{router}:route={route_taken}", source="routed_graph")],
    }


def _router_jd_node(state: CareerState) -> dict:
    """空操作节点；纯粹为了能让 router_jd 在 trace 里留痕。
    LangGraph 的 conditional edge 不会自动写 state，所以这里显式打标。"""
    label = _route_after_job_input(state)
    return _mark_route(state, label, router="after_job_input")


def _router_match_node(state: CareerState) -> dict:
    return _mark_route(state, _route_after_gap(state), router="after_gap")


def _router_match_valid_node(state: CareerState) -> dict:
    return _mark_route(state, _route_after_known_match(state), router="after_match")


def _router_today_node(state: CareerState) -> dict:
    return _mark_route(state, _route_after_today(state), router="after_today")


def _marked_route(state: CareerState) -> str:
    """条件边仅消费 router 节点已记录的决策，不二次计算。"""
    return state.get("_route_code") or "error"


def build_routed_graph():
    """带条件分支的 CareerFlow；JDAnalysis 在匹配前只生成一次。"""
    g = StateGraph(CareerState)
    g.add_node("profile", node_guard(profile_node, fail_policy="critical"))
    g.add_node("job_input", node_guard(job_input_node, fail_policy="critical"))
    g.add_node("router_jd", _router_jd_node)             # router 是纯路由，不需 guard
    g.add_node("jd_analyze", node_guard(jd_analyze_node, fail_policy="critical"))
    g.add_node("match", node_guard(match_node, fail_policy="critical"))
    g.add_node("router_match_valid", _router_match_valid_node)
    g.add_node("gap", node_guard(gap_node, fail_policy="critical"))
    g.add_node("router_match", _router_match_node)
    g.add_node("plan", node_guard(plan_node, fail_policy="optional"))
    g.add_node("today", node_guard(today_node, fail_policy="optional"))
    g.add_node("router_today", _router_today_node)
    g.add_node("resume", node_guard(resume_node, fail_policy="optional"))
    g.add_node("critic", node_guard(critic_node, fail_policy="optional"))
    g.add_node("application_suggest",
               node_guard(application_suggest_node, fail_policy="optional"))

    g.set_entry_point("profile")
    g.add_conditional_edges(
        "profile", _route_after_critical, {"ok": "job_input", "error": END})
    g.add_edge("job_input", "router_jd")
    g.add_conditional_edges(
        "router_jd",
        _marked_route,
        {"too_short": END, "ok": "jd_analyze", "error": END},
    )
    g.add_conditional_edges(
        "jd_analyze", _route_after_critical, {"ok": "match", "error": END})
    g.add_edge("match", "router_match_valid")
    g.add_conditional_edges(
        "router_match_valid", _marked_route, {"ok": "gap", "error": END})
    g.add_edge("gap", "router_match")
    g.add_conditional_edges(
        "router_match",
        _marked_route,
        {
            "suitable": "plan",
            "stretch": "plan",
            "not_recommended": "application_suggest",
            "error": END,
        },
    )
    g.add_edge("plan", "today")
    g.add_edge("today", "router_today")
    g.add_conditional_edges(
        "router_today",
        _marked_route,
        {"suitable": "resume", "stretch": END,
         "not_recommended": END, "error": END},
    )
    g.add_edge("resume", "critic")
    g.add_edge("critic", "application_suggest")
    g.add_edge("application_suggest", END)
    return g.compile()


_COMPILED_ROUTED = None


def get_routed_graph():
    global _COMPILED_ROUTED
    if _COMPILED_ROUTED is None:
        _COMPILED_ROUTED = build_routed_graph()
    return _COMPILED_ROUTED


def run_career_flow_routed(jd_text: str, *, jd_title: str = "未命名 JD",
                           skip_llm: bool = True, budget: dict = None,
                           run_id: str = None) -> dict:
    """带条件路由版的 CareerFlow。失败/短 JD 不会拖累下游节点。

    [L3] ``budget``（``make_budget()``）开启循环级预算闸（超限优雅降级）；
    ``run_id`` 开启每节点 checkpoint 落盘，崩溃后用 ``resume_career_flow(run_id)`` 续跑。
    """
    state: CareerState = {
        "jd_text": jd_text,
        "jd_title": jd_title,
        "skip_llm": skip_llm,
        "trace": [],
        "requires_confirmation": [],
        "errors": [],
        "route_history": [],
        "_run_date": datetime.date.today().isoformat(),
    }
    if budget is not None:
        state["_budget"] = dict(budget)     # [L3] 循环级预算；不修改调用方 dict
    if run_id is not None:
        state["_run_id"] = run_id           # [L3] checkpoint / resume
    final = get_routed_graph().invoke(state)
    return _finalize_memory(dict(final))


def resume_career_flow(run_id: str) -> dict:
    """[L3] 从最后一次 checkpoint replay 续跑到结束。无 checkpoint → FileNotFoundError。

    续跑给全新（无）预算，不带着已耗尽的旧预算，避免一上来就被判超限。
    """
    ckpt = load_checkpoint(run_id)
    if not ckpt:
        raise FileNotFoundError(f"无 checkpoint：{run_id}")
    state = dict(ckpt.get("state") or {})
    state.pop("_budget", None)
    state["_run_id"] = run_id
    state = _merge_patch_for_snapshot(state, {
        "trace": [_trace_event(
            "resume", f"replay_from={ckpt.get('node')}", source="checkpoint")]
    })
    final = get_routed_graph().invoke(state)
    return _finalize_memory(dict(final))


if __name__ == "__main__":
    import json
    demo_jd = (
        "岗位名称：大模型应用开发实习生\n公司：示例\n工作地点：上海\n"
        "学历要求：本科及以上\n专业要求：计算机、人工智能\n经验要求：实习\n"
        "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
    )
    out = run_career_flow(demo_jd, jd_title="DEMO 大模型应用实习")
    # 简化打印
    summary = {
        "status": out.get("match_report", {}).get("status"),
        "direction": out.get("match_report", {}).get("direction"),
        "gap_total": out.get("gaps", {}).get("total"),
        "plan_weeks": [w["focus"] for w in out.get("plan_outline", [])],
        "today": out.get("today_advice", {}).get("headline"),
        "resume_keywords": out.get("resume_skeleton", {}).get("keywords_hit"),
        "application": out.get("application_suggestion", {}).get("decision"),
        "confirm_pending": len(out.get("requires_confirmation", [])),
        "trace_steps": len(out.get("trace", [])),
        "errors": out.get("errors", []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
