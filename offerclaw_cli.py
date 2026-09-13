# -*- coding: utf-8 -*-
"""offerclaw_cli.py — OfferClaw command-line interface for OpenClaw integration.

Direct CLI entry point — no FastAPI server needed.
OpenClaw skill calls these subcommands via shell; each prints JSON to stdout.

Usage:
    python offerclaw_cli.py today
    python offerclaw_cli.py profile
    python offerclaw_cli.py match "JD 原文..."
    python offerclaw_cli.py query "我的求职方向是什么"
    python offerclaw_cli.py daily
    python offerclaw_cli.py log "今天学了 LangGraph 条件路由"
    python offerclaw_cli.py weekly
    python offerclaw_cli.py profile-suggestion list --status pending
    python offerclaw_cli.py profile-suggestion accept <id>
    python offerclaw_cli.py doctor
    python offerclaw_cli.py health
"""

import json
import os
import sys
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

_env_local = os.path.join(BASE_DIR, ".env.local")
if os.path.exists(_env_local):
    with open(_env_local, encoding="utf-8") as _f:
        for _ln in _f:
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


def _json_out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_today():
    from career_agent import get_today_advice
    _json_out(get_today_advice())


def cmd_profile():
    from profile_loader import load_profile
    p = load_profile()
    _json_out(p)


def cmd_match(jd_text):
    from profile_loader import load_profile
    from match_job import run_match, format_report
    from jd_parser import analyze_jd
    from semantic_matcher import align_requirements

    profile = load_profile()
    analysis = analyze_jd(jd_text).model_dump(mode="json")
    alignment = align_requirements(profile, analysis).model_dump(mode="json")
    report = run_match(
        profile, jd_text, jd_analysis=analysis, semantic_alignment=alignment,
    )
    from domain_status import match_status_code
    _json_out({
        "conclusion": report.conclusion,
        "status_code": match_status_code(report.conclusion).value,
        "reason": report.conclusion_reason,
        "direction": report.direction,
        "suggestions": report.suggestions,
        "gap_list": report.gap_list,
        "jd_analysis": analysis,
        "semantic_alignment": alignment,
    })


def _default_gaps_from_profile():
    """无显式缺口时，按画像现状给一份默认缺口（大模型应用工程师方向）。"""
    return (
        "技能缺口：\n"
        "- Python 工程能力需系统提升\n"
        "- 缺少 RAG / 向量检索的系统知识与实战\n"
        "- 缺少 Agent / 工具调用 / 工作流编排的项目经历\n"
        "经历缺口：\n"
        "- 缺少端到端、可写进简历的大模型应用开发项目"
    )


def _extract_weekly_themes(plan_md):
    """从计划正文抽取每周主题行，做微信友好摘要。"""
    import re
    themes = []
    for ln in plan_md.splitlines():
        m = re.match(r"\s*(Week\s*\d+[^\n]*主题[:：][^\n]+)", ln)
        if m:
            themes.append(m.group(1).strip())
    return themes


def _plan_review_result(instruction: str = "") -> dict:
    """Generate a reviewed plan draft; never save the current plan here."""
    from career_multi_agent import preview_portfolio_scope, start_agent_flow

    preview = preview_portfolio_scope(instruction=(instruction or "").strip())
    result = start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=preview["scope_snapshot_id"],
        revision_note=(instruction or "").strip(),
    )
    artifact = result.get("artifact") or {}
    return {
        "status": result.get("status"), "thread_id": result.get("thread_id"),
        "requires_confirmation": True, "saved_path": artifact.get("saved_path", ""),
        "artifact": artifact, "review_report": result.get("review_report") or {},
        "scope_preview": preview,
        "wechat_summary": (
            "计划草稿已由 Plan Agent 生成并经 Critic 审查，尚未修改当前计划。"
            f"请检查 thread_id={result.get('thread_id')} 后明确批准、提出修改或放弃。"
        ),
    }


def cmd_plan(gaps=None):
    _json_out(_plan_review_result(gaps or ""))


def _query_result(question: str, conversation_id: str = "") -> dict:
    from conversation_context import record_successful_turn
    from memory_layers import record_business_event
    from query_service import execute_query

    conversation_id = (conversation_id or "").strip() or f"wechat:{uuid.uuid4()}"
    operation_id = f"wechat-query:{uuid.uuid4()}"
    user_message_id = f"msg_{uuid.uuid4()}"
    assistant_message_id = f"msg_{uuid.uuid4()}"
    record_business_event(
        "conversation_message",
        {"role": "user", "content": question, "state": "sent",
         "message_id": user_message_id, "source_refs": []},
        actor="user", source="wechat", operation_id=f"{operation_id}:user",
        entity_type="conversation_message", entity_id=user_message_id,
        conversation_id=conversation_id,
    )
    try:
        result = execute_query(
            question, conversation_id=conversation_id,
        ).to_dict()
        record_business_event(
            "conversation_message",
            {"role": "assistant", "content": result.get("answer") or "(empty response)",
             "state": "completed", "message_id": assistant_message_id,
             "source_refs": result.get("sources") or []},
            actor="assistant", source="wechat", operation_id=f"{operation_id}:assistant",
            entity_type="conversation_message", entity_id=assistant_message_id,
            conversation_id=conversation_id, causation_id=operation_id,
        )
        record_successful_turn(
            conversation_id, result, turn_id=assistant_message_id,
        )
    except Exception as exc:
        record_business_event(
            "conversation_message",
            {"role": "assistant", "content": str(exc), "state": "failed",
             "message_id": assistant_message_id, "source_refs": []},
            actor="assistant", source="wechat", operation_id=f"{operation_id}:assistant",
            entity_type="conversation_message", entity_id=assistant_message_id,
            conversation_id=conversation_id, causation_id=operation_id,
        )
        raise
    result.update({"conversation_id": conversation_id,
                   "user_message_id": user_message_id,
                   "assistant_message_id": assistant_message_id})
    return result


def cmd_query(question, conversation_id=""):
    """KB 命中才答：检索→相关性门槛→命中则基于 KB 合成答案+标注来源；未命中坦白说没有。

    门槛逻辑统一收口在 rag_gate.gated_query（与 Web /api/query 共用同一套）。
    """
    _json_out(_query_result(question, conversation_id))


def cmd_daily():
    daily_path = os.path.join(BASE_DIR, "daily_log.md")
    if not os.path.exists(daily_path):
        _json_out({"error": "daily_log.md not found"})
        return
    with open(daily_path, "r", encoding="utf-8") as f:
        content = f.read()
    import re
    blocks = re.findall(r"(## \d{4}-\d{2}-\d{2}.*?)(?=\n## \d{4}-\d{2}-\d{2}|\Z)", content, re.DOTALL)
    recent = blocks[-3:] if blocks else []
    _json_out({
        "total_entries": len(blocks),
        "recent": [b.strip()[:500] for b in recent],
    })


def _parse_structured_log(content):
    """解析轻量结构化留痕文本，返回 (tag, done[], todo[], notes)。

    支持的前缀（中英文冒号皆可，分号/换行分隔多项）：
      主线: / tag:        → 主线标签
      完成: / done:       → 已完成项
      未完成: / todo:     → 未完成项
      笔记: / note:       → 自由笔记
    无任何前缀时，整段作为 notes（freeform）。
    """
    import re

    tag, done, todo = "", [], []
    note_parts = []
    matched = False
    field_pattern = re.compile(
        r"(?:^|[;；])\s*(主线|tag|完成|已完成|done|未完成|todo|笔记|note)\s*[：:]",
        re.I,
    )

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        markers = list(field_pattern.finditer(line))
        if not markers:
            note_parts.append(line)
            continue
        matched = True
        prefix = line[:markers[0].start()].strip(" ;；")
        if prefix:
            note_parts.append(prefix)
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(line)
            value = line[marker.end():end].strip(" ;；")
            if not value:
                continue
            field = marker.group(1).lower()
            if field in {"主线", "tag"}:
                tag = value
            elif field in {"完成", "已完成", "done"}:
                done.extend(x.strip() for x in re.split(r"[;；、]", value) if x.strip())
            elif field in {"未完成", "todo"}:
                todo.extend(x.strip() for x in re.split(r"[;；、]", value) if x.strip())
            else:
                note_parts.append(value)

    notes = "\n".join(note_parts).strip()
    if not matched:
        notes = content.strip()
    return tag, done, todo, notes


def cmd_log(content):
    """Stage a daily log preview. A separate confirmation performs the write."""
    from wechat_bridge import stage_daily_log

    _json_out(stage_daily_log(content))


def _run_summary(mode, date_str=None):
    """Run summary_tool with the active interpreter and return structured status."""
    import datetime
    import subprocess

    date_str = date_str or datetime.date.today().isoformat()
    args = [sys.executable, os.path.join(BASE_DIR, "summary_tool.py")]
    if mode == "weekly":
        args.append("--weekly")
    args += ["--date", date_str]
    try:
        proc = subprocess.run(
            args, cwd=BASE_DIR, capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:
        return {"status": "error", "mode": mode, "date": date_str, "error": str(exc)}

    output = proc.stdout or ""
    saved = next((line for line in output.splitlines() if "复盘已保存" in line), "")
    errors = [line.strip() for line in (output + "\n" + (proc.stderr or "")).splitlines()
              if "[ERROR]" in line]
    status = "ok" if proc.returncode == 0 else ("skipped" if proc.returncode == 2 else "error")
    result = {
        "status": status,
        "mode": mode,
        "date": date_str,
        "saved": saved.replace("[OK] ", "").strip(),
    }
    if errors:
        result["error"] = errors[-1].replace("[ERROR]", "").strip()
    elif proc.returncode != 0:
        result["error"] = (proc.stderr or output).strip()[-500:]
    return result


def _daily_review_result(date_str=None):
    """Run a review without fabricating a missing user log."""
    import datetime

    date_str = date_str or datetime.date.today().isoformat()
    result = _run_summary("daily", date_str)
    active = []
    try:
        from memory_layers import SemanticMemory, get_active_adjustments
        active = get_active_adjustments(SemanticMemory())
    except Exception:
        pass
    result.update({"active_adjustments": active, "auto_logged": False,
                   "write_policy": "只生成复盘建议，不补写用户留痕或修改计划"})
    return result


def cmd_review(date_str=None):
    """触发当日（或指定日期）结构化复盘并返回次日调整建议。"""
    _json_out(_daily_review_result(date_str))


def cmd_grow():
    """Generate evidence-backed profile suggestions without changing the profile."""
    from profile_evolution import suggest_profile_updates
    _json_out(suggest_profile_updates())


def _refresh_state_result(source_names=None):
    """Safely replace indexed state sources without deleting the old version first.

    ``replace_source`` embeds and writes the new content before deleting stale IDs. A
    failed source therefore remains queryable at its previous version.
    """
    import contextlib
    import io

    import chromadb
    from rag_ingest import replace_source
    from rag_tools import get_collection_name

    state_files = {
        "user_profile.md": "profile",
        "daily_log.md": "log",
        "applications.md": "application",
        "interview_story_bank.md": "story",
    }
    if source_names is not None:
        wanted = set(source_names)
        unknown = sorted(wanted - set(state_files))
        if unknown:
            return {"status": "error", "error": "未知状态来源：" + "、".join(unknown)}
        state_files = {name: kind for name, kind in state_files.items() if name in wanted}

    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    try:
        col = client.get_collection(get_collection_name())
    except Exception as exc:
        return {"status": "error", "error": f"collection 不可用：{exc}"}

    sources = {}
    for fname, stype in state_files.items():
        if not os.path.exists(os.path.join(BASE_DIR, fname)):
            sources[fname] = {"status": "skipped", "reason": "文件不存在"}
            continue
        try:
            # Keep CLI stdout as one parseable JSON document for the WeChat agent.
            with contextlib.redirect_stdout(io.StringIO()):
                result = replace_source(fname, col, source_type=stype)
            if result.get("status") == "ok":
                sources[fname] = {
                    "status": "ok",
                    "chunks": result.get("chunks", 0),
                    "stale_removed": result.get("stale_removed", 0),
                }
            else:
                sources[fname] = {
                    "status": "failed",
                    "error": result.get("status", "unknown"),
                    "old_version_preserved": True,
                }
        except Exception as exc:
            sources[fname] = {
                "status": "failed",
                "error": str(exc)[:500],
                "old_version_preserved": True,
            }

    any_failed = any(item.get("status") == "failed" for item in sources.values())
    return {
        "status": "partial" if any_failed else "ok",
        "sources": sources,
        "warning": ("有来源刷新失败；旧版本仍保留，可安全重跑 refresh-state"
                    if any_failed else None),
        "note": "微信问答即时生效；Web 服务的向量检索需重启后生效",
    }


def cmd_refresh_state():
    """安全重摄取画像、日志、投递池和面试故事库的状态快照。"""
    _json_out(_refresh_state_result())


def _compact_suggestion(item):
    evidence = []
    for ref in item.get("evidence_refs") or []:
        evidence.append({
            "evidence_id": ref.get("evidence_id", ""),
            "source_quote": str(ref.get("source_quote", ""))[:240],
            "verification": ref.get("verification", ""),
        })
    return {
        "suggestion_id": item.get("suggestion_id", ""),
        "status": item.get("status", ""),
        "base_revision": item.get("base_revision"),
        "field_path": item.get("field_path", ""),
        "operation": item.get("operation", ""),
        "current_value": item.get("current_value"),
        "proposed_value": item.get("proposed_value"),
        "rationale": item.get("rationale", ""),
        "evidence_refs": evidence,
    }


def _profile_suggestion_result(action, suggestion_id="", *, status="", value_json="",
                               reason="", operation_id=""):
    from profile_store import decide_suggestion, list_suggestions

    if action == "list":
        rows = list_suggestions(status)
        return {"status": "ok", "count": len(rows),
                "suggestions": [_compact_suggestion(row) for row in rows]}

    rows = list_suggestions()
    suggestion = next((row for row in rows if row.get("suggestion_id") == suggestion_id), None)
    if not suggestion:
        return {"status": "error", "error": "画像建议不存在", "suggestion_id": suggestion_id}
    if action == "show":
        return {"status": "ok", "suggestion": suggestion}
    if action not in {"accept", "modify", "reject"}:
        return {"status": "error", "error": f"不支持的画像建议操作：{action}"}

    decision = {"accept": "accepted", "modify": "modified", "reject": "rejected"}[action]
    modified_value = None
    if action == "modify":
        try:
            modified_value = json.loads(value_json)
        except json.JSONDecodeError as exc:
            return {"status": "error", "error": f"--value-json 不是有效 JSON：{exc.msg}"}

    operation_id = operation_id or f"wechat:profile-suggestion:{action}:{suggestion_id}"
    try:
        result = decide_suggestion(
            suggestion_id, decision, modified_value=modified_value,
            base_revision=suggestion.get("base_revision"), reason=reason,
            operation_id=operation_id,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "suggestion_id": suggestion_id}

    result["operation_id"] = operation_id
    if action in {"accept", "modify"}:
        # Replays also retry the derived-index refresh, so a transient index failure is recoverable.
        result["refresh_state"] = _refresh_state_result(["user_profile.md"])
        if result["refresh_state"].get("status") != "ok":
            result["status"] = "partial"
    return result


def cmd_profile_suggestion(action, suggestion_id="", **kwargs):
    _json_out(_profile_suggestion_result(action, suggestion_id, **kwargs))


def _weekly_result(date_str=None):
    import datetime

    date_str = date_str or datetime.date.today().isoformat()
    review = _run_summary("weekly", date_str)

    try:
        from profile_evolution import suggest_profile_updates
        grow = suggest_profile_updates()
    except Exception as exc:
        grow = {"status": "error", "error": str(exc)}

    try:
        from career_agent import _assess_plan_drift
        from plan_gen import summarize_plan_for_automation
        plan = summarize_plan_for_automation(date_str)
        drift = _assess_plan_drift(date_str, plan)
        plan_status = "ok"
    except Exception as exc:
        plan = {"has_plan": False, "error": str(exc)}
        drift = {"level": "none", "message": "", "evidence": []}
        plan_status = "error"

    refresh_state = _refresh_state_result()

    expired = bool(plan.get("expired"))
    should_replan = expired or drift.get("level") == "warn"
    if expired:
        replan_reason = "当前计划已过期，建议在微信确认后重新生成计划。"
    elif should_replan:
        replan_reason = drift.get("message") or "计划与实际进度明显偏离，建议确认后重排。"
    else:
        replan_reason = ""

    pending_actions = []
    if grow.get("has_updates"):
        pending_actions.append("review_profile_suggestions")
    if should_replan:
        pending_actions.append("confirm_replan")

    step_statuses = [
        review.get("status"), grow.get("status"), plan_status,
        refresh_state.get("status"),
    ]
    status = "ok" if all(value == "ok" for value in step_statuses) else "partial"
    lines = ["本周闭环已执行："]
    lines.append(f"- 周复盘：{review.get('saved') or review.get('error') or review.get('status')}")
    if grow.get("wechat_summary"):
        lines.append(grow["wechat_summary"])
    elif grow.get("error"):
        lines.append(f"- 画像审计：失败（{grow['error']}）")
    if plan_status == "error":
        lines.append(f"- 计划：读取失败（{plan['error']}），未做任何修改。")
    elif should_replan:
        lines.append(f"- 计划：{replan_reason} 尚未自动修改。")
    else:
        lines.append("- 计划：无需重排。")
    if refresh_state.get("status") == "ok":
        lines.append("- 派生索引：已完成失败安全刷新。")
    else:
        lines.append("- 派生索引：部分来源刷新失败；失败来源的旧向量块已保留。")
    lines.append("- 写入边界：仅保存复盘/画像建议并刷新可重建的派生索引；未修改计划、画像或业务文件。")

    return {
        "status": status,
        "date": date_str,
        "review": review,
        "grow": grow,
        "plan": {**plan, "status": plan_status, "drift": drift},
        "should_replan": should_replan,
        "replan_reason": replan_reason,
        "refresh_state": refresh_state,
        "pending_actions": pending_actions,
        "wechat_summary": "\n".join(lines),
    }


def cmd_weekly(date_str=None):
    """Run the deterministic weekly review/grow/drift/refresh workflow."""
    _json_out(_weekly_result(date_str))


def cmd_doctor():
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "doctor.py")],
        cwd=BASE_DIR, capture_output=True, text=True, timeout=30
    )
    lines = result.stdout.strip().split("\n")
    summary = lines[-1] if lines else "unknown"
    _json_out({"output": result.stdout.strip(), "summary": summary})


def cmd_health():
    import chromadb
    try:
        from rag_tools import get_collection_name
        client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
        collection_name = get_collection_name()
        col = client.get_collection(collection_name)
        _json_out({
            "status": "healthy",
            "chroma_db": "connected",
            "collection": collection_name,
            "chunks": col.count(),
        })
    except Exception as e:
        _json_out({"status": "unhealthy", "error": str(e)})


def cmd_attachment(path: str, purpose: str, title: str = ""):
    from wechat_bridge import stage_attachment

    _json_out(stage_attachment(path, purpose, title=title))


def cmd_pending(action: str, action_id: str = "", status: str = "pending"):
    from wechat_bridge import WeChatPendingStore, decide_pending

    if action == "list":
        rows = WeChatPendingStore().list(status=status)
        _json_out({"status": "ok", "count": len(rows), "actions": rows})
    elif action == "show":
        row = WeChatPendingStore().get(action_id)
        _json_out({"status": "ok" if row else "error", "action": row,
                   "error": "待审批操作不存在" if not row else ""})
    else:
        _json_out(decide_pending(action_id, action))


def cmd_application_list():
    from applications_store import list_applications

    rows = list_applications()
    _json_out({"status": "ok", "count": len(rows), "applications": rows})


def cmd_application_update(application_id: str, status_code: str):
    from wechat_bridge import stage_application_update

    _json_out(stage_application_update(application_id, status_code=status_code))


def cmd_match_source(action_id: str):
    from wechat_bridge import source_text

    source = source_text(action_id, purposes={"jd"})
    cmd_match(source["text"])


def _resume_wechat_summary(result: dict, resume_scope: str) -> str:
    label = "项目经历" if resume_scope == "project_section" else "完整简历"
    artifact = result.get("artifact") or {}
    review = result.get("review_report") or {}
    artifact_status = str(artifact.get("status") or "")
    review_status = str(
        review.get("review_status")
        or ((artifact.get("validation") or {}).get("critic") or {}).get("review_status")
        or ""
    )
    thread_id = result.get("thread_id") or ""
    if result.get("status") == "error" or artifact_status == "generation_error":
        errors = (artifact.get("validation") or {}).get("errors") or []
        reason = str(errors[0]) if errors else "生成服务不可用"
        return f"{label}生成失败（{reason}），未进入 Critic 评审，也未写入正式文件。"
    if artifact_status == "review_unavailable" or review_status == "unavailable":
        return (
            f"{label}草稿已生成，但 Critic 暂不可用；当前产物未评审、未写入正式文件。"
            f"可稍后重试，或检查 thread_id={thread_id} 后选择保存未评审草稿。"
        )
    if review_status == "completed":
        return (
            f"{label}已完成 Resume Agent、硬校验与 Critic 流程，尚未写入正式文件。"
            f"请审阅 thread_id={thread_id} 后明确批准或提出修改。"
        )
    return (
        f"{label}草稿已生成，但尚未完成 Critic 评审，也未写入正式文件。"
        f"请检查 thread_id={thread_id} 后重试评审。"
    )


def _resume_review_result(*, resume_scope: str, application_id: str = "",
                          jd_version_id: str = "", project_action_id: str = "",
                          resume_action_id: str = "", project_name: str = "",
                          remember_project: bool = False,
                          change_request: str = "") -> dict:
    from career_multi_agent import start_agent_flow
    from wechat_bridge import source_text

    project_text = ""
    project_repo_url = ""
    resume_source_text = ""
    if project_action_id:
        source = source_text(project_action_id, purposes={"project"})
        project_text = source["text"]
        project_name = project_name or source.get("title", "")
    if resume_action_id:
        source = source_text(resume_action_id, purposes={"resume"})
        resume_source_text = source["text"]
    result = start_agent_flow(
        task="resume", resume_scope=resume_scope,
        application_id=application_id, jd_version_id=jd_version_id,
        project_repo_url=project_repo_url, project_text=project_text,
        project_name=project_name, stage_project_memory=remember_project,
        resume_source_text=resume_source_text, revision_note=change_request,
    )
    artifact = result.get("artifact") or {}
    return {
        "status": result.get("status"), "thread_id": result.get("thread_id"),
        "resume_scope": resume_scope, "requires_confirmation": True,
        "saved_path": artifact.get("saved_path", ""), "artifact": artifact,
        "review_report": result.get("review_report") or {},
        "wechat_summary": _resume_wechat_summary(result, resume_scope),
    }


def cmd_resume_start(**kwargs):
    _json_out(_resume_review_result(**kwargs))


def cmd_artifact(action: str, thread_id: str, *, change_request: str = ""):
    from career_multi_agent import continue_agent_flow, get_agent_flow

    if action == "status":
        _json_out(get_agent_flow(thread_id))
        return
    decisions = {
        "approve": "approve", "reject": "reject",
        "request-changes": "request_changes",
        "save-unreviewed": "save_unreviewed_draft",
    }
    if action not in decisions:
        raise ValueError("artifact action 非法")
    current = get_agent_flow(thread_id)
    if action == "request-changes" and not change_request.strip():
        raise ValueError("提出修改时必须提供 --change-request")
    result = continue_agent_flow(
        thread_id, decision=decisions[action], change_request=change_request,
        artifact_revision=current.get("artifact_revision"),
    )
    _json_out(result)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python offerclaw_cli.py "
            "{today|profile|match|query|plan|daily|log|review|grow|refresh-state|weekly|"
            "profile-suggestion|attachment|pending|applications|application-update|"
            "match-source|resume-start|artifact|doctor|health} [args]"
        )
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "today":
        cmd_today()
    elif cmd == "plan":
        cmd_plan(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "profile":
        cmd_profile()
    elif cmd == "match":
        if len(sys.argv) < 3:
            print("Usage: python offerclaw_cli.py match '<JD text>'")
            sys.exit(1)
        cmd_match(sys.argv[2])
    elif cmd == "query":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py query")
        parser.add_argument("question")
        parser.add_argument("--conversation-id", default="")
        args = parser.parse_args(sys.argv[2:])
        cmd_query(args.question, args.conversation_id)
    elif cmd == "daily":
        cmd_daily()
    elif cmd == "log":
        if len(sys.argv) < 3:
            print("Usage: python offerclaw_cli.py log '<content>'")
            sys.exit(1)
        cmd_log(sys.argv[2])
    elif cmd == "review":
        cmd_review(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "grow":
        cmd_grow()
    elif cmd == "refresh-state":
        cmd_refresh_state()
    elif cmd == "weekly":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py weekly")
        parser.add_argument("--date", help="Reference date in YYYY-MM-DD format")
        args = parser.parse_args(sys.argv[2:])
        cmd_weekly(args.date)
    elif cmd == "profile-suggestion":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py profile-suggestion")
        subparsers = parser.add_subparsers(dest="action", required=True)

        list_parser = subparsers.add_parser("list")
        list_parser.add_argument("--status", default="pending")

        show_parser = subparsers.add_parser("show")
        show_parser.add_argument("suggestion_id")

        for action in ("accept", "reject"):
            decision_parser = subparsers.add_parser(action)
            decision_parser.add_argument("suggestion_id")
            decision_parser.add_argument("--reason")
            decision_parser.add_argument("--operation-id")

        modify_parser = subparsers.add_parser("modify")
        modify_parser.add_argument("suggestion_id")
        modify_parser.add_argument("--value-json", required=True)
        modify_parser.add_argument("--reason")
        modify_parser.add_argument("--operation-id")

        args = parser.parse_args(sys.argv[2:])
        cmd_profile_suggestion(
            action=args.action,
            suggestion_id=getattr(args, "suggestion_id", None),
            status=getattr(args, "status", None),
            value_json=getattr(args, "value_json", None),
            reason=getattr(args, "reason", None),
            operation_id=getattr(args, "operation_id", None),
        )
    elif cmd == "attachment":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py attachment")
        parser.add_argument("path")
        parser.add_argument("--purpose", required=True,
                            choices=["jd", "resume", "project", "knowledge", "daily_log"])
        parser.add_argument("--title", default="")
        args = parser.parse_args(sys.argv[2:])
        cmd_attachment(args.path, args.purpose, args.title)
    elif cmd == "pending":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py pending")
        sub = parser.add_subparsers(dest="action", required=True)
        list_parser = sub.add_parser("list")
        list_parser.add_argument("--status", default="pending")
        for action in ("show", "confirm", "reject"):
            action_parser = sub.add_parser(action)
            action_parser.add_argument("action_id")
        args = parser.parse_args(sys.argv[2:])
        cmd_pending(args.action, getattr(args, "action_id", ""),
                    getattr(args, "status", "pending"))
    elif cmd == "applications":
        cmd_application_list()
    elif cmd == "application-update":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py application-update")
        parser.add_argument("application_id")
        parser.add_argument("--status-code", required=True)
        args = parser.parse_args(sys.argv[2:])
        cmd_application_update(args.application_id, args.status_code)
    elif cmd == "match-source":
        if len(sys.argv) < 3:
            raise ValueError("match-source 需要附件 action_id")
        cmd_match_source(sys.argv[2])
    elif cmd == "resume-start":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py resume-start")
        parser.add_argument("--scope", dest="resume_scope", required=True,
                            choices=["full_resume", "project_section"])
        parser.add_argument("--application-id", default="")
        parser.add_argument("--jd-version-id", default="")
        parser.add_argument("--project-action-id", default="")
        parser.add_argument("--resume-action-id", default="")
        parser.add_argument("--project-name", default="")
        parser.add_argument("--remember-project", action="store_true")
        parser.add_argument("--change-request", default="")
        args = parser.parse_args(sys.argv[2:])
        cmd_resume_start(**vars(args))
    elif cmd == "artifact":
        import argparse
        parser = argparse.ArgumentParser(prog="offerclaw_cli.py artifact")
        parser.add_argument("action", choices=[
            "status", "approve", "reject", "request-changes", "save-unreviewed"])
        parser.add_argument("thread_id")
        parser.add_argument("--change-request", default="")
        args = parser.parse_args(sys.argv[2:])
        cmd_artifact(args.action, args.thread_id, change_request=args.change_request)
    elif cmd == "doctor":
        cmd_doctor()
    elif cmd == "health":
        cmd_health()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        _json_out({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        sys.exit(1)
