# -*- coding: utf-8 -*-
"""Deterministic WeChat-facing CLI workflow tests."""
from __future__ import annotations

import json
import sys
import types

import offerclaw_cli as cli


def _module(name, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def test_refresh_state_replaces_each_source_safely(tmp_path, monkeypatch, capsys):
    for name in ("user_profile.md", "daily_log.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    collection = object()
    calls = []

    class Client:
        def __init__(self, path):
            assert path == str(tmp_path / "chroma_db")

        def get_collection(self, name):
            assert name == "test-collection"
            return collection

    def replace_source(name, passed_collection, source_type):
        print("internal ingest progress")
        calls.append((name, source_type))
        assert passed_collection is collection
        return {"status": "ok", "chunks": 2, "stale_removed": 1}

    monkeypatch.setattr(cli, "BASE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "chromadb", _module("chromadb", PersistentClient=Client))
    monkeypatch.setitem(sys.modules, "rag_tools", _module(
        "rag_tools", get_collection_name=lambda: "test-collection",
    ))
    monkeypatch.setitem(sys.modules, "rag_ingest", _module(
        "rag_ingest", replace_source=replace_source,
    ))

    result = cli._refresh_state_result(["user_profile.md", "daily_log.md"])

    assert result["status"] == "ok"
    assert calls == [("user_profile.md", "profile"), ("daily_log.md", "log")]
    assert result["sources"]["user_profile.md"]["stale_removed"] == 1
    assert capsys.readouterr().out == ""


def test_refresh_state_reports_partial_and_preserves_failed_source(tmp_path, monkeypatch):
    for name in ("user_profile.md", "daily_log.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    class Client:
        def __init__(self, path):
            pass

        def get_collection(self, name):
            return object()

    def replace_source(name, collection, source_type):
        if name == "daily_log.md":
            raise RuntimeError("embedding unavailable")
        return {"status": "ok", "chunks": 3, "stale_removed": 2}

    monkeypatch.setattr(cli, "BASE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "chromadb", _module("chromadb", PersistentClient=Client))
    monkeypatch.setitem(sys.modules, "rag_tools", _module(
        "rag_tools", get_collection_name=lambda: "test-collection",
    ))
    monkeypatch.setitem(sys.modules, "rag_ingest", _module(
        "rag_ingest", replace_source=replace_source,
    ))

    result = cli._refresh_state_result(["user_profile.md", "daily_log.md"])

    assert result["status"] == "partial"
    assert result["sources"]["user_profile.md"]["status"] == "ok"
    failed = result["sources"]["daily_log.md"]
    assert failed["status"] == "failed"
    assert failed["old_version_preserved"] is True


def test_profile_suggestion_accept_uses_revision_stable_id_and_refreshes(monkeypatch):
    suggestion = {"suggestion_id": "sg-1", "status": "pending", "base_revision": 7}
    decisions = []

    def decide(suggestion_id, decision, **kwargs):
        decisions.append((suggestion_id, decision, kwargs))
        return {"status": "ok", "result_revision": 8}

    monkeypatch.setitem(sys.modules, "profile_store", _module(
        "profile_store", list_suggestions=lambda status="": [suggestion],
        decide_suggestion=decide,
    ))
    monkeypatch.setattr(cli, "_refresh_state_result", lambda sources=None: {
        "status": "ok", "sources": sources,
    })

    first = cli._profile_suggestion_result("accept", "sg-1", reason="事实确认")
    second = cli._profile_suggestion_result("accept", "sg-1", reason="事实确认")

    assert first["result_revision"] == 8
    assert first["refresh_state"]["sources"] == ["user_profile.md"]
    assert decisions[0][2]["base_revision"] == 7
    assert decisions[0][2]["operation_id"] == "wechat:profile-suggestion:accept:sg-1"
    assert decisions[1][2]["operation_id"] == decisions[0][2]["operation_id"]
    assert second["status"] == "ok"


def test_profile_suggestion_modify_parses_json_and_reject_does_not_refresh(monkeypatch):
    suggestion = {"suggestion_id": "sg-2", "status": "pending", "base_revision": 3}
    decisions = []
    refreshes = []

    def decide(suggestion_id, decision, **kwargs):
        decisions.append((decision, kwargs))
        return {"status": "ok", "result_revision": 4 if decision == "modified" else None}

    monkeypatch.setitem(sys.modules, "profile_store", _module(
        "profile_store", list_suggestions=lambda status="": [suggestion],
        decide_suggestion=decide,
    ))
    monkeypatch.setattr(cli, "_refresh_state_result", lambda sources=None: (
        refreshes.append(sources) or {"status": "ok"}
    ))

    modified = cli._profile_suggestion_result(
        "modify", "sg-2", value_json='{"name":"Python","level":3}',
    )
    rejected = cli._profile_suggestion_result("reject", "sg-2")

    assert decisions[0][1]["modified_value"] == {"name": "Python", "level": 3}
    assert modified["refresh_state"]["status"] == "ok"
    assert decisions[1][0] == "rejected"
    assert "refresh_state" not in rejected
    assert refreshes == [["user_profile.md"]]


def test_profile_suggestion_list_show_and_invalid_json(monkeypatch):
    suggestion = {
        "suggestion_id": "sg-3", "status": "pending", "base_revision": 5,
        "field_path": "/skills/0/level", "evidence_refs": [],
    }
    monkeypatch.setitem(sys.modules, "profile_store", _module(
        "profile_store", list_suggestions=lambda status="": [suggestion],
        decide_suggestion=lambda *args, **kwargs: {},
    ))

    listed = cli._profile_suggestion_result("list", status="pending")
    shown = cli._profile_suggestion_result("show", "sg-3")
    invalid = cli._profile_suggestion_result("modify", "sg-3", value_json="{bad")

    assert listed["count"] == 1
    assert shown["suggestion"] == suggestion
    assert invalid["status"] == "error"
    assert "有效 JSON" in invalid["error"]


def test_resume_wechat_summary_never_claims_critic_completed_on_generation_error():
    result = {
        "status": "error", "thread_id": "agent-1",
        "artifact": {
            "status": "generation_error",
            "validation": {"errors": ["OPENAI_API_KEY 未配置"]},
        },
        "review_report": {},
    }
    summary = cli._resume_wechat_summary(result, "project_section")
    assert "生成失败" in summary
    assert "未进入 Critic" in summary
    assert "已完成" not in summary


def test_resume_wechat_summary_distinguishes_unavailable_and_completed_review():
    unavailable = cli._resume_wechat_summary({
        "status": "waiting_approval", "thread_id": "agent-2",
        "artifact": {"status": "review_unavailable"},
        "review_report": {"review_status": "unavailable"},
    }, "full_resume")
    completed = cli._resume_wechat_summary({
        "status": "waiting_approval", "thread_id": "agent-3",
        "artifact": {"status": "ready_for_approval"},
        "review_report": {"review_status": "completed"},
    }, "full_resume")
    assert "Critic 暂不可用" in unavailable and "未评审" in unavailable
    assert "已完成 Resume Agent、硬校验与 Critic" in completed


def _weekly_modules(monkeypatch, order, *, drift_level="info", expired=False,
                    grow_status="ok"):
    monkeypatch.setattr(cli, "_run_summary", lambda mode, date: (
        order.append("review") or {"status": "ok", "saved": "weekly.md"}
    ))
    monkeypatch.setitem(sys.modules, "profile_evolution", _module(
        "profile_evolution", suggest_profile_updates=lambda: (
            order.append("grow") or {"status": grow_status, "has_updates": True,
                                      "wechat_summary": "画像建议待确认"}
        ),
    ))
    monkeypatch.setitem(sys.modules, "plan_gen", _module(
        "plan_gen", summarize_plan_for_automation=lambda date: (
            order.append("plan") or {"has_plan": True, "expired": expired,
                                      "plan_file": "plan.md"}
        ),
    ))
    monkeypatch.setitem(sys.modules, "career_agent", _module(
        "career_agent", _assess_plan_drift=lambda date, plan: (
            order.append("drift") or {"level": drift_level, "message": "进度偏离"}
        ),
    ))
    monkeypatch.setattr(cli, "_refresh_state_result", lambda: (
        order.append("refresh") or {"status": "ok"}
    ))


def test_weekly_runs_fixed_order_and_warn_only_requests_replan(monkeypatch):
    order = []
    _weekly_modules(monkeypatch, order, drift_level="warn")

    result = cli._weekly_result("2026-09-12")

    assert order == ["review", "grow", "plan", "drift", "refresh"]
    assert result["status"] == "ok"
    assert result["should_replan"] is True
    assert result["pending_actions"] == ["review_profile_suggestions", "confirm_replan"]
    assert "尚未自动修改" in result["wechat_summary"]


def test_weekly_info_drift_does_not_replan_and_partial_failure_continues(monkeypatch):
    order = []
    _weekly_modules(monkeypatch, order, drift_level="info", grow_status="error")

    result = cli._weekly_result("2026-09-12")

    assert order == ["review", "grow", "plan", "drift", "refresh"]
    assert result["status"] == "partial"
    assert result["should_replan"] is False
    assert "confirm_replan" not in result["pending_actions"]


def test_weekly_expired_plan_requests_consent_without_writing_plan(monkeypatch):
    order = []
    _weekly_modules(monkeypatch, order, expired=True)

    result = cli._weekly_result("2026-09-12")

    assert result["should_replan"] is True
    assert "过期" in result["replan_reason"]
    assert "confirm_replan" in result["pending_actions"]


def test_weekly_plan_failure_is_partial_and_refresh_still_runs(monkeypatch):
    order = []
    _weekly_modules(monkeypatch, order)
    monkeypatch.setitem(sys.modules, "plan_gen", _module(
        "plan_gen", summarize_plan_for_automation=lambda date: (
            order.append("plan") or (_ for _ in ()).throw(RuntimeError("plan unreadable"))
        ),
    ))

    result = cli._weekly_result("2026-09-12")

    assert order == ["review", "grow", "plan", "refresh"]
    assert result["status"] == "partial"
    assert result["plan"]["status"] == "error"
    assert result["should_replan"] is False
    assert "未做任何修改" in result["wechat_summary"]


def test_weekly_refresh_failure_is_partial_and_old_chunks_are_reported(monkeypatch):
    order = []
    _weekly_modules(monkeypatch, order)
    monkeypatch.setattr(cli, "_refresh_state_result", lambda: (
        order.append("refresh") or {"status": "partial", "failed": ["user_profile.md"]}
    ))
    result = cli._weekly_result("2026-09-12")

    assert order == ["review", "grow", "plan", "drift", "refresh"]
    assert result["status"] == "partial"
    assert result["refresh_state"]["failed"] == ["user_profile.md"]
    assert "旧向量块已保留" in result["wechat_summary"]
