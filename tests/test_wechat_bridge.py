# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import types

import pytest

import offerclaw_cli as cli
from wechat_bridge import (
    WeChatPendingStore,
    decide_pending,
    source_text,
    stage_application_update,
    stage_attachment,
    stage_daily_log,
)


def _module(name, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def test_daily_log_requires_separate_confirmation(tmp_path, monkeypatch):
    store = WeChatPendingStore(tmp_path / "wechat.sqlite3")
    writes = []
    monkeypatch.setitem(sys.modules, "summary_tool", _module(
        "summary_tool",
        append_structured_daily_log=lambda **kwargs: (
            writes.append(kwargs) or {"status": "ok", "log_id": "log_1"}
        ),
    ))

    staged = stage_daily_log(
        "主线:补技能\n完成:实现检索评测\n笔记:需要补充失败案例", store=store,
    )
    assert staged["status"] == "pending"
    assert staged["requires_confirmation"] is True
    assert writes == []

    confirmed = decide_pending(staged["action_id"], "confirm", store=store)
    replayed = decide_pending(staged["action_id"], "confirm", store=store)
    assert confirmed["status"] == "confirmed"
    assert writes[0]["operation_id"] == f"wechat:{staged['action_id']}"
    assert len(writes) == 1
    assert replayed["replayed"] is True


def test_application_update_uses_machine_enum_and_preview(tmp_path, monkeypatch):
    store = WeChatPendingStore(tmp_path / "wechat.sqlite3")
    writes = []
    current = {
        "application_id": "app_1", "company": "示例公司", "position": "产品经理",
        "status": "准备投递", "status_code": "preparing",
    }
    monkeypatch.setitem(sys.modules, "applications_store", _module(
        "applications_store", get_application=lambda _app_id: current,
        patch_application=lambda app_id, **changes: (
            writes.append((app_id, changes)) or {"status": "ok"}
        ),
    ))

    staged = stage_application_update("app_1", status_code="applied", store=store)
    assert staged["preview"]["after"] == {
        "status": "已投递", "status_code": "applied"}
    assert writes == []
    decide_pending(staged["action_id"], "confirm", store=store)
    assert writes[0][1]["status"] == "已投递"


def test_attachment_is_source_only_until_used_by_workflow(tmp_path):
    source = tmp_path / "project.md"
    source.write_text("# 项目\n\n使用 Python、RAG 与 FastAPI 实现检索和评测。" * 4,
                      encoding="utf-8")
    store = WeChatPendingStore(tmp_path / "wechat.sqlite3")

    staged = stage_attachment(str(source), "project", store=store)
    loaded = source_text(staged["action_id"], purposes={"project"}, store=store)

    assert staged["requires_confirmation"] is False
    assert loaded["purpose"] == "project"
    assert "FastAPI" in loaded["text"]
    with pytest.raises(ValueError, match="不需要确认写入"):
        decide_pending(staged["action_id"], "confirm", store=store)


def test_vector_database_and_image_files_are_rejected(tmp_path):
    store = WeChatPendingStore(tmp_path / "wechat.sqlite3")
    vector = tmp_path / "chroma.sqlite3"
    vector.write_bytes(b"not-a-portable-vector-library")
    image = tmp_path / "resume.png"
    image.write_bytes(b"fake")

    with pytest.raises(ValueError, match="不导入外部向量数据库"):
        stage_attachment(str(vector), "knowledge", store=store)
    with pytest.raises(ValueError, match="不支持图片 OCR"):
        stage_attachment(str(image), "resume", store=store)


def test_cli_plan_creates_review_draft_without_saved_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "career_multi_agent", _module(
        "career_multi_agent",
        preview_portfolio_scope=lambda **kwargs: {"scope_snapshot_id": "scope_1"},
        start_agent_flow=lambda **kwargs: {
            "status": "waiting_approval", "thread_id": "agent_1",
            "artifact": {"status": "ready_for_approval", "saved_path": ""},
            "review_report": {"verdict": "pass"},
        },
    ))

    result = cli._plan_review_result("按最新复盘调整")
    assert result["status"] == "waiting_approval"
    assert result["requires_confirmation"] is True
    assert result["saved_path"] == ""
