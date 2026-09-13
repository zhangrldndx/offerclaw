# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from action_capabilities import ACTION_CAPABILITIES
from rag_route_registry import READ_SOURCE_REGISTRY
from wechat_bridge import WeChatPendingStore
from wechat_data_bridge import REQUEST_SCHEMA as BRIDGE_SCHEMA, handle as bridge_handle
import wechat_dispatch as dispatcher
from wechat_dispatch import (
    REQUEST_SCHEMA, BridgeError, _bridge_command, call_bridge, classify_privacy, dispatch,
)
from wechat_policy import ACTION_POLICIES, READ_POLICIES, policy_report
from wechat_action_router import ACTION_EXAMPLES, action_router_contract, match_action_capability


def _request(text: str, *, message_id: str = "msg_1", sender_id: str = "self_1",
             conversation_id: str = "dm_1", trigger: str = "user",
             automation_kind: str = "") -> dict:
    return {
        "schema_version": REQUEST_SCHEMA,
        "message_id": message_id,
        "conversation_id": conversation_id,
        "channel": "openclaw-weixin",
        "account_id": "account_1",
        "sender_id": sender_id,
        "is_group": False,
        "text": text,
        "media": [],
        "trigger": trigger,
        "automation_kind": automation_kind,
    }


def _empty_query(question: str = "") -> dict:
    return {
        "status": "error", "answer": "", "reply_parts": [], "routes": [],
        "sources": [], "freshness": "", "data_version": "",
        "model_usage": {"offerclaw_calls": 0, "openclaw_calls": 0},
    }


def _audit_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("OFFERCLAW_WECHAT_AUDIT_PATH", str(path))
    return path


def test_every_registered_capability_has_privacy_mode_and_formatter():
    report = policy_report()
    assert report["status"] == "ok"
    assert set(READ_POLICIES) == {item.key for item in READ_SOURCE_REGISTRY}
    assert set(ACTION_POLICIES) == {item.capability_id for item in ACTION_CAPABILITIES}
    assert len(ACTION_POLICIES) == 30
    for policy in (*READ_POLICIES.values(), *ACTION_POLICIES.values()):
        assert policy.mode
        assert policy.privacy in {"public", "personal", "sensitive"}
        assert policy.formatter
        assert policy.model_allowed is False


def test_every_action_has_local_intent_examples_and_matches_uniquely():
    report = action_router_contract()
    assert report == {
        "status": "ok", "registered_count": 30, "example_count": 30,
        "missing": [], "extra": [],
    }
    for capability_id, examples in ACTION_EXAMPLES.items():
        assert examples
        for text in examples:
            match = match_action_capability(text)
            assert match is not None, (capability_id, text)
            assert match.capability_id == capability_id, (capability_id, text, match)


@pytest.mark.parametrize(
    "text",
    [
        "查看我投过哪些公司", "总结我的当前计划", "列出画像建议",
        "最近的学习记录是什么", "告诉我知识库里有哪些 RAG 资料",
        "知识库里关于 RAG 评估门槛有哪些记录",
        "这个岗位有哪些投递记录", "画像里记录了哪些技能",
        "分析我的复盘并给出建议", "分析项目证据的主要缺口",
    ],
)
def test_read_questions_are_not_misclassified_as_actions(text):
    assert match_action_capability(text) is None


def test_common_wechat_local_read_corpus_meets_95_percent_coverage(
        tmp_path, monkeypatch):
    """Keep colloquial local reads broad enough to avoid phrase-specific fixes."""
    import wechat_index

    _audit_in(tmp_path, monkeypatch)
    monkeypatch.setattr(wechat_index, "index_status", lambda: {
        "status": "healthy", "source_count": 1,
        "covered_source_count": 1, "chunk_count": 1,
    })
    cases = [
        ("总结一下我的画像", "profile_plan.get_profile"),
        ("我的职业方向是什么", "profile_plan.get_profile"),
        ("我会哪些技能", "profile_plan.get_profile"),
        ("目前准备投递的岗位", "application_state.list_current"),
        ("已经申请过什么岗位", "application_state.list_current"),
        ("哪些公司投过", "application_state.list_current"),
        ("今天有什么任务", "today.get"),
        ("今日安排", "today.get"),
        ("今天建议我做什么", "today.get"),
        ("最近的执行记录", "reflection_memory.get_recent"),
        ("学习记录", "reflection_memory.get_recent"),
        ("最近有什么日志", "reflection_memory.get_recent"),
        ("本周计划", "profile_plan.get_plan"),
        ("查看学习计划", "profile_plan.get_plan"),
        ("我的计划", "profile_plan.get_plan"),
        ("列出画像建议", "profile.suggestions.list"),
        ("有哪些画像建议", "profile.suggestions.list"),
        ("当前画像建议", "profile.suggestions.list"),
        ("做个健康检查", "system_diagnostics.inspect"),
        ("数据桥状态", "system_diagnostics.inspect"),
        ("系统现在是否正常", "system_diagnostics.inspect"),
    ]

    expected_by_text = dict(cases)

    def bridge(operation, payload):
        assert operation == "query.answer"
        capability = expected_by_text[payload["question"]]
        source, operation_name = capability.split(".", 1)
        return {
            "status": "ok", "answer": f"已回答：{payload['question']}",
            "reply_parts": [f"已回答：{payload['question']}"],
            "routes": [{"source": source, "operation": operation_name}],
            "sources": ["real-source"], "freshness": "2026-09-13T12:00:00+08:00",
            "data_version": "real-v1", "trace_id": "trace-1",
            "model_usage": {"offerclaw_calls": 0, "openclaw_calls": 0},
        }

    covered = 0
    misses = []
    store = WeChatPendingStore(tmp_path / "common-corpus.sqlite3")
    for index, (text, expected) in enumerate(cases):
        result = dispatch(_request(text, message_id=f"common_{index}"),
                          bridge=bridge, store=store)
        if (result["handled"] is True
                and result["privacy"]["model_exposure"] == "none"
                and result["route"]["capability_id"] == expected):
            covered += 1
        else:
            misses.append((text, expected, result["route"]["capability_id"]))

    assert covered / len(cases) >= 0.95, misses


@pytest.mark.parametrize(
    ("text", "capability_id", "expected_status"),
    [
        ("删除投递 app_123", "application.delete", "error"),
        ("生成一份新简历", "resume.generate_full", "error"),
        ("分析并制定一个新计划", "plan.generate", "error"),
        ("批准一个知识库候选", "knowledge.approve", "ok"),
        ("归档一条记忆", "memory.archive", "ok"),
        ("修改我的画像", "profile.update", "ok"),
    ],
)
def test_action_policies_are_applied_before_reads_or_model_fallback(
        tmp_path, monkeypatch, text, capability_id, expected_status):
    _audit_in(tmp_path, monkeypatch)
    result = dispatch(
        _request(text), bridge=lambda *_args: pytest.fail("action guidance must stay local"),
        store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    assert result["handled"] is True
    assert result["status"] == expected_status
    assert result["route"]["capability_id"] == capability_id
    assert result["privacy"]["model_exposure"] == "none"
    assert result["requires_confirmation"] is False
    assert not result["pending_action"]


def test_local_profile_reply_is_zero_model_and_uses_bridge_result(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    calls = []

    def bridge(operation, payload):
        calls.append((operation, payload))
        assert operation == "query.answer"
        return {
            "status": "ok", "answer": (
                "真实地点，熟练技能 Python\n\n"
                "数据来源：Windows OfferClaw 真实数据｜截至 2026-09-13T12:00:00+08:00"
            ),
            "reply_parts": [
                "真实地点，熟练技能 Python\n\n"
                "数据来源：Windows OfferClaw 真实数据｜截至 2026-09-13T12:00:00+08:00"
            ],
            "routes": [{"source": "profile_plan", "operation": "get_profile"}],
            "sources": ["user_profile.md"], "freshness": "2026-09-13T12:00:00+08:00",
            "data_version": "rev-real-1", "trace_id": "trace-real-1",
            "model_usage": {"offerclaw_calls": 0, "openclaw_calls": 0},
        }

    result = dispatch(_request("请把我当前的画像总结一下"), bridge=bridge,
                      store=WeChatPendingStore(tmp_path / "pending.sqlite3"))

    assert result["handled"] is True
    assert result["privacy"]["model_exposure"] == "none"
    assert "真实地点" in result["reply_text"]
    assert "数据来源：Windows OfferClaw 真实数据" in result["reply_text"]
    assert "截至 " in result["reply_text"]
    assert len(calls) == 1 and calls[0][0] == "query.answer"
    assert calls[0][1]["question"] == "请把我当前的画像总结一下"


def test_public_unknown_falls_back_with_original_text_only(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    result = dispatch(
        _request("北京天气怎么样"),
        bridge=lambda operation, payload: _empty_query(payload.get("question", "")),
        store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    assert classify_privacy("北京天气怎么样") == "public"
    assert result["handled"] is False
    assert result["reply_text"] == ""
    assert result["privacy"] == {"classification": "public", "model_exposure": "openclaw"}


def test_sensitive_unknown_fails_closed_without_model(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    result = dispatch(
        _request("结合我的附件和简历分析一下"),
        bridge=lambda operation, payload: _empty_query(payload.get("question", "")),
        store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    assert result["handled"] is True
    assert result["status"] == "error"
    assert result["privacy"]["model_exposure"] == "none"


def test_known_local_failure_never_falls_back(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)

    def failed_bridge(_operation, _payload):
        raise RuntimeError("bridge unavailable")

    result = dispatch(
        _request("查看我的画像"), bridge=failed_bridge,
        store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    assert result["handled"] is True
    assert result["privacy"]["model_exposure"] == "none"
    assert "没有转交给模型" in result["reply_text"]


def test_application_write_is_previewed_bound_and_idempotent(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    store = WeChatPendingStore(tmp_path / "pending.sqlite3")
    commits = []

    def bridge(operation, payload):
        if operation == "application.preview":
            return {
                "status": "preview", "payload": payload, "base_hash": "base-v1",
                "preview": {
                    "company": "真实公司", "position": "工程师",
                    "before": {"status": "准备投递"}, "after": {"status": "已投递"},
                },
            }
        if operation == "application.commit":
            commits.append(dict(payload))
            return {"status": "ok"}
        raise AssertionError(operation)

    request = _request("更新 app_real_1 状态为已投递", message_id="write_1")
    preview = dispatch(request, bridge=bridge, store=store)
    replayed_preview = dispatch(request, bridge=bridge, store=store)
    action_id = preview["pending_action"]["action_id"]
    assert action_id == replayed_preview["pending_action"]["action_id"]
    assert preview["requires_confirmation"] is True
    assert commits == []

    confirmed = dispatch(
        _request(f"确认 {action_id}", message_id="confirm_1"), bridge=bridge, store=store,
    )
    replayed = dispatch(
        _request(f"确认 {action_id}", message_id="confirm_2"), bridge=bridge, store=store,
    )
    assert confirmed["status"] == "ok"
    assert replayed["status"] == "ok"
    assert len(commits) == 1
    assert commits[0]["operation_id"] == f"wechat:{action_id}"


def test_concurrent_confirmation_commits_once(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    db_path = tmp_path / "pending.sqlite3"
    store = WeChatPendingStore(db_path)
    preview = store.stage(
        "bridge.application_update", {"application_id": "app_1", "status_code": "applied"},
        {"before": "preparing", "after": "applied"},
        context={"channel": "openclaw-weixin", "account_id": "account_1",
                 "sender_id": "self_1", "conversation_id": "dm_1", "message_id": "stage"},
        idempotency_key="concurrent", base_hash="base-v1",
    )
    commits = []

    def bridge(operation, payload):
        assert operation == "application.commit"
        commits.append(payload["operation_id"])
        return {"status": "ok"}

    def confirm(index):
        return dispatch(
            _request(f"确认 {preview['action_id']}", message_id=f"confirm_{index}"),
            bridge=bridge, store=WeChatPendingStore(db_path),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(confirm, (1, 2)))
    assert len(commits) == 1
    assert all(item["handled"] and item["privacy"]["model_exposure"] == "none"
               for item in results)


@pytest.mark.parametrize("failure", ["conflict", "crash"])
def test_commit_conflict_or_interruption_never_retries_write(
        tmp_path, monkeypatch, failure):
    _audit_in(tmp_path, monkeypatch)
    store = WeChatPendingStore(tmp_path / "pending.sqlite3")

    def bridge(operation, payload):
        if operation == "application.preview":
            return {"status": "preview", "payload": payload, "base_hash": "old",
                    "preview": {"before": {}, "after": {}}}
        if failure == "conflict":
            return {"status": "conflict", "error": "revision changed"}
        raise RuntimeError("child process interrupted")

    preview = dispatch(
        _request("更新 app_1 状态为已投递", message_id=f"stage_{failure}"),
        bridge=bridge, store=store,
    )
    action_id = preview["pending_action"]["action_id"]
    result = dispatch(
        _request(f"确认 {action_id}", message_id=f"confirm_{failure}"),
        bridge=bridge, store=store,
    )
    replay = dispatch(
        _request(f"确认 {action_id}", message_id=f"replay_{failure}"),
        bridge=lambda *_args: pytest.fail("failed action must not execute again"), store=store,
    )
    assert result["status"] == "error"
    assert replay["status"] == "error"
    assert store.get(action_id)["status"] == "failed"


def test_confirmation_rejects_other_sender_and_expired_action(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    store = WeChatPendingStore(tmp_path / "pending.sqlite3")
    context = {
        "channel": "openclaw-weixin", "account_id": "account_1",
        "sender_id": "self_1", "conversation_id": "dm_1", "message_id": "write_1",
    }
    row = store.stage(
        "bridge.daily_log", {"tag": "test"}, {"tag": "test"},
        context=context, idempotency_key="expiry-test", base_hash="v1",
    )

    with pytest.raises(PermissionError, match="不属于"):
        dispatch(
            _request(f"确认 {row['action_id']}", sender_id="other"),
            bridge=lambda *_args: pytest.fail("must not commit"), store=store,
        )

    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE pending_actions SET expires_at='2000-01-01T00:00:00+00:00' WHERE action_id=?",
            (row["action_id"],),
        )
    expired = dispatch(
        _request(f"确认 {row['action_id']}", message_id="confirm_expired"),
        bridge=lambda *_args: pytest.fail("must not commit"), store=store,
    )
    assert expired["status"] == "error"
    assert "已过期" in expired["reply_text"]


def test_smart_query_context_uses_stable_opaque_identifiers(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    db_path = tmp_path / "pending.sqlite3"
    payloads = []

    def bridge(operation, payload):
        assert operation == "query.answer"
        payloads.append(payload)
        return {
            "status": "ok", "answer": "智能查询结果",
            "reply_parts": ["智能查询结果"],
            "routes": [{"source": "application_state", "operation": "list_current"}],
            "sources": ["applications.md"], "freshness": "2026-09-13T12:00:00+08:00",
            "data_version": "real-v1", "trace_id": "trace-context",
            "model_usage": {"offerclaw_calls": 1, "openclaw_calls": 0},
        }

    first = dispatch(_request("我最近投了哪些公司", message_id="page_1"), bridge=bridge,
                     store=WeChatPendingStore(db_path))
    followup = dispatch(_request("这些公司下一步做什么", message_id="context_1"), bridge=bridge,
                        store=WeChatPendingStore(db_path))
    assert first["privacy"]["model_exposure"] == "offerclaw"
    assert followup["privacy"]["model_exposure"] == "offerclaw"
    assert payloads[0]["conversation_id"] == payloads[1]["conversation_id"]
    assert payloads[0]["message_id"] != payloads[1]["message_id"]
    serialized = json.dumps(payloads, ensure_ascii=False)
    assert "self_1" not in serialized and "dm_1" not in serialized
    assert "page_1" not in serialized and "context_1" not in serialized


@pytest.mark.parametrize(
    ("text", "decision", "modified_value"),
    [
        ("请接受画像建议 sug_1", "accepted", None),
        ("拒绝画像建议 sug_1", "rejected", None),
        ('修改后接受画像建议 sug_1 值 ["Python","RAG"]', "modified", ["Python", "RAG"]),
    ],
)
def test_profile_suggestion_decisions_are_explicit_previews(
        tmp_path, monkeypatch, text, decision, modified_value):
    _audit_in(tmp_path, monkeypatch)
    store = WeChatPendingStore(tmp_path / "pending.sqlite3")

    def bridge(operation, payload):
        assert operation == "suggestions.show"
        return {"status": "ok", "suggestion": {
            "suggestion_id": "sug_1", "field_path": "熟练技能",
            "current_value": ["Python"], "proposed_value": ["Python", "FastAPI"],
            "base_revision": 8,
        }}

    result = dispatch(_request(text), bridge=bridge, store=store)
    row = store.get(result["pending_action"]["action_id"])
    assert result["requires_confirmation"] is True
    assert row["payload"]["decision"] == decision
    assert row["payload"]["modified_value"] == modified_value
    assert row["base_revision"] == "8"


def test_cron_markers_are_local_and_zero_model(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)

    def bridge(operation, payload):
        if operation == "today.get":
            return {"advice": {"headline": "今日任务", "today": "2026-09-12"}}
        raise AssertionError(operation)

    result = dispatch(
        _request("offerclaw://automation/morning?v=1", trigger="cron",
                 automation_kind="morning", sender_id="automation"),
        bridge=bridge, store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    assert result["handled"] is True
    assert result["route"]["capability_id"] == "automation.morning"
    assert result["privacy"]["model_exposure"] == "none"


def test_attachment_is_quarantined_source_only_and_never_sent_to_model(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    workspace = tmp_path / "openclaw-media"
    workspace.mkdir()
    attachment = workspace / "project.txt"
    attachment.write_text("真实项目材料，只用于本地隔离。" * 8, encoding="utf-8")
    request = _request("这是项目附件")
    request["media"] = [{"path": str(attachment), "workspace_dir": str(workspace)}]
    store = WeChatPendingStore(tmp_path / "pending.sqlite3")

    result = dispatch(
        request, bridge=lambda *_args: pytest.fail("attachment must not call Windows bridge"),
        store=store,
    )
    row = store.get(result["reply_text"].split("ID ", 1)[1].splitlines()[0])
    assert result["handled"] is True
    assert result["privacy"]["model_exposure"] == "none"
    assert row["action_type"] == "attachment_source"
    assert row["preview"]["writes_business_data"] is False
    quarantine = Path(row["payload"]["quarantine_path"])
    assert quarantine.is_file() and quarantine.name != attachment.name


def test_attachment_path_must_be_inside_openclaw_workspace(tmp_path, monkeypatch):
    _audit_in(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("不能通过伪造路径读取本地文件。" * 8, encoding="utf-8")
    workspace = tmp_path / "different-workspace"
    workspace.mkdir()
    request = _request("这是项目附件")
    request["media"] = [{"path": str(outside), "workspace_dir": str(workspace)}]
    with pytest.raises(PermissionError, match="不在"):
        dispatch(request, bridge=lambda *_args: {},
                 store=WeChatPendingStore(tmp_path / "pending.sqlite3"))


def test_audit_records_hashes_but_not_message_or_identity(tmp_path, monkeypatch):
    audit = _audit_in(tmp_path, monkeypatch)
    request = _request("查看我的画像", message_id="private-message-id", sender_id="private-user-id")
    dispatch(
        request,
        bridge=lambda *_args: {"profile": {}, "revision": "r1"},
        store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    content = audit.read_text(encoding="utf-8")
    assert "查看我的画像" not in content
    assert "private-message-id" not in content
    assert "private-user-id" not in content
    assert "message_hash" in content and "conversation_hash" in content


def test_audit_prunes_entries_older_than_thirty_days(tmp_path, monkeypatch):
    audit = _audit_in(tmp_path, monkeypatch)
    audit.write_text(
        json.dumps({"at": "2000-01-01T00:00:00+00:00", "trace_id": "old"}) + "\n",
        encoding="utf-8",
    )
    dispatch(
        _request("查看我的画像"),
        bridge=lambda *_args: {"profile": {}, "revision": "r1"},
        store=WeChatPendingStore(tmp_path / "pending.sqlite3"),
    )
    content = audit.read_text(encoding="utf-8")
    assert '"trace_id":"old"' not in content
    assert "wxtrace_" in content


def test_windows_bridge_rejects_unknown_fields_and_path_traversal():
    with pytest.raises(ValueError, match="未知字段"):
        bridge_handle({
            "schema_version": BRIDGE_SCHEMA, "operation": "health.get",
            "payload": {}, "command": "whoami",
        })


@pytest.mark.parametrize(
    "patch",
    [
        {"unexpected": "whoami"},
        {"is_group": "false"},
        {"trigger": "system"},
        {"automation_kind": "weekly"},
        {"reply_to": {"body": "ok", "path": "C:/secret"}},
        {"media": [{"path": "a", "workspace_dir": "b", "command": "whoami"}]},
    ],
)
def test_dispatch_rejects_unknown_fields_and_type_confusion(tmp_path, patch):
    request = _request("公开文本")
    request.update(patch)
    with pytest.raises((ValueError, PermissionError)):
        dispatch(request, bridge=lambda *_args: pytest.fail("must reject before bridge"),
                 store=WeChatPendingStore(tmp_path / "pending.sqlite3"))


def test_dispatch_rejects_unknown_or_forged_cron_marker(tmp_path):
    request = _request(
        "offerclaw://automation/weekly?v=1", trigger="cron",
        automation_kind="morning", sender_id="automation",
    )
    with pytest.raises(PermissionError, match="标记"):
        dispatch(request, bridge=lambda *_args: pytest.fail("must reject before bridge"),
                 store=WeChatPendingStore(tmp_path / "pending.sqlite3"))


def test_group_identity_mismatch_and_oversized_input_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_WECHAT_ALLOWED_ACCOUNT_IDS", "account_1")
    monkeypatch.setenv("OFFERCLAW_WECHAT_ALLOWED_SENDER_IDS", "self_1")
    group = _request("公开文本")
    group["is_group"] = True
    with pytest.raises(PermissionError, match="群聊"):
        dispatch(group, bridge=lambda *_args: {},
                 store=WeChatPendingStore(tmp_path / "group.sqlite3"))
    with pytest.raises(PermissionError, match="发送者"):
        dispatch(_request("公开文本", sender_id="unpaired"), bridge=lambda *_args: {},
                 store=WeChatPendingStore(tmp_path / "sender.sqlite3"))
    oversized = _request("x" * 130_000)
    with pytest.raises(ValueError, match="过大|过长"):
        dispatch(oversized, bridge=lambda *_args: {},
                 store=WeChatPendingStore(tmp_path / "oversized.sqlite3"))
    with pytest.raises(ValueError, match="未知字段"):
        bridge_handle({
            "schema_version": BRIDGE_SCHEMA, "operation": "health.get",
            "payload": {"path": "C:/secret"},
        })
    with pytest.raises(ValueError, match="不允许"):
        bridge_handle({
            "schema_version": BRIDGE_SCHEMA, "operation": "source.read",
            "payload": {"source_id": "../secret.txt"},
        })


def test_windows_bridge_command_forces_utf8(monkeypatch):
    monkeypatch.setenv("OFFERCLAW_WINDOWS_PYTHON", "/mnt/c/runtime/python.exe")
    monkeypatch.setenv("OFFERCLAW_WINDOWS_BRIDGE_SCRIPT", "C:/repo/wechat_data_bridge.py")
    command, _cwd = _bridge_command()
    assert command == [
        "/mnt/c/runtime/python.exe", "-X", "utf8", "C:/repo/wechat_data_bridge.py",
    ]


def test_bridge_timeout_strips_keys_and_opens_circuit_after_three_failures(monkeypatch):
    dispatcher._FAILURES.clear()
    monkeypatch.delenv("OFFERCLAW_WINDOWS_PYTHON", raising=False)
    monkeypatch.delenv("OFFERCLAW_WINDOWS_BRIDGE_SCRIPT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-child")
    calls = []

    def timeout(*args, **kwargs):
        calls.append((args, kwargs))
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "must-not-enter-child" not in " ".join(map(str, args[0]))
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(dispatcher.subprocess, "run", timeout)
    for _ in range(3):
        with pytest.raises(BridgeError):
            call_bridge("health.get", {}, timeout=1)
    with pytest.raises(BridgeError, match="熔断"):
        call_bridge("health.get", {}, timeout=1)
    assert len(calls) == 3
    dispatcher._FAILURES.clear()


def test_web_and_windows_bridge_share_canonical_read_service():
    root = Path(__file__).resolve().parents[1]
    bridge = (root / "wechat_data_bridge.py").read_text(encoding="utf-8")
    web = (root / "rag_api.py").read_text(encoding="utf-8")
    for function_name in ("profile_snapshot", "applications_snapshot", "suggestions_snapshot"):
        assert function_name in bridge
        assert function_name in web
    assert 'data_version=snapshot.get("revision")' in web
    assert '"data_version": snapshot["revision"]' in web


def test_plugin_contract_intercepts_interactive_messages_and_uses_stdin():
    root = Path(__file__).resolve().parents[1]
    plugin_dir = root / "integrations" / "openclaw" / "offerclaw-direct-reply"
    entry = (plugin_dir / "index.ts").read_text(encoding="utf-8")
    runtime = (plugin_dir / "runtime.ts").read_text(encoding="utf-8")
    manifest = json.loads((plugin_dir / "openclaw.plugin.json").read_text(encoding="utf-8"))
    package = json.loads((plugin_dir / "package.json").read_text(encoding="utf-8"))
    assert manifest["hooks"] == ["reply_dispatch"]
    assert manifest["version"] == package["version"] == "1.1.0"
    assert '"reply_dispatch"' in entry
    assert '"before_dispatch"' not in entry
    assert '"inbound_claim"' not in entry
    assert '"before_agent_reply"' not in entry
    assert '["wechat-dispatch", "--stdin"]' in runtime
    assert "shell: false" in runtime
    assert "child.stdin.end" in runtime and "JSON.stringify(request)" in runtime
    assert 'scope.chatType === "direct"' in runtime
    assert "process.kill(-child.pid" in runtime
    assert 'privacy.classification === "public"' in runtime
    assert 'privacy.model_exposure === "openclaw"' in runtime
    assert "!media.present" in runtime
    assert "event?.ctx" not in runtime.split("logger.info(", 1)[1].split(");", 1)[0]


def test_reply_text_mode_is_cron_only(monkeypatch, capsys):
    monkeypatch.setattr(dispatcher.sys, "argv", ["wechat_dispatch.py", "--reply-text"])
    request = _request(
        "offerclaw://automation/morning?v=1", trigger="cron",
        automation_kind="morning", sender_id="automation",
    )
    monkeypatch.setattr(dispatcher.sys, "stdin", type("Input", (), {
        "buffer": __import__("io").BytesIO(json.dumps(request).encode("utf-8"))
    })())
    monkeypatch.setattr(dispatcher, "dispatch", lambda _request: {
        "handled": True,
        "reply_text": "本地结果",
        "privacy": {"classification": "sensitive", "model_exposure": "none"},
    })
    assert dispatcher.main() == 0
    assert capsys.readouterr().out.strip() == "本地结果"


def test_reply_text_mode_rejects_non_cron(monkeypatch):
    monkeypatch.setattr(dispatcher.sys, "argv", ["wechat_dispatch.py", "--reply-text"])
    request = _request("你好")
    monkeypatch.setattr(dispatcher.sys, "stdin", type("Input", (), {
        "buffer": __import__("io").BytesIO(json.dumps(request).encode("utf-8"))
    })())
    monkeypatch.setattr(dispatcher, "dispatch", lambda _request: {
        "handled": True,
        "reply_text": "不应输出",
        "privacy": {"classification": "public", "model_exposure": "none"},
    })
    with pytest.raises(PermissionError, match="纯文本输出"):
        dispatcher.main()
