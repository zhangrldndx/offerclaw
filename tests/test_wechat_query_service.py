# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

from fastapi.testclient import TestClient
import pytest

import business_read_service
import query_service
from model_call_context import record_model_request, record_model_response


def test_reply_parts_preserve_complete_long_answer():
    text = ("第一段。" * 280) + "\n\n" + ("第二段。" * 310) + "\n最后一行。"
    parts = query_service.split_reply_parts(text, 500)
    assert len(parts) > 2
    assert all(0 < len(part) <= 500 for part in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
    assert "已安全截断" not in "".join(parts)


def test_canonical_query_result_tracks_offerclaw_calls_without_content(
        monkeypatch, tmp_path):
    def gated(question, top_k, **kwargs):
        record_model_request({"model": "planner-model", "messages": [question]})
        record_model_response({"model": "planner-model", "usage": {
            "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16,
        }})
        return {
            "answer": "规范答案", "mode": "multi_source", "service_mode": "advise",
            "routes": [{"source": "profile_plan", "operation": "get_profile"}],
            "sources": ["user_profile.md"], "freshness": "2026-09-13T12:00:00+08:00",
            "intent_frame": {"service_mode": "advise"}, "decision": "answer",
        }

    monkeypatch.setitem(sys.modules, "rag_gate", types.SimpleNamespace(gated_query=gated))
    monkeypatch.setattr(business_read_service, "authoritative_data_version", lambda: "data-v1")
    result = query_service.execute_query(
        "结合我的画像给建议", conversation_id="conv_opaque", timeout_seconds=5,
        trace_id="trace-fixed",
    )
    payload = result.to_dict()
    assert payload["answer"] == "规范答案"
    assert payload["service_mode"] == "advise"
    assert payload["model_usage"] == {
        "offerclaw_calls": 1, "openclaw_calls": 0,
        "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16,
        "models": ["planner-model"],
    }
    assert "结合我的画像给建议" not in json.dumps(payload["model_usage"], ensure_ascii=False)


def _fake_execution() -> query_service.QueryExecutionResult:
    return query_service.QueryExecutionResult(
        payload={
            "answer": "Windows 规范答案", "mode": "multi_source",
            "service_mode": "recall", "answer_action": "answer",
            "routes": [{"source": "profile_plan", "operation": "get_profile"}],
            "sources": ["user_profile.md"], "freshness": "2026-09-13T12:00:00+08:00",
            "intent_frame": {"service_mode": "recall"}, "decision": "answer",
        },
        model_usage={"offerclaw_calls": 2, "openclaw_calls": 0, "models": ["m"]},
        data_version="a" * 64, trace_id="wxquery_fixed", status="ok",
    )


def test_internal_endpoint_requires_loopback_origin_and_token(tmp_path, monkeypatch):
    import conversation_context
    import rag_api

    token_path = tmp_path / "wechat-query.token"
    token_path.write_text("a" * 64, encoding="ascii")
    monkeypatch.setenv("OFFERCLAW_WECHAT_QUERY_TOKEN_FILE", str(token_path))
    monkeypatch.setattr(query_service, "execute_query", lambda *_args, **_kwargs: _fake_execution())
    compact = []
    monkeypatch.setattr(
        conversation_context, "record_successful_turn",
        lambda conversation_id, result, **kwargs: compact.append((conversation_id, result, kwargs)),
    )
    body = {
        "schema_version": "offerclaw.wechat-query.request.v1",
        "question": "我的画像是什么？",
        "conversation_id": "conv_" + "1" * 64,
        "message_id": "msg_" + "2" * 64,
        "operation_id": "op_" + "3" * 64,
        "top_k": 5,
    }
    headers = {
        "X-OfferClaw-Internal-Token": "a" * 64,
        "X-OfferClaw-Traffic-Origin": "wechat_direct",
    }
    loopback = TestClient(rag_api.app, client=("127.0.0.1", 50000))
    response = loopback.post("/api/internal/wechat-query", json=body, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert result["answer"].startswith("Windows 规范答案")
    assert result["model_usage"]["openclaw_calls"] == 0
    assert result["reply_parts"]
    assert compact and compact[0][0] == body["conversation_id"]

    assert loopback.post(
        "/api/internal/wechat-query", json=body,
        headers={**headers, "X-OfferClaw-Internal-Token": "b" * 64},
    ).status_code == 401
    remote = TestClient(rag_api.app, client=("192.168.1.10", 50000))
    assert remote.post(
        "/api/internal/wechat-query", json=body, headers=headers,
    ).status_code == 403
    assert loopback.post(
        "/api/internal/wechat-query", json={**body, "path": "C:\\private"}, headers=headers,
    ).status_code == 422


def test_windows_bridge_query_answer_has_fixed_contract(monkeypatch):
    import wechat_data_bridge

    seen = []
    monkeypatch.setattr(
        wechat_data_bridge, "_query_service_request",
        lambda path, **kwargs: seen.append((path, kwargs)) or {
            "status": "ok", "answer": "ok", "reply_parts": ["ok"],
        },
    )
    result = wechat_data_bridge.handle({
        "schema_version": wechat_data_bridge.REQUEST_SCHEMA,
        "operation": "query.answer",
        "payload": {
            "question": "问题", "conversation_id": "conv_" + "1" * 64,
            "message_id": "msg_" + "2" * 64, "operation_id": "op_" + "3" * 64,
            "top_k": 5,
        },
    })
    assert result["status"] == "ok"
    assert seen[0][0] == "/api/internal/wechat-query"
    assert set(seen[0][1]["payload"]) == {
        "schema_version", "question", "conversation_id", "message_id", "operation_id", "top_k",
    }
    with pytest.raises(ValueError, match="未知字段"):
        wechat_data_bridge.handle({
            "schema_version": wechat_data_bridge.REQUEST_SCHEMA,
            "operation": "query.answer",
            "payload": {"question": "问题", "url": "http://example.invalid"},
        })
