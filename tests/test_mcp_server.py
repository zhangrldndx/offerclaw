# -*- coding: utf-8 -*-
"""MCP Server（Streamable HTTP）协议合规 + 注册表漂移测试。

覆盖：
1. initialize 握手（版本协商：支持版本回显 / 未知版本回落基线）；
2. 通知消息 → 202 无响应体（规范要求）；
3. tools/list ↔ REGISTRY 漂移校验（A8 同款纪律：schema 单一来源）；
4. tools/call 成功路径（确定性工具，零 LLM）与 isError 路径；
5. JSON-RPC 协议错误：parse error / batch 拒绝 / method not found / unknown tool；
6. 安全：Origin 校验防 DNS rebinding（恶意来源 403，本机/无 Origin 放行）；
7. 传输面：GET /mcp → 405（无状态服务器不提供推流，规范允许）。
"""
import json
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient

from mcp_server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    handle_mcp_message,
    origin_allowed,
)
from tools_registry import REGISTRY


@pytest.fixture(scope="module")
def client():
    from rag_api import app
    return TestClient(app)


def _rpc(method, msg_id=1, params=None):
    body = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


# ---------- 1. initialize 握手 ----------

def test_initialize_echoes_supported_version(client):
    r = client.post("/mcp", json=_rpc("initialize", params={
        "protocolVersion": SUPPORTED_PROTOCOL_VERSIONS[0],
        "capabilities": {}, "clientInfo": {"name": "t", "version": "0"},
    }))
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "offerclaw-mcp"


def test_initialize_falls_back_on_unknown_version(client):
    r = client.post("/mcp", json=_rpc("initialize", params={"protocolVersion": "1999-01-01"}))
    assert r.json()["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION


# ---------- 2. 通知 → 202 ----------

def test_notification_returns_202_no_body(client):
    r = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202
    assert r.content == b""


# ---------- 3. tools/list ↔ REGISTRY 漂移校验 ----------

def test_tools_list_matches_registry_exactly(client):
    r = client.post("/mcp", json=_rpc("tools/list"))
    tools = r.json()["result"]["tools"]
    assert {t["name"] for t in tools} == set(REGISTRY.list_names())
    for t in tools:
        assert t["description"], f"{t['name']} 缺 description"
        assert isinstance(t["inputSchema"], dict) and t["inputSchema"].get("type") == "object", \
            f"{t['name']} 的 inputSchema 不是合法 JSON Schema 对象"


# ---------- 4. tools/call ----------

def test_tools_call_deterministic_tool(client):
    r = client.post("/mcp", json=_rpc("tools/call", params={
        "name": "list_applications", "arguments": {},
    }))
    result = r.json()["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert isinstance(payload, dict)


def test_tools_call_tool_failure_maps_to_is_error(client):
    # match_jd 缺必填参数 → Tool.call 包成 {"error": ...} → isError=True（不升级为协议错误）
    r = client.post("/mcp", json=_rpc("tools/call", params={
        "name": "match_jd", "arguments": {},
    }))
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["isError"] is True


def test_tools_call_unknown_tool_is_invalid_params(client):
    r = client.post("/mcp", json=_rpc("tools/call", params={"name": "no_such_tool"}))
    assert r.json()["error"]["code"] == -32602


# ---------- 5. 协议错误 ----------

def test_parse_error_on_malformed_json(client):
    r = client.post("/mcp", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32700


def test_batch_rejected(client):
    r = client.post("/mcp", json=[_rpc("ping", 1), _rpc("ping", 2)])
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600


def test_method_not_found(client):
    r = client.post("/mcp", json=_rpc("resources/list"))
    assert r.json()["error"]["code"] == -32601


# ---------- 6. Origin 安全 ----------

def test_evil_origin_rejected(client):
    r = client.post("/mcp", json=_rpc("ping"), headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 403


def test_local_and_absent_origin_allowed():
    assert origin_allowed(None) is True
    assert origin_allowed("http://localhost:8000") is True
    assert origin_allowed("http://127.0.0.1:3000") is True
    assert origin_allowed("https://attacker.io") is False


# ---------- 7. 传输面 ----------

def test_get_mcp_returns_405(client):
    assert client.get("/mcp").status_code == 405


def test_ping(client):
    r = client.post("/mcp", json=_rpc("ping"))
    assert r.json()["result"] == {}


# ---------- 纯函数层（不经 HTTP）----------

def test_handle_message_pure_function_notification():
    status, payload = handle_mcp_message(
        json.dumps({"jsonrpc": "2.0", "method": "notifications/x"}).encode())
    assert status == 202 and payload is None


def test_handle_message_rejects_non_jsonrpc():
    status, payload = handle_mcp_message(b'{"foo": 1}')
    assert status == 400 and payload["error"]["code"] == -32600
