# -*- coding: utf-8 -*-
"""mcp_server.py — OfferClaw MCP Server（Streamable HTTP 传输，手写协议层）

把 ``tools_registry.REGISTRY`` 的全部工具按 **MCP（Model Context Protocol）**
规范暴露给任意 MCP 客户端（Claude Code / Cursor / 自研 Agent 等）。

传输选型（面试可拆解的三个决策）：
1. **只做 Streamable HTTP，不做 stdio**——OfferClaw 本体是常驻 FastAPI 服务
   （45+ 路由），MCP 端点挂在同一服务的 ``POST /mcp`` 上正是生产部署形态；
   旧版 HTTP+SSE 双端点传输已于 2025-03-26 规范弃用，本实现直接按新规范落地。
2. **无状态模式**——规范允许服务器不分配 ``Mcp-Session-Id``；单用户本地服务
   无会话恢复需求，无状态实现最简且横向扩展友好（无粘性会话）。
3. **响应统一 application/json**——规范允许服务器对 POST 请求返回单个 JSON
   而非 SSE 流；本项目工具均为确定性快速调用（且自带 A7 墙钟预算），无需流式。
   GET /mcp（服务器主动推流）规范允许不提供 → FastAPI 自动 405。

安全（规范 MUST 项）：校验 ``Origin`` 头防 DNS rebinding——浏览器场景下恶意
页面可把外部域名解析到 127.0.0.1 迂回访问本地服务；仅放行本机来源与无 Origin
的服务端调用。

协议覆盖：initialize / notifications/* / ping / tools/list / tools/call；
tools/list 从 REGISTRY 动态生成（schema 与执行零重复定义，新增工具自动同步，
配套漂移校验测试见 tests/test_mcp_server.py）。
"""

from __future__ import annotations

import json
import os
import sys
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools_registry import REGISTRY

# 协议版本协商：客户端请求的版本在支持列表内则回显，否则回落到基线版本。
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26")
DEFAULT_PROTOCOL_VERSION = "2025-03-26"

SERVER_INFO = {"name": "offerclaw-mcp", "version": "1.0.0"}
SERVER_INSTRUCTIONS = (
    "OfferClaw 求职 Agent 的工具集：JD 匹配三档结论、JD 抽取、简历骨架、"
    "今日建议、投递记录查询、CareerFlow 全流程。所有工具确定性执行、不调 LLM。"
)

# Origin 白名单：本机来源；无 Origin（curl / 服务端调用）放行。
_ALLOWED_ORIGIN_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

# JSON-RPC 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def origin_allowed(origin: str | None) -> bool:
    """规范 MUST：校验 Origin 防 DNS rebinding。无 Origin = 非浏览器调用，放行。"""
    if not origin:
        return True
    try:
        host = urlparse(origin).hostname or ""
    except ValueError:
        return False
    return host in _ALLOWED_ORIGIN_HOSTS


def tool_to_mcp_schema(tool) -> dict:
    """OpenAI function schema → MCP tool 声明：parameters 本就是 JSON Schema，直接复用。"""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.parameters or {"type": "object", "properties": {}},
    }


def _ok(msg_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _handle_initialize(msg_id, params: dict) -> dict:
    requested = str(params.get("protocolVersion", ""))
    version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
    return _ok(msg_id, {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": SERVER_INFO,
        "instructions": SERVER_INSTRUCTIONS,
    })


def _handle_tools_list(msg_id) -> dict:
    tools = [tool_to_mcp_schema(REGISTRY.get(n)) for n in REGISTRY.list_names()]
    return _ok(msg_id, {"tools": tools})


def _handle_tools_call(msg_id, params: dict) -> dict:
    name = params.get("name", "")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _err(msg_id, INVALID_PARAMS, "arguments 必须是对象")
    try:
        tool = REGISTRY.get(name)
    except KeyError:
        return _err(msg_id, INVALID_PARAMS, f"Unknown tool: {name}")
    # Tool.call 永远返回 dict（异常已包成 {"error": ...}，且带 A7 墙钟预算）
    out = tool.call(**arguments)
    is_error = isinstance(out, dict) and "error" in out
    text = json.dumps(out, ensure_ascii=False, default=str)
    # 规范：工具执行失败放 result.isError，不升级为协议错误（LLM 可读错误自行决策）
    return _ok(msg_id, {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    })


def handle_mcp_message(raw_body: bytes) -> tuple[int, dict | None]:
    """处理一条 Streamable HTTP POST 消息。

    返回 ``(http_status, response_json)``；``response_json is None`` 表示
    202 Accepted 无响应体（通知类消息，规范要求）。
    """
    try:
        msg = json.loads(raw_body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 400, _err(None, PARSE_ERROR, "Parse error: 请求体不是合法 JSON")

    # JSON-RPC 批处理：2025-06-18 版规范已移除，本实现不支持
    if isinstance(msg, list):
        return 400, _err(None, INVALID_REQUEST, "不支持 JSON-RPC batch（2025-06-18 规范已移除批处理）")
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return 400, _err(None, INVALID_REQUEST, "不是合法的 JSON-RPC 2.0 消息")

    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # 通知（无 id）：规范要求 202 Accepted 无响应体
    if msg_id is None:
        return 202, None

    if method == "initialize":
        return 200, _handle_initialize(msg_id, params)
    if method == "ping":
        return 200, _ok(msg_id, {})
    if method == "tools/list":
        return 200, _handle_tools_list(msg_id)
    if method == "tools/call":
        return 200, _handle_tools_call(msg_id, params)
    return 200, _err(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
