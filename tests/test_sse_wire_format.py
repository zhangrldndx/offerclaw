# -*- coding: utf-8 -*-
"""SSE wire 格式统一（前端 P0 还债）的钉死测试。

历史问题：三个流式端点曾各说各话——/api/plan/stream 用 ``{text}``+``{done:true}``、
/api/stream 用命名事件 ``event: meta`` + ``{delta}``、/api/resume/*/stream 用
``[DONE]`` 字符串哨兵——前端被迫维护三套互不兼容的手写解析循环。

统一后（rag_api.py::_sse_event）：每条消息一行 ``data: <json>``，payload 必带
``type`` ∈ {meta, delta, done, error}，delta 文本统一在 ``text`` 字段。
本文件三层钉死：

  1. 单元：_sse_event 输出形状。
  2. 源码级 lint：rag_api.py 不允许出现绕过 helper 的裸 ``yield "data:`` /
     ``[DONE]`` 哨兵——新增流式端点必须走 _sse_event，否则此测试直接红。
  3. 端到端：/api/stream 打桩后逐行断言统一序列（stage → meta → delta → done），
     且不再出现旧格式的 ``event:`` 命名行 / ``delta`` 字段。
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RAG_API_SRC = Path(__file__).resolve().parent.parent / "rag_api.py"


# ---------------------------------------------------------------- 1. 单元

def test_sse_event_shape():
    import rag_api

    line = rag_api._sse_event({"type": "delta", "text": "你好"})
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    payload = json.loads(line[len("data: "):])
    assert payload == {"type": "delta", "text": "你好"}
    # ensure_ascii=False —— 中文不转义，前端直接可读
    assert "你好" in line


# ---------------------------------------------------- 2. 源码级 lint（防绕过）

def test_no_stream_yield_bypasses_sse_event():
    src = RAG_API_SRC.read_text(encoding="utf-8")
    assert "[DONE]" not in src, "字符串哨兵 [DONE] 不得回潮——统一用 type:done 事件"
    bare_yields = re.findall(r'yield f?"(?:event:|data:)', src)
    assert not bare_yields, (
        f"发现 {len(bare_yields)} 处绕过 _sse_event 的裸 SSE yield——"
        "所有流式输出必须经 _sse_event()"
    )


# ------------------------------------------------------------ 3. 端到端

@pytest.fixture()
def client(monkeypatch):
    # 端点内是 ``from rag_gate import gated_query_stream``（函数内 import），
    # 所以打桩要落在 rag_gate 模块上。
    import rag_api
    import rag_gate

    def fake_stream(query, top_k):
        yield {"type": "stage", "stage": "planning", "status": "active",
               "label": "正在理解问题并规划检索路径", "detail": "识别数据源"}
        yield {"type": "meta", "in_kb": True, "mode": "kb",
               "sources": ["a.md"], "matched_by": "dense", "best_distance": 0.1,
               "planner_mode": "rule", "service_mode": "explain",
               "requested_action": "",
               "routes": [{"source": "reference_kb", "operation": "search",
                           "depends_on": ["application_state"]}],
               "source_status": {"reference_kb": {"status": "ok", "count": 1}},
               "coverage": {"reference_kb": 1}, "freshness": "2026-08-21",
               "resolved_entities": {"companies": ["甲公司"], "positions": ["AI工程师"]},
               "latency_ms": {"planning": 1, "retrieval": 2}}
        yield {"type": "delta", "text": "第一段"}
        yield {"type": "delta", "text": "第二段"}
        yield {"type": "done"}

    monkeypatch.setattr(rag_gate, "gated_query_stream", fake_stream)
    return TestClient(rag_api.app)


def _events(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if not line.strip():
            continue
        assert line.startswith("data: "), f"非统一格式行: {line!r}"
        out.append(json.loads(line[len("data: "):]))
    return out


def test_api_stream_unified_sequence(client):
    r = client.post("/api/stream", json={"query": "什么是RAG", "top_k": 5})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    # 旧格式不得回潮
    assert "event:" not in body
    assert '"delta":' not in body  # 旧格式的 {delta: ...} 字段名（key 位）；
    # 新格式里 delta 只出现在 "type" 的值位（"type": "delta"）

    events = _events(body)
    assert [e["type"] for e in events] == ["stage", "meta", "delta", "delta", "done"]
    assert events[0]["stage"] == "planning"
    assert events[0]["status"] == "active"
    assert events[1]["in_kb"] is True
    assert events[1]["sources"] == ["a.md"]
    assert events[1]["planner_mode"] == "rule"
    assert events[1]["service_mode"] == "explain"
    assert events[1]["requested_action"] == ""
    assert events[1]["routes"][0]["source"] == "reference_kb"
    assert events[1]["coverage"] == {"reference_kb": 1}
    assert events[1]["resolved_entities"]["companies"] == ["甲公司"]
    assert events[2]["text"] == "第一段"


def test_api_stream_error_event(client, monkeypatch):
    import rag_gate

    def boom(query, top_k):
        raise RuntimeError("检索炸了")
        yield  # pragma: no cover — 使其成为 generator

    monkeypatch.setattr(rag_gate, "gated_query_stream", boom)
    r = client.post("/api/stream", json={"query": "x", "top_k": 1})
    events = _events(r.text)
    assert events[-1]["type"] == "error"
    assert "检索炸了" in events[-1]["error"]
