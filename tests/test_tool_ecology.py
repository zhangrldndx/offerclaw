# -*- coding: utf-8 -*-
"""Tool ecology + ReAct Agent 测试（V4 §3）。

所有测试默认不依赖 LLM。LLM mode 仅在当前 chat-completion key 存在时跑一条 smoke。
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from day1_api_starter import API_KEY_ENV


JD_AI = (
    "岗位名称：AI Agent 应用开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：硕士及以上\n经验要求：实习\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)


# --------- Tool 注册表自身 ---------

def test_registry_lists_all_expected_tools():
    from tools_registry import REGISTRY

    names = REGISTRY.list_names()
    expected = {"match_jd", "extract_jd", "resume_skeleton",
                "today_advice", "list_applications", "career_flow"}
    assert expected.issubset(set(names)), f"缺 tool：{expected - set(names)}"


def test_registry_to_openai_schemas_well_formed():
    from tools_registry import REGISTRY

    schemas = REGISTRY.to_openai_schemas()
    assert len(schemas) >= 6
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        for k in ("name", "description", "parameters"):
            assert k in fn
        assert fn["parameters"]["type"] == "object"


def test_registry_register_duplicate_raises():
    from tools_registry import ToolRegistry, Tool

    r = ToolRegistry()
    t = Tool(name="x", description="d", parameters={"type": "object"},
             fn=lambda: {"ok": True})
    r.register(t)
    with pytest.raises(ValueError):
        r.register(t)


def test_tool_call_wraps_typeerror():
    """Tool 收到不匹配参数必须返回 error dict，而不是抛 TypeError。"""
    from tools_registry import REGISTRY
    out = REGISTRY.get("match_jd").call(wrong_arg="oops")
    assert "error" in out


# --------- 各 tool 单测 ---------

def test_tool_match_jd_returns_status():
    from tools_registry import REGISTRY
    out = REGISTRY.get("match_jd").call(jd_text=JD_AI, jd_title="AI 实习")
    assert out["status"] in ("当前适合投递", "当前暂不建议投递", "中长期可转向")
    assert "direction" in out
    assert isinstance(out["gap_count"], int)


def test_tool_extract_jd_returns_structured():
    from tools_registry import REGISTRY
    out = REGISTRY.get("extract_jd").call(raw=JD_AI)
    assert isinstance(out.get("skills_detected"), list)
    # 至少识别出 Python / LangGraph 中的一个
    found = [s.lower() for s in out["skills_detected"]]
    assert any("python" in s or "langgraph" in s or "rag" in s for s in found)


def test_tool_resume_skeleton_returns_keywords():
    from tools_registry import REGISTRY
    out = REGISTRY.get("resume_skeleton").call(jd_text=JD_AI)
    assert "headline" in out and out["headline"]
    assert isinstance(out["keywords_hit"], list)
    # JD 含 python / langgraph / rag，应至少命中 1 个
    assert len(out["keywords_hit"]) >= 1


def test_tool_today_advice_returns_headline():
    from tools_registry import REGISTRY
    out = REGISTRY.get("today_advice").call()
    assert "today" in out
    assert isinstance(out.get("headline"), str)


def test_tool_list_applications_returns_total():
    from tools_registry import REGISTRY
    out = REGISTRY.get("list_applications").call(limit=5)
    assert "total" in out and isinstance(out["total"], int)
    assert isinstance(out["items"], list)
    assert len(out["items"]) <= 5


def test_tool_career_flow_runs_end_to_end():
    from tools_registry import REGISTRY
    out = REGISTRY.get("career_flow").call(jd_text=JD_AI, jd_title="AI 实习")
    assert out["status"] in ("当前适合投递", "当前暂不建议投递", "中长期可转向")
    assert out["route_taken"]
    assert isinstance(out["resume_keywords"], list)


# --------- ReAct deterministic ---------

@pytest.mark.parametrize("message,expected_tool", [
    ("今天该做什么？", "today_advice"),
    ("我投了哪些岗位？", "list_applications"),
    ("帮我跑完整 CareerFlow：" + JD_AI, "career_flow"),
    ("帮我抽取这段 JD：" + JD_AI, "extract_jd"),
    ("匹配一下：" + JD_AI, "match_jd"),
    ("给我个简历骨架：" + JD_AI, "resume_skeleton"),
])
def test_react_deterministic_picks_correct_tool(message, expected_tool):
    from react_agent import run
    out = run(message, mode="deterministic")
    assert out["mode"] == "deterministic"
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == expected_tool


def test_react_deterministic_no_match_returns_help():
    from react_agent import run
    out = run("怎么改 Tailwind 主题色？", mode="deterministic")
    assert out["tool_calls"] == []
    assert "no_route_matched" in out["errors"]


def test_react_deterministic_match_without_jd_in_message():
    """识别到 match 意图但消息里没 JD（消息太短）→ 友好提示，不调 tool。"""
    from react_agent import run
    out = run("匹配", mode="deterministic")
    assert out["tool_calls"] == []
    assert "no_jd_in_message" in out["errors"]


def test_react_llm_mode_fallback_when_no_key(monkeypatch):
    """没有 KEY 时 LLM 模式必须降级到 deterministic 且能跑通。

    注意：``get_llm_config()`` 内部会调用 ``load_local_env()`` 重新读 .env.local，
    若用 ``delenv`` 删除后该 key 仍会被 .env.local 重新注入，无法真正模拟"无 key"。
    故用 ``setenv("")`` —— load_local_env 不会覆盖已存在（即使为空）的 key，
    从而让 ``get_llm_config()`` 返回空 key，正确触发降级路径。
    """
    monkeypatch.setenv(API_KEY_ENV, "")
    monkeypatch.setenv("ZHIPU_API_KEY", "")
    from react_agent import run
    out = run("今天该做什么？", mode="llm")
    # 降级后 mode 仍为 deterministic
    assert out["mode"] == "deterministic"
    assert f"no_{API_KEY_ENV.lower()}:fallback_to_deterministic" in out["errors"]
    assert out["tool_calls"][0]["name"] == "today_advice"


@pytest.mark.skipif(
    not os.environ.get(API_KEY_ENV),
    reason=f"需要 {API_KEY_ENV} 才能跑真实 LLM 模式",
)
def test_react_llm_mode_real_smoke():
    from react_agent import run
    out = run("今天我应该做什么？", mode="llm", max_steps=2)
    assert out["mode"] == "llm"
    # 至少调一次 tool 或者直接给答复
    assert out["answer"]


# --------- /api/agent endpoint ---------

def test_api_agent_endpoint_deterministic():
    from fastapi.testclient import TestClient
    from rag_api import app
    client = TestClient(app)
    resp = client.post("/api/agent", json={
        "message": "今天该做什么？",
        "mode": "deterministic",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "deterministic"
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["name"] == "today_advice"
    assert body["answer"]


def test_api_agent_endpoint_match_jd_via_message():
    from fastapi.testclient import TestClient
    from rag_api import app
    client = TestClient(app)
    resp = client.post("/api/agent", json={
        "message": "匹配一下：" + JD_AI,
        "mode": "deterministic",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tool_calls"][0]["name"] == "match_jd"
    result = body["tool_calls"][0]["result"]
    assert result.get("status") in (
        "当前适合投递", "当前暂不建议投递", "中长期可转向")
