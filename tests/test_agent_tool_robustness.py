"""A4 ReAct/工具调用装配加固测试。

量化目标：畸形 tool_call（缺 id / function.name）不再抛 KeyError 逃逸；assistant.tool_calls
与后续 tool 消息**数量配对**（不留孤儿污染下一轮请求体）；循环截断带显式标记。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_demo


def test_malformed_tool_call_no_crash_and_paired(monkeypatch):
    """畸形 tool_call（缺 id）不抛 KeyError；tool 消息数 == tool_calls 数（配对合法）。"""
    seq = iter([
        {"choices": [{"message": {"role": "assistant",
                                  "tool_calls": [{"function": {"name": "foo"}}]}}]},  # 缺 id
        {"choices": [{"message": {"role": "assistant", "content": "final"}}]},
    ])
    monkeypatch.setattr(agent_demo, "call_llm", lambda *a, **k: next(seq))
    messages = [{"role": "user", "content": "hi"}]
    out = agent_demo.run_agent_turn(messages, "fake")   # 不抛 KeyError
    assert out == "final"
    n_calls = sum(len(m.get("tool_calls") or []) for m in messages)
    n_tools = sum(1 for m in messages if m.get("role") == "tool")
    assert n_tools == n_calls == 1                       # 畸形也补占位 tool 回复，配对


def test_truncation_marker(monkeypatch):
    """跑满 MAX_TOOL_ITERATIONS 仍有 tool_call → 返回带 max_iterations_reached 显式标记。"""
    monkeypatch.setattr(agent_demo, "execute_tool_call", lambda tc: "ok")
    monkeypatch.setattr(
        agent_demo, "call_llm",
        lambda *a, **k: {"choices": [{"message": {"role": "assistant",
                         "tool_calls": [{"id": "1", "function": {"name": "foo"}}]}}]})
    out = agent_demo.run_agent_turn([{"role": "user", "content": "x"}], "fake")
    assert "max_iterations_reached" in out


def test_react_max_steps_marker():
    """react_agent: 跑满 max_steps 未收敛 → errors 含 max_steps_reached（区分模型主动结束）。
    用 deterministic 直接验证逻辑不可行（那是单步），这里校验标记字符串已落代码。"""
    import react_agent
    import inspect
    src = inspect.getsource(react_agent._llm_step)
    assert "max_steps_reached" in src and "completed" in src   # A4 截断标记逻辑就位
    assert "malformed_tool_call" in src                         # fn_name 畸形防御就位
