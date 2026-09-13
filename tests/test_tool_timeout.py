"""A7 工具执行墙钟预算测试。

量化目标：工具死循环/阻塞超过 TOOL_TIMEOUT 时，Tool.call 在预算内返回 {'error':'tool_timeout'}
而非无限期挂死整个 ReAct loop；正常工具不受影响。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools_registry import Tool


def _tool(fn, name="t"):
    return Tool(name=name, description="d", parameters={}, fn=fn)


def test_tool_timeout_does_not_hang(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT", "0.3")
    t = _tool(lambda: time.sleep(5), name="slow")
    start = time.time()
    out = t.call()
    elapsed = time.time() - start
    assert out.get("error") == "tool_timeout" and out["tool"] == "slow"
    assert elapsed < 2.0                          # 0.3s 内返回，不等 5s（loop 不挂死）


def test_tool_normal_within_budget():
    assert _tool(lambda: {"ok": 1}).call() == {"ok": 1}


def test_tool_non_dict_wrapped():
    assert _tool(lambda: "hello").call() == {"result": "hello"}


def test_tool_param_mismatch_handled():
    out = _tool(lambda x: x).call()               # 缺必填参数 x
    assert "参数不匹配" in out.get("error", "")


def test_tool_exception_wrapped():
    def boom():
        raise ValueError("boom")
    assert "ValueError" in _tool(boom).call().get("error", "")
