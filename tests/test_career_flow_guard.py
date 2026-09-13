"""A3 节点级 guard + patch 语义测试。

量化目标：节点不原地修改 state；异常被转成 errors/trace patch。
关键节点额外返回 fatal_error，供路由明确停止。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from career_flow import node_guard


def test_node_guard_catches_records_continues():
    def boom_node(state):
        raise RuntimeError("injected match failure")

    state = {}
    out = node_guard(boom_node)(state)
    assert state == {}                                           # 输入 state 不变
    assert any(e["node"] == "boom" for e in out["errors"])
    assert any(t["action"] == "node_failed" for t in out["trace"])
    assert not out.get("fatal_error")                            # 默认是 optional


def test_node_guard_passthrough_on_success():
    def ok_node(state):
        return {"x": 1}

    state = {}
    assert node_guard(ok_node)(state)["x"] == 1                  # 正常节点透明放行
    assert not state.get("errors")                                # 不误记错误


def test_node_guard_critical_failure_marks_fatal():
    def match_node(state):
        raise RuntimeError("matcher unavailable")

    out = node_guard(match_node, fail_policy="critical")({})
    assert out["fatal_error"] is True
    assert out["errors"][0]["fatal"] is True


def test_node_guard_preserves_name():
    def match_node(state):
        return state

    assert node_guard(match_node).__name__ == "match_node"        # functools.wraps 保留元信息
