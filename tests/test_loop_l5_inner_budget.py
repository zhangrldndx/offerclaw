# -*- coding: utf-8 -*-
"""[L5] 内循环 token 预算 + 工具结果智能截断 —— 量化测试。

baseline（改造前）：单轮 ReAct 在 messages 无界累积；工具结果硬截 ``[:2000]``（会把后置的
关键字段如 error 切掉）；名义 max_steps 越高越易在第 N 轮 LLM 调用时才爆窗。
改造后：① 每步前预算预检，超 REACT_MAX_CTX_CHARS 提前 ``ctx_budget_exhausted`` 止损；
② 工具结果智能截断——dict 优先保留 error/status/summary 等关键字段。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import react_agent
from react_agent import _truncate_tool_result, _messages_chars, REACT_MAX_CTX_CHARS


# ── 智能截断 ─────────────────────────────────────────────────

def test_small_result_unchanged():
    r = {"status": "ok", "value": 42}
    assert _truncate_tool_result(r, limit=2000) == json.dumps(r, ensure_ascii=False)


def test_smart_truncate_preserves_key_fields_vs_hard_cut():
    # 关键字段 error 放在大 data 之后：硬截 [:2000] 会把它切掉，智能截断保留
    big = {"data": "x" * 5000, "error": "CRITICAL_FAILURE", "status": "failed"}
    hard = json.dumps(big, ensure_ascii=False)[:2000]
    assert "CRITICAL_FAILURE" not in hard                 # 改造前：硬截丢掉 error
    smart = _truncate_tool_result(big, limit=2000)
    assert "CRITICAL_FAILURE" in smart and "failed" in smart   # 改造后：关键字段保留
    assert len(smart) <= 2000 and '"_truncated": true' in smart


def test_non_dict_tail_truncated():
    s = _truncate_tool_result("y" * 5000, limit=100)
    assert len(s) <= 100 and s.endswith("…[截断]")          # 严格 ≤ limit（含 marker 已预留）


def test_key_fields_too_big_falls_back():
    big = {"error": "z" * 5000}                           # 关键字段本身就超 → 退回尾截
    out = _truncate_tool_result(big, limit=200)
    assert len(out) <= 200                                # 严格 ≤ limit


# ── 上下文预算闸 ─────────────────────────────────────────────

def test_messages_chars_sums_content():
    msgs = [{"role": "user", "content": "ab"}, {"role": "tool", "content": "cde"}]
    assert _messages_chars(msgs) == 5


def test_budget_condition_detects_overflow():
    msgs = [{"role": "tool", "content": "x" * (REACT_MAX_CTX_CHARS + 1)}]
    assert _messages_chars(msgs) > REACT_MAX_CTX_CHARS    # 超预算可被预检捕获


# ── 接线断言（_llm_step 的 HTTP 直连 mock 成本高，沿用 A4 源码断言模式）──

def test_budget_and_truncate_wired_into_llm_step():
    import inspect
    src = inspect.getsource(react_agent._llm_step)
    assert "ctx_budget_exhausted" in src                  # 预算预检止损已接线
    assert "_truncate_tool_result(result)" in src         # 智能截断已替换硬截 [:2000]
    assert "[:2000]" not in src                            # 旧硬截已移除
