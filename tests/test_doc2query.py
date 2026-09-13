# -*- coding: utf-8 -*-
"""doc2query（Round 9 / P1）单元测试：纯函数合并逻辑 + 解析健壮性 + 接线纪律。

策略沿用 A4/L5：LLM/HTTP 不真调（成本高且不稳定），核心合并逻辑抽成纯函数直测；
默认关闭的行为用源码接线断言钉死（防止未来有人把 d2q 从 flag 后面挪出来）。
"""
import json
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rag_doc2query import (_parse_questions, d2q_enabled, merge_parent_hits,
                           QUESTIONS_PER_CHUNK)


# ---------- merge_parent_hits 纯函数 ----------

def _mk(src):
    return {"source": src, "source_type": "t"}


def test_merge_new_parent_appended_and_sorted():
    docs, metas, dists = ["原文A"], [_mk("a.md")], [0.5]
    hits = [("pid1", "原文B", _mk("b.md"), 0.2)]
    d, m, dd = merge_parent_hits(docs, metas, dists, hits)
    assert d == ["原文B", "原文A"]          # 距离更小的父块排前
    assert dd == [0.2, 0.5]


def test_merge_same_parent_keeps_min_distance():
    docs, metas, dists = ["原文A"], [_mk("a.md")], [0.5]
    hits = [("pid1", "原文A", _mk("a.md"), 0.3)]   # 同一父块,合成问题距离更近
    d, m, dd = merge_parent_hits(docs, metas, dists, hits)
    assert len(d) == 1 and dd == [0.3]             # 不重复、取更小距离


def test_merge_same_parent_worse_distance_ignored():
    docs, metas, dists = ["原文A"], [_mk("a.md")], [0.2]
    hits = [("pid1", "原文A", _mk("a.md"), 0.9)]
    _, _, dd = merge_parent_hits(docs, metas, dists, hits)
    assert dd == [0.2]                             # 更差的距离不覆盖


def test_merge_missing_parent_skipped():
    d, m, dd = merge_parent_hits([], [], [], [("pid", None, _mk("x"), 0.1)])
    assert d == [] and dd == []                    # 父块已删（失效映射）不炸、不混入 None


# ---------- 问题解析健壮性 ----------

def test_parse_clean_json_array():
    qs = _parse_questions('["问题一是什么", "问题二怎么办", "问题三行不行"]')
    assert len(qs) == 3


def test_parse_json_wrapped_in_prose():
    raw = '好的，以下是问题：\n["数据库崩了怎么恢复", "为啥读写不互相挡"]\n希望有帮助！'
    qs = _parse_questions(raw)
    assert qs == ["数据库崩了怎么恢复", "为啥读写不互相挡"]


def test_parse_degrades_to_lines():
    raw = "1. 这个东西是干嘛用的\n2. 不用它行不行\n3. 它和老办法差在哪"
    qs = _parse_questions(raw)
    assert len(qs) == 3 and qs[0] == "这个东西是干嘛用的"


def test_parse_caps_at_limit():
    arr = json.dumps([f"问题{i}号是什么" for i in range(10)], ensure_ascii=False)
    assert len(_parse_questions(arr)) == QUESTIONS_PER_CHUNK


def test_parse_garbage_returns_empty():
    assert _parse_questions("") == []
    assert _parse_questions("嗯。") == []


# ---------- 默认关 + 接线纪律 ----------

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_DOC2QUERY", raising=False)
    assert d2q_enabled() is False


def test_gate_wiring_is_flag_guarded():
    """源码断言：rag_gate 里 d2q 合并必须在 d2q_enabled() 判断之内(A4 同款纪律)。"""
    src = open(os.path.join(BASE_DIR, "rag_gate.py"), encoding="utf-8").read()
    assert "query_d2q_and_merge" in src
    # 2026-08-22 重构后守卫变为 profile 感知:d2q_active = d2q_enabled() 或 profile 显式覆盖,
    # 再 if d2q_active: 调用。断言意图不变——调用必须位于源自 d2q_enabled() 的开关之后。
    derive_pos = src.find("d2q_active = (d2q_enabled()")
    guard_pos = src.find("if d2q_active:")
    call_pos = src.find("query_d2q_and_merge(emb")
    assert 0 < derive_pos < guard_pos < call_pos, "d2q 调用必须被 d2q_enabled() 派生的开关守卫"
