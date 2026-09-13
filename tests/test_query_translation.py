# -*- coding: utf-8 -*-
"""查询翻译通道:默认关 / 契约解析 / 缓存 / 专有名词保留 / fail-soft。全部打桩零 LLM。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_translate as T  # noqa: E402


def _fresh(monkeypatch, tmp_path):
    """每个用例一份独立缓存,互不污染。"""
    monkeypatch.setattr(T, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(T, "_MEM", {})
    monkeypatch.setattr(T, "_LOADED", False)


def test_default_off(monkeypatch):
    monkeypatch.delenv("RAG_QUERY_TRANSLATE", raising=False)
    assert T.translate_enabled() is False
    monkeypatch.setenv("RAG_QUERY_TRANSLATE", "1")
    assert T.translate_enabled() is True


def test_only_translates_chinese():
    assert T.needs_translation("在有限窗口内制造无限上下文的错觉") is True
    assert T.needs_translation("What is ReAct?") is False   # 已在"英问英"条件,翻了浪费


def test_parse_tolerates_code_fence_and_prose():
    raw = ('好的，结果如下：\n```json\n{"english_query": "infinite context illusion",'
           ' "keywords": ["paging"], "preserve_terms": []}\n```\n希望有帮助')
    got = T._parse(raw)
    assert got["english_query"] == "infinite context illusion"
    assert got["keywords"] == ["paging"]


def test_parse_rejects_contract_violation():
    assert T._parse("") is None
    assert T._parse("no json here") is None
    assert T._parse('{"keywords": ["x"]}') is None            # 缺核心字段 → 宁退回原 query
    assert T._parse('{"english_query": "  "}') is None        # 空串同样不可用


def test_preserve_terms_backfilled(monkeypatch, tmp_path):
    """query 里本来的英文专有名词必须出现在英文 query 里——否则丢掉最强的词法信号。"""
    _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(T, "_chat_raw", None, raising=False)
    import rag_gate
    monkeypatch.setattr(rag_gate, "_chat", lambda *a, **k: json.dumps(
        {"english_query": "paging mechanism of the memory system",
         "keywords": ["virtual context"], "preserve_terms": []}))
    got = T.translate_query("MemGPT 的分页机制是怎么设计的")
    assert "MemGPT" in got["english_query"] and "MemGPT" in got["preserve_terms"]


def test_cache_hit_avoids_second_llm_call(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    calls = []
    import rag_gate
    monkeypatch.setattr(rag_gate, "_chat", lambda *a, **k: calls.append(1) or json.dumps(
        {"english_query": "q", "keywords": [], "preserve_terms": []}))
    T.translate_query("上下文分页调度")
    T.translate_query("  上下文分页调度  ")       # 归一化后同 key → 不该再调一次
    assert len(calls) == 1
    assert os.path.exists(T.CACHE_PATH)           # 已落盘,跨进程复用


def test_failure_is_cached_to_avoid_retry_storm(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    calls = []
    import rag_gate
    monkeypatch.setattr(rag_gate, "_chat", lambda *a, **k: calls.append(1) or "")
    assert T.translate_query("无法翻译的问题") is None
    assert T.translate_query("无法翻译的问题") is None
    assert len(calls) == 1                        # 失败也记缓存,不反复烧钱


def test_fail_soft_when_llm_raises(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    import rag_gate

    def _boom(*a, **k):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(rag_gate, "_chat", _boom)
    assert T.translate_query("任意中文问题") is None   # 绝不把异常抛回检索主链路


def test_english_query_for_appends_keywords(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    import rag_gate
    monkeypatch.setattr(rag_gate, "_chat", lambda *a, **k: json.dumps(
        {"english_query": "infinite context illusion",
         "keywords": ["infinite context", "paging", "virtual memory"],
         "preserve_terms": []}))
    s = T.english_query_for("在有限窗口内制造无限上下文的错觉")
    assert s.startswith("infinite context illusion")
    assert "paging" in s and "virtual memory" in s
    assert s.count("infinite context") == 1        # 已在 english_query 里的不重复拼
