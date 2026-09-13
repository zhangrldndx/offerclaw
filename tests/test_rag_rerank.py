# -*- coding: utf-8 -*-
"""rag_rerank 单元测试：开关、降级、按分数重排（不依赖真实 reranker 模型）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_rerank as rr


def test_rerank_disabled_truncates(monkeypatch):
    """RAG_RERANK=0 → 不调模型，直接截断 top_k。"""
    monkeypatch.setenv("RAG_RERANK", "0")
    docs = ["a", "b", "c", "d"]
    metas = [{"source": s} for s in "abcd"]
    dists = [0.1, 0.2, 0.3, 0.4]
    rd, rm, rdi, rs = rr.rerank("q", docs, metas, dists, top_k=2)
    assert rd == ["a", "b"] and rdi == [0.1, 0.2] and rs == []


def test_rerank_empty_docs():
    assert rr.rerank("q", [], [], [], top_k=5) == ([], [], [], [])


def test_rerank_falls_back_when_model_unavailable(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setattr(rr, "_load_reranker", lambda: None)
    docs = ["a", "b", "c"]
    rd, _, _, rs = rr.rerank("q", docs, [{}, {}, {}], [0.1, 0.2, 0.3], top_k=2)
    assert rd == ["a", "b"] and rs == []   # 降级=截断，不报错


def test_rerank_reorders_by_score(monkeypatch):
    """交叉编码器把高分文档提前：原序 a,b,c → 分数 c>a>b → 重排 c,a。"""
    monkeypatch.setenv("RAG_RERANK", "1")

    class _FakeCE:
        def predict(self, pairs):
            # pairs 顺序 a,b,c → 给 c 最高分
            return [0.5, 0.1, 0.9]
    monkeypatch.setattr(rr, "_load_reranker", lambda: _FakeCE())
    docs = ["doc-a", "doc-b", "doc-c"]
    metas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
    dists = [0.20, 0.21, 0.22]
    rd, rm, rdi, rs = rr.rerank("q", docs, metas, dists, top_k=2)
    assert rd == ["doc-c", "doc-a"]          # 按分数降序
    assert rm[0]["source"] == "c"
    assert rdi == [0.22, 0.20]               # 原始距离跟着 doc 走
    assert rs == [0.9, 0.5]                   # rerank 分数


def test_rerank_enabled_env(monkeypatch):
    monkeypatch.setenv("RAG_RERANK", "0")
    assert rr.rerank_enabled() is False
    monkeypatch.setenv("RAG_RERANK", "1")
    assert rr.rerank_enabled() is True
