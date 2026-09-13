# -*- coding: utf-8 -*-
"""Round 10 精排语体桥单测:max 合分逻辑 + 键对齐 + 默认关接线(A4 同款纪律)。"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import rag_rerank
from rag_doc2query import doc_key


class _FakeModel:
    """确定性打分:按 pair 文本长度打分,便于断言 max 逻辑。"""
    def __init__(self, table):
        self.table = table  # {pair第二元素: 分数}
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.table.get(p[1], 0.1) for p in pairs]


def _run(monkeypatch, synth_map, table):
    monkeypatch.setattr(rag_rerank, "_load_reranker", lambda: _FakeModel(table))
    monkeypatch.setenv("RAG_RERANK", "1")
    docs = ["书面块A" * 20, "书面块B" * 20]
    metas = [{"source": "a.md"}, {"source": "b.md"}]
    return rag_rerank.rerank("口语问题", docs, metas, [0.5, 0.6], 2, synth_map=synth_map), docs, metas


def test_bridge_max_lifts_matched_doc(monkeypatch):
    docs = ["书面块A" * 20, "书面块B" * 20]
    metas = [{"source": "a.md"}, {"source": "b.md"}]
    # 直连分:A=0.2 B=0.5;A 的合成问题=0.9 → 桥后 A=0.9 应排第一
    table = {docs[0]: 0.2, docs[1]: 0.5, "A的口语问法": 0.9}
    smap = {doc_key(docs[0], metas[0]): "A的口语问法"}
    (rd, rm, rs, rscore), _, _ = _run(monkeypatch, smap, table)
    assert rd[0].startswith("书面块A") and rscore[0] == 0.9


def test_bridge_direct_higher_wins(monkeypatch):
    docs = ["书面块A" * 20, "书面块B" * 20]
    metas = [{"source": "a.md"}, {"source": "b.md"}]
    table = {docs[0]: 0.8, docs[1]: 0.5, "A的口语问法": 0.3}   # 直连分更高 → 保留直连
    smap = {doc_key(docs[0], metas[0]): "A的口语问法"}
    (rd, _, _, rscore), _, _ = _run(monkeypatch, smap, table)
    assert rd[0].startswith("书面块A") and rscore[0] == 0.8


def test_no_map_identical_to_before(monkeypatch):
    docs = ["书面块A" * 20, "书面块B" * 20]
    table = {docs[0]: 0.2, docs[1]: 0.5}
    (rd, _, _, rscore), _, _ = _run(monkeypatch, None, table)
    assert rd[0].startswith("书面块B") and rscore == [0.5, 0.2]   # 无桥 = 原行为


def test_scores_truncated_to_docs(monkeypatch):
    """附加 pair 的分数绝不能泄漏进返回的 rscore(曾是最容易写错的下标处)。"""
    docs = ["书面块A" * 20]
    metas = [{"source": "a.md"}]
    table = {docs[0]: 0.2, "Q": 0.9}
    monkeypatch.setattr(rag_rerank, "_load_reranker", lambda: _FakeModel(table))
    monkeypatch.setenv("RAG_RERANK", "1")   # conftest 全局关 rerank,此处需显式开
    rd, rm, rs, rscore = rag_rerank.rerank("q", docs, metas, [0.4], 5,
                                           synth_map={doc_key(docs[0], metas[0]): "Q"})
    assert len(rscore) == 1 and rscore[0] == 0.9


def test_gate_wiring_flag_guarded():
    src = open(os.path.join(BASE_DIR, "rag_gate.py"), encoding="utf-8").read()
    g = src.find('RAG_RERANK_BRIDGE')
    c = src.find("build_synth_map(emb")
    assert 0 < g < c, "语体桥必须被 RAG_RERANK_BRIDGE 守卫(默认关)"
