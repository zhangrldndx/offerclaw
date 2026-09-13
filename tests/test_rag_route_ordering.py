# -*- coding: utf-8 -*-
"""回归测试：audience 路由必须在 rerank 完整精排池上运行，再统一截 top_k。

Bug（2026-07-04 修复）：rag_gate._retrieve_and_classify 原先先 rerank 到 top_k、
再做 apply_audience_routing——路由只能在已截断的 top_k 内重排，若目标 career 文件
被 rerank 挤出 top_k 就永久丢失（ca03「算法岗怎么转」目标文件在完整召回池 rank #1-2，
却被提前截掉 → rank 0）。修复：rerank 整个 recall 池、先路由、最后截 top_k。

本文件两层守护：
1. 结构不变式（无模型/离线可跑）：monkeypatch rerank + apply_audience_routing，
   断言路由拿到的候选池 > top_k（即"先路由后截断"的顺序没被改回去）。
2. 端到端（需 reranker 模型 + 索引，缺失自动 skip）：ca03 命中 rank 1。
"""

import os

import pytest


def test_routing_sees_full_pool_before_truncation(monkeypatch):
    """结构不变式：apply_audience_routing 收到的候选数必须 > top_k。

    这直接钉死 bug——若有人把 rerank 改回截断到 top_k，路由拿到的池就 == top_k，
    本测试失败。用 monkeypatch 绕开 chroma/模型，纯验证调用顺序。
    """
    import rag_gate

    TOP_K = 5
    POOL = 20  # RECALL_N 量级

    # 桩：chroma 查询返回 POOL 个候选
    class _FakeCol:
        def query(self, **kw):
            n = kw.get("n_results", POOL)
            return {
                "documents": [[f"doc{i}" for i in range(n)]],
                "metadatas": [[{"source": f"src{i}.md"} for i in range(n)]],
                "distances": [[0.1 + i * 0.01 for i in range(n)]],
            }

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_collection(self, *a, **k):
            return _FakeCol()

    import chromadb

    monkeypatch.setattr(chromadb, "PersistentClient", _FakeClient)
    monkeypatch.setattr(rag_gate, "get_collection_name", lambda: "x", raising=False)
    # 让 rerank/bm25/hyde 变成恒等直通，但 rerank 保留"传入多少 top_k 就返回多少"的语义
    monkeypatch.setenv("RAG_RERANK", "1")
    import rag_rerank

    monkeypatch.setattr(rag_rerank, "rerank_enabled", lambda: True)
    monkeypatch.setattr(
        rag_rerank,
        "rerank",
        # **kw 吸收后续新增的可选参数(synth_map / alt_query / alt_idx …)——本测试钉的是
        # "路由拿到的池必须大于 top_k"这条调用顺序不变式,不该被 rerank 签名扩展绑住。
        lambda q, docs, metas, dists, top_k, **kw: (
            docs[:top_k],
            metas[:top_k],
            dists[:top_k],
            [0.9] * min(len(docs), top_k),
        ),
    )
    import rag_bm25

    monkeypatch.setattr(rag_bm25, "bm25_enabled", lambda: False)
    import rag_hyde

    monkeypatch.setattr(rag_hyde, "hyde_expand", lambda q, **_: q)
    monkeypatch.setattr(
        rag_gate, "get_embeddings_batch", lambda xs: [[0.0] * 8 for _ in xs], raising=False
    )

    seen_pool_sizes = []
    import rag_route

    orig = rag_route.apply_audience_routing

    def _spy(question, docs, metas, dists):
        seen_pool_sizes.append(len(docs))
        return orig(question, docs, metas, dists)

    monkeypatch.setattr(rag_route, "apply_audience_routing", _spy)

    rag_gate._retrieve_and_classify("算法岗怎么转大模型应用开发", top_k=TOP_K)

    assert seen_pool_sizes, "apply_audience_routing 未被调用"
    assert seen_pool_sizes[0] > TOP_K, (
        f"路由只拿到 {seen_pool_sizes[0]} 个候选（<= top_k={TOP_K}）——"
        "rerank 又在路由前截断了，ca03 类 bug 会复发"
    )


@pytest.mark.skipif(
    os.environ.get("RAG_RERANK", "1") == "0",
    reason="需真实 reranker 模型 + 索引（RAG_RERANK=0 时跳过，如 CI）",
)
def test_ca03_hits_rank_1_end_to_end():
    """端到端：'算法岗怎么转' 的目标 algorithm_transition_path 命中 rank 1。"""
    try:
        from rag_gate import _retrieve_and_classify
    except Exception as e:  # pragma: no cover
        pytest.skip(f"检索链不可用：{e}")
    try:
        g = _retrieve_and_classify("算法岗怎么转大模型应用开发", top_k=5)
    except Exception as e:
        pytest.skip(f"索引/模型缺失：{e}")
    srcs = [(m or {}).get("source", "").lower() for m in g.get("metas", [])]
    rank = next((i + 1 for i, s in enumerate(srcs) if "algorithm_transition" in s), 0)
    assert rank == 1, f"ca03 应命中 rank 1，实际 rank={rank}（srcs={srcs}）"
