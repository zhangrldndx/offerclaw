"""Round 2 BM25 + RRF 融合单元测试（纯函数为主，不依赖真实库）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_bm25 import _tokenize, bm25_search, bm25_enabled, rrf_fuse


def test_tokenize_mixed_zh_en():
    toks = _tokenize("混合检索 Hybrid Search 稀疏向量 BM25")
    assert "hybrid" in toks and "search" in toks  # 英文小写化
    assert "bm25" in toks
    assert any("稀疏" in t for t in toks)          # 中文被 jieba 切出


def test_tokenize_empty_safe():
    assert _tokenize("") == []
    assert _tokenize(None) == []


def test_bm25_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("RAG_BM25", "0")
    assert not bm25_enabled()
    assert bm25_search("任意查询", 5) == []


def test_bm25_metadata_filter_is_applied_before_top_n(monkeypatch):
    import rag_bm25

    class FakeBM25:
        @staticmethod
        def get_scores(_tokens):
            # The globally strongest item is outside the requested batch; the
            # best allowed item must still survive a top_n=1 query.
            return [100.0, 9.0, 8.0]

    monkeypatch.setenv("RAG_BM25", "1")
    monkeypatch.setattr(rag_bm25, "_build_index", lambda _name=None: {
        "bm25": FakeBM25(),
        "docs": ["other batch", "right batch", "second right"],
        "metas": [
            {"source": "2026-06-19_x.md", "owner_scope": "curated"},
            {"source": "2026-06-03_x.md", "owner_scope": "curated"},
            {"source": "2026-06-03_y.md", "owner_scope": "curated"},
        ],
    })
    hits = bm25_search(
        "Transformer", 1,
        metadata_filters={"source": {"2026-06-03_x.md", "2026-06-03_y.md"}},
    )
    assert [hit[0] for hit in hits] == ["right batch"]


def test_rrf_both_paths_rank_first():
    """同一文档两路都召回 → RRF 相加 → 排第一。"""
    vec_docs = ["A", "B"]
    vec_metas = [{"source": "a"}, {"source": "b"}]
    vec_dists = [0.10, 0.20]
    bm25 = [("A", {"source": "a"}, 5.0), ("C", {"source": "c"}, 3.0)]
    docs, metas, dists = rrf_fuse(vec_docs, vec_metas, vec_dists, bm25,
                                  top_n=10, bm25_only_dist=0.9)
    assert docs[0] == "A"          # 双路命中，RRF 最高
    assert "C" in docs            # 仅 BM25 命中的也被救回候选池


def test_rrf_preserves_and_placeholders_dist():
    """向量文档保留真实距离；仅 BM25 文档用占位距离。"""
    docs, metas, dists = rrf_fuse(
        ["A"], [{"source": "a"}], [0.12],
        [("C", {"source": "c"}, 4.0)],
        top_n=10, bm25_only_dist=0.88,
    )
    assert dists[docs.index("A")] == 0.12   # 向量真实距离
    assert dists[docs.index("C")] == 0.88   # BM25-only 占位


def test_rrf_top_n_truncates():
    vec_docs = [f"D{i}" for i in range(30)]
    vec_metas = [{"source": str(i)} for i in range(30)]
    vec_dists = [0.1 + i * 0.01 for i in range(30)]
    docs, _m, _d = rrf_fuse(vec_docs, vec_metas, vec_dists, [], top_n=20)
    assert len(docs) == 20
    assert docs[0] == "D0"   # 向量 rank1 仍最前


def test_rrf_empty_bm25_keeps_vector_order():
    """BM25 空（降级）→ 融合退化为向量原序。"""
    docs, _m, _d = rrf_fuse(["X", "Y", "Z"], [{}, {}, {}], [0.1, 0.2, 0.3],
                            [], top_n=10)
    assert docs == ["X", "Y", "Z"]
