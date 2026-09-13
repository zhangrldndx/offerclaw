# -*- coding: utf-8 -*-
"""语言配额召回:默认关 / 加性池不扰动主池 / 惰性 env / 语言判定 / 池观测。全部打桩零模型。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_lang as lang  # noqa: E402
import rag_quota as q  # noqa: E402


def test_default_off_and_opt_in(monkeypatch):
    """默认关——测正才采纳(同 HyDE/doc2query/CRAG/paper_route 的家法)。"""
    monkeypatch.delenv("RAG_EN_QUOTA", raising=False)
    assert q.quota_enabled() is False
    monkeypatch.setenv("RAG_EN_QUOTA", "1")
    assert q.quota_enabled() is True


def test_quota_k_and_collection_are_lazy(monkeypatch):
    """惰性读 env:模块级冻结曾让 Phase 4 的换集合 A/B 完全不生效(跑出 6% vs 96% 的离谱值)。"""
    monkeypatch.delenv("RAG_QUOTA_COLLECTION", raising=False)
    assert q.quota_collection() == "kb_paper_bge_v1"
    monkeypatch.setenv("RAG_QUOTA_COLLECTION", "kb_other_v9")
    assert q.quota_collection() == "kb_other_v9"      # 运行时改必须生效
    monkeypatch.setenv("RAG_EN_QUOTA_K", "7")
    assert q.quota_k() == 7
    monkeypatch.setenv("RAG_EN_QUOTA_K", "非数字")
    assert q.quota_k() == 10                           # 坏值回默认,不炸


def test_append_quota_pool_never_disturbs_main_pool():
    """**配额的核心不变量**:主池成员与顺序逐位不变,英文只追加在尾部。

    2026-08-18 实测教训:改成"融合后截断到固定池深"会让英文挤掉中文,腾出的位置放进
    基线里本进不了精排的另一个中文块,把正确答案顶下去——held-out 52 因此 -3.8pp,
    三题回退里两题的凶手根本不是英文。截断会把"增加"悄悄变回"顶替"。
    """
    docs = [f"zh{i}" for i in range(5)]
    metas = [{"source": f"z{i}.md"} for i in range(5)]
    dists = [0.1 * i for i in range(5)]
    quota = [("en1", {"source": "p.md"}, 0.9), ("en2", {"source": "p.md"}, 0.95)]
    od, om, ok, prov = q.append_quota_pool(docs, metas, dists, quota)
    assert od[:5] == docs and om[:5] == metas and ok[:5] == dists   # 主池逐位不变
    assert od[5:] == ["en1", "en2"]                                 # 英文在尾部
    assert prov["en1"] == ["dense_en_quota"] and prov["zh0"] == ["dense_global"]


def test_append_quota_pool_dedups_without_reordering():
    """同一块既在主池又被配额召回 → 不重复,且保持主池位置(只在血缘上记两个通道)。"""
    docs, metas, dists = ["a", "b"], [{"source": "x"}, {"source": "y"}], [0.2, 0.3]
    od, om, ok, prov = q.append_quota_pool(docs, metas, dists,
                                           [("a", {"source": "x"}, 0.9)])
    assert od == ["a", "b"] and ok == [0.2, 0.3]
    assert prov["a"] == ["dense_global", "dense_en_quota"]


def test_append_quota_pool_fallback_dist_for_lexical():
    """词法通道没有真实向量距离 → 用调用方给的占位距离(与主链路 bm25_only_dist 同义)。"""
    od, _om, ok, _p = q.append_quota_pool(
        ["a"], [{"source": "x"}], [0.2], [], [("lex", {"source": "p"}, None)],
        fallback_dist=0.73)
    assert od == ["a", "lex"] and ok == [0.2, 0.73]


def test_rerank_pool_is_additive_not_truncated(monkeypatch):
    monkeypatch.delenv("RAG_RERANK_POOL", raising=False)
    assert q.rerank_pool_size(20, 5) == 25          # 相加,不是截到某个固定值
    monkeypatch.setenv("RAG_RERANK_POOL", "24")
    assert q.rerank_pool_size(20, 5) == 24          # 显式覆盖(池深消融用)


def test_cap_per_source_off_by_default(monkeypatch):
    monkeypatch.delenv("RAG_MAX_CHUNKS_PER_SOURCE", raising=False)
    docs = ["a", "b", "c"]
    metas = [{"source": "s.md"}] * 3
    assert q.cap_per_source(docs, metas, [0.1, 0.2, 0.3])[0] == docs   # 默认不启用
    monkeypatch.setenv("RAG_MAX_CHUNKS_PER_SOURCE", "2")
    assert q.cap_per_source(docs, metas, [0.1, 0.2, 0.3])[0] == ["a", "b"]


def test_rrf_multi_gives_each_channel_top1_equal_weight():
    """配额必须是独立通道:各通道第 1 名同权,英文才不会因距离偏大被融合截断丢掉。"""
    docs, _m, _d, prov = q.rrf_multi(
        {"dense_global": [("zh", {}, 0.2)], "dense_en_quota": [("en", {}, 0.9)]},
        top_n=2)
    assert set(docs) == {"zh", "en"}
    assert prov["en"] == ["dense_en_quota"]


def test_pool_stats_counts_english_and_target():
    st = q.pool_stats([{"document_language": "zh"}, {"document_language": "en"}],
                      [0.2, 0.9], {"a": ["dense_global"]}, ["a", "b"])
    assert st["english_candidate_count"] == 1 and st["first_english_rank"] == 2
    assert st["sources"] == ["", ""]


def test_detect_language_bands():
    assert lang.detect_language("This is a purely English technical passage about memory.") == "en"
    assert lang.detect_language("这是一段完全中文的技术说明，用来描述检索链路的行为表现。") == "zh"
    assert lang.detect_language("x" * 5) == "unknown"          # 样本太小不参与配额
    assert lang.is_quota_lang("en") and lang.is_quota_lang("mixed")
    assert not lang.is_quota_lang("zh")


def test_quota_candidates_fail_soft_without_collection(monkeypatch):
    """集合缺失 → 空列表(配额通道静默退化),绝不阻断主链路。"""
    monkeypatch.setattr(q, "_get_quota_collection", lambda: None)
    assert q.quota_candidates([[0.0] * 8], k=5) == []
