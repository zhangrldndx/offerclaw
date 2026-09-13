# -*- coding: utf-8 -*-
"""排名融合 / 保底槽位 / English False Winner 保护 / 可达性自检。全部打桩零模型。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_fusion as f  # noqa: E402
import rag_reachability as rc  # noqa: E402

EN = {"document_language": "en", "source": "paper.md"}
ZH = {"document_language": "zh", "source": "zh.md"}


def test_fusion_off_by_default(monkeypatch):
    monkeypatch.delenv("RAG_RANK_FUSION", raising=False)
    assert f.fusion_mode() == "off"
    d, m, k, s = f.fuse_prerank_rerank(["b", "a"], ["a", "b"], [ZH, EN], [0.1, 0.2], [0.9, 0.1])
    assert d == ["a", "b"] and s == [0.9, 0.1]        # 逐字节不变


def test_fusion_rrf_and_blend_reorder(monkeypatch):
    pre = ["x", "y"]                                   # 初排:x 第1
    post = ["y", "x"]                                  # 精排:y 第1
    monkeypatch.setenv("RAG_RANK_FUSION", "rrf")
    d, _m, _k, _s = f.fuse_prerank_rerank(pre, post, [ZH, EN], [0.1, 0.2], [0.9, 0.8])
    assert set(d) == {"x", "y"}                        # 两侧各赢一次 → 稳定并列,不崩
    monkeypatch.setenv("RAG_RANK_FUSION", "blend")
    monkeypatch.setenv("RAG_FUSION_ALPHA", "1.0")      # 全权重给精排 = 退化为纯精排
    d2, _m, _k, _s = f.fuse_prerank_rerank(pre, post, [ZH, EN], [0.1, 0.2], [0.9, 0.8])
    assert d2 == post


def test_fusion_unknown_mode_is_failsafe(monkeypatch):
    monkeypatch.setenv("RAG_RANK_FUSION", "没这个模式")
    d, _m, _k, _s = f.fuse_prerank_rerank(["b"], ["a", "b"], [ZH, EN], [0.1, 0.2], [0.9, 0.1])
    assert d == ["a", "b"]                             # 未知模式 → 不改行为


def test_reserve_slot_promotes_champion_without_touching_top1():
    """保底只保 Top3 的覆盖,**不动精排的 Top1 判断**——这是它与等权融合的本质区别。"""
    docs = ["a", "b", "c", "d", "en1"]
    metas = [ZH, ZH, ZH, ZH, EN]
    d, m, k, s = f.reserve_slot(docs, metas, [0.1] * 5, [0.9, 0.8, 0.7, 0.6, 0.5],
                                champion="en1", slot=3)
    assert d[0] == "a"                                 # Top1 不动
    assert d[2] == "en1"                               # 冠军进第 3 位
    assert d == ["a", "b", "en1", "c", "d"]            # 其余顺次后退,不丢候选


def test_reserve_slot_noop_when_already_in_front_or_absent():
    docs, metas = ["en1", "a"], [EN, ZH]
    assert f.reserve_slot(docs, metas, [0.1, 0.2], [0.9, 0.8], "en1", 3)[0] == docs
    assert f.reserve_slot(docs, metas, [0.1, 0.2], [0.9, 0.8], "不在池里", 3)[0] == docs
    assert f.reserve_slot(docs, metas, [0.1, 0.2], [0.9, 0.8], None, 3)[0] == docs


def test_english_false_winner_margin_is_asymmetric():
    """英文险胜中文 → 让位;英文大比分赢 → 保留(真论文题不受影响)。"""
    docs, metas, dists = ["en1", "zh1"], [EN, ZH], [0.5, 0.6]
    # 险胜(差 0.01 < margin 0.05)→ 中文上位
    d, _m, _k, _s = f.protect_top1_margin(docs, metas, dists, [0.90, 0.89], margin=0.05)
    assert d[0] == "zh1"
    # 大胜(差 0.60 ≥ margin)→ 英文保持 Top1
    d2, _m, _k, _s = f.protect_top1_margin(docs, metas, dists, [0.95, 0.35], margin=0.05)
    assert d2[0] == "en1"


def test_margin_off_by_default_and_ignores_chinese_top1(monkeypatch):
    monkeypatch.delenv("RAG_EN_TOP1_MARGIN", raising=False)
    docs, metas = ["en1", "zh1"], [EN, ZH]
    assert f.protect_top1_margin(docs, metas, [0.5, 0.6], [0.90, 0.89])[0] == docs
    # Top1 本就是中文 → 与本保护无关,不干预
    assert f.protect_top1_margin(["zh1", "en1"], [ZH, EN], [0.5, 0.6],
                                 [0.90, 0.89], margin=0.5)[0] == ["zh1", "en1"]


def test_reachability_flags_unreachable_english_corpus(monkeypatch):
    """英文语料存在但入口全关 → 报 PAPER_RETRIEVAL_UNREACHABLE。

    这条跨模块不变量是 2026-08-18 那次架构级回归的直接产物:主库纯化 + 路由默认关
    两个**各自正确**的决策叠加,让 194 块论文在生产上完全不可达,而没有任何测试会失败。
    """
    monkeypatch.setattr(rc, "_collection_count", lambda name: 194)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": False, "paper_route": False})
    r = rc.check_reachability(strict=False)
    assert r["ok"] is False and r["code"] == rc.PAPER_RETRIEVAL_UNREACHABLE
    assert "RAG_EN_QUOTA=1" in r["detail"]              # 报错要带可执行的修法


def test_reachability_ok_when_any_entrypoint_open_or_no_corpus(monkeypatch):
    monkeypatch.setattr(rc, "_collection_count", lambda name: 194)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": True, "paper_route": False})
    assert rc.check_reachability(strict=False)["ok"] is True
    monkeypatch.setattr(rc, "_collection_count", lambda name: 0)   # 没有英文语料
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": False, "paper_route": False})
    assert rc.check_reachability(strict=False)["ok"] is True


def test_reachability_strict_raises(monkeypatch):
    monkeypatch.setattr(rc, "_collection_count", lambda name: 194)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": False, "paper_route": False})
    try:
        rc.check_reachability(strict=True)
        assert False, "strict 模式应抛出"
    except RuntimeError as e:
        assert rc.PAPER_RETRIEVAL_UNREACHABLE in str(e)
