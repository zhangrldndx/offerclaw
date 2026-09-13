# -*- coding: utf-8 -*-
"""分工精排(英文模型桥):默认关 / sigmoid 同尺度 / 只碰英文候选。零真模型。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_rerank as rr  # noqa: E402


class _Base:
    max_seq_length = 512

    def predict(self, pairs):
        return [0.9, 0.2][: len(pairs)]      # 中文候选 0.9,英文候选 0.2(跨语惩罚形状)


class _Mini:
    max_seq_length = 512

    def predict(self, pairs):
        return [2.0] * len(pairs)            # logit 2.0 → sigmoid ≈ 0.881


def test_en_bridge_default_off(monkeypatch):
    """不设 RAG_RERANK_EN_ONNX_DIR → 行为与改造前逐字节等价(en_bridge_idx 被忽略)。"""
    monkeypatch.delenv("RAG_RERANK_EN_ONNX_DIR", raising=False)
    monkeypatch.setattr(rr, "_load_reranker", lambda model_name=None: _Base())
    monkeypatch.setattr(rr, "rerank_enabled", lambda: True)
    docs = ["中文块", "english chunk"]
    metas = [{"source": "zh.md"}, {"source": "p.md", "source_type": "paper"}]
    d, _m, _k, s = rr.rerank("q", docs, metas, [0.2, 0.9], 2, en_bridge_idx=[1])
    assert d[0] == "中文块" and s == [0.9, 0.2]


def test_en_bridge_lifts_english_via_sigmoid(monkeypatch):
    """桥开启:英文候选分 = max(base, sigmoid(mini logit));中文候选一个字节不动。"""
    monkeypatch.setenv("RAG_RERANK_EN_ONNX_DIR", "/fake/dir")
    monkeypatch.setattr(rr, "_load_reranker", lambda model_name=None: _Base())
    monkeypatch.setattr(rr, "rerank_enabled", lambda: True)
    rr._CE_CACHE[("en_bridge", "/fake/dir")] = _Mini()   # 注入假模型,绕过真实加载
    try:
        docs = ["中文块", "english chunk"]
        metas = [{"source": "zh.md"}, {"source": "p.md", "source_type": "paper"}]
        d, _m, _k, s = rr.rerank("q", docs, metas, [0.2, 0.9], 2, en_bridge_idx=[1])
        assert d[0] == "中文块"                      # 0.9 仍胜 0.881(中文不被夺位)
        assert abs(s[1] - 0.8808) < 0.001            # 英文分被抬到 sigmoid(2.0)
        assert s[0] == 0.9                           # 中文分逐字节不动
    finally:
        rr._CE_CACHE.pop(("en_bridge", "/fake/dir"), None)


def test_en_bridge_never_lowers_scores(monkeypatch):
    """max 语义保底:mini 打分低于 base 时英文分不降(坏模型不至于伤害现状)。"""
    monkeypatch.setenv("RAG_RERANK_EN_ONNX_DIR", "/fake/dir")

    class _Weak:
        max_seq_length = 512

        def predict(self, pairs):
            return [-5.0] * len(pairs)               # sigmoid ≈ 0.007

    monkeypatch.setattr(rr, "_load_reranker", lambda model_name=None: _Base())
    monkeypatch.setattr(rr, "rerank_enabled", lambda: True)
    rr._CE_CACHE[("en_bridge", "/fake/dir")] = _Weak()
    try:
        docs = ["中文块", "english chunk"]
        metas = [{"source": "zh.md"}, {"source": "p.md", "source_type": "paper"}]
        _d, _m, _k, s = rr.rerank("q", docs, metas, [0.2, 0.9], 2, en_bridge_idx=[1])
        assert s[1] == 0.2                           # max(0.2, 0.007) = 原分
    finally:
        rr._CE_CACHE.pop(("en_bridge", "/fake/dir"), None)


def test_en_evidence_gate_semantics(monkeypatch):
    """英文证据门:默认关;开启后只在「主门拒 + Top1 是英文 + 分数≥τ」时接受,
    且证据按名次取前 3(英文候选距离超 rescue,按距离过滤会得到空 chunks)。"""
    import rag_gate as rg

    # 用源码断言钉住三个语义(比整链 mock 稳):τ 来自 RAG_EN_GATE_MIN、
    # 只认英文 Top1、en_evidence 分支按名次取证据
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rag_gate.py"), encoding="utf-8").read()
    assert 'os.environ.get("RAG_EN_GATE_MIN"' in src
    gate_pos = src.find('_en_gate_raw')
    assert 0 < src.find('rerank_top >= _en_tau', gate_pos)
    assert 0 < src.find('"en_evidence"', gate_pos)
    # 接受分支在 in_kb 块内按名次取前 3,不走距离 cutoff。
    # 2026-08-27:口语证据门加入后与英文证据门共用这一分支(两条路进来的候选距离
    # 都按构造超过 rescue 阈值),所以这里认的是「_en_accept 参与的那个分支」。
    en_branch = src.find('if _en_accept or _rr_accept', gate_pos)
    assert 0 < en_branch < src.find('list(zip(docs, metas))[:3]')
    # 共用分支后仍必须按来源分辨 matched_by,不能把口语放行记成 en_evidence
    # 钉映射本身而不是钉整行:第三条通道(答案含量门)加进来时,
    # 「_en_accept -> en_evidence」和「兜底 -> rerank_confidence」必须原样成立。
    assert '"en_evidence" if _en_accept' in src
    assert src.rstrip().count('else "rerank_confidence")') == 1
    # 默认关:_en_gate_raw 为空则整段短路
    assert "(not in_kb) and _en_gate_raw" in src


def test_profiles_carry_bridge_and_gate():
    """两个生产 Profile 都带分工桥;仅 paper_quality 开英文证据门(τ=0.80)。"""
    import rag_profiles as rp
    dm = rp.profile_env("default_mixed")
    pq = rp.profile_env("paper_quality")
    assert "RAG_RERANK_EN_ONNX_DIR" in dm and "RAG_RERANK_EN_ONNX_DIR" in pq
    assert "RAG_EN_GATE_MIN" not in dm            # 默认路径宁拒不编
    assert pq["RAG_EN_GATE_MIN"] == "0.80"


def test_en_bridge_expands_home_path(monkeypatch):
    """真实 `.env.local` 可写 `~/...`，模型加载边界必须统一展开。"""
    import rag_rerank_onnx

    seen = []

    class _Bridge:
        def __init__(self, path):
            seen.append(path)

        def predict(self, pairs):
            return [2.0 for _ in pairs]

    monkeypatch.setenv("RAG_RERANK_EN_ONNX_DIR", "~/offerclaw-en-bridge")
    monkeypatch.setattr(rag_rerank_onnx, "OnnxCrossEncoder", _Bridge)
    monkeypatch.setattr(rr, "_load_reranker", lambda model_name=None: _Base())
    monkeypatch.setattr(rr, "rerank_enabled", lambda: True)
    monkeypatch.setattr(rr, "_CE_CACHE", {})
    rr.rerank("q", ["paper"], [{}], [0.9], 1, en_bridge_idx=[0])
    assert seen == [os.path.expanduser("~/offerclaw-en-bridge")]
