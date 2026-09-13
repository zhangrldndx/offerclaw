"""Round 6 证据型门控 _evidence_gate 单元测试（纯函数）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_gate import _evidence_gate


def test_vector_reject_always_false():
    # 距离/词法初判不相关 → 直接 False，不看 reranker
    assert _evidence_gate(False, 0.99) is False
    assert _evidence_gate(False, None) is False


def test_rerank_none_falls_back_to_distance():
    # rerank 关闭（rerank_top=None）→ 退化为纯距离门控（兼容测试默认）
    assert _evidence_gate(True, None) is True


def test_rerank_veto(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_GATE_MIN", "0.85")
    assert _evidence_gate(True, 0.95) is True    # reranker 认可相关
    assert _evidence_gate(True, 0.644) is False  # reranker 否决（Docker 类：距离近但不相关）


def test_env_threshold_override(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_GATE_MIN", "0.5")
    assert _evidence_gate(True, 0.644) is True    # 阈值降到 0.5，0.644 通过


def test_boundary(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_GATE_MIN", "0.85")
    assert _evidence_gate(True, 0.85) is True     # 恰等于阈值 → 通过
    assert _evidence_gate(True, 0.849) is False


# ── Round 6.1 反向边缘救援（修 LoRA 漏判）─────────────────────

def test_rescue_edge_best_high_rerank():
    """距离稍超 strong（0.758）但 reranker 极高（0.99）→ 救回（「LoRA 是什么」case）。"""
    assert _evidence_gate(False, 0.99, best=0.758, strong=0.73) is True


def test_rescue_not_when_too_far():
    """best 0.85 > 救援上限 0.80 → 距离太远不救（防误救「今天天气」best 0.97 类）。"""
    assert _evidence_gate(False, 0.99, best=0.85, strong=0.73) is False


def test_rescue_not_when_rerank_not_high():
    """reranker 0.77 < 0.95 → 不够相关不救（防误救 Vue3 这类边缘负样本）。"""
    assert _evidence_gate(False, 0.77, best=0.758, strong=0.73) is False
