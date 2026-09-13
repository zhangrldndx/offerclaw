# -*- coding: utf-8 -*-
"""P1 论文域回退路由:零劫持 / 双证据门 / fail-soft / 返回形状。全部打桩零模型。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_paper_route as pr  # noqa: E402


def test_zero_hijack_when_main_strong():
    assert pr.maybe_paper_route("任意问题", main_in_kb=True) is None   # 主库强证据永不触发


def test_default_off_and_opt_in(monkeypatch):
    """默认关(测正才采纳:比分门下净增益 0);显式开启才生效。"""
    monkeypatch.delenv("RAG_PAPER_ROUTE", raising=False)
    assert pr.paper_route_enabled() is False
    assert pr.maybe_paper_route("论文问题", main_in_kb=False) is None
    monkeypatch.setenv("RAG_PAPER_ROUTE", "1")
    assert pr.paper_route_enabled() is True


def test_gate_calibrated_rule(monkeypatch):
    assert pr._paper_gate(best_dist=0.40, rerank_top=0.80) is True    # 精排强
    assert pr._paper_gate(best_dist=0.30, rerank_top=0.00) is True    # 距离兜底
    assert pr._paper_gate(best_dist=0.40, rerank_top=0.01) is False   # 双弱=拒(近域负例区)


def test_gate_does_not_outrank_main_evidence():
    """2026-08-18 回归修复:主库弱但更强时不夺权(52 题真尺子测出 -7.7pp 的那条路)。"""
    assert pr._paper_gate(0.40, 0.80, main_rerank_top=0.95) is False   # 主库更强 → 让位
    assert pr._paper_gate(0.40, 0.80, main_rerank_top=0.10) is True    # 论文更强 → 接管
    assert pr._paper_gate(0.30, 0.00, main_rerank_top=0.02) is False   # 距离够但精排输 → 让位


def test_route_returns_gated_shape(monkeypatch):
    monkeypatch.setenv("RAG_PAPER_ROUTE", "1")
    monkeypatch.setattr(pr, "_query_papers", lambda q, n: {
        "docs": ["paper chunk"], "metas": [{"source": "paper_react.md"}], "dists": [0.20]})
    import rag_rerank
    monkeypatch.setattr(rag_rerank, "rerank",
                        lambda q, d, m, di, k: (d, m, di, [0.9]))
    out = pr.maybe_paper_route("哪篇工作把推理和行动交替", main_in_kb=False)
    assert out and out["in_kb"] and out["paper_route"]
    assert out["sources"] == ["paper_react.md"] and out["matched_by"] == "paper_rerank"
    for k in ("chunks", "best", "rerank_top", "docs", "metas", "dists"):
        assert k in out


def test_fail_soft_on_missing_collection(monkeypatch):
    monkeypatch.setenv("RAG_PAPER_ROUTE", "1")
    monkeypatch.setattr(pr, "_query_papers", lambda q, n: None)      # 集合/模型缺失
    assert pr.maybe_paper_route("论文问题", main_in_kb=False) is None


def test_missing_collection_short_circuits_before_model_load(monkeypatch):
    """集合缺失时不得走到 e5 加载(2026-08-10 实测:死路径每查询白付 1.1GB 加载费)。"""
    import rag_tools

    def _boom(*a, **k):
        raise AssertionError("集合缺失仍尝试加载模型")

    monkeypatch.setattr(rag_tools, "_load_local_model", _boom)
    monkeypatch.setattr(pr, "_get_paper_collection", lambda: None)
    assert pr._query_papers("论文问题", n=5) is None                  # fail-soft 保留
    assert pr.maybe_paper_route("论文问题", main_in_kb=False) is None


def test_collection_probe_cached(monkeypatch):
    """探测结论(含缺失)进程内只查一次。"""
    import sys as _sys
    import types
    probes = []

    def _client(path):
        probes.append(path)
        raise RuntimeError("no such collection")

    monkeypatch.setitem(_sys.modules, "chromadb",
                        types.SimpleNamespace(PersistentClient=_client))
    monkeypatch.setattr(pr, "_PAPER_COL_CACHE", {})
    assert pr._get_paper_collection() is None
    assert pr._get_paper_collection() is None
    assert len(probes) == 1


def test_explicit_paper_quality_is_scoped_and_gated(monkeypatch, tmp_path):
    """增强只由显式论文入口选择；分数未过 0.80 时不得 grounded。"""
    monkeypatch.setenv("RAG_PAPER_PROFILE", "paper_quality")
    monkeypatch.setenv("RAG_EN_GATE_MIN", "0.80")
    monkeypatch.setenv("RAG_RERANK_EN_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(pr, "_query_papers_bge", lambda q, n: {
        "docs": ["english paper"],
        "metas": [{"source": "paper_react.md", "source_type": "paper"}],
        "dists": [0.8],
    })
    import rag_rerank
    calls = []

    def _rerank(q, d, m, di, k, **kwargs):
        calls.append(kwargs.get("en_bridge_idx"))
        return d, m, di, [0.81]

    monkeypatch.setattr(rag_rerank, "rerank", _rerank)
    out = pr.retrieve_papers_explicit("哪篇论文提出 ReAct")
    assert out and out["retrieval_profile"] == "paper_quality"
    assert out["matched_by"] == "paper_quality_en_gate"
    assert calls == [[0]]


def test_paper_quality_below_gate_does_not_bypass_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_PAPER_PROFILE", "paper_quality")
    monkeypatch.setenv("RAG_RERANK_EN_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(pr, "_query_papers_bge", lambda q, n: {
        "docs": ["weak paper"], "metas": [{"source": "weak.md"}], "dists": [0.9],
    })
    import rag_rerank
    monkeypatch.setattr(rag_rerank, "rerank",
                        lambda q, d, m, di, k, **kw: (d, m, di, [0.79]))
    monkeypatch.setattr(pr, "_retrieve_paper_evidence",
                        lambda q, top_k=3, main_rerank_top=None: {"baseline": True})
    assert pr.retrieve_papers_explicit("论文问题") is None


def test_unknown_paper_profile_is_fail_closed(monkeypatch):
    monkeypatch.setenv("RAG_PAPER_PROFILE", "surprise")
    assert pr.paper_profile() == "baseline"
