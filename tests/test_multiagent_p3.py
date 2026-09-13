# -*- coding: utf-8 -*-
"""多-agent 升级 P3 单测:CRAG 恢复循环(默认关)+ JD 解析 LLM 富化(默认关)+ job_input 接线。

纪律(docs/MULTI_AGENT_UPGRADE.md):RAG_CRAG=0 逐字节等价(全量套件已证);CRAG 深度守卫防双改写;
LLM 富化绝不产硬门槛;jd_struct 不喂 match_job(三档保持纯规则)。
"""
import inspect
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import career_flow
import jd_parser
import match_job
import rag_gate


# ---------- ⑤ CRAG 恢复循环 ----------

def test_crag_recover_returns_first_in_kb(monkeypatch):
    calls = []

    def fake_rc(q, top_k=5, _crag_depth=0):
        calls.append((q, _crag_depth))
        return {"in_kb": True, "chunks": ["x"], "sources": ["s"]}

    monkeypatch.setattr(rag_gate, "_crag_query_rewrites", lambda q: ["改写A", "改写B"])
    monkeypatch.setattr(rag_gate, "_retrieve_and_classify", fake_rc)
    out = rag_gate._crag_recover("弱证据 query", 5)
    assert out and out["in_kb"]
    assert calls[0][1] == 1                 # 递归调用 depth=1(深度守卫防双改写)


def test_crag_recover_none_when_all_weak(monkeypatch):
    monkeypatch.setattr(rag_gate, "_crag_query_rewrites", lambda q: ["a", "b"])
    monkeypatch.setattr(rag_gate, "_retrieve_and_classify",
                        lambda q, top_k=5, _crag_depth=0: {"in_kb": False})
    assert rag_gate._crag_recover("q", 5) is None    # 都救不回 → None(将维持原拒答)


def test_crag_query_rewrites_returns_bounded_list():
    cands = rag_gate._crag_query_rewrites("大模型 应用 工程师 需要 什么 技能")
    assert isinstance(cands, list) and len(cands) <= 2


def test_crag_default_off_signature_backward_compatible():
    # _retrieve_and_classify 新增 _crag_depth 默认 0,旧调用 (question[, top_k]) 不受影响
    params = inspect.signature(rag_gate._retrieve_and_classify).parameters
    assert params["_crag_depth"].default == 0


# ---------- ① JD 解析 LLM 富化(默认关) ----------

def test_jd_llm_analysis_is_grounded_and_never_decides_application(tmp_path):
    jd = "岗位：后端\n岗位职责\n搭建 Kubernetes 服务\n任职要求\n熟悉 gRPC"
    output = json.dumps({
        "title": "后端", "company": "", "location": "", "job_type": "",
        "responsibilities": ["搭建 Kubernetes 服务"],
        "requirements": [{
            "text": "熟悉 gRPC", "kind": "skill", "modality": "required",
            "priority": 0.8, "alternatives": [], "evidence_spans": ["熟悉 gRPC"],
        }],
        "keywords": [{
            "canonical_name": "gRPC", "surface_forms": ["gRPC"],
            "category": "technology", "importance": 0.8,
            "requirement_indexes": [0], "evidence_spans": ["gRPC"],
        }], "warnings": [],
    }, ensure_ascii=False)
    analysis = jd_parser.analyze_jd(
        jd, mode="intelligent", model="fake", cache_dir=tmp_path,
        caller=lambda *_: output,
    )
    assert analysis.source == "llm"
    assert "gRPC" in analysis.keyword_names()
    assert analysis.responsibilities == ["搭建 Kubernetes 服务"]
    assert not hasattr(analysis, "application_decision")


def test_jd_llm_error_degrades_to_deterministic(tmp_path):
    analysis = jd_parser.analyze_jd(
        "岗位：后端", mode="intelligent", model="fake", cache_dir=tmp_path,
        caller=lambda *_: (_ for _ in ()).throw(RuntimeError("llm down")),
    )
    assert analysis.source == "deterministic"
    assert any("llm_fallback" in warning for warning in analysis.warnings)


# ---------- job_input 接线 ----------

def test_job_input_then_jd_analyze_stores_new_contract():
    state = {"jd_text": "岗位名称：大模型应用实习\n技术要求：Python、RAG、LangGraph（需 30 字符以上）",
             "skip_llm": True}
    out = career_flow.job_input_node(state)
    analyzed = career_flow.jd_analyze_node({**state, **out})
    assert "jd_analysis" in analyzed and analyzed["jd_analysis"]["source"] == "deterministic"
    assert "python" in [item["canonical_name"].lower()
                        for item in analyzed["jd_analysis"]["keywords"]]


def test_jd_analyze_reuses_only_matching_hash_and_schema():
    first_jd = "岗位职责\n负责 Python 服务开发\n任职要求\n熟悉 Python 与 FastAPI"
    second_jd = "岗位职责\n负责 Go 服务开发\n任职要求\n熟悉 Go 与 Kubernetes"
    existing = jd_parser.analyze_jd(
        first_jd, mode="deterministic").model_dump(mode="json")

    assert career_flow.jd_analyze_node({
        "jd_text": first_jd, "jd_analysis": existing, "skip_llm": True,
    }) == {}

    refreshed = career_flow.jd_analyze_node({
        "jd_text": second_jd, "jd_analysis": existing, "skip_llm": True,
        "match_report": {"status": "stale"},
        "resume_skeleton": {"keywords_hit": ["Python"]},
    })
    assert refreshed["jd_analysis"]["input_hash"] != existing["input_hash"]
    assert refreshed["jd_analysis"]["schema_version"] == jd_parser.JD_ANALYSIS_SCHEMA_VERSION
    assert refreshed["match_report"] == {}
    assert refreshed["resume_skeleton"] == {}


# ---------- §6.1 结构性保证:jd_struct 不喂 match_job ----------

def test_match_job_never_sees_jd_struct():
    # 匹配三档保持纯规则——run_match 签名里根本没有 jd_struct 参(LLM 决不碰硬门槛裁决)
    assert "jd_struct" not in inspect.signature(match_job.run_match).parameters
