# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py — 主链路（match → plan → summary → rag）的离线测试

不调 LLM、不依赖网络；只测：
- 模块可导入
- match_job.run_match 返回结构正确
- plan_gen / summary_tool / rag_query 模块可调
- LangGraph 的 _msg_to_dict 等关键工具函数可用
- chroma_db 可被 rag_query 模块连上
"""
from __future__ import annotations
import json
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------- import smoke ----------

def test_import_match_job():
    import match_job  # noqa: F401


def test_import_plan_gen():
    import plan_gen  # noqa: F401


def test_import_summary_tool():
    import summary_tool  # noqa: F401


def test_import_rag_query():
    import rag_query  # noqa: F401


def test_import_rag_graph():
    import rag_graph  # noqa: F401


def test_import_pipeline():
    import pipeline  # noqa: F401


# ---------- match_job 结构测试 ----------

def test_match_job_returns_report_with_status():
    """对一份硬否决 JD，run_match 应返回带 status 字段的 report。"""
    import match_job
    profile_path = os.path.join(ROOT, "profiles", "p1_demo_ai.json")
    if not os.path.exists(profile_path):
        pytest.skip("profiles/p1_demo_ai.json missing")
    profile = json.loads(open(profile_path, encoding="utf-8").read())

    jd_java = "Java 高级开发\n精通 Spring / MyBatis\n北京"
    rep = match_job.run_match(profile, jd_java, jd_title="java-test")
    # MatchReport 是 dataclass / 普通对象，至少一个状态/结论字段
    has_status = (
        hasattr(rep, "status")
        or hasattr(rep, "verdict")
        or hasattr(rep, "conclusion")
        or isinstance(rep, dict)
    )
    assert has_status, f"run_match return type unexpected: {type(rep)}"


def test_match_job_three_personas():
    """3 个 persona × 1 JD，run_match 都能正常返回。"""
    import match_job
    pdir = os.path.join(ROOT, "profiles")
    persona_files = [f for f in os.listdir(pdir) if f.startswith("p") and f.endswith(".json")]
    if len(persona_files) < 2:
        pytest.skip("not enough persona files")
    jd = "AI 应用开发实习生\n上海\n本科及以上\n熟悉 Python / LLM / Prompt"
    for pf in persona_files:
        profile = json.loads(open(os.path.join(pdir, pf), encoding="utf-8").read())
        rep = match_job.run_match(profile, jd, jd_title=f"smoke-{pf}")
        assert rep is not None, f"{pf} returned None"


# ---------- LangGraph 工具函数 ----------

def test_msg_to_dict_handles_basemessage_like():
    """rag_graph._msg_to_dict 应能处理 BaseMessage-like 对象（可序列化）。"""
    import rag_graph
    fn = getattr(rag_graph, "_msg_to_dict", None)
    if fn is None:
        pytest.skip("_msg_to_dict not exposed")

    class FakeMsg:
        type = "human"
        content = "hello"

    out = fn(FakeMsg())
    json.dumps(out)  # 必须可 JSON 序列化
    assert isinstance(out, dict)


# ---------- ChromaDB 连接 ----------

def test_chroma_db_reachable():
    db = os.path.join(ROOT, "chroma_db")
    if not os.path.exists(db):
        pytest.skip("chroma_db not built; run python rag_ingest.py")
    import chromadb
    from rag_tools import get_collection_name
    client = chromadb.PersistentClient(path=db)
    col = client.get_collection(get_collection_name())
    assert col.count() > 0


# ---------- 数据契约一致性 ----------

def test_data_contract_files_present():
    """DATA_CONTRACT.md 列出的核心 System Layer 文件都得真存在。"""
    must = [
        "SOUL.md", "target_rules.md", "source_policy.md",
        "match_job.py", "plan_gen.py", "summary_tool.py",
        "rag_ingest.py", "rag_query.py", "rag_graph.py",
        "rag_api.py", "logging_utils.py", "eval_rag.py",
        "DATA_CONTRACT.md",
    ]
    missing = [f for f in must if not os.path.exists(os.path.join(ROOT, f))]
    assert not missing, f"DATA_CONTRACT 列出但缺失：{missing}"


def test_eval_set_schema():
    p = os.path.join(ROOT, "tests", "rag_eval_set.json")
    data = json.loads(open(p, encoding="utf-8").read())
    assert "items" in data
    assert len(data["items"]) >= 50, f"评估集应 >= 50 题，当前 {len(data['items'])}"
    cats = {it["category"] for it in data["items"]}
    assert {"fact", "explain", "cross_doc"} <= cats
    for it in data["items"]:
        assert "q" in it and "expected_sources" in it
        assert isinstance(it["expected_sources"], list) and len(it["expected_sources"]) >= 1
