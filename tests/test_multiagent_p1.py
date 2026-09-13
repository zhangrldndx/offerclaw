# -*- coding: utf-8 -*-
"""多-agent 升级 P1 单测:jd_parser fallback / resume_critic 三腿 / supervisor 串行编排。

纪律对照 docs/MULTI_AGENT_UPGRADE.md:①默认纯代码 fallback、④编造硬拦=代码、Supervisor 零 LLM。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import career_flow
import jd_parser
import resume_critic
import supervisor


# ---------- ① jd_parser ----------

_JD = "岗位名称：大模型应用开发实习生\n技术要求：Python、LangGraph、RAG、FastAPI\n学历：硕士"


def test_jd_deterministic_contract_and_no_hard_decision():
    analysis = jd_parser.analyze_jd(_JD, mode="deterministic")
    assert analysis.schema_version == jd_parser.JD_ANALYSIS_SCHEMA_VERSION
    kws = [k.lower() for k in analysis.keyword_names()]
    assert "python" in kws and "rag" in kws
    assert "大模型应用开发实习生" in analysis.title
    assert analysis.source == "deterministic"
    # 抽取结果不包含投递结论；硬淘汰仍只能由 match_job 决定。
    assert not hasattr(analysis, "application_decision")


def test_parse_jd_compat_adapter_uses_deterministic_analysis():
    assert jd_parser.parse_jd(_JD)["_source"] == "deterministic"


def test_parse_jd_use_llm_degrades_without_model():
    # 显式清空分析模型，验证不会误用缓存或触发真实网络。
    analysis = jd_parser.analyze_jd(_JD, mode="intelligent", model="")
    assert analysis.source == "deterministic"
    assert any("no_model" in warning for warning in analysis.warnings)


# ---------- ④ resume_critic ----------

def test_keyword_coverage():
    cov = resume_critic.keyword_coverage("我用 Python 和 RAG 做了检索", ["Python", "RAG", "Redis"])
    assert set(cov["hit"]) == {"Python", "RAG"} and cov["miss"] == ["Redis"]
    assert abs(cov["coverage"] - 2 / 3) < 1e-6


def test_fabrication_blacklist_flag():
    flags = resume_critic.fabrication_flags("本系统支持自动投递并对接了 LinkedIn")
    terms = {f["term"] for f in flags}
    assert "自动投递" in terms and "LinkedIn" in terms


def test_fabrication_stale_number_flag():
    # "118 chunks" 与 metrics.json 当前真值(3340)不符 → stale_number
    flags = resume_critic.fabrication_flags("知识库 118 chunks，pytest 37 通过")
    kinds = [(f.get("field"), f.get("found")) for f in flags if f["kind"] == "stale_number"]
    assert ("chunks", "118") in kinds


def test_current_numbers_not_flagged():
    # 用当前真值则不该被误判为 stale。
    # 2026-08-08 修正:原硬编码 3340,真值刷新(用户入库 +4 块 → 3344)后测试即腐烂;
    # 改为动态读 metrics.json——测试跟着单一事实源走,不再随真值刷新过期。
    import json
    import os
    _metrics = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "metrics.json")
    cur_chunks = json.load(open(_metrics, encoding="utf-8"))["current"]["chunks"]
    flags = resume_critic.fabrication_flags(f"知识库 {cur_chunks} chunks")
    assert not any(f["kind"] == "stale_number" for f in flags)


def test_bare_database_flagged_but_vector_db_ok():
    assert resume_critic._blacklist_flags("我用了数据库存储")           # 裸"数据库"→flag
    assert not resume_critic._blacklist_flags("我用了向量数据库 ChromaDB")  # 向量数据库合法


def test_critic_verdict_reject_on_fabrication():
    r = resume_critic.critic_report("支持自动投递", jd_keywords=["Python"], profile={})
    assert r["verdict"] == "reject" and r["semantic_flags"] is None


def test_critic_verdict_pass_on_clean_covered():
    r = resume_critic.critic_report(
        "我用 Python、RAG、LangGraph 搭建了求职 Agent",
        jd_keywords=["Python", "RAG", "LangGraph"], profile={})
    assert r["verdict"] == "pass" and not r["fabrication_flags"]


def test_critic_verdict_needs_fix_on_low_coverage():
    r = resume_critic.critic_report(
        "一段和 JD 完全无关的话",
        jd_keywords=["Python", "RAG", "LangGraph", "FastAPI", "MCP", "Redis"], profile={})
    assert r["verdict"] == "needs_fix"


# ---------- Supervisor ----------

def test_supervisor_ledger_has_all_five_subsystems():
    assert set(supervisor.SUPERVISOR) == {
        "A_三档路由", "B_预算_checkpoint", "C_停滞检测", "D_三层记忆", "E_节奏触发"}


def test_supervisor_serial_ranks_and_independent_budget(monkeypatch):
    received_budgets = []

    def fake_run(jd_text, *, jd_title, skip_llm, budget=None):
        received_budgets.append(budget)
        status = {"a": "当前适合投递", "b": "中长期可转向",
                  "c": "当前暂不建议投递"}.get(jd_text, "当前适合投递")
        return {"match_report": {"status": status}, "route_taken": status, "jd_title": jd_title}

    monkeypatch.setattr(career_flow, "run_career_flow_routed", fake_run)

    out = supervisor.run_supervisor(
        [{"jd_text": "c", "jd_title": "T3"},
         {"jd_text": "a", "jd_title": "T1"},
         {"jd_text": "b", "jd_title": "T2"}],
        budget={"max_steps": 5})

    assert out["n"] == 3
    # 适合优先排序:T1(适合) < T2(中长期) < T3(暂不建议)
    assert out["ranked_titles"] == ["T1", "T2", "T3"]
    assert set(out["buckets"]) == {"当前适合投递", "中长期可转向", "当前暂不建议投递"}
    # 每份 JD 独立预算副本(无跨 JD 串写)
    assert len(received_budgets) == 3
    assert received_budgets[0] is not received_budgets[1]


def test_supervisor_parallel_equals_serial(monkeypatch):
    # [P4·E3 correctness] 并行终态与串行终态逐字段相等(并行只提速不改结果)
    def fake_run(jd_text, *, jd_title, skip_llm, budget=None):
        return {"match_report": {"status": "当前适合投递"}, "route_taken": "r", "jd_title": jd_title}

    monkeypatch.setattr(career_flow, "run_career_flow_routed", fake_run)
    jds = [{"jd_text": f"jd{i}", "jd_title": f"T{i}"} for i in range(4)]
    seq = supervisor.run_supervisor(jds)
    par = supervisor.run_supervisor(jds, parallel=True)
    assert [(r["jd_title"], r["status"]) for r in seq["runs"]] == \
           [(r["jd_title"], r["status"]) for r in par["runs"]]
    assert par["parallel"] is True and seq["parallel"] is False


def test_supervisor_string_jds_and_titles():
    def fake_run(jd_text, *, jd_title, skip_llm, budget=None):
        return {"match_report": {"status": "当前适合投递"}, "route_taken": "", "jd_title": jd_title}

    import unittest.mock as mock
    with mock.patch.object(career_flow, "run_career_flow_routed", fake_run):
        out = supervisor.run_supervisor(["JD 文本一", "JD 文本二"])
    assert out["n"] == 2 and out["runs"][0]["jd_title"] == "JD#1"


# ---------- CareerState channel 声明 ----------

def test_careerstate_has_new_channels():
    ann = career_flow.CareerState.__annotations__
    assert "jd_struct" in ann and "plan_md" in ann and "critic_report" in ann
