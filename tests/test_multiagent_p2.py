# -*- coding: utf-8 -*-
"""多-agent 升级 P2 单测:Critic 接进 CareerFlow(产-查分离)+ 语义腿 + 图接线。

纪律(docs/MULTI_AGENT_UPGRADE.md):默认 skip_llm 走骨架 → critic 透传零行为改变;
编造硬拦=代码(reject);LLM 语义腿只 flag(最多 needs_fix,不 reject)。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import career_flow
import resume_critic


# ---------- critic_node 接线 ----------

def test_critic_node_transparent_on_skeleton():
    # 默认 skip_llm → resume 出骨架(mode=skeleton) → critic 透传,不产 critic_report
    state = {"resume_skeleton": {"mode": "skeleton"}, "skip_llm": True}
    out = career_flow.critic_node(state)
    assert "critic_report" not in out


def test_critic_node_transparent_when_no_md():
    state = {"resume_skeleton": {"mode": "llm", "llm_md": ""}, "skip_llm": True}
    assert "critic_report" not in career_flow.critic_node(state)


def test_critic_node_rejects_fabrication_llm_mode():
    state = {"resume_skeleton": {"mode": "llm", "llm_md": "本系统支持自动投递并对接 LinkedIn"},
             "jd_text": "Python", "profile": {}, "skip_llm": True,
             "jd_struct": {"keywords": ["Python"]}}
    out = career_flow.critic_node(state)
    assert out["critic_report"]["verdict"] == "reject"
    # 未过 → 挂 requires_confirmation(不代改,守"写入需确认")
    assert any("Critic" in c.get("suggested_patch", "")
               for c in out.get("requires_confirmation", []))


def test_critic_node_pass_clean_llm_mode():
    state = {"resume_skeleton": {"mode": "llm", "llm_md": "我用 Python 和 RAG 搭了求职 Agent"},
             "profile": {}, "skip_llm": True, "jd_struct": {"keywords": ["Python", "RAG"]}}
    out = career_flow.critic_node(state)
    assert out["critic_report"]["verdict"] == "pass"
    assert not out.get("requires_confirmation")


# ---------- 语义腿(腿2,LLM,默认关) ----------

def test_semantic_leg_flags_drive_needs_fix(monkeypatch):
    import day1_api_starter
    import plan_gen
    monkeypatch.setattr(day1_api_starter, "get_llm_config", lambda: {"api_key": "x"})
    monkeypatch.setattr(plan_gen, "call_llm_plain",
                        lambda messages, api_key, max_tokens=800: "这句夸大了\n那句无据")
    r = resume_critic.critic_report("一段覆盖了 Python 的简历",
                                    jd_keywords=["Python"], profile={"技能": ["Python"]}, use_llm=True)
    assert r["semantic_flags"] and len(r["semantic_flags"]) == 2
    assert r["verdict"] == "needs_fix"          # 无编造硬拦,但 LLM 标了语义问题 → 建议改写


def test_semantic_leg_none_means_pass(monkeypatch):
    import day1_api_starter
    import plan_gen
    monkeypatch.setattr(day1_api_starter, "get_llm_config", lambda: {"api_key": "x"})
    monkeypatch.setattr(plan_gen, "call_llm_plain", lambda *a, **k: "NONE")
    r = resume_critic.critic_report("我用 Python 做了 RAG",
                                    jd_keywords=["Python", "RAG"], profile={}, use_llm=True)
    assert r["semantic_flags"] == [] and r["verdict"] == "pass"


def test_semantic_leg_no_key_degrades_to_none(monkeypatch):
    import day1_api_starter
    monkeypatch.setattr(day1_api_starter, "get_llm_config", lambda: {"api_key": None})
    r = resume_critic.critic_report("我用 Python 做了 RAG",
                                    jd_keywords=["Python", "RAG"], profile={}, use_llm=True)
    assert r["semantic_flags"] is None and r["verdict"] == "pass"


def test_semantic_leg_cannot_override_code_reject(monkeypatch):
    # LLM 说"没问题"(NONE),但代码腿抓到编造 → 仍 reject(硬拦交代码,LLM 不得推翻)
    import day1_api_starter
    import plan_gen
    monkeypatch.setattr(day1_api_starter, "get_llm_config", lambda: {"api_key": "x"})
    monkeypatch.setattr(plan_gen, "call_llm_plain", lambda *a, **k: "NONE")
    r = resume_critic.critic_report("本项目自动投递", jd_keywords=["x"], profile={}, use_llm=True)
    assert r["verdict"] == "reject"


# ---------- 图接线 ----------

def test_both_graphs_have_critic_node():
    for build in (career_flow.build_graph, career_flow.build_routed_graph):
        nodes = build().get_graph().nodes
        assert "critic" in nodes
