# -*- coding: utf-8 -*-
"""CareerFlow 阶段 1/2：正确性与 LangGraph state 合并契约。"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import career_flow


JD = (
    "岗位名称：AI Agent 应用开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n专业要求：计算机/AI\n经验要求：实习\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)


def _no_memory(monkeypatch):
    monkeypatch.setattr(
        career_flow, "_learn_from_flow",
        lambda _: {"ok": True, "skipped": True, "node": "memory"})


def test_status_code_is_stable_and_unknown_does_not_fall_into_stretch():
    assert career_flow._match_status_code({"status": "当前适合投递"}) == "suitable"
    assert career_flow._match_status_code({"status": "当前不适合投递"}) == "unknown"
    assert career_flow._route_after_gap(
        {"match_report": {"status": "信息不足，建议补充后再判断"}}) == "error"


def test_short_jd_stops_before_parser(monkeypatch):
    import jd_parser

    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("short JD must not reach parser")

    monkeypatch.setattr(jd_parser, "parse_jd", should_not_run)
    state = {"jd_text": "招人", "skip_llm": True}
    before = copy.deepcopy(state)
    patch = career_flow.job_input_node(state)
    assert state == before
    assert patch["jd_valid"] is False
    assert called is False


def test_latin_keyword_boundaries_avoid_substring_false_positive():
    from jd_parser import analyze_jd

    assert "rag" not in {x.lower() for x in analyze_jd(
        "object storage platform", mode="deterministic").keyword_names()}
    assert "agent" not in {x.lower() for x in analyze_jd(
        "agentic workflow", mode="deterministic").keyword_names()}
    found = {x.lower() for x in analyze_jd(
        "任职要求\nBuild a RAG agent with Python", mode="deterministic").keyword_names()}
    assert {"rag", "agent"}.issubset(found)


def test_application_proposal_has_stable_id_and_replay_is_idempotent():
    state = {
        "jd_title": "AI 应用工程师",
        "_run_date": "2026-08-22",
        "match_report": {
            "status": "当前适合投递",
            "status_code": "suitable",
            "direction": "AI 应用",
        },
    }
    first = career_flow.application_suggest_node(state)
    proposal = first["requires_confirmation"][0]
    assert proposal["id"].startswith("proposal_")
    assert proposal["confirm_required"] is True

    merged = career_flow._merge_patch_for_snapshot(state, first)
    assert career_flow.application_suggest_node(merged) == {}
    assert len(career_flow._merge_confirmation_records([proposal], [proposal])) == 1


def test_critic_uses_input_hash_and_does_not_repeat(monkeypatch):
    import resume_critic

    calls = 0

    def fake_report(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "verdict": "needs_fix",
            "fabrication_flags": [],
            "keyword_coverage": {"coverage": 0.2},
        }

    monkeypatch.setattr(resume_critic, "critic_report", fake_report)
    state = {
        "resume_skeleton": {"mode": "llm", "llm_md": "Python 项目"},
        "jd_text": "Python RAG",
        "profile": {"技能": ["Python"]},
        "jd_struct": {"keywords": ["Python", "RAG"]},
        "skip_llm": True,
    }
    first = career_flow.critic_node(state)
    merged = career_flow._merge_patch_for_snapshot(state, first)
    second = career_flow.critic_node(merged)
    assert calls == 1
    assert second == {}
    assert first["critic_report"]["_input_hash"].startswith("critic_input_")


def test_unknown_match_result_routes_to_visible_error(monkeypatch):
    _no_memory(monkeypatch)
    import match_job

    report = SimpleNamespace(
        conclusion="信息不足，建议补充后再判断",
        direction="",
        gap_list={},
        suggestions=[],
        requirement_analysis={},
    )
    monkeypatch.setattr(match_job, "run_match", lambda *args, **kwargs: report)
    monkeypatch.setattr(match_job, "format_report", lambda _: "insufficient")
    out = career_flow.run_career_flow_routed(JD, jd_title="unknown")
    assert out["match_report"]["status_code"] == "unknown"
    assert out["route_taken"] == "error:stop"
    assert "plan" not in {entry["node"] for entry in out["trace"]}


def test_routed_flow_records_full_route_history(monkeypatch):
    _no_memory(monkeypatch)
    out = career_flow.run_career_flow_routed(JD, jd_title="route history")
    assert out["match_report"]["status_code"] in {
        "suitable", "stretch", "not_recommended"}
    assert out["route_history"]
    assert all({"router", "code", "label", "ts"} <= set(item)
               for item in out["route_history"])


def test_checkpoint_failure_is_observable_in_guard(monkeypatch):
    monkeypatch.setattr(
        career_flow, "_checkpoint",
        lambda *args, **kwargs: {
            "ok": False, "skipped": False, "node": "ok", "error": "disk full"})

    def ok_node(state):
        return {"value": 1}

    patch = career_flow.node_guard(ok_node)({"_run_id": "run_1"})
    assert patch["checkpoint_status"]["ok"] is False
    assert any(item["node"] == "checkpoint" for item in patch["errors"])


def test_memory_failure_is_visible_in_final_state(monkeypatch):
    monkeypatch.setattr(
        career_flow, "_learn_from_flow",
        lambda _: {"ok": False, "skipped": False,
                   "node": "memory", "error": "permission denied"})
    out = career_flow._finalize_memory({"trace": [], "errors": []})
    assert out["memory_status"]["ok"] is False
    assert any(item["node"] == "memory" for item in out["errors"])
