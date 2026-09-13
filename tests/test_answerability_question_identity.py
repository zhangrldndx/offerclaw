# -*- coding: utf-8 -*-
"""Original-question identity contracts for answerability shadow judging."""

from __future__ import annotations

import json


def test_reference_route_retrieves_with_subquery_but_carries_original_question(
        monkeypatch):
    from rag_multi_source import execute_plan
    from rag_query_plan import QueryPlan, RouteSpec

    monkeypatch.setenv("RAG_PARALLEL_ROUTES", "0")
    original = "向量数据库与传统数据库属于相互替代关系吗？"
    retrieval = "向量数据库与传统数据库的职责关系"
    plan = QueryPlan(
        planner_mode="rule",
        routes=[RouteSpec("reference_kb", "search", retrieval)],
    )
    seen = {}

    def retrieve(query, top_k, **kwargs):
        seen.update({"query": query, "top_k": top_k, "kwargs": kwargs})
        return {
            "in_kb": False, "chunks": [], "docs": [], "metas": [],
            "sources": [],
        }

    execute_plan(plan, original, 5, retrieve=retrieve)

    assert seen["query"] == retrieval
    assert seen["kwargs"]["_answerability_shadow_route"] == "reference_kb"
    assert seen["kwargs"]["_answerability_shadow_question"] == original


def test_online_shadow_observes_original_and_retrieval_questions(monkeypatch):
    import rag_gate
    import rag_shadow_answerability
    from tests.test_retrieval_trace import _install_retrieval_stubs

    _install_retrieval_stubs(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_CRAG", "0")
    captured = {}

    def observe(question, trace, **kwargs):
        captured.update({"question": question, "trace": trace, **kwargs})

    monkeypatch.setattr(rag_shadow_answerability, "observe", observe)
    original = "Planner的职责是执行子任务而非设计行动路径吗？"
    retrieval = "Planner 与 Executor 的职责边界"

    from traffic_origin import reset_traffic_origin, set_traffic_origin

    token = set_traffic_origin("agent_generated")
    try:
        rag_gate._retrieve_and_classify(
            retrieval, top_k=3, source_types={"doc"},
            allow_paper_route=False,
            _answerability_shadow_route="reference_kb",
            _answerability_shadow_question=original,
        )
    finally:
        reset_traffic_origin(token)

    assert captured["question"] == original
    assert captured["retrieval_question"] == retrieval
    assert captured["route"] == "reference_kb"
    assert captured["traffic_origin"] == "agent_generated"


def test_traffic_judge_uses_original_for_premise_and_subquery_for_scope(
        monkeypatch):
    import rag_answerability
    from scripts.run_answerability_shadow_traffic_agent import judge_selected_rows

    original = "向量数据库与传统数据库属于相互替代关系吗？"
    retrieval = "向量数据库与传统数据库的职责关系"
    document = "两者承担不同职责，形成互补，而非相互替代。"
    calls = {"cache": [], "grade": []}

    def cache_key(model, question, chunk, retrieval_question=None):
        calls["cache"].append((model, question, chunk, retrieval_question))
        return "fixture-key"

    def grade(question, chunk, **kwargs):
        calls["grade"].append((question, chunk, kwargs))
        return {
            "grade": 3, "question_form": "polar",
            "direct_answer": "proposition_false",
            "premise_status": "refuted", "relation": "contradicts",
            "reason": "证据明确说明互补",
        }

    monkeypatch.setattr(rag_answerability, "cache_key", cache_key)
    monkeypatch.setattr(rag_answerability, "grade", grade)
    monkeypatch.setattr(rag_answerability, "_load_cache", lambda: {})
    monkeypatch.setattr(rag_answerability, "flush_cache", lambda: None)
    rows = [{
        "judge_selected": True,
        "question": original,
        "retrieval_question": retrieval,
        "candidates_private": [{
            "rank": 1, "chunk_id": "chunk-1", "rerank_score": 0.99,
            "document": document,
        }],
        "verdicts": [],
    }]

    result = judge_selected_rows(
        rows, depth=1, workers=1, model="fixture-judge",
    )

    assert result["judge_jobs"] == 1
    assert calls["cache"] == [
        ("fixture-judge", original, document, retrieval),
    ]
    assert calls["grade"][0][0:2] == (original, document)
    assert calls["grade"][0][2]["retrieval_question"] == retrieval
    assert rows[0]["verdicts"][0]["action"] == "correct_premise"


def _package_fixture():
    question = "本地原始问题"
    retrieval = "中性检索式"
    reference = {
        "traffic_id": "traffic-1",
        "traffic_origin": "agent_generated",
        "stratum": "main",
        "producer": "deterministic",
        "producer_model": "",
        "transform": "exact_replay",
        "seed_question_sha256": "1" * 64,
        "question": question,
        "retrieval_question": retrieval,
        "original_question_sha256": "2" * 64,
        "retrieval_question_sha256": "3" * 64,
        "reference_capture": 1,
        "features": {"rerank_top": 0.99, "rerank_margin": 0.02},
        "gate_decision": True,
        "effective_hit": True,
        "routes": [{"source": "reference_kb", "operation": "search"}],
        "retrieval_profile": "baseline",
        "index_fingerprint": "fp",
        "reranker_requested": "r",
        "reranker_actual": "r",
        "reranker_status": "ok",
        "planning_ms": 1.0,
        "execution_ms": 2.0,
        "judge_selected": True,
        "sampling_probability": 1.0,
        "sampling_weight": 1.0,
        "sampling_reason": "all_gate_accepts",
        "candidates_private": [{
            "rank": 1, "chunk_id": "chunk-1", "source": "source.md",
            "source_type": "resource", "heading_path": "section",
            "rerank_score": 0.99, "document": "evidence",
        }],
        "verdicts": [{
            "rank": 1,
            "verdict": {
                "grade": 3, "question_form": "polar",
                "direct_answer": "proposition_false",
                "premise_status": "refuted", "relation": "contradicts",
                "reason": "refuted",
            },
            "action": "correct_premise", "cache_hit": False,
            "latency_ms": 3.0, "chunk_id": "chunk-1",
            "rerank_score": 0.99,
        }],
    }
    route = {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "traffic_id": "traffic-1", "question_sha256": "2" * 64,
        "traffic_origin": "agent_generated", "stratum": "main",
        "producer": "deterministic", "transform": "exact_replay",
        "seed_question_sha256": "1" * 64, "decision": "answer",
        "resolver_mode": "llm", "planner_engine": "gpt",
        "planner_version": "v1", "route_model": "m",
        "routes": [{"source": "reference_kb", "operation": "search"}],
        "has_reference_kb": True, "planning_ms": 1.0,
    }
    return route, reference


def test_public_calibration_declares_original_question_role():
    from scripts.run_answerability_shadow_traffic_agent import (
        JUDGE_QUESTION_ROLE, package_rows,
    )

    route, reference = _package_fixture()
    _, public, _, _ = package_rows(
        [route], [reference], judge_depth=1, judge_model="fixture-judge",
    )

    assert JUDGE_QUESTION_ROLE == "original_user_question"
    assert public[0]["judge_question_role"] == "original_user_question"
    assert public[0]["verdicts"][0]["question_form"] == "polar"
    assert public[0]["verdicts"][0]["direct_answer"] == "proposition_false"
    assert public[0]["verdicts"][0]["premise_status"] == "refuted"
    encoded = json.dumps(public, ensure_ascii=False)
    assert reference["question"] not in encoded
    assert reference["retrieval_question"] not in encoded


def test_manifest_declares_original_question_role(monkeypatch, tmp_path):
    import scripts.run_answerability_shadow_traffic_agent as agent

    monkeypatch.setattr(
        agent, "load_reference_probe_topics",
        lambda: ([{"topic_id": "a" * 64}], {}),
    )
    monkeypatch.setattr(
        agent, "build_reference_probe_traffic",
        lambda *args, **kwargs: [{"question": "fixture"}],
    )
    monkeypatch.setattr(
        agent, "run_route_faithful_retrieval",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(agent, "sample_for_judging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        agent, "judge_selected_rows",
        lambda *args, **kwargs: {
            "judge_jobs": 0, "judge_cache_hits": 0, "judge_unavailable": 0,
        },
    )
    summary = {
        "generated_queries": 1,
        "reference_kb_queries": 0,
        "traffic_origin": "agent_generated_reference_probe",
    }
    monkeypatch.setattr(
        agent, "package_rows",
        lambda *args, **kwargs: ([], [], [], dict(summary)),
    )
    monkeypatch.setattr(agent, "validate_public_exports", lambda *args: None)
    monkeypatch.setattr(agent, "_git_state", lambda: {})
    output = tmp_path / "package"

    assert agent.main([
        "--output-dir", str(output),
        "--count", "1",
        "--producer", "reference_probe",
        "--producer-model", "fixture-producer",
        "--judge-model", "fixture-judge",
    ]) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_question_role"] == "original_user_question"
