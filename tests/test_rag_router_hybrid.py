# -*- coding: utf-8 -*-

from __future__ import annotations

import json


def _candidate(source, operation, answer_object, score, role="answer_source"):
    return {
        "key": f"{source}.{operation}", "source": source, "operation": operation,
        "answer_object": answer_object, "source_role": role, "score": score,
        "semantic_score": score, "frame_bonus": 0.0, "description": "test candidate",
    }


def _semantic_payload(*routes, service="recall", decision="answer", clarification=""):
    return json.dumps({
        "decision": decision, "interaction_kind": "query",
        "service_mode": service, "action_request": None,
        "tasks": ([{
            "task_id": "t", "subquery": "用户问题",
            "answer_routes": list(routes), "context_routes": [],
            "filters": {}, "depends_on": [],
        }] if decision == "answer" else []),
        "capability_ids": [], "clarification": clarification,
    }, ensure_ascii=False)


def test_route_registry_covers_every_executable_route_and_valid_roles():
    from rag_query_plan import ROUTE_OPERATIONS
    from rag_route_registry import READ_SOURCE_REGISTRY, SOURCE_ROLES

    registered = {item.key for item in READ_SOURCE_REGISTRY}
    executable = {
        f"{source}.{operation}"
        for source, operations in ROUTE_OPERATIONS.items() for operation in operations
    }
    assert registered == executable
    assert all(item.source_role in SOURCE_ROLES for item in READ_SOURCE_REGISTRY)


def test_route_prototype_cache_can_be_prebuilt_without_personal_data(monkeypatch, tmp_path):
    import rag_route_registry

    monkeypatch.setattr(rag_route_registry, "_cache_path",
                        lambda _profile: tmp_path / "routes.json")
    monkeypatch.setattr(rag_route_registry, "_profile_key",
                        lambda: ("test", {"provider": "fake", "model": "fake", "dimensions": 2}))

    def embed(texts):
        return [[float(len(text) % 7), 1.0] for text in texts]

    first = rag_route_registry.prebuild_route_prototype_cache(embed)
    second = rag_route_registry.prebuild_route_prototype_cache(embed)
    assert first["cache"] == "miss" and second["cache"] == "hit"
    assert first["route_count"] == len(rag_route_registry.plannable_route_definitions())
    raw = (tmp_path / "routes.json").read_text(encoding="utf-8")
    assert "user_profile" not in raw and "applications.md" not in raw


def test_frozen_router_eval_set_has_planned_suite_sizes():
    from pathlib import Path
    from eval_rag_routes import _expanded_cases

    path = Path(__file__).parent / "fixtures" / "rag_router_heldout_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = _expanded_cases(data)
    assert len(cases) == 480
    counts = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts == {
        "semantic_role_contrast": 200,
        "operation_contrast": 100,
        "multi_turn_reference": 80,
        "composite_multi_source": 100,
    }


def test_obsolete_hybrid_setting_is_ignored_and_never_runs_prototypes(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "hybrid")
    called = []

    def ranker(*_args, **_kwargs):
        called.append("embedding")
        raise AssertionError("hard route must not call semantic ranker")

    plan = plan_query(
        "我投过哪几家公司？", semantic_ranker=ranker,
        llm_call=lambda _payload: called.append("llm") or _semantic_payload(
            "application_state.list_ever_applied"
        ),
    )
    assert called == ["llm"]
    assert plan.resolver_mode == "llm"
    assert {(r.source, r.operation) for r in plan.routes} == {
        ("application_state", "list_ever_applied")}
    assert plan.confidence_factors["llm_calls"] == 1
    assert plan.intent_frame.decision_reasons == ["structured_v4_contract_validated"]


def test_prototype_score_cannot_authorize_an_online_route(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "hybrid")
    def ranker(*_args, **_kwargs):
        raise AssertionError("prototype ranker is offline only")

    plan = plan_query(
        "知识库中的混合检索怎么实现", semantic_ranker=ranker,
        llm_call=lambda _payload: _semantic_payload(
            "reference_kb.search", service="explain"
        ),
    )
    assert plan.planner_mode == "llm"
    assert plan.resolver_mode == "llm"
    assert {(r.source, r.operation) for r in plan.routes} == {
        ("reference_kb", "search")}


def test_v3_model_requests_clarification_without_prototype_margins(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "hybrid")

    def ranker(*_args, **_kwargs):
        raise AssertionError("prototype ranker is offline only")

    plan = plan_query(
        "帮我看看这些经历", semantic_ranker=ranker,
        llm_call=lambda _payload: _semantic_payload(
            service="recall", decision="clarify",
            clarification="请说明是项目经历还是投递经历。",
        ),
    )
    assert plan.decision == "clarify"
    assert plan.routes == []
    assert "还是" in plan.clarification


def test_v3_planner_owns_complementary_source_selection(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "hybrid")
    payload = json.dumps({
        "decision": "answer", "interaction_kind": "query",
        "service_mode": "advise", "action_request": None,
        "tasks": [{"task_id": "t", "subquery": "投递经验和技术概念",
                   "answer_routes": ["application_experience.search", "reference_kb.search"],
                   "context_routes": [], "filters": {}, "depends_on": []}],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query(
        "结合我的投递经验，同时解释对应技术概念",
        semantic_ranker=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("prototype ranker is offline only")
        ),
        llm_call=lambda _payload: payload,
    )
    pairs = {(r.source, r.operation) for r in plan.routes}
    assert ("application_experience", "search") in pairs  # 明确要求，必须保留
    assert ("reference_kb", "search") in pairs
    assert ("profile_plan", "get_gaps") not in pairs
    assert plan.confidence_factors["llm_calls"] == 1


def test_llm_cannot_invent_storage_target_or_non_candidate_route(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "hybrid")

    def ranker(_question, _frame, **_kwargs):
        return ([
            _candidate("reference_kb", "search", "reference_knowledge", 0.60),
            _candidate("paper_kb", "search", "paper_knowledge", 0.58),
        ], {})

    def arbiter(_payload):
        return json.dumps({
            "decision": "answer",
            "routes": [
                {"source": "project_memory", "operation": "search",
                 "source_role": "answer_source"},
                {"source": "reference_kb", "operation": "search",
                 "source_role": "answer_source", "filters": {"path": "/tmp/private"}},
            ],
        })

    plan = plan_query("请同时综合看看", semantic_ranker=ranker, llm_call=arbiter)
    assert plan.resolver_mode == "fallback"
    assert all("/tmp/private" not in json.dumps(r.filters) for r in plan.routes)


def test_removed_shadow_alias_has_no_runtime_effect(monkeypatch, tmp_path):
    import rag_query_plan

    monkeypatch.setenv("RAG_ROUTER_MODE", "shadow")
    plan = rag_query_plan.plan_query(
        "我想看看个人成果",
        semantic_ranker=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("prototype ranker is offline only")
        ),
        llm_call=lambda _payload: _semantic_payload(
            "project_memory.list_catalog", service="recall"
        ),
    )
    assert plan.resolver_mode == "llm"
    assert {(r.source, r.operation) for r in plan.routes} == {
        ("project_memory", "list_catalog")
    }
    assert plan.intent_frame.decision_reasons == ["structured_v4_contract_validated"]


def test_clarification_plan_returns_sse_without_retrieval(monkeypatch):
    import rag_query_plan
    from rag_gate import gated_query_stream

    plan = rag_query_plan.QueryPlan(
        planner_mode="semantic", resolver_mode="semantic", decision="clarify",
        clarification="你要查个人项目还是投递记录？", routes=[],
        reason="答案中心冲突",
    )
    monkeypatch.setattr(rag_query_plan, "plan_query", lambda *_args, **_kwargs: plan)
    events = list(gated_query_stream("看看这些经历"))
    stage_events = [e for e in events if e["type"] == "stage"]
    payload_events = [e for e in events if e["type"] != "stage"]
    assert stage_events[-1]["stage"] == "clarification"
    assert payload_events[0]["type"] == "meta" and payload_events[0]["mode"] == "clarification"
    assert payload_events[0]["decision"] == "clarify"
    assert payload_events[1] == {"type": "delta", "text": "你要查个人项目还是投递记录？"}
    assert payload_events[-1]["type"] == "done"


def test_missing_personal_evidence_is_visible_and_not_replaced_by_reference(monkeypatch):
    from rag_multi_source import execute_plan, multi_source_messages
    from rag_query_plan import QueryPlan, RouteSpec

    monkeypatch.setenv("RAG_PARALLEL_ROUTES", "0")
    plan = QueryPlan(
        planner_mode="rule",
        routes=[
            RouteSpec("project_memory", "search", "我的项目", required=True),
            RouteSpec("reference_kb", "search", "技术解释"),
        ],
    )

    def retrieve(_query, _top_k, **kwargs):
        if kwargs.get("source_types") == {"project_context"}:
            return {"in_kb": False, "chunks": [], "docs": [], "metas": [], "sources": []}
        return {"in_kb": True, "chunks": ["通用教程内容"], "docs": ["通用教程内容"],
                "metas": [{"source": "guide.md", "source_type": "resource",
                           "owner_scope": "curated"}], "sources": ["guide.md"],
                "matched_by": "hybrid"}

    result = execute_plan(plan, "问题", 5, retrieve=retrieve)
    assert result.missing_personal_routes == ["project_memory"]
    system = multi_source_messages("问题", plan, result)[0]["content"]
    assert "必须明确写“未找到个人记录”" in system
