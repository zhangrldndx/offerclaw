# -*- coding: utf-8 -*-
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_structured_runtime():
    import structured_llm

    structured_llm._reset_structured_runtime_for_tests()
    yield
    structured_llm._reset_structured_runtime_for_tests()


def _v2_payload(*, route: str = "reference_kb.search",
                service: str = "explain") -> str:
    return json.dumps({
        "decision": "answer",
        "service_mode": service,
        "tasks": [{
            "task_id": "t1",
            "subquery": "用户问题",
            "answer_routes": [route],
            "context_routes": [],
            "filters": {},
            "depends_on": [],
        }],
        "capability_ids": [],
        "clarification": "",
    }, ensure_ascii=False)


def test_v4_schema_separates_speech_act_and_turn_relation():
    from semantic_query_planner import SEMANTIC_QUERY_SCHEMA_VERSION, SemanticQueryPlan

    schema = json.dumps(SemanticQueryPlan.model_json_schema(), ensure_ascii=False)
    properties = SemanticQueryPlan.model_json_schema()["properties"]
    assert "personal_scope" not in properties
    assert all(value not in schema for value in (
        '"answer_object"', '"operation"', '"source_roles"', '"route_keys"',
    ))
    assert all(value in schema for value in (
        '"answer_routes"', '"context_routes"', '"depends_on"',
        '"interaction_kind"', '"turn_relation"', '"action_request"', '"field_changes"',
    ))
    assert SEMANTIC_QUERY_SCHEMA_VERSION == "semantic-query-plan-v4"
    assert "correction" not in properties["interaction_kind"]["enum"]
    assert set(properties["turn_relation"]["enum"]) == {
        "standalone", "continuation", "correction",
    }


def test_v3_correction_output_is_translated_without_restoring_old_routing():
    from semantic_query_planner import SemanticQueryPlan

    payload = json.loads(_v2_payload())
    payload["interaction_kind"] = "correction"
    parsed = SemanticQueryPlan.model_validate(payload)

    assert parsed.interaction_kind == "query"
    assert parsed.turn_relation == "correction"


def test_model_cannot_author_relation_target_turn_id():
    from semantic_query_planner import SemanticQueryPlan

    payload = json.loads(_v2_payload())
    payload["turn_relation"] = "correction"
    payload["relation_target_turn_id"] = "forged_by_model"

    with pytest.raises(ValueError, match="relation_target_turn_id"):
        SemanticQueryPlan.model_validate(payload)


def test_action_schema_repair_receives_original_request(monkeypatch):
    from rag_query_plan import plan_query

    calls = []
    incomplete = json.loads(_v2_payload(route="product_help.guide"))
    incomplete.update({
        "interaction_kind": "command", "service_mode": "guide",
        "action_request": None,
        "capability_ids": ["application.update_status"],
    })
    repaired = {
        **incomplete,
        "action_request": {
            "verb": "update", "target_type": "application",
            "field_changes": [{"field": "status_code", "value_code": "applied"}],
            "target_refs": [], "commitment": "requested",
        },
    }

    def caller(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(incomplete, ensure_ascii=False)
        assert "original_request" in prompt
        assert "帮我把这条投递状态改成已投递" in prompt
        return json.dumps(repaired, ensure_ascii=False)

    plan = plan_query("帮我把这条投递状态改成已投递", llm_call=caller)

    assert len(calls) == 2
    assert plan.decision == "answer"
    assert plan.intent_frame.interaction_kind == "command"
    assert plan.intent_frame.action_request["field_changes"] == [
        {"field": "status_code", "value_code": "applied"}
    ]


def test_planner_prompt_exposes_daily_attachment_vs_formal_kb_boundary():
    from semantic_query_planner import _static_system_prompt

    prompt = _static_system_prompt()
    assert "每日执行附件的保存/检索边界选reflection" in prompt
    assert "主动导入正式资料库才选knowledge" in prompt
    assert "PDF/图片只作为本地附件引用" in prompt


def test_v2_derives_scope_object_operation_and_answer_role(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "我的项目有哪些？",
        llm_call=lambda _prompt: _v2_payload(
            route="project_memory.list_catalog", service="recall",
        ),
    )
    assert plan.intent_frame.personal_scope == "current"
    assert plan.intent_frame.answer_object == "project"
    assert plan.intent_frame.operations == ["list"]
    assert plan.routes[0].source_role == "answer_source"


def test_hidden_recent_log_route_remains_executable_but_not_plannable(monkeypatch):
    import rag_query_plan
    from rag_route_registry import get_route_definition, plannable_route_definitions

    definition = get_route_definition("profile_plan", "get_recent_log")
    assert definition is not None and definition.planner_visible is False
    assert "get_recent_log" in rag_query_plan.ROUTE_OPERATIONS["profile_plan"]
    assert definition not in plannable_route_definitions()

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = rag_query_plan.plan_query(
        "最近学习了什么？",
        llm_call=lambda _prompt: _v2_payload(
            route="profile_plan.get_recent_log", service="recall",
        ),
        semantic_ranker=lambda *_args, **_kwargs: ([], {"status": "empty"}),
    )
    assert plan.decision == "clarify"
    assert "route_not_plannable" in plan.fallback_reason


def test_recall_cannot_use_general_knowledge_as_personal_fact(monkeypatch):
    import rag_query_plan

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = rag_query_plan.plan_query(
        "我以前做过哪些项目？",
        llm_call=lambda _prompt: _v2_payload(
            route="reference_kb.search", service="recall",
        ),
        semantic_ranker=lambda *_args, **_kwargs: ([], {"status": "empty"}),
    )
    assert plan.decision == "clarify"
    assert "recall_requires_personal_source" in plan.fallback_reason


def test_required_entity_uses_registry_producer_dependency(monkeypatch):
    import rag_query_plan

    payload = {
        "decision": "answer", "service_mode": "advise",
        "tasks": [
            {"task_id": "app", "subquery": "华为投递",
             "answer_routes": ["application_state.list_current"],
             "filters": {"company": "华为"}, "depends_on": []},
            {"task_id": "fit", "subquery": "推荐项目",
             "answer_routes": ["project_memory.rank_for_application"],
             "depends_on": ["app"]},
        ],
    }
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = rag_query_plan.plan_query(
        "给华为投递推荐项目",
        llm_call=lambda _prompt: json.dumps(payload, ensure_ascii=False),
        semantic_ranker=lambda *_args, **_kwargs: ([], {"status": "empty"}),
    )
    assert plan.decision == "clarify"
    assert "missing_required_entity" in plan.fallback_reason

    payload["tasks"].insert(1, {
        "task_id": "jd", "subquery": "读取绑定JD",
        "answer_routes": ["application_jd.get_bound_jd"],
        "depends_on": ["app"],
    })
    payload["tasks"][-1]["depends_on"] = ["app", "jd"]
    valid = rag_query_plan.plan_query(
        "给华为投递推荐项目并参考绑定JD",
        llm_call=lambda _prompt: json.dumps(payload, ensure_ascii=False),
    )
    assert valid.decision == "answer"
    assert {route.source for route in valid.routes} == {
        "application_state", "application_jd", "project_memory",
    }


def test_prompt_has_stable_compact_prefix_and_dynamic_user_only():
    from semantic_query_planner import _COMPACT_SCHEMA_INSTRUCTIONS, _messages

    first = _messages("问题甲", ["上下文甲"])
    second = _messages("问题乙", ["上下文乙"])
    assert first[0] == second[0]
    assert "问题甲" not in first[0]["content"]
    assert "上下文甲" not in first[0]["content"]
    assert json.loads(first[1]["content"]) == {
        "question": "问题甲", "context": ["上下文甲"],
    }
    total_chars = sum(len(item["content"]) for item in first) + len(
        _COMPACT_SCHEMA_INSTRUCTIONS
    )
    # 为避免“每日附件本地引用”与“正式知识库导入”误路由，静态前缀保留
    # 一条能力边界；整体仍远低于 2200-token 预算。
    # 2026-08-31 +47:"本系统工程事实"类归属规则(UI 实测 4/4 库内事实题
    # 被个人记录路截胡,见 UI_JOURNEY_FINDINGS 发现 1)。棘轮惯例:上限
    # 钉在当前用量,再加字须再来此有意识地动钉。
    assert total_chars <= 6500
    assert "profile_plan.get_recent_log" not in first[0]["content"]


def test_planner_uses_compact_output_budget_and_memory_lru(monkeypatch):
    import semantic_query_planner as planner
    from semantic_query_planner import SemanticQueryPlan
    from structured_llm import StructuredCallMeta

    planner.clear_semantic_plan_cache()
    monkeypatch.setenv("RAG_ROUTE_MODEL", "gpt-test")
    calls = []

    def fake_call(model_type, messages, **kwargs):
        calls.append(kwargs)
        return SemanticQueryPlan.model_validate(json.loads(_v2_payload())), StructuredCallMeta(
            calls=1, model="gpt-test",
        )

    monkeypatch.setattr(planner, "call_structured", fake_call)
    first = planner.plan_semantically("解释缓存测试", [])
    second = planner.plan_semantically("解释缓存测试", [])
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 600
    assert calls[0]["lane"] == "online"
    assert calls[0]["schema_instructions"] == planner._COMPACT_SCHEMA_INSTRUCTIONS
    assert first.cache_hit is False and second.cache_hit is True
    assert all("解释缓存测试" not in key for key in planner._PLAN_CACHE)


def test_prototype_ranker_refuses_request_time_cold_build(monkeypatch):
    import rag_route_registry as registry

    registry.clear_route_prototype_runtime()
    monkeypatch.setattr(
        registry, "_profile_key",
        lambda: ("cold-test", {"provider": "fake", "model": "fake", "dimensions": 2}),
    )
    embed_calls = []
    with pytest.raises(RuntimeError, match="route_prototypes_not_ready"):
        registry.rank_route_candidates(
            "解释 BM25", object(),
            embedder=lambda texts: embed_calls.append(texts) or [[1.0, 0.0]],
        )
    assert embed_calls == []


def test_obsolete_shadow_setting_cannot_restore_the_background_router(monkeypatch):
    import rag_query_plan as planner

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent_shadow")
    calls = []
    plan = planner.plan_query(
        "请解释一个新问题",
        llm_call=lambda prompt: calls.append(prompt) or _v2_payload(),
    )

    assert len(calls) == 1
    assert plan.resolver_mode == "llm"
    assert plan.planner_engine == "structured_llm"
    assert all("shadow" not in reason for reason in plan.intent_frame.decision_reasons)
