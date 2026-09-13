# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "top_ask_service_modes_v1.json"


def test_service_mode_frozen_set_meets_contract():
    from eval_rag_routes import evaluate

    report = evaluate(FIXTURE, mode="A")
    assert report["count"] == 50
    assert report["macro_f1"] == 1.0
    assert report["service_mode_accuracy"] == 1.0
    assert report["answer_object_accuracy"] == 1.0
    assert report["operation_accuracy"] == 1.0
    assert report["failures"] == []


def test_three_registries_have_distinct_read_only_responsibilities():
    from action_capabilities import ACTION_CAPABILITIES
    from product_help import CARD_CAPABILITY_REGISTRY
    from rag_route_registry import READ_SOURCE_REGISTRY, SERVICE_REGISTRY

    assert set(SERVICE_REGISTRY) == {"guide", "recall", "explain", "advise", "diagnose"}
    assert all("allow_general_fallback" in item for item in SERVICE_REGISTRY.values())
    assert SERVICE_REGISTRY["explain"]["allow_general_fallback"] is True
    assert all(not SERVICE_REGISTRY[mode]["allow_general_fallback"]
               for mode in ("guide", "recall", "advise", "diagnose"))

    route_keys = {item.key for item in READ_SOURCE_REGISTRY}
    assert "product_help.guide" in route_keys
    assert "system_diagnostics.inspect" in route_keys
    assert all(item.source_role for item in READ_SOURCE_REGISTRY)

    required = {"card_id", "ui_entry", "available", "supported_inputs",
                "required_inputs", "limitations", "steps", "boundary"}
    assert CARD_CAPABILITY_REGISTRY
    assert all(required <= set(item) for item in CARD_CAPABILITY_REGISTRY.values())
    assert ACTION_CAPABILITIES
    assert all(item.top_chat_policy == "guide_only" for item in ACTION_CAPABILITIES)
    assert all(item.capability_id and item.target_type and item.verb
               for item in ACTION_CAPABILITIES)


def test_guide_and_diagnose_are_locked_local_routes_without_retrieval():
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query

    calls = []

    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Guide/Diagnose must not call retrieval")

    guide_q = (
        "使用华为 JD，按企业知识库问答系统模板，改写 OfferClaw 项目，"
        "保留 4 条职责。"
    )
    guide_plan = rule_plan_query(guide_q)
    guide = execute_plan(guide_plan, guide_q, 5, retrieve)
    assert calls == []
    assert guide_plan.intent_frame.requested_action == "generate"
    assert [route.source for route in guide_plan.routes] == ["product_help"]
    assert all(route.locked and route.required for route in guide_plan.routes)
    assert "简历工坊" in guide.deterministic_answer
    assert "不能指定某一份模板" in guide.deterministic_answer
    assert "不能精确指定输出职责条数" in guide.deterministic_answer
    assert "知识库维护" not in guide.deterministic_answer

    diagnose_q = "为什么没找到我的项目？"
    diagnose_plan = rule_plan_query(diagnose_q)
    diagnose = execute_plan(diagnose_plan, diagnose_q, 5, retrieve)
    assert calls == []
    assert [route.source for route in diagnose_plan.routes] == ["system_diagnostics"]
    assert "只读诊断" in diagnose.deterministic_answer
    assert "不会修改记录" in diagnose.deterministic_answer


def test_conceptual_why_is_explain_not_system_diagnose():
    from rag_query_plan import rule_plan_query

    conceptual = rule_plan_query("为什么 RAG 需要重排？")
    failed_query = rule_plan_query("为什么这次检索路由不对？")
    assert conceptual.intent_frame.service_mode == "explain"
    assert {(r.source, r.operation) for r in conceptual.routes} == {
        ("reference_kb", "search")
    }
    assert failed_query.intent_frame.service_mode == "diagnose"
    assert {(r.source, r.operation) for r in failed_query.routes} == {
        ("system_diagnostics", "inspect")
    }


def test_personal_storage_location_does_not_add_curated_kb_route():
    from rag_query_plan import rule_plan_query

    plan = rule_plan_query("知识库里有哪些我审批过的个人项目？")
    assert plan.intent_frame.service_mode == "recall"
    assert {(route.source, route.operation) for route in plan.routes} == {
        ("project_memory", "list_approved")
    }


def test_general_application_method_does_not_read_personal_applications():
    from rag_query_plan import rule_plan_query

    plan = rule_plan_query("企业官网投递通常有哪些注意事项？")
    assert plan.intent_frame.service_mode == "explain"
    assert {(route.source, route.operation) for route in plan.routes} == {
        ("reference_kb", "search")
    }


def test_v3_planner_preserves_complementary_sources(monkeypatch):
    import json
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "interaction_kind": "query",
        "service_mode": "advise", "action_request": None,
        "tasks": [{"task_id": "t", "subquery": "项目中的 RRF",
                   "answer_routes": ["project_memory.search", "reference_kb.search"],
                   "context_routes": [], "filters": {}, "depends_on": []}],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query(
        "根据我的项目材料介绍 RRF 的作用",
        llm_call=lambda _prompt: payload,
        semantic_ranker=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("prototype ranker is offline only")
        ),
    )
    assert {(route.source, route.operation) for route in plan.routes} == {
        ("project_memory", "search"), ("reference_kb", "search")
    }
    assert plan.confidence_factors["llm_calls"] == 1


def test_evidence_gate_allows_general_answer_only_for_explain(monkeypatch):
    import rag_gate
    import rag_query_plan
    import rag_tools
    from rag_multi_source import ExecutionResult

    explain = rag_query_plan.rule_plan_query("不要使用内部资料，解释什么是倒排索引")
    monkeypatch.setattr(rag_query_plan, "plan_query", lambda *_args, **_kwargs: explain)
    monkeypatch.setattr(
        rag_gate, "_execute_query_plan", lambda *_args, **_kwargs: ExecutionResult(),
    )
    monkeypatch.setattr(
        rag_gate, "synthesize_fallback_answer", lambda *_args, **_kwargs: "普通模型解释",
    )
    monkeypatch.setattr(rag_tools, "has_embedding_api_key", lambda: False)
    answer = rag_gate.gated_query("问题")
    assert answer["mode"] == "general_fallback"
    assert "普通模型解释" in answer["answer"]

    recall = rag_query_plan.rule_plan_query("我的个人项目有哪些？")
    monkeypatch.setattr(rag_query_plan, "plan_query", lambda *_args, **_kwargs: recall)
    answer = rag_gate.gated_query("问题")
    assert answer["mode"] != "general_fallback"
    assert "没有找到可用的个人记录" in answer["answer"]
