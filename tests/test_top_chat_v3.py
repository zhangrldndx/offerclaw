# -*- coding: utf-8 -*-
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest


def _command_payload(*, status="已投递", commitment="requested",
                     turn_relation="standalone"):
    return json.dumps({
        "decision": "answer",
        "interaction_kind": "command",
        "turn_relation": turn_relation,
        "service_mode": "guide",
        "action_request": {
            "verb": "update",
            "target_type": "application",
            "field_changes": [{"field": "status_code", "value_code": status}],
            "target_refs": [],
            "commitment": commitment,
        },
        "tasks": [{
            "task_id": "guide",
            "subquery": "说明如何更新投递状态",
            "answer_routes": ["product_help.guide"],
            "context_routes": [], "filters": {}, "depends_on": [],
        }],
        "capability_ids": ["application.update_status"],
        "clarification": "",
    }, ensure_ascii=False)


def _query_payload(route="application_state.get_next_actions",
                   turn_relation="standalone"):
    return json.dumps({
        "decision": "answer", "interaction_kind": "query",
        "turn_relation": turn_relation,
        "service_mode": "recall", "action_request": None,
        "tasks": [{
            "task_id": "read", "subquery": "读取投递下一步",
            "answer_routes": [route], "context_routes": [],
            "filters": {}, "depends_on": [],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)


def test_write_command_uses_v3_action_contract_and_never_enters_recall(
        monkeypatch, tmp_path):
    from memory_store import MemoryStore
    from rag_multi_source import execute_plan
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    store = MemoryStore(tmp_path)
    event_count = store.stats()["events"]
    applications_path = Path(__file__).parents[1] / "applications.md"
    applications_hash = (
        hashlib.sha256(applications_path.read_bytes()).hexdigest()
        if applications_path.is_file() else ""
    )
    calls = []
    question = "帮我把这条投递状态改成已投递"
    plan = plan_query(
        question, llm_call=lambda prompt: calls.append(prompt) or _command_payload(),
    )

    assert len(calls) == 1
    assert plan.intent_frame.interaction_kind == "command"
    assert plan.intent_frame.service_mode == "guide"
    assert plan.intent_frame.action_request["verb"] == "update"
    assert plan.intent_frame.action_request["target_type"] == "application"
    assert plan.intent_frame.action_request["field_changes"] == [
        {"field": "status_code", "value_code": "applied"}
    ]
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]
    assert plan.intent_frame.context_resolution["status"] == "not_required_for_guide"
    assert plan.intent_frame.context_resolution["original_status"] == "missing_antecedent"

    def forbidden_retrieval(*_args, **_kwargs):
        raise AssertionError("Guide must not read applications or a vector index")

    result = execute_plan(plan, question, 5, forbidden_retrieval)
    assert result.coverage == {"product_help": 1}
    assert "没有修改任何数据" in result.deterministic_answer
    assert "application.update_status" in result.deterministic_answer
    assert "status_code=applied" in result.deterministic_answer
    assert store.stats()["events"] == event_count
    if applications_hash:
        assert hashlib.sha256(applications_path.read_bytes()).hexdigest() == applications_hash


@pytest.mark.parametrize("obsolete_mode", ["legacy", "intelligent_shadow", "hybrid", "shadow"])
def test_obsolete_router_mode_cannot_downgrade_free_text(monkeypatch, obsolete_mode):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", obsolete_mode)
    plan = plan_query(
        "帮我把这条投递状态改成已投递",
        llm_call=lambda _prompt: _command_payload(),
    )

    assert plan.planner_engine == "structured_llm"
    assert plan.intent_frame.interaction_kind == "command"
    assert plan.intent_frame.service_mode == "guide"
    assert plan.intent_frame.capability_ids == ["application.update_status"]
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]


def test_invalid_or_conflicting_machine_status_fails_closed(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "把投递改成随便看看",
        llm_call=lambda _prompt: _command_payload(status="随便看看"),
        semantic_ranker=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prototype fallback must not run")
        ),
    )
    assert plan.decision == "clarify"
    assert plan.routes == []
    assert plan.planner_engine == "structured_llm_unavailable"
    assert "invalid_application_status_code" in plan.fallback_reason


def test_valid_command_missing_entity_is_normalized_to_generic_guide(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.loads(_command_payload())
    payload.update({
        "decision": "clarify", "tasks": [],
        "clarification": "请提供具体投递记录。",
    })
    plan = plan_query(
        "帮我把这条投递状态改成已投递",
        llm_call=lambda _prompt: json.dumps(payload, ensure_ascii=False),
    )
    assert plan.decision == "answer"
    assert plan.clarification == ""
    assert plan.intent_frame.interaction_kind == "command"
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]


def test_omitted_capability_is_recovered_only_from_unique_typed_match(monkeypatch):
    from action_capabilities import infer_unique_capability_id
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.loads(_command_payload())
    payload.update({
        "decision": "clarify", "tasks": [], "capability_ids": [],
        "clarification": "请提供具体投递记录。",
    })
    plan = plan_query(
        "帮我把这条投递状态改成已投递",
        llm_call=lambda _prompt: json.dumps(payload, ensure_ascii=False),
    )
    assert plan.decision == "answer"
    assert plan.intent_frame.capability_ids == ["application.update_status"]
    assert infer_unique_capability_id({
        "verb": "update", "target_type": "application", "field_changes": [],
    }) == ""


def test_free_text_stable_id_no_longer_bypasses_speech_act_planner(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []
    plan = plan_query(
        "把投递 app_abc123 的状态设为已投递",
        llm_call=lambda prompt: calls.append(prompt) or _command_payload(),
    )
    assert len(calls) == 1
    assert plan.intent_frame.interaction_kind == "command"
    assert plan.planner_engine == "structured_llm"


@pytest.mark.parametrize("question", [
    "把这条投递改成已投递",
    "把这条投递设为已投递",
    "请标记这条投递为已投递",
    "投递状态调整到已投递",
    "这条申请换成已投递状态",
])
def test_action_paraphrases_all_reach_the_same_structured_planner(
        monkeypatch, question):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []
    plan = plan_query(
        question, llm_call=lambda prompt: calls.append(prompt) or _command_payload(),
    )
    assert len(calls) == 1
    assert plan.intent_frame.interaction_kind == "command"
    assert plan.routes[0].filters["capability_ids"] == ["application.update_status"]


def test_reported_or_negated_write_is_not_accepted_as_a_command(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    for commitment in ("reported", "negated", "hypothetical"):
        plan = plan_query(
            "这是一个非执行语境",
            llm_call=lambda _prompt, value=commitment: _command_payload(
                commitment=value
            ),
        )
        assert plan.decision == "clarify"
        assert plan.routes == []


def test_trusted_structured_read_is_the_only_zero_llm_fast_path():
    from rag_query_plan import plan_structured_read

    plan = plan_structured_read(
        "application_state", "list_current",
        filters={"application_ids": ["app_abc123"]},
    )
    assert plan.planner_engine == "structured_read"
    assert plan.confidence_factors["llm_calls"] == 0
    assert plan.intent_frame.routing_assurance == "deterministic"


def test_application_next_action_has_shared_output_contract(monkeypatch):
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "这些公司下一步呢？", llm_call=lambda _prompt: _query_payload(),
    )
    # No server-side antecedent means the read is held for clarification, but
    # the route contract itself no longer says application_action.
    assert plan.decision == "clarify"
    from rag_route_registry import route_output_contracts
    assert [item.to_dict() for item in route_output_contracts(
        "application_state", "get_next_actions"
    )] == [{"entity_type": "application", "projection": "next_action"}]


def test_server_context_binds_latest_compatible_real_ids(monkeypatch, tmp_path):
    from conversation_context import record_successful_turn
    from memory_store import MemoryStore
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    store = MemoryStore(tmp_path)
    record_successful_turn("conv_test", {
        "decision": "answer", "service_mode": "recall",
        "routes": [{"source": "application_state", "operation": "list_current"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "application", "projection": "summary"}
        ]},
        "resolved_entities": {"application_ids": ["app_one", "app_two"]},
    }, turn_id="turn_apps", store=store)
    record_successful_turn("conv_test", {
        "decision": "answer", "service_mode": "explain",
        "routes": [{"source": "reference_kb", "operation": "search"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "knowledge", "projection": "reference"}
        ]},
        "resolved_entities": {},
    }, turn_id="turn_kb", store=store)

    prompts = []
    plan = plan_query(
        "这些公司下一步呢？", conversation_id="conv_test",
        llm_call=lambda prompt: prompts.append(prompt) or _query_payload(),
    )
    assert plan.decision == "answer"
    assert plan.routes[0].filters["application_ids"] == ["app_one", "app_two"]
    assert plan.routes[0].filters["strict_entity_filter"] is True
    assert plan.intent_frame.context_resolution["turn_id"] == "turn_apps"
    assert "turn_kb" not in prompts[0]
    assert "app_one" in prompts[0] and "app_two" in prompts[0]


def test_explicit_new_topic_does_not_inherit_previous_entities(monkeypatch, tmp_path):
    from conversation_context import record_successful_turn
    from memory_store import MemoryStore
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    store = MemoryStore(tmp_path)
    record_successful_turn("conv_switch", {
        "decision": "answer", "service_mode": "recall",
        "routes": [{"source": "application_state", "operation": "list_current"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "application", "projection": "summary"}
        ]},
        "resolved_entities": {"application_ids": ["app_previous"]},
    }, turn_id="turn_previous", store=store)
    payload = json.dumps({
        "decision": "answer", "interaction_kind": "query",
        "service_mode": "explain", "action_request": None,
        "tasks": [{
            "task_id": "explain", "subquery": "BM25 的作用",
            "answer_routes": ["reference_kb.search"],
            "context_routes": [], "filters": {}, "depends_on": [],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    prompts = []
    plan = plan_query(
        "解释 BM25 的作用", conversation_id="conv_switch",
        llm_call=lambda prompt: prompts.append(prompt) or payload,
    )
    assert plan.decision == "answer"
    assert plan.intent_frame.context_resolution == {}
    assert "app_previous" not in prompts[0]


def test_singular_reference_to_multiple_ids_requires_clarification(monkeypatch, tmp_path):
    from conversation_context import record_successful_turn
    from memory_store import MemoryStore
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    store = MemoryStore(tmp_path)
    record_successful_turn("conv_multi", {
        "decision": "answer", "service_mode": "recall",
        "routes": [{"source": "application_state", "operation": "list_current"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "application", "projection": "summary"}
        ]},
        "resolved_entities": {"application_ids": ["app_one", "app_two"]},
    }, turn_id="turn_multi", store=store)
    plan = plan_query(
        "这个下一步呢？", conversation_id="conv_multi",
        llm_call=lambda _prompt: _query_payload(),
    )
    assert plan.decision == "clarify"
    assert plan.planner_engine == "context_gate"
    assert plan.intent_frame.routing_assurance == "ambiguous"


def test_command_can_use_one_bound_id_but_still_only_returns_guidance(monkeypatch, tmp_path):
    from conversation_context import record_successful_turn
    from memory_store import MemoryStore
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    store = MemoryStore(tmp_path)
    record_successful_turn("conv_one", {
        "decision": "answer", "service_mode": "recall",
        "routes": [{"source": "application_state", "operation": "list_current"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "application", "projection": "summary"}
        ]},
        "resolved_entities": {"application_ids": ["app_one"]},
    }, turn_id="turn_one", store=store)
    plan = plan_query(
        "把这条投递改成已投递", conversation_id="conv_one",
        llm_call=lambda _prompt: _command_payload(),
    )
    assert plan.intent_frame.action_request["target_refs"] == [
        {"entity_type": "application", "entity_id": "app_one"}
    ]
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]


def test_continuation_binds_latest_turn_only_after_semantic_classification(
        monkeypatch, tmp_path):
    from conversation_context import record_successful_turn
    from memory_store import MemoryStore
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    store = MemoryStore(tmp_path)
    record_successful_turn("conv_continue", {
        "decision": "answer", "service_mode": "recall",
        "routes": [{"source": "application_state", "operation": "list_current"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "application", "projection": "summary"}
        ]},
        "resolved_entities": {"application_ids": ["app_previous"]},
    }, turn_id="turn_previous", store=store)
    prompts = []
    plan = plan_query(
        "继续说下一步。", conversation_id="conv_continue",
        llm_call=lambda prompt: prompts.append(prompt) or _query_payload(
            turn_relation="continuation"
        ),
    )

    assert "app_previous" not in prompts[0]
    assert plan.intent_frame.turn_relation == "continuation"
    assert plan.intent_frame.relation_target_turn_id == "turn_previous"
    assert plan.intent_frame.context_resolution["status"] == "relation_resolved"
    assert plan.routes[0].filters["application_ids"] == ["app_previous"]


def test_correction_is_orthogonal_to_command_and_uses_server_turn_id(
        monkeypatch, tmp_path):
    from conversation_context import record_successful_turn
    from memory_store import MemoryStore
    from rag_query_plan import plan_query

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    store = MemoryStore(tmp_path)
    record_successful_turn("conv_correct", {
        "decision": "answer", "service_mode": "recall",
        "routes": [{"source": "application_state", "operation": "list_current"}],
        "intent_frame": {"output_contracts": [
            {"entity_type": "application", "projection": "summary"}
        ]},
        "resolved_entities": {"application_ids": ["app_one"]},
    }, turn_id="turn_to_correct", store=store)
    plan = plan_query(
        "不对，是把这条投递改成已投递。", conversation_id="conv_correct",
        llm_call=lambda _prompt: _command_payload(turn_relation="correction"),
    )

    assert plan.intent_frame.interaction_kind == "command"
    assert plan.intent_frame.turn_relation == "correction"
    assert plan.intent_frame.relation_target_turn_id == "turn_to_correct"
    assert plan.intent_frame.service_mode == "guide"


def test_expired_turn_is_not_a_reliable_antecedent(tmp_path):
    from conversation_context import resolve_conversation_context
    from memory_store import MemoryStore

    store = MemoryStore(tmp_path)
    old = (dt.datetime.now(dt.timezone.utc).astimezone()
           - dt.timedelta(hours=25)).isoformat(timespec="seconds")
    store.append_conversation_turn(
        conversation_id="conv_old", turn_id="turn_old", topic="application",
        service_mode="recall",
        output_contracts=[{"entity_type": "application", "projection": "summary"}],
        resolved_entities={"application_ids": ["app_old"]},
        capability_ids=[], completed_at=old,
    )
    resolved = resolve_conversation_context(
        "这条投递呢？", "conv_old", store=store,
    )
    assert resolved.status == "missing_antecedent"


def test_frontend_does_not_send_raw_question_history():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "ragQuestionHistory" not in source
    assert "conversation_id: TOP_CHAT_CONVERSATION_ID" in source
