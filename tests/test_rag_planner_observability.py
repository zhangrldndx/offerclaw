from types import SimpleNamespace


def _plan(**factors):
    frame = SimpleNamespace(
        service_mode="explain",
        interaction_kind="query",
        turn_relation="standalone",
        relation_target_turn_id="",
        action_request={},
        requested_action="answer",
        routing_assurance="schema_validated",
        decision_reasons=["structured_v4_contract_validated"],
        context_resolution={},
        to_dict=lambda: {
            "service_mode": "explain", "interaction_kind": "query",
            "turn_relation": "standalone", "relation_target_turn_id": "",
            "action_request": {}, "routing_assurance": "schema_validated",
            "decision_reasons": ["structured_v4_contract_validated"],
            "context_resolution": {},
        },
    )
    return SimpleNamespace(
        intent_frame=frame,
        planner_mode="llm",
        resolver_mode="llm",
        decision="answer",
        clarification="",
        routes=[],
        candidate_routes=[],
        confidence_factors=factors,
        reason="test",
        planner_engine="structured_llm",
        planner_version="llm-first-v3",
        schema_version="semantic-query-plan-v2",
        route_model="gpt-5.6",
        repair_used=False,
        fallback_reason="",
    )


def test_plan_meta_exposes_structured_runtime_metrics():
    from rag_gate import _plan_meta

    meta = _plan_meta(
        _plan(
            planner_queue_ms=1.25,
            planner_provider_ms=320.5,
            planner_wall_ms=325.0,
            planner_deadline_ms=8000.0,
            planner_cache_hit=True,
            planner_timeout_stage="",
            planner_late_response=False,
            planner_circuit_state="closed",
            prompt_tokens=1800,
            completion_tokens=120,
            structured_output_mode="json_object",
        ),
        planning_ms=326.0,
    )

    assert meta["planner_queue_ms"] == 1.25
    assert meta["planner_provider_ms"] == 320.5
    assert meta["planner_wall_ms"] == 325.0
    assert meta["planner_deadline_ms"] == 8000.0
    assert meta["planner_cache_hit"] is True
    assert meta["planner_circuit_state"] == "closed"
    assert meta["prompt_tokens"] == 1800
    assert meta["completion_tokens"] == 120
    assert meta["structured_output_mode"] == "json_object"


def test_synth_model_defaults_to_global_gpt(monkeypatch):
    import rag_gate

    monkeypatch.delenv("RAG_SYNTH_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6")
    assert rag_gate._synth_model() == "gpt-5.6"
