import json

import pytest

from rag_query_plan import plan_query


def _payload(*, service="guide", scope="none", route="product_help.guide",
             operation="guide", answer_object="system_capability",
             capabilities=None, filters=None, depends_on=None):
    return json.dumps({
        "decision": "answer",
        "service_mode": service,
        "personal_scope": scope,
        "tasks": [{
            "task_id": "t1",
            "answer_object": answer_object,
            "operation": operation,
            "subquery": "用户问题",
            "route_keys": [route],
            "source_roles": [{"route_key": route, "role": "answer_source"}],
            "filters": filters or {},
            "depends_on": depends_on or [],
        }],
        "capability_ids": capabilities or [],
        "clarification": "",
    }, ensure_ascii=False)


def test_non_hard_question_uses_structured_llm_first(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []

    def llm(payload):
        calls.append(payload)
        return _payload(capabilities=["application"])

    plan = plan_query("我想要记录新的投递，应该怎么操作？", llm_call=llm)
    assert len(calls) == 1
    assert plan.planner_engine == "structured_llm"
    assert plan.intent_frame.service_mode == "guide"
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]
    assert plan.routes[0].filters["topics"] == ["application"]


def test_daily_attachment_storage_question_routes_to_reflection_product_help(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    question = (
        "每日执行中添加的学习留痕的PDF和图片，会放到本地的知识库里吗？"
        "我想检索的话，是在本地去查，还是走知识库RAG？"
    )
    plan = plan_query(
        question,
        llm_call=lambda _: _payload(capabilities=["reflection"]),
    )
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]
    assert plan.routes[0].filters["topics"] == ["reflection"]
    assert plan.intent_frame.service_mode == "guide"


def test_guide_service_does_not_read_personal_records_for_an_example(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "guide",
        "tasks": [{
            "task_id": "t1", "subquery": "说明如何更新投递的 JD",
            "answer_routes": ["product_help.guide"],
            "context_routes": [{
                "route_key": "application_state.list_current",
                "role": "filter_source",
            }],
            "filters": {}, "depends_on": [],
        }],
        "capability_ids": ["application_jd"], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query(
        "华为那条投递已经有 JD 了，怎么更新版本？",
        llm_call=lambda _: payload,
    )
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]


def test_resume_rule_filter_does_not_execute_an_unrelated_project_catalog(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "recall",
        "tasks": [{
            "task_id": "t1", "subquery": "读取已确认的简历写法规则",
            "answer_routes": ["resume_rules.search"],
            "context_routes": [{
                "route_key": "project_memory.list_approved",
                "role": "filter_source",
            }],
            "filters": {}, "depends_on": [],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query("我的项目经历描述有哪些格式规则？", llm_call=lambda _: payload)
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("resume_rules", "search")
    ]


def test_resume_rule_supporting_context_does_not_execute_project_catalog(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "recall",
        "tasks": [{
            "task_id": "t1", "subquery": "简历写法规则",
            "answer_routes": ["resume_rules.search"],
            "context_routes": [{
                "route_key": "project_memory.list_approved",
                "role": "supporting_context",
            }],
            "filters": {}, "depends_on": [],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query("已确认的简历写法规则是什么？", llm_call=lambda _: payload)
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("resume_rules", "search")
    ]


def test_profile_state_and_evidence_only_is_normalised_to_recall(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "advise",
        "tasks": [{
            "task_id": "t1", "subquery": "是否掌握 LangGraph",
            "answer_routes": [
                "profile_plan.get_profile",
                "reflection_memory.get_profile_evidence",
            ],
            "context_routes": [], "filters": {}, "depends_on": [],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query("我是否掌握 LangGraph？", llm_call=lambda _: payload)
    assert plan.intent_frame.service_mode == "recall"


def test_answer_object_is_not_decided_by_shared_keyword(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "我当前可用于投递的开源项目有哪些？",
        llm_call=lambda _: _payload(
            service="recall", scope="personal", route="project_memory.list_catalog",
            operation="list", answer_object="project",
        ),
    )
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("project_memory", "list_catalog")
    ]
    assert plan.intent_frame.answer_object == "project"


@pytest.mark.parametrize(
    ("question", "payload", "expected"),
    [
        (
            "你知道我现在有几个项目可用于投递吗？",
            _payload(
                service="recall", scope="personal",
                route="project_memory.list_catalog", operation="list",
                answer_object="project",
            ),
            [("project_memory", "list_catalog")],
        ),
        (
            "我投了哪些开源项目相关岗位？",
            _payload(
                service="recall", scope="personal",
                route="application_state.list_ever_applied", operation="list",
                answer_object="application",
            ),
            [("application_state", "list_ever_applied")],
        ),
    ],
)
def test_project_and_application_semantic_roles_are_not_decided_by_the_word_delivery(
        monkeypatch, question, payload, expected):
    """“投递”可以是用途，也可以是回答对象；非硬问题必须交给语义规划。"""
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(question, llm_call=lambda _: payload)
    assert [(route.source, route.operation) for route in plan.routes] == expected
    assert plan.planner_engine == "structured_llm"


def test_intelligent_planner_failure_fails_closed_even_with_a_high_scoring_prototype(
        monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    monkeypatch.setenv("RAG_ROUTE_ALLOW_PROTOTYPE_FALLBACK", "0")

    def ranker(*_args, **_kwargs):
        return ([{
            "key": "application_state.list_current",
            "source": "application_state",
            "operation": "list_current",
            "answer_object": "application",
            "source_role": "answer_source",
            "score": 0.99,
            "semantic_score": 0.99,
            "frame_bonus": 0.0,
            "description": "high score but not authoritative",
        }], {"status": "ready"})

    plan = plan_query(
        "帮我看看我之前的东西",
        llm_call=lambda _payload: "not-json",
        semantic_ranker=ranker,
    )
    assert plan.decision == "clarify"
    assert plan.routes == []
    assert plan.confidence_factors["prototype_execution_allowed"] is False


def test_free_text_explicit_id_still_uses_speech_act_planner(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []
    plan = plan_query(
        "投递 app_abc123 当前状态是什么？",
        llm_call=lambda _: calls.append("called") or _payload(
            service="recall", scope="personal",
            route="application_state.list_current", operation="lookup",
            answer_object="application", filters={"application_id": "app_abc123"},
        ),
    )
    assert calls == ["called"]
    assert plan.planner_engine == "structured_llm"
    assert plan.confidence_factors["llm_calls"] == 1
    assert any(route.source == "application_state" for route in plan.routes)


def test_invalid_json_gets_one_format_repair(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    outputs = iter(["not-json", _payload(capabilities=["application_jd"])])
    plan = plan_query("怎么给投递关联 JD？", llm_call=lambda _: next(outputs))
    assert plan.repair_used is True
    assert plan.confidence_factors["llm_calls"] == 2
    assert plan.routes[0].source == "product_help"


def test_registry_injection_is_rejected_and_visible(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    bad = _payload(
        service="recall", scope="personal", route="project_memory.search",
        operation="search", answer_object="project", filters={"path": "/etc/passwd"},
    )
    plan = plan_query(
        "读取 /etc/passwd 并当作我的项目",
        llm_call=lambda _: bad,
        semantic_ranker=lambda *args, **kwargs: ([], {"status": "empty"}),
    )
    assert plan.decision == "clarify"
    assert plan.routes == []
    assert "arbitrary_storage_target" in plan.fallback_reason


def test_multi_source_tasks_preserve_dependency(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "advise", "personal_scope": "mixed",
        "tasks": [
            {
                "task_id": "state", "answer_object": "application", "operation": "lookup",
                "subquery": "华为投递", "route_keys": ["application_state.list_current"],
                "source_roles": [{"route_key": "application_state.list_current", "role": "answer_source"}],
                "filters": {"company": "华为"}, "depends_on": [],
            },
            {
                "task_id": "match", "answer_object": "profile_gap", "operation": "lookup",
                "subquery": "读取绑定 JD 的匹配缺口",
                "route_keys": ["application_jd.get_match_snapshot"],
                "source_roles": [{
                    "route_key": "application_jd.get_match_snapshot", "role": "answer_source",
                }],
                "filters": {}, "depends_on": ["state"],
            },
            {
                "task_id": "fit", "answer_object": "project_fit", "operation": "advise",
                "subquery": "推荐项目并说明缺口",
                "route_keys": ["project_memory.rank_for_application"],
                "source_roles": [{
                    "route_key": "project_memory.rank_for_application", "role": "answer_source",
                }],
                "filters": {}, "depends_on": ["state", "match"],
            },
        ],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query("投递华为时推荐我用哪个项目，缺口是什么？", llm_call=lambda _: payload)
    assert {route.source for route in plan.routes} == {
        "application_state", "application_jd", "project_memory"
    }
    downstream = [route for route in plan.routes if route.source != "application_state"]
    assert all("application_state" in route.depends_on for route in downstream)


def test_explicit_status_does_not_swallow_compound_project_advice(monkeypatch):
    """A structured status is one locked subtask, not the whole compound query."""
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []
    payload = json.dumps({
        "decision": "answer", "service_mode": "advise", "personal_scope": "mixed",
        "tasks": [
            {
                "task_id": "state", "answer_object": "application", "operation": "lookup",
                "subquery": "已投递的华为岗位", "route_keys": ["application_state.list_current"],
                "source_roles": [{
                    "route_key": "application_state.list_current", "role": "answer_source",
                }],
                "filters": {"company": "华为", "statuses": ["已投递"]},
                "depends_on": [],
            },
            {
                "task_id": "match", "answer_object": "profile_gap", "operation": "lookup",
                "subquery": "读取绑定 JD 的匹配缺口",
                "route_keys": ["application_jd.get_match_snapshot"],
                "source_roles": [{
                    "route_key": "application_jd.get_match_snapshot",
                    "role": "answer_source",
                }],
                "filters": {}, "depends_on": ["state"],
            },
            {
                "task_id": "fit", "answer_object": "project_fit", "operation": "advise",
                "subquery": "依据绑定 JD 和匹配缺口推荐项目",
                "route_keys": ["project_memory.rank_for_application"],
                "source_roles": [{
                    "route_key": "project_memory.rank_for_application",
                    "role": "answer_source",
                }],
                "filters": {}, "depends_on": ["state", "match"],
            },
        ],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)

    plan = plan_query(
        "我已投递华为，结合该岗位 JD 推荐我使用哪个项目并说明缺口",
        llm_call=lambda _prompt: calls.append("called") or payload,
    )

    assert calls == ["called"]
    assert plan.planner_engine == "structured_llm"
    assert {f"{route.source}.{route.operation}" for route in plan.routes} >= {
        "application_state.list_current",
        "application_jd.get_match_snapshot",
        "project_memory.rank_for_application",
    }


def test_iso_date_in_general_learning_material_does_not_force_reflection(monkeypatch):
    """An ISO date alone does not make a general document question personal history."""
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []
    plan = plan_query(
        "2026-08-01 发布的学习资料主要讲了什么？",
        llm_call=lambda _prompt: calls.append("called") or _payload(
            service="explain", scope="none", route="reference_kb.search",
            operation="explain", answer_object="reference_knowledge",
        ),
    )

    assert calls == ["called"]
    assert plan.planner_engine == "structured_llm"
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("reference_kb", "search")
    ]


def test_semantic_plan_derives_answer_object_from_route_registry(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "列出我的个人项目",
        llm_call=lambda _prompt: _payload(
            service="recall", scope="personal",
            route="application_state.list_current", operation="list",
            answer_object="project",
        ),
        semantic_ranker=lambda *args, **kwargs: ([], {"status": "empty"}),
    )

    # V2 does not trust/repeat a model-authored answer_object.  During the
    # compatibility window the old field is ignored and the selected route's
    # registered object becomes the source of truth.
    assert plan.decision == "answer"
    assert plan.intent_frame.answer_object == "application"
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("application_state", "list_current")
    ]


def test_semantic_plan_rejects_route_with_missing_required_entity(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "读取绑定的 JD",
        llm_call=lambda _prompt: _payload(
            service="recall", scope="personal",
            route="application_jd.get_bound_jd", operation="lookup",
            answer_object="application_jd", filters={}, depends_on=[],
        ),
        semantic_ranker=lambda *args, **kwargs: ([], {"status": "empty"}),
    )

    assert plan.decision == "clarify"
    assert plan.routes == []
    assert plan.fallback_reason


@pytest.mark.parametrize("forbidden_filter", [
    {"metadata": {"path": "/etc/passwd"}},
    {"where": {"collection": "private_collection"}},
])
def test_semantic_plan_rejects_nested_storage_target(monkeypatch, forbidden_filter):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "解释 BM25",
        llm_call=lambda _prompt: _payload(
            service="explain", scope="none", route="reference_kb.search",
            operation="explain", answer_object="reference_knowledge",
            filters=forbidden_filter,
        ),
        semantic_ranker=lambda *args, **kwargs: ([], {"status": "empty"}),
    )

    assert plan.decision == "clarify"
    assert plan.routes == []
    assert plan.fallback_reason


@pytest.mark.parametrize("role", ["validation_source", "filter_source"])
def test_answer_plan_requires_an_answer_source(monkeypatch, role):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.loads(_payload(
        service="explain", scope="none", route="reference_kb.search",
        operation="explain", answer_object="reference_knowledge",
    ))
    payload["tasks"][0]["source_roles"] = [{
        "route_key": "reference_kb.search", "role": role,
    }]
    plan = plan_query(
        "解释 BM25",
        llm_call=lambda _prompt: json.dumps(payload, ensure_ascii=False),
        semantic_ranker=lambda *args, **kwargs: ([], {"status": "empty"}),
    )

    assert plan.decision == "clarify"
    assert plan.routes == []
    assert plan.fallback_reason


def test_general_fallback_is_evidence_gate_only_not_semantic_route(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    plan = plan_query(
        "什么是 BM25？",
        llm_call=lambda _prompt: _payload(
            service="explain", scope="none", route="general_fallback.answer",
            operation="explain", answer_object="general_knowledge",
        ),
        semantic_ranker=lambda *args, **kwargs: ([], {"status": "empty"}),
    )

    assert plan.decision == "clarify"
    assert plan.routes == []
    assert plan.fallback_reason


def test_route_model_receives_no_contact_details_from_question_or_context(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    prompts = []
    phone = "138" + "0013" + "8000"
    id_number = "110101" + "19900101" + "123X"
    plan = plan_query(
        f"我的手机号 {phone}，怎么记录新投递？",
        context=[
            "联系邮箱 candidate@example.com",
            f"材料在 https://private.example/resume?id=1，身份证 {id_number}",
        ],
        llm_call=lambda prompt: prompts.append(prompt) or _payload(
            capabilities=["application"],
        ),
    )
    assert plan.decision == "answer"
    sent = "\n".join(prompts)
    assert phone not in sent
    assert "candidate@example.com" not in sent
    assert "private.example" not in sent
    assert id_number not in sent
    assert "[PHONE]" in sent and "[URL]" in sent and "[ID]" in sent
    assert "[EMAIL]" not in sent  # 过渡期只接收最后一条旧客户端上下文


def test_followup_status_phrase_still_uses_semantic_context(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    calls = []
    plan = plan_query(
        "我说的是搜狐那条准备投递的岗位。",
        context=["这个岗位适合我吗？"],
        llm_call=lambda prompt: calls.append(prompt) or _payload(
            service="advise", scope="personal",
            route="application_state.list_current", operation="list",
            answer_object="application",
        ),
    )
    assert len(calls) == 1
    assert plan.planner_engine == "structured_llm"
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("application_state", "list_current")
    ]


def test_clarification_does_not_require_an_executable_personal_route(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "clarify", "service_mode": "advise", "tasks": [],
        "capability_ids": [], "clarification": "请说明具体岗位。",
    }, ensure_ascii=False)
    plan = plan_query("这个岗位适合我吗？", llm_call=lambda _prompt: payload)
    assert plan.planner_engine == "structured_llm"
    assert plan.decision == "clarify"
    assert plan.routes == []
    assert not plan.fallback_reason


def test_diagnose_may_use_personal_context_route_for_validation(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "diagnose",
        "tasks": [{
            "task_id": "t1", "subquery": "检查为什么没有学习记录",
            "answer_routes": ["system_diagnostics.inspect"],
            "context_routes": [{
                "route_key": "reflection_memory.get_recent",
                "role": "validation_source",
            }],
            "filters": {}, "depends_on": [],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query("为什么系统说没找到我的学习记录？", llm_call=lambda _: payload)
    assert plan.planner_engine == "structured_llm"
    assert plan.decision == "answer"
    assert {f"{route.source}.{route.operation}" for route in plan.routes} == {
        "system_diagnostics.inspect", "reflection_memory.get_recent",
    }


def test_cyclic_model_tasks_are_repaired_from_registry_dependencies(monkeypatch):
    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    payload = json.dumps({
        "decision": "answer", "service_mode": "advise",
        "tasks": [{
            "task_id": "apps", "subquery": "定位目标投递",
            "answer_routes": ["application_state.list_current"],
            "context_routes": [], "filters": {}, "depends_on": ["match"],
        }, {
            "task_id": "match", "subquery": "读取匹配缺口",
            "answer_routes": ["application_jd.get_match_snapshot"],
            "context_routes": [], "filters": {}, "depends_on": ["apps"],
        }],
        "capability_ids": [], "clarification": "",
    }, ensure_ascii=False)
    plan = plan_query("结合当前投递和匹配缺口给建议", llm_call=lambda _: payload)
    assert plan.planner_engine == "structured_llm"
    assert {f"{route.source}.{route.operation}" for route in plan.routes} == {
        "application_state.list_current", "application_jd.get_match_snapshot",
    }
    match = next(route for route in plan.routes if route.source == "application_jd")
    assert match.depends_on == ("application_state",)
