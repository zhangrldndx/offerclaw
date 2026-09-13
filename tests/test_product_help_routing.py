import json
from pathlib import Path


def _fixture():
    path = Path(__file__).parent / "fixtures" / "product_help_intent_eval_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_product_operations_route_to_locked_deterministic_help():
    from rag_query_plan import _plan_validation_errors, rule_plan_query

    for case in _fixture()["positive"]:
        plan = rule_plan_query(case["q"])
        assert [(route.source, route.operation) for route in plan.routes] == [
            ("product_help", "guide")
        ], case["q"]
        route = plan.routes[0]
        assert route.required and route.locked
        assert route.filters["topics"][0] == case["topic"]
        assert route.filters["action"] == case["action"]
        assert plan.intent_frame.answer_objects == ["system_capability"]
        assert plan.intent_frame.operations == ["guide"]
        assert plan.intent_frame.personal_scope == "none"
        assert _plan_validation_errors(plan) == []


def test_natural_operation_howto_uses_v3_and_skips_prototype_router(monkeypatch):
    """开放表达由同一次结构化 LLM 识别，原型排序没有执行资格。"""
    from rag_query_plan import plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "intelligent")
    called = []

    def ranker(*_args, **_kwargs):
        called.append("semantic")
        raise AssertionError("locked Guide must not call the semantic router")

    payload = json.dumps({
        "decision": "answer", "interaction_kind": "how_to",
        "service_mode": "guide",
        "action_request": {"verb": "create", "target_type": "application",
                           "field_changes": [], "target_refs": [],
                           "commitment": "hypothetical"},
        "tasks": [{"task_id": "guide", "subquery": "记录投递",
                   "answer_routes": ["product_help.guide"],
                   "context_routes": [], "filters": {}, "depends_on": []}],
        "capability_ids": ["application.create"], "clarification": "",
    }, ensure_ascii=False)
    llm_calls = []
    plan = plan_query(
        "我想要记录我的新投递简历该怎么操作？",
        semantic_ranker=ranker,
        llm_call=lambda _prompt: llm_calls.append(1) or payload,
    )

    assert called == []
    assert llm_calls == [1]
    assert plan.intent_frame.service_mode == "guide"
    assert plan.intent_frame.interaction_kind == "how_to"
    assert plan.intent_frame.requested_action == "create"
    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]
    assert plan.routes[0].locked and plan.routes[0].required


def test_personal_facts_and_general_advice_do_not_become_product_help():
    from rag_query_plan import rule_plan_query

    for question in _fixture()["negative"]:
        plan = rule_plan_query(question)
        assert not any(route.source == "product_help" for route in plan.routes), question
        assert "system_capability" not in plan.intent_frame.answer_objects, question


def test_explicit_knowledge_content_lookup_is_not_product_management_help():
    from rag_query_plan import rule_plan_query

    question = "请只从我的知识库查：状态刷新采用什么替换策略？失败时旧块怎么处理？"
    plan = rule_plan_query(question)

    assert [(route.source, route.operation) for route in plan.routes] == [
        ("reference_kb", "search")
    ]
    assert "product_help" not in plan.intent_frame.domains


def test_actual_retrieval_failure_still_routes_to_diagnostics():
    from rag_query_plan import rule_plan_query

    plan = rule_plan_query("为什么这次知识库检索失败，没有结果？")

    assert [(route.source, route.operation) for route in plan.routes] == [
        ("system_diagnostics", "inspect")
    ]


def test_product_help_execution_is_read_only_and_needs_no_retrieval():
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query

    calls = []

    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        return {"in_kb": False, "chunks": [], "docs": [], "metas": []}

    plan = rule_plan_query("当前我有新的项目你可以帮助我添加进去嘛？")
    result = execute_plan(plan, "当前我有新的项目你可以帮助我添加进去嘛？", 5, retrieve)

    assert calls == []
    assert result.coverage == {"product_help": 1}
    assert result.source_status["product_help"]["status"] == "ok"
    assert "简历工坊" in result.deterministic_answer
    assert "待确认区" in result.deterministic_answer
    assert "不会直接新增" in result.deterministic_answer


def test_delete_help_does_not_claim_the_action_was_performed():
    from product_help import render_product_help

    text = render_product_help(["application"], "delete")
    assert "不执行删除" in text
    assert "尚未开放" in text
    assert "删除成功" not in text


def test_daily_attachment_help_distinguishes_local_refs_from_formal_kb_index():
    from product_help import render_product_help

    text = render_product_help(["reflection"], "locate")
    assert "daily_attachments/日期/" in text
    assert "daily_log.md" in text
    assert "reflection_memory" in text
    assert "不会自动进入正式知识库" in text
    assert "不会解析每日附件的 PDF 正文" in text
    assert "知识库维护" in text and "reference_kb" in text


def test_plan_help_registers_daily_task_link_lineage():
    from product_help import render_product_help

    text = render_product_help(["plan"], "locate")
    assert "关联计划任务" in text
    assert "GET /api/plan/today" in text
    assert "plans/ 中最新计划文件" in text
    assert "task_id 写入 daily_log.md" in text
    assert "不会自动改写计划或把任务标成完成" in text


def test_resume_agent_and_project_scope_register_one_review_workflow():
    from product_help import CARD_CAPABILITY_REGISTRY, render_product_help

    text = render_product_help(["resume_agent", "resume_project"], "locate")
    assert "完整 Markdown 简历草稿" in text
    assert "Resume Agent 负责生成和修改" in text
    assert "独立 Resume Critic Agent" in text
    assert "resume_drafts/<application_id>/" in text
    assert "单个项目经历" in text
    assert "产物范围设为“项目经历”" in text
    assert "产物继续经过硬校验、独立 Critic" in text
    assert "同一个生成按钮" in text
    assert CARD_CAPABILITY_REGISTRY["resume_agent"]["card_id"] == "resume_workshop"
    assert CARD_CAPABILITY_REGISTRY["resume_project"]["card_id"] == "resume_workshop"


def test_semantic_prompt_exposes_resume_entry_contrast():
    from semantic_query_planner import _capability_registry_for_prompt, _static_system_prompt

    capabilities = _capability_registry_for_prompt()
    assert "resume.generate_full" in capabilities["ids"]
    assert "resume.generate_project" in capabilities["ids"]
    assert "完整简历" in capabilities["h"]["resume_agent"]
    assert "项目经历段" in capabilities["h"]["resume_project"]
    prompt = _static_system_prompt()
    assert "command/how_to须给action_request和唯一动作能力ID" in prompt


def test_focused_guide_prompt_requires_comparing_real_processing_chains():
    from rag_multi_source import ExecutionResult, product_help_messages
    from rag_query_plan import QueryPlan, RequestFrame, RouteSpec

    question = "简历工坊的项目段按钮和完整简历按钮有什么区别？都会调用 Resume Agent 和 Critic 吗？"
    plan = QueryPlan(
        planner_mode="llm", resolver_mode="llm",
        intent_frame=RequestFrame(
            service_mode="guide", answer_object="system_capability",
            answer_objects=["system_capability"], operations=["guide"],
        ),
        routes=[RouteSpec(
            "product_help", "guide", question,
            filters={"topics": ["resume_agent", "resume_project"], "action": "locate"},
            required=True,
        )],
    )
    execution = ExecutionResult(
        deterministic_answer=(
            "简历工坊只有一个 Resume Agent → Critic 按钮。\n"
            "完整简历和项目经历是同一审查工作流的两个产物范围，二者都会经过硬校验和独立 Critic。"
        ),
        resolved_entities={"capability_topics": ["resume_agent", "resume_project"]},
    )
    system = product_help_messages(question, plan, execution)[0]["content"]
    assert "产物粒度" in system
    assert "是否会触发相同 Agent" in system
    assert "resume_agent、resume_project" in system
    assert "纠正先前的产品说明" in system
    assert "准确按钮" in system


def test_resume_workshop_multi_agent_comparison_routes_both_real_pipelines():
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query

    question = "简历工坊的分析和生成简历段和按JD生成简历段，哪个功能涉及到了多agent协作？"
    plan = rule_plan_query(question)

    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]
    route = plan.routes[0]
    assert route.required and route.locked
    assert route.filters["topics"] == ["resume_agent", "resume_project"]

    calls = []

    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        return {"in_kb": False, "chunks": [], "docs": [], "metas": []}

    result = execute_plan(plan, question, 5, retrieve)
    assert calls == []
    assert "产物范围=项目经历" in result.deterministic_answer
    assert "项目经历使用所选 JD 定制" in result.deterministic_answer
    assert "至多一次自动修订" in result.deterministic_answer
    assert "产物范围=完整简历" in result.deterministic_answer
    assert "独立 Critic Agent" in result.deterministic_answer


def test_product_correction_uses_current_registry_and_prior_question_context():
    from rag_query_plan import rule_plan_query

    previous = "哪个简历功能涉及多 Agent 协作？"
    question = "不对，完整简历入口应该在简历工坊，不是在投递管理。"
    plan = rule_plan_query(question, context=[previous])

    assert [(route.source, route.operation) for route in plan.routes] == [
        ("product_help", "guide")
    ]
    route = plan.routes[0]
    assert route.filters["request_kind"] == "correction"
    assert route.filters["topics"] == ["resume_agent", "resume_project"]
    assert "explicit_product_correction" in plan.intent_frame.deterministic_signals


def test_streaming_guide_renders_registered_facts_without_a_second_llm(
        monkeypatch):
    import rag_gate
    import rag_query_plan
    from rag_query_plan import QueryPlan, RequestFrame, RouteSpec

    question = "每日执行的关联计划部分怎么使用？下拉列表来自哪里？"
    plan = QueryPlan(
        planner_mode="llm",
        resolver_mode="llm",
        intent_frame=RequestFrame(
            service_mode="guide", answer_object="system_capability",
            answer_objects=["system_capability"], operations=["guide"],
        ),
        routes=[RouteSpec(
            "product_help", "guide", question,
            filters={"topics": ["plan"], "action": "locate"},
            required=True,
        )],
    )
    monkeypatch.setattr(rag_query_plan, "plan_query", lambda *_args, **_kwargs: plan)

    captured = {}

    def focused_stream(messages):
        captured["messages"] = messages
        return iter([
            "它用于把本次留痕关联到一条计划任务。",
            "列表来自 /api/plan/today，不会自动修改计划。",
        ])

    monkeypatch.setattr(rag_gate, "_chat_stream", focused_stream)
    events = list(rag_gate.gated_query_stream(question))
    answer = "".join(item.get("text", "") for item in events
                     if item.get("type") == "delta")
    assert "修改学习计划或计划任务" in answer
    assert "GET /api/plan/today" in answer
    assert captured == {}
