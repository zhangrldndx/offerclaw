# -*- coding: utf-8 -*-
"""多意图规划、结构化投递事实、多源证据与隐私回归。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _compatibility_router_mode(monkeypatch):
    """Keep compatibility-contract tests independent from local UI rollout mode."""
    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")


def _pairs(plan):
    return {(r.source, r.operation) for r in plan.routes}


def test_unlabelled_application_question_routes_to_deterministic_operations():
    from rag_query_plan import rule_plan_query as plan_query

    plan = plan_query("我投了哪几家？我接下来还需要去那些公司的官网（投递）？给我信息")
    assert _pairs(plan) == {
        ("application_state", "list_ever_applied"),
        ("application_state", "list_pending_submission"),
        ("application_state", "get_next_actions"),
    }
    assert plan.planner_mode == "rule"


def test_complex_question_is_decomposed_into_personal_and_reference_routes():
    from rag_query_plan import rule_plan_query as plan_query

    plan = plan_query(
        "目前我失败企业中，根据我的经验总结，说明我缺少什么能力，"
        "结合我提供你的参考资料，普及相关概念和技术应用知识，给出下一步学习建议"
    )
    pairs = _pairs(plan)
    assert ("application_state", "list_failed") in pairs
    assert ("application_experience", "search") in pairs
    assert ("profile_plan", "get_gaps") in pairs
    assert ("reference_kb", "search") in pairs
    assert "实时投递事实" in plan.answer_sections
    assert "参考资料与概念补充" in plan.answer_sections


def test_application_advice_resolves_anaphora_as_dependent_routes():
    from rag_query_plan import rule_plan_query as plan_query

    plan = plan_query("我最近要投递哪些企业？投递该企业有什么注意事项？")
    assert _pairs(plan) == {
        ("application_state", "list_pending_submission"),
        ("application_jd", "search_bound_jd"),
        ("application_experience", "search"),
        ("reference_kb", "search"),
    }
    semantic_routes = [r for r in plan.routes if r.source != "application_state"]
    assert all(r.depends_on == ("application_state",) for r in semantic_routes)
    assert all(r.bindings["companies"] == "application_state.companies"
               for r in semantic_routes)
    assert all(r.subquery == "投递该企业有什么注意事项" for r in semantic_routes)
    assert "按企业与岗位说明注意事项" in plan.answer_sections


def test_application_followups_cover_requirements_and_company_interview_advice():
    from rag_query_plan import rule_plan_query as plan_query

    requirements = plan_query("我准备投哪些公司？这些岗位需要哪些核心能力？")
    assert _pairs(requirements) == {
        ("application_state", "list_pending_submission"),
        ("application_jd", "search_bound_jd"),
        ("reference_kb", "search"),
    }
    ref = next(r for r in requirements.routes if r.source == "reference_kb")
    assert ref.filters["intent"] == "role_requirements"
    assert ref.depends_on == ("application_state",)

    interview = plan_query("我投过哪几家？这些公司的面试流程和注意事项是什么？")
    assert _pairs(interview) == {
        ("application_state", "list_ever_applied"),
        ("application_jd", "search_bound_jd"),
        ("application_experience", "search"),
        ("reference_kb", "search"),
    }


def test_general_application_advice_does_not_read_private_state():
    from rag_query_plan import rule_plan_query as plan_query

    assert _pairs(plan_query("一般应届生如何投递简历")) == {("reference_kb", "search")}
    assert _pairs(plan_query("不要查询我的投递，解释一下 RAG")) == {("reference_kb", "search")}


def test_explicit_paper_intent_uses_separate_route():
    from rag_query_plan import rule_plan_query as plan_query

    assert _pairs(plan_query("结合论文说明 ReAct 的研究依据")) == {("paper_kb", "search")}


def test_application_fact_view_keeps_status_semantics_and_missing_url_visible():
    from applications_store import render_application_facts, select_application_facts

    rows = [
        {"日期": "2026-08-01", "公司": "甲公司", "岗位": "AI 工程师",
         "来源": "官网 https://jobs.example/a", "地点": "上海", "当前状态": "已拒绝",
         "下一步动作": "复盘", "备注": "08-01 已投递 · 08-20 已拒绝"},
        {"日期": "2026-08-21", "公司": "乙公司", "岗位": "RAG 工程师",
         "来源": "官网", "地点": "北京", "当前状态": "准备投递",
         "下一步动作": "定稿简历", "备注": "08-21 准备投递"},
        {"日期": "2026-08-20", "公司": "丙公司", "岗位": "后端",
         "来源": "BOSS", "地点": "杭州", "当前状态": "不投递",
         "下一步动作": "—", "备注": "08-20 不投递"},
    ]
    facts = select_application_facts(
        ["list_ever_applied", "list_pending_submission", "list_failed", "get_next_actions"],
        rows=rows,
    )
    assert [r["company"] for r in facts["ever_applied"]] == ["甲公司"]
    assert [r["company"] for r in facts["failed"]] == ["甲公司"]
    assert [r["company"] for r in facts["pending_submission"]] == ["乙公司"]
    answer = render_application_facts(facts)
    assert "https://jobs.example/a" not in answer  # 失败企业段不额外展示官网
    assert "乙公司" in answer and "官网链接：未记录" in answer
    assert "丙公司" not in answer


def test_multi_source_execution_filters_failed_experience_and_redacts_contact(monkeypatch):
    import applications_store
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_PARALLEL_ROUTES", "0")
    monkeypatch.setattr(applications_store, "list_applications", lambda: [{
        "日期": "2026-08-20", "公司": "甲公司", "岗位": "AI 工程师",
        "来源": "官网", "地点": "上海", "当前状态": "已拒绝",
        "下一步动作": "复盘", "备注": "08-01 已投递 · 08-20 已拒绝",
    }])
    monkeypatch.setattr(applications_store, "list_experiences", lambda: [{
        "company": "甲公司", "position": "AI 工程师", "stage": "技术面",
        "date": "2026-08-20", "summary": "RAG 评估不足，联系人微信 wx_test_1234",
        "path": "knowledge_base/experience_posts/a.md",
    }])

    calls = []

    def fake_retrieve(question, top_k, **kwargs):
        calls.append(kwargs)
        st = next(iter(kwargs.get("source_types") or {"resource"}))
        doc = f"{st} 资料说明 RAG 评估需要离线测试集与 groundedness 指标。"
        return {"in_kb": True, "chunks": [doc], "docs": [doc],
                "metas": [{"source": "guide.md", "source_type": st}],
                "matched_by": "vector", "sources": ["guide.md"]}

    plan = plan_query(
        "目前我失败企业中，根据我的经验总结说明能力缺口，结合参考资料解释技术概念"
    )
    out = execute_plan(plan, "问题", 5, retrieve=fake_retrieve)
    assert "甲公司" in out.deterministic_answer
    assert any(item.route == "application_experience" for item in out.evidence)
    assert any(item.route == "reference_kb" for item in out.evidence)
    assert all("wx_test_1234" not in item.text for item in out.evidence)
    assert any("[已脱敏账号]" in item.text for item in out.evidence)
    assert any(kwargs.get("exclude_source_types") for kwargs in calls)


def test_dependent_retrieval_binds_state_entities_and_filters_other_company_experience(monkeypatch):
    import applications_store
    from rag_multi_source import execute_plan, multi_source_messages
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_PARALLEL_ROUTES", "0")
    monkeypatch.setattr(applications_store, "list_applications", lambda: [{
        "日期": "2026-08-21", "公司": "华为计算产品线", "岗位": "AI应用工程师",
        "来源": "官网", "地点": "南京", "当前状态": "准备投递",
        "下一步动作": "定稿简历", "备注": "08-21 准备投递",
    }])
    monkeypatch.setattr(applications_store, "list_experiences", lambda: [
        {"company": "无关公司", "position": "后端", "stage": "面试",
         "date": "2026-08-20", "summary": "无关经验", "path": "other.md"},
        {"company": "华为计算产品线", "position": "AI应用工程师", "stage": "投递",
         "date": "2026-08-21", "summary": "重视基础、项目成果和限时代码",
         "path": "huawei.md"},
    ])
    seen_queries = []

    def fake_retrieve(query, top_k, **kwargs):
        seen_queries.append(query)
        if kwargs.get("source_types") == {"experience"}:
            chunks = [
                "无关公司 后端面试需要准备 Java",
                "华为计算产品线 AI应用工程师重视基础和项目成果",
            ]
            metas = [
                {"source": "other.md", "source_type": "experience"},
                {"source": "huawei.md", "source_type": "experience"},
            ]
        else:
            chunks = ["AI 应用岗位投递前应核对项目证据与基础能力"]
            metas = [{"source": "guide.md", "source_type": "resource"}]
        return {"in_kb": True, "chunks": chunks, "docs": chunks, "metas": metas,
                "matched_by": "hybrid", "sources": [metas[0]["source"]]}

    plan = plan_query("我最近要投递哪些企业？投递该企业有什么注意事项？")
    out = execute_plan(plan, "原问题", 5, retrieve=fake_retrieve)
    assert {key: out.resolved_entities[key] for key in (
        "applications", "companies", "positions"
    )} == {
        "applications": [{"company": "华为计算产品线", "position": "AI应用工程师",
                          "status": "准备投递"}],
        "companies": ["华为计算产品线"], "positions": ["AI应用工程师"]}
    assert all("华为计算产品线" in query and "AI应用工程师" in query
               for query in seen_queries)
    assert any(item.source_id == "huawei.md" for item in out.evidence)
    assert all(item.source_id != "other.md" for item in out.evidence)
    assert sum(item.source_type == "experience" for item in out.evidence) == 1
    system = multi_source_messages("原问题", plan, out)[0]["content"]
    assert "华为计算产品线｜AI应用工程师" in system
    assert "不得使用“上述企业”“该岗位”等代词替代" in system


def test_local_embedding_routes_are_serial_by_default(monkeypatch):
    import threading
    import time

    import rag_tools
    from rag_multi_source import execute_plan
    from rag_query_plan import QueryPlan, RouteSpec

    monkeypatch.delenv("RAG_PARALLEL_ROUTES", raising=False)
    monkeypatch.setattr(rag_tools, "get_embedding_config", lambda: {"provider": "local"})
    plan = QueryPlan(
        planner_mode="rule",
        routes=[
            RouteSpec("project_memory", "search", "项目"),
            RouteSpec("reference_kb", "search", "资料"),
        ],
        entities={},
        answer_sections=[],
        reason="test",
    )
    lock = threading.Lock()
    active = 0
    peak = 0
    calls = 0

    def fake_retrieve(question, top_k, **kwargs):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"in_kb": False, "chunks": [], "docs": [], "metas": []}

    execute_plan(plan, "问题", 5, retrieve=fake_retrieve)
    assert calls == 2
    assert peak == 1


def test_state_only_stream_is_deterministic_and_does_not_call_vector_or_llm(monkeypatch):
    import applications_store
    import rag_gate
    import rag_query_plan

    structured_plan = rag_query_plan.plan_structured_read(
        "application_state", "list_pending_submission",
        subquery="读取待投递记录",
    )
    monkeypatch.setattr(
        rag_query_plan, "plan_query", lambda *_args, **_kwargs: structured_plan,
    )

    monkeypatch.setattr(applications_store, "list_applications", lambda: [{
        "日期": "2026-08-21", "公司": "华为", "岗位": "AI 应用工程师",
        "来源": "官网", "地点": "南京", "当前状态": "准备投递",
        "下一步动作": "定稿简历", "备注": "08-21 准备投递",
    }])
    monkeypatch.setattr(rag_gate, "_retrieve_and_classify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应向量检索")))
    monkeypatch.setattr(rag_gate, "_chat_stream",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用 LLM")))
    events = list(rag_gate.gated_query_stream("我还需要去哪些公司官网投递？"))
    stage_events = [e for e in events if e["type"] == "stage"]
    payload_events = [e for e in events if e["type"] != "stage"]
    assert [e["type"] for e in payload_events] == ["meta", "delta", "done"]
    assert any(e["stage"] == "retrieval" and e["status"] == "completed"
               for e in stage_events)
    assert payload_events[0]["planner_mode"] == "structured"
    assert payload_events[0]["routes"][0]["source"] == "application_state"
    assert "华为" in payload_events[1]["text"] and "官网链接：未记录" in payload_events[1]["text"]


def test_privacy_redactor_removes_contacts_but_keeps_dates_and_companies():
    from rag_multi_source import redact_private_text

    phone = "138" + "0013" + "8000"
    text = f"华为 2026-08-21，手机 {phone}，邮箱 a@example.com，微信: wx_demo_99"
    out = redact_private_text(text)
    assert "华为" in out and "2026-08-21" in out
    assert phone not in out and "a@example.com" not in out and "wx_demo_99" not in out
