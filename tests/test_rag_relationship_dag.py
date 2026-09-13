import json
from pathlib import Path
from types import SimpleNamespace


FIT_ROUTES = {
    ("application_state", "list_current"),
    ("application_jd", "search_bound_jd"),
    ("application_jd", "get_match_snapshot"),
    ("project_memory", "rank_for_application"),
}


def _fixture():
    path = Path(__file__).parent / "fixtures" / "rag_relationship_dag_eval_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_project_fit_questions_build_locked_relationship_dag():
    from rag_query_plan import _plan_validation_errors, rule_plan_query

    for question in _fixture()["positive"]:
        plan = rule_plan_query(question)
        pairs = {(route.source, route.operation) for route in plan.routes}
        assert pairs == FIT_ROUTES, question
        assert plan.intent_frame.filters["relation_intent"] == "application_project_fit"
        assert "project_fit" in plan.intent_frame.answer_objects
        assert all(route.locked and route.required for route in plan.routes)
        state = next(route for route in plan.routes if route.source == "application_state")
        project = next(route for route in plan.routes if route.source == "project_memory")
        assert state.source_role == "filter_source"
        assert state.filters["strict_entity_filter"] is True
        assert project.source_role == "answer_source"
        assert project.depends_on == ("application_state", "application_jd")
        assert _plan_validation_errors(plan) == []


def test_nearby_questions_do_not_trigger_project_fit_workflow():
    from rag_query_plan import rule_plan_query

    for question in _fixture()["negative"]:
        plan = rule_plan_query(question)
        assert not any(route.operation == "rank_for_application" for route in plan.routes), question
        assert "application_project_fit" not in plan.intent_frame.signals, question


def test_application_entity_filter_prefers_company_and_fails_closed(monkeypatch):
    import applications_store as store

    views = [
        {"application_id": "app_huawei", "company": "华为计算产品线",
         "position": "AI应用工程师", "status": "准备投递", "date": "2026-08-21",
         "next_action": "准备面试", "jd_version_id": "jdv_huawei", "terminal": False,
         "ever_applied": False, "pending_submission": True, "decision_pending": False,
         "failed": False},
        {"application_id": "app_sohu", "company": "搜狐", "position": "AI产品实习生",
         "status": "准备投递", "date": "2026-08-22", "next_action": "官网投递",
         "jd_version_id": "jdv_sohu", "terminal": False, "ever_applied": False,
         "pending_submission": True, "decision_pending": False, "failed": False},
    ]
    monkeypatch.setattr(store, "application_fact_views", lambda rows=None: list(views))
    selected = store.select_application_facts(
        ["list_current"], filters={"entity_query": "投递华为的岗位", "strict_entity_filter": True},
    )
    assert [row["application_id"] for row in selected["current"]] == ["app_huawei"]
    scalar = store.select_application_facts(
        ["list_current"], filters={"company": "华为"},
    )
    assert [row["application_id"] for row in scalar["current"]] == ["app_huawei"]
    missing = store.select_application_facts(
        ["list_current"], filters={"entity_query": "投递不存在公司的岗位", "strict_entity_filter": True},
    )
    assert missing["current"] == []


def test_reference_retrieval_uses_topic_and_compiles_date_to_source_filter(
        monkeypatch, tmp_path):
    import rag_multi_source
    from rag_multi_source import _reference_source_filter, _retrieval_route_query

    kb = tmp_path / "knowledge_base" / "learning_resources"
    kb.mkdir(parents=True)
    (kb / "2026-06-03_transformer.md").write_text("# Transformer", encoding="utf-8")
    (kb / "2026-06-19_transformer.md").write_text("# Other batch", encoding="utf-8")
    pending = tmp_path / "knowledge_base" / "_pending"
    pending.mkdir()
    (pending / "2026-06-03_unapproved.md").write_text("# pending", encoding="utf-8")
    monkeypatch.setattr(rag_multi_source, "BASE_DIR", str(tmp_path))

    route = SimpleNamespace(
        subquery="2026-06-03 Transformer 作用",
        filters={"date": "2026-06-03", "topic": "Transformer 作用"},
        depends_on=(),
    )
    semantic_query = _retrieval_route_query("reference_kb", [route], "原问题", [])
    assert "2026-06-03" not in semantic_query
    assert "Transformer" in semantic_query and "作用" in semantic_query
    assert _reference_source_filter([route]) == {"2026-06-03_transformer.md"}

    missing = SimpleNamespace(filters={"date": "2026-01-01"})
    assert next(iter(_reference_source_filter([missing]))).startswith(
        "__no_approved_kb_source_for_"
    )


def test_direct_experience_query_filters_a_named_company(monkeypatch):
    import applications_store
    from rag_multi_source import _direct_experience_evidence

    monkeypatch.setattr(applications_store, "list_experiences", lambda: [{
        "company": "华为计算产品线", "position": "AI应用工程师",
        "stage": "面试", "date": "2026-08-01", "summary": "华为复盘",
        "path": "huawei.md",
    }, {
        "company": "搜狐", "position": "AI产品",
        "stage": "笔试", "date": "2026-08-02", "summary": "搜狐复盘",
        "path": "sohu.md",
    }])
    items = _direct_experience_evidence(query="华为那条记录的经验是什么")
    assert [item.metadata["company"] for item in items] == ["华为计算产品线"]


def test_execute_project_fit_dag_passes_stable_relationship_evidence(monkeypatch):
    import application_jd_store as jd
    import applications_store as applications
    import project_memory
    from rag_multi_source import execute_plan, multi_source_messages
    from rag_query_plan import rule_plan_query

    target = {
        "application_id": "app_huawei", "company": "华为计算产品线",
        "position": "AI应用工程师", "status": "准备投递", "date": "2026-08-21",
        "next_action": "准备面试", "jd_id": "jd_huawei", "jd_version_id": "jdv_huawei",
        "match_id": "match_huawei", "terminal": False, "ever_applied": False,
        "pending_submission": True, "decision_pending": False, "failed": False,
    }
    monkeypatch.setattr(applications, "select_application_facts", lambda *a, **k: {
        "operations": ["list_current"], "current": [target], "freshness": "2026-08-21", "total": 1,
    })
    monkeypatch.setattr(applications, "render_application_facts", lambda facts: "华为确定性事实")
    monkeypatch.setattr(jd, "search_bound_jds", lambda targets, query="": [{
        **target, "captured_at": "2026-08-20", "source_url": "", "text": "要求 RAG、Python 和效果评估",
        "path": "application_jds/jd_huawei/jdv_huawei.md",
    }])
    monkeypatch.setattr(jd, "load_match_snapshot", lambda *a, **k: {
        "match_id": "match_huawei", "generated_at": "2026-08-20", "status": "中长期可转向",
        "direction": "主方向", "gap_list": {"技能缺口": ["缺少效果评估证据"]},
        "suggestions": ["补充可量化项目证据"], "requirement_analysis": {},
    })
    monkeypatch.setattr(project_memory, "list_project_catalog", lambda **k: [{
        "project_id": "project_offerclaw", "name": "OfferClaw", "summary": "RAG 求职助手",
        "source_path": "docs/RESUME_PROJECT.md", "review_status": "resume_confirmed",
        "approval_label": "已形成正式简历项目段", "resume_usage_label": "已用于简历",
    }])
    monkeypatch.setattr(project_memory, "render_project_record", lambda row: "项目：OfferClaw\n说明：RAG 求职助手")
    monkeypatch.setattr(project_memory, "render_project_evidence", lambda row: (
        "项目：OfferClaw\n说明：RAG 求职助手\n正文：实现混合检索、评测与多源路由"
    ))

    question = "投递华为的岗位，你推荐我使用哪个项目，缺口是什么？"
    plan = rule_plan_query(question)
    execution = execute_plan(
        plan, question, 5,
        lambda *a, **k: {"in_kb": False, "chunks": [], "docs": [], "metas": []},
    )
    types = {item.source_type for item in execution.evidence}
    assert {"application_jd", "application_match", "project_context"} <= types
    edges = execution.resolved_entities["relationship_edges"]
    assert {edge["relation"] for edge in edges} == {
        "bound_to", "evaluated_by", "compare_project_candidate",
    }
    system = multi_source_messages(question, plan, execution)[0]["content"]
    assert "关系感知的项目适配比较" in system
    assert "不得凭通用知识推荐项目" in system
