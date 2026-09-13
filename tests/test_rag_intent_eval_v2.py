# -*- coding: utf-8 -*-

import json
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "rag_intent_eval_v2.json"


def test_contrastive_intent_suite_routes_all_160_queries(monkeypatch):
    """相近词面必须按个人/通用、事实/语义、单源/混合意图区分。"""
    from rag_query_plan import rule_plan_query as plan_query

    # This is the frozen regression contract for the compatibility router.
    # Keep it offline even when a developer's .env.local enables intelligent
    # shadow planning for the running UI.
    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    monkeypatch.setenv("RAG_LLM_PLANNER", "0")
    monkeypatch.setenv("RAG_SCOPED_LLM_PLANNER", "0")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    total = sum(len(family["queries"]) for family in data["families"])
    assert total == 160

    failures = []
    for family in data["families"]:
        expected_pairs = {tuple(pair) for pair in family.get("must_pairs", [])}
        expected_sources = set(family.get("must_sources", []))
        excluded = set(family.get("must_not_sources", []))
        for query in family["queries"]:
            plan = plan_query(query)
            pairs = {(route.source, route.operation) for route in plan.routes}
            sources = {source for source, _operation in pairs}
            missing_pairs = expected_pairs - pairs
            missing_sources = expected_sources - sources
            forbidden = excluded & sources
            if missing_pairs or missing_sources or forbidden:
                failures.append({
                    "family": family["id"], "query": query,
                    "actual": sorted(pairs), "missing_pairs": sorted(missing_pairs),
                    "missing_sources": sorted(missing_sources),
                    "forbidden": sorted(forbidden),
                    "frame": plan.intent_frame.to_dict(),
                })
    assert not failures, json.dumps(failures, ensure_ascii=False, indent=2)


def test_anaphora_uses_only_recent_user_question_context(monkeypatch):
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    monkeypatch.setenv("RAG_SCOPED_LLM_PLANNER", "0")
    plan = plan_query(
        "那这个呢",
        context=["无关的更早问题", "请总结我的 OfferClaw 项目"],
    )
    assert ("project_memory", "search") in {
        (route.source, route.operation) for route in plan.routes
    }
    assert plan.intent_frame.personal_scope in {"artifact", "history"}
    assert "anaphora" in plan.intent_frame.signals


def test_project_inventory_treats_application_as_usage_not_answer_source(monkeypatch):
    """最小对照：中心询问对象决定路由，不能由“投递”单个修饰词决定。"""
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    monkeypatch.setenv("RAG_LLM_PLANNER", "0")
    monkeypatch.setenv("RAG_SCOPED_LLM_PLANNER", "0")
    cases = {
        "我当前可用于投递的开源项目有哪些？": {
            ("project_memory", "list_catalog")},
        "我当前可用的开源项目有哪些？": {
            ("project_memory", "list_catalog")},
        "我有哪些已审批的项目材料？": {
            ("project_memory", "list_approved")},
        "我投了哪些公司？": {
            ("application_state", "list_ever_applied")},
        "我投了哪些开源项目相关岗位？": {
            ("application_state", "list_ever_applied"),
            ("project_memory", "search")},
        "如何把我的项目写进简历？": {
            ("project_memory", "search"), ("resume_rules", "search")},
    }
    for query, expected in cases.items():
        plan = plan_query(query)
        pairs = {(route.source, route.operation) for route in plan.routes}
        assert pairs == expected, (query, sorted(pairs), plan.intent_frame.to_dict())
    inventory = plan_query("我当前可用于投递的开源项目有哪些？")
    assert inventory.intent_frame.answer_object == "project"
    assert inventory.intent_frame.usage_contexts == ["application"]
    assert "application_state" not in {route.source for route in inventory.routes}


def test_project_catalog_is_deterministic_and_preserves_confirmation_status(tmp_path):
    from project_memory import list_project_catalog

    project_dir = tmp_path / "knowledge_base" / "project_context"
    project_dir.mkdir(parents=True)
    (project_dir / "approved.md").write_text(
        '---\ntitle: "Demo 项目现状"\nsource_url: "https://github.com/me/demo"\n'
        'source_type: project_context\nreview_status: approved\n---\n'
        '# Demo\n\n## 一句话定位\n\n一个经过用户确认的开源项目。',
        encoding="utf-8",
    )
    (project_dir / "pending.md").write_text(
        '---\ntitle: "Pending"\nsource_type: project_context\nreview_status: pending\n---\n正文',
        encoding="utf-8",
    )
    resume_dir = tmp_path / "docs"
    resume_dir.mkdir()
    (resume_dir / "RESUME_PROJECT.md").write_text(
        '# OfferClaw 简历项目段\n\n这是一个项目经历。', encoding="utf-8")
    (resume_dir / "project_one_pager.md").write_text(
        '# OfferClaw\n\nGitHub: https://github.com/me/offerclaw', encoding="utf-8")

    catalog = list_project_catalog(root=tmp_path, open_source_only=True)
    assert {item["name"] for item in catalog} == {"Demo", "OfferClaw"}
    assert {item["review_status"] for item in catalog} == {"approved", "resume_confirmed"}
    assert all(item["name"] != "Pending" for item in catalog)
    approved = list_project_catalog(root=tmp_path, approved_only=True)
    assert [item["name"] for item in approved] == ["Demo"]


def test_project_inventory_executes_without_vector_search(monkeypatch):
    import project_memory
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    called = False

    def unexpected_retrieve(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("项目清单不应调用向量检索")

    synthetic_record = {
        "project_id": "project_sample",
        "name": "SampleProject",
        "title": "SampleProject 合成项目",
        "summary": "只用于验证确定性项目目录，不对应真实个人或仓库。",
        "source_url": "https://example.com/sample-project",
        "source_path": "synthetic/project.md",
        "review_status": "approved",
        "approval_label": "合成测试已批准",
        "resume_usage": "not_recorded",
        "resume_usage_label": "合成测试未记录",
        "open_source": True,
        "updated_at": "2026-01-01",
    }
    monkeypatch.setattr(
        project_memory,
        "list_project_catalog",
        lambda **_kwargs: [dict(synthetic_record)],
    )

    plan = plan_query("我当前可用的开源项目有哪些？")
    result = execute_plan(plan, "问题", 5, retrieve=unexpected_retrieve)
    assert not called
    assert result.source_status["project_memory"]["status"] == "ok"
    assert {item.metadata["name"] for item in result.evidence} == {"SampleProject"}
    assert all(item.matched_by == "deterministic_catalog" for item in result.evidence)


def test_history_overview_covers_all_valid_logs_and_excludes_orphans():
    from reflection_memory import history_overview_document

    inventory = {
        "daily": [
            {"id": "log_a", "date_from": "2026-01-01", "source_status": "valid",
             "text": "A", "metadata": {"main_tag": "补项目", "completed": ["完成 A"]}},
            {"id": "log_b", "date_from": "2026-01-02", "source_status": "valid",
             "text": "B", "metadata": {"main_tag": "补技能", "completed": ["完成 B"]}},
        ],
        "reflections": [
            {"id": "refl_ok", "date_from": "2026-01-02", "source_status": "valid",
             "text": "有效复盘", "metadata": {"blockers": ["被 C 阻塞"]}},
            {"id": "refl_bad", "date_from": "2026-01-03", "source_status": "orphaned",
             "text": "不应进入回答的孤立内容", "metadata": {}},
        ],
        "stats": {"orphaned_reflections": 1},
    }
    doc = history_overview_document(inventory)
    assert doc["matched_by"] == "deterministic_history"
    assert doc["metadata"]["source_log_ids"] == ["log_a", "log_b"]
    assert doc["metadata"]["reflection_ids"] == ["refl_ok"]
    assert "完成 A" in doc["text"] and "完成 B" in doc["text"]
    assert "不应进入回答的孤立内容" not in doc["text"]
    assert "1 条孤立复盘" in doc["text"]


def test_vector_route_contract_rejects_cross_domain_collisions(monkeypatch):
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    monkeypatch.setenv("RAG_PARALLEL_ROUTES", "0")
    calls = []

    def fake_retrieve(_query, _top_k, **kwargs):
        calls.append(kwargs)
        chunks = ["私人项目中的 BM25", "策展教程中的 BM25"]
        metas = [
            {"source": "mine.md", "source_type": "project_context", "owner_scope": "personal"},
            {"source": "guide.md", "source_type": "resource", "owner_scope": "curated"},
        ]
        return {"in_kb": True, "chunks": chunks, "docs": chunks, "metas": metas,
                "matched_by": "hybrid", "sources": ["mine.md", "guide.md"]}

    personal = execute_plan(
        plan_query("总结我的 OfferClaw 项目"), "问题", 5, retrieve=fake_retrieve)
    assert {item.source_id for item in personal.evidence} == {"mine.md"}
    assert calls[-1]["metadata_filters"] == {"owner_scope": {"personal"}}

    general = execute_plan(
        plan_query("解释 BM25 原理"), "问题", 5, retrieve=fake_retrieve)
    assert {item.source_id for item in general.evidence} == {"guide.md"}
    assert calls[-1]["metadata_filters"] == {"owner_scope": {"curated"}}


def test_mixed_answer_labels_general_fallback_when_curated_evidence_is_missing(monkeypatch):
    from rag_multi_source import ExecutionResult, EvidenceItem, multi_source_messages
    from rag_query_plan import rule_plan_query as plan_query

    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    plan = plan_query("根据我以前的学习记录解释 BM25 原理")
    execution = ExecutionResult(
        evidence=[EvidenceItem(
            "daily_execution", "daily_log.md", "2026-01-01", "live_execution_fact",
            "deterministic", "完成过一次检索练习", "reflection_memory",
        )],
        source_status={
            "reflection_memory": {"status": "ok", "count": 1},
            "reference_kb": {"status": "no_evidence", "count": 0},
        },
    )
    system = multi_source_messages("问题", plan, execution)[0]["content"]
    assert "通用知识补充（未经知识库验证）" in system
    assert "不得给通用知识伪造证据编号" in system
