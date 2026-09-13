# -*- coding: utf-8 -*-
import json
import os
import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import application_jd_store as jd
import applications_store as apps
import career_multi_agent as agents
import plan_gen
import resume_critic
from rag_api import app


OLD_TABLE = """# 投递追踪

| 日期 | 公司 | 岗位 | 来源 | 地点 | 匹配结论 | 样本定位 | 当前状态 | 下一步动作 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-22 | 华为计算产品线 | AI应用工程师 | 官网 | 上海 | 当前适合投递 | Agent | 准备投递 | 完成官网投递 | 08-22 准备投递 |
"""


def _jd(company: str, position: str) -> str:
    return f"""公司：{company}
岗位名称：{position}
工作地点：上海
岗位职责：负责 RAG、Agent、FastAPI 应用开发和评测。
任职要求：熟悉 Python、向量检索、重排与端到端项目实践。
"""


@pytest.fixture()
def portfolio_env(tmp_path, monkeypatch):
    app_path = tmp_path / "applications.md"
    app_path.write_text(OLD_TABLE, encoding="utf-8")
    monkeypatch.setattr(apps, "APPLICATIONS_PATH", str(app_path))
    monkeypatch.setattr(jd, "SNAPSHOT_DIR", str(tmp_path / "application_jds"))
    monkeypatch.setattr(jd, "PROFILE_PATH", str(tmp_path / "profile.md"))
    (tmp_path / "profile.md").write_text("# profile\nPython / RAG", encoding="utf-8")
    monkeypatch.setattr(jd, "_match_payload", lambda text, title, jd_analysis=None: {
        "status": "当前适合投递", "direction": "Agent 应用工程", "summary": "ok",
        "gap_list": {"技能缺口": ["缺少 RAG 评测实战"],
                     "硬门槛缺口": ["学历信息不足"]},
        "suggestions": ["补一个评测实验"],
        "requirement_analysis": {"skills": ["RAG", "Python"]},
    })
    monkeypatch.setattr(agents, "_safe_profile", lambda: {
        "方向优先级": ["Agent 应用工程"], "熟练技能": ["Python"],
        "技能证据": ["RAG", "FastAPI"],
    })
    monkeypatch.setattr(plan_gen, "retrieve_learning_resources", lambda *a, **k: [])
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: None)
    monkeypatch.setattr(agents, "RESUME_DRAFT_DIR", str(tmp_path / "resume_drafts"))
    monkeypatch.setattr(agents, "PLAN_DRAFT_DIR", str(tmp_path / "unreviewed_plan_drafts"))
    monkeypatch.setenv("CAREER_AGENT_DB_PATH", str(tmp_path / "agent.sqlite3"))
    monkeypatch.setenv("CAREER_AGENT_CRITIC_LLM", "1")
    agents.reset_agent_runtime_for_tests()

    apps.ensure_application_schema()
    first = apps.application_fact_views()[0]
    art1 = jd.prepare_approved_artifacts(
        application_id=first["application_id"], jd_text=_jd("华为计算产品线", "AI应用工程师"),
        company="华为计算产品线", position="AI应用工程师", location="上海",
        expected_hash=jd.jd_content_hash(_jd("华为计算产品线", "AI应用工程师")),
    )
    apps.patch_application(
        first["application_id"], jd_id=art1["jd_id"], jd_version_id=art1["jd_version_id"],
        match_id=art1["match_id"], include_in_plan=True, plan_priority="high",
    )

    second_add = apps.upsert_application(
        "搜狐", "大模型应用开发", "面试中", force_new=True,
        next_action="准备一面", include_in_plan=False,
    )
    art2 = jd.prepare_approved_artifacts(
        application_id=second_add["application_id"], jd_text=_jd("搜狐", "大模型应用开发"),
        company="搜狐", position="大模型应用开发", location="北京",
        expected_hash=jd.jd_content_hash(_jd("搜狐", "大模型应用开发")),
    )
    apps.patch_application(
        second_add["application_id"], jd_id=art2["jd_id"], jd_version_id=art2["jd_version_id"],
        match_id=art2["match_id"], include_in_plan=True, plan_priority="medium",
    )

    saved_plans = []

    def fake_save(content, edited_by_user=False):
        path = tmp_path / f"plan_{len(saved_plans) + 1}{'_user' if edited_by_user else ''}.md"
        path.write_text(content, encoding="utf-8")
        saved_plans.append(path)
        return str(path)

    monkeypatch.setattr(plan_gen, "save_plan", fake_save)
    yield {"tmp": tmp_path, "saved_plans": saved_plans,
           "first_id": first["application_id"], "second_id": second_add["application_id"],
           "first_jd": art1["jd_version_id"], "second_jd": art2["jd_version_id"]}
    agents.reset_agent_runtime_for_tests()


def _valid_plan_llm(messages, **_kwargs):
    system = messages[0]["content"]
    request = json.loads(messages[-1]["content"])
    if "Plan Critic Agent" in system:
        requirements = [item for item in request["review_contract"]["criteria"]
                        if item.get("source") == "user_requirement"]
        first_change_review = bool(requirements and request["review_phase"] == "initial")
        return json.dumps({
            "verdict": "revise" if first_change_review else "pass",
            "criteria": [], "findings": [],
            "unmet_requirement_ids": ([item["requirement_id"] for item in requirements]
                                      if first_change_review else []),
            "evidence_conflicts": [],
            "revision_brief": ["落实用户本轮修改要求"] if first_change_review else [],
            "clarification_questions": [], "summary": "计划评审完成",
        }, ensure_ascii=False)
    context = request["context"]
    tasks = []
    all_gaps = [*(context.get("gaps") or []), *(context.get("profile_goals") or [])]
    for index, gap in enumerate(all_gaps):
        if not gap.get("learnable"):
            continue
        sources = gap.get("sources") or []
        source_type = "application_jd" if sources else "profile_general"
        tasks.append({
            "task_id": f"task_gap_{index}", "title": f"完成 {gap['text']}",
            "objective": gap["text"], "time_window": "Week 1",
            "estimated_hours": 1, "deliverable": f"{gap['text']} 的练习记录",
            "gap_ids": [gap["gap_id"]],
            "application_ids": [x["application_id"] for x in sources],
            "jd_version_ids": [x["jd_version_id"] for x in sources],
            "resource_source_ids": [], "dependency_ids": [],
            "source_type": source_type,
        })
    start = dt.date.fromisoformat(request["start_date"])
    end = dt.date.fromisoformat(request["end_date"])
    schedule = []
    cursor = start
    index = 0
    while cursor <= end:
        task_ids = [tasks[index]["task_id"]] if index < len(tasks) else []
        schedule.append({
            "date": cursor.isoformat(),
            "focus": tasks[index]["title"] if index < len(tasks) else "恢复与机动",
            "core_task_ids": task_ids, "optional_task_ids": [],
            "planned_hours": 1 if task_ids else 0,
        })
        index += 1
        cursor += dt.timedelta(days=1)
    return json.dumps({
        "schema_version": "plan.v2",
        "portfolio_summary": "先完成高优先级岗位动作，再合并多个 JD 的共同能力缺口。",
        "priority_decisions": [{"application_id": context["applications"][0]["application_id"],
                                "reason": "高优先级且临近投递"}],
        "common_gap_tasks": tasks, "role_specific_tasks": [],
        "application_actions": [{
            "application_id": item["application_id"],
            "action": item.get("next_action") or "完成投递下一步",
            "time_window": "本周", "deliverable": "更新投递状态",
        } for item in context["applications"]],
        "decision_warnings": [{"text": "学历信息只能核验，不能通过学习改变",
                               "gap_ids": [x["gap_id"] for x in context["gaps"]
                                           if x["kind"] == "hard_constraint"]}],
        "schedule": schedule,
        "applied_requirement_ids": [item["requirement_id"]
                                    for item in request.get("user_requirements") or []],
        "change_summary": ["按冻结范围生成完整日计划"],
    }, ensure_ascii=False)


def _valid_resume_llm(messages, **_kwargs):
    system = messages[0]["content"]
    request = json.loads(messages[-1]["content"])
    if "Resume Critic Agent" in system:
        requirements = [item for item in request["review_contract"]["criteria"]
                        if item.get("source") == "user_requirement"]
        first_change_review = bool(requirements and request["review_phase"] == "initial")
        return json.dumps({
            "verdict": "revise" if first_change_review else "pass",
            "criteria": [], "findings": [],
            "unmet_requirement_ids": ([item["requirement_id"] for item in requirements]
                                      if first_change_review else []),
            "evidence_conflicts": [],
            "revision_brief": ["落实用户本轮修改要求"] if first_change_review else [],
            "clarification_questions": [], "summary": "简历评审完成",
        }, ensure_ascii=False)
    refs = [item["evidence_ref"] for item in request["evidence_bundle"]]
    profile_ref = next((ref for ref in refs if ref.startswith("profile:")), refs[0])
    project_ref = next((ref for ref in refs if ref == "project:offerclaw"), profile_ref)
    target = request["target"]
    artifact_scope = request.get("output_scope", "full_resume")
    current = request.get("current_resume_spec") or {}
    old_blocks = [block for section in current.get("sections") or []
                  for block in section.get("blocks") or []]
    ids = [block["block_id"] for block in old_blocks]
    claim_ids = [block.get("claim_id") for block in old_blocks]
    sections = [
        {"section_id": "project", "title": "项目经历", "blocks": [{
            "block_id": ids[0] if ids else "project_offerclaw",
            "claim_id": (claim_ids[0] if claim_ids else "") or "claim_project_offerclaw",
            "kind": "bullet", "text": "使用 Python、FastAPI、RAG 与重排构建 OfferClaw，并建立检索评测流程。",
            "evidence_refs": [project_ref],
        }]},
    ] if artifact_scope == "project_section" else [
            {"section_id": "summary", "title": "求职摘要", "blocks": [{
                "block_id": ids[0] if ids else "summary_profile",
                "claim_id": (claim_ids[0] if claim_ids else "") or "claim_summary_profile",
                "kind": "paragraph", "text": "具备 Python 与 RAG 项目实践，关注可复现评测和工程交付。",
                "evidence_refs": [profile_ref, project_ref],
            }]},
            {"section_id": "skills", "title": "技能", "blocks": [{
                "block_id": ids[1] if len(ids) > 1 else "skills_profile",
                "claim_id": (claim_ids[1] if len(claim_ids) > 1 else "") or "claim_skills_profile",
                "kind": "bullet", "text": "Python、RAG、FastAPI",
                "evidence_refs": [profile_ref, project_ref],
            }]},
            {"section_id": "project", "title": "项目经历", "blocks": [{
                "block_id": ids[2] if len(ids) > 2 else "project_offerclaw",
                "claim_id": (claim_ids[2] if len(claim_ids) > 2 else "") or "claim_project_offerclaw",
                "kind": "bullet", "text": "使用 Python、FastAPI、RAG 与重排构建 OfferClaw，并建立检索评测流程。",
                "evidence_refs": [project_ref],
            }]},
        ]
    return json.dumps({
        "schema_version": "resume.v2",
        "artifact_scope": artifact_scope,
        "target_application_id": target["application_id"],
        "jd_version_id": target["jd_version_id"],
        "headline": "AI 应用工程候选人",
        "sections": sections,
        "applied_requirement_ids": [item["requirement_id"]
                                    for item in request.get("user_requirements") or []],
        "change_summary": ["按目标 JD 调整摘要与项目经历"],
    }, ensure_ascii=False)


def test_scope_defaults_to_all_confirmed_current_targets(portfolio_env):
    out = agents.preview_portfolio_scope(allow_llm=False)
    assert {x["application_id"] for x in out["included"]} == {
        portfolio_env["first_id"], portfolio_env["second_id"]}
    assert out["resolver_mode"] == "rule"
    assert out["estimated_llm_calls"]["total"] == 1
    assert not Path(os.environ["CAREER_AGENT_DB_PATH"]).exists()


def test_scope_collapses_an_exact_active_jd_duplicate(portfolio_env):
    duplicate = apps.upsert_application(
        "华为计算产品线", "AI应用工程师", "准备投递", force_new=True,
        include_in_plan=False, plan_priority="low",
    )
    duplicate_artifact = jd.prepare_approved_artifacts(
        application_id=duplicate["application_id"],
        jd_text=_jd("华为计算产品线", "AI应用工程师"),
        company="华为计算产品线", position="AI应用工程师", location="上海",
        expected_hash=jd.jd_content_hash(_jd("华为计算产品线", "AI应用工程师")),
    )
    apps.patch_application(
        duplicate["application_id"], jd_id=duplicate_artifact["jd_id"],
        jd_version_id=duplicate_artifact["jd_version_id"],
        match_id=duplicate_artifact["match_id"], include_in_plan=True,
        plan_priority="low",
    )

    scope = agents.preview_portfolio_scope(allow_llm=False)

    assert {item["application_id"] for item in scope["included"]} == {
        portfolio_env["first_id"], portfolio_env["second_id"],
    }
    collapsed = next(item for item in scope["excluded"]
                     if item["application_id"] == duplicate["application_id"])
    assert collapsed["duplicate_of"] == portfolio_env["first_id"]
    assert "重复保存" in collapsed["reason"]


def test_natural_language_scope_uses_company_and_direction_together(portfolio_env):
    out = agents.preview_portfolio_scope(
        instruction="只考虑华为的 AI 岗位", allow_llm=False)
    assert [x["application_id"] for x in out["included"]] == [portfolio_env["first_id"]]
    assert any(x["application_id"] == portfolio_env["second_id"] for x in out["excluded"])


def test_scope_becomes_stale_after_application_change(portfolio_env):
    out = agents.preview_portfolio_scope(allow_llm=False)
    apps.patch_application(portfolio_env["first_id"], status="已投递")
    with pytest.raises(RuntimeError, match="已变化"):
        agents.get_scope_snapshot(out["scope_snapshot_id"])


def test_plan_agent_interrupts_before_write_and_approve_is_idempotent(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_call_llm", _valid_plan_llm)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=scope["scope_snapshot_id"],
        start_date="2026-08-22", end_date="2026-09-22",
    )
    assert started["status"] == "waiting_approval"
    assert started["artifact"]["status"] == "ready_for_approval"
    assert started["budget"]["used_llm_calls"] == 2
    assert started["review_report"]["review_status"] == "completed"
    metric_ids = [item["call_id"] for item in started["metrics"]]
    assert len(metric_ids) == len(set(metric_ids))
    from plan_daily import parse_plan_days
    assert len(parse_plan_days(started["artifact"]["content_md"])["days"]) == 32
    assert portfolio_env["saved_plans"] == []

    approved = agents.resume_agent_flow(started["thread_id"], decision="approve")
    assert approved["status"] == "completed"
    assert Path(approved["artifact"]["saved_path"]).exists()
    assert "OFFERCLAW_PORTFOLIO_SCOPE=" in Path(
        approved["artifact"]["saved_path"]).read_text(encoding="utf-8")
    again = agents.resume_agent_flow(started["thread_id"], decision="approve")
    assert again["artifact"]["saved_path"] == approved["artifact"]["saved_path"]
    assert len(portfolio_env["saved_plans"]) == 1


def test_plan_change_request_runs_frozen_review_cycle(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_call_llm", _valid_plan_llm)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(task="portfolio_plan",
                                      scope_snapshot_id=scope["scope_snapshot_id"])
    edited = agents.resume_agent_flow(
        started["thread_id"], decision="request_changes",
        change_request="周末只做复盘，并保留机动时间。",
        artifact_revision=started["artifact_revision"],
    )
    assert edited["status"] == "waiting_approval"
    assert edited["artifact"]["status"] == "ready_for_approval"
    assert edited["budget"]["used_llm_calls"] == 3
    assert len(edited["requirements"]) == 1
    assert any(item["source"] == "user_requirement"
               for item in edited["review_contract"]["criteria"])
    assert portfolio_env["saved_plans"] == []
    approved = agents.resume_agent_flow(started["thread_id"], decision="approve")
    assert Path(approved["artifact"]["saved_path"]).exists()


def test_sqlite_checkpoint_restores_waiting_approval_after_restart(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_call_llm", _valid_plan_llm)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(task="portfolio_plan",
                                      scope_snapshot_id=scope["scope_snapshot_id"])
    thread_id = started["thread_id"]
    agents.reset_agent_runtime_for_tests()
    restored = agents.get_agent_flow(thread_id)
    assert restored["status"] == "waiting_approval"
    assert restored["artifact"]["artifact_id"] == started["artifact"]["artifact_id"]
    rejected = agents.resume_agent_flow(thread_id, decision="reject")
    assert rejected["status"] == "completed"
    assert rejected["artifact"]["saved_path"] == ""
    assert portfolio_env["saved_plans"] == []


def test_approval_rechecks_scope_freshness(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_call_llm", _valid_plan_llm)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=scope["scope_snapshot_id"])
    apps.patch_application(portfolio_env["first_id"], status="已投递")
    stale = agents.resume_agent_flow(
        started["thread_id"], decision="approve",
        artifact_revision=started["artifact_revision"])
    assert stale["status"] == "waiting_approval"
    assert stale["artifact"]["status"] == "blocked"
    assert "context_stale" in stale["artifact"]["validation"]
    assert portfolio_env["saved_plans"] == []


def test_resume_approval_reloads_changed_active_jd_and_reruns_review(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_call_llm", _valid_resume_llm)
    started = agents.start_agent_flow(
        task="resume", application_id=portfolio_env["first_id"],
        jd_version_id=portfolio_env["first_jd"],
    )
    changed_jd = _jd("华为计算产品线", "AI应用工程师") + "\n优先要求：具备离线评测集治理经验。\n"
    newer = jd.prepare_approved_artifacts(
        application_id=portfolio_env["first_id"], jd_text=changed_jd,
        company="华为计算产品线", position="AI应用工程师", location="上海",
        expected_hash=jd.jd_content_hash(changed_jd),
    )
    apps.patch_application(
        portfolio_env["first_id"], jd_id=newer["jd_id"],
        jd_version_id=newer["jd_version_id"], match_id=newer["match_id"],
    )

    refreshed = agents.resume_agent_flow(
        started["thread_id"], decision="approve",
        artifact_revision=started["artifact_revision"],
    )
    assert refreshed["status"] == "waiting_approval"
    assert refreshed["artifact"]["status"] == "ready_for_approval"
    assert refreshed["artifact"]["artifact_revision"] > started["artifact_revision"]
    assert refreshed["artifact"]["source_refs"][0]["jd_version_id"] == newer["jd_version_id"]
    assert refreshed["budget"]["used_llm_calls"] == 3

    approved = agents.resume_agent_flow(
        started["thread_id"], decision="approve",
        artifact_revision=refreshed["artifact_revision"],
    )
    assert approved["artifact"]["status"] == "approved"


def test_resume_writer_and_critic_use_at_most_two_calls(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_llm_config", lambda: ("test-key", "TEST_KEY"))
    monkeypatch.setattr(agents, "_call_llm", _valid_resume_llm)
    started = agents.start_agent_flow(
        task="resume", application_id=portfolio_env["first_id"],
        jd_version_id=portfolio_env["first_jd"],
    )
    assert started["status"] == "waiting_approval"
    assert started["budget"]["used_llm_calls"] == 2
    assert started["artifact"]["status"] == "ready_for_approval"
    assert started["artifact"]["validation"]["critic"]["review_status"] == "completed"
    assert "命中分析" not in started["artifact"]["content_md"]
    assert not (portfolio_env["tmp"] / "resume_drafts").exists()
    approved = agents.resume_agent_flow(started["thread_id"], decision="approve")
    assert Path(approved["artifact"]["saved_path"]).exists()


def test_generic_project_section_uses_resume_critic_and_waits_before_write(
        portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_llm_config", lambda: ("test-key", "TEST_KEY"))
    calls = []

    def fake(messages, **kwargs):
        calls.append(messages[0]["content"])
        return _valid_resume_llm(messages, **kwargs)

    monkeypatch.setattr(agents, "_call_llm", fake)
    started = agents.start_agent_flow(
        task="resume", resume_scope="project_section",
        project_text=("OfferClaw 使用 Python、FastAPI 和 RAG 实现求职工作流，"
                      "包含检索评测、状态追踪与人工审批。" * 3),
        project_name="OfferClaw",
    )

    assert started["status"] == "waiting_approval"
    assert started["resume_scope"] == "project_section"
    assert started["artifact"]["kind"] == "resume_project"
    assert started["artifact"]["saved_path"] == ""
    assert started["artifact"]["status"] == "ready_for_approval"
    assert sum("Resume Critic Agent" in value for value in calls) == 1
    assert len(started["artifact"]["content_md"]) >= 40

    approved = agents.resume_agent_flow(
        started["thread_id"], decision="approve",
        artifact_revision=started["artifact_revision"],
    )
    assert approved["artifact"]["status"] == "approved"
    assert Path(approved["artifact"]["saved_path"]).exists()
    assert "projects" in Path(approved["artifact"]["saved_path"]).parts

    full_context = agents._resume_context(
        portfolio_env["first_id"], portfolio_env["first_jd"],
    )
    assert any(
        item["evidence_ref"].startswith("project_section:")
        for item in full_context["project_sources"]
    )


def test_resume_nodes_reuse_preloaded_jd_analysis(monkeypatch):
    import resume_builder

    keywords = [{
        "canonical_name": "Python", "surface_forms": ["Python"],
        "importance": 0.9,
    }]
    context = {
        "application": {
            "application_id": "app_test", "jd_version_id": "jdv_test",
            "company": "示例公司", "position": "后端工程师",
        },
        "jd": {"jd_text": "任职要求：熟悉 Python"},
        "match": {}, "profile": {"熟练技能": ["Python"]},
        "project_evidence": "OfferClaw 使用 Python、RAG 与 FastAPI。",
        "memory_context": {}, "jd_analysis": {"keywords": keywords},
    }
    loads = []
    monkeypatch.setattr(
        agents, "_resume_context",
        lambda *_args: loads.append(1) or context,
    )
    loaded = agents._load_context_node({
        "agent_task": "resume", "application_id": "app_test",
        "jd_version_id": "jdv_test",
    })
    assert loaded["agent_context"] is context
    assert len(loads) == 1

    monkeypatch.setattr(
        agents, "_resume_context",
        lambda *_args: (_ for _ in ()).throw(AssertionError("context reloaded")),
    )
    monkeypatch.setattr(resume_builder, "build_resume_markdown", lambda *_a, **_k: {
        "resume_md": "## 基础简历\n已有内容",
    })
    monkeypatch.setattr(agents, "_call_llm", _valid_resume_llm)
    monkeypatch.setattr(jd, "profile_fingerprint", lambda: "profile_hash")
    writer = agents._resume_writer_node({
        "application_id": "app_test", "jd_version_id": "jdv_test",
        "agent_context": context,
        "agent_budget": {"used_llm_calls": 0, "max_llm_calls": 4},
        "agent_metrics": [],
    })

    seen = []
    monkeypatch.setattr(resume_critic, "critic_report", lambda *_a, **kwargs: (
        seen.append(kwargs.get("jd_keywords")) or {
            "keyword_coverage": {"coverage": 1.0},
            "fabrication_flags": [], "semantic_flags": None, "verdict": "pass",
        }
    ))
    checked = agents._resume_hard_validator_node({
        "agent_artifact": writer["agent_artifact"], "artifact_spec": writer["artifact_spec"],
        "agent_context": context, "agent_metrics": [],
    })
    assert seen == [keywords]
    assert checked["hard_validation"]["valid"] is True


def test_hard_resume_fabrication_still_runs_critic(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_llm_config", lambda: ("test-key", "TEST_KEY"))
    calls = []

    def fake(messages, **kwargs):
        calls.append(messages[0]["content"])
        raw = _valid_resume_llm(messages, **kwargs)
        if "Resume Critic Agent" not in messages[0]["content"]:
            value = json.loads(raw)
            value["sections"][2]["blocks"][0]["text"] = "使用 React 建设了线上服务并服务上千用户。"
            value["sections"][2]["blocks"][0]["evidence_refs"] = [
                value["sections"][0]["blocks"][0]["evidence_refs"][0]]
            return json.dumps(value, ensure_ascii=False)
        return raw

    monkeypatch.setattr(agents, "_call_llm", fake)
    started = agents.start_agent_flow(
        task="resume", application_id=portfolio_env["first_id"],
        jd_version_id=portfolio_env["first_jd"],
    )
    assert started["artifact"]["status"] == "needs_revision"
    assert started["budget"]["used_llm_calls"] == 4
    assert sum("Resume Critic Agent" in value for value in calls) == 2
    assert "approve" not in started["interrupt"]["options"]
    assert started["artifact"]["validation"]["critic"]["review_status"] == "completed"
    still_waiting = agents.resume_agent_flow(started["thread_id"], decision="approve")
    assert still_waiting["status"] == "waiting_approval"


def test_parseable_plan_hard_errors_still_run_critic(portfolio_env, monkeypatch):
    critic_calls = []

    def fake(messages, **kwargs):
        raw = _valid_plan_llm(messages, **kwargs)
        if "Plan Critic Agent" in messages[0]["content"]:
            critic_calls.append(1)
            return raw
        value = json.loads(raw)
        value["schedule"] = []
        return json.dumps(value, ensure_ascii=False)

    monkeypatch.setattr(agents, "_call_llm", fake)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=scope["scope_snapshot_id"])
    codes = {item["code"] for item in
             started["artifact"]["validation"]["hard_validator"]["findings"]}
    assert started["artifact"]["status"] == "needs_revision"
    assert started["budget"]["used_llm_calls"] == 4
    assert len(critic_calls) == 2
    assert "incomplete_calendar" in codes


def test_unknown_plan_priority_id_is_reviewed_instead_of_becoming_generation_error(
        portfolio_env, monkeypatch):
    critic_calls = []

    def fake(messages, **kwargs):
        raw = _valid_plan_llm(messages, **kwargs)
        if "Plan Critic Agent" in messages[0]["content"]:
            critic_calls.append(1)
            return raw
        value = json.loads(raw)
        value["priority_decisions"][0]["application_id"] = "app_unknown"
        return json.dumps(value, ensure_ascii=False)

    monkeypatch.setattr(agents, "_call_llm", fake)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=scope["scope_snapshot_id"])
    codes = {item["code"] for item in
             started["artifact"]["validation"]["hard_validator"]["findings"]}

    assert started["artifact"]["status"] == "needs_revision"
    assert started["budget"]["used_llm_calls"] == 4
    assert len(critic_calls) == 2
    assert "unknown_priority_application" in codes


def test_failed_author_call_consumes_reserved_budget(portfolio_env, monkeypatch):
    monkeypatch.setattr(
        agents, "_call_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    scope = agents.preview_portfolio_scope(allow_llm=False)
    result = agents.start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=scope["scope_snapshot_id"])
    assert result["artifact"]["status"] == "generation_error"
    assert result["budget"]["used_llm_calls"] == 1
    author_metric = next(item for item in result["metrics"]
                         if item["node"] == "portfolio_plan_agent")
    assert author_metric["llm_calls"] == 1


def test_critic_unavailable_cannot_be_presented_as_pass(portfolio_env, monkeypatch):
    def fake(messages, **kwargs):
        if "Plan Critic Agent" in messages[0]["content"]:
            return "provider returned non-json review"
        return _valid_plan_llm(messages, **kwargs)

    monkeypatch.setattr(agents, "_call_llm", fake)
    scope = agents.preview_portfolio_scope(allow_llm=False)
    started = agents.start_agent_flow(
        task="portfolio_plan", scope_snapshot_id=scope["scope_snapshot_id"])
    assert started["artifact"]["status"] == "review_unavailable"
    assert started["review_report"]["review_status"] == "unavailable"
    assert started["review_contract"]["criteria"]
    assert started["review_report"]["review_contract"] == started["review_contract"]
    assert "approve" not in started["interrupt"]["options"]
    assert "save_unreviewed_draft" in started["interrupt"]["options"]
    saved = agents.resume_agent_flow(
        started["thread_id"], decision="save_unreviewed_draft",
        artifact_revision=started["artifact_revision"])
    assert saved["artifact"]["status"] == "saved_unreviewed"
    assert Path(saved["artifact"]["saved_path"]).exists()
    assert portfolio_env["saved_plans"] == []


def test_resume_change_request_freezes_contract_for_final_review(portfolio_env, monkeypatch):
    critic_contracts = []

    def fake(messages, **kwargs):
        if "Resume Critic Agent" in messages[0]["content"]:
            critic_contracts.append(json.loads(messages[-1]["content"])[
                "review_contract"]["contract_id"])
        return _valid_resume_llm(messages, **kwargs)

    monkeypatch.setattr(agents, "_call_llm", fake)
    started = agents.start_agent_flow(
        task="resume", application_id=portfolio_env["first_id"],
        jd_version_id=portfolio_env["first_jd"])
    changed = agents.resume_agent_flow(
        started["thread_id"], decision="request_changes",
        change_request="项目经历第一条优先说明检索评测。",
        artifact_revision=started["artifact_revision"])
    assert changed["artifact"]["status"] == "ready_for_approval"
    assert changed["budget"]["used_llm_calls"] == 3
    assert critic_contracts[-2] == critic_contracts[-1]
    with pytest.raises(RuntimeError, match="草稿版本已变化"):
        agents.resume_agent_flow(
            started["thread_id"], decision="approve",
            artifact_revision=started["artifact_revision"])


def test_agent_context_redacts_irrelevant_contacts():
    phone = "138" + "0013" + "8000"
    text = f"联系邮箱 me@example.com，手机 {phone}，微信: offer_claw_123"
    redacted = agents._redact_text(text)
    assert "me@example.com" not in redacted
    assert phone not in redacted
    assert "offer_claw_123" not in redacted


def test_agent_api_scope_and_sse_start(portfolio_env, monkeypatch):
    monkeypatch.setattr(agents, "_call_llm", _valid_plan_llm)
    client = TestClient(app)
    preview = client.post("/api/agent/plan/scopes/preview", json={
        "instruction": "", "application_ids": [], "include_profile_goals": True,
    })
    assert preview.status_code == 200
    started = client.post("/api/agent/flows/start", json={
        "task": "portfolio_plan",
        "scope_snapshot_id": preview.json()["scope_snapshot_id"],
        "start_date": "2026-08-22", "end_date": "2026-09-22",
    })
    assert started.status_code == 200
    events = [json.loads(line[6:]) for line in started.text.splitlines()
              if line.startswith("data: ")]
    assert {event["type"] for event in events} >= {
        "meta", "scope", "node", "artifact", "interrupt", "done"}
    done = next(x for x in events if x["type"] == "done")
    assert done["status"] == "waiting_approval"
    artifact = next(x for x in events if x["type"] == "artifact")["artifact"]
    changed = client.post(f"/api/agent/flows/{done['thread_id']}/resume", json={
        "decision": "request_changes",
        "change_request": "把周末安排为复盘和机动时间。",
        "artifact_revision": artifact["artifact_revision"],
    })
    assert changed.status_code == 200
    assert changed.json()["artifact"]["status"] == "ready_for_approval"
    assert changed.json()["review_contract"]["criteria"][-1]["source"] == "user_requirement"
