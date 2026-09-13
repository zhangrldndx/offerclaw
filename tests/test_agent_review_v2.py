# -*- coding: utf-8 -*-
import datetime as dt

from review_protocol import (
    CriticOutput,
    ResumeSpec,
    default_review_contract,
    finalize_review,
    hard_validate_plan,
    hard_validate_resume,
    make_requirement,
    merge_critic_contract,
    normalize_plan_schedule,
    render_resume,
    resume_spec_from_markdown,
)


def _resume_context():
    return {
        "application": {"application_id": "app_1", "jd_version_id": "jdv_1"},
        "profile": {"熟练技能": ["Python"]},
        "project_evidence": "OfferClaw 使用 Python 和 FastAPI。",
        "memory_context": {},
    }


def _resume_spec():
    return {
        "schema_version": "resume.v2",
        "target_application_id": "app_1",
        "jd_version_id": "jdv_1",
        "headline": "AI 应用工程候选人",
        "sections": [
            {"section_id": "summary", "title": "求职摘要", "blocks": [{
                "block_id": "summary_1", "kind": "paragraph",
                "claim_id": "claim_summary_1",
                "text": "熟悉 Python，关注可靠的工程交付。",
                "evidence_refs": ["profile:熟练技能"],
            }]},
            {"section_id": "skills", "title": "技能", "blocks": [{
                "block_id": "skills_1", "kind": "bullet",
                "claim_id": "claim_skills_1",
                "text": "Python、FastAPI",
                "evidence_refs": ["profile:熟练技能", "project:offerclaw"],
            }]},
            {"section_id": "project", "title": "项目经历", "blocks": [{
                "block_id": "project_1", "kind": "bullet",
                "claim_id": "claim_project_1",
                "text": "使用 Python 与 FastAPI 构建 OfferClaw 求职工作流。",
                "evidence_refs": ["project:offerclaw"],
            }]},
        ],
        "applied_requirement_ids": [], "change_summary": [],
    }


def test_resume_hard_validator_requires_traceable_claims_after_manual_edit():
    spec = _resume_spec()
    content = render_resume(spec)
    assert hard_validate_resume(spec, content, _resume_context())["valid"] is True

    edited = content.replace("可靠的工程交付", "拥有十万线上用户")
    reparsed = resume_spec_from_markdown(
        edited, application_id="app_1", jd_version_id="jdv_1",
        previous_spec=spec,
    )
    report = hard_validate_resume(reparsed, edited, _resume_context())
    assert report["valid"] is False
    assert "missing_evidence_ref" in {item["code"] for item in report["findings"]}


def test_resume_manual_edit_preserves_requirement_scope_and_distinct_claims():
    requirement = make_requirement("项目经历优先突出检索评测")
    spec = _resume_spec()
    spec["applied_requirement_ids"] = [requirement["requirement_id"]]
    spec["sections"][2]["blocks"].append({
        "block_id": "project_2", "kind": "bullet",
        "claim_id": "claim_project_2", "text": "Python、FastAPI",
        "evidence_refs": ["project:offerclaw"],
    })
    content = render_resume(spec)

    reparsed = resume_spec_from_markdown(
        content, application_id="app_1", jd_version_id="jdv_1",
        previous_spec=spec,
    )
    blocks = [block for section in reparsed["sections"] for block in section["blocks"]]
    claim_ids = [block["claim_id"] for block in blocks]
    repeated = [block for block in blocks if block["text"] == "Python、FastAPI"]

    assert reparsed["applied_requirement_ids"] == [requirement["requirement_id"]]
    assert len(claim_ids) == len(set(claim_ids))
    assert {block["claim_id"] for block in repeated} == {
        "claim_skills_1", "claim_project_2"}


def test_legacy_risky_term_is_allowed_when_the_claim_cites_matching_evidence():
    spec = _resume_spec()
    spec["sections"][1]["blocks"][0]["text"] = "Python、React、FastAPI"
    context = _resume_context()
    context["profile"]["熟练技能"] = ["Python", "React"]
    report = hard_validate_resume(
        spec, render_resume(spec), context,
        deterministic_report={
            "fabrication_flags": [{"kind": "fabrication", "term": "React"}],
            "keyword_coverage": {"coverage": 1.0},
        },
    )
    assert report["valid"] is True


def test_review_contract_keeps_fixed_and_user_criteria_when_critic_refines_it():
    requirement = make_requirement("项目经历优先突出检索评测")
    contract = default_review_contract("resume", [requirement])
    proposed = [{
        "criterion_id": f"user_{requirement['requirement_id']}",
        "source": "user_requirement", "title": "突出检索评测",
        "acceptance": "项目经历首条说明检索评测方法与结果",
        "blocking": False, "requirement_id": requirement["requirement_id"],
    }]
    merged = merge_critic_contract(contract, proposed)
    by_id = {item["criterion_id"]: item for item in merged["criteria"]}
    assert "truthfulness" in by_id
    assert by_id[f"user_{requirement['requirement_id']}"]["blocking"] is True
    acceptance = by_id[f"user_{requirement['requirement_id']}"]["acceptance"]
    assert requirement["text"] in acceptance
    assert "项目经历首条说明检索评测方法与结果" in acceptance


def test_hard_error_overrides_critic_pass():
    contract = default_review_contract("resume", [])
    output = CriticOutput(
        verdict="pass", summary="语言质量通过", criteria=[], findings=[],
        unmet_requirement_ids=[], evidence_conflicts=[], revision_brief=[],
        clarification_questions=[],
    )
    report = finalize_review(
        kind="resume", artifact_revision=1, output=output,
        base_contract=contract,
        hard_report={"valid": False, "findings": [{
            "finding_id": "hard_1", "code": "unsupported_claim",
            "severity": "error", "message": "存在无证据主张", "artifact_ref": "project_1",
        }], "stats": {}},
    )
    assert report["verdict"] == "revise"
    assert any(item.get("source") == "hard_validator" for item in report["findings"])


def test_critic_blocked_for_clarification_stays_blocked_with_hard_errors():
    contract = default_review_contract("resume", [])
    output = CriticOutput(
        verdict="blocked", summary="需要用户补充可核验数据",
        evidence_conflicts=["量化结果没有来源"],
        clarification_questions=["请提供统计口径和时间范围。"],
    )
    report = finalize_review(
        kind="resume", artifact_revision=1, output=output,
        base_contract=contract,
        hard_report={"valid": False, "findings": [{
            "finding_id": "hard_1", "code": "unsupported_claim",
            "severity": "error", "message": "量化结果缺少证据",
            "artifact_ref": "project_1",
        }], "stats": {}},
    )
    assert report["verdict"] == "blocked"


def test_blocking_critic_finding_overrides_inconsistent_pass():
    contract = default_review_contract("resume", [])
    output = CriticOutput.model_validate({
        "verdict": "pass",
        "findings": [{
            "finding_id": "finding_truth", "criterion_id": "truthfulness",
            "severity": "major", "reason": "项目效果与证据不一致",
            "artifact_ref": "project_1", "evidence_refs": ["project:offerclaw"],
            "fix_instruction": "删除未经证实的效果描述",
        }],
    })
    report = finalize_review(
        kind="resume", artifact_revision=1, output=output,
        base_contract=contract,
        hard_report={"valid": True, "findings": [], "stats": {}},
    )
    assert report["verdict"] == "revise"
    assert report["findings"][0]["blocking"] is True


def test_final_review_cannot_mutate_frozen_contract():
    base = default_review_contract("resume", [])
    initial = finalize_review(
        kind="resume", artifact_revision=1,
        output=CriticOutput.model_validate({
            "verdict": "revise",
            "criteria": [{
                "criterion_id": "jd_rag", "source": "jd", "title": "RAG 针对性",
                "acceptance": "项目自然呈现检索与评测证据", "blocking": True,
            }],
            "revision_brief": ["补充 RAG 项目证据"],
        }),
        base_contract=base,
        hard_report={"valid": True, "findings": [], "stats": {}},
    )
    frozen = initial["review_contract"]
    final = finalize_review(
        kind="resume", artifact_revision=2,
        output=CriticOutput.model_validate({
            "verdict": "pass",
            "criteria": [
                {"criterion_id": "jd_rag", "source": "jd", "title": "已篡改",
                 "acceptance": "临时降低标准", "blocking": False},
                {"criterion_id": "jd_new", "source": "jd", "title": "临时新增",
                 "acceptance": "最终轮新增标准", "blocking": True},
            ],
        }),
        base_contract=frozen,
        hard_report={"valid": True, "findings": [], "stats": {}},
        final_review=True, previous_report=initial,
    )
    assert final["review_contract"] == frozen


def test_plan_validator_rejects_duplicate_ids_and_missing_daily_identity():
    start = dt.date(2026, 9, 14)
    end = start + dt.timedelta(days=13)
    task = {
        "task_id": "task_same", "title": "完成检索评测", "objective": "补齐评测",
        "time_window": "Week 1", "estimated_hours": 1,
        "deliverable": "评测报告", "gap_ids": ["gap_1"],
        "application_ids": ["app_1"], "jd_version_ids": ["jdv_1"],
        "resource_source_ids": [], "source_type": "application_jd",
    }
    data = normalize_plan_schedule({
        "portfolio_summary": "验证重复身份不会被静默合并",
        "common_gap_tasks": [task, dict(task)], "role_specific_tasks": [],
        "application_actions": [{"application_id": "app_1", "action": "完成投递",
                                 "time_window": "本周", "deliverable": "状态记录"}],
        "schedule": [],
    }, start.isoformat(), end.isoformat())
    context = {
        "applications": [{"application_id": "app_1", "jd_version_id": "jdv_1"}],
        "gaps": [{"gap_id": "gap_1", "learnable": True}],
        "profile_goals": [], "resources": [],
    }
    report = hard_validate_plan(data, context, start.isoformat(), end.isoformat())
    codes = {item["code"] for item in report["findings"]}
    assert report["valid"] is False
    assert "duplicate_task_id" in codes


def test_resume_spec_renderer_never_adds_review_metadata():
    parsed = ResumeSpec.model_validate(_resume_spec())
    content = render_resume(parsed.model_dump(mode="json"))
    assert "## 求职摘要" in content
    assert "## 项目经历" in content
    assert "命中分析" not in content
