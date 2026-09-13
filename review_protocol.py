# -*- coding: utf-8 -*-
"""Shared artifact and review contracts for the v2 Agent workflows.

This module deliberately contains no workflow nodes and performs no LLM calls.
It defines the data passed between author agents, deterministic validators and
critic agents so those responsibilities cannot silently collapse together.
"""
from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


WORKFLOW_VERSION = "v2"
RUBRIC_VERSION = "2026-09-10"


def stable_id(prefix: str, value: Any, size: int = 20) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:size]


class UserRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(pattern=r"^req_[0-9a-f]{20}$")
    text: str = Field(min_length=1, max_length=2000)
    source: Literal["start", "user_change", "remembered_preference"] = "user_change"
    persistent: bool = False


def make_requirement(text: str, *, source: str = "user_change",
                     persistent: bool = False) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        raise ValueError("修改要求不能为空")
    value = UserRequirement(
        requirement_id=stable_id("req_", {"text": cleaned, "source": source}),
        text=cleaned,
        source=source,
        persistent=persistent,
    )
    return value.model_dump(mode="json")


class ResumeBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str = Field(min_length=3, max_length=100)
    claim_id: str = Field(default="", max_length=100)
    kind: Literal["paragraph", "bullet"] = "bullet"
    text: str = Field(min_length=1, max_length=1200)
    evidence_refs: list[str] = Field(default_factory=list, max_length=30)


class ResumeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=1, max_length=120)
    blocks: list[ResumeBlock] = Field(min_length=1, max_length=80)


class ResumeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["resume.v2"] = "resume.v2"
    artifact_scope: Literal["full_resume", "project_section"] = "full_resume"
    target_application_id: str = Field(default="", max_length=120)
    jd_version_id: str = Field(default="", max_length=160)
    headline: str = Field(default="候选人", min_length=1, max_length=160)
    sections: list[ResumeSection] = Field(min_length=1, max_length=20)
    applied_requirement_ids: list[str] = Field(default_factory=list, max_length=100)
    change_summary: list[str] = Field(default_factory=list, max_length=30)


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str = Field(min_length=3, max_length=120)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(default="", max_length=600)
    time_window: str = Field(default="", max_length=120)
    estimated_hours: float = Field(gt=0, le=24)
    deliverable: str = Field(min_length=1, max_length=600)
    gap_ids: list[str] = Field(default_factory=list, max_length=100)
    application_ids: list[str] = Field(default_factory=list, max_length=100)
    jd_version_ids: list[str] = Field(default_factory=list, max_length=100)
    resource_source_ids: list[str] = Field(default_factory=list, max_length=100)
    dependency_ids: list[str] = Field(default_factory=list, max_length=100)
    source_type: Literal["application_jd", "profile_general"] = "application_jd"


class PlanDay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str
    focus: str = Field(default="", max_length=160)
    # Cardinality is a hard business rule, not a parsing rule. Keeping the
    # schema permissive lets the Critic see an otherwise parseable bad plan.
    core_task_ids: list[str] = Field(default_factory=list, max_length=20)
    optional_task_ids: list[str] = Field(default_factory=list, max_length=20)
    planned_hours: float = Field(default=0, ge=0, le=24)

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        dt.date.fromisoformat(value)
        return value


class PlanPriorityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    application_id: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=600)


class PlanApplicationAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    application_id: str = Field(min_length=1, max_length=120)
    action: str = Field(min_length=1, max_length=600)
    time_window: str = Field(default="", max_length=120)
    deliverable: str = Field(min_length=1, max_length=600)


class PlanDecisionWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=800)
    gap_ids: list[str] = Field(default_factory=list, max_length=100)


class PlanSpec(BaseModel):
    """Canonical plan artifact owned by the Plan Agent."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["plan.v2"] = "plan.v2"
    portfolio_summary: str = Field(min_length=1, max_length=2000)
    priority_decisions: list[PlanPriorityDecision] = Field(default_factory=list, max_length=100)
    common_gap_tasks: list[PlanTask] = Field(default_factory=list, max_length=300)
    role_specific_tasks: list[PlanTask] = Field(default_factory=list, max_length=300)
    application_actions: list[PlanApplicationAction] = Field(default_factory=list, max_length=200)
    decision_warnings: list[PlanDecisionWarning] = Field(default_factory=list, max_length=100)
    schedule: list[PlanDay] = Field(default_factory=list, max_length=56)
    applied_requirement_ids: list[str] = Field(default_factory=list, max_length=100)
    change_summary: list[str] = Field(default_factory=list, max_length=30)


class HardFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str
    code: str
    severity: Literal["error", "warning"]
    message: str
    artifact_ref: str = ""


class HardValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    findings: list[HardFinding] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class ReviewCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str
    source: Literal["fixed", "jd", "user_requirement"] = "fixed"
    title: str
    acceptance: str
    blocking: bool = True
    requirement_id: str = ""


class ReviewContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_id: str
    rubric_version: str = RUBRIC_VERSION
    artifact_kind: Literal["resume", "resume_project", "portfolio_plan"]
    criteria: list[ReviewCriterion]
    frozen: bool = True


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str
    criterion_id: str
    severity: Literal["critical", "major", "minor"]
    artifact_ref: str = ""
    reason: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=30)
    fix_instruction: str = ""
    is_regression: bool = False


class CriticOutput(BaseModel):
    """The only output contract accepted from a Critic Agent LLM call."""

    model_config = ConfigDict(extra="forbid")
    verdict: Literal["pass", "pass_with_warnings", "revise", "blocked"]
    criteria: list[ReviewCriterion] = Field(default_factory=list, max_length=40)
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=60)
    unmet_requirement_ids: list[str] = Field(default_factory=list, max_length=100)
    evidence_conflicts: list[str] = Field(default_factory=list, max_length=40)
    revision_brief: list[str] = Field(default_factory=list, max_length=40)
    clarification_questions: list[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(default="", max_length=1200)


def default_review_contract(kind: str, requirements: list[dict[str, Any]],
                            *, previous_id: str = "") -> dict[str, Any]:
    if kind in {"resume", "resume_project"}:
        fixed = (
            ("truthfulness", "事实与证据", "每项候选人主张均有有效证据，不夸大或偷换概念", True),
            ("jd_relevance", "岗位针对性", (
                "绑定 JD 时内容回应其重要要求；通用项目经历不得假装针对不存在的 JD"
                if kind == "resume_project" else
                "内容回应 JD 的重要要求，取舍与目标岗位相关"
            ), True),
            ("language_quality", "表达质量", "语言清楚、具体、无明显重复，不机械堆叠关键词", False),
            ("resume_structure", "简历结构", (
                "正文仅包含一段可直接放入简历的项目经历，不含命中分析或模型说明"
                if kind == "resume_project" else
                "正文为可直接使用的完整简历，不含命中分析或模型说明"
            ), True),
        )
    else:
        fixed = (
            ("scope_alignment", "目标范围", "计划只服务于冻结范围内的岗位、JD 与画像目标", True),
            ("priority_quality", "优先级取舍", "优先级、共同缺口和岗位特有任务的取舍有充分依据", True),
            ("plan_feasibility", "现实可执行性", "任务顺序、投入和交付物在计划周期内可执行", True),
            ("learning_coherence", "学习连贯性", "任务依赖合理，并能回应复盘、阻碍和既有进展", False),
        )
    criteria = [
        ReviewCriterion(criterion_id=cid, source="fixed", title=title,
                        acceptance=acceptance, blocking=blocking).model_dump(mode="json")
        for cid, title, acceptance, blocking in fixed
    ]
    for raw in requirements:
        req = UserRequirement.model_validate(raw)
        criteria.append(ReviewCriterion(
            criterion_id=f"user_{req.requirement_id}", source="user_requirement",
            title="用户明确要求", acceptance=f"准确落实：{req.text}", blocking=True,
            requirement_id=req.requirement_id,
        ).model_dump(mode="json"))
    payload = {"kind": kind, "criteria": criteria, "rubric": RUBRIC_VERSION}
    return ReviewContract(
        contract_id=stable_id("review_contract_", payload),
        artifact_kind=kind,
        criteria=criteria,
    ).model_dump(mode="json")


def merge_critic_contract(base: dict[str, Any], proposed: list[dict[str, Any]]) -> dict[str, Any]:
    """Allow the Critic to clarify criteria without deleting or weakening them."""
    contract = ReviewContract.model_validate(base)
    by_id = {item.criterion_id: item for item in contract.criteria}
    for raw in proposed or []:
        try:
            item = ReviewCriterion.model_validate(raw)
        except Exception:
            continue
        original = by_id.get(item.criterion_id)
        if not original:
            # The first Critic pass may add JD-derived criteria, but it may not
            # invent extra user requirements or replace fixed policy.
            if item.source == "jd" and item.criterion_id.startswith("jd_"):
                item.requirement_id = ""
                by_id[item.criterion_id] = item
            continue
        # Fixed policy text is immutable. A user criterion always retains the
        # original requirement and may only gain a concrete acceptance clause.
        if original.source == "user_requirement" and item.acceptance:
            detail = item.acceptance.strip()
            if detail and detail not in original.acceptance:
                original.acceptance = f"{original.acceptance}；细化验收：{detail}"
        elif original.source == "jd":
            original.acceptance = item.acceptance or original.acceptance
            original.title = item.title or original.title
        original.blocking = bool(original.blocking or item.blocking)
    contract.criteria = list(by_id.values())
    contract.contract_id = stable_id("review_contract_", {
        "kind": contract.artifact_kind,
        "rubric": contract.rubric_version,
        "criteria": [item.model_dump(mode="json") for item in contract.criteria],
    })
    return contract.model_dump(mode="json")


def parse_critic_output(raw: str) -> CriticOutput:
    text = str(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Critic 没有返回 JSON 对象")
    return CriticOutput.model_validate(json.loads(text[start:end + 1]))


def finalize_review(*, kind: str, artifact_revision: int,
                    output: CriticOutput, base_contract: dict[str, Any],
                    hard_report: dict[str, Any], final_review: bool = False,
                    previous_report: dict[str, Any] | None = None) -> dict[str, Any]:
    # The first Critic pass compiles the contract. Later passes receive and
    # return the exact frozen contract even if the model proposes new criteria.
    if final_review:
        contract = ReviewContract.model_validate(base_contract).model_dump(mode="json")
    else:
        contract = merge_critic_contract(base_contract, [x.model_dump(mode="json")
                                                           for x in output.criteria])
    known_criteria = {item["criterion_id"] for item in contract["criteria"]}
    criteria_by_id = {item["criterion_id"]: item for item in contract["criteria"]}
    findings = []
    for item in output.findings:
        if item.criterion_id not in known_criteria:
            continue
        value = item.model_dump(mode="json")
        value["source"] = "critic"
        value["blocking"] = bool(
            criteria_by_id[item.criterion_id]["blocking"]
            and item.severity in {"critical", "major"}
        )
        findings.append(value)
    hard = HardValidationReport.model_validate(hard_report)
    for item in hard.findings:
        findings.append({
            "finding_id": item.finding_id,
            "criterion_id": "truthfulness" if kind in {"resume", "resume_project"} else "plan_feasibility",
            "severity": "critical" if item.severity == "error" else "minor",
            "artifact_ref": item.artifact_ref,
            "reason": item.message,
            "evidence_refs": [],
            "fix_instruction": item.message,
            "is_regression": False,
            "source": "hard_validator",
            "blocking": item.severity == "error",
        })
    verdict = output.verdict
    if not hard.valid and verdict in {"pass", "pass_with_warnings"}:
        verdict = "revise"
    requirement_ids = {item["requirement_id"] for item in contract["criteria"]
                       if item["source"] == "user_requirement"}
    unmet = sorted(set(output.unmet_requirement_ids) & requirement_ids)
    if unmet and verdict in {"pass", "pass_with_warnings"}:
        verdict = "revise"
    if (any(item.get("blocking") for item in findings)
            and verdict in {"pass", "pass_with_warnings"}):
        verdict = "revise"
    if output.evidence_conflicts or output.clarification_questions:
        verdict = "blocked"
    if verdict == "pass" and findings:
        verdict = "pass_with_warnings"
    if final_review and verdict == "revise" and previous_report:
        previous_blocking = {x.get("criterion_id") for x in previous_report.get("findings", [])
                             if x.get("severity") in {"critical", "major"}}
        for item in findings:
            if item.get("severity") == "minor" and item.get("criterion_id") not in previous_blocking:
                item["blocking"] = False
        has_blocking = any(item.get("blocking") for item in findings)
        if not has_blocking and not unmet and not output.evidence_conflicts and not output.clarification_questions:
            verdict = "pass_with_warnings"
    return {
        "review_id": stable_id("review_", {
            "contract": contract["contract_id"], "revision": artifact_revision,
            "verdict": verdict, "findings": findings,
        }),
        "artifact_revision": artifact_revision,
        "rubric_version": contract["rubric_version"],
        "review_contract": contract,
        "verdict": verdict,
        "findings": findings,
        "unmet_requirement_ids": unmet,
        "evidence_conflicts": list(output.evidence_conflicts),
        "revision_brief": list(output.revision_brief),
        "clarification_questions": list(output.clarification_questions),
        "summary": output.summary,
        "review_status": "completed",
        "final_review": final_review,
    }


def evidence_registry(context: dict[str, Any]) -> dict[str, str]:
    registry: dict[str, str] = {}
    for key, value in (context.get("profile") or {}).items():
        if value not in (None, "", [], {}):
            registry[f"profile:{key}"] = json.dumps(value, ensure_ascii=False, default=str)
    material = str(context.get("project_evidence") or "").strip()
    if material:
        # Keep the aggregate reference for v2 checkpoints while exposing each
        # approved source separately for new claims and provenance inspection.
        registry["project:offerclaw"] = material
    for item in context.get("project_sources") or []:
        evidence_ref = str(item.get("evidence_ref") or "").strip()
        content = str(item.get("content") or "").strip()
        if evidence_ref and content:
            registry[evidence_ref] = content
    resume_source = str(context.get("resume_source") or "").strip()
    if resume_source:
        registry["resume:uploaded"] = resume_source
    for item in (context.get("memory_context") or {}).get("semantic", []):
        memory_id = str(item.get("memory_id") or "")
        if memory_id:
            registry[f"memory:{memory_id}"] = json.dumps(item.get("value"), ensure_ascii=False)
    return registry


def render_resume(spec: dict[str, Any]) -> str:
    parsed = ResumeSpec.model_validate(spec)
    lines = [f"# {parsed.headline}", ""]
    for section in parsed.sections:
        lines.extend([f"## {section.title}", ""])
        for block in section.blocks:
            if block.kind == "bullet":
                lines.append(f"- {block.text}")
            else:
                lines.extend([block.text, ""])
        if lines[-1] != "":
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resume_spec_from_markdown(content: str, *, application_id: str,
                              jd_version_id: str,
                              artifact_scope: str = "full_resume",
                              previous_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reparse an advanced manual edit; only unchanged text keeps old evidence."""
    previous_claims: dict[str, list[dict[str, Any]]] = {}
    applied_requirement_ids: list[str] = []
    change_summary: list[str] = []
    if previous_spec:
        try:
            old = ResumeSpec.model_validate(previous_spec)
            applied_requirement_ids = list(old.applied_requirement_ids)
            change_summary = list(old.change_summary)
            for section in old.sections:
                for block in section.blocks:
                    key = re.sub(r"\s+", " ", block.text).strip()
                    previous_claims.setdefault(key, []).append({
                        "block_id": block.block_id,
                        "claim_id": block.claim_id,
                        "evidence_refs": block.evidence_refs,
                    })
        except Exception:
            pass
    headline = "候选人"
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in str(content or "").splitlines():
        line = raw.strip()
        if line.startswith("# ") and not line.startswith("## "):
            headline = line[2:].strip() or headline
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            current = {"section_id": stable_id("section_", title, 16),
                       "title": title, "blocks": []}
            sections.append(current)
            continue
        if not line or current is None:
            continue
        kind = "bullet" if re.match(r"^[-*]\s+", line) else "paragraph"
        text = re.sub(r"^[-*]\s+", "", line).strip()
        normalized = re.sub(r"\s+", " ", text).strip()
        candidates = previous_claims.get(normalized) or []
        previous = candidates.pop(0) if candidates else {}
        generated_block_id = stable_id("block_", {"section": current["section_id"],
                                                    "text": text}, 16)
        current["blocks"].append({
            "block_id": previous.get("block_id") or generated_block_id,
            "claim_id": previous.get("claim_id") or stable_id(
                "claim_", {"section": current["section_id"], "text": text}, 16),
            "kind": kind, "text": text,
            "evidence_refs": previous.get("evidence_refs") or [],
        })
    sections = [item for item in sections if item["blocks"]]
    return ResumeSpec(
        artifact_scope=artifact_scope,
        target_application_id=application_id, jd_version_id=jd_version_id,
        headline=headline, sections=sections,
        applied_requirement_ids=applied_requirement_ids,
        change_summary=[*change_summary, "用户通过高级编辑修改了简历正文"][-30:],
    ).model_dump(mode="json")


def hard_validate_resume(spec: dict[str, Any], content: str, context: dict[str, Any],
                         deterministic_report: dict[str, Any] | None = None,
                         requirements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    try:
        parsed = ResumeSpec.model_validate(spec)
    except Exception as exc:
        return HardValidationReport(valid=False, findings=[HardFinding(
            finding_id=stable_id("hard_", str(exc)), code="resume_schema_invalid",
            severity="error", message=f"简历结构无效：{str(exc)[:300]}")]).model_dump(mode="json")
    registry = evidence_registry(context)
    blocks_by_id: dict[str, ResumeBlock] = {}
    seen_sections: set[str] = set()
    seen_titles: set[str] = set()
    seen_blocks: set[str] = set()
    seen_claims: set[str] = set()
    for section in parsed.sections:
        if section.section_id in seen_sections:
            findings.append(_hard("duplicate_section_id", "error",
                                  f"重复 section_id：{section.section_id}", section.section_id))
        seen_sections.add(section.section_id)
        normalized_title = re.sub(r"\s+", "", section.title).casefold()
        if normalized_title in seen_titles:
            findings.append(_hard("duplicate_section_title", "error",
                                  f"重复简历章节：{section.title}", section.section_id))
        seen_titles.add(normalized_title)
        for block in section.blocks:
            blocks_by_id[block.block_id] = block
            if block.block_id in seen_blocks:
                findings.append(_hard("duplicate_block_id", "error",
                                      f"重复 block_id：{block.block_id}", block.block_id))
            seen_blocks.add(block.block_id)
            if not block.claim_id:
                findings.append(_hard("missing_claim_id", "error",
                                      "候选人主张缺少 claim_id", block.block_id))
            elif block.claim_id in seen_claims:
                findings.append(_hard("duplicate_claim_id", "error",
                                      f"重复 claim_id：{block.claim_id}", block.block_id))
            seen_claims.add(block.claim_id)
            if not block.evidence_refs:
                findings.append(_hard("missing_evidence_ref", "error",
                                      "候选人主张缺少 evidence_refs", block.block_id))
            unknown = [ref for ref in block.evidence_refs if ref not in registry]
            if unknown:
                findings.append(_hard("unknown_evidence_ref", "error",
                                      f"引用了不存在的证据：{', '.join(unknown[:3])}", block.block_id))
    expected_scope = str(context.get("resume_scope") or "full_resume")
    if parsed.artifact_scope != expected_scope:
        findings.append(_hard("resume_scope_mismatch", "error",
                              "简历产物范围与冻结上下文不一致"))
    required_sections = ({"project"} if expected_scope == "project_section"
                         else {"summary", "skills", "project"})
    for section_id in sorted(required_sections - seen_sections):
        findings.append(_hard("missing_resume_section", "error",
                              f"完整简历缺少必要章节：{section_id}", section_id))
    if (expected_scope == "full_resume" and
            any((context.get("profile") or {}).get(key) for key in ("学历", "专业", "学校"))):
        if "education" not in seen_sections:
            findings.append(_hard("missing_resume_section", "error",
                                  "完整简历缺少教育经历章节", "education"))
    if parsed.target_application_id != str((context.get("application") or {}).get("application_id") or ""):
        findings.append(_hard("application_mismatch", "error", "简历投递 ID 与冻结上下文不一致"))
    if parsed.jd_version_id != str((context.get("application") or {}).get("jd_version_id") or ""):
        findings.append(_hard("jd_version_mismatch", "error", "简历 JD 版本与冻结上下文不一致"))
    required_ids = {str(item.get("requirement_id") or "") for item in requirements or []}
    missing_requirements = sorted(required_ids - set(parsed.applied_requirement_ids))
    for requirement_id in missing_requirements:
        findings.append(_hard("unapplied_requirement", "error",
                              f"Resume Agent 未声明落实用户要求：{requirement_id}",
                              requirement_id))
    if "命中分析" in content:
        findings.append(_hard("analysis_leaked", "error", "正式简历正文包含“命中分析”元数据"))
    minimum_chars = 40 if expected_scope == "project_section" else 80
    if len(str(content or "").strip()) < minimum_chars:
        message = ("项目经历正文过短，无法进行可靠审查"
                   if expected_scope == "project_section" else
                   "简历正文过短，无法作为完整简历审查")
        findings.append(_hard("resume_too_short", "error", message))
    deterministic_report = deterministic_report or {}
    for flag in deterministic_report.get("fabrication_flags") or []:
        term = str(flag.get("term") or "").replace("(裸)", "").strip()
        if term:
            normalized_term = re.sub(r"\s+", "", term).casefold()
            matching = [block for block in blocks_by_id.values()
                        if normalized_term in re.sub(r"\s+", "", block.text).casefold()]
            unsupported = []
            for block in matching:
                cited = "\n".join(registry.get(ref, "") for ref in block.evidence_refs)
                if normalized_term not in re.sub(r"\s+", "", cited).casefold():
                    unsupported.append(block)
            for block in unsupported:
                findings.append(_hard(
                    "unsupported_claim", "error",
                    f"主张“{term}”未被该段引用的证据支持", block.block_id))
            if matching:
                continue
        findings.append(_hard("unsupported_claim", "error",
                              str(flag.get("claim") or flag.get("reason") or flag)[:300]))
    coverage = ((deterministic_report.get("keyword_coverage") or {}).get("coverage"))
    if (str((context.get("jd") or {}).get("jd_text") or "").strip()
            and coverage is not None and float(coverage) < 0.15):
        findings.append(_hard("low_keyword_coverage", "warning",
                              f"JD 关键词确定性覆盖率仅 {float(coverage):.0%}"))
    valid = not any(item["severity"] == "error" for item in findings)
    return HardValidationReport(valid=valid, findings=findings,
                                stats={"sections": len(parsed.sections),
                                       "blocks": len(seen_blocks),
                                       "keyword_coverage": coverage}).model_dump(mode="json")


def _hard(code: str, severity: str, message: str, artifact_ref: str = "") -> dict[str, Any]:
    return HardFinding(finding_id=stable_id("hard_", {"code": code, "ref": artifact_ref,
                                                       "message": message}),
                       code=code, severity=severity, message=message,
                       artifact_ref=artifact_ref).model_dump(mode="json")


def normalized_period(start_date: str = "", end_date: str = "") -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    try:
        start = dt.date.fromisoformat(start_date) if start_date else today
    except ValueError as exc:
        raise ValueError("计划开始日期无效") from exc
    try:
        end = dt.date.fromisoformat(end_date) if end_date else start + dt.timedelta(days=27)
    except ValueError as exc:
        raise ValueError("计划结束日期无效") from exc
    days = (end - start).days + 1
    if days < 14 or days > 56:
        raise ValueError("计划周期必须为 14 至 56 天")
    return start, end


def normalize_plan_schedule(data: dict[str, Any], start_date: str, end_date: str) -> dict[str, Any]:
    """Normalize a PlanSpec without inventing missing author decisions.

    A complete daily calendar is an Agent output requirement. Filling missing
    dates or assigning unscheduled tasks in code would hide a weak plan from
    both the hard validator and the Critic.
    """
    normalized_period(start_date, end_date)
    parsed = PlanSpec.model_validate(data or {})
    return parsed.model_dump(mode="json")


def hard_validate_plan(data: dict[str, Any], context: dict[str, Any],
                       start_date: str, end_date: str,
                       *, capacity: dict[str, float] | None = None,
                       requirements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    try:
        start, end = normalized_period(start_date, end_date)
    except ValueError as exc:
        return HardValidationReport(valid=False, findings=[_hard(
            "invalid_period", "error", str(exc))]).model_dump(mode="json")
    tasks: list[PlanTask] = []
    seen: set[str] = set()
    for field in ("common_gap_tasks", "role_specific_tasks"):
        for raw in data.get(field) or []:
            try:
                task = PlanTask.model_validate(raw)
            except Exception as exc:
                findings.append(_hard("invalid_task", "error", f"任务结构无效：{str(exc)[:240]}"))
                continue
            if task.task_id in seen:
                findings.append(_hard("duplicate_task_id", "error",
                                      f"重复 task_id：{task.task_id}", task.task_id))
            seen.add(task.task_id)
            tasks.append(task)
    applications = {str(item.get("application_id")): item for item in context.get("applications") or []}
    gaps = {str(item.get("gap_id")): item for item in [*(context.get("gaps") or []),
                                                        *(context.get("profile_goals") or [])]}
    resources = {str(item.get("resource_id")) for item in context.get("resources") or []}
    for task in tasks:
        if not task.gap_ids:
            findings.append(_hard("missing_gap_ref", "error", "任务缺少 gap_ids", task.task_id))
        for gap_id in task.gap_ids:
            if gap_id not in gaps or not gaps[gap_id].get("learnable"):
                findings.append(_hard("invalid_gap_ref", "error", f"任务引用不可学习或不存在的 gap：{gap_id}", task.task_id))
        if task.source_type == "application_jd":
            if not task.application_ids or not task.jd_version_ids:
                findings.append(_hard("missing_jd_ref", "error", "JD 任务缺少投递或 JD 引用", task.task_id))
            for app_id in task.application_ids:
                expected = str((applications.get(app_id) or {}).get("jd_version_id") or "")
                if not expected or expected not in task.jd_version_ids:
                    findings.append(_hard("jd_ref_mismatch", "error", f"投递 {app_id} 的 JD 版本不一致", task.task_id))
        for resource_id in task.resource_source_ids:
            if resource_id not in resources:
                findings.append(_hard("unknown_resource", "error", f"学习资源不存在：{resource_id}", task.task_id))
    task_map = {task.task_id: task for task in tasks}
    for task in tasks:
        for dependency_id in task.dependency_ids:
            if dependency_id == task.task_id:
                findings.append(_hard("self_dependency", "error", "任务不能依赖自身", task.task_id))
            elif dependency_id not in task_map:
                findings.append(_hard("unknown_dependency", "error",
                                      f"任务依赖不存在：{dependency_id}", task.task_id))
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return True
        if task_id in visited or task_id not in task_map:
            return False
        visiting.add(task_id)
        cyclic = any(visit(dep) for dep in task_map[task_id].dependency_ids)
        visiting.remove(task_id)
        visited.add(task_id)
        return cyclic

    if any(visit(task_id) for task_id in task_map):
        findings.append(_hard("dependency_cycle", "error", "任务依赖存在循环"))

    actions = [item for item in data.get("application_actions") or [] if isinstance(item, dict)]
    action_ids = [str(item.get("application_id") or "") for item in actions]
    for app_id in action_ids:
        if app_id not in applications:
            findings.append(_hard("unknown_application_action", "error",
                                  f"投递动作引用未知投递：{app_id}", app_id))
    for app_id in sorted(set(applications) - set(action_ids)):
        findings.append(_hard("missing_application_action", "error",
                              f"投递缺少明确下一步动作：{app_id}", app_id))
    for item in data.get("priority_decisions") or []:
        app_id = str(item.get("application_id") or "")
        if app_id not in applications:
            findings.append(_hard("unknown_priority_application", "error",
                                  f"优先级判断引用未知投递：{app_id}", app_id))
    for item in data.get("decision_warnings") or []:
        for gap_id in item.get("gap_ids") or []:
            if str(gap_id) not in gaps:
                findings.append(_hard("unknown_warning_gap", "error",
                                      f"决策提醒引用未知缺口：{gap_id}", str(gap_id)))
    required_ids = {str(item.get("requirement_id") or "") for item in requirements or []}
    applied_ids = {str(value) for value in data.get("applied_requirement_ids") or []}
    for requirement_id in sorted(required_ids - applied_ids):
        findings.append(_hard("unapplied_requirement", "error",
                              f"Plan Agent 未声明落实用户要求：{requirement_id}",
                              requirement_id))
    learnable_gaps = {gap_id for gap_id, item in gaps.items() if item.get("learnable")}
    covered_gaps = {gap_id for task in tasks for gap_id in task.gap_ids}
    for gap_id in sorted(learnable_gaps - covered_gaps):
        findings.append(_hard("uncovered_gap", "error", f"可学习缺口未被任务覆盖：{gap_id}", gap_id))
    schedule = data.get("schedule") or []
    expected_dates = [(start + dt.timedelta(days=index)).isoformat()
                      for index in range((end - start).days + 1)]
    actual_dates = [str(item.get("date") or "") for item in schedule if isinstance(item, dict)]
    if actual_dates != expected_dates:
        findings.append(_hard("incomplete_calendar", "error", "日计划未连续覆盖整个计划周期"))
    scheduled: list[str] = []
    cap = {"weekday": 3.0, "weekend": 5.0, **(capacity or {})}
    weekly_hours: dict[int, float] = {}
    weekly_capacity: dict[int, float] = {}
    for day in schedule:
        if not isinstance(day, dict):
            continue
        core = [str(value) for value in day.get("core_task_ids") or []]
        optional = [str(value) for value in day.get("optional_task_ids") or []]
        if len(core) > 2 or len(optional) > 1:
            findings.append(_hard("daily_task_limit", "error", "每天最多 2 个核心任务和 1 个可选任务", str(day.get("date") or "")))
        for task_id in core + optional:
            if task_id not in task_map:
                findings.append(_hard("unknown_scheduled_task", "error", f"日计划引用未知任务：{task_id}", str(day.get("date") or "")))
            scheduled.append(task_id)
        try:
            date = dt.date.fromisoformat(str(day.get("date") or ""))
        except ValueError:
            continue
        hours = float(day.get("planned_hours") or 0)
        daily_cap = cap["weekend"] if date.weekday() >= 5 else cap["weekday"]
        if hours > daily_cap + 1e-9:
            findings.append(_hard("daily_capacity_exceeded", "error",
                                  f"{date.isoformat()} 计划 {hours:g}h，超过上限 {daily_cap:g}h", date.isoformat()))
        week = ((date - start).days // 7) + 1
        weekly_hours[week] = weekly_hours.get(week, 0) + hours
        weekly_capacity[week] = weekly_capacity.get(week, 0) + daily_cap
    duplicates = sorted({task_id for task_id in scheduled if scheduled.count(task_id) > 1})
    for task_id in duplicates:
        findings.append(_hard("duplicate_schedule_ref", "error", f"同一具体任务被重复排期：{task_id}", task_id))
    for task_id in sorted(set(task_map) - set(scheduled)):
        findings.append(_hard("unscheduled_task", "error", f"任务未进入日计划：{task_id}", task_id))
    first_day: dict[str, int] = {}
    for index, day in enumerate(schedule):
        if not isinstance(day, dict):
            continue
        for value in [*(day.get("core_task_ids") or []),
                      *(day.get("optional_task_ids") or [])]:
            first_day.setdefault(str(value), index)
    for task in tasks:
        if task.task_id not in first_day:
            continue
        for dependency_id in task.dependency_ids:
            if dependency_id in first_day and first_day[dependency_id] >= first_day[task.task_id]:
                findings.append(_hard("dependency_order", "error",
                                      f"依赖任务 {dependency_id} 必须早于当前任务", task.task_id))
    for week, hours in weekly_hours.items():
        if hours > weekly_capacity.get(week, 0) * 0.85 + 1e-9:
            findings.append(_hard("weekly_buffer_missing", "error",
                                  f"第 {week} 周未保留至少 15% 机动时间"))
    valid = not any(item["severity"] == "error" for item in findings)
    return HardValidationReport(valid=valid, findings=findings,
                                stats={"task_count": len(tasks), "day_count": len(schedule),
                                       "covered_gap_count": len(covered_gaps & learnable_gaps),
                                       "total_gap_count": len(learnable_gaps)}).model_dump(mode="json")


def plan_spec_from_markdown(content: str, previous_spec: dict[str, Any]) -> dict[str, Any]:
    """Reparse the editable daily layer into the canonical plan structure.

    Stable task definitions and evidence links come from the previous spec;
    the edited Markdown must retain those IDs. New free-form tasks therefore
    cannot bypass evidence validation through the advanced editor.
    """
    from plan_daily import parse_plan_days

    parsed = parse_plan_days(str(content or ""))
    days = parsed.get("days") or []
    if not days:
        raise ValueError("手动编辑后的计划缺少可解析的逐日安排")
    out = json.loads(json.dumps(previous_spec, ensure_ascii=False))
    known = {str(item.get("task_id")) for field in ("common_gap_tasks", "role_specific_tasks")
             for item in out.get(field) or [] if item.get("task_id")}
    schedule = []
    for day in days:
        core = [str(item.get("task_id") or "") for item in day.get("task_items") or []
                if not item.get("optional")]
        optional = [str(item.get("task_id") or "") for item in day.get("task_items") or []
                    if item.get("optional") and str(item.get("text") or "").strip() != "无"]
        if any(not task_id or task_id not in known for task_id in core + optional):
            raise ValueError(f"{day.get('label')} 包含缺失或未知 task_id")
        block_lines = str(content or "").splitlines()[day["header_line"]:day["block_end"]]
        hours = 0.0
        for line in block_lines:
            match = re.search(r"预计投入[^\d]*(\d+(?:\.\d+)?)\s*h", line, re.I)
            if match:
                hours = float(match.group(1))
                break
        schedule.append({"date": day["date"].isoformat(), "focus": day.get("label") or "",
                         "core_task_ids": core, "optional_task_ids": optional,
                         "planned_hours": hours})
    out["schedule"] = schedule
    return out


def artifact_diff(before: str, after: str) -> dict[str, Any]:
    old = str(before or "").splitlines()
    new = str(after or "").splitlines()
    added = removed = 0
    for line in difflib.ndiff(old, new):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    headings = [line.lstrip("# ").strip() for line in new if line.startswith("## ")]
    return {"added_lines": added, "removed_lines": removed,
            "changed": old != new, "section_titles": headings}
