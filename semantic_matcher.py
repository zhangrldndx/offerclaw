# -*- coding: utf-8 -*-
"""Evidence-grounded semantic alignment between one JD and one user profile.

The LLM is deliberately limited to a translation task: relate an exact JD
requirement to one or more existing profile evidence items.  It cannot decide
hard gates or the final three-tier match result.  Those decisions remain in
``match_job`` and consume only the validated alignment contract produced here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from structured_llm import StructuredCallMeta, TextCaller, call_structured


SEMANTIC_MATCH_SCHEMA_VERSION = "semantic-match-v1"
SEMANTIC_MATCHER_VERSION = "evidence-aligner-v1"

_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}

EvidenceType = Literal[
    "declared_skill", "capability_evidence", "project", "work",
    "research", "competition",
]
Relation = Literal["direct", "transferable", "partial", "unsupported", "contradicted"]


class ProfileEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(min_length=1, max_length=96)
    evidence_type: EvidenceType
    source_ref: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=5000)


class _AlignmentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(min_length=1, max_length=96)
    relation: Relation
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(min_length=1, max_length=600)


class _AlignmentDraftBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alignments: list[_AlignmentDraft] = Field(default_factory=list, max_length=160)


class RequirementAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    requirement_text: str
    requirement_kind: str
    modality: str
    relation: Relation
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    rationale: str


class SemanticMatchAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SEMANTIC_MATCH_SCHEMA_VERSION
    input_hash: str
    evidence_hash: str
    source: Literal["llm", "cache", "deterministic"]
    status: Literal["completed", "not_requested", "degraded"]
    model: str = ""
    alignments: list[RequirementAlignment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    call_meta: dict[str, Any] = Field(default_factory=dict)


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _clean_text(value: object, limit: int = 5000) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _fallback_evidence(profile: dict[str, Any]) -> list[ProfileEvidence]:
    """Build auditable evidence for programmatic/test profiles.

    A project count is intentionally not evidence.  Only user-provided content
    can support a semantic capability relation.
    """
    mapping: tuple[tuple[str, EvidenceType], ...] = (
        ("熟练技能", "declared_skill"),
        ("会用技能", "declared_skill"),
        ("技能证据", "capability_evidence"),
        ("项目技能", "project"),
    )
    out: list[ProfileEvidence] = []
    for field, evidence_type in mapping:
        values = profile.get(field) or []
        if not isinstance(values, list):
            values = [values]
        for index, value in enumerate(values):
            text = _clean_text(value, 1200)
            if not text or "待补充" in text:
                continue
            source_ref = f"profile:{field}:{index}"
            out.append(ProfileEvidence(
                evidence_id=_stable_id("ev_", source_ref, text),
                evidence_type=evidence_type,
                source_ref=source_ref,
                text=text,
            ))
    return out


def build_profile_evidence(profile: dict[str, Any]) -> list[ProfileEvidence]:
    """Return a de-duplicated, positive-evidence-only catalog."""
    raw_items = profile.get("_evidence_catalog") or []
    items: list[ProfileEvidence] = []
    for raw in raw_items:
        try:
            items.append(ProfileEvidence.model_validate(raw))
        except Exception:
            continue
    items.extend(_fallback_evidence(profile))
    deduped: dict[str, ProfileEvidence] = {}
    seen_content: set[tuple[str, str]] = set()
    for item in items:
        content_key = (item.evidence_type, " ".join(item.text.casefold().split()))
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        # An ID collision with different content is not silently accepted.
        previous = deduped.get(item.evidence_id)
        if previous and previous.text != item.text:
            item = item.model_copy(update={
                "evidence_id": _stable_id("ev_", item.source_ref, item.text),
            })
        deduped[item.evidence_id] = item
    return list(deduped.values())[:120]


def _eligible_requirements(jd_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in jd_analysis.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        requirement_id = str(item.get("requirement_id") or "").strip()
        kind = str(item.get("kind") or "")
        modality = str(item.get("modality") or "unknown")
        text = _clean_text(item.get("text"), 1200)
        # Hard qualifications remain fully deterministic. Context statements
        # and negated requirements are not candidate capabilities.
        if (not requirement_id or not text or kind == "hard"
                or modality == "context"):
            continue
        out.append({
            "requirement_id": requirement_id,
            "text": text,
            "kind": kind,
            "modality": modality,
            "priority": item.get("priority", 0.0),
        })
    return out


def _input_hash(jd_analysis: dict[str, Any]) -> str:
    stable = [{key: item.get(key) for key in (
        "requirement_id", "text", "kind", "modality", "priority"
    )} for item in _eligible_requirements(jd_analysis)]
    return hashlib.sha256(json.dumps(
        stable, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _evidence_hash(evidence: list[ProfileEvidence]) -> str:
    stable = [item.model_dump(mode="json") for item in evidence]
    return hashlib.sha256(json.dumps(
        stable, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _messages(requirements: list[dict[str, Any]],
              evidence: list[ProfileEvidence]) -> list[dict[str, str]]:
    system = (
        "你是岗位要求与候选人证据的语义对齐器，不是岗位录用决策者。"
        "逐项判断 JD requirement 与现有 profile evidence 的关系。"
        "direct 表示证据直接证明同一能力或经历；transferable 表示证据证明可迁移的"
        "底层能力但场景不同；partial 表示只覆盖要求的一部分或熟练度不足；"
        "unsupported 表示没有证据；contradicted 表示证据明确相反。"
        "只能引用输入中存在的 requirement_id 和 evidence_id，不得补写候选人事实。"
        "项目/工作经验要求不能只用技能名称证明；‘会用’不能自动满足‘精通’。"
        "每个 requirement_id 恰好输出一次。unsupported 不得附 evidence_ids。"
        "不要输出适合投递、中长期可转向、不建议投递、分数或录用结论。"
        "只返回符合 JSON Schema 的 JSON。"
    )
    payload = {
        "requirements": requirements,
        "profile_evidence": [item.model_dump(mode="json") for item in evidence],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _ground(requirements: list[dict[str, Any]], evidence: list[ProfileEvidence],
            draft: _AlignmentDraftBundle, *, model: str,
            input_hash: str, evidence_hash: str,
            meta: StructuredCallMeta) -> SemanticMatchAssessment:
    requirement_by_id = {item["requirement_id"]: item for item in requirements}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    accepted: dict[str, RequirementAlignment] = {}
    warnings: list[str] = []
    for index, item in enumerate(draft.alignments):
        requirement = requirement_by_id.get(item.requirement_id)
        if requirement is None:
            warnings.append(f"unknown_requirement_ref:{index}:{item.requirement_id}")
            continue
        if item.requirement_id in accepted:
            warnings.append(f"duplicate_requirement_alignment:{item.requirement_id}")
            continue
        refs = list(dict.fromkeys(item.evidence_ids))
        unknown = [value for value in refs if value not in evidence_by_id]
        if unknown:
            warnings.append(f"unknown_evidence_ref:{item.requirement_id}")
            continue
        if item.relation == "unsupported" and refs:
            warnings.append(f"unsupported_with_evidence:{item.requirement_id}")
            refs = []
        if item.relation in {"direct", "transferable", "partial", "contradicted"} and not refs:
            warnings.append(f"relation_without_evidence:{item.requirement_id}")
            continue
        cited = [evidence_by_id[value] for value in refs]
        relation: Relation = item.relation
        # A named skill is not proof that the user performed a responsibility
        # or accumulated experience. Keep that distinction deterministic.
        if (requirement["kind"] in {"experience", "responsibility"}
                and relation in {"direct", "transferable"}
                and cited
                and all(value.evidence_type in {
                    "declared_skill", "capability_evidence"
                } for value in cited)):
            relation = "partial"
            warnings.append(f"experience_relation_downgraded:{item.requirement_id}")
        accepted[item.requirement_id] = RequirementAlignment(
            requirement_id=item.requirement_id,
            requirement_text=requirement["text"],
            requirement_kind=requirement["kind"],
            modality=requirement["modality"],
            relation=relation,
            evidence_ids=refs,
            evidence_types=list(dict.fromkeys(value.evidence_type for value in cited)),
            evidence_sources=list(dict.fromkeys(value.source_ref for value in cited)),
            rationale=item.rationale.strip(),
        )
    valid_alignment_count = len(accepted)
    for requirement in requirements:
        requirement_id = requirement["requirement_id"]
        if requirement_id in accepted:
            continue
        warnings.append(f"missing_requirement_alignment:{requirement_id}")
        accepted[requirement_id] = RequirementAlignment(
            requirement_id=requirement_id,
            requirement_text=requirement["text"],
            requirement_kind=requirement["kind"],
            modality=requirement["modality"],
            relation="unsupported", evidence_ids=[], evidence_types=[],
            evidence_sources=[], rationale="模型未返回可验证的证据对应，按无证据处理",
        )
    status = "completed" if valid_alignment_count else "degraded"
    if not valid_alignment_count:
        warnings.append("no_valid_model_alignment")
    return SemanticMatchAssessment(
        input_hash=input_hash, evidence_hash=evidence_hash,
        source="llm", status=status, model=model,
        alignments=[accepted[item["requirement_id"]] for item in requirements],
        warnings=list(dict.fromkeys(warnings)), call_meta=meta.to_dict(),
    )


def _resolved_model() -> str:
    explicit = os.environ.get("MATCH_SEMANTIC_MODEL", "").strip()
    if explicit:
        return explicit
    try:
        from day1_api_starter import get_llm_config, load_local_env
        load_local_env()
        return str((get_llm_config() or {}).get("model") or "")
    except Exception:
        return ""


def _cache_path(input_hash: str, evidence_hash: str, model: str,
                cache_dir: str | Path | None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else (
        Path(__file__).resolve().parent / ".offerclaw" / "cache" / "semantic_match"
    )
    raw = f"{input_hash}|{evidence_hash}|{SEMANTIC_MATCHER_VERSION}|{model}"
    return base / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()}.json"


def _cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.Lock())


def _load_cache(path: Path) -> SemanticMatchAssessment | None:
    try:
        value = SemanticMatchAssessment.model_validate_json(path.read_text(encoding="utf-8"))
        if value.status != "completed":
            return None
        return value.model_copy(update={"source": "cache"})
    except Exception:
        return None


def _save_cache(path: Path, value: SemanticMatchAssessment) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        pass


def _empty_assessment(jd_analysis: dict[str, Any], evidence: list[ProfileEvidence],
                      *, status: Literal["not_requested", "degraded"],
                      warning: str = "") -> SemanticMatchAssessment:
    return SemanticMatchAssessment(
        input_hash=_input_hash(jd_analysis), evidence_hash=_evidence_hash(evidence),
        source="deterministic", status=status,
        warnings=[warning] if warning else [],
    )


def align_requirements(profile: dict[str, Any], jd_analysis: dict[str, Any], *,
                       enabled: bool = True, model: str | None = None,
                       cache_dir: str | Path | None = None,
                       caller: TextCaller | None = None) -> SemanticMatchAssessment:
    """Call the semantic aligner once and return a validated assessment.

    Failure is fail-visible and conservative. No alignment is returned as a
    successful LLM review when the model, schema, or evidence is unavailable.
    """
    evidence = build_profile_evidence(profile)
    requirements = _eligible_requirements(jd_analysis)
    if not enabled:
        return _empty_assessment(jd_analysis, evidence, status="not_requested")
    if not requirements:
        return _empty_assessment(
            jd_analysis, evidence, status="degraded", warning="no_eligible_requirements",
        )
    if not evidence:
        return _empty_assessment(
            jd_analysis, evidence, status="degraded", warning="no_profile_evidence",
        )
    resolved_model = model if model is not None else _resolved_model()
    if not resolved_model and caller is None:
        return _empty_assessment(
            jd_analysis, evidence, status="degraded", warning="llm_unavailable:no_model",
        )
    input_hash = _input_hash(jd_analysis)
    evidence_hash = _evidence_hash(evidence)
    path = _cache_path(input_hash, evidence_hash, resolved_model or "injected", cache_dir)
    with _cache_lock(path):
        cached = _load_cache(path)
        if cached is not None:
            return cached
        draft, meta = call_structured(
            _AlignmentDraftBundle, _messages(requirements, evidence),
            model=resolved_model or None,
            timeout_seconds=float(os.environ.get(
                "MATCH_SEMANTIC_TIMEOUT_SECONDS", "30"
            ) or 30),
            max_tokens=3200, repair=True, caller=caller,
            reasoning_effort=os.environ.get(
                "MATCH_SEMANTIC_REASONING_EFFORT", "low"
            ).strip() or None,
            lane="online",
        )
        if draft is None:
            warning = "llm_fallback:" + (",".join(meta.errors) or "invalid_output")
            value = _empty_assessment(
                jd_analysis, evidence, status="degraded", warning=warning,
            ).model_copy(update={"model": resolved_model, "call_meta": meta.to_dict()})
            return value
        value = _ground(
            requirements, evidence, draft, model=resolved_model,
            input_hash=input_hash, evidence_hash=evidence_hash, meta=meta,
        )
        if value.status == "completed":
            _save_cache(path, value)
        return value


__all__ = [
    "ProfileEvidence", "RequirementAlignment", "SemanticMatchAssessment",
    "SEMANTIC_MATCH_SCHEMA_VERSION", "align_requirements", "build_profile_evidence",
]
