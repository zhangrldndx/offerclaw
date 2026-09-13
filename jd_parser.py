# -*- coding: utf-8 -*-
"""Versioned, evidence-grounded JD analysis shared by all consumers.

The model may extract requirements, modalities and terminology. Code verifies
that every accepted item points to an exact span in the original JD. This
module never makes the final hard-gate/application decision.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from structured_llm import StructuredCallMeta, TextCaller, call_structured


JD_ANALYSIS_SCHEMA_VERSION = "jd-analysis-v3"
JD_ANALYZER_VERSION = "evidence-extractor-v3"
JD_DETERMINISTIC_MODEL = "deterministic-segmenter-v1"

_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1200)
    start: int = -1
    end: int = -1


class JDRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1200)
    kind: Literal["hard", "skill", "experience", "responsibility", "nice_to_have"]
    modality: Literal["required", "preferred", "alternative", "context", "unknown"]
    priority: float = Field(ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list, max_length=20)
    evidence_spans: list[EvidenceSpan] = Field(min_length=1, max_length=8)


class JDKeyword(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str = Field(min_length=1, max_length=120)
    surface_forms: list[str] = Field(min_length=1, max_length=12)
    category: str = Field(default="other", max_length=80)
    importance: float = Field(ge=0.0, le=1.0)
    requirement_ids: list[str] = Field(default_factory=list, max_length=20)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list, max_length=8)


class JDAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = JD_ANALYSIS_SCHEMA_VERSION
    input_hash: str
    source: Literal["llm", "deterministic", "cache", "legacy"]
    model: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    job_type: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[JDRequirement] = Field(default_factory=list)
    keywords: list[JDKeyword] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def keyword_names(self) -> list[str]:
        return [item.canonical_name for item in self.keywords]


class _DraftRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1200)
    kind: Literal["hard", "skill", "experience", "responsibility", "nice_to_have"]
    modality: Literal["required", "preferred", "alternative", "context", "unknown"]
    priority: float = Field(ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list, max_length=20)
    evidence_spans: list[str] = Field(min_length=1, max_length=8)


class _DraftKeyword(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str = Field(min_length=1, max_length=120)
    surface_forms: list[str] = Field(min_length=1, max_length=12)
    category: str = Field(default="other", max_length=80)
    importance: float = Field(ge=0.0, le=1.0)
    requirement_indexes: list[int] = Field(default_factory=list, max_length=20)
    evidence_spans: list[str] = Field(default_factory=list, max_length=8)


class _JDAnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = ""
    company: str = ""
    location: str = ""
    job_type: str = ""
    responsibilities: list[str] = Field(default_factory=list, max_length=80)
    requirements: list[_DraftRequirement] = Field(default_factory=list, max_length=120)
    keywords: list[_DraftKeyword] = Field(default_factory=list, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=30)


def _input_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _find_span(jd_text: str, evidence: str) -> EvidenceSpan | None:
    value = str(evidence or "").strip()
    if not value:
        return None
    start = jd_text.find(value)
    if start < 0:
        parts = [re.escape(part) for part in re.split(r"\s+", value) if part]
        if not parts:
            return None
        match = re.search(r"\s*".join(parts), jd_text)
        if match is None:
            return None
        start, end = match.span()
        return EvidenceSpan(text=jd_text[start:end], start=start, end=end)
    return EvidenceSpan(text=jd_text[start:start + len(value)],
                        start=start, end=start + len(value))


def _find_term_span(jd_text: str, term: str) -> EvidenceSpan | None:
    """Locate one lexical term without matching ``rag`` inside ``storage``."""
    value = str(term or "").strip()
    if not value:
        return None
    if any("\u4e00" <= char <= "\u9fff" for char in value):
        pattern = re.escape(value)
    else:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
    match = re.search(pattern, jd_text, re.I)
    if match is None:
        return None
    return EvidenceSpan(text=jd_text[match.start():match.end()],
                        start=match.start(), end=match.end())


def term_in_text(text: str, term: str) -> bool:
    """Shared lexical matcher with English token boundaries."""
    return _find_term_span(text or "", term or "") is not None


def _normalized_evidence_text(value: str) -> str:
    value = _clean_line(str(value or "")) if "_clean_line" in globals() else str(value or "").strip()
    return re.sub(r"[\s，,。；;：:（）()【】\[\]‘’'\"`]+", "", value).lower()


def _requirement_supported(text: str, spans: list[EvidenceSpan]) -> bool:
    normalized = _normalized_evidence_text(text)
    if not normalized:
        return False
    for span in spans:
        evidence = _normalized_evidence_text(span.text)
        if normalized == evidence:
            return True
        shorter, longer = sorted((normalized, evidence), key=len)
        if shorter and shorter in longer and len(shorter) / max(1, len(longer)) >= 0.82:
            return True
    return False


def _first(patterns: tuple[str, ...], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.I | re.M)
        if match:
            return match.group(1).strip().splitlines()[0][:120]
    return ""


def _field_fallback(jd_text: str) -> tuple[str, str, str, str]:
    title = _first((
        r"^(?:岗位名称|职位名称|招聘岗位|岗位|职位)\s*[：:]\s*(.+)$",
        r"(?:职位详情|岗位详情)\s*\n+\s*([^\n]{4,100})",
    ), jd_text)
    company = _first((
        r"^(?:公司名称|招聘公司|Company|公司)\s*[：:]\s*(.+)$",
        r"公司信息\s*\n+\s*([^\n]{2,80})",
    ), jd_text)
    location = _first((
        r"^(?:工作地点|工作城市|办公地点|办公地址|职位地点|Location|城市|地点)\s*[：:]\s*([^\n，,；;|｜]+)",
    ), jd_text)
    job_type = _first((
        r"^(?:岗位性质|工作性质|招聘类型|用工类型|类型)\s*[：:]\s*([^\n，,；;|｜]+)",
    ), jd_text)
    if not title:
        for line in jd_text.splitlines():
            value = line.strip(" #\t")
            if value and 3 < len(value) <= 100:
                title = value
                break
    if not location:
        for city in ("北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "西安", "苏州"):
            if city in jd_text[:500]:
                location = city
                break
    if not job_type:
        for value in ("实习", "全职", "校招", "社招", "兼职"):
            if value in jd_text[:500]:
                job_type = value
                break
    return title, company, location, job_type


_SECTION_HEADING = re.compile(
    r"^(?:#+\s*)?(岗位职责|工作职责|职位描述|职责描述|职责|Responsibilities|"
    r"任职要求|岗位要求|职位要求|基本要求|要求|Requirements|加分项|优先条件|Nice to have)\s*[：:]?\s*$",
    re.I,
)
_PREFERRED_MARKERS = ("优先", "加分", "更佳", "为佳", "nice to have", "preferred")
_ALTERNATIVE_MARKERS = ("或", "或者", "任一", "至少一种", "至少一门", "one of", "either")
_REQUIRED_MARKERS = ("必须", "要求", "应具备", "需具备", "熟练", "精通", "至少", "required")
_CONTEXT_MARKERS = ("团队使用", "目前使用", "技术栈包括", "项目使用", "了解即可", "不要求")
_HARD_MARKERS = (
    "学历", "学位", "专业", "毕业", "年经验", "年以上", "年龄",
    "英语", "语言", "证书", "工作地点",
)
_EXPERIENCE_MARKERS = ("经验", "经历", "项目背景", "从业")


def _clean_line(line: str) -> str:
    return re.sub(r"^\s*(?:[-*•·]|\d+[.)、]|[（(]?\d+[）)])\s*", "", line).strip()


def _line_modality(line: str, section: str) -> str:
    lower = line.lower()
    if any(marker in lower for marker in _CONTEXT_MARKERS):
        return "context"
    if section == "preferred" or any(marker in lower for marker in _PREFERRED_MARKERS):
        return "preferred"
    if any(marker in lower for marker in _ALTERNATIVE_MARKERS):
        return "alternative"
    if any(marker in lower for marker in _REQUIRED_MARKERS) or section == "requirements":
        return "required"
    return "unknown"


def _line_kind(line: str, section: str, modality: str) -> str:
    lower = line.lower()
    if section == "responsibilities":
        return "responsibility"
    if modality == "preferred":
        return "nice_to_have"
    if any(marker in lower for marker in _HARD_MARKERS):
        return "hard"
    if any(marker in lower for marker in _EXPERIENCE_MARKERS):
        return "experience"
    return "skill"


def _alternatives(line: str, modality: str) -> list[str]:
    if modality != "alternative":
        return []
    values = [part.strip(" ，,；;。") for part in re.split(r"(?:或者|或|/|\bor\b)", line, flags=re.I)]
    return [value for value in values if value and value != line][:12]


def _deterministic_analysis(jd_text: str) -> JDAnalysis:
    title, company, location, job_type = _field_fallback(jd_text)
    requirements: list[JDRequirement] = []
    responsibilities: list[str] = []
    section = "unknown"
    for raw_line in jd_text.splitlines():
        stripped = raw_line.strip()
        heading = _SECTION_HEADING.match(stripped)
        inline = re.match(
            r"^(岗位职责|工作职责|职位描述|职责描述|职责|Responsibilities|任职要求|岗位要求|"
            r"职位要求|基本要求|要求|Requirements|加分项|优先条件|Nice to have|"
            r"学历要求|专业要求|经验要求|技术要求|技能要求|语言要求)\s*[：:]\s*(.+)$",
            stripped, re.I,
        )
        if inline:
            heading_key = inline.group(1).lower()
            section = ("responsibilities" if any(x in heading_key for x in ("职责", "描述", "responsib"))
                       else "preferred" if any(x in heading_key for x in ("加分", "优先", "nice"))
                       else "requirements")
            # Field-shaped requirement lines need the key in their evidence;
            # otherwise “专业要求：计算机” degrades into an untyped “计算机”
            # and cannot be safely used by the corresponding hard gate.
            raw_line = (stripped if heading_key in {
                "学历要求", "专业要求", "经验要求", "技术要求", "技能要求", "语言要求",
            } else inline.group(2))
            heading = None
        if heading:
            key = heading.group(1).lower()
            section = ("responsibilities" if any(x in key for x in ("职责", "描述", "responsib"))
                       else "preferred" if any(x in key for x in ("加分", "优先", "nice"))
                       else "requirements")
            continue
        line = _clean_line(raw_line)
        if len(line) < 4 or section == "unknown":
            continue
        span = _find_span(jd_text, line)
        if span is None:
            continue
        modality = _line_modality(line, section)
        kind = _line_kind(line, section, modality)
        priority = 0.9 if modality == "required" else 0.55 if modality == "alternative" else (
            0.4 if modality == "preferred" else 0.25
        )
        requirement_id = _stable_id("req", line, span.start)
        requirements.append(JDRequirement(
            requirement_id=requirement_id, text=line, kind=kind,
            modality=modality, priority=priority,
            alternatives=_alternatives(line, modality), evidence_spans=[span],
        ))
        if kind == "responsibility":
            responsibilities.append(line)

    from context_budget import keywords_from
    requirement_text = "\n".join(item.text for item in requirements) or jd_text
    tokens = keywords_from(requirement_text, top=80)
    # Preserve meaningful surface phrases without a project-local skill
    # whitelist.  Candidates are sourced from the JD itself or the shared,
    # editable career-domain registry and still require exact source evidence.
    phrase_candidates = re.findall(r"[\u4e00-\u9fff]{4,12}", requirement_text)
    phrase_candidates += [
        match.group(0).strip() for match in re.finditer(
            r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9+.#-]*\s+){1,2}"
            r"[A-Za-z][A-Za-z0-9+.#-]*(?![A-Za-z0-9])",
            requirement_text,
        )
    ]
    try:
        from career_domains import detect_domain, load_domain_templates
        domain_id = detect_domain(jd_text)
        domain = next((item for item in load_domain_templates()
                       if item.get("id") == domain_id), None)
        if domain:
            phrase_candidates += [
                str(value) for value in domain.get("competency_keywords") or []
                if _find_term_span(jd_text, str(value)) is not None
            ]
    except Exception:
        pass
    for candidate in phrase_candidates:
        value = candidate.strip(" ，,；;。()（）【】")
        if value and value.lower() not in {str(token).lower() for token in tokens}:
            tokens.append(value)
    stop = {
        "要求", "负责", "相关", "工作", "岗位", "职位", "能力", "经验", "以上",
        "进行", "具备", "熟悉", "掌握", "优先", "能够", "以及", "我们", "具有",
        "熟练", "熟练掌握", "开发", "应用", "内容", "团队", "参与", "完成",
        "requirements", "responsibilities", "the", "and", "with", "for", "you",
    }
    keywords: list[JDKeyword] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = token.strip()
        key = normalized.lower()
        if len(normalized) < 2 or key in stop or key in seen:
            continue
        related = [item.requirement_id for item in requirements
                   if _find_term_span(item.text, normalized) is not None]
        match_start = -1
        match_end = -1
        for requirement in requirements:
            if requirement.requirement_id not in related:
                continue
            local = _find_term_span(requirement.evidence_spans[0].text, normalized)
            if local is not None:
                match_start = requirement.evidence_spans[0].start + local.start
                match_end = requirement.evidence_spans[0].start + local.end
                break
        if match_start < 0:
            match = _find_term_span(jd_text, normalized)
            if match is None:
                continue
            match_start, match_end = match.start, match.end
        seen.add(key)
        importance = max(
            (item.priority for item in requirements if item.requirement_id in related),
            default=0.35,
        )
        category = "technology" if re.search(r"[A-Za-z+#.]", normalized) else "competency"
        keywords.append(JDKeyword(
            canonical_name=normalized, surface_forms=[jd_text[match_start:match_end]],
            category=category, importance=importance, requirement_ids=related,
            evidence_spans=[EvidenceSpan(
                text=jd_text[match_start:match_end], start=match_start, end=match_end)],
        ))
    return JDAnalysis(
        input_hash=_input_hash(jd_text), source="deterministic", model=JD_DETERMINISTIC_MODEL,
        title=title, company=company, location=location, job_type=job_type,
        responsibilities=list(dict.fromkeys(responsibilities)),
        requirements=requirements, keywords=keywords,
        warnings=[] if requirements else ["deterministic_no_requirement_section"],
    )


def _analysis_messages(jd_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "你是 JD 证据化抽取器，不做候选人匹配、不决定是否投递。"
            "抽取公司、岗位、地点、性质、职责、要求和关键词。每条 requirement 的"
            "evidence_spans、每个 keyword 的 evidence_spans 必须逐字复制自 JD 原文；"
            "不得把常识、推断或规范化名称当原文证据。正确区分必须、优先/加分、任选其一、"
            "上下文描述和‘不要求经验’等否定。requirement_indexes 使用从 0 开始的要求序号。"
            "kind 的口径固定：hard=可由代码核验的资格/硬门槛，包括学历专业、毕业届次、"
            "明确经验年限、工作地点/到岗方式、语言等级或工作语言、证书和其他明确必须条件；"
            "skill=工具技术或可学习能力；experience=没有形成硬年限/必须措辞的项目或经历要求；"
            "responsibility=入职后职责；nice_to_have=加分项。kind=hard 只是候选标签，"
            "出现‘A或B/任选/其中一门/至少一种/满足其一’时 modality 必须是 alternative，"
            "不能因为同时出现‘至少/要求’改成 required；可替代的编程或自然语言池属于 skill，"
            "只有单一强制工作语言、明确语言等级或证书才属于 hard。"
            "最终淘汰仍由代码根据原文措辞核验。"
            "只输出满足 JSON schema 的 JSON。"
        )},
        {"role": "user", "content": json.dumps({"jd_text": jd_text[:30000]}, ensure_ascii=False)},
    ]


def _ground_draft(jd_text: str, draft: _JDAnalysisDraft, *, model: str) -> JDAnalysis:
    warnings = list(draft.warnings)
    requirements: list[JDRequirement] = []
    index_to_id: dict[int, str] = {}
    for index, item in enumerate(draft.requirements):
        spans = [span for value in item.evidence_spans
                 if (span := _find_span(jd_text, value)) is not None]
        if not spans:
            warnings.append(f"dropped_ungrounded_requirement:{index}")
            continue
        source_text = spans[0].text.strip()
        if not _requirement_supported(item.text, spans):
            # Never preserve a model paraphrase that the cited source cannot
            # support. The exact cited span is still a valid requirement
            # candidate, so normalize the text to that span instead of losing
            # recall. Downstream hard gates independently verify its wording.
            warnings.append(f"normalized_requirement_to_evidence:{index}")
        grounded_alternatives = [
            value for value in item.alternatives
            if any(term_in_text(span.text, value) for span in spans)
        ]
        if len(grounded_alternatives) != len(item.alternatives):
            warnings.append(f"dropped_unsupported_alternative:{index}")
        if item.modality == "alternative" and not grounded_alternatives:
            grounded_alternatives = _alternatives(source_text, "alternative")
        requirement_id = _stable_id("req", source_text, spans[0].start)
        index_to_id[index] = requirement_id
        requirements.append(JDRequirement(
            requirement_id=requirement_id, text=source_text,
            kind=item.kind, modality=item.modality, priority=item.priority,
            alternatives=grounded_alternatives, evidence_spans=spans,
        ))
    for value in draft.responsibilities:
        span = _find_span(jd_text, value)
        if span is None or any(item.text == span.text.strip() for item in requirements):
            continue
        requirement_id = _stable_id("req", span.text.strip(), span.start)
        requirements.append(JDRequirement(
            requirement_id=requirement_id, text=span.text.strip(),
            kind="responsibility", modality="unknown", priority=0.25,
            alternatives=[], evidence_spans=[span],
        ))
    valid_ids = {item.requirement_id for item in requirements}
    keywords: list[JDKeyword] = []
    seen: set[str] = set()
    for index, item in enumerate(draft.keywords):
        spans = [span for value in item.evidence_spans
                 if (span := _find_span(jd_text, value)) is not None]
        forms = [value for value in item.surface_forms
                 if _find_term_span(jd_text, value) is not None]
        requirement_ids = [index_to_id[i] for i in item.requirement_indexes if i in index_to_id]
        if not spans:
            spans = [span for value in forms
                     if (span := _find_term_span(jd_text, value)) is not None]
        if not spans or not forms:
            warnings.append(f"dropped_ungrounded_keyword:{index}:{item.canonical_name}")
            continue
        canonical = item.canonical_name.strip()
        if not any(term_in_text(span.text, canonical) for span in spans):
            # A free-form canonical name cannot be proven equivalent to a
            # grounded surface form without another semantic model. Preserve
            # the exact surface instead of accepting a hallucinated concept.
            canonical = forms[0]
            warnings.append(f"canonicalized_keyword_to_surface:{index}")
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(JDKeyword(
            canonical_name=canonical,
            surface_forms=list(dict.fromkeys(forms or [span.text for span in spans])),
            category=item.category, importance=item.importance,
            requirement_ids=[value for value in requirement_ids if value in valid_ids],
            evidence_spans=spans,
        ))
    title0, company0, location0, job_type0 = _field_fallback(jd_text)

    def grounded_field(name: str, candidate: str, fallback: str) -> str:
        value = str(candidate or "").strip()
        if not value:
            return fallback
        if _find_span(jd_text, value) is not None:
            return value
        warnings.append(f"dropped_ungrounded_field:{name}")
        return fallback

    grounded_responsibilities = [
        value for value in draft.responsibilities if _find_span(jd_text, value) is not None
    ]
    return JDAnalysis(
        input_hash=_input_hash(jd_text), source="llm", model=model,
        title=grounded_field("title", draft.title, title0),
        company=grounded_field("company", draft.company, company0),
        location=grounded_field("location", draft.location, location0),
        job_type=grounded_field("job_type", draft.job_type, job_type0),
        responsibilities=list(dict.fromkeys(grounded_responsibilities)),
        requirements=requirements, keywords=keywords,
        warnings=list(dict.fromkeys(warnings)),
    )


def _resolved_model() -> str:
    try:
        from day1_api_starter import load_local_env
        load_local_env()
    except Exception:
        pass
    explicit = os.environ.get("JD_ANALYSIS_MODEL", "").strip()
    if explicit:
        return explicit
    try:
        from day1_api_starter import get_llm_config
        return str((get_llm_config() or {}).get("model") or "")
    except Exception:
        return ""


def _cache_path(jd_text: str, model: str, cache_dir: str | Path | None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else (
        Path(__file__).resolve().parent / ".offerclaw" / "cache" / "jd_analysis"
    )
    key = hashlib.sha256(
        f"{_input_hash(jd_text)}|{JD_ANALYSIS_SCHEMA_VERSION}|"
        f"{JD_ANALYZER_VERSION}|{model}".encode("utf-8")
    ).hexdigest()
    return base / f"{key}.json"


def _cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.Lock())


def _load_cache(path: Path) -> JDAnalysis | None:
    try:
        value = JDAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
        return value.model_copy(update={
            "source": "cache",
            "warnings": list(value.warnings) + [f"cached_from:{value.source}"],
        })
    except Exception:
        return None


def _save_cache(path: Path, analysis: JDAnalysis) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        pass


def _shadow_log(jd_text: str, deterministic: JDAnalysis,
                intelligent: JDAnalysis | None, meta: StructuredCallMeta) -> None:
    try:
        path = Path(__file__).resolve().parent / ".offerclaw" / "jd_analysis_shadow.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "input_hash": _input_hash(jd_text),
            "deterministic_requirements": len(deterministic.requirements),
            "deterministic_keywords": len(deterministic.keywords),
            "intelligent_requirements": len(intelligent.requirements) if intelligent else 0,
            "intelligent_keywords": len(intelligent.keywords) if intelligent else 0,
            "schema_valid": intelligent is not None,
            "repair_used": meta.repair_used,
            "errors": meta.errors,
            "model": meta.model,
            "gateway": meta.gateway,
            "reasoning_effort": meta.reasoning_effort,
            "structured_output_mode": meta.structured_output_mode,
            "json_capability": meta.json_capability,
            "queue_ms": round(meta.planner_queue_ms, 1),
            "provider_ms": round(meta.planner_provider_ms, 1),
            "wall_ms": round(meta.planner_wall_ms, 1),
            "deadline_ms": round(meta.planner_deadline_ms, 1),
            "timeout_stage": meta.planner_timeout_stage,
            "late_response": meta.planner_late_response,
            "circuit_state": meta.planner_circuit_state,
            "prompt_tokens": meta.prompt_tokens,
            "completion_tokens": meta.completion_tokens,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def analyze_jd(jd_text: str, *, mode: str | None = None,
               model: str | None = None, cache_dir: str | Path | None = None,
               caller: TextCaller | None = None) -> JDAnalysis:
    """Analyze one JD once and return a source-grounded, versioned contract."""
    text = (jd_text or "").strip()
    deterministic = _deterministic_analysis(text)
    # Deployments enable shadow explicitly; library consumers stay deterministic.
    selected_mode = (mode or os.environ.get("JD_ANALYZER_MODE", "deterministic")).strip().lower()
    if selected_mode not in {"deterministic", "shadow", "intelligent"}:
        selected_mode = "deterministic"
    if selected_mode == "deterministic" or not text:
        return deterministic
    route_model = model if model is not None else _resolved_model()
    if not route_model and caller is None:
        return deterministic.model_copy(update={
            "warnings": deterministic.warnings + ["llm_unavailable:no_model"],
        })
    cache_path = _cache_path(text, route_model or "injected", cache_dir)
    # Keyed single-flight: callers analyzing the same JD in one process share
    # the first result. The cache is rechecked after acquiring the lock.
    with _cache_lock(cache_path):
        cached = _load_cache(cache_path)
        if cached is not None:
            if selected_mode == "shadow":
                _shadow_log(text, deterministic, cached, StructuredCallMeta(model=route_model))
                return deterministic.model_copy(update={
                    "warnings": deterministic.warnings + ["shadow_candidate_from_cache"],
                })
            return cached
        draft, meta = call_structured(
            _JDAnalysisDraft, _analysis_messages(text), model=route_model or None,
            timeout_seconds=float(os.environ.get("JD_ANALYSIS_TIMEOUT_SECONDS", "30") or 30),
            max_tokens=3500, repair=True, caller=caller,
            reasoning_effort=os.environ.get(
                "JD_ANALYSIS_REASONING_EFFORT", "low"
            ).strip() or None,
            lane="shadow" if selected_mode == "shadow" else "online",
        )
        intelligent = _ground_draft(text, draft, model=route_model) if draft is not None else None
        if intelligent is not None:
            _save_cache(cache_path, intelligent)
    if selected_mode == "shadow":
        _shadow_log(text, deterministic, intelligent, meta)
        return deterministic.model_copy(update={
            "warnings": deterministic.warnings + [
                "shadow_candidate_valid" if intelligent else "shadow_candidate_invalid"
            ],
        })
    if intelligent is None:
        return deterministic.model_copy(update={
            "warnings": deterministic.warnings + [
                "llm_fallback:" + (",".join(meta.errors) or "unknown")
            ],
        })
    return intelligent


def legacy_to_analysis(jd_text: str, legacy: dict[str, Any] | None) -> JDAnalysis:
    """One-cycle checkpoint adapter. New checkpoints must store jd_analysis only."""
    if not legacy:
        return _deterministic_analysis(jd_text)
    deterministic = _deterministic_analysis(jd_text)
    names = [str(item) for item in legacy.get("keywords", []) if str(item).strip()]
    existing = {item.canonical_name.lower() for item in deterministic.keywords}
    keywords = list(deterministic.keywords)
    warnings = list(deterministic.warnings) + ["adapted_from_legacy_jd_struct"]
    for name in names:
        if name.lower() in existing:
            continue
        span = _find_span(jd_text, name)
        if span is None:
            warnings.append(f"legacy_keyword_dropped_ungrounded:{name}")
            continue
        keywords.append(JDKeyword(
            canonical_name=name, surface_forms=[span.text], category="legacy",
            importance=0.35, evidence_spans=[span],
        ))
    return deterministic.model_copy(update={
        "source": "legacy", "keywords": keywords, "warnings": warnings,
    })


def parse_jd(jd_text: str, *, use_llm: bool = False) -> dict[str, Any]:
    """Deprecated six-key adapter kept for one compatibility cycle."""
    analysis = analyze_jd(jd_text, mode="intelligent" if use_llm else "deterministic")
    return {
        "hard_requirements": [item.text for item in analysis.requirements
                              if item.kind == "hard" and item.modality == "required"],
        "nice_to_have": [item.text for item in analysis.requirements
                         if item.modality == "preferred"],
        "keywords": analysis.keyword_names(),
        "responsibilities": list(analysis.responsibilities),
        "title": analysis.title,
        "_source": analysis.source,
    }
