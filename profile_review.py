# -*- coding: utf-8 -*-
"""Evidence-grounded profile review and versioned profile storage.

The LLM may extract and compare meaning, but every persisted evidence record and
profile patch is grounded and authorized by deterministic code.  SQLite is the
source of truth; ``user_profile.md`` is a human-readable export/import surface.
"""
from __future__ import annotations

import copy
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from io_utils import atomic_write_text
from memory_models import EventEnvelope, validate_payload
from memory_store import MemoryStore, new_id, now_iso
from structured_llm import StructuredCallMeta, TextCaller, call_structured


PROFILE_SCHEMA_VERSION = "profile-v2"
PROFILE_REVIEW_VERSION = "profile-review-v2"
PROFILE_ID = "user_profile"

ActivityLevel = Literal["observed", "practiced", "delivered"]
VerificationType = Literal[
    "source_grounded", "artifact_checked", "test_result",
    "external_result", "user_attested",
]
SuggestionStatus = Literal[
    "generating", "pending", "needs_evidence", "accepted", "modified",
    "rejected", "stale", "review_unavailable",
]

SKILL_RUBRIC = {
    "1": "了解概念，尚不能完成实际任务",
    "2": "能在示例或指导下完成练习",
    "3": "能独立完成明确范围任务，并有可验证交付物",
    "4": "有多次端到端交付，包含评估、排障或维护证据",
    "5": "有重复真实场景成果，并能完成架构决策或指导他人",
}

DEFAULT_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {"capability_id": "cap_ai_engineering", "name": "AI 应用工程", "aliases": ["AI 应用开发"],
     "parent_id": None, "node_type": "domain"},
    {"capability_id": "cap_programming", "name": "编程与工程", "aliases": ["软件工程"],
     "parent_id": None, "node_type": "domain"},
    {"capability_id": "cap_rag_engineering", "name": "RAG 工程", "aliases": ["RAG", "检索增强生成", "向量检索"],
     "parent_id": "cap_ai_engineering", "node_type": "capability"},
    {"capability_id": "cap_rag_chunking", "name": "文档解析与分块", "aliases": ["分块", "chunking"],
     "parent_id": "cap_rag_engineering", "node_type": "atomic"},
    {"capability_id": "cap_rag_retrieval", "name": "检索", "aliases": ["语义检索", "混合检索", "BM25"],
     "parent_id": "cap_rag_engineering", "node_type": "atomic"},
    {"capability_id": "cap_rag_rerank", "name": "重排", "aliases": ["rerank", "reranker"],
     "parent_id": "cap_rag_engineering", "node_type": "atomic"},
    {"capability_id": "cap_rag_evaluation", "name": "RAG 评估", "aliases": ["RAGAS", "检索评估"],
     "parent_id": "cap_rag_engineering", "node_type": "atomic"},
    {"capability_id": "cap_rag_deployment", "name": "RAG 部署", "aliases": ["服务部署"],
     "parent_id": "cap_rag_engineering", "node_type": "atomic"},
    {"capability_id": "cap_rag_diagnosis", "name": "RAG 诊断", "aliases": ["检索诊断", "故障诊断"],
     "parent_id": "cap_rag_engineering", "node_type": "atomic"},
    {"capability_id": "cap_agent_workflow", "name": "Agent / Workflow", "aliases": ["Agent", "工作流编排"],
     "parent_id": "cap_ai_engineering", "node_type": "capability"},
    {"capability_id": "cap_langgraph", "name": "LangGraph", "aliases": ["LangGraph 工作流", "LangGraph 状态机", "状态图编排"],
     "parent_id": "cap_agent_workflow", "node_type": "atomic"},
    {"capability_id": "cap_llm_api", "name": "LLM API 调用", "aliases": ["模型 API 调用", "function calling"],
     "parent_id": "cap_ai_engineering", "node_type": "atomic"},
    {"capability_id": "cap_prompt", "name": "Prompt 设计", "aliases": ["Prompt 工程", "提示词工程"],
     "parent_id": "cap_ai_engineering", "node_type": "atomic"},
    {"capability_id": "cap_python", "name": "Python 工程", "aliases": ["Python"],
     "parent_id": "cap_programming", "node_type": "atomic"},
    {"capability_id": "cap_fastapi", "name": "FastAPI", "aliases": ["FastAPI 接口开发"],
     "parent_id": "cap_programming", "node_type": "atomic"},
    {"capability_id": "cap_data_processing", "name": "数据处理", "aliases": ["数据分析"],
     "parent_id": "cap_programming", "node_type": "atomic"},
    {"capability_id": "cap_frontend", "name": "前端基础", "aliases": ["前端开发"],
     "parent_id": "cap_programming", "node_type": "atomic"},
    {"capability_id": "cap_matlab", "name": "MATLAB", "aliases": [],
     "parent_id": "cap_programming", "node_type": "atomic"},
    {"capability_id": "cap_unmapped", "name": "待归类能力", "aliases": [],
     "parent_id": None, "node_type": "capability"},
)


class ProfileSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    title: str
    body_markdown: str = ""


class ProfileSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment_id: str
    capability_id: str
    name: str
    level: int = Field(ge=1, le=5)
    source_section_id: str
    source_line: int = Field(ge=0)
    source_format: Literal["inline_score", "table_score"]
    scope: str = "global"
    evidence_ids: list[str] = Field(default_factory=list)
    last_verified_at: str = ""


class ProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = PROFILE_SCHEMA_VERSION
    preamble_markdown: str
    sections: list[ProfileSection]
    matching_fields: dict[str, Any]
    skills: list[ProfileSkill]
    legacy_notes: list[str] = Field(default_factory=list)


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1, max_length=500)
    source_event_id: str = Field(min_length=1, max_length=96)
    source_quote: str = Field(min_length=1, max_length=1200)
    capability_id: str = Field(default="", max_length=96)
    new_capability_candidate: str = Field(default="", max_length=120)
    activity_level: ActivityLevel
    scope: str = Field(default="global", max_length=240)
    verification_candidate: VerificationType = "source_grounded"
    criteria_ids: list[str] = Field(default_factory=list, max_length=12)
    explicit_statement: bool = False
    relation: Literal["direct", "partial", "transferable", "unrelated"] = "direct"
    rationale: str = Field(default="", max_length=600)


class EvidenceExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[EvidenceCandidate] = Field(default_factory=list, max_length=120)


class NewCapabilityDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    parent_id: str = Field(default="", max_length=96)
    aliases: list[str] = Field(default_factory=list, max_length=12)


class ProfileChangeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_path: str = Field(min_length=1, max_length=240)
    current_value: Any = None
    observed_change: str = Field(min_length=1, max_length=600)
    operation: Literal["add", "replace", "remove"]
    proposed_value: Any = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    counter_evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    capability_id: str = Field(default="", max_length=96)
    new_capability: NewCapabilityDraft | None = None
    rationale: str = Field(min_length=1, max_length=800)
    requirement_status: Literal["satisfied", "needs_evidence", "conflict"] = "satisfied"


class ProfileAuditDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changes: list[ProfileChangeDraft] = Field(default_factory=list, max_length=40)


def _hash_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(_norm(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _default_capability_id(label: str) -> str:
    normalized = _norm(label)
    exact: dict[str, str] = {}
    for item in DEFAULT_CAPABILITIES:
        for alias in [item["name"], *(item.get("aliases") or [])]:
            exact[_norm(alias)] = item["capability_id"]
    if normalized in exact:
        return exact[normalized]
    matches = [(len(alias), capability_id) for alias, capability_id in exact.items()
               if len(alias) >= 2 and (alias in normalized or normalized in alias)]
    return max(matches, default=(0, ""))[1]


def _profile_sections(content: str) -> tuple[str, list[ProfileSection]]:
    matches = list(re.finditer(r"(?m)^##\s*(\d+)\.\s*([^\n]+)\s*$", content or ""))
    if not matches:
        return content.strip(), []
    preamble = content[:matches[0].start()].rstrip()
    sections: list[ProfileSection] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[match.end():end].strip("\r\n")
        sections.append(ProfileSection(
            section_id=match.group(1), title=match.group(2).strip(), body_markdown=body,
        ))
    return preamble, sections


def _matching_fields(content: str) -> dict[str, Any]:
    from profile_loader import (
        parse_acceptable_locations, parse_directions, parse_education,
        parse_explicit_not, parse_industry_preference, parse_location,
        parse_major, parse_salary, parse_target_job_type, parse_work_mode,
    )

    def line(label: str) -> str:
        match = re.search(
            rf"(?m)^\s*-?\s*{re.escape(label)}\s*[：:]\s*(.+?)\s*$", content,
        )
        return match.group(1).strip() if match else ""

    return {
        "姓名": line("姓名 / 昵称") or line("姓名"),
        "学历": parse_education(content), "专业": parse_major(content),
        "所在地": parse_location(content), "可接受地域": parse_acceptable_locations(content),
        "方向优先级": parse_directions(content), "目标岗位类型": parse_target_job_type(content),
        "行业偏好": parse_industry_preference(content), "明确不做": parse_explicit_not(content),
        "工作性质偏好": parse_work_mode(content), "期望薪资": parse_salary(content),
        "兴趣爱好": line("兴趣爱好"), "擅长的思维方式": line("擅长的思维方式"),
        "不擅长或不感兴趣的方向": line("不擅长或不感兴趣的方向"),
        "每天可投入": line("每天可投入（小时）"), "每周可投入": line("每周可投入（小时）"),
        "黄金时段": line("黄金时段"), "不可打扰时段": line("不可打扰时段"),
    }


def _profile_skills(sections: list[ProfileSection]) -> list[ProfileSkill]:
    out: list[ProfileSkill] = []
    for section in sections:
        if section.section_id not in {"3", "9"}:
            continue
        for line_index, raw in enumerate(section.body_markdown.splitlines()):
            if section.section_id == "3":
                match = re.match(r"^\s*-\s*([^：:]+)[：:]\s*.*?(?<!\d)([1-5])/5(?:\D|$)", raw)
                source_format = "inline_score"
            else:
                match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([1-5])\s*\|", raw)
                source_format = "table_score"
            if not match:
                continue
            name, level = match.group(1).strip(), int(match.group(2))
            capability_id = _default_capability_id(name) or _stable_id("cap", name)
            out.append(ProfileSkill(
                assessment_id=_stable_id("asmt", section.section_id, line_index, name),
                capability_id=capability_id, name=name, level=level,
                source_section_id=section.section_id, source_line=line_index,
                source_format=source_format,
            ))
    return out


def parse_profile_markdown(content: str) -> ProfileSpec:
    if not content.strip():
        raise ValueError("画像内容不能为空")
    preamble, sections = _profile_sections(content)
    required = {"0", "2", "3", "9", "10"}
    missing = sorted(required - {item.section_id for item in sections})
    if missing:
        raise ValueError("画像缺少必要章节：" + "、".join(f"§{item}" for item in missing))
    return ProfileSpec(
        preamble_markdown=preamble, sections=sections,
        matching_fields=_matching_fields(content), skills=_profile_skills(sections),
    )


def render_profile_markdown(spec: ProfileSpec) -> str:
    pieces = [spec.preamble_markdown.rstrip()]
    for section in spec.sections:
        pieces.append(f"## {section.section_id}. {section.title}\n{section.body_markdown.rstrip()}")
    return "\n\n".join(piece for piece in pieces if piece).rstrip() + "\n"


def _profile_diff(before: ProfileSpec, after: ProfileSpec) -> dict[str, Any]:
    changed_fields = []
    keys = sorted(set(before.matching_fields) | set(after.matching_fields))
    for key in keys:
        if before.matching_fields.get(key) != after.matching_fields.get(key):
            changed_fields.append({
                "field_path": f"/fields/{key}", "before": before.matching_fields.get(key),
                "after": after.matching_fields.get(key),
            })
    before_skills = {item.assessment_id: item.model_dump(mode="json") for item in before.skills}
    after_skills = {item.assessment_id: item.model_dump(mode="json") for item in after.skills}
    changed_skills = []
    for key in sorted(set(before_skills) | set(after_skills)):
        if before_skills.get(key) != after_skills.get(key):
            changed_skills.append({"assessment_id": key, "before": before_skills.get(key),
                                   "after": after_skills.get(key)})
    section_changes = [item.section_id for item in after.sections if next(
        (old.body_markdown for old in before.sections if old.section_id == item.section_id), None,
    ) != item.body_markdown]
    return {"changed_fields": changed_fields, "changed_skills": changed_skills,
            "changed_sections": section_changes}


def _carry_skill_identity(before: ProfileSpec, after: ProfileSpec) -> ProfileSpec:
    """Preserve assessment identity and evidence across Markdown re-parsing."""
    by_capability: dict[str, list[ProfileSkill]] = {}
    for skill in before.skills:
        by_capability.setdefault(skill.capability_id, []).append(skill)
    used: set[str] = set()
    for skill in after.skills:
        matches = [item for item in by_capability.get(skill.capability_id, [])
                   if item.assessment_id not in used]
        if not matches:
            matches = [item for item in before.skills
                       if _norm(item.name) == _norm(skill.name) and item.assessment_id not in used]
        if len(matches) != 1:
            continue
        previous = matches[0]
        used.add(previous.assessment_id)
        skill.assessment_id = previous.assessment_id
        skill.capability_id = previous.capability_id
        skill.scope = previous.scope
        skill.evidence_ids = list(previous.evidence_ids)
        skill.last_verified_at = previous.last_verified_at
    return after


class ProfileConflictError(RuntimeError):
    pass


class ProfileValidationError(ValueError):
    pass


class ProfileRepository:
    def __init__(self, store: MemoryStore | None = None, *,
                 profile_path: str | Path | None = None) -> None:
        self.store = store or MemoryStore()
        self.profile_path = Path(profile_path or Path(__file__).resolve().parent / "user_profile.md")
        self._ensure_initialized()

    def _seed_capabilities(self, conn) -> None:
        stamp = now_iso()
        for item in DEFAULT_CAPABILITIES:
            conn.execute(
                """INSERT OR IGNORE INTO profile_capabilities(
                    capability_id,name,aliases_json,parent_id,node_type,rubric_json,
                    lifecycle,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                (item["capability_id"], item["name"],
                 json.dumps(item.get("aliases") or [], ensure_ascii=False),
                 item.get("parent_id"), item["node_type"],
                 json.dumps(SKILL_RUBRIC, ensure_ascii=False), "active", stamp, stamp),
            )

    def _ensure_profile_capabilities(self, conn, spec: ProfileSpec) -> None:
        stamp = now_iso()
        for skill in spec.skills:
            conn.execute(
                """INSERT OR IGNORE INTO profile_capabilities(
                    capability_id,name,aliases_json,parent_id,node_type,rubric_json,
                    lifecycle,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                (skill.capability_id, skill.name, "[]", None, "atomic",
                 json.dumps(SKILL_RUBRIC, ensure_ascii=False), "active", stamp, stamp),
            )

    def _backup_legacy_sources(self) -> list[str]:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return []
        sources = [self.profile_path, self.profile_path.parent / "logs" / "profile" / "suggestions.jsonl"]
        existing = [path for path in sources if path.exists()]
        if not existing:
            return []
        backup_dir = self.store.base_dir / "profile_backups" / (
            dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + new_id("backup").split("_", 1)[1][:8]
        )
        backup_dir.mkdir(parents=True, exist_ok=False)
        saved = []
        for source in existing:
            target = backup_dir / source.name
            shutil.copy2(source, target)
            saved.append(str(target))
        return saved

    def _legacy_suggestion_states(self) -> list[dict[str, Any]]:
        """Read the old append-only proposal log without trusting its conclusions."""
        path = self.profile_path.parent / "logs" / "profile" / "suggestions.jsonl"
        if not path.exists():
            return []
        states: dict[str, dict[str, Any]] = {}
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                raw = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            suggestion_id = str(raw.get("suggestion_id") or "").strip()
            if not suggestion_id:
                continue
            if raw.get("action") == "proposed" and isinstance(raw.get("suggestion"), dict):
                states[suggestion_id] = {
                    **raw["suggestion"], "suggestion_id": suggestion_id,
                    "legacy_line": line_no,
                }
            elif raw.get("action") == "decision" and suggestion_id in states:
                states[suggestion_id]["legacy_decision"] = str(raw.get("decision") or "")
        return list(states.values())

    def _import_legacy_suggestions(self, conn, *, revision: int,
                                   target_context_id: str) -> int:
        """Keep old proposals for audit while preventing them from becoming patches."""
        rows = self._legacy_suggestion_states()
        if not rows:
            return 0
        stamp, audit_id = now_iso(), new_id("profile_audit")
        conn.execute(
            """INSERT INTO profile_audits(
                audit_id,base_revision,target_context_id,trigger_kind,evidence_cutoff,status,
                model_meta_json,error,created_at,completed_at)
               VALUES(?,?,?,?,?,'completed',?,'',?,?)""",
            (audit_id, revision, target_context_id, "legacy_migration", stamp,
             json.dumps({"source": "legacy_suggestions_jsonl", "trusted": False}), stamp, stamp),
        )
        imported = 0
        for raw in rows:
            legacy_id = str(raw.get("suggestion_id") or new_id("legacy_profile_suggestion"))
            suggestion_id = legacy_id if not conn.execute(
                "SELECT 1 FROM profile_suggestions WHERE suggestion_id=?", (legacy_id,),
            ).fetchone() else new_id("legacy_profile_suggestion")
            current_value = raw.get("current_text") or raw.get("target_anchor") or ""
            proposed_value = raw.get("proposed_text") or ""
            evidence_hash = str(raw.get("evidence_hash") or _hash_text(json.dumps(
                raw.get("evidence_refs") or [], ensure_ascii=False, sort_keys=True,
            )))
            conn.execute(
                """INSERT INTO profile_suggestions(
                    suggestion_id,audit_id,base_revision,target_context_id,field_path,operation,
                    current_value_json,proposed_value_json,observed_change,evidence_ids_json,
                    counter_evidence_ids_json,rationale,requirement_status,status,capability_id,
                    new_capability_json,evidence_hash,model_meta_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'stale',NULL,'{}',?,?,?,?)""",
                (suggestion_id, audit_id, revision, target_context_id,
                 f"/legacy/section/{raw.get('target_section') or 'unknown'}", "replace",
                 json.dumps(current_value, ensure_ascii=False),
                 json.dumps(proposed_value, ensure_ascii=False),
                 "旧版机械画像建议，仅保留审计记录", "[]", "[]",
                 "旧链路没有可核验的原始事件引用，迁移后不得批准。", "needs_evidence",
                 evidence_hash, json.dumps({
                     "source": "legacy_suggestions_jsonl", "trusted": False,
                     "legacy_line": raw.get("legacy_line"),
                     "legacy_decision": raw.get("legacy_decision", ""),
                 }, ensure_ascii=False), str(raw.get("created_at") or stamp), stamp),
            )
            imported += 1
        return imported

    def _ensure_initialized(self) -> None:
        with self.store._connect() as conn:
            if conn.execute("SELECT 1 FROM profile_state WHERE profile_id=?", (PROFILE_ID,)).fetchone():
                return
        if not self.profile_path.exists():
            raise FileNotFoundError(f"画像文件不存在：{self.profile_path}")
        content = self.profile_path.read_text(encoding="utf-8")
        spec = parse_profile_markdown(content)
        backups = self._backup_legacy_sources()
        stamp, revision_id = now_iso(), new_id("profile_rev")
        with self.store.transaction() as conn:
            # Another process may have initialized the profile while this one
            # parsed the legacy Markdown and created its backup.
            if conn.execute(
                    "SELECT 1 FROM profile_state WHERE profile_id=?", (PROFILE_ID,)).fetchone():
                return
            self._seed_capabilities(conn)
            self._ensure_profile_capabilities(conn, spec)
            cursor = conn.execute(
                """INSERT INTO profile_revisions(
                    revision_id,schema_version,profile_json,content_md,content_hash,
                    reason,actor,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (revision_id, PROFILE_SCHEMA_VERSION, spec.model_dump_json(), content,
                 _hash_text(content), "从 user_profile.md 初始化", "import", stamp),
            )
            revision = int(cursor.lastrowid)
            conn.execute("INSERT INTO profile_state VALUES(?,?,?)", (PROFILE_ID, revision, stamp))
            legacy_suggestions = self._import_legacy_suggestions(
                conn, revision=revision, target_context_id=self.store.active_goal_id(),
            )
            report = {"status": "ok", "source": str(self.profile_path),
                      "backups": backups, "legacy_notes": spec.legacy_notes,
                      "skill_count": len(spec.skills),
                      "legacy_suggestions_staled": legacy_suggestions}
            conn.execute(
                "INSERT INTO profile_migrations VALUES(?,?,?,?,?,?)",
                (new_id("profile_migration"), str(self.profile_path), _hash_text(content),
                 revision, json.dumps(report, ensure_ascii=False), stamp),
            )

    @staticmethod
    def _revision_value(row) -> dict[str, Any]:
        spec = ProfileSpec.model_validate_json(row["profile_json"])
        return {
            "revision": int(row["revision"]), "revision_id": row["revision_id"],
            "profile_spec": spec.model_dump(mode="json"), "content_md": row["content_md"],
            "base_hash": row["content_hash"], "created_at": row["created_at"],
            "reason": row["reason"], "actor": row["actor"],
        }

    def current(self) -> dict[str, Any]:
        with self.store._connect() as conn:
            row = self._current_row(conn)
        if not row:
            raise RuntimeError("正式画像尚未初始化")
        return self._revision_value(row)

    @staticmethod
    def _current_row(conn):
        return conn.execute(
            """SELECT r.* FROM profile_state s JOIN profile_revisions r
               ON r.revision=s.current_revision WHERE s.profile_id=?""", (PROFILE_ID,),
        ).fetchone()

    def capabilities(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM profile_capabilities"
        if not include_archived:
            sql += " WHERE lifecycle!='archived'"
        sql += " ORDER BY node_type,name"
        with self.store._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [{**dict(row), "aliases": json.loads(row["aliases_json"]),
                 "rubric": json.loads(row["rubric_json"])} for row in rows]

    def export_current(self) -> dict[str, Any]:
        current = self.current()
        atomic_write_text(str(self.profile_path), current["content_md"])
        try:
            from profile_loader import reset_cache
            reset_cache()
        except Exception:
            pass
        return {"path": str(self.profile_path), "content_hash": current["base_hash"]}

    def _replay_profile_edit(self, operation_id: str) -> dict[str, Any] | None:
        if not operation_id:
            return None
        event = self.store.get_event_by_operation(operation_id)
        if not event or event.get("kind") != "profile_edited":
            return None
        content_hash = str(event.get("profile_hash") or "")
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profile_revisions WHERE content_hash=? ORDER BY revision LIMIT 1",
                (content_hash,),
            ).fetchone()
        if not row:
            return None
        return {"status": "ok", **self._revision_value(row),
                "memory_event_id": event["event_id"], "replayed": True}

    def create_edit_preview(self, content_md: str, *, base_revision: int) -> dict[str, Any]:
        current = self.current()
        if int(base_revision) != current["revision"]:
            raise ProfileConflictError("画像版本已变化，请刷新后重试")
        try:
            candidate = parse_profile_markdown(content_md)
        except ValueError as exc:
            raise ProfileValidationError(str(exc)) from exc
        before = ProfileSpec.model_validate(current["profile_spec"])
        candidate = _carry_skill_identity(before, candidate)
        diff = _profile_diff(before, candidate)
        if not any(diff.values()):
            return {
                "status": "no_changes",
                "base_revision": current["revision"],
                "diff": diff,
            }
        preview_id, stamp = new_id("profile_preview"), now_iso()
        expires = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)).isoformat(timespec="seconds")
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO profile_edit_previews VALUES(?,?,?,?,?,?,?,NULL,?)""",
                (preview_id, current["revision"], candidate.model_dump_json(), content_md,
                 _hash_text(content_md), json.dumps(diff, ensure_ascii=False), expires, stamp),
            )
        return {"status": "preview", "preview_id": preview_id,
                "base_revision": current["revision"], "diff": diff, "expires_at": expires}

    def _event_envelope(self, kind: str, payload: dict[str, Any], *, actor: str,
                        source: str, operation_id: str, entity_type: str,
                        entity_id: str, target_context_id: str | None = None) -> dict[str, Any]:
        validated = validate_payload(kind, payload)
        stamp = now_iso()
        return EventEnvelope(
            event_id=new_id("ep"), schema_version=1, kind=kind,
            occurred_at=stamp, recorded_at=stamp, business_date=dt.date.today().isoformat(),
            actor=actor, source=source, traffic_origin="organic", operation_id=operation_id,
            target_context_id=target_context_id or self.store.active_goal_id(),
            entity_type=entity_type, entity_id=entity_id, payload=validated,
        ).model_dump(mode="json")

    def _insert_revision(self, conn, spec: ProfileSpec, content_md: str, *,
                         reason: str, actor: str) -> int:
        cursor = conn.execute(
            """INSERT INTO profile_revisions(
                revision_id,schema_version,profile_json,content_md,content_hash,
                reason,actor,created_at) VALUES(?,?,?,?,?,?,?,?)""",
            (new_id("profile_rev"), PROFILE_SCHEMA_VERSION, spec.model_dump_json(),
             content_md, _hash_text(content_md), reason[:500], actor, now_iso()),
        )
        revision = int(cursor.lastrowid)
        conn.execute(
            "UPDATE profile_state SET current_revision=?,updated_at=? WHERE profile_id=?",
            (revision, now_iso(), PROFILE_ID),
        )
        self._ensure_profile_capabilities(conn, spec)
        return revision

    def commit_edit_preview(self, preview_id: str, *, reason: str = "",
                            operation_id: str = "") -> dict[str, Any]:
        operation_id = operation_id or new_id("op")
        replay = self._replay_profile_edit(operation_id)
        if replay:
            return replay
        with self.store._connect() as conn:
            preview = conn.execute(
                "SELECT * FROM profile_edit_previews WHERE preview_id=?", (preview_id,),
            ).fetchone()
        if not preview or preview["committed_at"]:
            raise KeyError("画像编辑预览不存在或已提交")
        if dt.datetime.fromisoformat(preview["expires_at"]) < dt.datetime.now(dt.timezone.utc):
            raise ProfileConflictError("画像编辑预览已过期，请重新预览")
        current = self.current()
        if current["revision"] != int(preview["base_revision"]):
            raise ProfileConflictError("画像版本已变化，请重新预览")
        spec = ProfileSpec.model_validate_json(preview["profile_json"])
        content = preview["content_md"]
        before_snapshot = f"snap_{_hash_text(current['content_md'])}"
        after_snapshot = f"snap_{_hash_text(content)}"
        event = self._event_envelope(
            "profile_edited",
            {"before_snapshot_id": before_snapshot, "after_snapshot_id": after_snapshot,
             "profile_hash": _hash_text(content), "reason": reason[:500]},
            actor="user", source="profile_review", operation_id=operation_id,
            entity_type="profile", entity_id=PROFILE_ID,
        )
        replayed = False
        with self.store.transaction() as conn:
            existing = conn.execute("SELECT * FROM events WHERE operation_id=?", (operation_id,)).fetchone()
            if existing:
                replayed = True
            else:
                live_preview = conn.execute(
                    "SELECT * FROM profile_edit_previews WHERE preview_id=?", (preview_id,),
                ).fetchone()
                live_current = self._current_row(conn)
                if not live_preview or live_preview["committed_at"]:
                    raise ProfileConflictError("画像编辑预览已被其他操作提交")
                if not live_current or int(live_current["revision"]) != int(preview["base_revision"]):
                    raise ProfileConflictError("画像版本已变化，请重新预览")
                self.store.put_snapshot_in_transaction(
                    conn, current["content_md"], media_type="text/markdown", source_path="user_profile.md",
                )
                self.store.put_snapshot_in_transaction(
                    conn, content, media_type="text/markdown", source_path="user_profile.md",
                )
                revision = self._insert_revision(conn, spec, content, reason=reason, actor="user")
                conn.execute("UPDATE profile_edit_previews SET committed_at=? WHERE preview_id=?",
                             (now_iso(), preview_id))
                conn.execute("UPDATE profile_suggestions SET status='stale',updated_at=? "
                             "WHERE status IN ('pending','needs_evidence') AND base_revision!=?",
                             (now_iso(), revision))
                self.store.insert_event_in_transaction(conn, event)
        if replayed:
            replay = self._replay_profile_edit(operation_id)
            if replay:
                return replay
            raise ProfileConflictError("画像保存已由其他操作完成，请刷新")
        export_error = ""
        try:
            self.export_current()
        except Exception as exc:
            export_error = str(exc)
        result = {"status": "ok", **self.current(), "memory_event_id": event["event_id"]}
        if export_error:
            result["export_error"] = export_error
        return result

    def save_markdown(self, content_md: str, *, base_hash: str, reason: str = "",
                      operation_id: str = "") -> dict[str, Any]:
        replay = self._replay_profile_edit(operation_id)
        if replay:
            return replay
        current = self.current()
        if not base_hash:
            raise ProfileValidationError("保存画像必须提交当前内容哈希")
        if base_hash != current["base_hash"]:
            raise ProfileConflictError("画像已被其他操作更新，请刷新后重试")
        preview = self.create_edit_preview(content_md, base_revision=current["revision"])
        return self.commit_edit_preview(
            preview["preview_id"], reason=reason, operation_id=operation_id,
        )

    @staticmethod
    def _source_allowed(event: dict[str, Any]) -> bool:
        if event.get("deleted_at") or event.get("archived"):
            return False
        if event.get("actor") not in {"user", "import"}:
            return False
        kind = event.get("kind")
        if kind == "daily_log_recorded":
            return True
        if kind == "application_review_recorded":
            return True
        if kind == "knowledge_material_changed":
            return (event.get("action") == "promoted" and
                    event.get("source_type") == "project_context")
        return False

    def _source_text(self, event: dict[str, Any]) -> str:
        kind = event.get("kind")
        if kind == "daily_log_recorded":
            values: list[str] = []
            if event.get("tag"):
                values.append(str(event["tag"]))
            values.extend(str(item) for item in event.get("done") or [])
            values.extend(str(item) for item in event.get("incomplete") or [])
            if event.get("notes"):
                values.append(str(event["notes"]))
            return "\n".join(values)
        snapshot_id = str(event.get("snapshot_id") or "")
        snapshot = self.store.get_snapshot(snapshot_id) if snapshot_id else None
        return str((snapshot or {}).get("content") or "")

    def _pending_source_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.store._connect() as conn:
            processed = {row[0]: row[1] for row in conn.execute(
                "SELECT source_event_id,status FROM profile_evidence_extractions"
            )}
        events = [event for event in self.store.list_events(limit=300)
                  if self._source_allowed(event)
                  and processed.get(event["event_id"]) != "completed"]
        return events[-max(1, min(int(limit), 50)):]

    @staticmethod
    def _resolved_model() -> str:
        explicit = os.environ.get("PROFILE_REVIEW_MODEL", "").strip()
        if explicit:
            return explicit
        try:
            from day1_api_starter import get_llm_config, load_local_env
            load_local_env()
            return str((get_llm_config() or {}).get("model") or "")
        except Exception:
            return ""

    def _extraction_messages(self, events: list[dict[str, Any]]) -> list[dict[str, str]]:
        sources = [{
            "source_event_id": event["event_id"], "kind": event.get("kind"),
            "date": event.get("business_date") or event.get("date"),
            "text": self._source_text(event)[:5000],
        } for event in events]
        capabilities = [{key: item.get(key) for key in (
            "capability_id", "name", "aliases", "parent_id", "node_type"
        )} for item in self.capabilities()]
        system = (
            "你是画像证据语义提取器，不是画像修改器。只从 sources 的原文中提取用户实际"
            "接触、练习或交付的能力证据。source_quote 必须逐字来自对应 source 的 text；"
            "source_event_id 和 capability_id 只能引用输入 ID。若没有合适 capability，填写"
            "new_capability_candidate，不得编造 ID。计划、想学、否定、推测和助手生成内容"
            "不是正向证据。observed=阅读了解，practiced=实际练习，delivered=产生可核查结果。"
            "verification_candidate 只是候选，最终由代码核验。criteria_ids 只能引用输入中的"
            "原子能力 ID。explicit_statement 仅表示用户明确要求修改个人事实或偏好。"
            "只返回符合 JSON Schema 的 JSON。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(
                {"sources": sources, "capabilities": capabilities}, ensure_ascii=False,
            )},
        ]

    @staticmethod
    def _future_or_negative(text: str) -> bool:
        return bool(re.search(
            r"(?:想学|准备学习|计划学习|计划掌握|尚未|还没|未掌握|不会|不熟悉|仅了解|待补充)",
            text or "",
        ))

    @staticmethod
    def _quote_location(source_text: str, quote: str) -> tuple[str, int, int]:
        exact_start = source_text.find(quote)
        if exact_start >= 0:
            return "exact", exact_start, exact_start + len(quote)
        normalized_source, normalized_quote = _norm(source_text), _norm(quote)
        normalized_start = normalized_source.find(normalized_quote)
        if normalized_start >= 0:
            return "normalized", normalized_start, normalized_start + len(normalized_quote)
        return "missing", -1, -1

    def _ensure_provisional(self, conn, name: str, parent_id: str = "",
                            aliases: list[str] | None = None) -> str:
        name = re.sub(r"\s+", " ", name or "").strip()[:120]
        if not name:
            return "cap_unmapped"
        capability_id = _default_capability_id(name) or _stable_id("cap", name)
        existing = conn.execute(
            "SELECT capability_id FROM profile_capabilities WHERE capability_id=?", (capability_id,),
        ).fetchone()
        if existing:
            return str(existing[0])
        if parent_id and not conn.execute(
                "SELECT 1 FROM profile_capabilities WHERE capability_id=?", (parent_id,)).fetchone():
            parent_id = ""
        stamp = now_iso()
        conn.execute(
            """INSERT INTO profile_capabilities(
                capability_id,name,aliases_json,parent_id,node_type,rubric_json,
                lifecycle,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)""",
            (capability_id, name, json.dumps(aliases or [], ensure_ascii=False),
             parent_id or None, "atomic", json.dumps(SKILL_RUBRIC, ensure_ascii=False),
             "provisional", stamp, stamp),
        )
        return capability_id

    def _resolve_source_event_id(self, source_id: str) -> str:
        if self.store.get_event(source_id):
            return source_id
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT event_id FROM events WHERE entity_id=? AND deleted_at IS NULL "
                "ORDER BY seq DESC LIMIT 1", (source_id,),
            ).fetchone()
        return str(row[0]) if row else source_id

    def _ground_evidence(self, candidate: EvidenceCandidate,
                         source_events: dict[str, dict[str, Any]]) -> dict[str, Any]:
        source_event_id = self._resolve_source_event_id(candidate.source_event_id)
        event = source_events.get(source_event_id) or self.store.get_event(source_event_id)
        source_text = self._source_text(event) if event else ""
        quote = candidate.source_quote.strip()
        location_kind, quote_start, quote_end = self._quote_location(source_text, quote)
        rejection = ""
        if not event:
            rejection = "unknown_source_event"
        elif not self._source_allowed(event):
            rejection = "untrusted_source"
        elif not quote or location_kind == "missing":
            rejection = "source_quote_not_found"
        elif self._future_or_negative(quote) or self._future_or_negative(candidate.claim):
            rejection = "future_or_negative_statement"

        capability_id = candidate.capability_id.strip()
        with self.store._connect() as conn:
            known = bool(capability_id and conn.execute(
                "SELECT 1 FROM profile_capabilities WHERE capability_id=?", (capability_id,),
            ).fetchone())
        if not known and not candidate.new_capability_candidate.strip():
            rejection = rejection or "unknown_capability"
            capability_id = "cap_unmapped"

        requested = candidate.verification_candidate
        verification: VerificationType = "source_grounded"
        test_keyword = r"(?:测试|test|pytest|用例)"
        success_keyword = r"(?:全部通过|通过|passed?|成功)"
        if requested == "test_result" and (
                re.search(rf"{test_keyword}.{{0,80}}{success_keyword}", quote, re.I) or
                re.search(rf"{success_keyword}.{{0,80}}{test_keyword}", quote, re.I)):
            verification = "test_result"
        elif requested == "artifact_checked" and event and (
                event.get("kind") == "knowledge_material_changed" or
                any(Path(str(ref)).exists() for ref in event.get("attachment_refs") or [])):
            verification = "artifact_checked"
        elif requested == "external_result" and event and event.get("kind") == "application_review_recorded":
            verification = "external_result"
        elif candidate.activity_level == "delivered":
            verification = "user_attested"

        occurred_on = str((event or {}).get("business_date") or (event or {}).get("date") or "")
        origin_key = _hash_text("|".join((source_event_id, _norm(quote), capability_id or
                                          _norm(candidate.new_capability_candidate))))
        return {
            "evidence_id": new_id("profile_ev"), "source_event_id": source_event_id,
            "source_snapshot_id": str((event or {}).get("snapshot_id") or ""),
            "source_type": str((event or {}).get("kind") or "unknown"),
            "source_quote": quote, "quote_hash": _hash_text(_norm(quote)),
            "occurred_on": occurred_on or None, "claim": candidate.claim.strip(),
            "capability_id": capability_id, "new_capability": candidate.new_capability_candidate.strip(),
            "activity_level": candidate.activity_level, "verification": verification,
            "scope": candidate.scope.strip() or "global",
            "target_context_id": str((event or {}).get("target_context_id") or
                                      self.store.active_goal_id()),
            "rationale": candidate.rationale.strip(),
            "status": "rejected" if rejection else "validated",
            "rejection_reason": rejection, "origin_key": origin_key,
            "metadata": {
                "criteria_ids": list(dict.fromkeys(candidate.criteria_ids)),
                "explicit_statement": candidate.explicit_statement,
                "relation": candidate.relation,
                "requested_verification": requested,
                "quote_location": location_kind,
                "quote_start": quote_start,
                "quote_end": quote_end,
                "source_content_hash": _hash_text(source_text),
            },
        }

    def _persist_grounded_evidence(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        with self.store.transaction() as conn:
            for item in items:
                capability_id = item["capability_id"]
                new_capability = item.pop("new_capability", "")
                if new_capability:
                    capability_id = self._ensure_provisional(conn, new_capability)
                    item["capability_id"] = capability_id
                    item["origin_key"] = _hash_text("|".join((
                        item["source_event_id"], _norm(item["source_quote"]), capability_id,
                    )))
                existing = conn.execute(
                    "SELECT * FROM profile_evidence WHERE origin_key=?", (item["origin_key"],),
                ).fetchone()
                if existing:
                    saved.append(self._evidence_row(existing))
                    continue
                stamp = now_iso()
                conn.execute(
                    """INSERT INTO profile_evidence(
                        evidence_id,source_event_id,source_snapshot_id,source_type,source_quote,
                        quote_hash,occurred_on,claim,capability_id,activity_level,verification,
                        scope,target_context_id,rationale,status,rejection_reason,origin_key,
                        metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item["evidence_id"], item["source_event_id"], item["source_snapshot_id"] or None,
                     item["source_type"], item["source_quote"], item["quote_hash"], item["occurred_on"],
                     item["claim"], capability_id, item["activity_level"], item["verification"],
                     item["scope"], item["target_context_id"], item["rationale"], item["status"],
                     item["rejection_reason"], item["origin_key"],
                     json.dumps(item["metadata"], ensure_ascii=False), stamp, stamp),
                )
                saved.append({**item, "created_at": stamp, "updated_at": stamp})
        return saved

    @staticmethod
    def _evidence_row(row) -> dict[str, Any]:
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json") or "{}")
        return value

    def list_evidence(self, *, status: str = "validated",
                      target_context_id: str = "") -> list[dict[str, Any]]:
        clauses, params = [], []
        if status:
            clauses.append("status=?"); params.append(status)
        if target_context_id:
            clauses.append("target_context_id IN (?, 'global')"); params.append(target_context_id)
        sql = "SELECT * FROM profile_evidence"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY occurred_on,created_at"
        with self.store._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._evidence_row(row) for row in rows]

    def evidence(self, evidence_id: str, *, include_source: bool = False) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profile_evidence WHERE evidence_id=?", (evidence_id,),
            ).fetchone()
        if not row:
            return None
        value = self._evidence_row(row)
        if include_source:
            event = self.store.get_event(value["source_event_id"])
            value["source_event"] = ({key: event.get(key) for key in (
                "event_id", "seq", "kind", "occurred_at", "business_date", "actor",
                "source", "target_context_id", "entity_type", "entity_id",
            )} if event else None)
            value["source_text"] = self._source_text(event) if event else ""
        return value

    def ingest_candidates(self, candidates: list[EvidenceCandidate | dict[str, Any]]) -> dict[str, Any]:
        parsed: list[EvidenceCandidate] = []
        for raw in candidates:
            try:
                parsed.append(raw if isinstance(raw, EvidenceCandidate)
                              else EvidenceCandidate.model_validate(raw))
            except Exception:
                continue
        source_ids = {self._resolve_source_event_id(item.source_event_id) for item in parsed}
        source_events = {event_id: self.store.get_event(event_id) for event_id in source_ids}
        grounded = [self._ground_evidence(item, source_events) for item in parsed]
        saved = self._persist_grounded_evidence(grounded)
        return {"status": "ok", "items": saved,
                "validated": sum(item["status"] == "validated" for item in saved),
                "rejected": sum(item["status"] == "rejected" for item in saved)}

    def ingest_reflection_evidence(self, reflection: dict[str, Any]) -> dict[str, Any]:
        candidates = list(reflection.get("evidence_candidates") or [])
        if not candidates:
            source_ids = list(reflection.get("source_log_ids") or [])
            source_id = source_ids[0] if source_ids else ""
            level_map = {"observed": "observed", "practiced": "practiced", "verified": "delivered"}
            for raw in reflection.get("skill_evidence") or []:
                if not source_id or not isinstance(raw, dict):
                    continue
                skill = str(raw.get("skill") or "").strip()
                candidates.append({
                    "claim": f"用户记录了 {skill} 的实践证据", "source_event_id": source_id,
                    "source_quote": str(raw.get("evidence") or "").strip(),
                    "new_capability_candidate": skill,
                    "activity_level": level_map.get(str(raw.get("level") or ""), "observed"),
                    "verification_candidate": "user_attested",
                    "scope": "global", "rationale": "legacy_skill_evidence",
                })
        return self.ingest_candidates(candidates)

    def extract_pending_evidence(self, *, caller: TextCaller | None = None,
                                 model: str | None = None, limit: int = 20) -> dict[str, Any]:
        events = self._pending_source_events(limit=limit)
        if not events:
            return {"status": "ok", "sources": 0, "validated": 0, "rejected": 0}
        resolved_model = model if model is not None else self._resolved_model()
        source_ids = [event["event_id"] for event in events]
        if not resolved_model and caller is None:
            with self.store.transaction() as conn:
                for event_id in source_ids:
                    conn.execute(
                        """INSERT INTO profile_evidence_extractions VALUES(?,?,?,?,?)
                           ON CONFLICT(source_event_id) DO UPDATE SET status=excluded.status,
                           model_meta_json=excluded.model_meta_json,error=excluded.error,
                           processed_at=excluded.processed_at""",
                        (event_id, "review_unavailable", "{}", "llm_unavailable:no_model", now_iso()),
                    )
            return {"status": "review_unavailable", "sources": len(events),
                    "error": "llm_unavailable:no_model"}
        draft, meta = call_structured(
            EvidenceExtractionDraft, self._extraction_messages(events),
            model=resolved_model or None, timeout_seconds=float(os.environ.get(
                "PROFILE_REVIEW_TIMEOUT_SECONDS", "45") or 45), max_tokens=3600,
            repair=True, caller=caller,
            reasoning_effort=os.environ.get("PROFILE_REVIEW_REASONING_EFFORT", "low") or None,
            lane="online",
        )
        if draft is None:
            error = ",".join(meta.errors) or "invalid_output"
            with self.store.transaction() as conn:
                for event_id in source_ids:
                    conn.execute(
                        """INSERT INTO profile_evidence_extractions VALUES(?,?,?,?,?)
                           ON CONFLICT(source_event_id) DO UPDATE SET status=excluded.status,
                           model_meta_json=excluded.model_meta_json,error=excluded.error,
                           processed_at=excluded.processed_at""",
                        (event_id, "review_unavailable", json.dumps(meta.to_dict()), error, now_iso()),
                    )
            return {"status": "review_unavailable", "sources": len(events), "error": error,
                    "model_meta": meta.to_dict()}
        source_events = {event["event_id"]: event for event in events}
        grounded = [self._ground_evidence(item, source_events) for item in draft.candidates]
        saved = self._persist_grounded_evidence(grounded)
        with self.store.transaction() as conn:
            for event_id in source_ids:
                conn.execute(
                    """INSERT INTO profile_evidence_extractions VALUES(?,?,?,?,?)
                       ON CONFLICT(source_event_id) DO UPDATE SET status=excluded.status,
                       model_meta_json=excluded.model_meta_json,error='',processed_at=excluded.processed_at""",
                    (event_id, "completed", json.dumps(meta.to_dict()), "", now_iso()),
                )
        return {"status": "ok", "sources": len(events), "items": saved,
                "validated": sum(item["status"] == "validated" for item in saved),
                "rejected": sum(item["status"] == "rejected" for item in saved),
                "model_meta": meta.to_dict()}

    def _capability_graph(self) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
        capabilities = {item["capability_id"]: item for item in self.capabilities(include_archived=True)}
        children: dict[str, set[str]] = {}
        for item in capabilities.values():
            parent = str(item.get("parent_id") or "")
            if parent:
                children.setdefault(parent, set()).add(item["capability_id"])
        descendants: dict[str, set[str]] = {}
        for capability_id in capabilities:
            found, pending = set(), list(children.get(capability_id, set()))
            while pending:
                child = pending.pop()
                if child in found:
                    continue
                found.add(child)
                pending.extend(children.get(child, set()))
            descendants[capability_id] = found
        return capabilities, descendants

    def capability_bounds(self, target_context_id: str = "") -> dict[str, dict[str, Any]]:
        target_context_id = target_context_id or self.store.active_goal_id()
        capabilities, descendants = self._capability_graph()
        evidence = self.list_evidence(status="validated", target_context_id=target_context_id)
        bounds: dict[str, dict[str, Any]] = {}
        for capability_id, capability in capabilities.items():
            related = {capability_id, *descendants.get(capability_id, set())}
            rows = [item for item in evidence if item["capability_id"] in related and
                    item.get("metadata", {}).get("relation") != "unrelated"]
            if not rows:
                bounds[capability_id] = {"max_supported_level": 0, "evidence_ids": [],
                                         "dates": [], "source_count": 0, "criteria_ids": []}
                continue
            maximum = 1
            if any(item["activity_level"] in {"practiced", "delivered"} for item in rows):
                maximum = 2
            strong = [item for item in rows if item["activity_level"] == "delivered" and
                      item["verification"] in {"artifact_checked", "test_result", "external_result"} and
                      item.get("metadata", {}).get("relation") in {"direct", "partial"}]
            dates = {str(item.get("occurred_on") or "") for item in rows if item.get("occurred_on")}
            sources = {item["source_event_id"] for item in rows}
            criteria = {criterion for item in rows for criterion in
                        item.get("metadata", {}).get("criteria_ids", []) if criterion in related}
            atomic = capability.get("node_type") == "atomic"
            project_verified = any(item["source_type"] == "knowledge_material_changed" for item in strong)
            if strong and (atomic or len(criteria) >= 2) and (project_verified or len(dates) >= 2):
                maximum = 3
            quality_cue = any(re.search(
                r"(?:评估|指标|排障|诊断|监控|部署|维护|回归|压测)",
                item.get("claim", "") + " " + item.get("source_quote", ""),
            ) for item in strong)
            required_coverage = 1 if atomic else 3
            if (len({item["source_event_id"] for item in strong}) >= 3 and len(dates) >= 2 and
                    (atomic or len(criteria) >= required_coverage) and quality_cue):
                maximum = 4
            bounds[capability_id] = {
                "max_supported_level": maximum,
                "evidence_ids": [item["evidence_id"] for item in rows],
                "dates": sorted(dates), "source_count": len(sources),
                "criteria_ids": sorted(criteria),
                "strong_evidence_count": len(strong),
            }
        return bounds

    @staticmethod
    def _json_equal(left: Any, right: Any) -> bool:
        return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
            right, ensure_ascii=False, sort_keys=True,
        )

    @staticmethod
    def _current_value(spec: ProfileSpec, field_path: str) -> Any:
        if field_path.startswith("/fields/"):
            return spec.matching_fields.get(field_path.removeprefix("/fields/"))
        match = re.fullmatch(r"/skills/([^/]+)/level", field_path)
        if match:
            skill = next((item for item in spec.skills if item.assessment_id == match.group(1)), None)
            return skill.level if skill else None
        if field_path.startswith("/skills/new"):
            return None
        if field_path == "/projects":
            section = next((item for item in spec.sections if item.section_id == "4"), None)
            return section.body_markdown if section else ""
        return None

    def _audit_messages(self, current: dict[str, Any], evidence: list[dict[str, Any]],
                        bounds: dict[str, dict[str, Any]],
                        new_evidence_ids: set[str]) -> list[dict[str, str]]:
        spec = ProfileSpec.model_validate(current["profile_spec"])
        allowed_paths = [f"/fields/{key}" for key in spec.matching_fields]
        allowed_paths.extend(f"/skills/{item.assessment_id}/level" for item in spec.skills)
        allowed_paths.extend(["/skills/new", "/projects"])
        profile = {
            "revision": current["revision"], "matching_fields": spec.matching_fields,
            "skills": [item.model_dump(mode="json") for item in spec.skills],
        }
        evidence_payload = [{key: item.get(key) for key in (
            "evidence_id", "source_event_id", "source_type", "source_quote", "occurred_on",
            "claim", "capability_id", "activity_level", "verification", "scope",
            "target_context_id", "rationale", "metadata",
        )} for item in evidence]
        system = (
            "你是 OfferClaw 的画像语义审查器。比较当前画像和经过代码验证的 evidence，"
            "提出字段级候选修改。你只能使用 allowed_field_paths，并且每条修改必须引用输入中的"
            "evidence_id，且至少引用一个 new_evidence_id。画像技能等级与单条证据强度不同；"
            "技能上调不得超过 capability_bounds。"
            "不要因为一条子技能证据宣称整个综合能力已掌握。身份、学历、地域、正式职业目标、"
            "薪资、排除项、可投入时间只能在 evidence.metadata.explicit_statement=true 时建议修改。"
            "兴趣和工作方式需要至少三条独立证据并跨两天。/projects 只允许建议加入项目材料原文，"
            "proposed_value 必须是 {name, body_markdown}，body_markdown 逐字来自被引用来源。"
            "新能力使用 /skills/new，提供 new_capability 和 {name, level}；能力 5 级不得由普通审计"
            "建议。不要输出 Markdown 画像或未被证据支持的事实。证据不足时将 requirement_status"
            "设为 needs_evidence。只返回符合 JSON Schema 的 JSON。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "profile": profile, "allowed_field_paths": allowed_paths,
                "evidence": evidence_payload, "new_evidence_ids": sorted(new_evidence_ids),
                "capability_bounds": bounds,
            }, ensure_ascii=False)},
        ]

    def _related_capabilities(self, capability_id: str) -> set[str]:
        capabilities, descendants = self._capability_graph()
        related = {capability_id, *descendants.get(capability_id, set())}
        current = capability_id
        while current in capabilities and capabilities[current].get("parent_id"):
            current = str(capabilities[current]["parent_id"])
            related.add(current)
        return related

    def _project_value_grounded(self, value: Any, evidence: list[dict[str, Any]]) -> bool:
        if not isinstance(value, dict):
            return False
        name = str(value.get("name") or "").strip()
        body = str(value.get("body_markdown") or "").strip()
        if not name or len(body) < 20:
            return False
        for item in evidence:
            event = self.store.get_event(item["source_event_id"])
            if event and item["source_type"] == "knowledge_material_changed" and (
                    _norm(body) in _norm(self._source_text(event))):
                return True
        return False

    @staticmethod
    def _evidence_strength(item: dict[str, Any]) -> tuple[int, int]:
        activity = {"observed": 1, "practiced": 2, "delivered": 3}
        verification = {
            "source_grounded": 1, "user_attested": 1,
            "artifact_checked": 3, "test_result": 3, "external_result": 3,
        }
        return (activity.get(str(item.get("activity_level")), 0),
                verification.get(str(item.get("verification")), 0))

    def _has_stronger_evidence_than_rejection(
            self, conn, field_path: str, evidence_ids: list[str],
            evidence_by_id: dict[str, dict[str, Any]]) -> bool:
        rejected = conn.execute(
            """SELECT evidence_ids_json FROM profile_suggestions
               WHERE field_path=? AND status='rejected' ORDER BY updated_at DESC LIMIT 1""",
            (field_path,),
        ).fetchone()
        if not rejected:
            return True
        old_ids = set(json.loads(rejected["evidence_ids_json"] or "[]"))
        added = [evidence_by_id[item] for item in evidence_ids
                 if item not in old_ids and item in evidence_by_id]
        if not added:
            return False
        old_rows = [evidence_by_id[item] for item in old_ids if item in evidence_by_id]
        old_max = max((self._evidence_strength(item) for item in old_rows), default=(0, 0))
        # A further independent item of the same strongest kind strengthens the
        # aggregate; a weaker observation does not reopen a rejected proposal.
        return any(self._evidence_strength(item) >= old_max for item in added)

    @staticmethod
    def _level_five_supported(evidence: list[dict[str, Any]]) -> bool:
        strong = [item for item in evidence if item.get("activity_level") == "delivered" and
                  item.get("verification") in {
                      "artifact_checked", "test_result", "external_result",
                  }]
        explicit = any(
            item.get("metadata", {}).get("explicit_statement") and re.search(
                r"(?:5\s*/\s*5|5\s*级|五级|专家级)", str(item.get("source_quote") or ""), re.I,
            )
            for item in evidence
        )
        scopes = {_norm(item.get("scope")) for item in strong if _norm(item.get("scope"))}
        return explicit and len({item["source_event_id"] for item in strong}) >= 3 and len(scopes) >= 2

    @staticmethod
    def _explicit_value_supported(field: str, operation: str, proposed_value: Any,
                                  evidence: list[dict[str, Any]]) -> bool:
        quotes = "\n".join(str(item.get("source_quote") or "") for item in evidence)
        normalized_quotes = _norm(quotes)
        if operation == "remove":
            current_markers = ("不再", "删除", "取消", "移除", "没有", "不限")
            return any(marker in quotes for marker in current_markers)
        values = proposed_value if isinstance(proposed_value, list) else [proposed_value]
        normalized_values = [_norm(item) for item in values if _norm(item)]
        if not normalized_values:
            return False
        # The model may identify the field semantics, but every concrete value
        # applied to protected fields must occur in the user's exact quote.
        return all(value in normalized_quotes for value in normalized_values)

    def _evaluate_change(self, change: ProfileChangeDraft, spec: ProfileSpec,
                         evidence_by_id: dict[str, dict[str, Any]],
                         bounds: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
        evidence_ids = list(dict.fromkeys(change.evidence_ids))
        counter_ids = list(dict.fromkeys(change.counter_evidence_ids))
        if any(item not in evidence_by_id for item in evidence_ids + counter_ids):
            return "needs_evidence", "unknown_evidence_reference", change.capability_id
        cited = [evidence_by_id[item] for item in evidence_ids]
        if not cited:
            return "needs_evidence", "missing_evidence", change.capability_id
        current = self._current_value(spec, change.field_path)
        if not self._json_equal(current, change.current_value):
            return "needs_evidence", "current_value_mismatch", change.capability_id
        if change.requirement_status != "satisfied":
            return "needs_evidence", change.requirement_status, change.capability_id
        if change.operation == "replace" and self._json_equal(current, change.proposed_value):
            return "needs_evidence", "no_effect", change.capability_id

        skill_match = re.fullmatch(r"/skills/([^/]+)/level", change.field_path)
        if skill_match:
            skill = next((item for item in spec.skills if item.assessment_id == skill_match.group(1)), None)
            if not skill:
                return "needs_evidence", "unknown_skill_assessment", change.capability_id
            try:
                proposed_level = int(change.proposed_value)
            except (TypeError, ValueError):
                return "needs_evidence", "invalid_skill_level", skill.capability_id
            if proposed_level < 1 or proposed_level > 5:
                return "needs_evidence", "automatic_level_out_of_range", skill.capability_id
            related = self._related_capabilities(skill.capability_id)
            if any(item["capability_id"] not in related for item in cited):
                return "needs_evidence", "unrelated_skill_evidence", skill.capability_id
            if proposed_level <= skill.level:
                return "needs_evidence", "automatic_downgrade_or_noop_not_allowed", skill.capability_id
            maximum = int(bounds.get(skill.capability_id, {}).get("max_supported_level") or 0)
            if proposed_level == 5 and not self._level_five_supported(cited):
                return "needs_evidence", "level_five_requires_explicit_multiscenario_evidence", skill.capability_id
            if proposed_level < 5 and proposed_level > maximum:
                return "needs_evidence", "skill_level_exceeds_code_bound", skill.capability_id
            return "pending", "", skill.capability_id

        if change.field_path.startswith("/skills/new"):
            if not change.new_capability or not isinstance(change.proposed_value, dict):
                return "needs_evidence", "invalid_new_capability", change.capability_id
            capability_id = change.capability_id or _default_capability_id(
                change.new_capability.name) or _stable_id("cap", change.new_capability.name)
            if any(item["capability_id"] != capability_id for item in cited):
                return "needs_evidence", "new_capability_evidence_mismatch", capability_id
            try:
                proposed_level = int(change.proposed_value.get("level"))
            except (TypeError, ValueError):
                return "needs_evidence", "invalid_skill_level", capability_id
            maximum = int(bounds.get(capability_id, {}).get("max_supported_level") or 0)
            if proposed_level < 1 or proposed_level > 5:
                return "needs_evidence", "automatic_level_out_of_range", capability_id
            if proposed_level == 5 and not self._level_five_supported(cited):
                return "needs_evidence", "level_five_requires_explicit_multiscenario_evidence", capability_id
            if proposed_level < 5 and proposed_level > min(4, maximum):
                return "needs_evidence", "skill_level_exceeds_code_bound", capability_id
            return "pending", "", capability_id

        if change.field_path == "/projects":
            if not all(item["source_type"] == "knowledge_material_changed" for item in cited):
                return "needs_evidence", "project_requires_confirmed_project_source", ""
            if not self._project_value_grounded(change.proposed_value, cited):
                return "needs_evidence", "project_body_not_grounded", ""
            return "pending", "", ""

        if not change.field_path.startswith("/fields/"):
            return "needs_evidence", "field_not_allowed", ""
        field = change.field_path.removeprefix("/fields/")
        explicit_only = {
            "姓名", "学历", "专业", "所在地", "可接受地域", "方向优先级",
            "目标岗位类型", "行业偏好", "明确不做", "工作性质偏好", "期望薪资",
            "每天可投入", "每周可投入", "黄金时段", "不可打扰时段",
        }
        inferred = {"兴趣爱好", "擅长的思维方式", "不擅长或不感兴趣的方向"}
        if field not in explicit_only | inferred:
            return "needs_evidence", "field_not_allowed", ""
        if field in explicit_only and not all(
                item.get("metadata", {}).get("explicit_statement") for item in cited):
            return "needs_evidence", "explicit_statement_required", ""
        if field in explicit_only and not self._explicit_value_supported(
                field, change.operation, change.proposed_value, cited):
            return "needs_evidence", "explicit_value_not_found_in_source", ""
        if field in inferred:
            source_count = len({item["source_event_id"] for item in cited})
            day_count = len({item.get("occurred_on") for item in cited if item.get("occurred_on")})
            if source_count < 3 or day_count < 2:
                return "needs_evidence", "inferred_field_threshold_not_met", ""
        return "pending", "", ""

    def run_audit(self, *, max_items: int = 5, trigger_kind: str = "manual",
                  target_context_id: str = "", caller: TextCaller | None = None,
                  model: str | None = None, extract_sources: bool = True) -> dict[str, Any]:
        target_context_id = target_context_id or self.store.active_goal_id()
        extraction = (self.extract_pending_evidence(caller=caller, model=model)
                      if extract_sources else {"status": "skipped"})
        current = self.current()
        evidence = self.list_evidence(status="validated", target_context_id=target_context_id)
        with self.store._connect() as conn:
            previously_reviewed = {str(row[0]) for row in conn.execute(
                """SELECT DISTINCT ae.evidence_id FROM profile_audit_evidence ae
                   JOIN profile_audits a ON a.audit_id=ae.audit_id
                   WHERE a.target_context_id=? AND a.status='completed'""",
                (target_context_id,),
            )}
        new_evidence_ids = {item["evidence_id"] for item in evidence
                            if item["evidence_id"] not in previously_reviewed}
        bounds = self.capability_bounds(target_context_id)
        audit_id, stamp = new_id("profile_audit"), now_iso()
        evidence_cutoff = max((item.get("created_at") or "" for item in evidence), default=stamp)
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO profile_audits(
                    audit_id,base_revision,target_context_id,trigger_kind,evidence_cutoff,status,
                    model_meta_json,error,created_at,completed_at) VALUES(?,?,?,?,?,'generating','{}','',?,NULL)""",
                (audit_id, current["revision"], target_context_id, trigger_kind, evidence_cutoff, stamp),
            )
        if not evidence:
            with self.store.transaction() as conn:
                conn.execute("UPDATE profile_audits SET status='completed',completed_at=? WHERE audit_id=?",
                             (now_iso(), audit_id))
            existing = self.list_suggestions()
            return {"status": "ok", "audit_id": audit_id, "created": [],
                    "pending": sum(item["status"] == "pending" for item in existing),
                    "needs_evidence": sum(item["status"] == "needs_evidence" for item in existing),
                    "extraction": extraction}
        if not new_evidence_ids:
            with self.store.transaction() as conn:
                conn.execute("UPDATE profile_audits SET status='completed',completed_at=? WHERE audit_id=?",
                             (now_iso(), audit_id))
            existing = self.list_suggestions()
            return {"status": "ok", "audit_id": audit_id, "created": [],
                    "pending": sum(item["status"] == "pending" for item in existing),
                    "needs_evidence": sum(item["status"] == "needs_evidence" for item in existing),
                    "reason": "no_new_evidence", "extraction": extraction}
        resolved_model = model if model is not None else self._resolved_model()
        if not resolved_model and caller is None:
            with self.store.transaction() as conn:
                conn.execute(
                    "UPDATE profile_audits SET status='review_unavailable',error=?,completed_at=? WHERE audit_id=?",
                    ("llm_unavailable:no_model", now_iso(), audit_id),
                )
            return {"status": "review_unavailable", "audit_id": audit_id,
                    "created": [], "pending": 0, "needs_evidence": 0,
                    "error": "llm_unavailable:no_model", "extraction": extraction}
        draft, meta = call_structured(
            ProfileAuditDraft, self._audit_messages(current, evidence, bounds, new_evidence_ids),
            model=resolved_model or None, timeout_seconds=float(os.environ.get(
                "PROFILE_REVIEW_TIMEOUT_SECONDS", "45") or 45), max_tokens=3600,
            repair=True, caller=caller,
            reasoning_effort=os.environ.get("PROFILE_REVIEW_REASONING_EFFORT", "low") or None,
            lane="online",
        )
        if draft is None:
            error = ",".join(meta.errors) or "invalid_output"
            with self.store.transaction() as conn:
                conn.execute(
                    "UPDATE profile_audits SET status='review_unavailable',model_meta_json=?,error=?,completed_at=? WHERE audit_id=?",
                    (json.dumps(meta.to_dict()), error, now_iso(), audit_id),
                )
            return {"status": "review_unavailable", "audit_id": audit_id,
                    "created": [], "pending": 0, "needs_evidence": 0,
                    "error": error, "model_meta": meta.to_dict(), "extraction": extraction}

        spec = ProfileSpec.model_validate(current["profile_spec"])
        evidence_by_id = {item["evidence_id"]: item for item in evidence}
        created: list[dict[str, Any]] = []
        with self.store.transaction() as conn:
            for change in draft.changes[:max(1, min(int(max_items), 20))]:
                status, gate_reason, capability_id = self._evaluate_change(
                    change, spec, evidence_by_id, bounds,
                )
                if not new_evidence_ids.intersection(change.evidence_ids):
                    status, gate_reason = "needs_evidence", "change_does_not_cite_new_evidence"
                if change.new_capability:
                    capability_id = self._ensure_provisional(
                        conn, change.new_capability.name, change.new_capability.parent_id,
                        change.new_capability.aliases,
                    )
                evidence_ids = list(dict.fromkeys(change.evidence_ids))
                counter_ids = list(dict.fromkeys(change.counter_evidence_ids))
                evidence_hash = _hash_text(json.dumps(
                    {"evidence": evidence_ids, "counter": counter_ids},
                    ensure_ascii=False, sort_keys=True,
                ))
                duplicate = conn.execute(
                    """SELECT 1 FROM profile_suggestions WHERE field_path=? AND evidence_hash=?
                       AND status IN ('pending','needs_evidence','accepted','modified','rejected')""",
                    (change.field_path, evidence_hash),
                ).fetchone()
                if duplicate:
                    continue
                if not self._has_stronger_evidence_than_rejection(
                        conn, change.field_path, evidence_ids, evidence_by_id):
                    continue
                suggestion_id, created_at = new_id("profile_suggestion"), now_iso()
                requirement_status = "satisfied" if status == "pending" else "needs_evidence"
                rationale = change.rationale.strip()
                if gate_reason:
                    rationale += f"\n[代码校验] {gate_reason}"
                conn.execute(
                    """INSERT INTO profile_suggestions(
                        suggestion_id,audit_id,base_revision,target_context_id,field_path,operation,
                        current_value_json,proposed_value_json,observed_change,evidence_ids_json,
                        counter_evidence_ids_json,rationale,requirement_status,status,capability_id,
                        new_capability_json,evidence_hash,model_meta_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (suggestion_id, audit_id, current["revision"], target_context_id,
                     change.field_path, change.operation,
                     json.dumps(change.current_value, ensure_ascii=False),
                     json.dumps(change.proposed_value, ensure_ascii=False),
                     change.observed_change, json.dumps(evidence_ids), json.dumps(counter_ids),
                     rationale, requirement_status, status, capability_id or None,
                     change.new_capability.model_dump_json() if change.new_capability else "{}",
                     evidence_hash, json.dumps(meta.to_dict()), created_at, created_at),
                )
                created.append({"suggestion_id": suggestion_id, "status": status,
                                "field_path": change.field_path})
            conn.execute(
                "UPDATE profile_audits SET status='completed',model_meta_json=?,completed_at=? WHERE audit_id=?",
                (json.dumps(meta.to_dict()), now_iso(), audit_id),
            )
            conn.executemany(
                "INSERT INTO profile_audit_evidence(audit_id,evidence_id,is_new) VALUES(?,?,?)",
                [(audit_id, item["evidence_id"], int(item["evidence_id"] in new_evidence_ids))
                 for item in evidence],
            )
        rows = self.list_suggestions()
        return {"status": "ok", "audit_id": audit_id, "created": created,
                "pending": sum(item["status"] == "pending" for item in rows),
                "needs_evidence": sum(item["status"] == "needs_evidence" for item in rows),
                "model_meta": meta.to_dict(), "extraction": extraction}

    def _suggestion_row(self, row, base_hash: str = "") -> dict[str, Any]:
        value = dict(row)
        for source, target in (("current_value_json", "current_value"),
                               ("proposed_value_json", "proposed_value"),
                               ("evidence_ids_json", "evidence_ids"),
                               ("counter_evidence_ids_json", "counter_evidence_ids"),
                               ("new_capability_json", "new_capability"),
                               ("model_meta_json", "model_meta")):
            value[target] = json.loads(value.pop(source) or ("[]" if "ids" in source else "{}"))
        value["evidence_refs"] = [self.evidence(item) for item in value["evidence_ids"]]
        value["evidence_refs"] = [item for item in value["evidence_refs"] if item]
        value["counter_evidence_refs"] = [self.evidence(item) for item in value["counter_evidence_ids"]]
        value["counter_evidence_refs"] = [item for item in value["counter_evidence_refs"] if item]
        if not base_hash:
            with self.store._connect() as conn:
                revision = conn.execute(
                    "SELECT content_hash FROM profile_revisions WHERE revision=?",
                    (value["base_revision"],),
                ).fetchone()
            base_hash = str(revision[0]) if revision else ""
        value["base_profile_hash"] = base_hash
        value["target_section"] = (
            "3/9" if value["field_path"].startswith("/skills/") else
            "4" if value["field_path"] == "/projects" else "画像字段"
        )
        value["current_text"] = json.dumps(value["current_value"], ensure_ascii=False)
        value["proposed_text"] = json.dumps(value["proposed_value"], ensure_ascii=False)
        return value

    def list_suggestions(self, status: str = "") -> list[dict[str, Any]]:
        current = self.current()
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE profile_suggestions SET status='stale',updated_at=? "
                "WHERE status IN ('pending','needs_evidence') AND base_revision!=?",
                (now_iso(), current["revision"]),
            )
        clauses, params = [], []
        if status:
            clauses.append("status=?"); params.append(status)
        sql = "SELECT * FROM profile_suggestions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        with self.store._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        with self.store._connect() as conn:
            revision_hashes = {
                int(row["revision"]): str(row["content_hash"])
                for row in conn.execute("SELECT revision,content_hash FROM profile_revisions")
            }
        return [self._suggestion_row(row, revision_hashes.get(int(row["base_revision"]), ""))
                for row in rows]

    @staticmethod
    def _replace_simple_field(spec: ProfileSpec, field: str, value: Any) -> None:
        labels = {
            "姓名": ("1", ("姓名 / 昵称", "姓名")), "学历": ("1", ("学历层次", "学历")),
            "专业": ("1", ("专业",)), "所在地": ("1", ("所在地",)),
            "可接受地域": ("1", ("可接受工作地域", "可接受地域")),
            "目标岗位类型": ("2", ("目标岗位类型",)), "行业偏好": ("2", ("行业偏好",)),
            "工作性质偏好": ("2", ("工作性质偏好",)), "期望薪资": ("2", ("期望薪资区间", "期望薪资")),
            "兴趣爱好": ("8", ("兴趣爱好",)), "擅长的思维方式": ("8", ("擅长的思维方式",)),
            "不擅长或不感兴趣的方向": ("8", ("不擅长或不感兴趣的方向",)),
            "每天可投入": ("10", ("每天可投入（小时）",)),
            "每周可投入": ("10", ("每周可投入（小时）",)),
            "黄金时段": ("10", ("黄金时段",)), "不可打扰时段": ("10", ("不可打扰时段",)),
        }
        if field in {"方向优先级", "明确不做"}:
            section = next(item for item in spec.sections if item.section_id == "2")
            lines = section.body_markdown.splitlines()
            label = "目标方向（按优先级排序）" if field == "方向优先级" else "明确不做的方向"
            start = next((index for index, line in enumerate(lines)
                          if re.match(rf"^\s*-\s*{re.escape(label)}\s*[：:]", line)), -1)
            if start < 0:
                raise ProfileValidationError(f"画像中找不到字段：{field}")
            end = start + 1
            while end < len(lines) and not re.match(r"^\s*-\s*[^\s-].*[：:]", lines[end]):
                end += 1
            values = value if isinstance(value, list) else [value]
            replacement = [lines[start]] + [
                (f"  {index}. {item}" if field == "方向优先级" else f"  - {item}")
                for index, item in enumerate(values, 1)
            ]
            section.body_markdown = "\n".join(lines[:start] + replacement + lines[end:])
            return
        if field not in labels:
            raise ProfileValidationError(f"画像字段不支持自动应用：{field}")
        section_id, names = labels[field]
        section = next(item for item in spec.sections if item.section_id == section_id)
        rendered = "/".join(str(item) for item in value) if isinstance(value, list) else str(value)
        lines = section.body_markdown.splitlines()
        for index, line in enumerate(lines):
            for name in names:
                match = re.match(rf"^(\s*-\s*{re.escape(name)}\s*[：:]).*$", line)
                if match:
                    lines[index] = match.group(1) + rendered
                    section.body_markdown = "\n".join(lines)
                    return
        raise ProfileValidationError(f"画像中找不到字段：{field}")

    def _apply_suggestion(self, spec: ProfileSpec, suggestion: dict[str, Any],
                          final_value: Any) -> ProfileSpec:
        updated = copy.deepcopy(spec)
        field_path = suggestion["field_path"]
        skill_match = re.fullmatch(r"/skills/([^/]+)/level", field_path)
        if skill_match:
            skill = next((item for item in updated.skills
                          if item.assessment_id == skill_match.group(1)), None)
            if not skill:
                raise ProfileValidationError("画像技能字段已不存在")
            section = next(item for item in updated.sections
                           if item.section_id == skill.source_section_id)
            lines = section.body_markdown.splitlines()
            if skill.source_line >= len(lines):
                raise ProfileValidationError("画像技能字段位置已失效")
            if skill.source_format == "inline_score":
                lines[skill.source_line], count = re.subn(
                    r"(?<!\d)[1-5]/5", f"{int(final_value)}/5", lines[skill.source_line], count=1,
                )
            else:
                lines[skill.source_line], count = re.subn(
                    r"^(\|\s*[^|]+\|\s*)[1-5](\s*\|)",
                    rf"\g<1>{int(final_value)}\g<2>", lines[skill.source_line], count=1,
                )
            if count != 1:
                raise ProfileValidationError("画像技能等级格式已变化")
            section.body_markdown = "\n".join(lines)
        elif field_path.startswith("/skills/new"):
            if not isinstance(final_value, dict):
                raise ProfileValidationError("新增能力必须包含 name 和 level")
            name, level = str(final_value.get("name") or "").strip(), int(final_value.get("level"))
            section = next(item for item in updated.sections if item.section_id == "3")
            section.body_markdown = section.body_markdown.rstrip() + f"\n- {name}：{level}/5（经证据审计并由用户确认）"
        elif field_path == "/projects":
            if not isinstance(final_value, dict):
                raise ProfileValidationError("项目建议格式无效")
            body = str(final_value.get("body_markdown") or "").strip()
            section = next(item for item in updated.sections if item.section_id == "4")
            section.body_markdown = section.body_markdown.rstrip() + "\n\n" + body
        elif field_path.startswith("/fields/"):
            self._replace_simple_field(updated, field_path.removeprefix("/fields/"), final_value)
        else:
            raise ProfileValidationError("画像建议字段不受支持")
        reparsed = _carry_skill_identity(
            updated, parse_profile_markdown(render_profile_markdown(updated)),
        )
        for skill in reparsed.skills:
            if suggestion.get("capability_id") and (
                    skill.assessment_id == skill_match.group(1) if skill_match else
                    skill.name == str((final_value or {}).get("name") if isinstance(final_value, dict) else "")):
                skill.capability_id = suggestion["capability_id"]
                skill.evidence_ids = list(suggestion.get("evidence_ids") or [])
                skill.last_verified_at = now_iso()
        return reparsed

    def decide_suggestion(self, suggestion_id: str, decision: str, *,
                          base_revision: int | None = None, modified_value: Any = None,
                          reason: str = "", operation_id: str = "") -> dict[str, Any]:
        if decision not in {"accepted", "modified", "rejected"}:
            raise ProfileValidationError("decision 必须是 accepted / modified / rejected")
        operation_id = operation_id or new_id("op")
        with self.store._connect() as conn:
            prior = conn.execute(
                "SELECT * FROM profile_suggestion_decisions WHERE operation_id=?", (operation_id,),
            ).fetchone()
            row = conn.execute(
                "SELECT * FROM profile_suggestions WHERE suggestion_id=?", (suggestion_id,),
            ).fetchone()
        if prior:
            return {"status": "ok", "suggestion_id": suggestion_id,
                    "result_revision": prior["result_revision"], "replayed": True}
        if not row:
            raise KeyError("画像建议不存在")
        suggestion = self._suggestion_row(row)
        if suggestion["status"] != "pending":
            raise ProfileValidationError("只有 pending 建议可以处理")
        current = self.current()
        expected_revision = int(base_revision if base_revision is not None else suggestion["base_revision"])
        if current["revision"] != int(suggestion["base_revision"]) or current["revision"] != expected_revision:
            with self.store.transaction() as conn:
                conn.execute("UPDATE profile_suggestions SET status='stale',updated_at=? WHERE suggestion_id=?",
                             (now_iso(), suggestion_id))
            raise ProfileConflictError("画像版本已变化，请重新审计建议")

        final_value = suggestion["proposed_value"] if decision == "accepted" else modified_value
        result_revision: int | None = None
        new_content = current["content_md"]
        updated_spec = ProfileSpec.model_validate(current["profile_spec"])
        if decision != "rejected":
            if final_value is None or final_value == "":
                raise ProfileValidationError("修改后的画像值不能为空")
            if suggestion["field_path"].endswith("/level") and isinstance(final_value, str):
                match = re.search(r"[1-5]", final_value)
                if not match:
                    raise ProfileValidationError("技能等级必须是 1–5")
                final_value = int(match.group(0))
            change = ProfileChangeDraft(
                field_path=suggestion["field_path"], current_value=suggestion["current_value"],
                observed_change=suggestion["observed_change"], operation=suggestion["operation"],
                proposed_value=final_value, evidence_ids=suggestion["evidence_ids"],
                counter_evidence_ids=suggestion["counter_evidence_ids"],
                capability_id=suggestion.get("capability_id") or "",
                new_capability=(NewCapabilityDraft.model_validate(suggestion["new_capability"])
                                if suggestion.get("new_capability") else None),
                rationale=suggestion["rationale"], requirement_status="satisfied",
            )
            evidence_by_id = {item["evidence_id"]: item for item in
                              self.list_evidence(status="validated",
                                                 target_context_id=suggestion["target_context_id"])}
            status, gate_reason, _ = self._evaluate_change(
                change, updated_spec, evidence_by_id,
                self.capability_bounds(suggestion["target_context_id"]),
            )
            if status != "pending":
                raise ProfileValidationError("修改后的建议未通过证据校验：" + gate_reason)
            updated_spec = self._apply_suggestion(updated_spec, suggestion, final_value)
            new_content = render_profile_markdown(updated_spec)

        decision_event = self._event_envelope(
            "profile_suggestion_decided",
            {"suggestion_id": suggestion_id, "decision": decision,
             "reason": reason[:500], "evidence_ids": suggestion["evidence_ids"]},
            actor="user", source="profile_review", operation_id=f"{operation_id}:decision",
            entity_type="profile_suggestion", entity_id=suggestion_id,
            target_context_id=suggestion["target_context_id"],
        )
        profile_event = None
        if decision != "rejected":
            before_snapshot = f"snap_{_hash_text(current['content_md'])}"
            after_snapshot = f"snap_{_hash_text(new_content)}"
            profile_event = self._event_envelope(
                "profile_edited",
                {"before_snapshot_id": before_snapshot, "after_snapshot_id": after_snapshot,
                 "profile_hash": _hash_text(new_content), "reason": reason[:500]},
                actor="user", source="profile_review", operation_id=f"{operation_id}:profile",
                entity_type="profile", entity_id=PROFILE_ID,
                target_context_id=suggestion["target_context_id"],
            )
        stamp = now_iso()
        with self.store.transaction() as conn:
            concurrent_replay = conn.execute(
                "SELECT * FROM profile_suggestion_decisions WHERE operation_id=?", (operation_id,),
            ).fetchone()
            if concurrent_replay:
                return {"status": "ok", "suggestion_id": suggestion_id,
                        "result_revision": concurrent_replay["result_revision"], "replayed": True}
            live_suggestion = conn.execute(
                "SELECT status,base_revision FROM profile_suggestions WHERE suggestion_id=?",
                (suggestion_id,),
            ).fetchone()
            live_current = self._current_row(conn)
            if not live_suggestion or live_suggestion["status"] != "pending":
                raise ProfileConflictError("画像建议已被其他操作处理")
            if (not live_current or int(live_current["revision"]) != expected_revision or
                    int(live_suggestion["base_revision"]) != expected_revision):
                raise ProfileConflictError("画像版本已变化，请重新审计建议")
            if decision != "rejected":
                self.store.put_snapshot_in_transaction(
                    conn, current["content_md"], media_type="text/markdown", source_path="user_profile.md",
                )
                self.store.put_snapshot_in_transaction(
                    conn, new_content, media_type="text/markdown", source_path="user_profile.md",
                )
                result_revision = self._insert_revision(
                    conn, updated_spec, new_content, reason=reason or "用户确认画像建议", actor="user",
                )
                if suggestion.get("capability_id"):
                    conn.execute(
                        "UPDATE profile_capabilities SET lifecycle='active',updated_at=? "
                        "WHERE capability_id=? AND lifecycle='provisional'",
                        (stamp, suggestion["capability_id"]),
                    )
                conn.execute(
                    "UPDATE profile_suggestions SET status='stale',updated_at=? "
                    "WHERE status IN ('pending','needs_evidence') AND suggestion_id!=?",
                    (stamp, suggestion_id),
                )
            conn.execute(
                "UPDATE profile_suggestions SET status=?,updated_at=? WHERE suggestion_id=?",
                (decision, stamp, suggestion_id),
            )
            conn.execute(
                """INSERT INTO profile_suggestion_decisions(
                    decision_id,suggestion_id,decision,final_value_json,reason,operation_id,
                    result_revision,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (new_id("profile_decision"), suggestion_id, decision,
                 json.dumps(final_value, ensure_ascii=False), reason[:500], operation_id,
                 result_revision, stamp),
            )
            self.store.insert_event_in_transaction(conn, decision_event)
            if profile_event:
                self.store.insert_event_in_transaction(conn, profile_event)
        export_error = ""
        if decision != "rejected":
            try:
                self.export_current()
            except Exception as exc:
                export_error = str(exc)
        result = {"status": "ok", "decision": decision, "suggestion_id": suggestion_id,
                  "result_revision": result_revision,
                  "memory_event_id": decision_event["event_id"]}
        if export_error:
            result["export_error"] = export_error
        return result

    def migration_report(self) -> dict[str, Any]:
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profile_migrations ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return {"status": "missing"}
        value = dict(row)
        value["report"] = json.loads(value.pop("report_json") or "{}")
        return value


def format_audit_markdown(result: dict[str, Any], repository: ProfileRepository) -> str:
    lines = ["## 画像证据审计"]
    if result.get("status") == "review_unavailable":
        return "\n".join(lines + ["- LLM 审查暂不可用；有效证据已保留，正式画像未变化。"])
    created_ids = {item["suggestion_id"] for item in result.get("created") or []}
    suggestions = [item for item in repository.list_suggestions() if item["suggestion_id"] in created_ids]
    if not suggestions:
        return "\n".join(lines + ["- 本轮没有通过证据门槛的画像修改建议。"])
    for index, item in enumerate(suggestions, 1):
        lines.extend([
            f"### 建议 {index}：{item['field_path']}",
            f"- 当前值：{item['current_text']}", f"- 建议值：{item['proposed_text']}",
            f"- 状态：{item['status']}", f"- 理由：{item['rationale']}",
            "- 证据：" + "；".join(
                f"{evidence.get('occurred_on') or '日期未知'} {evidence.get('source_quote', '')[:120]}"
                for evidence in item.get("evidence_refs") or []
            ),
        ])
    return "\n".join(lines)


__all__ = [
    "EvidenceCandidate", "EvidenceExtractionDraft", "ProfileAuditDraft",
    "ProfileConflictError", "ProfileRepository", "ProfileSpec",
    "ProfileValidationError", "format_audit_markdown", "parse_profile_markdown",
    "render_profile_markdown",
]
