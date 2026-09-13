# -*- coding: utf-8 -*-
"""Server-side dialogue state for typed follow-up references.

Only completed routing/execution metadata is stored here. Raw chat text remains
in episodic memory and is never concatenated to guess a follow-up antecedent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
import re
from typing import Any

from memory_store import MemoryStore, new_id


CONTEXT_TTL_HOURS = 24

_REFERENCE_RE = re.compile(
    r"这些|那些|这条|那条|这个|那个|它们|它|上述|前面(?:的)?|刚才(?:的)?|上一(?:个|条|轮)"
)
_SINGULAR_RE = re.compile(r"这条|那条|这个|那个|它|上一(?:个|条)")
_ENTITY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("application_jd", ("jd", "职位描述", "岗位要求")),
    ("application", ("公司", "岗位", "投递", "申请")),
    ("project", ("项目", "仓库")),
    ("resume", ("简历", "经历段")),
    ("plan", ("计划", "任务", "安排")),
    ("profile", ("画像", "能力", "技能")),
    ("reflection", ("复盘", "学习记录", "执行记录", "留痕")),
    ("knowledge", ("资料", "知识库", "论文", "文档")),
)


@dataclass
class ContextResolution:
    status: str = "none"
    source: str = "server"
    turn_id: str = ""
    topic: str = ""
    entity_type: str = ""
    target_refs: list[dict[str, str]] = field(default_factory=list)
    output_contracts: list[dict[str, str]] = field(default_factory=list)
    capability_ids: list[str] = field(default_factory=list)
    reason: str = ""
    legacy_text: str = ""

    def to_dict(self, *, include_legacy_text: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_legacy_text:
            value.pop("legacy_text", None)
        return value

    def planner_payload(self) -> dict[str, Any] | None:
        if self.status == "none":
            return None
        return self.to_dict(include_legacy_text=True)


def _cutoff() -> str:
    return (
        dt.datetime.now(dt.timezone.utc).astimezone()
        - dt.timedelta(hours=CONTEXT_TTL_HOURS)
    ).isoformat(timespec="seconds")


def _expected_entity_type(question: str) -> str:
    q = str(question or "").lower()
    for entity_type, cues in _ENTITY_CUES:
        if any(cue in q for cue in cues):
            return entity_type
    return ""


def _entity_types(turn: dict[str, Any]) -> set[str]:
    return {
        str(item.get("entity_type") or "")
        for item in turn.get("output_contracts") or []
        if item.get("entity_type")
    }


def _target_refs(turn: dict[str, Any], entity_type: str) -> list[dict[str, str]]:
    resolved = dict(turn.get("resolved_entities") or {})
    if entity_type == "system_capability":
        return [{"entity_type": entity_type, "entity_id": str(value)}
                for value in (turn.get("capability_ids") or []) if str(value)]
    keys = {
        "application": ("application_ids", "applications"),
        "application_jd": ("jd_version_ids", "application_jds"),
        "project": ("project_ids", "projects"),
        "resume": ("resume_ids", "resumes"),
        "plan": ("plan_ids", "plans"),
        "reflection": ("reflection_ids", "reflections"),
    }.get(entity_type, (f"{entity_type}_ids", f"{entity_type}s"))
    refs: list[dict[str, str]] = []
    for raw in resolved.get(keys[0]) or []:
        entity_id = str(raw or "").strip()
        if entity_id:
            refs.append({"entity_type": entity_type, "entity_id": entity_id})
    if not refs:
        for item in resolved.get(keys[1]) or []:
            if not isinstance(item, dict):
                continue
            entity_id = str(
                item.get(f"{entity_type}_id")
                or item.get("application_id")
                or item.get("project_id")
                or item.get("id") or ""
            ).strip()
            if entity_id:
                refs.append({"entity_type": entity_type, "entity_id": entity_id})
    return list({item["entity_id"]: item for item in refs}.values())


def resolve_conversation_context(question: str, conversation_id: str = "", *,
                                 legacy_context: list[str] | None = None,
                                 store: MemoryStore | None = None) -> ContextResolution:
    """Expose a typed prior-turn candidate and resolve explicit references.

    A candidate is never applied by this function. The semantic planner first
    decides whether the current turn is standalone, a continuation, or a
    correction; only the latter two may bind the server-authored turn ID.
    """
    question = str(question or "").strip()
    has_reference = bool(_REFERENCE_RE.search(question))
    expected = _expected_entity_type(question)
    if conversation_id:
        store = store or MemoryStore()
        turns = store.conversation_turns(
            conversation_id, limit=20, active_after=_cutoff(),
        )
        if turns and not has_reference:
            turn = turns[0]
            contract_types = _entity_types(turn)
            entity_type = next((
                str(item.get("entity_type") or "")
                for item in turn.get("output_contracts") or []
                if item.get("entity_type")
            ), next(iter(contract_types), ""))
            return ContextResolution(
                status="candidate", turn_id=str(turn.get("turn_id") or ""),
                topic=str(turn.get("topic") or ""), entity_type=entity_type,
                target_refs=_target_refs(turn, entity_type) if entity_type else [],
                output_contracts=list(turn.get("output_contracts") or []),
                capability_ids=list(turn.get("capability_ids") or []),
                reason="latest_successful_turn_available",
            )
        compatible = [turn for turn in turns
                      if not expected or expected in _entity_types(turn)]
        if compatible:
            turn = compatible[0]
            entity_type = expected or next(iter(_entity_types(turn)), "")
            refs = _target_refs(turn, entity_type) if entity_type else []
            if _SINGULAR_RE.search(question) and len(refs) > 1:
                return ContextResolution(
                    status="ambiguous", turn_id=str(turn.get("turn_id") or ""),
                    topic=str(turn.get("topic") or ""), entity_type=entity_type,
                    target_refs=refs,
                    output_contracts=list(turn.get("output_contracts") or []),
                    capability_ids=list(turn.get("capability_ids") or []),
                    reason="singular_reference_has_multiple_entities",
                )
            if refs or not entity_type:
                return ContextResolution(
                    status="resolved", turn_id=str(turn.get("turn_id") or ""),
                    topic=str(turn.get("topic") or ""), entity_type=entity_type,
                    target_refs=refs,
                    output_contracts=list(turn.get("output_contracts") or []),
                    capability_ids=list(turn.get("capability_ids") or []),
                    reason="latest_type_compatible_successful_turn",
                )
            return ContextResolution(
                status="missing_entity_id", turn_id=str(turn.get("turn_id") or ""),
                topic=str(turn.get("topic") or ""), entity_type=entity_type,
                reason="antecedent_has_no_stable_entity_id",
            )
    if not has_reference:
        return ContextResolution(status="none", reason="no_reference_expression")
    legacy = [str(item).strip()[:600] for item in (legacy_context or []) if str(item).strip()]
    if legacy:
        return ContextResolution(
            status="legacy_client_context", source="legacy_client",
            reason="server_context_unavailable_using_latest_client_turn",
            legacy_text=legacy[-1],
        )
    return ContextResolution(
        status="missing_antecedent", entity_type=expected,
        reason="reference_has_no_compatible_successful_turn",
    )


def record_successful_turn(conversation_id: str, result: dict[str, Any], *,
                           turn_id: str = "", store: MemoryStore | None = None) -> dict[str, Any] | None:
    """Persist a compact context record from a completed query result/meta."""
    if not conversation_id or result.get("decision", "answer") != "answer":
        return None
    routes = list(result.get("routes") or [])
    if not routes:
        return None
    frame = dict(result.get("intent_frame") or {})
    contracts = list(frame.get("output_contracts") or [])
    if not contracts:
        contracts = [
            {"entity_type": value, "projection": "legacy"}
            for value in (frame.get("answer_objects") or []) if value
        ]
    resolved = dict(result.get("resolved_entities") or {})
    contract_types = {
        str(item.get("entity_type") or "") for item in contracts
        if item.get("entity_type")
    }
    if "knowledge" in contract_types and not resolved.get("knowledge_ids"):
        resolved["knowledge_ids"] = [
            str(value) for value in (result.get("sources") or []) if str(value)
        ]
    capabilities = list(
        resolved.get("capability_ids")
        or frame.get("capability_ids")
        or []
    )
    topics = list(dict.fromkeys(
        str(item.get("entity_type") or "") for item in contracts
        if item.get("entity_type")
    ))
    store = store or MemoryStore()
    return store.append_conversation_turn(
        conversation_id=conversation_id,
        turn_id=turn_id or new_id("turn"),
        topic="+".join(topics) or str(frame.get("service_mode") or "unknown"),
        service_mode=str(result.get("service_mode") or frame.get("service_mode") or ""),
        output_contracts=contracts,
        resolved_entities=resolved,
        capability_ids=capabilities,
    )


def clear_conversation_context(conversation_id: str,
                               store: MemoryStore | None = None) -> int:
    if not conversation_id:
        return 0
    return (store or MemoryStore()).clear_conversation(conversation_id)


__all__ = [
    "CONTEXT_TTL_HOURS", "ContextResolution", "clear_conversation_context",
    "record_successful_turn", "resolve_conversation_context",
]
