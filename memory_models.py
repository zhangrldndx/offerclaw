# -*- coding: utf-8 -*-
"""Validated event contracts for personal memory."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain_status import ApplicationStatusCode, MatchStatusCode


_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class EventPayload(BaseModel):
    model_config = ConfigDict(extra="allow")


class MatchPayload(EventPayload):
    status_code: MatchStatusCode = MatchStatusCode.UNKNOWN
    status: str = ""
    direction: str = ""
    jd_title: str = ""


class ReflectionPayload(EventPayload):
    date: str = ""
    reflection_kind: Literal["daily", "weekly"] = "daily"
    deviation_score: int = Field(default=0, ge=0, le=100)
    completed: list[Any] = Field(default_factory=list)
    incomplete: list[Any] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    skill_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_candidates: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        if value:
            dt.date.fromisoformat(value)
        return value


class ConversationPayload(EventPayload):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)
    state: Literal["sent", "completed", "failed", "interrupted"] = "completed"
    message_id: str = ""
    source_refs: list[Any] = Field(default_factory=list)


class DailyLogPayload(EventPayload):
    log_id: str
    date: str
    status: Literal["done", "partial", "blocked", "deferred"] = "done"
    task_id: str = ""
    done: list[str] = Field(default_factory=list)
    incomplete: list[str] = Field(default_factory=list)
    notes: str = ""
    minutes: int | None = Field(default=None, ge=0, le=1440)


class ApplicationPayload(EventPayload):
    application_id: str
    status_code: ApplicationStatusCode
    status: str
    company: str = ""
    position: str = ""
    long_term_follow: bool | None = None


class ApplicationReviewPayload(EventPayload):
    application_id: str
    company: str = ""
    position: str = ""
    stage: str
    snapshot_id: str = Field(pattern=r"^snap_[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    saved_path: str


class ProfileEditedPayload(EventPayload):
    before_snapshot_id: str = Field(pattern=r"^snap_[0-9a-f]{64}$")
    after_snapshot_id: str = Field(pattern=r"^snap_[0-9a-f]{64}$")
    profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(default="", max_length=500)


class PlanSavedPayload(EventPayload):
    snapshot_id: str = Field(pattern=r"^snap_[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    saved_path: str = Field(min_length=1, max_length=500)
    edited_by_user: bool = False


class HistoricalSnapshotPayload(EventPayload):
    source_kind: str
    snapshot_id: str = Field(pattern=r"^snap_[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str = ""
    historical_snapshot: bool = True


class KnowledgeMaterialPayload(EventPayload):
    action: Literal["submitted", "promoted", "indexed"]
    source_path: str = ""
    source_type: str = ""
    snapshot_id: str | None = None
    content_hash: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class ResumeArtifactPayload(EventPayload):
    action: Literal["generated", "approved"]
    snapshot_id: str = Field(pattern=r"^snap_[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_id: str = ""
    saved_path: str = ""


class GoalSwitchedPayload(EventPayload):
    previous_context_id: str
    context_id: str
    name: str = Field(min_length=1, max_length=120)
    status: Literal["active"] = "active"
    version: int = Field(ge=1)


class ProfileSuggestionPayload(EventPayload):
    suggestion_id: str
    decision: Literal["accepted", "rejected", "modified"]
    reason: str = Field(default="", max_length=500)
    evidence_ids: list[str] = Field(default_factory=list)


PAYLOAD_MODELS: dict[str, type[EventPayload]] = {
    "match_run": MatchPayload,
    "career_flow_run": MatchPayload,
    "match_completed": MatchPayload,
    "reflection": ReflectionPayload,
    "conversation_message": ConversationPayload,
    "daily_log_recorded": DailyLogPayload,
    "application_changed": ApplicationPayload,
    "application_review_recorded": ApplicationReviewPayload,
    "profile_edited": ProfileEditedPayload,
    "profile_suggestion_decided": ProfileSuggestionPayload,
    "plan_saved": PlanSavedPayload,
    "historical_snapshot_imported": HistoricalSnapshotPayload,
    "knowledge_material_changed": KnowledgeMaterialPayload,
    "resume_artifact_changed": ResumeArtifactPayload,
    "goal_context_switched": GoalSwitchedPayload,
}


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(pattern=r"^ep_[0-9a-fA-F-]{36}$")
    schema_version: int = 1
    kind: str
    occurred_at: str
    recorded_at: str
    business_date: str | None = None
    actor: Literal["user", "assistant", "system", "import"]
    source: str = Field(min_length=1, max_length=100)
    traffic_origin: str = Field(min_length=1, max_length=50)
    operation_id: str | None = None
    causation_id: str | None = None
    conversation_id: str | None = None
    target_context_id: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    archived: bool = False
    payload: dict[str, Any]

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value: str) -> str:
        if not _KIND_RE.fullmatch(value):
            raise ValueError("kind 必须是 snake_case machine-readable code")
        return value

    @field_validator("occurred_at", "recorded_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        dt.datetime.fromisoformat(value)
        return value

    @field_validator("business_date")
    @classmethod
    def valid_business_date(cls, value: str | None) -> str | None:
        if value:
            dt.date.fromisoformat(value)
        return value


def validate_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = PAYLOAD_MODELS.get(kind, EventPayload)
    return model.model_validate(payload).model_dump(mode="json")
