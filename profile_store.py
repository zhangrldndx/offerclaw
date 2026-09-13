# -*- coding: utf-8 -*-
"""Compatibility facade for versioned, evidence-grounded profile review."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from profile_review import (
    ProfileConflictError,
    ProfileRepository,
    ProfileValidationError,
    parse_profile_markdown,
)


BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "user_profile.md"
EVENT_DIR = BASE_DIR / "logs" / "profile"
EVENT_PATH = EVENT_DIR / "suggestions.jsonl"
REQUIRED_SECTIONS = ("## 0.", "## 2.", "## 3.", "## 9.", "## 10.")


def _repository() -> ProfileRepository:
    return ProfileRepository(profile_path=PROFILE_PATH)


def content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def read_profile() -> dict[str, Any]:
    current = _repository().current()
    try:
        display_path = str(PROFILE_PATH.relative_to(BASE_DIR))
    except ValueError:
        display_path = str(PROFILE_PATH)
    return {**current, "path": display_path}


def _validate(content: str) -> None:
    if not content.strip() or len(content) > 200_000:
        raise ProfileValidationError("画像内容为空或超过 200000 字符")
    try:
        parse_profile_markdown(content)
    except ValueError as exc:
        raise ProfileValidationError(str(exc)) from exc


def save_profile(content: str, base_hash: str, *, reason: str = "",
                 source: str = "用户手动编辑",
                 operation_id: str | None = None) -> dict[str, Any]:
    """Save an explicit user edit through the structured profile repository."""
    _validate(content)
    return _repository().save_markdown(
        content, base_hash=base_hash, reason=reason or source,
        operation_id=operation_id or "",
    )


def create_edit_preview(content: str, base_revision: int) -> dict[str, Any]:
    _validate(content)
    return _repository().create_edit_preview(content, base_revision=base_revision)


def commit_edit_preview(preview_id: str, *, reason: str = "",
                        operation_id: str = "") -> dict[str, Any]:
    return _repository().commit_edit_preview(
        preview_id, reason=reason, operation_id=operation_id,
    )


def list_suggestions(status: str = "") -> list[dict[str, Any]]:
    return _repository().list_suggestions(status)


def audit_profile_suggestions(max_items: int = 5, *, trigger_kind: str = "manual",
                              caller=None, model: str | None = None,
                              extract_sources: bool = True) -> dict[str, Any]:
    """Run the bounded LLM review; never fall back to mechanical score changes."""
    return _repository().run_audit(
        max_items=max_items, trigger_kind=trigger_kind,
        caller=caller, model=model, extract_sources=extract_sources,
    )


def decide_suggestion(suggestion_id: str, decision: str, *, modified_text: str = "",
                      modified_value: Any = None, base_revision: int | None = None,
                      reason: str = "", operation_id: str = "") -> dict[str, Any]:
    if modified_value is None and modified_text:
        modified_value = modified_text.strip()
    return _repository().decide_suggestion(
        suggestion_id, decision, base_revision=base_revision,
        modified_value=modified_value, reason=reason, operation_id=operation_id,
    )


def get_evidence(evidence_id: str) -> dict[str, Any] | None:
    return _repository().evidence(evidence_id, include_source=True)


def migration_report() -> dict[str, Any]:
    return _repository().migration_report()


__all__ = [
    "ProfileConflictError", "ProfileValidationError", "audit_profile_suggestions",
    "commit_edit_preview", "content_hash", "create_edit_preview", "decide_suggestion",
    "get_evidence", "list_suggestions", "migration_report", "read_profile", "save_profile",
]
