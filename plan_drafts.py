# -*- coding: utf-8 -*-
"""Reviewable plan drafts with optimistic approval checks."""
from __future__ import annotations

import hashlib
from typing import Any

from memory_store import MemoryStore, new_id, now_iso


class PlanDraftNotFoundError(KeyError):
    pass


class StalePlanDraftError(RuntimeError):
    pass


def _content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _current_plan_version() -> dict[str, Any]:
    from plan_gen import load_latest_plan
    current = load_latest_plan()
    if not current:
        return {"filename": "", "mtime": 0, "content_hash": ""}
    return {
        "filename": current.get("filename") or "",
        "mtime": int(current.get("mtime") or 0),
        "content_hash": _content_hash(current.get("content") or ""),
    }


def create_plan_draft(content: str, *, changes: list[str] | None = None,
                      daily_days: int = -1, source: str = "plan_gen") -> dict[str, Any]:
    from memory_layers import record_business_event
    store = MemoryStore()
    draft_id = new_id("draft")
    snapshot = store.put_snapshot(content, media_type="text/markdown")
    data = {
        "draft_id": draft_id,
        "status": "pending",
        "content": content,
        "content_hash": snapshot["content_hash"],
        "snapshot_id": snapshot["snapshot_id"],
        "target_context_id": store.active_goal_id(),
        "base_plan": _current_plan_version(),
        "changes": list(changes or []),
        "daily_days": int(daily_days),
        "created_at": now_iso(),
    }
    store.set_runtime(f"plan_draft:{draft_id}", data)
    event = record_business_event(
        "plan_draft_generated",
        {key: value for key, value in data.items() if key != "content"},
        actor="assistant", source=source, operation_id=f"plan-draft:{draft_id}:generated",
        entity_type="plan_draft", entity_id=draft_id,
        target_context_id=data["target_context_id"],
    )
    data["event_id"] = event["event_id"]
    store.set_runtime(f"plan_draft:{draft_id}", data)
    return data


def decide_plan_draft(draft_id: str, decision: str) -> dict[str, Any]:
    from memory_layers import record_business_event
    store = MemoryStore()
    existing = store.get_event_by_operation(f"plan-draft:{draft_id}:{decision}")
    if existing:
        return {"draft_id": draft_id, "decision": decision,
                "saved_path": existing.get("saved_path") or "", "replayed": True}
    key = f"plan_draft:{draft_id}"
    draft = store.get_runtime(key)
    if not isinstance(draft, dict) or draft.get("status") != "pending":
        raise PlanDraftNotFoundError(draft_id)
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if decision == "approve":
        if store.active_goal_id() != draft.get("target_context_id"):
            raise StalePlanDraftError("职业目标已变化，请按当前目标重新生成计划草稿")
        if _current_plan_version().get("content_hash") != draft.get("base_plan", {}).get("content_hash"):
            raise StalePlanDraftError("当前计划已更新，请重新生成草稿后再批准")
        from plan_gen import save_plan
        saved_path = save_plan(str(draft.get("content") or ""),
                               operation_id=f"plan-draft:{draft_id}:save")
    else:
        saved_path = ""
    record_business_event(
        "plan_draft_decided",
        {"draft_id": draft_id, "decision": decision,
         "snapshot_id": draft.get("snapshot_id"), "saved_path": saved_path},
        actor="user", source="plan_approval",
        operation_id=f"plan-draft:{draft_id}:{decision}",
        entity_type="plan_draft", entity_id=draft_id,
        causation_id=draft.get("event_id"), target_context_id=draft.get("target_context_id"),
    )
    store.delete_runtime(key)
    return {"draft_id": draft_id, "decision": decision, "saved_path": saved_path}
