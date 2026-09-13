# -*- coding: utf-8 -*-
"""Recoverable bridge between human-readable files and SQLite memory events."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from io_utils import atomic_write_text, file_lock
from memory_store import new_id


class MemoryFileConflictError(RuntimeError):
    pass


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest() if value else ""


def _hash_text_bytes(value: bytes) -> str:
    if not value:
        return ""
    normalized = value.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_text_with_memory(path: str | Path, text: str, *, event_kind: str,
                           event_payload: dict[str, Any], event_options: dict[str, Any],
                           operation_id: str | None = None,
                           append: bool = False,
                           expected_before_hash: str = "") -> dict[str, Any]:
    """Atomically write a file and durably queue its idempotent memory event."""
    from memory_layers import EpisodicMemory, record_business_event
    target = Path(path).resolve()
    operation_id = operation_id or new_id("op")
    event_options = {**event_options, "operation_id": operation_id}
    epi = EpisodicMemory()
    existing_event = epi.store.get_event_by_operation(operation_id)
    if existing_event:
        return {"operation_id": operation_id,
                "memory_event_id": existing_event["event_id"], "replayed": True}
    previous = epi.store.get_file_operation(operation_id)
    if previous and previous["status"] == "pending":
        epi.store.recover_file_operations()
        existing_event = epi.store.get_event_by_operation(operation_id)
        if existing_event:
            return {"operation_id": operation_id,
                    "memory_event_id": existing_event["event_id"], "replayed": True}
        previous = epi.store.get_file_operation(operation_id)
    if previous and previous["status"] == "conflict":
        raise MemoryFileConflictError("previous operation conflicts with the current file")
    with file_lock(str(target)):
        before = target.read_bytes() if target.exists() else b""
        if expected_before_hash and _hash_text_bytes(before) != expected_before_hash:
            raise MemoryFileConflictError("source file changed before commit")
        if append:
            text = before.decode("utf-8") + text
        after = text.encode("utf-8")
        epi.store.begin_file_operation(
            operation_id, event_kind, str(target), _hash_bytes(before), _hash_bytes(after),
            {"event_kind": event_kind, "event_payload": event_payload,
             "event_options": event_options},
        )
        try:
            atomic_write_text(str(target), text)
        except BaseException:
            epi.store.finish_file_operation(operation_id, status="failed")
            raise
    try:
        event = record_business_event(event_kind, event_payload, **event_options)
    except Exception as exc:
        # Keep the operation pending. recover_file_operations() can replay the
        # event because the full validated intent is in detail_json.
        return {"operation_id": operation_id, "memory_event_id": "", "memory_error": str(exc)}
    epi.store.finish_file_operation(operation_id, status="committed",
                                    detail={"event_id": event["event_id"]})
    return {"operation_id": operation_id, "memory_event_id": event["event_id"]}
