# -*- coding: utf-8 -*-
"""Canonical read projections shared by Web APIs and the WeChat data bridge."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
_VERSION_FILES = (
    "user_profile.md", "daily_log.md", "applications.md", "interview_story_bank.md",
)
_VERSION_ROOTS = (
    "knowledge_base", "learning_resources", "summaries", "plans", "application_jds",
)


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def authoritative_data_version() -> str:
    """Hash authoritative business inputs without including derived indexes or chat state."""
    digest = hashlib.sha256()
    candidates = [BASE_DIR / name for name in _VERSION_FILES]
    for root_name in _VERSION_ROOTS:
        root = BASE_DIR / root_name
        if root.is_dir():
            candidates.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt"}
                and not any(part.startswith("_") for part in path.relative_to(root).parts)
            )
    for path in sorted((path for path in candidates if path.is_file()), key=lambda p: p.as_posix()):
        digest.update(path.relative_to(BASE_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def profile_snapshot() -> dict[str, Any]:
    from profile_loader import load_profile

    path = BASE_DIR / "user_profile.md"
    raw = path.read_bytes() if path.is_file() else b""
    source_hash = hashlib.sha256(raw).hexdigest()
    return {
        "status": "ok",
        "profile": load_profile(path=str(path), use_cache=False),
        "content_md": raw.decode("utf-8", errors="replace"),
        "revision": source_hash[:16] if source_hash else "missing",
        "base_hash": source_hash,
        "source": "user_profile.md",
        "as_of": dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        ) if path.is_file() else "",
    }


def applications_snapshot() -> dict[str, Any]:
    from applications_store import application_fact_views

    rows = application_fact_views()
    return {
        "status": "ok", "count": len(rows), "applications": rows,
        "revision": stable_json_hash(rows),
    }


def suggestions_snapshot(status: str = "") -> dict[str, Any]:
    from profile_store import list_suggestions

    try:
        rows = list_suggestions(status)
    except FileNotFoundError:
        # A public clone intentionally has no private user_profile.md yet.
        rows = []
    return {
        "status": "ok", "count": len(rows), "suggestions": rows,
        "revision": stable_json_hash(rows),
    }


def suggestion_snapshot(suggestion_id: str) -> dict[str, Any]:
    rows = suggestions_snapshot()["suggestions"]
    row = next((item for item in rows if item.get("suggestion_id") == suggestion_id), None)
    return {
        "status": "ok" if row else "error", "suggestion": row,
        "error": "画像建议不存在" if not row else "",
        "revision": stable_json_hash(row) if row else "",
    }


__all__ = [
    "applications_snapshot", "authoritative_data_version", "profile_snapshot", "stable_json_hash",
    "suggestion_snapshot", "suggestions_snapshot",
]
