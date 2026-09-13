# -*- coding: utf-8 -*-
"""Stdin-only Windows data owner used by the WSL WeChat dispatcher.

The process intentionally has no listener and no arbitrary command/path API.
It runs next to the authoritative Windows OfferClaw files so existing Windows
file locks and SQLite transactions remain the only writers of business state.
"""
from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REQUEST_SCHEMA = "offerclaw.windows-bridge.request.v1"
RESPONSE_SCHEMA = "offerclaw.windows-bridge.response.v1"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
SOURCE_FILES = {
    "user_profile.md": "profile",
    "daily_log.md": "log",
    "applications.md": "application",
    "interview_story_bank.md": "project_context",
}
SOURCE_ROOTS = {
    "knowledge_base": "resource",
    "learning_resources": "learning_resource",
    "summaries": "reflection_summary",
    "plans": "plan",
    "application_jds": "application_jd",
}
SOURCE_EXTENSIONS = {".md", ".markdown", ".txt"}
QUERY_SERVICE_URL = "http://127.0.0.1:8000"

_PAYLOAD_FIELDS: dict[str, set[str]] = {
    "source.manifest": set(),
    "source.read": {"source_id"},
    "profile.get": set(),
    "applications.list": set(),
    "today.get": set(),
    "daily.get": set(),
    "plan.get": {"date"},
    "suggestions.list": {"status"},
    "suggestions.show": {"suggestion_id"},
    "query.execute": {"question"},
    "query.answer": {
        "question", "conversation_id", "message_id", "operation_id", "top_k",
    },
    "query.health": set(),
    "application.preview": {"application_id", "status_code"},
    "application.commit": {"application_id", "status_code", "base_hash", "operation_id"},
    "daily.preview": {"content"},
    "daily.commit": {"tag", "done", "todo", "notes", "base_hash", "operation_id"},
    "suggestions.commit": {
        "suggestion_id", "decision", "base_revision", "modified_value", "reason",
        "base_hash", "operation_id",
    },
    "health.get": set(),
}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_hash(value: Any) -> str:
    from business_read_service import stable_json_hash

    return stable_json_hash(value)


def _source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[Path, str]] = []
    for name, source_type in SOURCE_FILES.items():
        candidates.append((BASE_DIR / name, source_type))
    for root_name, default_type in SOURCE_ROOTS.items():
        root = BASE_DIR / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS
                    and not any(part.startswith("_") for part in path.relative_to(root).parts)):
                candidates.append((path, default_type))
    for path, default_type in candidates:
        if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
            continue
        relative = path.relative_to(BASE_DIR).as_posix()
        raw = path.read_bytes()
        source_type = default_type
        if relative.startswith("knowledge_base/"):
            head = raw[:4000].decode("utf-8", errors="ignore")
            match = re.search(r"^source_type:\s*[\"']?([^\s\"']+)", head, re.M | re.I)
            if match:
                source_type = match.group(1)
        rows.append({
            "source_id": relative,
            "source_type": source_type,
            "size": len(raw),
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha_bytes(raw),
        })
    return sorted(rows, key=lambda item: item["source_id"])


def _resolve_source(source_id: str) -> Path:
    normalized = str(source_id or "").strip().replace("\\", "/")
    manifest = {row["source_id"] for row in _source_rows()}
    if normalized not in manifest:
        raise ValueError("source_id 不允许访问：不在真实数据白名单中")
    path = (BASE_DIR / normalized).resolve()
    if BASE_DIR.resolve() not in path.parents:
        raise ValueError("source_id 越出 OfferClaw 数据目录")
    return path


class _ReadOnlySemanticMemory:
    def get(self, _key: str) -> dict:
        return {}

    def set(self, _key: str, _value: Any) -> None:
        return None


def _recent_daily() -> dict:
    from summary_tool import extract_recent_blocks

    path = BASE_DIR / "daily_log.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    recent = extract_recent_blocks(text, days=7)
    return {
        "status": "ok", "source": "daily_log.md",
        "as_of": dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        ) if path.is_file() else "",
        "content": recent[:12000],
    }


def _execute_query(question: str) -> dict:
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query

    plan = rule_plan_query(question)

    def no_vector_retrieve(*_args, **_kwargs):
        return {"in_kb": False, "chunks": [], "docs": [], "metas": []}

    execution = execute_plan(plan, question, 5, retrieve=no_vector_retrieve)
    return {
        "status": "ok",
        "plan": plan.to_dict(),
        "execution": {
            "deterministic_answer": execution.deterministic_answer,
            "evidence": [item.to_dict() for item in execution.evidence],
            "source_status": execution.source_status,
            "sources": execution.sources,
            "coverage": execution.coverage,
            "freshness": execution.freshness,
            "missing_personal_routes": execution.missing_personal_routes,
            "resolved_entities": execution.resolved_entities,
        },
    }


def _query_token_path() -> Path:
    configured = os.environ.get("OFFERCLAW_WECHAT_QUERY_TOKEN_FILE", "").strip()
    return Path(configured) if configured else Path.home() / ".offerclaw-runtime" / "wechat-query.token"


def _query_service_request(path: str, *, payload: dict[str, Any] | None = None,
                           timeout: float = 42.0) -> dict[str, Any]:
    """Call the one fixed loopback service; proxy settings are deliberately ignored."""
    import requests

    try:
        token = _query_token_path().read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeError("本机查询令牌不可用") from exc
    if len(token) < 32:
        raise RuntimeError("本机查询令牌无效")
    headers = {
        "X-OfferClaw-Internal-Token": token,
        "X-OfferClaw-Traffic-Origin": "wechat_direct",
    }
    session = requests.Session()
    session.trust_env = False
    url = QUERY_SERVICE_URL + path
    try:
        response = (
            session.post(url, json=payload, headers=headers, timeout=timeout)
            if payload is not None
            else session.get(url, headers=headers, timeout=min(timeout, 5.0))
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("OfferClaw Windows 查询服务不可用") from exc
    if not isinstance(result, dict) or result.get("status") not in {"ok", "partial"}:
        raise RuntimeError("OfferClaw Windows 查询服务返回无效")
    return result


def _answer_query(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("question 不能为空")
    request = {
        "schema_version": "offerclaw.wechat-query.request.v1",
        "question": question,
        "conversation_id": str(payload.get("conversation_id") or ""),
        "message_id": str(payload.get("message_id") or ""),
        "operation_id": str(payload.get("operation_id") or ""),
        "top_k": int(payload.get("top_k") or 5),
    }
    return _query_service_request("/api/internal/wechat-query", payload=request)


def _dispatch(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    if operation == "source.manifest":
        return {"status": "ok", "sources": _source_rows()}
    if operation == "source.read":
        path = _resolve_source(payload.get("source_id", ""))
        raw = path.read_bytes()
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("来源文件超过微信索引桥上限")
        return {
            "status": "ok", "source_id": path.relative_to(BASE_DIR).as_posix(),
            "sha256": _sha_bytes(raw), "text": raw.decode("utf-8", errors="replace"),
        }
    if operation == "profile.get":
        from business_read_service import profile_snapshot

        return {key: value for key, value in profile_snapshot().items() if key != "content_md"}
    if operation == "applications.list":
        from business_read_service import applications_snapshot

        return applications_snapshot()
    if operation == "today.get":
        from career_agent import get_today_advice

        return {"status": "ok", "advice": get_today_advice(sem=_ReadOnlySemanticMemory())}
    if operation == "daily.get":
        return _recent_daily()
    if operation == "plan.get":
        from plan_gen import summarize_plan_for_automation

        return {"status": "ok", "plan": summarize_plan_for_automation(payload.get("date"))}
    if operation == "suggestions.list":
        from business_read_service import suggestions_snapshot

        return suggestions_snapshot(str(payload.get("status") or "pending"))
    if operation == "suggestions.show":
        from business_read_service import suggestion_snapshot

        return suggestion_snapshot(str(payload.get("suggestion_id") or ""))
    if operation == "query.execute":
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("question 不能为空")
        return _execute_query(question)
    if operation == "query.answer":
        return _answer_query(payload)
    if operation == "query.health":
        return _query_service_request("/api/internal/wechat-health", timeout=5.0)
    if operation == "application.preview":
        from applications_store import get_application
        from domain_status import APPLICATION_STATUS_LABELS, application_status_code

        current = get_application(str(payload.get("application_id") or ""))
        if not current:
            raise ValueError("找不到 application_id")
        code = application_status_code(str(payload.get("status_code") or ""))
        return {
            "status": "preview", "capability_id": "application.update_status",
            "payload": {"application_id": current["application_id"], "status_code": code.value},
            "preview": {"company": current.get("company"), "position": current.get("position"),
                        "before": {"status": current.get("status"), "status_code": current.get("status_code")},
                        "after": {"status": APPLICATION_STATUS_LABELS[code], "status_code": code.value}},
            "base_hash": _json_hash(current),
        }
    if operation == "application.commit":
        from applications_store import get_application, patch_application
        from domain_status import APPLICATION_STATUS_LABELS, application_status_code

        application_id = str(payload.get("application_id") or "")
        current = get_application(application_id)
        if _json_hash(current) != str(payload.get("base_hash") or ""):
            return {"status": "conflict", "error": "投递记录已变化，请重新生成预览"}
        code = application_status_code(str(payload.get("status_code") or ""))
        return patch_application(
            application_id, status=APPLICATION_STATUS_LABELS[code],
            operation_id=str(payload.get("operation_id") or ""),
        )
    if operation == "daily.preview":
        from offerclaw_cli import _parse_structured_log

        content = str(payload.get("content") or "")
        tag, done, todo, notes = _parse_structured_log(content)
        if not any((tag, done, todo, notes)):
            raise ValueError("留痕内容不能为空")
        path = BASE_DIR / "daily_log.md"
        return {
            "status": "preview", "capability_id": "daily_log.create",
            "payload": {"tag": tag, "done": done, "todo": todo, "notes": notes},
            "preview": {"tag": tag, "done": done, "todo": todo, "notes": notes},
            "base_hash": _sha_bytes(path.read_bytes()) if path.is_file() else "",
        }
    if operation == "daily.commit":
        from summary_tool import append_structured_daily_log

        path = BASE_DIR / "daily_log.md"
        current_hash = _sha_bytes(path.read_bytes()) if path.is_file() else ""
        if current_hash != str(payload.get("base_hash") or ""):
            return {"status": "conflict", "error": "每日留痕已变化，请重新生成预览"}
        return append_structured_daily_log(
            tag=str(payload.get("tag") or ""), done=payload.get("done") or [],
            todo=payload.get("todo") or [], notes=str(payload.get("notes") or ""),
            operation_id=str(payload.get("operation_id") or ""),
        )
    if operation == "suggestions.commit":
        from profile_store import decide_suggestion, list_suggestions

        suggestion_id = str(payload.get("suggestion_id") or "")
        row = next((item for item in list_suggestions() if item.get("suggestion_id") == suggestion_id), None)
        if not row:
            raise ValueError("画像建议不存在")
        decision = str(payload.get("decision") or "")
        if decision not in {"accepted", "modified", "rejected"}:
            raise ValueError("画像建议决定非法")
        return decide_suggestion(
            suggestion_id, decision, modified_value=payload.get("modified_value"),
            base_revision=int(payload.get("base_revision") or row.get("base_revision")),
            reason=str(payload.get("reason") or "微信明确确认"),
            operation_id=str(payload.get("operation_id") or ""),
        )
    if operation == "health.get":
        manifest = _source_rows()
        return {
            "status": "healthy", "authoritative_root": str(BASE_DIR),
            "source_count": len(manifest), "policy": "windows_single_writer",
        }
    raise ValueError(f"operation 不在允许列表：{operation}")


def handle(request: dict[str, Any]) -> dict[str, Any]:
    unknown_request_fields = set(request) - {"schema_version", "operation", "payload"}
    if unknown_request_fields:
        raise ValueError(f"bridge request 包含未知字段：{sorted(unknown_request_fields)}")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("bridge schema_version 不受支持")
    operation = str(request.get("operation") or "")
    if operation not in _PAYLOAD_FIELDS:
        raise ValueError(f"operation 不在允许列表：{operation}")
    payload = request.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("payload 必须是 JSON object")
    unknown_payload_fields = set(payload) - _PAYLOAD_FIELDS[operation]
    if unknown_payload_fields:
        raise ValueError(f"payload 包含未知字段：{sorted(unknown_payload_fields)}")
    result = _dispatch(operation, payload)
    return {"schema_version": RESPONSE_SCHEMA, "operation": operation, **result}


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("bridge request 过大")
    request = json.loads(raw.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("bridge request 必须是 JSON object")
    print(json.dumps(handle(request), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({
            "schema_version": RESPONSE_SCHEMA, "status": "error",
            "error_type": type(exc).__name__, "error": str(exc)[:500],
        }, ensure_ascii=False, separators=(",", ":")))
        raise SystemExit(1)
