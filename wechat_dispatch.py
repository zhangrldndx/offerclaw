# -*- coding: utf-8 -*-
"""Deterministic, model-before-dispatch WeChat request router."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from action_capabilities import ACTION_CAPABILITY_REGISTRY, action_guidance
from io_utils import atomic_write_text, file_lock
from wechat_bridge import (
    WeChatPendingStore, request_context, stage_attachment,
)
from wechat_policy import ACTION_POLICIES, READ_POLICIES, policy_report
from wechat_action_router import action_router_contract, match_action_capability


REQUEST_SCHEMA = "offerclaw.wechat.request.v1"
RESPONSE_SCHEMA = "offerclaw.wechat.response.v1"
BRIDGE_REQUEST_SCHEMA = "offerclaw.windows-bridge.request.v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_TEXT_CHARS = 12_000
MAX_REPLY_CHARS = 1_800
_REQUEST_FIELDS = {
    "schema_version", "message_id", "conversation_id", "channel", "account_id",
    "sender_id", "is_group", "text", "reply_to", "media", "trigger", "automation_kind",
}
_REPLY_FIELDS = {"id", "body"}
_MEDIA_FIELDS = {"path", "workspace_dir", "content_type", "kind", "message_id"}
_WORKERS = threading.BoundedSemaphore(2)
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_GUARD = threading.Lock()
_FAILURES: dict[str, list[float]] = {}


class BridgeError(RuntimeError):
    pass


def _hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _opaque_identity(value: str, store: WeChatPendingStore, kind: str) -> str:
    configured = os.environ.get("OFFERCLAW_WECHAT_IDENTITY_SALT_FILE", "").strip()
    path = Path(configured) if configured else store.path.parent / "wechat-identity.salt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        raw = secrets.token_bytes(32)
        try:
            with path.open("xb") as handle:
                handle.write(raw)
        except FileExistsError:
            pass
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    salt = path.read_bytes()
    if len(salt) < 32:
        raise PermissionError("微信身份盐文件无效")
    digest = hmac.new(salt, str(value or "").encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{kind}_{digest}"


def _session_lock(key: str) -> threading.RLock:
    with _SESSION_GUARD:
        return _SESSION_LOCKS.setdefault(_hash(key)[:24], threading.RLock())


def _try_process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except (BlockingIOError, OSError):
        handle.close()
        return None


def _unlock_process(handle) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@contextmanager
def _cross_process_execution(store: WeChatPendingStore, context: dict[str, str]):
    """Serialize one session and enforce two worker slots across launcher processes."""
    lock_dir = store.path.parent / "wechat_locks"
    scope = "|".join(context[name] for name in (
        "channel", "account_id", "sender_id", "conversation_id",
    ))
    timeout = max(1.0, min(float(os.environ.get("OFFERCLAW_WECHAT_QUEUE_TIMEOUT", "40")), 40.0))
    deadline = time.monotonic() + timeout
    session_handle = None
    slot_handle = None
    try:
        session_path = lock_dir / f"session-{_hash(scope)[:24]}.lock"
        while session_handle is None and time.monotonic() < deadline:
            session_handle = _try_process_lock(session_path)
            if session_handle is None:
                time.sleep(0.025)
        if session_handle is None:
            raise TimeoutError("同一微信会话仍有请求在执行，请稍后重试")
        while slot_handle is None and time.monotonic() < deadline:
            for index in range(2):
                slot_handle = _try_process_lock(lock_dir / f"worker-{index}.lock")
                if slot_handle is not None:
                    break
            if slot_handle is None:
                time.sleep(0.025)
        if slot_handle is None:
            raise TimeoutError("OfferClaw 本地 worker 正忙，请稍后重试")
        yield
    finally:
        if slot_handle is not None:
            _unlock_process(slot_handle)
        if session_handle is not None:
            _unlock_process(session_handle)


def _scope_values(name: str) -> set[str]:
    return {item.strip() for item in os.environ.get(name, "").split(",") if item.strip()}


def _context(request: dict[str, Any]) -> dict[str, str]:
    return request_context(
        channel=request.get("channel"), account_id=request.get("account_id"),
        sender_id=request.get("sender_id"), conversation_id=request.get("conversation_id"),
        message_id=request.get("message_id"),
    )


def _validate_request(request: dict[str, Any]) -> None:
    if set(request) - _REQUEST_FIELDS:
        raise ValueError(f"wechat request 包含未知字段：{sorted(set(request) - _REQUEST_FIELDS)}")
    try:
        encoded_size = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("wechat request 不是可验证的 JSON 数据") from exc
    if encoded_size > MAX_REQUEST_BYTES:
        raise ValueError("wechat dispatch request 过大")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("wechat schema_version 不受支持")
    if str(request.get("channel") or "") != "openclaw-weixin":
        raise PermissionError("只允许 openclaw-weixin")
    if request.get("is_group") is not False:
        raise PermissionError("群聊入口已关闭")
    for key in ("message_id", "conversation_id", "account_id", "sender_id"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise PermissionError(f"缺少可信身份字段：{key}")
    trigger = request.get("trigger", "user")
    if trigger not in {"user", "cron"}:
        raise PermissionError("trigger 不受支持")
    if trigger == "cron":
        automation_kind = request.get("automation_kind")
        expected_marker = f"offerclaw://automation/{automation_kind}?v=1"
        if automation_kind not in {"morning", "evening", "weekly"} or request.get("text") != expected_marker:
            raise PermissionError("定时任务标记不受支持")
    elif request.get("automation_kind") not in {None, ""}:
        raise PermissionError("普通消息不能声明 automation_kind")
    accounts = _scope_values("OFFERCLAW_WECHAT_ALLOWED_ACCOUNT_IDS")
    senders = _scope_values("OFFERCLAW_WECHAT_ALLOWED_SENDER_IDS")
    if accounts and str(request.get("account_id")) not in accounts:
        raise PermissionError("微信账号不在本地直返允许列表")
    if (str(request.get("trigger") or "user") != "cron"
            and senders and str(request.get("sender_id")) not in senders):
        raise PermissionError("发送者不在本地直返允许列表")
    if not isinstance(request.get("text", ""), str):
        raise ValueError("消息正文必须是字符串")
    text = request.get("text") or ""
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("消息正文过长")
    media = request.get("media") or []
    if not isinstance(media, list) or len(media) > 4:
        raise ValueError("附件列表非法或过多")
    reply_to = request.get("reply_to") or {}
    if not isinstance(reply_to, dict) or set(reply_to) - _REPLY_FIELDS:
        raise ValueError("回复上下文字段非法")
    if any(not isinstance(value, str) for value in reply_to.values()):
        raise ValueError("回复上下文必须是字符串")
    if len(str(reply_to.get("body") or "")) > MAX_TEXT_CHARS:
        raise ValueError("回复上下文过长")
    for item in media:
        if not isinstance(item, dict) or set(item) - _MEDIA_FIELDS:
            raise ValueError("附件字段非法")
        if any(not isinstance(value, str) for value in item.values()):
            raise ValueError("附件字段必须是字符串")


def _validated_media_path(item: dict[str, Any]) -> Path:
    if not isinstance(item, dict):
        raise ValueError("附件信息必须是 JSON object")
    raw_source = str(item.get("path") or "").strip()
    raw_workspace = str(item.get("workspace_dir") or "").strip()
    if not raw_source or not raw_workspace:
        raise PermissionError("附件缺少 OpenClaw 本地路径或工作目录")
    source = Path(raw_source).expanduser().resolve()
    workspace = Path(raw_workspace).expanduser().resolve()
    if not source.is_file() or (source != workspace and workspace not in source.parents):
        raise PermissionError("附件路径不在 OpenClaw 提供的隔离工作目录中")
    return source


def _attachment_purpose(text: str) -> str:
    if re.search(r"简历|resume|cv", text, re.I):
        return "resume"
    if re.search(r"项目|project", text, re.I):
        return "project"
    if re.search(r"\bJD\b|岗位描述|职位描述", text, re.I):
        return "jd"
    if re.search(r"知识库|资料|knowledge", text, re.I):
        return "knowledge"
    raise ValueError("请在附件消息中说明用途：JD、简历、项目或知识库资料")


def _bridge_command() -> tuple[list[str], str | None]:
    python_bin = os.environ.get("OFFERCLAW_WINDOWS_PYTHON", "").strip()
    script = os.environ.get("OFFERCLAW_WINDOWS_BRIDGE_SCRIPT", "").strip()
    cwd = os.environ.get("OFFERCLAW_WINDOWS_REPO_WSL", "").strip() or None
    if not python_bin and not script:
        return [sys.executable, str(Path(__file__).with_name("wechat_data_bridge.py"))], str(
            Path(__file__).resolve().parent
        )
    if not python_bin or not script:
        raise BridgeError("Windows 数据桥配置不完整")
    return [python_bin, "-X", "utf8", script], cwd


def call_bridge(operation: str, payload: dict[str, Any], *, timeout: int = 15) -> dict[str, Any]:
    now = time.monotonic()
    recent = [value for value in _FAILURES.get("bridge", []) if now - value < 30]
    if len(recent) >= 3:
        raise BridgeError("Windows 数据桥熔断中，请 30 秒后重试")
    command, cwd = _bridge_command()
    request = json.dumps({
        "schema_version": BRIDGE_REQUEST_SCHEMA,
        "operation": operation, "payload": payload,
    }, ensure_ascii=False, separators=(",", ":"))
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY"}
    }
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if operation == "query.answer":
        timeout = max(timeout, 43)
    try:
        process = subprocess.run(
            command, input=request, text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=timeout, cwd=cwd, env=env, shell=False,
        )
        result = json.loads((process.stdout or "").strip())
        if process.returncode or result.get("status") == "error":
            raise BridgeError(str(result.get("error") or "Windows 数据桥执行失败"))
        _FAILURES["bridge"] = []
        return result
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, BridgeError) as exc:
        recent.append(time.monotonic())
        _FAILURES["bridge"] = recent[-3:]
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError(str(exc)) from exc


_SECRET_RE = re.compile(
    r"(?i)(?:api[_ -]?key|access[_ -]?token|secret|password|密码)\s*[:=：]|"
    r"\bsk-[A-Za-z0-9_-]{12,}\b|[A-Za-z]:\\|/mnt/[a-z]/|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)|(?<!\d)\d{17}[0-9Xx](?!\d)"
)
_PERSONAL_RE = re.compile(
    r"我|本人|我们|咱|画像|简历|投递|申请|公司|岗位|学习记录|"
    r"日志|留痕|计划|项目经历|面试经历|附件|微信|收入|薪资|住址|电话|邮箱"
)
_LOCAL_SYSTEM_RE = re.compile(
    r"OfferClaw|OpenClaw|系统状态|健康检查|数据桥|本地数据|知识库|画像建议|"
    r"wxact_[A-Za-z0-9_-]+|wxpage_[A-Fa-f0-9]+",
    re.I,
)


def classify_privacy(text: str, media: list[dict[str, Any]] | None = None) -> str:
    if media or _SECRET_RE.search(text or ""):
        return "sensitive"
    if _PERSONAL_RE.search(text or "") or _LOCAL_SYSTEM_RE.search(text or ""):
        return "personal"
    return "public"


def _page(text: str, _context_key: str) -> str:
    """Compatibility wrapper; complete replies are split by the response contract."""
    return text


def _next_page(text: str, context: dict[str, str],
               bridge: Callable[[str, dict[str, Any]], dict[str, Any]],
               store: WeChatPendingStore) -> str | None:
    match = re.fullmatch(r"\s*下一页\s+(wxpage_[a-f0-9]{12})\s*", text)
    if not match:
        return None
    token = match.group(1)
    cursor = store.cursor(token, context=context)
    if not cursor:
        return "分页已过期或不属于当前会话，请重新查询。"
    if cursor["kind"] == "applications":
        result = bridge("applications.list", {})
        rows_by_id = {
            str(item.get("application_id") or ""): item
            for item in result.get("applications") or []
        }
        rows = [rows_by_id[item] for item in cursor["references"] if item in rows_by_id]
        return _format_reference_page(
            "applications", rows, cursor["offset"], token, store,
            lambda page: _format_applications({
                "applications": page, "count": len(rows), "revision": result.get("revision", ""),
            }, offset=cursor["offset"]),
        )
    if cursor["kind"] == "suggestions":
        result = bridge("suggestions.list", {"status": "pending"})
        rows_by_id = {
            str(item.get("suggestion_id") or ""): item
            for item in result.get("suggestions") or []
        }
        rows = [rows_by_id[item] for item in cursor["references"] if item in rows_by_id]
        return _format_reference_page(
            "suggestions", rows, cursor["offset"], token, store,
            lambda page: _format_suggestions({
                "suggestions": page, "count": len(rows), "revision": result.get("revision", ""),
            }, offset=cursor["offset"]),
        )
    store.advance_cursor(token, None)
    return "分页类型已失效，请重新查询。"


def _format_reference_page(kind: str, rows: list[dict[str, Any]], offset: int,
                           token: str, store: WeChatPendingStore,
                           formatter: Callable[[list[dict[str, Any]]], str]) -> str:
    page_rows = rows[offset:offset + 5]
    if not page_rows:
        store.advance_cursor(token, None)
        return "已经是最后一页。"
    next_offset = offset + len(page_rows)
    if next_offset < len(rows):
        store.advance_cursor(token, next_offset)
        suffix = f"\n\n下一页：回复 下一页 {token}"
    else:
        store.advance_cursor(token, None)
        suffix = ""
    return formatter(page_rows) + suffix


def _format_profile(result: dict[str, Any]) -> str:
    profile = result.get("profile") or {}
    fields = (
        ("所在地", profile.get("所在地")), ("目标岗位", profile.get("目标岗位类型")),
        ("方向优先级", profile.get("方向优先级")), ("行业偏好", profile.get("行业偏好")),
        ("期望薪资", profile.get("期望薪资")), ("熟练技能", profile.get("熟练技能")),
        ("会用技能", profile.get("会用技能")), ("项目数量", profile.get("项目数量")),
        ("实习数量", profile.get("实习数量")), ("英语自评", profile.get("英语自评")),
    )
    lines = ["当前正式画像："]
    for label, value in fields:
        if value not in (None, "", []):
            rendered = "、".join(map(str, value)) if isinstance(value, list) else str(value)
            lines.append(f"- {label}：{rendered}")
    lines.append(f"- 数据版本：revision {result.get('revision') or '未记录'}")
    return "\n".join(lines)


def _format_applications(result: dict[str, Any], *, offset: int = 0) -> str:
    rows = result.get("applications") or []
    total = int(result.get("count", len(rows)) or len(rows))
    lines = [f"当前共有 {total} 条投递记录（第 {offset + 1}-{offset + len(rows)} 条）："]
    for item in rows:
        lines.append(
            f"- {item.get('company') or '未记录'}｜{item.get('position') or '未记录'}"
            f"｜{item.get('status') or '未记录'}｜{item.get('date') or '日期未记录'}"
            f"｜ID {item.get('application_id') or '未记录'}"
        )
        if item.get("next_action") and item.get("next_action") != "—":
            lines.append(f"  下一步：{item['next_action']}")
    lines.append(f"数据版本：{result.get('revision', '')[:12]}")
    return "\n".join(lines)


def _format_today(result: dict[str, Any]) -> str:
    value = result.get("advice") or {}
    lines = [value.get("headline") or "今日建议"]
    if value.get("reason"):
        lines.append(value["reason"])
    for item in (value.get("next_actions") or [])[:6]:
        lines.append(f"- {item}")
    drift = value.get("plan_drift") or {}
    if drift.get("level") in {"info", "warn"} and drift.get("message"):
        lines.append(f"计划偏离：{drift['message']}")
    lines.append(f"来源：{value.get('source') or '本地规则'}｜截至 {value.get('today') or ''}")
    return "\n".join(lines)


def _format_plan(result: dict[str, Any]) -> str:
    plan = result.get("plan") or {}
    if not plan.get("has_plan"):
        return "当前没有可读取的正式计划。"
    week = plan.get("current_week") or {}
    lines = [f"当前计划：{plan.get('period') or '周期未记录'}"]
    lines.append(f"- 第 {week.get('n', '—')} 周：{week.get('theme') or '主题未记录'}")
    if week.get("deliverable"):
        lines.append(f"- 本周交付：{week['deliverable']}")
    for task in (plan.get("today_tasks") or [])[:5]:
        lines.append(f"- 今日：{task}")
    if plan.get("expired"):
        lines.append("- 当前计划已过期；不会自动重排。")
    lines.append(f"来源：{plan.get('plan_file') or 'plans/'}")
    return "\n".join(lines)


def _format_daily(result: dict[str, Any]) -> str:
    content = str(result.get("content") or "").strip()
    return ("最近 7 天的真实执行记录：\n" + content) if content else "最近 7 天没有真实执行记录。"


def _format_suggestions(result: dict[str, Any], *, offset: int = 0) -> str:
    rows = result.get("suggestions") or []
    if not rows:
        return "当前没有待确认的画像建议。"
    total = int(result.get("count", len(rows)) or len(rows))
    lines = [f"当前有 {total} 条画像建议（第 {offset + 1}-{offset + len(rows)} 条）："]
    for item in rows:
        lines.append(
            f"- {item.get('suggestion_id')}｜{item.get('field_path') or '字段未记录'}"
            f"｜建议：{str(item.get('proposed_value') or '')[:180]}"
        )
    lines.append("接受或拒绝建议会先生成预览，不会直接修改画像。")
    return "\n".join(lines)


def _first_reference_page(kind: str, result: dict[str, Any], *,
                          context: dict[str, str], store: WeChatPendingStore) -> str:
    field = "applications" if kind == "applications" else "suggestions"
    id_field = "application_id" if kind == "applications" else "suggestion_id"
    rows = result.get(field) or []
    references = [str(item.get(id_field) or "") for item in rows if item.get(id_field)]
    store.remember_context(kind, references, context=context)
    page_rows = rows[:5]
    page_result = {field: page_rows, "count": len(rows), "revision": result.get("revision", "")}
    reply = (_format_applications(page_result) if kind == "applications"
             else _format_suggestions(page_result))
    if len(rows) > len(page_rows):
        token = store.create_cursor(kind, references, len(page_rows), context=context)
        reply += f"\n\n下一页：回复 下一页 {token}"
    return reply


def _format_evidence(result: dict[str, Any]) -> str:
    execution = result.get("execution") or {}
    if execution.get("deterministic_answer"):
        return str(execution["deterministic_answer"])
    evidence = execution.get("evidence") or []
    if not evidence:
        return ""
    lines = ["本地真实数据中的相关记录："]
    for item in evidence[:5]:
        lines.append(f"\n来源：{item.get('source_id') or '本地记录'}")
        lines.append(str(item.get("text") or "")[:900])
    if execution.get("freshness"):
        lines.append(f"\n截至：{execution['freshness']}")
    return "\n".join(lines)


def _response(*, status: str, handled: bool, reply_text: str = "",
              capability_id: str = "", privacy: str = "personal",
              model_exposure: str = "none", requires_confirmation: bool = False,
              pending_action: dict[str, Any] | None = None,
              freshness: dict[str, Any] | None = None, trace_id: str = "",
              reply_parts: list[str] | None = None,
              model_usage: dict[str, Any] | None = None,
              data_version: str = "", routes: list[dict[str, Any]] | None = None,
              sources: list[str] | None = None) -> dict[str, Any]:
    if reply_parts is None:
        from query_service import split_reply_parts
        reply_parts = split_reply_parts(reply_text, MAX_REPLY_CHARS)
    return {
        "schema_version": RESPONSE_SCHEMA, "status": status, "handled": handled,
        "reply_text": reply_text, "reply_parts": reply_parts,
        "route": {"capability_id": capability_id,
        "mode": "openclaw_fallback" if not handled else "local"},
        "routes": routes or [], "sources": sources or [],
        "requires_confirmation": requires_confirmation,
        "pending_action": pending_action,
        "freshness": freshness or {},
        "privacy": {"classification": privacy, "model_exposure": model_exposure},
        "model_usage": model_usage or {"offerclaw_calls": 0, "openclaw_calls": 0},
        "data_version": data_version,
        "trace_id": trace_id,
    }


def _audit(response: dict[str, Any], request: dict[str, Any], started: float) -> None:
    path = Path(os.environ.get("OFFERCLAW_WECHAT_AUDIT_PATH") or
                Path(__file__).resolve().parent / ".offerclaw" / "wechat_audit.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    record = {
        "at": now.isoformat(timespec="seconds"),
        "trace_id": response.get("trace_id"),
        "message_hash": _hash(str(request.get("message_id") or ""))[:16],
        "conversation_hash": _hash(str(request.get("conversation_id") or ""))[:16],
        "capability_id": (response.get("route") or {}).get("capability_id"),
        "status": response.get("status"), "handled": response.get("handled"),
        "model_exposure": (response.get("privacy") or {}).get("model_exposure"),
        "offerclaw_model_calls": (response.get("model_usage") or {}).get("offerclaw_calls", 0),
        "openclaw_model_calls": (response.get("model_usage") or {}).get("openclaw_calls", 0),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    with file_lock(str(path)):
        retained: list[str] = []
        cutoff = now.astimezone(dt.timezone.utc) - dt.timedelta(days=30)
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines()[-20_000:]:
                    value = json.loads(line)
                    stamp = dt.datetime.fromisoformat(str(value.get("at") or ""))
                    if stamp.astimezone(dt.timezone.utc) >= cutoff:
                        retained.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            except (OSError, ValueError, json.JSONDecodeError):
                retained = []
        retained.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        atomic_write_text(str(path), "\n".join(retained) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _stage_bridge_action(store: WeChatPendingStore, request: dict[str, Any],
                         action_type: str, bridge_result: dict[str, Any]) -> dict[str, Any]:
    context = _context(request)
    idempotency = "|".join((context["channel"], context["account_id"],
                             context["sender_id"], context["message_id"], action_type))
    return store.stage(
        action_type, bridge_result.get("payload") or {}, bridge_result.get("preview") or {},
        context=context, idempotency_key=idempotency,
        base_revision=bridge_result.get("base_revision") or "",
        base_hash=bridge_result.get("base_hash") or "",
    )


def _confirm(store: WeChatPendingStore, request: dict[str, Any], action_id: str,
             decision: str,
             bridge: Callable[[str, dict[str, Any]], dict[str, Any]]) -> tuple[str, str]:
    context = _context(request)
    row = store.get(action_id)
    if not row:
        return "error", "待确认操作不存在。"
    store.assert_context(row, context)
    if row["status"] in {"confirmed", "rejected"}:
        return "ok", f"该操作已经是 {row['status']} 状态，没有重复执行。"
    if decision == "reject":
        store.transition(action_id, "pending", "rejected", context=context)
        return "ok", "已拒绝该操作，真实数据没有变化。"
    if not store.transition(action_id, "pending", "executing", context=context):
        return "error", "操作状态已经变化，请重新查询。"
    payload = dict(row["payload"])
    payload.update({"base_hash": row.get("base_hash"),
                    "operation_id": f"wechat:{action_id}"})
    operations = {
        "bridge.application_update": "application.commit",
        "bridge.daily_log": "daily.commit",
        "bridge.suggestion": "suggestions.commit",
    }
    try:
        operation = operations[row["action_type"]]
        result = bridge(operation, payload)
        if result.get("status") in {"conflict", "error"}:
            store.transition(action_id, "executing", "failed", result=result,
                             error=result.get("error", ""), context=context)
            return "error", str(result.get("error") or "提交失败")
        store.transition(action_id, "executing", "confirmed", result=result, context=context)
        try:
            from wechat_index import sync_real_index
            sync_result = sync_real_index(bridge, max_changed=4)
        except Exception as exc:
            sync_result = {"status": "partial", "error": str(exc)[:160]}
        suffix = "" if sync_result.get("status") == "ok" else "；派生索引稍后可安全重试"
        return "ok", f"操作已确认并写入真实数据{suffix}。"
    except Exception as exc:
        store.transition(action_id, "executing", "failed", error=str(exc), context=context)
        return "error", f"提交失败，未把失败当成成功：{exc}"


def _route_request(request: dict[str, Any], bridge: Callable[[str, dict[str, Any]], dict[str, Any]],
                   store: WeChatPendingStore) -> dict[str, Any]:
    text = str(request.get("text") or "").strip()
    context = _context(request)
    context_key = "|".join(context[name] for name in ("channel", "account_id", "sender_id", "conversation_id"))
    trace_id = "wxtrace_" + _hash(context["message_id"] + context_key)[:16]
    privacy = classify_privacy(text, request.get("media"))

    media = request.get("media") or []
    if media:
        purpose = _attachment_purpose(text)
        staged = []
        for index, item in enumerate(media):
            path = _validated_media_path(item)
            staged.append(stage_attachment(
                str(path), purpose, store=store, context=context,
                idempotency_key="|".join((
                    context["channel"], context["account_id"], context["sender_id"],
                    context["message_id"], "attachment", str(index),
                )),
                source_only=True,
            ))
        lines = ["附件已在本机隔离，仅作为临时只读来源："]
        for row in staged:
            preview = row.get("preview") or {}
            lines.append(
                f"- {preview.get('filename') or '附件'}｜{preview.get('chars') or 0} 字"
                f"｜ID {row.get('action_id')}"
            )
        lines.append("未写入真实数据、未建立正式索引、未发送给模型；隔离副本将在 24 小时后清理。")
        return _response(
            status="ok", handled=True, reply_text=_page("\n".join(lines), context_key),
            capability_id="attachment.quarantine", privacy="sensitive",
            model_exposure="none", trace_id=trace_id,
        )

    next_page = _next_page(text, context, bridge, store)
    if next_page is not None:
        return _response(status="ok", handled=True, reply_text=next_page,
                         capability_id="pagination.next", privacy="sensitive", trace_id=trace_id)

    decision = re.fullmatch(r"\s*(确认|拒绝)\s+(wxact_[A-Za-z0-9_-]+)\s*", text)
    if decision:
        status, reply = _confirm(
            store, request, decision.group(2),
            "confirm" if decision.group(1) == "确认" else "reject", bridge,
        )
        return _response(status=status, handled=True, reply_text=reply,
                         capability_id="pending.confirm", privacy="sensitive", trace_id=trace_id)

    if re.search(r"待确认|待审批|pending", text, re.I):
        rows = [row for row in store.list("pending", context=context) if not row["is_expired"]]
        reply = "当前没有待确认操作。" if not rows else "当前待确认操作：\n" + "\n".join(
            f"- {row['action_id']}｜{row['action_type']}｜到期 {row['expires_at']}"
            for row in rows[:10]
        )
        return _response(status="ok", handled=True, reply_text=reply,
                         capability_id="pending.list", privacy="sensitive", trace_id=trace_id)

    automation = str(request.get("automation_kind") or "") if request.get("trigger") == "cron" else ""
    if automation == "morning":
        reply = _format_today(bridge("today.get", {}))
        return _response(status="ok", handled=True, reply_text=_page(reply, context_key),
                         capability_id="automation.morning", privacy="sensitive", trace_id=trace_id)
    if automation == "evening":
        plan = _format_plan(bridge("plan.get", {}))
        daily = bridge("daily.get", {})
        has_log = bool(str(daily.get("content") or "").strip())
        reply = (
            "今天已有真实留痕，可按需补充。" if has_log else
            "今天还没有可读取的真实留痕，请按实际完成情况记录，不会代写完成事实。"
        ) + "\n\n" + plan
        return _response(status="ok", handled=True, reply_text=_page(reply, context_key),
                         capability_id="automation.evening", privacy="sensitive", trace_id=trace_id)
    if automation == "weekly":
        plan = _format_plan(bridge("plan.get", {}))
        daily = _format_daily(bridge("daily.get", {}))
        suggestions = _format_suggestions(bridge("suggestions.list", {"status": "pending"}))
        reply = (
            "本周本地闭环摘要（零模型调用）：\n\n" + plan + "\n\n" + daily + "\n\n" + suggestions
            + "\n\n当前真实数据模式不会自动运行 grow、生成复盘或重排计划。"
        )
        return _response(status="ok", handled=True, reply_text=_page(reply, context_key),
                         capability_id="automation.weekly", privacy="sensitive", trace_id=trace_id)

    if re.search(r"(?:记录|添加|补记).{0,12}(?:学习|留痕|日志|今天)", text):
        content = re.sub(r"^(?:请|帮我)?(?:记录|添加|补记)(?:一下)?", "", text).strip(" ：:")
        preview = bridge("daily.preview", {"content": content})
        row = _stage_bridge_action(store, request, "bridge.daily_log", preview)
        value = preview.get("preview") or {}
        reply = (
            f"准备记录以下真实留痕：\n- 主线：{value.get('tag') or '未填写'}"
            f"\n- 完成：{'；'.join(value.get('done') or []) or '未填写'}"
            f"\n- 未完成：{'；'.join(value.get('todo') or []) or '未填写'}"
            f"\n- 笔记：{value.get('notes') or '未填写'}"
            f"\n\n确认写入请回复：确认 {row['action_id']}"
        )
        return _response(status="ok", handled=True, reply_text=reply,
                         capability_id="daily_log.create", privacy="sensitive",
                         requires_confirmation=True, pending_action={"action_id": row["action_id"]},
                         trace_id=trace_id)

    app_update = re.search(
        r"(?:更新|修改).{0,8}(app_[A-Za-z0-9_-]+).{0,12}(?:状态(?:为|到)?\s*)?"
        r"(preparing|evaluated|applied|waiting|interviewing|offered|rejected|abandoned|"
        r"准备投递|已评估|已投递|等待反馈|面试中|已Offer|已 Offer|已拒绝|主动放弃)", text, re.I,
    )
    if app_update:
        labels = {"准备投递": "preparing", "已评估": "evaluated", "已投递": "applied",
                  "等待反馈": "waiting", "面试中": "interviewing", "已offer": "offered",
                  "已 offer": "offered", "已拒绝": "rejected", "主动放弃": "abandoned"}
        raw_status = app_update.group(2)
        status_code = labels.get(raw_status.lower(), raw_status.lower())
        preview = bridge("application.preview", {
            "application_id": app_update.group(1), "status_code": status_code,
        })
        row = _stage_bridge_action(store, request, "bridge.application_update", preview)
        value = preview.get("preview") or {}
        reply = (
            f"准备修改投递：{value.get('company')}｜{value.get('position')}"
            f"\n- 修改前：{(value.get('before') or {}).get('status')}"
            f"\n- 修改后：{(value.get('after') or {}).get('status')}"
            f"\n\n确认写入请回复：确认 {row['action_id']}"
        )
        return _response(status="ok", handled=True, reply_text=reply,
                         capability_id="application.update_status", privacy="sensitive",
                         requires_confirmation=True, pending_action={"action_id": row["action_id"]},
                         trace_id=trace_id)

    modified_suggestion = re.fullmatch(
        r"\s*(?:请)?(?:修改后接受|修改并接受)\s*(?:画像)?建议\s+"
        r"([A-Za-z0-9_-]+)\s+(?:值|为|改为)\s+(.+?)\s*", text, re.S,
    )
    suggestion_action = modified_suggestion or re.search(
        r"(?:请)?(接受|拒绝)\s*(?:画像)?建议\s*([A-Za-z0-9_-]+)", text,
    )
    if suggestion_action:
        if modified_suggestion:
            suggestion_id = modified_suggestion.group(1)
            try:
                modified_value = json.loads(modified_suggestion.group(2))
            except json.JSONDecodeError as exc:
                raise ValueError("修改值必须是合法 JSON，例如：\"Python\" 或 [\"Python\",\"RAG\"]") from exc
            decision_name = "modified"
        else:
            suggestion_id = suggestion_action.group(2)
            modified_value = None
            decision_name = "accepted" if suggestion_action.group(1) == "接受" else "rejected"
        shown = bridge("suggestions.show", {"suggestion_id": suggestion_id})
        suggestion = shown.get("suggestion") or {}
        bridge_preview = {
            "payload": {"suggestion_id": suggestion_id, "decision": decision_name,
                        "base_revision": suggestion.get("base_revision"),
                        "modified_value": modified_value},
            "preview": {"field_path": suggestion.get("field_path"),
                        "current_value": suggestion.get("current_value"),
                        "proposed_value": (modified_value if decision_name == "modified"
                                           else suggestion.get("proposed_value")),
                        "decision": decision_name},
            "base_revision": suggestion.get("base_revision"),
        }
        row = _stage_bridge_action(store, request, "bridge.suggestion", bridge_preview)
        reply = (
            f"画像建议决定预览：{suggestion_id}\n- 字段：{suggestion.get('field_path')}"
            f"\n- 当前值：{suggestion.get('current_value')}"
            f"\n- 建议值：{bridge_preview['preview'].get('proposed_value')}"
            f"\n- 决定：{decision_name}\n\n确认执行请回复：确认 {row['action_id']}"
        )
        return _response(status="ok", handled=True, reply_text=reply,
                         capability_id=("profile.reject_suggestion" if decision_name == "rejected"
                                        else "profile.approve_suggestion"),
                         privacy="sensitive", requires_confirmation=True,
                         pending_action={"action_id": row["action_id"]}, trace_id=trace_id)

    action_match = match_action_capability(text)
    if action_match:
        capability_id = action_match.capability_id
        policy = ACTION_POLICIES[capability_id]
        capability = ACTION_CAPABILITY_REGISTRY[capability_id]
        if policy.mode == "model_blocked":
            reply = (
                f"已识别为 {capability_id}，但该功能需要使用真实个人资料生成内容。"
                "当前真实数据模式禁止把这些资料发送给远程模型；请等待 HTTPS 端点或本地模型启用。"
            )
        elif policy.mode == "unavailable":
            reply = (
                f"已识别为 {capability_id}，但此操作当前明确不支持。"
                "没有创建待确认操作，也没有修改任何真实数据。\n\n"
                + "\n".join(capability.steps)
            )
        elif policy.mode == "preview_confirm":
            required = "、".join(capability.required_inputs)
            reply = (
                f"已识别为 {capability_id}，但当前消息缺少生成安全预览所需的唯一参数：{required}。"
                "请补充明确 ID 和目标值；在返回 action_id 且你再次确认前，不会写入真实数据。"
            )
        else:
            reply = action_guidance([capability_id])
        return _response(
            status="ok" if policy.mode == "safe_guidance" else "error",
            handled=True, reply_text=_page(reply, context_key),
            capability_id=capability_id, privacy=policy.privacy,
            model_exposure="none", trace_id=trace_id,
        )

    model_action = re.search(
        r"(?:生成|重排|重做|改写|帮我写|规划).{0,12}"
        r"(?:画像分析|成长分析|个人分析|复盘)", text,
    )
    if model_action:
        return _response(
            status="error", handled=True,
            reply_text="该功能需要把真实个人资料交给生成模型。当前真实数据模式已禁用这类模型调用；请先配置 HTTPS 端点或本地模型。",
            capability_id="model_sensitive_generation", privacy="sensitive", trace_id=trace_id,
        )

    if _SECRET_RE.search(text):
        return _response(
            status="error", handled=True,
            reply_text="消息中可能包含密钥、联系方式或本机路径，本次未进入智能问答，也未发送给任何模型。",
            capability_id="privacy.secret_rejected", privacy="sensitive",
            model_exposure="none", trace_id=trace_id,
        )

    query_payload = {
        "question": text,
        "conversation_id": _opaque_identity(context["conversation_id"], store, "conv"),
        "message_id": _opaque_identity(context["message_id"], store, "msg"),
        "operation_id": _opaque_identity(
            "|".join((context["account_id"], context["sender_id"], context["message_id"])),
            store, "op",
        ),
        "top_k": 5,
    }
    result = bridge("query.answer", query_payload)
    routes = [item for item in result.get("routes") or [] if isinstance(item, dict)]
    capability = next((
        f"{item.get('source')}.{item.get('operation')}"
        for item in routes if item.get("source") and item.get("operation")
    ), "query.answer")
    reply = str(result.get("answer") or "").strip()
    usage = result.get("model_usage") or {"offerclaw_calls": 0, "openclaw_calls": 0}
    result_status = str(result.get("status") or "error")
    if not reply or result_status not in {"ok", "partial"}:
        if privacy == "public" and os.environ.get("OFFERCLAW_PUBLIC_LLM_FALLBACK", "1") == "1":
            return _response(
                status="fallback", handled=False, capability_id=capability,
                privacy="public", model_exposure="openclaw", trace_id=trace_id,
                model_usage=usage,
            )
        return _response(
            status="error", handled=True,
            reply_text="OfferClaw 智能查询没有得到可安全展示的结果。本次没有转交给 OpenClaw 模型。",
            capability_id=capability, privacy=privacy,
            model_exposure="offerclaw" if usage.get("offerclaw_calls") else "none",
            trace_id=trace_id, model_usage=usage,
        )
    return _response(
        status=result_status, handled=True, reply_text=reply,
        reply_parts=[str(item) for item in result.get("reply_parts") or [] if str(item)],
        capability_id=capability, privacy=privacy,
        model_exposure="offerclaw" if usage.get("offerclaw_calls") else "none",
        freshness={"authoritative": "windows_live", "as_of": result.get("freshness")},
        trace_id=str(result.get("trace_id") or trace_id), model_usage=usage,
        data_version=str(result.get("data_version") or ""), routes=routes,
        sources=[str(item) for item in result.get("sources") or [] if str(item)],
    )


def dispatch(request: dict[str, Any], *, bridge: Callable[[str, dict[str, Any]], dict[str, Any]] = call_bridge,
             store: WeChatPendingStore | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    _validate_request(request)
    context = _context(request)
    repository = store or WeChatPendingStore()
    repository.cleanup()
    with _WORKERS, _session_lock(context["conversation_id"]), _cross_process_execution(repository, context):
        try:
            response = _route_request(request, bridge, repository)
        except PermissionError:
            raise
        except Exception as exc:
            privacy = classify_privacy(str(request.get("text") or ""), request.get("media"))
            trace_id = "wxtrace_" + _hash(context["message_id"])[:16]
            if privacy == "public" and os.environ.get("OFFERCLAW_PUBLIC_LLM_FALLBACK", "1") == "1":
                response = _response(
                    status="fallback", handled=False,
                    capability_id="dispatcher.public_fallback", privacy="public",
                    model_exposure="openclaw", trace_id=trace_id,
                )
            else:
                response = _response(
                    status="error", handled=True,
                    reply_text=f"OfferClaw 本地处理失败：{str(exc)[:240]}。本次没有转交给模型。",
                    capability_id="dispatcher.error", privacy=privacy,
                    trace_id=trace_id,
                )
        _audit(response, request, started)
        return response


def main() -> int:
    args = sys.argv[1:]
    if args not in ([], ["--reply-text"]):
        raise ValueError("wechat dispatch 参数不受支持")
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("wechat dispatch request 过大")
    request = json.loads(raw.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("wechat dispatch request 必须是 JSON object")
    response = dispatch(request)
    if args == ["--reply-text"]:
        privacy = response.get("privacy") or {}
        if (request.get("trigger") != "cron" or response.get("handled") is not True
                or privacy.get("model_exposure") != "none"):
            raise PermissionError("纯文本输出只允许已本地处理的定时任务")
        print(str(response.get("reply_text") or "OfferClaw 定时任务没有返回内容。"))
    else:
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({
            "schema_version": RESPONSE_SCHEMA, "status": "error", "handled": True,
            "reply_text": f"OfferClaw 请求被拒绝：{str(exc)[:240]}",
            "privacy": {"classification": "sensitive", "model_exposure": "none"},
        }, ensure_ascii=False, separators=(",", ":")))
        raise SystemExit(1)
