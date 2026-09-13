# -*- coding: utf-8 -*-
"""Reviewed gateway adapter for OfferClaw's WeChat channel.

The gateway may receive files and natural-language requests, but it does not
own business data.  This module stages user input, returns inspectable previews
and calls the existing OfferClaw services only after an explicit confirmation.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / ".offerclaw" / "wechat_gateway.sqlite3"
DEFAULT_CONFIRM_TTL_SECONDS = 15 * 60
DEFAULT_ATTACHMENT_TTL_SECONDS = 24 * 60 * 60
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
DOCUMENT_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}
SOURCE_PURPOSES = {"jd", "resume", "project"}
MUTATION_PURPOSES = {"knowledge", "daily_log"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


def _digest_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _expires_in(seconds: int) -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)).isoformat(
        timespec="seconds"
    )


def _expired(value: str) -> bool:
    if not value:
        return False
    try:
        return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc) <= dt.datetime.now(
            dt.timezone.utc
        )
    except ValueError:
        return True


def request_context(**values: Any) -> dict[str, str]:
    """Return the normalized identity fields bound to a WeChat action."""
    return {
        name: str(values.get(name) or "").strip()
        for name in ("channel", "account_id", "sender_id", "conversation_id", "message_id")
    }


class WeChatPendingStore:
    """Small durable store for gateway previews and explicit decisions."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        configured = path or os.environ.get("OFFERCLAW_WECHAT_DB_PATH") or DEFAULT_DB_PATH
        self.path = Path(configured)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pending_actions (
                    action_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('pending','executing','confirmed','rejected','failed')),
                    payload_json TEXT NOT NULL,
                    preview_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS pending_actions_status
                    ON pending_actions(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS wechat_session_refs (
                    scope_hash TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    references_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wechat_reply_cursors (
                    cursor_id TEXT PRIMARY KEY,
                    scope_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    references_json TEXT NOT NULL,
                    offset_value INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS wechat_reply_cursors_expiry
                    ON wechat_reply_cursors(expires_at);
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(pending_actions)")}
            migrations = {
                "channel": "TEXT NOT NULL DEFAULT ''",
                "account_id": "TEXT NOT NULL DEFAULT ''",
                "sender_id": "TEXT NOT NULL DEFAULT ''",
                "conversation_id": "TEXT NOT NULL DEFAULT ''",
                "message_id": "TEXT NOT NULL DEFAULT ''",
                "idempotency_key": "TEXT NOT NULL DEFAULT ''",
                "expires_at": "TEXT NOT NULL DEFAULT ''",
                "base_revision": "TEXT NOT NULL DEFAULT ''",
                "base_hash": "TEXT NOT NULL DEFAULT ''",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE pending_actions ADD COLUMN {name} {declaration}")
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS pending_actions_idempotency
                   ON pending_actions(idempotency_key) WHERE idempotency_key <> ''"""
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=8)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=8000")
        conn.execute("PRAGMA secure_delete=ON")
        conn.execute("PRAGMA journal_mode=DELETE")
        return conn

    def stage(self, action_type: str, payload: dict[str, Any],
              preview: dict[str, Any], *, content_hash: str = "",
              context: dict[str, Any] | None = None, idempotency_key: str = "",
              ttl_seconds: int = DEFAULT_CONFIRM_TTL_SECONDS,
              base_revision: str | int = "", base_hash: str = "") -> dict[str, Any]:
        identity = request_context(**(context or {}))
        idempotency_key = str(idempotency_key or "").strip()
        action_id = (
            "wxact_" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:24]
            if idempotency_key else _id("wxact")
        )
        stamp = _now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT action_id FROM pending_actions WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone() if idempotency_key else None
            if not existing:
                conn.execute(
                    """INSERT INTO pending_actions(
                       action_id,action_type,status,payload_json,preview_json,content_hash,
                       created_at,updated_at,result_json,error,channel,account_id,sender_id,
                       conversation_id,message_id,idempotency_key,expires_at,base_revision,base_hash
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (action_id, action_type, "pending",
                     json.dumps(payload, ensure_ascii=False),
                     json.dumps(preview, ensure_ascii=False), content_hash,
                     stamp, stamp, "{}", "", identity["channel"], identity["account_id"],
                     identity["sender_id"], identity["conversation_id"],
                     identity["message_id"], idempotency_key,
                     _expires_in(max(1, int(ttl_seconds))), str(base_revision or ""),
                     str(base_hash or "")),
                )
            else:
                action_id = str(existing["action_id"])
        return self.get(action_id) or {}

    def get(self, action_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_actions WHERE action_id=?", (action_id,)
            ).fetchone()
        return self._row(row) if row else None

    def list(self, status: str = "pending", limit: int = 20, *,
             context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        clauses: list[str] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        identity = request_context(**(context or {}))
        for name in ("channel", "account_id", "sender_id", "conversation_id"):
            if identity[name]:
                clauses.append(f"{name}=?")
                params.append(identity[name])
        params.append(max(1, min(int(limit), 100)))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pending_actions{where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row(row) for row in rows]

    def assert_context(self, row: dict[str, Any], context: dict[str, Any] | None) -> None:
        identity = request_context(**(context or {}))
        for name in ("channel", "account_id", "sender_id", "conversation_id"):
            expected = str(row.get(name) or "")
            if expected and expected != identity[name]:
                raise PermissionError("待确认操作不属于当前微信账号或会话")
        if row.get("is_expired"):
            raise TimeoutError("待确认操作已过期，请重新发起并检查预览")

    def transition(self, action_id: str, expected: str, status: str, *,
                   result: dict[str, Any] | None = None, error: str = "",
                   context: dict[str, Any] | None = None) -> bool:
        row = self.get(action_id)
        if not row:
            return False
        self.assert_context(row, context)
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE pending_actions SET status=?,updated_at=?,result_json=?,error=?
                   WHERE action_id=? AND status=?""",
                (status, _now(), json.dumps(result or {}, ensure_ascii=False),
                 str(error or "")[:1000], action_id, expected),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _scope_hash(context: dict[str, Any] | None) -> str:
        identity = request_context(**(context or {}))
        required = ("channel", "account_id", "sender_id", "conversation_id")
        if any(not identity[name] for name in required):
            raise PermissionError("会话上下文缺少可信身份字段")
        raw = "|".join(identity[name] for name in required)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def remember_context(self, kind: str, references: list[str], *,
                         context: dict[str, Any],
                         ttl_seconds: int = DEFAULT_CONFIRM_TTL_SECONDS) -> None:
        clean = [str(item) for item in references if str(item)][:100]
        stamp = _now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO wechat_session_refs(scope_hash,kind,references_json,updated_at,expires_at)
                   VALUES(?,?,?,?,?) ON CONFLICT(scope_hash) DO UPDATE SET
                   kind=excluded.kind,references_json=excluded.references_json,
                   updated_at=excluded.updated_at,expires_at=excluded.expires_at""",
                (self._scope_hash(context), str(kind), json.dumps(clean, ensure_ascii=False),
                 stamp, _expires_in(max(1, int(ttl_seconds)))),
            )

    def recent_context(self, *, context: dict[str, Any]) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM wechat_session_refs WHERE scope_hash=?",
                (self._scope_hash(context),),
            ).fetchone()
        if not row or _expired(str(row["expires_at"])):
            return None
        return {
            "kind": row["kind"], "references": json.loads(row["references_json"]),
            "updated_at": row["updated_at"], "expires_at": row["expires_at"],
        }

    def create_cursor(self, kind: str, references: list[str], offset: int, *,
                      context: dict[str, Any],
                      ttl_seconds: int = DEFAULT_CONFIRM_TTL_SECONDS) -> str:
        cursor_id = "wxpage_" + uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO wechat_reply_cursors(
                   cursor_id,scope_hash,kind,references_json,offset_value,created_at,expires_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (cursor_id, self._scope_hash(context), str(kind),
                 json.dumps([str(item) for item in references if str(item)][:100], ensure_ascii=False),
                 max(0, int(offset)), _now(), _expires_in(max(1, int(ttl_seconds)))),
            )
        return cursor_id

    def cursor(self, cursor_id: str, *, context: dict[str, Any]) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM wechat_reply_cursors WHERE cursor_id=? AND scope_hash=?",
                (str(cursor_id), self._scope_hash(context)),
            ).fetchone()
        if not row or _expired(str(row["expires_at"])):
            return None
        return {
            "cursor_id": row["cursor_id"], "kind": row["kind"],
            "references": json.loads(row["references_json"]),
            "offset": int(row["offset_value"]), "expires_at": row["expires_at"],
        }

    def advance_cursor(self, cursor_id: str, offset: int | None) -> None:
        with self._connect() as conn:
            if offset is None:
                conn.execute("DELETE FROM wechat_reply_cursors WHERE cursor_id=?", (cursor_id,))
            else:
                conn.execute(
                    "UPDATE wechat_reply_cursors SET offset_value=? WHERE cursor_id=?",
                    (max(0, int(offset)), cursor_id),
                )

    def cleanup(self) -> int:
        """Remove expired attachment files and prune action metadata after 30 days."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT action_id,payload_json FROM pending_actions WHERE expires_at <> '' AND expires_at <= ?",
                (_now(),),
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"])
                    quarantine = Path(str(payload.get("quarantine_path") or ""))
                    if quarantine.is_file() and quarantine.parent == self.path.parent / "wechat_attachments":
                        quarantine.unlink()
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat(
                timespec="seconds"
            )
            pruned = conn.execute(
                "DELETE FROM pending_actions WHERE created_at <= ?", (cutoff,)
            ).rowcount
            expired_refs = conn.execute(
                "DELETE FROM wechat_session_refs WHERE expires_at <= ?", (_now(),)
            ).rowcount
            expired_cursors = conn.execute(
                "DELETE FROM wechat_reply_cursors WHERE expires_at <= ?", (_now(),)
            ).rowcount
        return (len(rows) + max(0, int(pruned)) + max(0, int(expired_refs))
                + max(0, int(expired_cursors)))

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "action_id": row["action_id"], "action_type": row["action_type"],
            "status": row["status"], "payload": json.loads(row["payload_json"]),
            "preview": json.loads(row["preview_json"]),
            "content_hash": row["content_hash"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], "result": json.loads(row["result_json"]),
            "error": row["error"],
            "channel": row["channel"], "account_id": row["account_id"],
            "sender_id": row["sender_id"], "conversation_id": row["conversation_id"],
            "message_id": row["message_id"], "idempotency_key": row["idempotency_key"],
            "expires_at": row["expires_at"], "base_revision": row["base_revision"],
            "base_hash": row["base_hash"], "is_expired": _expired(row["expires_at"]),
        }


def _read_attachment(path: str) -> tuple[str, bytes, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("附件文件不存在或不是普通文件")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError("附件为空")
    if size > MAX_ATTACHMENT_BYTES:
        raise ValueError("附件超过 20 MB，未进入 OfferClaw")
    ext = source.suffix.lower()
    if ext not in DOCUMENT_EXTENSIONS:
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise ValueError("当前不支持图片 OCR；请发送可抽取文本的 PDF/DOCX/MD/TXT")
        if ext in {".sqlite", ".sqlite3", ".db", ".faiss", ".index"}:
            raise ValueError("不导入外部向量数据库文件；请上传原始文档，经确认后由 OfferClaw 重建索引")
        raise ValueError("仅支持 .md/.txt/.markdown/.pdf/.docx")
    raw = source.read_bytes()
    return source.name, raw, ext


def _extract_text(name: str, raw: bytes, ext: str) -> str:
    if ext in TEXT_EXTENSIONS:
        text = raw.decode("utf-8", errors="replace")
    else:
        from knowledge_crawler import extract_text_for_kb

        text = extract_text_for_kb(name, raw)
    text = text.strip()
    if len(text) < 30:
        raise ValueError("附件没有可用正文（至少 30 字）")
    return text


def stage_attachment(path: str, purpose: str, *, title: str = "",
                     store: WeChatPendingStore | None = None,
                     context: dict[str, Any] | None = None,
                     idempotency_key: str = "", source_only: bool = False) -> dict[str, Any]:
    """Parse a received attachment and create a non-authoritative preview."""
    purpose = str(purpose or "").strip()
    if purpose not in SOURCE_PURPOSES | MUTATION_PURPOSES:
        raise ValueError("purpose 必须是 jd/resume/project/knowledge/daily_log")
    name, raw, ext = _read_attachment(path)
    text = _extract_text(name, raw, ext)
    digest = hashlib.sha256(raw).hexdigest()
    action_type = "attachment_source" if source_only or purpose in SOURCE_PURPOSES else purpose
    repository = store or WeChatPendingStore()
    quarantine_dir = repository.path.parent / "wechat_attachments"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(quarantine_dir, 0o700)
    except OSError:
        pass
    quarantine_path = quarantine_dir / f"{uuid.uuid4().hex}.txt"
    quarantine_path.write_bytes(text.encode("utf-8"))
    try:
        os.chmod(quarantine_path, 0o600)
    except OSError:
        pass
    preview: dict[str, Any] = {
        "filename": name, "purpose": purpose, "title": title or Path(name).stem,
        "chars": len(text), "sha256": digest,
        "writes_business_data": False,
    }
    if purpose == "jd":
        from jd_parser import analyze_jd

        preview["jd_analysis"] = analyze_jd(text).model_dump(mode="json")
    payload = {"purpose": purpose, "filename": name,
               "title": title or Path(name).stem,
               "quarantine_path": str(quarantine_path),
               "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    staged = repository.stage(
        action_type, payload, preview, content_hash=digest, context=context,
        idempotency_key=idempotency_key, ttl_seconds=DEFAULT_ATTACHMENT_TTL_SECONDS,
    )
    return {"status": "pending",
            "requires_confirmation": purpose in MUTATION_PURPOSES and not source_only,
            **{key: staged[key] for key in ("action_id", "action_type", "preview")}}


def stage_daily_log(content: str, *, store: WeChatPendingStore | None = None,
                    context: dict[str, Any] | None = None,
                    idempotency_key: str = "") -> dict[str, Any]:
    from offerclaw_cli import _parse_structured_log

    tag, done, todo, notes = _parse_structured_log(content)
    if not any((tag, done, todo, notes)):
        raise ValueError("留痕内容不能为空")
    payload = {"tag": tag, "done": done, "todo": todo, "notes": notes}
    preview = {**payload, "writes": ["daily_log.md", "memory.sqlite3"]}
    daily_path = BASE_DIR / "daily_log.md"
    base_hash = hashlib.sha256(daily_path.read_bytes()).hexdigest() if daily_path.is_file() else ""
    row = (store or WeChatPendingStore()).stage(
        "daily_log", payload, preview, context=context,
        idempotency_key=idempotency_key, base_hash=base_hash,
    )
    return {"status": "pending", "requires_confirmation": True,
            "action_id": row["action_id"], "preview": preview}


def stage_application_update(application_id: str, *, status_code: str,
                             store: WeChatPendingStore | None = None,
                             context: dict[str, Any] | None = None,
                             idempotency_key: str = "") -> dict[str, Any]:
    from applications_store import get_application
    from domain_status import APPLICATION_STATUS_LABELS, application_status_code

    current = get_application(application_id)
    if not current:
        raise ValueError("找不到 application_id")
    code = application_status_code(status_code)
    label = APPLICATION_STATUS_LABELS[code]
    payload = {"application_id": application_id, "status_code": code.value,
               "status": label}
    preview = {
        "application_id": application_id,
        "company": current.get("company", ""), "position": current.get("position", ""),
        "before": {"status": current.get("status"), "status_code": current.get("status_code")},
        "after": {"status": label, "status_code": code.value},
        "writes": ["applications.md", "memory.sqlite3"],
    }
    row = (store or WeChatPendingStore()).stage(
        "application_update", payload, preview, context=context,
        idempotency_key=idempotency_key, base_hash=_digest_json(current),
        base_revision=current.get("updated_at") or current.get("recorded_at") or "",
    )
    return {"status": "pending", "requires_confirmation": True,
            "action_id": row["action_id"], "preview": preview}


def source_text(action_id: str, *, purposes: set[str] | None = None,
                store: WeChatPendingStore | None = None) -> dict[str, Any]:
    row = (store or WeChatPendingStore()).get(action_id)
    if not row or row["action_type"] != "attachment_source":
        raise ValueError("附件来源不存在")
    payload = row["payload"]
    if purposes and payload.get("purpose") not in purposes:
        raise ValueError("附件用途与当前操作不匹配")
    path = Path(str(payload.get("quarantine_path") or ""))
    if not path.is_file() or path.parent != (store or WeChatPendingStore()).path.parent / "wechat_attachments":
        raise ValueError("附件隔离文件不存在或已过期")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != payload.get("text_sha256"):
        raise ValueError("附件隔离文件完整性校验失败")
    text = raw.decode("utf-8")
    return {**payload, "text": text}


def decide_pending(action_id: str, decision: str, *,
                   store: WeChatPendingStore | None = None,
                   context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Confirm or reject a staged write. Source-only attachments are not writes."""
    repository = store or WeChatPendingStore()
    row = repository.get(action_id)
    if not row:
        raise ValueError("待审批操作不存在")
    repository.assert_context(row, context)
    if row["status"] in {"confirmed", "rejected"}:
        return {"status": row["status"], "action_id": action_id,
                "replayed": True, "result": row["result"]}
    if row["status"] != "pending":
        raise RuntimeError(f"待审批操作当前状态为 {row['status']}")
    if decision == "reject":
        repository.transition(action_id, "pending", "rejected", context=context)
        return {"status": "rejected", "action_id": action_id}
    if decision != "confirm":
        raise ValueError("decision 必须是 confirm 或 reject")
    if row["action_type"] == "attachment_source":
        raise ValueError("该附件是只读来源，不需要确认写入；请把 action_id 交给分析或生成命令")
    if row["action_type"] == "application_update" and row.get("base_hash"):
        from applications_store import get_application
        if _digest_json(get_application(row["payload"]["application_id"])) != row["base_hash"]:
            raise RuntimeError("投递记录已发生变化，请重新生成预览")
    if row["action_type"] == "daily_log" and row.get("base_hash"):
        daily_path = BASE_DIR / "daily_log.md"
        current_hash = hashlib.sha256(daily_path.read_bytes()).hexdigest() if daily_path.is_file() else ""
        if current_hash != row["base_hash"]:
            raise RuntimeError("每日留痕已发生变化，请重新生成预览")
    if not repository.transition(action_id, "pending", "executing", context=context):
        raise RuntimeError("审批状态已经变化，请刷新后重试")
    payload = row["payload"]
    try:
        if row["action_type"] == "daily_log":
            from summary_tool import append_structured_daily_log

            result = append_structured_daily_log(
                tag=payload.get("tag", ""), done=payload.get("done") or [],
                todo=payload.get("todo") or [], notes=payload.get("notes", ""),
                operation_id=f"wechat:{action_id}",
            )
        elif row["action_type"] == "application_update":
            from applications_store import patch_application

            result = patch_application(
                payload["application_id"], status=payload["status"],
                operation_id=f"wechat:{action_id}",
            )
        elif row["action_type"] == "knowledge":
            from knowledge_crawler import _score_and_save

            source = source_text(action_id, store=repository)
            result = _score_and_save(
                source["text"], url=f"(微信上传:{payload['filename']})",
                origin="微信附件上传", force_keep=True,
            )
        else:
            raise ValueError(f"不支持的待审批操作：{row['action_type']}")
        if result.get("status") not in {"ok", "duplicate"}:
            raise RuntimeError(str(result.get("error") or result))
        repository.transition(
            action_id, "executing", "confirmed", result=result, context=context,
        )
        return {"status": "confirmed", "action_id": action_id, "result": result}
    except Exception as exc:
        repository.transition(
            action_id, "executing", "failed", error=str(exc), context=context,
        )
        raise


__all__ = [
    "DEFAULT_ATTACHMENT_TTL_SECONDS", "DEFAULT_CONFIRM_TTL_SECONDS",
    "MAX_ATTACHMENT_BYTES", "WeChatPendingStore", "decide_pending", "request_context",
    "source_text", "stage_application_update", "stage_attachment", "stage_daily_log",
]
