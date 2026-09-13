# -*- coding: utf-8 -*-
"""Transactional storage for OfferClaw's personal memory.

SQLite is the source of truth.  Markdown business files remain user-readable
artifacts and JSON/JSONL files are compatibility exports only.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 6
_ORIGINAL_BASE_DIR = Path(__file__).resolve().parent / "logs" / "memory"


def _setting(name: str, default: float, cast):
    try:
        value = cast(os.environ.get(name, str(default)))
        return value if value > 0 else cast(default)
    except (TypeError, ValueError):
        return cast(default)


SOP_MIN_CASES = _setting("OFFERCLAW_SOP_MIN_CASES", 3, int)
SOP_MIN_DAYS = _setting("OFFERCLAW_SOP_MIN_DAYS", 2, int)
SOP_ACTIVATION_CONFIDENCE = _setting("OFFERCLAW_SOP_ACTIVATION_CONFIDENCE", .75, float)


def effective_base_dir(configured: str | os.PathLike[str] | None = None) -> Path:
    if configured:
        return Path(configured)
    environment_dir = os.environ.get("OFFERCLAW_MEMORY_DIR", "").strip()
    if environment_dir:
        return Path(environment_dir)
    # Production-like tests used to write thousands of synthetic CareerFlow
    # events into the user's real memory.  Keep implicit test stores isolated.
    test_id = os.environ.get("PYTEST_CURRENT_TEST", "").split(" (", 1)[0].strip()
    if test_id:
        suffix = hashlib.sha256(test_id.encode("utf-8")).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f"offerclaw-test-memory-{os.getpid()}-{suffix}"
    return _ORIGINAL_BASE_DIR


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


class _ClosingConnection(sqlite3.Connection):
    """SQLite context manager that also releases its Windows file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class MemoryStore:
    """Short-transaction SQLite repository with bounded busy retries."""

    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        self.base_dir = effective_base_dir(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "memory.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=8, isolation_level=None,
                               factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=8000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            # SQLite versions before the WAL-reset fix stay on the rollback
            # journal.  The DB is local and transactions are deliberately short.
            version = tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:3])
            journal = "WAL" if version >= (3, 51, 3) else "DELETE"
            conn.execute(f"PRAGMA journal_mode={journal}")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS goal_contexts (
                    context_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','archived')),
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_goal
                    ON goal_contexts(status) WHERE status='active';
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    business_date TEXT,
                    actor TEXT NOT NULL CHECK(actor IN ('user','assistant','system','import')),
                    source TEXT NOT NULL,
                    traffic_origin TEXT NOT NULL,
                    operation_id TEXT UNIQUE,
                    causation_id TEXT,
                    conversation_id TEXT,
                    target_context_id TEXT,
                    entity_type TEXT,
                    entity_id TEXT,
                    payload_json TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT,
                    FOREIGN KEY(target_context_id) REFERENCES goal_contexts(context_id)
                );
                CREATE INDEX IF NOT EXISTS events_kind_seq ON events(kind, seq DESC);
                CREATE INDEX IF NOT EXISTS events_target_seq ON events(target_context_id, seq DESC);
                CREATE INDEX IF NOT EXISTS events_entity ON events(entity_type, entity_id);
                CREATE TABLE IF NOT EXISTS evidence_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    content TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    source_path TEXT,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    memory_id TEXT PRIMARY KEY,
                    memory_key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    certainty TEXT NOT NULL CHECK(certainty IN ('explicit','inferred','pending')),
                    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('active','superseded','archived','deleted')),
                    target_context_id TEXT NOT NULL DEFAULT 'global',
                    first_supported_at TEXT NOT NULL,
                    last_supported_at TEXT NOT NULL,
                    expires_at TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    UNIQUE(memory_key, target_context_id)
                );
                CREATE TABLE IF NOT EXISTS semantic_evidence (
                    memory_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    stance TEXT NOT NULL CHECK(stance IN ('support','oppose')),
                    PRIMARY KEY(memory_id, event_id, stance),
                    FOREIGN KEY(memory_id) REFERENCES semantic_memories(memory_id) ON DELETE CASCADE,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );
                CREATE TABLE IF NOT EXISTS procedural_sops (
                    sop_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    body TEXT NOT NULL,
                    trigger_json TEXT NOT NULL,
                    target_context_id TEXT NOT NULL DEFAULT 'global',
                    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('candidate','active','suspended','archived','deleted')),
                    success_weight REAL NOT NULL DEFAULT 0,
                    failure_weight REAL NOT NULL DEFAULT 0,
                    independent_cases INTEGER NOT NULL DEFAULT 0,
                    evidence_days INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_evidence_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name, target_context_id)
                );
                CREATE TABLE IF NOT EXISTS sop_evidence (
                    sop_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK(outcome IN ('suggested','adopted','success','failure')),
                    weight REAL NOT NULL DEFAULT 1,
                    business_date TEXT,
                    PRIMARY KEY(sop_id, event_id, outcome),
                    FOREIGN KEY(sop_id) REFERENCES procedural_sops(sop_id) ON DELETE CASCADE,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );
                CREATE TABLE IF NOT EXISTS runtime_state (
                    state_key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_turn_contexts (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    service_mode TEXT NOT NULL,
                    output_contracts_json TEXT NOT NULL,
                    resolved_entities_json TEXT NOT NULL,
                    capability_ids_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversation_turns_recent
                    ON conversation_turn_contexts(conversation_id, completed_at DESC);
                CREATE TABLE IF NOT EXISTS operation_journal (
                    operation_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','committed','conflict','failed')),
                    target_path TEXT,
                    before_hash TEXT,
                    after_hash TEXT,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tombstones (
                    tombstone_id TEXT PRIMARY KEY,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    deleted_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    UNIQUE(object_type, object_id)
                );
                CREATE TABLE IF NOT EXISTS migration_map (
                    source_key TEXT PRIMARY KEY,
                    target_id TEXT,
                    status TEXT NOT NULL,
                    detail TEXT
                );
                CREATE TABLE IF NOT EXISTS memory_search_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_profile TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS memory_search_event
                    ON memory_search_chunks(event_id);
                CREATE INDEX IF NOT EXISTS memory_search_profile
                    ON memory_search_chunks(embedding_profile);
                CREATE TABLE IF NOT EXISTS profile_revisions (
                    revision INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision_id TEXT NOT NULL UNIQUE,
                    schema_version TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    content_md TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    actor TEXT NOT NULL CHECK(actor IN ('user','system','import')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profile_state (
                    profile_id TEXT PRIMARY KEY,
                    current_revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(current_revision) REFERENCES profile_revisions(revision)
                );
                CREATE TABLE IF NOT EXISTS profile_capabilities (
                    capability_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    parent_id TEXT,
                    node_type TEXT NOT NULL CHECK(node_type IN ('domain','capability','atomic')),
                    rubric_json TEXT NOT NULL,
                    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('provisional','active','archived')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name)
                );
                CREATE TABLE IF NOT EXISTS profile_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    source_event_id TEXT NOT NULL,
                    source_snapshot_id TEXT,
                    source_type TEXT NOT NULL,
                    source_quote TEXT NOT NULL,
                    quote_hash TEXT NOT NULL,
                    occurred_on TEXT,
                    claim TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    activity_level TEXT NOT NULL CHECK(activity_level IN ('observed','practiced','delivered')),
                    verification TEXT NOT NULL CHECK(verification IN ('source_grounded','artifact_checked','test_result','external_result','user_attested')),
                    scope TEXT NOT NULL,
                    target_context_id TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('candidate','validated','rejected','superseded','deleted')),
                    rejection_reason TEXT NOT NULL DEFAULT '',
                    origin_key TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(capability_id) REFERENCES profile_capabilities(capability_id)
                );
                CREATE INDEX IF NOT EXISTS profile_evidence_capability
                    ON profile_evidence(capability_id,status,occurred_on);
                CREATE INDEX IF NOT EXISTS profile_evidence_source
                    ON profile_evidence(source_event_id,status);
                CREATE TABLE IF NOT EXISTS profile_evidence_extractions (
                    source_event_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('completed','review_unavailable','failed')),
                    model_meta_json TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    processed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profile_audits (
                    audit_id TEXT PRIMARY KEY,
                    base_revision INTEGER NOT NULL,
                    target_context_id TEXT NOT NULL,
                    trigger_kind TEXT NOT NULL,
                    evidence_cutoff TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('generating','completed','review_unavailable','failed')),
                    model_meta_json TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(base_revision) REFERENCES profile_revisions(revision)
                );
                CREATE TABLE IF NOT EXISTS profile_audit_evidence (
                    audit_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    is_new INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(audit_id,evidence_id),
                    FOREIGN KEY(audit_id) REFERENCES profile_audits(audit_id) ON DELETE CASCADE,
                    FOREIGN KEY(evidence_id) REFERENCES profile_evidence(evidence_id)
                );
                CREATE TABLE IF NOT EXISTS profile_suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    audit_id TEXT NOT NULL,
                    base_revision INTEGER NOT NULL,
                    target_context_id TEXT NOT NULL,
                    field_path TEXT NOT NULL,
                    operation TEXT NOT NULL CHECK(operation IN ('add','replace','remove')),
                    current_value_json TEXT NOT NULL,
                    proposed_value_json TEXT NOT NULL,
                    observed_change TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    counter_evidence_ids_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    requirement_status TEXT NOT NULL CHECK(requirement_status IN ('satisfied','needs_evidence','conflict')),
                    status TEXT NOT NULL CHECK(status IN ('generating','pending','needs_evidence','accepted','modified','rejected','stale','review_unavailable')),
                    capability_id TEXT,
                    new_capability_json TEXT NOT NULL DEFAULT '{}',
                    evidence_hash TEXT NOT NULL,
                    model_meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(audit_id) REFERENCES profile_audits(audit_id),
                    FOREIGN KEY(base_revision) REFERENCES profile_revisions(revision)
                );
                CREATE INDEX IF NOT EXISTS profile_suggestions_status
                    ON profile_suggestions(status,created_at DESC);
                CREATE TABLE IF NOT EXISTS profile_suggestion_decisions (
                    decision_id TEXT PRIMARY KEY,
                    suggestion_id TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('accepted','modified','rejected')),
                    final_value_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    operation_id TEXT NOT NULL UNIQUE,
                    result_revision INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(suggestion_id) REFERENCES profile_suggestions(suggestion_id),
                    FOREIGN KEY(result_revision) REFERENCES profile_revisions(revision)
                );
                CREATE TABLE IF NOT EXISTS profile_edit_previews (
                    preview_id TEXT PRIMARY KEY,
                    base_revision INTEGER NOT NULL,
                    profile_json TEXT NOT NULL,
                    content_md TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    diff_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    committed_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(base_revision) REFERENCES profile_revisions(revision)
                );
                CREATE TABLE IF NOT EXISTS profile_migrations (
                    migration_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    revision INTEGER,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(revision) REFERENCES profile_revisions(revision)
                );
                """
            )
            self._migrate_sop_scope(conn)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            if not conn.execute("SELECT 1 FROM goal_contexts WHERE status='active'").fetchone():
                stamp = now_iso()
                conn.execute(
                    "INSERT OR IGNORE INTO goal_contexts VALUES(?,?,?,?,?,?)",
                    ("goal_default", "当前职业目标", "active", 1, stamp, stamp),
                )

    @staticmethod
    def _migrate_sop_scope(conn: sqlite3.Connection) -> None:
        """Upgrade the v1 global SOP name constraint without losing evidence."""
        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='procedural_sops'"
        ).fetchone()
        table_sql = "".join(str(sql_row[0] or "").lower().split()) if sql_row else ""
        if "unique(name,target_context_id)" in table_sql:
            return
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.executescript(
                """
                BEGIN IMMEDIATE;
                ALTER TABLE sop_evidence RENAME TO sop_evidence_v1;
                ALTER TABLE procedural_sops RENAME TO procedural_sops_v1;
                CREATE TABLE procedural_sops (
                    sop_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    body TEXT NOT NULL,
                    trigger_json TEXT NOT NULL,
                    target_context_id TEXT NOT NULL DEFAULT 'global',
                    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('candidate','active','suspended','archived','deleted')),
                    success_weight REAL NOT NULL DEFAULT 0,
                    failure_weight REAL NOT NULL DEFAULT 0,
                    independent_cases INTEGER NOT NULL DEFAULT 0,
                    evidence_days INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_evidence_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name, target_context_id)
                );
                CREATE TABLE sop_evidence (
                    sop_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK(outcome IN ('suggested','adopted','success','failure')),
                    weight REAL NOT NULL DEFAULT 1,
                    business_date TEXT,
                    PRIMARY KEY(sop_id, event_id, outcome),
                    FOREIGN KEY(sop_id) REFERENCES procedural_sops(sop_id) ON DELETE CASCADE,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );
                INSERT INTO procedural_sops SELECT * FROM procedural_sops_v1;
                INSERT INTO sop_evidence SELECT * FROM sop_evidence_v1;
                DROP TABLE sop_evidence_v1;
                DROP TABLE procedural_sops_v1;
                COMMIT;
                """
            )
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    @contextlib.contextmanager
    def transaction(self, retries: int = 4) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            for attempt in range(retries):
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == retries - 1:
                        raise
                    time.sleep(0.04 * (2**attempt))
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def active_goal_id(self) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT context_id FROM goal_contexts WHERE status='active'"
            ).fetchone()
        return str(row[0]) if row else "goal_default"

    @staticmethod
    def insert_event_in_transaction(conn: sqlite3.Connection,
                                    envelope: dict[str, Any]) -> sqlite3.Row:
        """Insert an already validated event using the caller's transaction."""
        value = dict(envelope)
        payload = dict(value.pop("payload", {}) or {})
        operation_id = value.get("operation_id")
        if operation_id:
            existing = conn.execute(
                "SELECT * FROM events WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if existing:
                return existing
        conn.execute(
            """INSERT INTO events(
                event_id,schema_version,kind,occurred_at,recorded_at,business_date,
                actor,source,traffic_origin,operation_id,causation_id,conversation_id,
                target_context_id,entity_type,entity_id,payload_json,archived)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                value["event_id"], value["schema_version"], value["kind"],
                value["occurred_at"], value["recorded_at"],
                value.get("business_date"), value["actor"], value["source"],
                value["traffic_origin"], operation_id, value.get("causation_id"),
                value.get("conversation_id"), value.get("target_context_id"),
                value.get("entity_type"), value.get("entity_id"),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                int(bool(value.get("archived"))),
            ),
        )
        return conn.execute(
            "SELECT * FROM events WHERE event_id=?", (value["event_id"],)
        ).fetchone()

    def append_event(self, envelope: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as conn:
            row = self.insert_event_in_transaction(conn, envelope)
        return self._event_row(row)

    @staticmethod
    def put_snapshot_in_transaction(conn: sqlite3.Connection, content: str, *,
                                    media_type: str = "text/plain",
                                    source_path: str = "") -> dict[str, Any]:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        snapshot_id = f"snap_{digest}"
        conn.execute(
            """INSERT OR IGNORE INTO evidence_snapshots
               (snapshot_id,content_hash,content,media_type,source_path,created_at,deleted_at)
               VALUES(?,?,?,?,?,?,NULL)""",
            (snapshot_id, digest, content, media_type, source_path or None, now_iso()),
        )
        return {"snapshot_id": snapshot_id, "content_hash": digest,
                "media_type": media_type, "source_path": source_path}

    def put_snapshot(self, content: str, *, media_type: str = "text/plain",
                     source_path: str = "") -> dict[str, Any]:
        with self.transaction() as conn:
            return self.put_snapshot_in_transaction(
                conn, content, media_type=media_type, source_path=source_path,
            )

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_snapshots WHERE snapshot_id=? AND deleted_at IS NULL",
                (snapshot_id,),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"] or "{}")
        return {
            "id": row["event_id"], "event_id": row["event_id"], "seq": row["seq"],
            "schema_version": row["schema_version"], "kind": row["kind"],
            "ts_iso": row["occurred_at"], "occurred_at": row["occurred_at"],
            "recorded_at": row["recorded_at"], "business_date": row["business_date"],
            "actor": row["actor"], "source": row["source"],
            "traffic_origin": row["traffic_origin"], "operation_id": row["operation_id"],
            "causation_id": row["causation_id"], "conversation_id": row["conversation_id"],
            "target_context_id": row["target_context_id"], "entity_type": row["entity_type"],
            "entity_id": row["entity_id"], "archived": bool(row["archived"]),
            "deleted_at": row["deleted_at"], **payload,
        }

    def get_event(self, event_id: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        sql = "SELECT * FROM events WHERE event_id=?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        with self._connect() as conn:
            row = conn.execute(sql, (event_id,)).fetchone()
        return self._event_row(row) if row else None

    def get_event_by_operation(self, operation_id: str) -> dict[str, Any] | None:
        if not operation_id:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE operation_id=?", (operation_id,)).fetchone()
        return self._event_row(row) if row else None

    def list_events(self, *, limit: int | None = None, kind: str = "",
                    include_archived: bool = True, include_deleted: bool = False,
                    target_context_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if kind:
            clauses.append("kind=?"); params.append(kind)
        if not include_archived:
            clauses.append("archived=0")
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if target_context_id:
            clauses.append("target_context_id IN (?, 'global')"); params.append(target_context_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = " ORDER BY seq DESC" if limit is not None else " ORDER BY seq ASC"
        sql = "SELECT * FROM events" + where + order
        if limit is not None:
            sql += " LIMIT ?"; params.append(max(0, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = [self._event_row(row) for row in rows]
        return out if limit is None else list(reversed(out))

    def set_runtime(self, key: str, value: Any) -> None:
        stamp = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO runtime_state VALUES(?,?,?)
                   ON CONFLICT(state_key) DO UPDATE SET value_json=excluded.value_json,
                   updated_at=excluded.updated_at""",
                (key, json.dumps(value, ensure_ascii=False), stamp),
            )

    def get_runtime(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM runtime_state WHERE state_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def all_runtime(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT state_key,value_json FROM runtime_state").fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def delete_runtime(self, key: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM runtime_state WHERE state_key=?", (key,))
        return bool(cur.rowcount)

    def append_conversation_turn(self, *, conversation_id: str, turn_id: str,
                                 topic: str, service_mode: str,
                                 output_contracts: list[dict[str, Any]],
                                 resolved_entities: dict[str, Any],
                                 capability_ids: list[str],
                                 completed_at: str | None = None) -> dict[str, Any]:
        stamp = completed_at or now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO conversation_turn_contexts(
                       turn_id,conversation_id,topic,service_mode,
                       output_contracts_json,resolved_entities_json,
                       capability_ids_json,completed_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(turn_id) DO UPDATE SET
                       topic=excluded.topic,
                       service_mode=excluded.service_mode,
                       output_contracts_json=excluded.output_contracts_json,
                       resolved_entities_json=excluded.resolved_entities_json,
                       capability_ids_json=excluded.capability_ids_json,
                       completed_at=excluded.completed_at""",
                (
                    turn_id, conversation_id, topic, service_mode,
                    json.dumps(output_contracts, ensure_ascii=False),
                    json.dumps(resolved_entities, ensure_ascii=False),
                    json.dumps(capability_ids, ensure_ascii=False), stamp,
                ),
            )
        return {
            "turn_id": turn_id, "conversation_id": conversation_id,
            "topic": topic, "service_mode": service_mode,
            "output_contracts": output_contracts,
            "resolved_entities": resolved_entities,
            "capability_ids": capability_ids, "completed_at": stamp,
        }

    def conversation_turns(self, conversation_id: str, *, limit: int = 20,
                           active_after: str | None = None) -> list[dict[str, Any]]:
        clauses = ["conversation_id=?"]
        params: list[Any] = [conversation_id]
        if active_after:
            clauses.append("completed_at>=?")
            params.append(active_after)
        params.append(max(1, min(int(limit), 100)))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_turn_contexts WHERE "
                + " AND ".join(clauses)
                + " ORDER BY completed_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [{
            "turn_id": row["turn_id"],
            "conversation_id": row["conversation_id"],
            "topic": row["topic"],
            "service_mode": row["service_mode"],
            "output_contracts": json.loads(row["output_contracts_json"]),
            "resolved_entities": json.loads(row["resolved_entities_json"]),
            "capability_ids": json.loads(row["capability_ids_json"]),
            "completed_at": row["completed_at"],
        } for row in rows]

    def clear_conversation(self, conversation_id: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_turn_contexts WHERE conversation_id=?",
                (conversation_id,),
            )
        return int(cursor.rowcount)

    def purge_expired_conversation_turns(self, active_after: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_turn_contexts WHERE completed_at<?",
                (active_after,),
            )
        return int(cursor.rowcount)

    def upsert_semantic(self, key: str, value: Any, *, memory_type: str = "legacy",
                        confidence: float = 1.0, certainty: str = "explicit",
                        target_context_id: str = "global", evidence_ids: list[str] | None = None,
                        opposing_evidence_ids: list[str] | None = None,
                        replace_evidence: bool = False,
                        expires_at: str | None = None) -> dict[str, Any]:
        stamp = now_iso(); memory_id = new_id("sem")
        encoded_value = json.dumps(value, ensure_ascii=False)
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM semantic_memories WHERE memory_key=? AND target_context_id=?",
                (key, target_context_id),
            ).fetchone()
            if existing:
                memory_id = existing["memory_id"]
                known_evidence = {row[0] for row in conn.execute(
                    "SELECT event_id FROM semantic_evidence WHERE memory_id=? AND stance='support'",
                    (memory_id,),
                )}
                known_opposing = {row[0] for row in conn.execute(
                    "SELECT event_id FROM semantic_evidence WHERE memory_id=? AND stance='oppose'",
                    (memory_id,),
                )}
                changed = any((
                    existing["memory_type"] != memory_type,
                    existing["value_json"] != encoded_value,
                    float(existing["confidence"]) != float(confidence),
                    existing["certainty"] != certainty,
                    existing["lifecycle"] != "active",
                    existing["expires_at"] != expires_at,
                    bool(set(evidence_ids or []) - known_evidence),
                    bool(set(opposing_evidence_ids or []) - known_opposing),
                    replace_evidence and (known_evidence != set(evidence_ids or []) or
                                          known_opposing != set(opposing_evidence_ids or [])),
                ))
                if changed:
                    conn.execute(
                        """UPDATE semantic_memories SET memory_type=?,value_json=?,confidence=?,certainty=?,
                           lifecycle='active',last_supported_at=?,expires_at=?,version=?,updated_at=? WHERE memory_id=?""",
                        (memory_type, encoded_value, confidence, certainty,
                         stamp, expires_at, existing["version"] + 1, stamp, memory_id),
                    )
            else:
                conn.execute(
                    """INSERT INTO semantic_memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (memory_id, key, memory_type, encoded_value, confidence,
                     certainty, "active", target_context_id, stamp, stamp, expires_at, 1, stamp),
                )
            if replace_evidence:
                conn.execute("DELETE FROM semantic_evidence WHERE memory_id=?", (memory_id,))
            for event_id in evidence_ids or []:
                conn.execute(
                    "INSERT OR IGNORE INTO semantic_evidence VALUES(?,?, 'support')",
                    (memory_id, event_id),
                )
            for event_id in opposing_evidence_ids or []:
                conn.execute(
                    "INSERT OR IGNORE INTO semantic_evidence VALUES(?,?, 'oppose')",
                    (memory_id, event_id),
                )
        return self.get_semantic(key, target_context_id=target_context_id) or {}

    def get_semantic(self, key: str, *, target_context_id: str = "global") -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_memories WHERE memory_key=? AND target_context_id=?",
                (key, target_context_id),
            ).fetchone()
            if not row:
                return None
            evidence = [r[0] for r in conn.execute(
                "SELECT event_id FROM semantic_evidence WHERE memory_id=? AND stance='support'", (row["memory_id"],)
            )]
        out = dict(row); out["value"] = json.loads(out.pop("value_json")); out["evidence_ids"] = evidence
        return out

    def list_semantic(self, *, include_inactive: bool = False,
                      target_context_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = ["lifecycle!='deleted'"], []
        if not include_inactive: clauses.append("lifecycle='active'")
        if target_context_id:
            clauses.append("target_context_id IN (?, 'global')"); params.append(target_context_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM semantic_memories" + where + " ORDER BY updated_at DESC", params).fetchall()
            out = []
            for row in rows:
                item = dict(row)
                item["value"] = json.loads(item.pop("value_json"))
                item["evidence_ids"] = [e[0] for e in conn.execute(
                    "SELECT event_id FROM semantic_evidence WHERE memory_id=? AND stance='support'",
                    (item["memory_id"],),
                )]
                item["opposing_evidence_ids"] = [e[0] for e in conn.execute(
                    "SELECT event_id FROM semantic_evidence WHERE memory_id=? AND stance='oppose'",
                    (item["memory_id"],),
                )]
                out.append(item)
        return out

    def semantic_evidence_events(self, memory_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.*,se.stance FROM semantic_evidence se
                   JOIN events e ON e.event_id=se.event_id
                   WHERE se.memory_id=? AND e.deleted_at IS NULL ORDER BY e.seq DESC""",
                (memory_id,),
            ).fetchall()
        out = []
        for row in rows:
            item = self._event_row(row)
            item["stance"] = row["stance"]
            out.append(item)
        return out

    def delete_semantic(self, key: str, *, target_context_id: str = "global") -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE semantic_memories SET lifecycle='deleted',updated_at=? WHERE memory_key=? AND target_context_id=?",
                (now_iso(), key, target_context_id),
            )
        return bool(cur.rowcount)

    def upsert_sop(self, name: str, body: str, trigger: dict[str, Any], *,
                   target_context_id: str = "global", lifecycle: str = "candidate") -> dict[str, Any]:
        stamp = now_iso(); sop_id = new_id("sop")
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT sop_id FROM procedural_sops WHERE name=? AND target_context_id=?",
                (name, target_context_id),
            ).fetchone()
            if existing:
                sop_id = existing[0]
                conn.execute(
                    "UPDATE procedural_sops SET body=?,trigger_json=?,target_context_id=?,lifecycle=?,updated_at=? WHERE sop_id=?",
                    (body, json.dumps(trigger, ensure_ascii=False), target_context_id, lifecycle, stamp, sop_id),
                )
            else:
                conn.execute(
                    """INSERT INTO procedural_sops(sop_id,name,body,trigger_json,target_context_id,lifecycle,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (sop_id, name, body, json.dumps(trigger, ensure_ascii=False), target_context_id, lifecycle, stamp, stamp),
                )
        return self.get_sop(name, target_context_id=target_context_id) or {}

    @staticmethod
    def _sop_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row); item["trigger_conditions"] = json.loads(item.pop("trigger_json") or "{}")
        total = float(item["success_weight"] + item["failure_weight"])
        item["confidence"] = (1.0 + item["success_weight"]) / (2.0 + total)
        trigger = item["trigger_conditions"]
        if len(trigger) == 1:
            key, value = next(iter(trigger.items()))
            item["trigger"] = str(value) if key == "text_contains" else f"{key}={value}"
        else:
            item["trigger"] = ""
        return item

    def get_sop(self, name: str, *, target_context_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            if target_context_id:
                row = conn.execute(
                    "SELECT * FROM procedural_sops WHERE name=? AND target_context_id=?",
                    (name, target_context_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM procedural_sops WHERE name=? ORDER BY CASE WHEN target_context_id='global' THEN 0 ELSE 1 END LIMIT 1",
                    (name,),
                ).fetchone()
        return self._sop_row(row) if row else None

    def list_sops(self, *, include_inactive: bool = True) -> list[dict[str, Any]]:
        where = " WHERE lifecycle!='deleted'" if include_inactive else " WHERE lifecycle='active'"
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM procedural_sops" + where + " ORDER BY updated_at DESC").fetchall()
            out = []
            for row in rows:
                item = self._sop_row(row)
                item["evidence"] = [dict(e) for e in conn.execute(
                    """SELECT event_id,outcome,weight,business_date FROM sop_evidence
                       WHERE sop_id=? ORDER BY business_date DESC""",
                    (item["sop_id"],),
                )]
                out.append(item)
        return out

    def remove_sop(self, name: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("UPDATE procedural_sops SET lifecycle='deleted',updated_at=? WHERE name=?", (now_iso(), name))
        return bool(cur.rowcount)

    def record_sop_outcome(self, name: str, event_id: str, outcome: str,
                           business_date: str = "", weight: float = 1.0) -> dict[str, Any]:
        if outcome not in {"suggested", "adopted", "success", "failure"}:
            raise ValueError("invalid SOP outcome")
        if weight <= 0:
            raise ValueError("SOP evidence weight must be positive")
        if outcome in {"success", "failure"} and not business_date:
            raise ValueError("execution outcomes require a business date")
        if business_date:
            dt.date.fromisoformat(business_date)
        with self.transaction() as conn:
            sop = conn.execute(
                "SELECT * FROM procedural_sops WHERE name=? ORDER BY CASE WHEN target_context_id='global' THEN 0 ELSE 1 END LIMIT 1",
                (name,),
            ).fetchone()
            if not sop: raise KeyError(name)
            inserted = conn.execute("INSERT OR IGNORE INTO sop_evidence VALUES(?,?,?,?,?)",
                                    (sop["sop_id"], event_id, outcome, weight,
                                     business_date or None)).rowcount
            if not inserted:
                return self._sop_row(sop)
            stats = conn.execute(
                """SELECT
                   SUM(CASE WHEN outcome='success' THEN weight ELSE 0 END),
                   SUM(CASE WHEN outcome='failure' THEN weight ELSE 0 END),
                   COUNT(DISTINCT CASE WHEN outcome IN ('success','failure') THEN event_id END),
                   COUNT(DISTINCT CASE WHEN outcome IN ('success','failure') THEN business_date END)
                   FROM sop_evidence WHERE sop_id=?""", (sop["sop_id"],)
            ).fetchone()
            success, failure, cases, days = float(stats[0] or 0), float(stats[1] or 0), int(stats[2] or 0), int(stats[3] or 0)
            confidence = (1 + success) / (2 + success + failure)
            lifecycle = str(sop["lifecycle"])
            if outcome in {"success", "failure"}:
                lifecycle = ("active" if cases >= SOP_MIN_CASES and days >= SOP_MIN_DAYS
                             and confidence >= SOP_ACTIVATION_CONFIDENCE else "candidate")
            if outcome == "failure":
                consecutive = int(sop["consecutive_failures"] or 0) + 1
            elif outcome == "success": consecutive = 0
            else: consecutive = int(sop["consecutive_failures"] or 0)
            if consecutive >= 2: lifecycle = "suspended"
            effective_at = ((business_date + "T23:59:59+00:00")
                            if outcome in {"success", "failure"} else sop["last_evidence_at"])
            conn.execute(
                """UPDATE procedural_sops SET success_weight=?,failure_weight=?,independent_cases=?,
                   evidence_days=?,consecutive_failures=?,last_evidence_at=?,lifecycle=?,updated_at=? WHERE sop_id=?""",
                (success, failure, cases, days, consecutive, effective_at, lifecycle, now_iso(), sop["sop_id"]),
            )
        return self.get_sop(name) or {}

    def switch_goal(self, name: str) -> dict[str, Any]:
        stamp = now_iso(); context_id = new_id("goal")
        with self.transaction() as conn:
            conn.execute("UPDATE goal_contexts SET status='archived',updated_at=? WHERE status='active'", (stamp,))
            conn.execute("INSERT INTO goal_contexts VALUES(?,?,?,?,?,?)", (context_id, name.strip(), "active", 1, stamp, stamp))
        return {"context_id": context_id, "name": name.strip(), "status": "active", "version": 1}

    def activate_goal(self, context_id: str) -> dict[str, Any]:
        stamp = now_iso()
        with self.transaction() as conn:
            target = conn.execute("SELECT * FROM goal_contexts WHERE context_id=?", (context_id,)).fetchone()
            if not target:
                raise KeyError(context_id)
            conn.execute("UPDATE goal_contexts SET status='archived',updated_at=? WHERE status='active'", (stamp,))
            conn.execute(
                "UPDATE goal_contexts SET status='active',version=version+1,updated_at=? WHERE context_id=?",
                (stamp, context_id),
            )
            row = conn.execute("SELECT * FROM goal_contexts WHERE context_id=?", (context_id,)).fetchone()
        return dict(row)

    def list_goals(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM goal_contexts ORDER BY created_at DESC")]

    def set_lifecycle(self, object_type: str, object_id: str, lifecycle: str) -> bool:
        allowed = {
            "event": {"active", "archived"},
            "semantic": {"active", "archived"},
            "sop": {"candidate", "active", "suspended", "archived"},
        }
        if lifecycle not in allowed.get(object_type, set()):
            raise ValueError(f"unsupported lifecycle for {object_type}: {lifecycle}")
        stamp = now_iso()
        with self.transaction() as conn:
            if object_type == "event":
                changed = conn.execute(
                    "UPDATE events SET archived=? WHERE event_id=? AND deleted_at IS NULL",
                    (1 if lifecycle == "archived" else 0, object_id),
                ).rowcount
            elif object_type == "semantic":
                changed = conn.execute(
                    "UPDATE semantic_memories SET lifecycle=?,updated_at=? WHERE memory_id=? AND lifecycle!='deleted'",
                    (lifecycle, stamp, object_id),
                ).rowcount
            else:
                changed = conn.execute(
                    "UPDATE procedural_sops SET lifecycle=?,updated_at=? WHERE sop_id=? AND lifecycle!='deleted'",
                    (lifecycle, stamp, object_id),
                ).rowcount
        return bool(changed)

    def replace_search_chunks(self, event_id: str, chunks: list[dict[str, Any]]) -> None:
        """Replace one event's derived search chunks in a short transaction."""
        stamp = now_iso()
        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_search_chunks WHERE event_id=?", (event_id,))
            conn.executemany(
                """INSERT INTO memory_search_chunks
                   (chunk_id,event_id,chunk_index,content,content_hash,embedding,
                    embedding_dim,embedding_profile,metadata_json,indexed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                [(row["chunk_id"], event_id, int(row["chunk_index"]), row["content"],
                  row["content_hash"], row["embedding"], int(row["embedding_dim"]),
                  row["embedding_profile"],
                  json.dumps(row.get("metadata") or {}, ensure_ascii=False), stamp)
                 for row in chunks],
            )

    def replace_all_search_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Atomically replace the derived personal vector index."""
        stamp = now_iso()
        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_search_chunks")
            conn.executemany(
                """INSERT INTO memory_search_chunks
                   (chunk_id,event_id,chunk_index,content,content_hash,embedding,
                    embedding_dim,embedding_profile,metadata_json,indexed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                [(row["chunk_id"], row["event_id"], int(row["chunk_index"]),
                  row["content"], row["content_hash"], row["embedding"],
                  int(row["embedding_dim"]), row["embedding_profile"],
                  json.dumps(row.get("metadata") or {}, ensure_ascii=False), stamp)
                 for row in chunks],
            )

    def list_search_chunks(self, embedding_profile: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_search_chunks WHERE embedding_profile=? ORDER BY chunk_id",
                (embedding_profile,),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            items.append(item)
        return items

    def delete_search_chunks(self, event_id: str) -> int:
        with self.transaction() as conn:
            return conn.execute(
                "DELETE FROM memory_search_chunks WHERE event_id=?", (event_id,)
            ).rowcount

    def begin_file_operation(self, operation_id: str, kind: str, target_path: str,
                             before_hash: str, expected_after_hash: str,
                             detail: dict[str, Any] | None = None) -> None:
        stamp = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO operation_journal
                   (operation_id,kind,status,target_path,before_hash,after_hash,detail_json,created_at,updated_at)
                   VALUES(?,?, 'pending',?,?,?,?,?,?)
                   ON CONFLICT(operation_id) DO NOTHING""",
                (operation_id, kind, target_path, before_hash or None,
                 expected_after_hash or None, json.dumps(detail or {}, ensure_ascii=False), stamp, stamp),
            )

    def get_file_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM operation_journal WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["detail"] = json.loads(item.pop("detail_json") or "{}")
        return item

    def finish_file_operation(self, operation_id: str, *, status: str = "committed",
                              detail: dict[str, Any] | None = None) -> None:
        if status not in {"committed", "conflict", "failed"}:
            raise ValueError("invalid operation journal status")
        with self.transaction() as conn:
            if detail is None:
                conn.execute("UPDATE operation_journal SET status=?,updated_at=? WHERE operation_id=?",
                             (status, now_iso(), operation_id))
            else:
                conn.execute(
                    "UPDATE operation_journal SET status=?,detail_json=?,updated_at=? WHERE operation_id=?",
                    (status, json.dumps(detail, ensure_ascii=False), now_iso(), operation_id),
                )

    def recover_file_operations(self) -> list[dict[str, Any]]:
        """Classify unfinished file writes without overwriting third-party edits."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM operation_journal WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            path = Path(item.get("target_path") or "")
            current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
            if current and current == (item.get("after_hash") or ""):
                detail = json.loads(item.get("detail_json") or "{}")
                try:
                    if detail.get("event_kind"):
                        from memory_layers import record_business_event
                        record_business_event(detail["event_kind"], detail.get("event_payload") or {},
                                              **(detail.get("event_options") or {}))
                    status = "committed"
                except Exception:
                    results.append({"operation_id": item["operation_id"], "status": "pending",
                                    "target_path": item.get("target_path")})
                    continue
            elif current == (item.get("before_hash") or ""):
                status = "failed"
            else:
                status = "conflict"
            self.finish_file_operation(item["operation_id"], status=status)
            results.append({"operation_id": item["operation_id"], "status": status,
                            "target_path": item.get("target_path")})
        return results

    def delete_object(self, object_type: str, object_id: str, reason: str = "") -> bool:
        stamp = now_iso()
        with self.transaction() as conn:
            changed = 0
            if object_type == "event":
                event_row = conn.execute(
                    "SELECT payload_json FROM events WHERE event_id=? AND deleted_at IS NULL",
                    (object_id,),
                ).fetchone()
                semantic_ids = [row[0] for row in conn.execute(
                    "SELECT DISTINCT memory_id FROM semantic_evidence WHERE event_id=?",
                    (object_id,),
                )]
                sop_ids = [row[0] for row in conn.execute(
                    "SELECT DISTINCT sop_id FROM sop_evidence WHERE event_id=?",
                    (object_id,),
                )]
                changed = conn.execute("UPDATE events SET deleted_at=? WHERE event_id=?", (stamp, object_id)).rowcount
                conn.execute("DELETE FROM memory_search_chunks WHERE event_id=?", (object_id,))
                conn.execute("DELETE FROM semantic_evidence WHERE event_id=?", (object_id,))
                conn.execute("DELETE FROM sop_evidence WHERE event_id=?", (object_id,))
                for memory_id in semantic_ids:
                    supports = conn.execute(
                        "SELECT COUNT(*) FROM semantic_evidence WHERE memory_id=? AND stance='support'",
                        (memory_id,),
                    ).fetchone()[0]
                    if not supports:
                        conn.execute(
                            "UPDATE semantic_memories SET lifecycle='archived',updated_at=? WHERE memory_id=? AND lifecycle!='deleted'",
                            (stamp, memory_id),
                        )
                for sop_id in sop_ids:
                    self._refresh_sop_stats(conn, sop_id, stamp)
                if event_row:
                    payload = json.loads(event_row[0] or "{}")
                    snapshot_ids = {str(value) for key, value in payload.items()
                                    if key.endswith("snapshot_id") and value}
                    for snapshot_id in snapshot_ids:
                        remaining = conn.execute(
                            "SELECT COUNT(*) FROM events WHERE deleted_at IS NULL AND payload_json LIKE ?",
                            (f'%"{snapshot_id}"%',),
                        ).fetchone()[0]
                        if not remaining:
                            conn.execute(
                                "UPDATE evidence_snapshots SET deleted_at=? WHERE snapshot_id=?",
                                (stamp, snapshot_id),
                            )
            elif object_type == "semantic":
                changed = conn.execute("UPDATE semantic_memories SET lifecycle='deleted',updated_at=? WHERE memory_id=?", (stamp, object_id)).rowcount
            elif object_type == "sop":
                changed = conn.execute("UPDATE procedural_sops SET lifecycle='deleted',updated_at=? WHERE sop_id=?", (stamp, object_id)).rowcount
            if changed:
                conn.execute("INSERT OR REPLACE INTO tombstones VALUES(?,?,?,?,?)",
                             (new_id("del"), object_type, object_id, stamp, reason[:500]))
        return bool(changed)

    @staticmethod
    def _refresh_sop_stats(conn: sqlite3.Connection, sop_id: str, stamp: str) -> None:
        stats = conn.execute(
            """SELECT
               SUM(CASE WHEN outcome='success' THEN weight ELSE 0 END),
               SUM(CASE WHEN outcome='failure' THEN weight ELSE 0 END),
               COUNT(DISTINCT CASE WHEN outcome IN ('success','failure') THEN event_id END),
               COUNT(DISTINCT CASE WHEN outcome IN ('success','failure') THEN business_date END),
               MAX(CASE WHEN outcome IN ('success','failure') THEN business_date END)
               FROM sop_evidence WHERE sop_id=?""",
            (sop_id,),
        ).fetchone()
        success, failure = float(stats[0] or 0), float(stats[1] or 0)
        cases, days = int(stats[2] or 0), int(stats[3] or 0)
        confidence = (1 + success) / (2 + success + failure)
        lifecycle = ("active" if cases >= SOP_MIN_CASES and days >= SOP_MIN_DAYS
                     and confidence >= SOP_ACTIVATION_CONFIDENCE else "candidate")
        outcomes = conn.execute(
            """SELECT se.outcome FROM sop_evidence se JOIN events e ON e.event_id=se.event_id
               WHERE se.sop_id=? AND se.outcome IN ('success','failure')
               ORDER BY e.seq DESC LIMIT 2""",
            (sop_id,),
        ).fetchall()
        consecutive_failures = 0
        for row in outcomes:
            if row[0] != "failure":
                break
            consecutive_failures += 1
        if consecutive_failures >= 2:
            lifecycle = "suspended"
        last_evidence_at = (str(stats[4]) + "T23:59:59+00:00") if stats[4] else None
        conn.execute(
            """UPDATE procedural_sops SET success_weight=?,failure_weight=?,independent_cases=?,
               evidence_days=?,consecutive_failures=?,last_evidence_at=?,lifecycle=?,updated_at=?
               WHERE sop_id=? AND lifecycle!='deleted'""",
            (success, failure, cases, days, consecutive_failures, last_evidence_at,
             lifecycle, stamp, sop_id),
        )

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            kinds = {r[0]: r[1] for r in conn.execute(
                "SELECT kind,COUNT(*) FROM events WHERE deleted_at IS NULL GROUP BY kind"
            )}
            return {
                "events": sum(kinds.values()), "event_kinds": kinds,
                "semantic": conn.execute("SELECT COUNT(*) FROM semantic_memories WHERE lifecycle='active'").fetchone()[0],
                "sops": conn.execute("SELECT COUNT(*) FROM procedural_sops WHERE lifecycle='active'").fetchone()[0],
                "search_chunks": conn.execute("SELECT COUNT(*) FROM memory_search_chunks").fetchone()[0],
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                "sqlite_version": sqlite3.sqlite_version,
            }
