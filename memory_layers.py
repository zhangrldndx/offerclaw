# -*- coding: utf-8 -*-
"""OfferClaw's three memory layers backed by transactional SQLite.

The public classes keep the original API so existing callers continue to work.
JSON/JSONL paths are materialized compatibility exports; memory.sqlite3 is the
source of truth.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from domain_status import MatchStatusCode, match_status_code
from memory_models import EventEnvelope, validate_payload
from memory_store import MemoryStore, new_id, now_iso


BASE_DIR_DEFAULT = str(Path(__file__).resolve().parent / "logs" / "memory")
_ORIGINAL_BASE_DIR = BASE_DIR_DEFAULT
_EXPORT_LOCK = threading.RLock()
_MODEL_DISTILL_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-distill")
_RUNTIME_KEYS = {"last_advice", "last_distilled_at"}
_ENVELOPE_INPUTS = {
    "event_id", "id", "schema_version", "occurred_at", "recorded_at", "ts_iso",
    "business_date", "actor", "source", "traffic_origin", "operation_id", "causation_id",
    "conversation_id", "target_context_id", "entity_type", "entity_id", "archived", "deleted_at",
}


def _setting(name: str, default: float, cast):
    try:
        value = cast(os.environ.get(name, str(default)))
        return value if value > 0 else cast(default)
    except (TypeError, ValueError):
        return cast(default)


SEMANTIC_MIN_EVIDENCE = _setting("OFFERCLAW_SEMANTIC_MIN_EVIDENCE", 3, int)
SEMANTIC_MIN_DAYS = _setting("OFFERCLAW_SEMANTIC_MIN_DAYS", 2, int)
STAGE_INTEREST_HALF_LIFE_DAYS = _setting("OFFERCLAW_STAGE_INTEREST_HALF_LIFE_DAYS", 30, int)
INFERRED_HALF_LIFE_DAYS = _setting("OFFERCLAW_INFERRED_HALF_LIFE_DAYS", 90, int)
INFERRED_ARCHIVE_DAYS = _setting("OFFERCLAW_INFERRED_ARCHIVE_DAYS", 180, int)
SOP_HALF_LIFE_DAYS = _setting("OFFERCLAW_SOP_HALF_LIFE_DAYS", 90, int)
SOP_ARCHIVE_DAYS = _setting("OFFERCLAW_SOP_ARCHIVE_DAYS", 180, int)


def _iso_now() -> str:
    return now_iso()


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: str, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _safe_load_json(path: str, default: dict) -> dict:
    import copy
    if not os.path.exists(path):
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else copy.deepcopy(default)
    except json.JSONDecodeError:
        import shutil
        import sys
        backup = f"{path}.corrupt.{int(dt.datetime.now().timestamp())}"
        try:
            shutil.copy2(path, backup)
        except OSError:
            backup = "(backup failed)"
        print(f"[memory] damaged JSON backed up to {backup}", file=sys.stderr)
        return copy.deepcopy(default)
    except OSError:
        return copy.deepcopy(default)


def _configured_base(base_dir: str | None) -> str | None:
    if base_dir:
        return base_dir
    if BASE_DIR_DEFAULT != _ORIGINAL_BASE_DIR:
        return BASE_DIR_DEFAULT
    return None


def _traffic_origin() -> str:
    try:
        from traffic_origin import current_traffic_origin
        return current_traffic_origin()
    except Exception:
        return "organic"


def _validate_event(event: Any) -> None:
    if not isinstance(event, dict):
        raise ValueError("memory event must be an object")
    kind = event.get("kind")
    if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", kind):
        raise ValueError("memory event kind must be a non-empty snake_case code")
    payload = {k: v for k, v in event.items() if k not in _ENVELOPE_INPUTS and k != "kind"}
    validate_payload(kind, payload)


def _event_envelope(event: dict[str, Any], store: MemoryStore) -> dict[str, Any]:
    _validate_event(event)
    raw = dict(event)
    kind = raw.pop("kind")
    payload = {k: v for k, v in raw.items() if k not in _ENVELOPE_INPUTS}
    if kind in {"match_run", "career_flow_run", "match_completed"}:
        payload["status_code"] = match_status_code(
            payload.get("status_code") or payload.get("status")
        ).value
    payload = validate_payload(kind, payload)
    stamp = str(raw.get("occurred_at") or raw.get("ts_iso") or now_iso())
    origin = str(raw.get("traffic_origin") or _traffic_origin())
    automated = origin in {"test", "stress", "historical_replay", "agent_generated"}
    envelope = EventEnvelope(
        event_id=new_id("ep"), schema_version=1, kind=kind,
        occurred_at=stamp, recorded_at=now_iso(),
        business_date=raw.get("business_date") or payload.get("date"),
        actor=raw.get("actor") or "system", source=raw.get("source") or "memory_api",
        traffic_origin=origin, operation_id=raw.get("operation_id"),
        causation_id=raw.get("causation_id"), conversation_id=raw.get("conversation_id"),
        target_context_id=raw.get("target_context_id") or store.active_goal_id(),
        entity_type=raw.get("entity_type"), entity_id=raw.get("entity_id"),
        archived=bool(raw.get("archived", automated)), payload=payload,
    )
    return envelope.model_dump(mode="json")


def _legacy_migration(store: MemoryStore, legacy_jsonl: Path) -> None:
    marker = f"legacy-jsonl:{legacy_jsonl.resolve()}"
    with store._connect() as conn:
        if conn.execute("SELECT 1 FROM migration_map WHERE source_key=?", (marker,)).fetchone():
            return
        already_has_events = bool(conn.execute("SELECT 1 FROM events LIMIT 1").fetchone())
    if already_has_events or not legacy_jsonl.exists():
        with store.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO migration_map VALUES(?,?,?,?)",
                         (marker, None, "skipped", "database already populated or source absent"))
        return
    backup_root = store.base_dir / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"pre_sqlite_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ("episodic.jsonl", "semantic.json", "procedural.json"):
        source = store.base_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    _atomic_write_json(str(backup_dir / "migration_report.json"), {
        "created_at": now_iso(), "source": str(legacy_jsonl),
        "source_sha256": hashlib.sha256(legacy_jsonl.read_bytes()).hexdigest(),
        "source_lines": len(legacy_jsonl.read_text(encoding="utf-8").splitlines()),
    })
    imported = quarantined = 0
    for line_no, line in enumerate(legacy_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        source_key = f"{marker}:{line_no}"
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("event is not an object")
            raw["actor"] = "import"
            raw["source"] = "legacy_jsonl"
            raw["traffic_origin"] = str(raw.get("traffic_origin") or "unclassified")
            if raw.get("kind") == "career_flow_run" and raw["traffic_origin"] != "organic":
                raw["archived"] = True
            event = store.append_event(_event_envelope(raw, store))
            with store.transaction() as conn:
                conn.execute("INSERT OR REPLACE INTO migration_map VALUES(?,?,?,?)",
                             (source_key, event["event_id"], "imported", str(raw.get("id") or "")))
            imported += 1
        except Exception as exc:
            with store.transaction() as conn:
                conn.execute("INSERT OR REPLACE INTO migration_map VALUES(?,?,?,?)",
                             (source_key, None, "quarantined", str(exc)[:500]))
            quarantined += 1
    with store.transaction() as conn:
        conn.execute("INSERT OR REPLACE INTO migration_map VALUES(?,?,?,?)",
                     (marker, None, "complete", json.dumps({"imported": imported, "quarantined": quarantined})))


class EpisodicMemory:
    FILE = "episodic.jsonl"

    def __init__(self, base_dir: str | None = None) -> None:
        self.store = MemoryStore(_configured_base(base_dir))
        self.base_dir = str(self.store.base_dir)
        self.path = str(self.store.base_dir / self.FILE)
        _legacy_migration(self.store, Path(self.path))

    def _export(self) -> None:
        from io_utils import file_lock
        with file_lock(self.path):
            rows = self.store.list_events(include_archived=True)
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            _atomic_write_text(self.path, text)

    def append(self, event: dict[str, Any], *, export: bool = True,
               index: bool = True) -> dict[str, Any]:
        with _EXPORT_LOCK:
            result = self.store.append_event(_event_envelope(dict(event), self.store))
            if export:
                self._export()
        if index:
            try:
                from memory_search import schedule_index_event
                schedule_index_event(result["event_id"], store=self.store)
            except Exception:
                pass
        return result

    def all(self) -> list[dict[str, Any]]:
        return self.store.list_events(include_archived=True)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.list_events(limit=limit, include_archived=True)

    def filter(self, predicate: Callable[[dict], bool]) -> list[dict]:
        return [event for event in self.all() if predicate(event)]

    def count_by(self, key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.all():
            value = str(event.get(key, ""))
            if value:
                counts[value] = counts.get(value, 0) + 1
        return counts


class SemanticMemory:
    FILE = "semantic.json"

    def __init__(self, base_dir: str | None = None) -> None:
        self.store = MemoryStore(_configured_base(base_dir))
        self.base_dir = str(self.store.base_dir)
        self.path = str(self.store.base_dir / self.FILE)
        self._import_legacy()
        self._migrate_nonsemantic_state()

    def _import_legacy(self) -> None:
        marker = f"legacy-semantic:{Path(self.path).resolve()}"
        with self.store._connect() as conn:
            if conn.execute("SELECT 1 FROM migration_map WHERE source_key=?", (marker,)).fetchone():
                return
        data = _safe_load_json(self.path, {})
        for key, value in data.items():
            if key in _RUNTIME_KEYS:
                self.store.set_runtime(key, value)
            elif key != "_meta":
                self.store.upsert_semantic(key, value, memory_type="legacy_import", certainty="pending")
        with self.store.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO migration_map VALUES(?,?,?,?)",
                         (marker, None, "complete", f"imported={max(0, len(data)-1)}"))

    def _migrate_nonsemantic_state(self) -> None:
        with self.store.transaction() as conn:
            rows = conn.execute(
                "SELECT memory_key,value_json FROM semantic_memories WHERE memory_key IN (?,?)",
                tuple(sorted(_RUNTIME_KEYS)),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO runtime_state(state_key,value_json,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(state_key) DO NOTHING",
                    (row["memory_key"], row["value_json"], now_iso()),
                )
            conn.execute(
                "UPDATE semantic_memories SET lifecycle='deleted',updated_at=? "
                "WHERE memory_key IN (?,?)",
                (now_iso(), *tuple(sorted(_RUNTIME_KEYS))),
            )
            empty_adjustments = conn.execute(
                "SELECT memory_id,value_json FROM semantic_memories "
                "WHERE memory_key=? AND lifecycle='active'", (ADJUSTMENTS_KEY,)
            ).fetchall()
            for row in empty_adjustments:
                try:
                    has_rules = bool(json.loads(row["value_json"] or "{}").get("rules"))
                except (json.JSONDecodeError, AttributeError):
                    has_rules = False
                if not has_rules:
                    conn.execute(
                        "UPDATE semantic_memories SET lifecycle='archived',updated_at=? WHERE memory_id=?",
                        (now_iso(), row["memory_id"]),
                    )

    def _export(self) -> None:
        from io_utils import file_lock
        with file_lock(self.path):
            data = {(row["memory_key"] if row["target_context_id"] == "global"
                     else f"{row['target_context_id']}:{row['memory_key']}"): row["value"]
                    for row in self.store.list_semantic(include_inactive=False)}
            data["_meta"] = {"updated_at": now_iso(), "source_of_truth": "memory.sqlite3"}
            _atomic_write_json(self.path, data)

    def get(self, key: str, default: Any = None) -> Any:
        if key in _RUNTIME_KEYS:
            return self.store.get_runtime(key, default)
        row = self.store.get_semantic(key)
        return row["value"] if row and row["lifecycle"] == "active" else default

    def set(self, key: str, value: Any) -> None:
        if key in _RUNTIME_KEYS:
            self.store.set_runtime(key, value)
            return
        # The export is not authoritative, but preserve a visibly corrupted
        # copy before replacing it so recovery/auditing remains possible.
        if os.path.exists(self.path):
            _safe_load_json(self.path, {})
        self.store.upsert_semantic(key, value)
        with _EXPORT_LOCK:
            self._export()

    def delete(self, key: str) -> bool:
        if key in _RUNTIME_KEYS:
            return self.store.delete_runtime(key)
        changed = self.store.delete_semantic(key)
        if changed:
            with _EXPORT_LOCK:
                self._export()
        return changed

    def all(self) -> dict[str, Any]:
        out = {row["memory_key"]: row["value"] for row in self.store.list_semantic()}
        out.update(self.store.all_runtime())
        out["_meta"] = {"updated_at": now_iso(), "source_of_truth": "memory.sqlite3"}
        return out


def _parse_trigger(trigger: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(trigger, dict):
        return trigger
    raw = str(trigger or "").strip()
    if "=" in raw:
        key, value = raw.split("=", 1)
        return {key.strip(): value.strip()}
    return {"text_contains": raw} if raw else {"global": True}


class ProceduralMemory:
    FILE = "procedural.json"

    def __init__(self, base_dir: str | None = None) -> None:
        self.store = MemoryStore(_configured_base(base_dir))
        self.base_dir = str(self.store.base_dir)
        self.path = str(self.store.base_dir / self.FILE)
        self._import_legacy()

    def _import_legacy(self) -> None:
        marker = f"legacy-procedural:{Path(self.path).resolve()}"
        with self.store._connect() as conn:
            if conn.execute("SELECT 1 FROM migration_map WHERE source_key=?", (marker,)).fetchone(): return
        data = _safe_load_json(self.path, {}).get("sops", {})
        for name, raw in data.items():
            self.store.upsert_sop(name, str(raw.get("body") or ""), _parse_trigger(raw.get("trigger", "")), lifecycle="candidate")
        with self.store.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO migration_map VALUES(?,?,?,?)", (marker, None, "complete", f"imported={len(data)}"))

    def _export(self) -> None:
        from io_utils import file_lock
        with file_lock(self.path):
            rows = self.store.list_sops(include_inactive=True)
            data = {"sops": {(row["name"] if row["target_context_id"] == "global"
                              else f"{row['target_context_id']}:{row['name']}"): row
                             for row in rows if row["lifecycle"] != "deleted"},
                    "_meta": {"updated_at": now_iso(), "source_of_truth": "memory.sqlite3"}}
            _atomic_write_json(self.path, data)

    def add(self, name: str, *, body: str, trigger: str | dict[str, Any] = "",
            lifecycle: str = "candidate", target_context_id: str = "global") -> dict:
        result = self.store.upsert_sop(name, body, _parse_trigger(trigger),
                                       target_context_id=target_context_id, lifecycle=lifecycle)
        with _EXPORT_LOCK: self._export()
        return result

    def get(self, name: str) -> dict | None:
        row = self.store.get_sop(name)
        return row if row and row["lifecycle"] != "deleted" else None

    def list(self) -> list[dict]:
        return [row for row in self.store.list_sops(include_inactive=True) if row["lifecycle"] != "deleted"]

    def remove(self, name: str) -> bool:
        changed = self.store.remove_sop(name)
        if changed:
            with _EXPORT_LOCK: self._export()
        return changed


def record_business_event(kind: str, payload: dict[str, Any], *, actor: str = "user",
                          source: str = "business_service", operation_id: str | None = None,
                          entity_type: str | None = None, entity_id: str | None = None,
                          conversation_id: str | None = None, causation_id: str | None = None,
                          business_date: str | None = None,
                          target_context_id: str | None = None) -> dict[str, Any]:
    episodic = EpisodicMemory()
    result = episodic.append({
        "kind": kind, **payload, "actor": actor, "source": source,
        "operation_id": operation_id, "entity_type": entity_type, "entity_id": entity_id,
        "conversation_id": conversation_id, "causation_id": causation_id,
        "business_date": business_date, "target_context_id": target_context_id,
    })
    should_distill = kind in {"profile_edited", "application_changed", "application_review_recorded",
                              "daily_log_recorded"}
    should_distill = should_distill or (kind == "conversation_message" and actor == "user")
    if should_distill:
        try:
            distill_to_semantic(episodic, SemanticMemory(str(episodic.store.base_dir)))
        except Exception:
            pass
        if actor == "user":
            schedule_model_assisted_distillation(episodic)
    return result


def _trusted(event: dict[str, Any]) -> bool:
    return (not event.get("archived") and not event.get("deleted_at") and
            event.get("traffic_origin", "organic") == "organic")


def _topic_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    latin = re.findall(r"[a-z][a-z0-9+#.]{2,}", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,12}", normalized)
    stop = {"今天", "之前", "什么", "怎么", "这个", "那个", "可以", "是否", "用户", "岗位"}
    return {token for token in latin + chinese if token not in stop}


def _markdown_section(content: str, heading_number: int) -> str:
    match = re.search(
        rf"(?ms)^##\s+{heading_number}\.[^\n]*\n(.*?)(?=^##\s+\d+\.|\Z)",
        content or "",
    )
    return match.group(1).strip() if match else ""


def _field_value(section: str, label: str) -> str:
    match = re.search(rf"(?m)^-\s*{re.escape(label)}[：:]\s*(.+)$", section or "")
    value = match.group(1).strip() if match else ""
    return "" if "【待补充】" in value else value


def _profile_facts(content: str) -> dict[str, Any]:
    basic = _markdown_section(content, 1)
    preference = _markdown_section(content, 2)
    availability = _markdown_section(content, 10)
    directions: list[str] = []
    direction_block = re.search(
        r"(?ms)^-\s*目标方向[^\n]*[：:]\s*\n(.*?)(?=^-\s*[^\n]+[：:]|\Z)", preference
    )
    if direction_block:
        directions = [item.strip() for item in re.findall(
            r"(?m)^\s*\d+[.、]\s*(.+?)\s*$", direction_block.group(1)
        ) if "【待补充】" not in item]
    exclusions: list[str] = []
    exclusion_block = re.search(
        r"(?ms)^-\s*明确不做的方向[：:]\s*\n(.*?)(?=^-\s*[^\n]+[：:]|\Z)", preference
    )
    if exclusion_block:
        exclusions = [item.strip() for item in re.findall(
            r"(?m)^\s*-\s*(.+?)\s*$", exclusion_block.group(1)
        ) if "【待补充】" not in item]
    identity = {
        "education": _field_value(basic, "学历层次"),
        "major": _field_value(basic, "专业"),
        "graduation": _field_value(basic, "毕业时间"),
        "current_location": _field_value(basic, "所在地"),
        "accepted_locations": _field_value(basic, "可接受工作地域"),
    }
    career = {
        "target_directions": directions,
        "position_type": _field_value(preference, "目标岗位类型"),
        "industry_preference": _field_value(preference, "行业偏好"),
        "excluded_directions": exclusions,
    }
    time_budget = {
        "daily": _field_value(availability, "每天可投入（小时）"),
        "weekly": _field_value(availability, "每周可投入（小时）"),
        "preferred_hours": _field_value(availability, "黄金时段"),
        "unavailable_hours": _field_value(availability, "不可打扰时段"),
    }
    return {"identity_constraints": {k: v for k, v in identity.items() if v},
            "career_preferences": {k: v for k, v in career.items() if v},
            "availability": {k: v for k, v in time_budget.items() if v}}


def _latest_profile_evidence(events: list[dict[str, Any]], sem: SemanticMemory) -> None:
    candidates = [event for event in events if event.get("kind") == "profile_edited" or
                  (event.get("kind") == "historical_snapshot_imported" and
                   event.get("source_kind") == "profile")]
    if not candidates:
        return
    event = max(candidates, key=lambda item: int(item.get("seq") or 0))
    snapshot_id = event.get("after_snapshot_id") or event.get("snapshot_id")
    snapshot = sem.store.get_snapshot(str(snapshot_id or ""))
    if not snapshot:
        return
    facts = _profile_facts(str(snapshot.get("content") or ""))
    evidence_ids = [event["event_id"]]
    if facts["identity_constraints"]:
        sem.store.upsert_semantic(
            "profile:identity_constraints", facts["identity_constraints"],
            memory_type="confirmed_fact", certainty="explicit", confidence=1.0,
            evidence_ids=evidence_ids,
        )
    if facts["availability"]:
        sem.store.upsert_semantic(
            "profile:availability", facts["availability"],
            memory_type="confirmed_constraint", certainty="explicit", confidence=1.0,
            evidence_ids=evidence_ids,
        )
    if facts["career_preferences"]:
        sem.store.upsert_semantic(
            "profile:career_preferences", facts["career_preferences"],
            memory_type="confirmed_preference", certainty="explicit", confidence=1.0,
            target_context_id=event.get("target_context_id") or sem.store.active_goal_id(),
            evidence_ids=evidence_ids,
        )


def _distill_execution_capacity(events: list[dict[str, Any]], sem: SemanticMemory) -> None:
    logs = [event for event in events if event.get("kind") == "daily_log_recorded"]
    independent: dict[str, dict[str, Any]] = {}
    for event in logs:
        independent[str(event.get("operation_id") or event["event_id"])] = event
    logs = list(independent.values())
    days = {str(event.get("business_date") or event.get("date") or "") for event in logs}
    if len(logs) < SEMANTIC_MIN_EVIDENCE or len(days - {""}) < SEMANTIC_MIN_DAYS:
        return
    minutes = sorted(int(event["minutes"]) for event in logs if event.get("minutes") is not None)
    completed = sum(len(event.get("done") or []) for event in logs)
    incomplete = sum(len(event.get("incomplete") or []) for event in logs)
    total = completed + incomplete
    value = {
        "observed_days": len(days - {""}),
        "median_minutes": minutes[len(minutes) // 2] if minutes else None,
        "completion_rate": round(completed / total, 3) if total else None,
        "basis": "daily_execution_records",
    }
    sem.store.upsert_semantic(
        "execution:observed_capacity", value, memory_type="behavior_pattern",
        certainty="inferred", confidence=min(.9, .6 + .04 * len(logs)),
        target_context_id=logs[-1].get("target_context_id") or sem.store.active_goal_id(),
        evidence_ids=[event["event_id"] for event in logs],
    )


def _distill_explicit_statements(events: list[dict[str, Any]], sem: SemanticMemory) -> None:
    patterns = (
        ("negative", re.compile(r"(?:我)?(?:明确)?(?:不喜欢|不希望|不考虑|排除|拒绝)\s*([^。！？\n]{2,80})")),
        ("positive", re.compile(r"(?:我)?(?:更)?(?:偏好|希望|倾向于|长期关注)\s*([^。！？\n]{2,80})")),
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("actor") != "user":
            continue
        if event.get("kind") == "conversation_message" and event.get("role") == "user":
            text = str(event.get("content") or "")
        elif event.get("kind") == "daily_log_recorded":
            text = str(event.get("notes") or "")
        elif event.get("kind") == "application_review_recorded":
            snapshot = sem.store.get_snapshot(str(event.get("snapshot_id") or ""))
            text = str((snapshot or {}).get("content") or "")
        else:
            continue
        for polarity, pattern in patterns:
            for match in pattern.finditer(text):
                topic = match.group(1).strip(" ，,：:；;")
                if not topic:
                    continue
                target = str(event.get("target_context_id") or sem.store.active_goal_id())
                key = f"explicit_preference:{_norm_task_key(topic, target)}"
                grouped.setdefault((key, target), []).append(
                    {"event": event, "polarity": polarity, "topic": topic,
                     "statement": match.group(0).strip()}
                )
    for (key, target), records in grouped.items():
        records.sort(key=lambda item: int(item["event"].get("seq") or 0))
        latest = records[-1]
        supports = [item["event"]["event_id"] for item in records
                    if item["polarity"] == latest["polarity"]]
        opposes = [item["event"]["event_id"] for item in records
                   if item["polarity"] != latest["polarity"]]
        sem.store.upsert_semantic(
            key, {"topic": latest["topic"], "polarity": latest["polarity"],
                  "statement": latest["statement"]},
            memory_type="confirmed_preference", certainty="explicit", confidence=1.0,
            target_context_id=target, evidence_ids=supports,
            opposing_evidence_ids=opposes, replace_evidence=True,
        )


def distill_to_semantic(epi: EpisodicMemory, sem: SemanticMemory) -> dict:
    active_goal = sem.store.active_goal_id()
    events = [event for event in epi.all() if _trusted(event) and
              event.get("target_context_id") in {active_goal, "global", None}]
    if not events:
        return {"distilled": False, "reason": "no_trusted_events"}
    _latest_profile_evidence(events, sem)
    _distill_execution_capacity(events, sem)
    _distill_explicit_statements(events, sem)
    status_dist: dict[str, int] = {}
    match_events = []
    for event in events:
        if event.get("kind") not in {"match_run", "career_flow_run", "match_completed"}:
            continue
        match_events.append(event)
        code = match_status_code(event.get("status_code") or event.get("status")).value
        status_dist[code] = status_dist.get(code, 0) + 1
    if status_dist:
        sem.store.upsert_semantic("match_status_distribution", status_dist,
                                  memory_type="derived_statistic", certainty="inferred", confidence=1.0,
                                  evidence_ids=[e["event_id"] for e in match_events])
    latest_app: dict[str, dict[str, Any]] = {}
    for event in events:
        candidate = event
        if (event.get("kind") == "historical_snapshot_imported" and
                event.get("source_kind") == "application"):
            snapshot = sem.store.get_snapshot(str(event.get("snapshot_id") or ""))
            try:
                historical = json.loads(str((snapshot or {}).get("content") or "{}"))
            except json.JSONDecodeError:
                historical = {}
            if isinstance(historical, dict):
                candidate = {**event, **{field: historical.get(field) for field in (
                    "application_id", "company", "position", "status", "status_code",
                    "long_term_follow", "next_action", "date")}}
        if candidate.get("kind") == "application_changed" or (
                candidate.get("kind") == "historical_snapshot_imported" and
                candidate.get("source_kind") == "application"):
            key = str(candidate.get("application_id") or candidate.get("entity_id") or "").strip()
            if key and int(candidate.get("seq") or 0) >= int(latest_app.get(key, {}).get("seq") or 0):
                latest_app[key] = candidate
    followed: dict[str, list[dict]] = {}
    for event in latest_app.values():
        if event.get("long_term_follow") is True:
            key = str(event.get("position") or event.get("entity_id") or "").strip()
            if key: followed.setdefault(key, []).append(event)
    applications_by_target: dict[str, list[dict[str, Any]]] = {}
    for event in latest_app.values():
        target = str(event.get("target_context_id") or "global")
        applications_by_target.setdefault(target, []).append(event)
    for target, app_events in applications_by_target.items():
        sem.store.upsert_semantic(
            "application:current_choices",
            [{"application_id": event.get("application_id"),
              "company": event.get("company"), "position": event.get("position"),
              "status_code": event.get("status_code"),
              "long_term_follow": bool(event.get("long_term_follow"))}
             for event in app_events],
            memory_type="confirmed_activity", certainty="explicit", confidence=1.0,
            target_context_id=target,
            evidence_ids=[event["event_id"] for event in app_events],
        )
    active_follow_keys: set[tuple[str, str]] = set()
    for position, evidence in followed.items():
        memory_key = f"long_term_interest:{_norm_task_key(position)}"
        target = evidence[-1].get("target_context_id") or "global"
        active_follow_keys.add((memory_key, target))
        sem.store.upsert_semantic(
            memory_key, {"position": position},
            memory_type="stage_interest", certainty="explicit", confidence=1.0,
            target_context_id=target,
            evidence_ids=[e["event_id"] for e in evidence],
        )
    for item in sem.store.list_semantic(include_inactive=False):
        if item["memory_key"].startswith("long_term_interest:") and (
                item["memory_key"], item["target_context_id"]) not in active_follow_keys:
            sem.store.set_lifecycle("semantic", item["memory_id"], "archived")
    topic_evidence: dict[str, list[dict]] = {}
    for event in events:
        if event.get("kind") == "conversation_message" and event.get("role") == "user":
            for topic in _topic_tokens(str(event.get("content") or "")):
                topic_evidence.setdefault(topic, []).append(event)
    interests = []
    for topic, evidence in topic_evidence.items():
        operations = {e.get("operation_id") or e.get("event_id") for e in evidence}
        days = {str(e.get("business_date") or e.get("occurred_at", "")[:10]) for e in evidence}
        if len(operations) >= SEMANTIC_MIN_EVIDENCE and len(days) >= SEMANTIC_MIN_DAYS:
            sem.store.upsert_semantic(
                f"frequent_topic:{_norm_task_key(topic)}", {"topic": topic},
                memory_type="stage_interest", certainty="inferred", confidence=min(.9, .6 + .05 * len(operations)),
                evidence_ids=[e["event_id"] for e in evidence],
                target_context_id=evidence[-1].get("target_context_id") or "global",
            )
            interests.append(topic)
    sem.store.set_runtime("last_distilled_at", now_iso())
    sem._export()
    return {"distilled": True, "n_events": len(events), "n_match_events": len(match_events),
            "status_dist": status_dist, "frequent_topics": interests, "long_term_interests": list(followed)}


def _free_text_value(event: dict[str, Any], store: MemoryStore) -> str:
    kind = str(event.get("kind") or "")
    if kind == "conversation_message" and event.get("role") == "user":
        return str(event.get("content") or "").strip()
    if kind == "daily_log_recorded":
        parts = [str(event.get("notes") or "")]
        for key in ("done", "incomplete", "blockers", "skill_evidence"):
            if event.get(key):
                parts.append(json.dumps(event[key], ensure_ascii=False))
        return "\n".join(part for part in parts if part).strip()
    if kind == "application_review_recorded":
        snapshot = store.get_snapshot(str(event.get("snapshot_id") or ""))
        return str((snapshot or {}).get("content") or "").strip()
    if kind == "reflection":
        parts = [str(event.get("user_reflection") or "")]
        for key in ("completed", "incomplete", "blockers", "skill_evidence"):
            if event.get(key):
                parts.append(json.dumps(event[key], ensure_ascii=False))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _decode_candidate_json(raw: str) -> list[dict[str, Any]]:
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", str(raw or "").strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        rows = parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return []


def _has_explicit_conflict(sem: SemanticMemory, topic: str) -> bool:
    normalized = _norm_task_text(topic)
    tokens = _topic_tokens(topic)
    for item in sem.store.list_semantic(include_inactive=False):
        if item.get("certainty") != "explicit":
            continue
        value_text = json.dumps(item.get("value") or {}, ensure_ascii=False)
        value_normalized = _norm_task_text(value_text)
        if normalized and (normalized in value_normalized or value_normalized in normalized):
            return True
        if tokens and tokens & _topic_tokens(value_text):
            return True
    return False


def distill_free_text_with_model(epi: EpisodicMemory, sem: SemanticMemory, *,
                                 llm_call: Callable | None = None,
                                 force: bool = False) -> dict[str, Any]:
    """Extract recurring free-text memories, then verify every claim in code."""
    target = sem.store.active_goal_id()
    source_rows = []
    for event in epi.all():
        if not _trusted(event) or event.get("actor") != "user":
            continue
        if event.get("target_context_id") not in {target, "global", None}:
            continue
        text = _free_text_value(event, sem.store)
        if not text:
            continue
        source_rows.append({
            "event": event, "text": text,
            "date": str(event.get("business_date") or event.get("occurred_at", "")[:10]),
            "operation": str(event.get("operation_id") or event["event_id"]),
        })
    max_seq = max((int(row["event"].get("seq") or 0) for row in source_rows), default=0)
    state = sem.store.get_runtime("model_distill_status", {}) or {}
    if not force and int(state.get("last_seq") or 0) >= max_seq:
        return {"status": "unchanged", "last_seq": max_seq, "accepted": 0}
    if len({row["operation"] for row in source_rows}) < SEMANTIC_MIN_EVIDENCE or len(
            {row["date"] for row in source_rows if row["date"]}) < SEMANTIC_MIN_DAYS:
        sem.store.set_runtime("model_distill_status", {
            "status": "waiting_for_evidence", "last_seq": max_seq, "updated_at": now_iso(),
        })
        return {"status": "waiting_for_evidence", "last_seq": max_seq, "accepted": 0}
    if not force and state.get("retry_after"):
        try:
            if dt.datetime.fromisoformat(str(state["retry_after"])) > dt.datetime.now(dt.timezone.utc).astimezone():
                return {"status": "retry_later", "last_seq": max_seq, "accepted": 0}
        except ValueError:
            pass

    compact = [{"event_id": row["event"]["event_id"], "date": row["date"],
                "operation_id": row["operation"], "text": row["text"][:1200]}
               for row in source_rows[-100:]]
    messages = [
        {"role": "system", "content": (
            "你只负责从用户原文中提出可核验的长期记忆候选。不要修改正式职业目标，不要把助手输出、计划内容、JD要求写成用户事实。"
            "只输出 JSON：{\"candidates\":[{\"kind\":\"interest|preference|behavior\","
            "\"topic\":\"...\",\"polarity\":\"positive|negative|neutral\",\"summary\":\"...\","
            "\"scope\":\"current_goal\",\"support\":[{\"event_id\":\"...\",\"quote\":\"原文连续片段\"}],"
            "\"oppose\":[{\"event_id\":\"...\",\"quote\":\"原文连续片段\"}]}]}。"
            "每个候选至少引用 3 个独立操作且跨 2 个日期；证据不足就不输出。"
        )},
        {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
    ]
    try:
        if llm_call is None:
            from rag_tools import chat_with_llm
            llm_call = chat_with_llm
        raw = llm_call(messages, temperature=0.0, max_tokens=2200)
        candidates = _decode_candidate_json(raw)
    except Exception as exc:
        retry_after = (dt.datetime.now(dt.timezone.utc).astimezone() + dt.timedelta(hours=1)).isoformat()
        sem.store.set_runtime("model_distill_status", {
            "status": "failed", "last_seq": int(state.get("last_seq") or 0),
            "error_type": type(exc).__name__, "retry_after": retry_after, "updated_at": now_iso(),
        })
        return {"status": "failed", "accepted": 0, "error_type": type(exc).__name__,
                "retry_after": retry_after}

    by_id = {row["event"]["event_id"]: row for row in source_rows}
    accepted = []; rejected = []
    kind_map = {"interest": "stage_interest", "preference": "inferred_preference",
                "behavior": "behavior_pattern"}
    for candidate in candidates[:20]:
        kind = str(candidate.get("kind") or "")
        topic = str(candidate.get("topic") or "").strip()
        summary = str(candidate.get("summary") or "").strip()
        polarity = str(candidate.get("polarity") or "neutral")
        if (kind not in kind_map or candidate.get("scope") != "current_goal" or
                not 2 <= len(topic) <= 80 or not 4 <= len(summary) <= 300 or
                polarity not in {"positive", "negative", "neutral"}):
            rejected.append({"topic": topic[:80], "reason": "invalid_contract"})
            continue
        supports = []
        support_quotes = []
        for reference in candidate.get("support") or []:
            if not isinstance(reference, dict):
                continue
            row = by_id.get(str(reference.get("event_id") or ""))
            quote = str(reference.get("quote") or "").strip()
            if row and len(quote) >= 4 and quote in row["text"]:
                supports.append(row)
                support_quotes.append({"event_id": row["event"]["event_id"], "quote": quote[:300]})
        unique_support = {row["operation"]: row for row in supports}
        support_days = {row["date"] for row in unique_support.values() if row["date"]}
        if (len(unique_support) < SEMANTIC_MIN_EVIDENCE or
                len(support_days) < SEMANTIC_MIN_DAYS):
            rejected.append({"topic": topic, "reason": "insufficient_verified_evidence"})
            continue
        if kind == "preference" and _has_explicit_conflict(sem, topic):
            rejected.append({"topic": topic, "reason": "explicit_memory_precedence"})
            continue
        opposing_ids = []
        for reference in candidate.get("oppose") or []:
            if not isinstance(reference, dict):
                continue
            row = by_id.get(str(reference.get("event_id") or ""))
            quote = str(reference.get("quote") or "").strip()
            if row and len(quote) >= 4 and quote in row["text"]:
                opposing_ids.append(row["event"]["event_id"])
        memory_key = f"model_{kind}:{_norm_task_key(topic, target)}"
        evidence_ids = [row["event"]["event_id"] for row in unique_support.values()]
        confidence = min(.9, .6 + .05 * len(unique_support))
        result = sem.store.upsert_semantic(
            memory_key,
            {"topic": topic, "polarity": polarity, "summary": summary,
             "support_quotes": support_quotes, "extraction": "model_assisted"},
            memory_type=kind_map[kind], certainty="inferred", confidence=confidence,
            target_context_id=target, evidence_ids=evidence_ids,
            opposing_evidence_ids=sorted(set(opposing_ids)), replace_evidence=True,
        )
        accepted.append({"memory_id": result["memory_id"], "topic": topic,
                         "evidence_count": len(evidence_ids), "evidence_days": len(support_days)})
    sem.store.set_runtime("model_distill_status", {
        "status": "ok", "last_seq": max_seq, "accepted": len(accepted),
        "rejected": len(rejected), "updated_at": now_iso(),
    })
    if accepted:
        sem._export()
    return {"status": "ok", "last_seq": max_seq, "accepted": accepted, "rejected": rejected}


def schedule_model_assisted_distillation(epi: EpisodicMemory) -> None:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get(
            "OFFERCLAW_MEMORY_MODEL_DISTILL", "1") != "1":
        return
    _MODEL_DISTILL_POOL.submit(
        distill_free_text_with_model, epi, SemanticMemory(str(epi.store.base_dir)))


ADJUSTMENTS_KEY = "daily_adjustments"


def record_reflection(epi: EpisodicMemory, reflection: dict) -> dict:
    event = {
        "kind": "reflection", "actor": "user", "source": "reflection_service",
        "reflection_id": reflection.get("reflection_id", ""),
        "reflection_kind": reflection.get("kind", "daily"), "date": reflection.get("date", ""),
        "date_from": reflection.get("date_from", reflection.get("date", "")),
        "date_to": reflection.get("date_to", reflection.get("date", "")),
        "main_tag": reflection.get("main_tag", ""),
        "deviation_score": int(reflection.get("deviation_score", 0) or 0),
        "completed": list(reflection.get("completed", []) or []),
        "incomplete": list(reflection.get("incomplete", []) or []),
        "incomplete_items": list(reflection.get("incomplete_items", []) or []),
        "blockers": list(reflection.get("blockers", []) or []),
        "next_day_suggestion": reflection.get("next_day_suggestion", ""),
        "skill_evidence": list(reflection.get("skill_evidence", []) or []),
        "evidence_candidates": list(reflection.get("evidence_candidates", []) or []),
        "source_log_ids": list(reflection.get("source_log_ids", []) or []),
        "summary_path": reflection.get("summary_path", ""),
        "content_hash": reflection.get("content_hash", ""),
        "source_status": reflection.get("source_status", "valid"),
        "operation_id": (f"reflection:{reflection.get('reflection_id')}" if reflection.get("reflection_id") else None),
        "entity_type": "reflection", "entity_id": reflection.get("reflection_id") or None,
        "business_date": reflection.get("date") or None,
    }
    return epi.append(event)


def _norm_task_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"^\s*(?:[-*+] |\d+[.、]\s*)", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _norm_task_key(text: str, target_context_id: str = "global") -> str:
    normalized = _norm_task_text(text)
    if not normalized: return ""
    digest = hashlib.sha256(f"{target_context_id}\0{normalized}".encode("utf-8")).hexdigest()
    return f"legacy_{digest[:24]}"


def _task_identity(item: Any, target_context_id: str) -> tuple[str, str]:
    if isinstance(item, dict):
        for field in ("task_id", "task_series_id", "gap_id", "skill_id"):
            if item.get(field): return f"{field}:{item[field]}", str(item.get("text") or item[field])
        text = str(item.get("text") or "")
    else:
        text = str(item or "")
    return _norm_task_key(text, target_context_id), text


def distill_reflections_to_semantic(epi: EpisodicMemory, sem: SemanticMemory,
                                    recent_n: int = 5, streak: int = 3) -> dict:
    target_context_id = sem.store.active_goal_id()
    eligible = [event for event in epi.all() if _trusted(event) and event.get("kind") == "reflection"
                and event.get("reflection_kind", "daily") == "daily"
                and event.get("source_status", "valid") == "valid" and event.get("date")
                and event.get("target_context_id") in {target_context_id, "global", None}]
    by_date: dict[str, dict] = {}
    for event in eligible:
        if event["date"] not in by_date or int(event.get("seq", 0)) > int(by_date[event["date"]].get("seq", 0)):
            by_date[event["date"]] = event
    dates = sorted(by_date)[-recent_n:]
    recent = [by_date[date] for date in dates]
    rules: list[dict[str, Any]] = []
    if len(recent) >= streak:
        tail = recent[-streak:]
        parsed = [dt.date.fromisoformat(event["date"]) for event in tail]
        consecutive = all((b - a).days == 1 for a, b in zip(parsed, parsed[1:]))
        if consecutive and all(int(event.get("deviation_score", 0)) >= 50 for event in tail):
            rules.append({"pattern": "high_deviation_streak",
                          "detail": f"最近 {streak} 个连续记录日偏离度均不低于 50；下次生成计划时应减少任务量并保留一个主线产出。",
                          "since": tail[0]["date"], "evidence_ids": [e["event_id"] for e in tail]})
    counter: dict[str, dict[str, Any]] = {}
    for event in recent:
        seen: set[str] = set()
        items = event.get("incomplete_items") or event.get("incomplete") or []
        target = str(event.get("target_context_id") or "global")
        for item in items:
            key, sample = _task_identity(item, target)
            if not key or key in seen: continue
            seen.add(key)
            slot = counter.setdefault(key, {"dates": set(), "sample": sample, "events": []})
            slot["dates"].add(event["date"]); slot["events"].append(event["event_id"])
    for key, slot in counter.items():
        if len(slot["dates"]) >= streak:
            rules.append({"pattern": f"recurring_incomplete:{key}",
                          "detail": f"“{slot['sample']}”在最近 {len(recent)} 个记录日中有 {len(slot['dates'])} 天未完成；下次生成计划时应拆细、减量或前置。",
                          "since": min(slot["dates"]), "evidence_ids": slot["events"]})
    evidence_ids = sorted({eid for rule in rules for eid in rule.get("evidence_ids", [])})
    if rules:
        sem.store.upsert_semantic(ADJUSTMENTS_KEY, {"rules": rules, "updated_at": now_iso()},
                                  memory_type="reflection_adjustment", certainty="inferred",
                                  confidence=.8, evidence_ids=evidence_ids,
                                  target_context_id=target_context_id)
    else:
        existing = sem.store.get_semantic(ADJUSTMENTS_KEY,
                                          target_context_id=target_context_id)
        if existing and existing.get("lifecycle") == "active":
            sem.store.set_lifecycle("semantic", existing["memory_id"], "archived")
    sem._export()
    return {"distilled": True, "n_reflections": len(recent), "rules": rules}


def get_active_adjustments(sem: SemanticMemory,
                           target_context_id: str | None = None) -> list[str]:
    target = target_context_id or sem.store.active_goal_id()
    row = sem.store.get_semantic(ADJUSTMENTS_KEY, target_context_id=target)
    data = row["value"] if row and row.get("lifecycle") == "active" else {}
    return [rule.get("detail", "") for rule in data.get("rules", []) if rule.get("detail")]


def record_career_flow_run(epi: EpisodicMemory, *, jd_title: str, status: str,
                           direction: str, status_code: str = "unknown") -> dict:
    code = match_status_code(status_code if status_code != "unknown" else status).value
    return epi.append({"kind": "career_flow_run", "jd_title": jd_title or "",
                       "status": status or "", "status_code": code, "direction": direction or "",
                       "actor": "system", "source": "career_flow"})


def distill_procedural_sops(epi: EpisodicMemory, proc: ProceduralMemory,
                            min_support: int = 2) -> dict:
    """Create reviewable candidates only; match suitability is not SOP success."""
    runs = [e for e in epi.all() if _trusted(e) and e.get("kind") == "career_flow_run"]
    by_direction: dict[str, dict[str, int]] = {}
    for event in runs:
        direction = str(event.get("direction") or "").strip()
        if not direction: continue
        slot = by_direction.setdefault(direction, {"fit": 0, "total": 0})
        slot["total"] += 1
        if match_status_code(event.get("status_code") or event.get("status")) == MatchStatusCode.SUITABLE:
            slot["fit"] += 1
    candidates = []
    for direction, support in by_direction.items():
        if support["fit"] >= min_support:
            proc.add(f"apply_direction:{direction}",
                     body=f"评估“{direction}”岗位时复用已验证的项目证据，并在执行后记录实际结果。",
                     trigger={"direction": direction}, lifecycle="candidate")
            candidates.append(direction)
    return {"distilled": True, "directions": candidates, "candidates": candidates, "support": by_direction}


def record_sop_outcome(proc: ProceduralMemory, name: str, event_id: str, outcome: str,
                       business_date: str = "", weight: float = 1.0) -> dict:
    if outcome not in {"suggested", "adopted", "success", "failure"}:
        raise ValueError("invalid SOP outcome")
    result = proc.store.record_sop_outcome(name, event_id, outcome, business_date, weight)
    with _EXPORT_LOCK: proc._export()
    return result


def _sop_matches(trigger: dict[str, Any], context: dict[str, Any]) -> bool:
    if trigger.get("global") is True: return True
    excluded = trigger.get("exclude") or {}
    if isinstance(excluded, dict):
        for key, unwanted in excluded.items():
            actual = context.get(key)
            if isinstance(actual, (list, tuple, set)):
                if str(unwanted) in {str(value) for value in actual}: return False
            elif str(actual or "") == str(unwanted): return False
    for key, wanted in trigger.items():
        if key in {"exclude", "global"}: continue
        if key == "text_contains":
            if str(wanted) not in " ".join(str(value) for value in context.values()): return False
            continue
        actual = context.get(key)
        if isinstance(actual, (list, tuple, set)):
            if str(wanted) not in {str(value) for value in actual}: return False
        elif str(actual or "") != str(wanted): return False
    return True


def recall_active_sops(proc: ProceduralMemory, context: str | dict[str, Any] = "",
                       limit: int = 3) -> list[dict[str, Any]]:
    """Return applicable SOPs with their match reason and execution evidence."""
    if isinstance(context, str):
        structured = {"text": context}
        for sop in proc.list():
            direction = sop.get("trigger_conditions", {}).get("direction")
            if direction and direction in context: structured["direction"] = direction
    else:
        structured = dict(context or {})
    goal_id = str(structured.get("target_context_id") or proc.store.active_goal_id())
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    ranked = []
    for sop in proc.store.list_sops(include_inactive=False):
        if sop["target_context_id"] not in {"global", goal_id}: continue
        last = sop.get("last_evidence_at")
        if last:
            try: age = max(0, (now - dt.datetime.fromisoformat(last)).days)
            except ValueError: age = 999
            if age >= SOP_ARCHIVE_DAYS:
                proc.store.set_lifecycle("sop", sop["sop_id"], "archived")
                continue
            factor = .5 ** (age / SOP_HALF_LIFE_DAYS)
            confidence = (1 + sop["success_weight"] * factor) / (
                2 + (sop["success_weight"] + sop["failure_weight"]) * factor)
        else:
            confidence = sop["confidence"]
        if not _sop_matches(sop["trigger_conditions"], structured):
            continue
        trigger = sop["trigger_conditions"]
        reason = ("通用 SOP" if trigger.get("global") is True else
                  "；".join(f"{key}={value}" for key, value in trigger.items()
                            if key != "exclude"))
        ranked.append((confidence, {"sop_id": sop["sop_id"], "name": sop["name"],
                                    "body": sop["body"], "confidence": confidence,
                                    "match_reason": reason,
                                    "target_context_id": sop["target_context_id"],
                                    "evidence": sop.get("evidence", [])}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in ranked[:max(1, limit)]]


def get_active_sops(proc: ProceduralMemory, context: str | dict[str, Any] = "",
                    limit: int = 3) -> list[str]:
    return [item["body"] for item in recall_active_sops(proc, context, limit)]


def active_semantic_memories(sem: SemanticMemory, *, target_context_id: str | None = None) -> list[dict]:
    goal_id = target_context_id or sem.store.active_goal_id()
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    out = []
    for item in sem.store.list_semantic(target_context_id=goal_id):
        if item["certainty"] == "inferred":
            try: age = max(0, (now - dt.datetime.fromisoformat(item["last_supported_at"])).days)
            except ValueError: age = 999
            if age >= INFERRED_ARCHIVE_DAYS:
                sem.store.set_lifecycle("semantic", item["memory_id"], "archived")
                continue
            half_life = (STAGE_INTEREST_HALF_LIFE_DAYS
                         if item["memory_type"] == "stage_interest"
                         else INFERRED_HALF_LIFE_DAYS)
            item["effective_confidence"] = item["confidence"] * (.5 ** (age / half_life))
        else: item["effective_confidence"] = item["confidence"]
        out.append(item)
    return out


def build_memory_context(*, purpose: str = "advice",
                         context: dict[str, Any] | None = None,
                         semantic_limit: int = 12,
                         reflection_limit: int = 3) -> dict[str, Any]:
    """Build the single versioned memory contract used by planning agents."""
    structured = dict(context or {})
    store = MemoryStore(_configured_base(None))
    goal_id = str(structured.get("target_context_id") or store.active_goal_id())
    semantic = active_semantic_memories(SemanticMemory(str(store.base_dir)),
                                        target_context_id=goal_id)
    semantic.sort(key=lambda item: (float(item.get("effective_confidence", 0)),
                                    str(item.get("updated_at", ""))), reverse=True)
    proc = ProceduralMemory(str(store.base_dir))
    structured["target_context_id"] = goal_id
    sop_details = recall_active_sops(proc, structured, limit=3)
    return {
        "purpose": purpose,
        "target_context_id": goal_id,
        "goal": next((item for item in store.list_goals()
                      if item["context_id"] == goal_id), None),
        "semantic": [{key: item.get(key) for key in (
            "memory_id", "memory_key", "memory_type", "value", "certainty",
            "effective_confidence", "version", "updated_at")}
            for item in semantic[:max(1, semantic_limit)]],
        "adjustments": get_active_adjustments(SemanticMemory(str(store.base_dir)), goal_id),
        "sops": [item["body"] for item in sop_details],
        "sop_details": sop_details,
        "recent_reflections": recent_reflection_lessons(
            EpisodicMemory(str(store.base_dir)), reflection_limit),
        "generated_at": now_iso(),
    }


def switch_goal_context(name: str) -> dict[str, Any]:
    if not str(name or "").strip(): raise ValueError("goal name cannot be empty")
    store = MemoryStore(_configured_base(None)); previous = store.active_goal_id()
    result = store.switch_goal(str(name).strip())
    record_business_event("goal_context_switched", {"previous_context_id": previous, **result},
                          actor="user", source="memory_service", entity_type="goal",
                          entity_id=result["context_id"], target_context_id=result["context_id"])
    return result


def activate_goal_context(context_id: str) -> dict[str, Any]:
    store = MemoryStore(_configured_base(None)); previous = store.active_goal_id()
    result = store.activate_goal(context_id)
    record_business_event("goal_context_switched",
                          {"previous_context_id": previous, **result},
                          actor="user", source="memory_service", entity_type="goal",
                          entity_id=context_id, target_context_id=context_id)
    return result


def recent_reflection_lessons(epi: EpisodicMemory, n: int = 3) -> list[str]:
    goal_id = epi.store.active_goal_id()
    events = [event for event in epi.all() if _trusted(event) and
              event.get("target_context_id") in {goal_id, "global", None}]
    reflections = [event for event in events if event.get("kind") == "reflection"
                   and event.get("reflection_kind", "daily") == "daily"
                   and event.get("source_status", "valid") == "valid"]
    by_date = {event.get("date"): event for event in reflections if event.get("date")}
    lessons: list[tuple[str, int, str]] = []
    for event in by_date.values():
        incomplete = event.get("incomplete") or []; blockers = event.get("blockers") or []
        suggestion = str(event.get("next_day_suggestion") or "").strip()
        deviation = int(event.get("deviation_score") or 0)
        detail = [f"偏离度 {deviation}"] if deviation > 0 else []
        if incomplete: detail.append(f"未完成 {len(incomplete)} 项（如：{str(incomplete[0])[:40]}）")
        if blockers: detail.append(f"阻碍：{str(blockers[0])[:60]}")
        if suggestion: detail.append(f"复盘生成建议：{suggestion[:80]}")
        if detail:
            lessons.append((event["date"], int(event.get("seq") or 0), "；".join(detail)))
    for event in events:
        if event.get("kind") == "daily_log_recorded" and str(event.get("notes") or "").strip():
            lessons.append((str(event.get("business_date") or event.get("date") or ""),
                            int(event.get("seq") or 0),
                            f"用户每日思考：{str(event['notes']).strip()[:160]}"))
        elif event.get("kind") == "application_review_recorded":
            snapshot = epi.store.get_snapshot(str(event.get("snapshot_id") or ""))
            content = str((snapshot or {}).get("content") or "")
            body = content.split("---", 2)[-1].strip() if content.startswith("---") else content
            if body:
                lessons.append((str(event.get("business_date") or ""), int(event.get("seq") or 0),
                                f"投递复盘（{event.get('company', '')} {event.get('position', '')}）：{body[-200:]}"))
    lessons.sort(key=lambda item: (item[0], item[1]))
    return [f"{date} {text}" for date, _seq, text in lessons[-max(1, n):]]


def memory_health(base_dir: str | None = None) -> dict[str, Any]:
    store = MemoryStore(_configured_base(base_dir)); stats = store.stats()
    with store._connect() as conn:
        stats.update({
            "quarantined_migrations": conn.execute("SELECT COUNT(*) FROM migration_map WHERE status='quarantined'").fetchone()[0],
            "pending_operations": conn.execute("SELECT COUNT(*) FROM operation_journal WHERE status='pending'").fetchone()[0],
            "failed_operations": conn.execute("SELECT COUNT(*) FROM operation_journal WHERE status IN ('failed','conflict')").fetchone()[0],
        })
    stats["model_distill"] = store.get_runtime("model_distill_status", {"status": "idle"})
    return stats
