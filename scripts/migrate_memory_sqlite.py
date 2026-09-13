# -*- coding: utf-8 -*-
"""Report and migrate OfferClaw personal memory into SQLite.

Run without flags for a read-only report.  ``--apply`` creates a timestamped
backup, imports legacy layer files, and backfills traceable current snapshots.
Every backfill uses a deterministic operation id, so reruns are idempotent.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEMORY_DIR = ROOT / "logs" / "memory"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _legacy_source() -> Path:
    backups = sorted((MEMORY_DIR / "backups").glob("pre_sqlite_*/episodic.jsonl"))
    return backups[0] if backups else MEMORY_DIR / "episodic.jsonl"


def legacy_report() -> dict:
    path = _legacy_source()
    kinds: collections.Counter[str] = collections.Counter()
    ids: collections.Counter[str] = collections.Counter()
    invalid = synthetic = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                invalid += 1
                continue
            kinds[str(row.get("kind") or "missing")] += 1
            if row.get("id"):
                ids[str(row["id"])] += 1
            raw = json.dumps(row, ensure_ascii=False).casefold()
            if any(token in raw for token in ("unittest ai", '"traced"', '"traffic_origin": "test"')):
                synthetic += 1
    duplicates = {key: count for key, count in ids.items() if count > 1}
    return {
        "source": str(path), "exists": path.exists(), "events": sum(kinds.values()),
        "kinds": dict(kinds), "invalid_lines": invalid,
        "duplicate_id_groups": len(duplicates),
        "duplicate_excess": sum(value - 1 for value in duplicates.values()),
        "suspected_synthetic": synthetic,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "",
    }


def _append_snapshot(epi, *, source_key: str, source_kind: str, title: str,
                     content: str, source_path: str, business_date: str = "") -> bool:
    if not content.strip():
        return False
    operation_id = f"backfill:{source_key}:{_digest(content)}"
    if epi.store.get_event_by_operation(operation_id):
        return False
    snapshot = epi.store.put_snapshot(content, media_type="text/markdown", source_path=source_path)
    epi.append({
        "kind": "historical_snapshot_imported", "actor": "import",
        "source": "memory_migration", "traffic_origin": "organic",
        "operation_id": operation_id,
        "entity_type": source_kind, "entity_id": source_key,
        "business_date": business_date or None, "source_kind": source_kind,
        "title": title, "snapshot_id": snapshot["snapshot_id"],
        "content_hash": snapshot["content_hash"], "source_path": source_path,
        "historical_snapshot": True,
    }, export=False, index=False)
    return True


def apply_migration() -> dict:
    from memory_layers import EpisodicMemory, ProceduralMemory, SemanticMemory
    epi = EpisodicMemory(str(MEMORY_DIR))
    SemanticMemory(str(MEMORY_DIR))
    ProceduralMemory(str(MEMORY_DIR))
    backfilled = 0
    errors: list[dict[str, str]] = []

    profile = ROOT / "user_profile.md"
    if profile.exists():
        backfilled += _append_snapshot(
            epi, source_key="user_profile", source_kind="profile", title="当前用户画像",
            content=profile.read_text(encoding="utf-8"), source_path="user_profile.md")

    try:
        from applications_store import application_fact_views
        for row in application_fact_views():
            content = json.dumps(row, ensure_ascii=False, indent=2)
            backfilled += _append_snapshot(
                epi, source_key=str(row.get("application_id") or _digest(content)[:16]),
                source_kind="application", title=f"{row.get('company', '')} · {row.get('position', '')}",
                content=content, source_path="applications.md", business_date=str(row.get("date") or ""))
    except Exception as exc:
        errors.append({"source": "applications.md", "error": str(exc)[:500]})

    try:
        from reflection_memory import daily_execution_documents, reflection_documents
        for doc in daily_execution_documents() + reflection_documents():
            if doc.get("source_status") != "valid":
                continue
            backfilled += _append_snapshot(
                epi, source_key=str(doc["id"]), source_kind=str(doc["source_type"]),
                title=str(doc.get("title") or "历史记录"), content=str(doc.get("text") or ""),
                source_path=str(doc.get("path") or ""), business_date=str(doc.get("date_to") or ""))
    except Exception as exc:
        errors.append({"source": "daily_log/reflections", "error": str(exc)[:500]})

    plans = ROOT / "plans"
    for path in sorted(plans.glob("plan_*.md")) if plans.exists() else []:
        try:
            content = path.read_text(encoding="utf-8")
            backfilled += _append_snapshot(
                epi, source_key=path.name, source_kind="plan", title=path.stem,
                content=content, source_path=str(path.relative_to(ROOT)))
        except Exception as exc:
            errors.append({"source": str(path.relative_to(ROOT)), "error": str(exc)[:500]})

    epi._export()
    return {"status": "partial" if errors else "ok", "backfilled": backfilled,
            "database": str(epi.store.path), "errors": errors,
            "stats": epi.store.stats()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default=str(MEMORY_DIR / "migration_report.json"))
    args = parser.parse_args()
    report = legacy_report()
    if args.apply:
        report["migration"] = apply_migration()
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
