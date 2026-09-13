# -*- coding: utf-8 -*-
"""Disposable WSL index synchronized from the authoritative Windows bridge."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

from io_utils import atomic_write_json
from rag_source_policy import evidence_allowed, infer_owner_scope, rag_source_excluded


BridgeCall = Callable[[str, dict[str, Any]], dict[str, Any]]
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / ".offerclaw" / "wechat-real-chroma"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / ".offerclaw" / "wechat-real-index.json"
DEFAULT_COLLECTION_NAME = "offerclaw_wechat_real_bge_base_zh_768"


def _paths() -> tuple[Path, Path]:
    index = Path(os.environ.get("OFFERCLAW_REAL_INDEX_DIR") or DEFAULT_INDEX_DIR)
    manifest = Path(os.environ.get("OFFERCLAW_REAL_INDEX_MANIFEST") or DEFAULT_MANIFEST_PATH)
    return index, manifest


def _eligible(row: dict[str, Any]) -> bool:
    source_id = str(row.get("source_id") or "")
    source_type = str(row.get("source_type") or "doc")
    owner = infer_owner_scope(source_type, source_id)
    return owner in {"personal", "curated"} and not rag_source_excluded(source_type, source_id)


def _collection():
    import chromadb

    index_dir, _ = _paths()
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(index_dir, 0o700)
    except OSError:
        pass
    client = chromadb.PersistentClient(path=str(index_dir))
    name = os.environ.get("OFFERCLAW_REAL_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)


def _indexed_source_ids(collection: Any) -> set[str] | None:
    """Return actual source coverage, or None for lightweight test doubles."""
    try:
        if int(collection.count()) == 0:
            return set()
        payload = collection.get(include=["metadatas"])
        return {
            str(item.get("source") or "")
            for item in (payload.get("metadatas") or [])
            if isinstance(item, dict) and item.get("source")
        }
    except (AttributeError, TypeError, ValueError):
        return None


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def sync_real_index(bridge: BridgeCall, *, max_changed: int | None = None) -> dict[str, Any]:
    """Refresh changed eligible sources, writing the manifest only after each safe replace."""
    _, manifest_path = _paths()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_manifest(manifest_path)
    previous = current.get("sources") if isinstance(current.get("sources"), dict) else {}
    remote = bridge("source.manifest", {}).get("sources") or []
    wanted = {row["source_id"]: row for row in remote if _eligible(row)}
    collection = _collection()
    indexed_sources = _indexed_source_ids(collection)
    changed = [row for key, row in wanted.items()
               if (previous.get(key) or {}).get("sha256") != row.get("sha256")
               or (indexed_sources is not None and key not in indexed_sources)]
    removed = sorted(set(previous) - set(wanted))
    limited = max_changed is not None and len(changed) > max_changed
    selected = changed if max_changed is None else changed[:max_changed]
    completed: dict[str, dict[str, Any]] = dict(previous)
    sources: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    for source_id in removed:
        try:
            collection.delete(where={"source": source_id})
            completed.pop(source_id, None)
        except Exception as exc:
            failures[source_id] = str(exc)[:200]

    from rag_ingest import replace_source_content
    for row in selected:
        source_id = row["source_id"]
        try:
            response = bridge("source.read", {"source_id": source_id})
            if response.get("sha256") != row.get("sha256"):
                raise RuntimeError("source changed during index refresh")
            with contextlib.redirect_stdout(io.StringIO()):
                result = replace_source_content(
                    source_id, response.get("text") or "", collection,
                    source_type=str(row.get("source_type") or "doc"),
                )
            if result.get("status") != "ok":
                raise RuntimeError(str(result.get("status") or "index replace failed"))
            completed[source_id] = row
            sources[source_id] = {
                "status": "ok", "chunks": result.get("chunks", 0),
                "stale_removed": result.get("stale_removed", 0),
            }
        except Exception as exc:
            failures[source_id] = str(exc)[:200]

    stamp = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    atomic_write_json(str(manifest_path), {
        "schema_version": "offerclaw.wechat-index-manifest.v1",
        "updated_at": stamp, "sources": completed,
    })
    try:
        os.chmod(manifest_path, 0o600)
    except OSError:
        pass
    pending = max(0, len(changed) - len(selected))
    status = "partial" if failures or limited else "ok"
    return {
        "status": status, "updated_at": stamp, "changed": len(changed),
        "processed": len(selected), "pending": pending, "removed": len(removed),
        "sources": sources, "failures": failures,
        "old_versions_preserved": bool(failures),
    }


def search_real_index(question: str, route_keys: list[str], *, top_k: int = 5) -> dict[str, Any]:
    """Retrieve local evidence without answer synthesis or any remote model call."""
    from rag_query import retrieve

    collection = _collection()
    candidates = retrieve(question, collection, top_k=max(12, top_k * 4))
    allowed_routes = [key.split(".", 1)[0] for key in route_keys]
    accepted = []
    max_distance = float(os.environ.get("OFFERCLAW_WECHAT_RAG_MAX_DISTANCE", "0.75"))
    for item in candidates:
        source = str(item.get("source") or "")
        source_type = ""
        try:
            found = collection.get(where={"source": source}, include=["metadatas"])
            source_type = str(((found.get("metadatas") or [{}])[0]).get("source_type") or "")
        except Exception:
            pass
        if item.get("distance") is None or float(item["distance"]) > max_distance:
            continue
        if allowed_routes and not any(
                evidence_allowed(route, source_type, "", source) for route in allowed_routes):
            continue
        accepted.append({
            "source": source, "title": item.get("title") or "",
            "distance": round(float(item["distance"]), 4),
            "text": str(item.get("document") or "")[:1200],
        })
        if len(accepted) >= top_k:
            break
    return {"status": "ok", "in_kb": bool(accepted), "evidence": accepted}


def index_status() -> dict[str, Any]:
    index_dir, manifest_path = _paths()
    manifest = _load_manifest(manifest_path)
    expected = set(manifest.get("sources") or {})
    indexed: set[str] | None = None
    chunk_count = 0
    if index_dir.is_dir() and manifest:
        try:
            collection = _collection()
            chunk_count = int(collection.count())
            indexed = _indexed_source_ids(collection)
        except Exception:
            indexed = None
    covered = len(expected & indexed) if indexed is not None else 0
    healthy = bool(expected) and indexed is not None and expected <= indexed and chunk_count > 0
    return {
        "status": "healthy" if healthy else ("degraded" if manifest else "uninitialized"),
        "updated_at": manifest.get("updated_at", ""),
        "source_count": len(expected), "covered_source_count": covered,
        "chunk_count": chunk_count,
    }


__all__ = ["index_status", "search_real_index", "sync_real_index"]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", required=True)
    parser.parse_args()
    from wechat_dispatch import call_bridge
    print(json.dumps(sync_real_index(call_bridge), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
