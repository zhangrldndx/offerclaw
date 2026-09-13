# -*- coding: utf-8 -*-
from __future__ import annotations

import json

import rag_ingest
import wechat_index


class _Collection:
    def __init__(self):
        self.deleted = []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


def test_real_index_keeps_previous_manifest_when_replace_fails(tmp_path, monkeypatch):
    manifest_path = tmp_path / "state" / "manifest.json"
    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_MANIFEST", str(manifest_path))
    collection = _Collection()
    monkeypatch.setattr(wechat_index, "_collection", lambda: collection)
    state = {"sha256": "hash-v1", "text": "真实项目第一版"}

    def bridge(operation, payload):
        if operation == "source.manifest":
            return {"sources": [{
                "source_id": "interview_story_bank.md", "source_type": "project_context",
                "sha256": state["sha256"], "size": len(state["text"]), "mtime_ns": 1,
            }]}
        if operation == "source.read":
            return {"source_id": "interview_story_bank.md", **state}
        raise AssertionError(operation)

    monkeypatch.setattr(
        rag_ingest, "replace_source_content",
        lambda source_id, content, passed_collection, source_type: {
            "status": "ok", "chunks": 2, "stale_removed": 1,
        },
    )
    first = wechat_index.sync_real_index(bridge)
    assert first["status"] == "ok"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sources"][
        "interview_story_bank.md"
    ]["sha256"] == "hash-v1"

    state.update({"sha256": "hash-v2", "text": "真实项目第二版"})

    def fail_replace(*_args, **_kwargs):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(rag_ingest, "replace_source_content", fail_replace)
    second = wechat_index.sync_real_index(bridge)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert second["status"] == "partial"
    assert second["old_versions_preserved"] is True
    assert "interview_story_bank.md" in second["failures"]
    assert persisted["sources"]["interview_story_bank.md"]["sha256"] == "hash-v1"


def test_real_index_limits_refresh_without_marking_unprocessed_sources_current(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_MANIFEST", str(manifest_path))
    monkeypatch.setattr(wechat_index, "_collection", _Collection)
    rows = [
        {"source_id": "interview_story_bank.md", "source_type": "project_context", "sha256": "h1"},
        {"source_id": "learning_resources/rag.md", "source_type": "learning_resource", "sha256": "h2"},
    ]

    def bridge(operation, payload):
        if operation == "source.manifest":
            return {"sources": rows}
        row = next(item for item in rows if item["source_id"] == payload["source_id"])
        return {"source_id": row["source_id"], "sha256": row["sha256"], "text": "正文"}

    monkeypatch.setattr(
        rag_ingest, "replace_source_content",
        lambda *_args, **_kwargs: {"status": "ok", "chunks": 1, "stale_removed": 0},
    )
    result = wechat_index.sync_real_index(bridge, max_changed=1)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))["sources"]
    assert result["status"] == "partial"
    assert result["processed"] == 1 and result["pending"] == 1
    assert len(persisted) == 1


def test_real_index_rebuilds_manifested_source_when_collection_is_empty(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "sources": {
            "interview_story_bank.md": {
                "source_id": "interview_story_bank.md",
                "source_type": "project_context",
                "sha256": "same-hash",
            },
        },
    }), encoding="utf-8")

    class EmptyCollection(_Collection):
        def count(self):
            return 0

        def get(self, **_kwargs):
            return {"metadatas": []}

    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_MANIFEST", str(manifest_path))
    monkeypatch.setattr(wechat_index, "_collection", EmptyCollection)

    def bridge(operation, payload):
        row = {
            "source_id": "interview_story_bank.md",
            "source_type": "project_context",
            "sha256": "same-hash",
        }
        if operation == "source.manifest":
            return {"sources": [row]}
        return {**row, "text": "正文"}

    calls = []
    monkeypatch.setattr(
        rag_ingest, "replace_source_content",
        lambda *args, **_kwargs: calls.append(args[0]) or {
            "status": "ok", "chunks": 1, "stale_removed": 0,
        },
    )

    result = wechat_index.sync_real_index(bridge)
    assert result["status"] == "ok"
    assert result["changed"] == 1
    assert calls == ["interview_story_bank.md"]


def test_real_index_collection_name_does_not_depend_on_embedding_cache_path():
    assert wechat_index.DEFAULT_COLLECTION_NAME == "offerclaw_wechat_real_bge_base_zh_768"


def test_index_status_reports_manifest_without_chunks_as_degraded(tmp_path, monkeypatch):
    index_dir = tmp_path / "chroma"
    index_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "updated_at": "2026-09-13T00:00:00+08:00",
        "sources": {"interview_story_bank.md": {"sha256": "hash"}},
    }), encoding="utf-8")

    class EmptyCollection(_Collection):
        def count(self):
            return 0

        def get(self, **_kwargs):
            return {"metadatas": []}

    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("OFFERCLAW_REAL_INDEX_MANIFEST", str(manifest_path))
    monkeypatch.setattr(wechat_index, "_collection", EmptyCollection)

    status = wechat_index.index_status()
    assert status["status"] == "degraded"
    assert status["source_count"] == 1
    assert status["covered_source_count"] == 0
    assert status["chunk_count"] == 0
