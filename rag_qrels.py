"""Strict loader for retrieval qrels overlays used by offline evaluation.

The overlay is deliberately separate from the historical benchmark JSON.  It
records reviewer-verified answer spans and the *current* stable chunk that
contains each span.  Retrieval output is never used as a source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "rag-qrels-overlay-v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OUTCOMES = {"accepted", "partial", "unsupported"}
_RELEVANCE = {"direct", "supporting"}
_EVIDENCE_SCOPES = {"child", "parent_expand_required"}


class QrelsValidationError(ValueError):
    """Raised when an overlay violates the qrels data contract."""


def normalize_answer_span(text: str) -> str:
    """Return the canonical representation used for answer-span hashes."""

    if not isinstance(text, str):
        raise QrelsValidationError("answer span must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def answer_span_hash(text: str) -> str:
    normalized = normalize_answer_span(text)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QrelsValidationError(message)


def validate_qrels_overlay(
    payload: Any,
    *,
    expected_query_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate and return an overlay payload.

    ``unsupported`` questions are intentionally allowed to have no targets:
    forcing a target in that case would turn a source-title coincidence into a
    false gold label.  Accepted and partial questions must have at least one
    reviewer-verified direct span.
    """

    _require(isinstance(payload, dict), "overlay root must be an object")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(isinstance(payload.get("reviewer_id"), str) and payload["reviewer_id"].strip(), "reviewer_id is required")
    _require(payload.get("base_set") == "tests/rag_bench_paraphrase_set.json", "base_set must reference the immutable 52-question set")
    index = payload.get("index")
    _require(isinstance(index, dict), "index must be an object")
    _require(isinstance(index.get("collection"), str) and index["collection"].strip(), "index.collection is required")
    _require(isinstance(index.get("count"), int) and index["count"] >= 0, "index.count must be non-negative")
    if "fingerprint" in index:
        _require(isinstance(index["fingerprint"], str) and _SHA256_RE.fullmatch(index["fingerprint"]) is not None, "index.fingerprint is invalid")
    _require(isinstance(payload.get("items"), list), "items must be a list")

    seen: set[str] = set()
    for item_index, item in enumerate(payload["items"]):
        prefix = f"items[{item_index}]"
        _require(isinstance(item, dict), f"{prefix} must be an object")
        query_id = item.get("query_id")
        _require(isinstance(query_id, str) and query_id.strip(), f"{prefix}.query_id is required")
        _require(query_id not in seen, f"duplicate query_id: {query_id}")
        seen.add(query_id)
        outcome = item.get("review_outcome")
        _require(outcome in _OUTCOMES, f"{query_id}: invalid review_outcome")
        _require(isinstance(item.get("review_note"), str) and item["review_note"].strip(), f"{query_id}: review_note is required")
        targets = item.get("relevant_targets")
        _require(isinstance(targets, list), f"{query_id}: relevant_targets must be a list")
        if outcome == "unsupported":
            _require(not targets, f"{query_id}: unsupported questions must not invent targets")
            continue
        _require(bool(targets), f"{query_id}: {outcome} questions need relevant targets")

        direct_count = 0
        target_keys: set[tuple[str, str, str]] = set()
        for target_index, target in enumerate(targets):
            tp = f"{query_id}.relevant_targets[{target_index}]"
            _require(isinstance(target, dict), f"{tp} must be an object")
            source = target.get("source")
            _require(isinstance(source, str) and source.endswith(".md"), f"{tp}.source must be a Markdown basename")
            _require(Path(source).name == source and ".." not in source, f"{tp}.source must not contain a path")
            heading_path = target.get("heading_path")
            _require(
                isinstance(heading_path, list)
                and heading_path
                and all(isinstance(part, str) and part.strip() for part in heading_path),
                f"{tp}.heading_path must be a non-empty string list",
            )
            chunk_id = target.get("chunk_id")
            _require(isinstance(chunk_id, str) and chunk_id.strip(), f"{tp}.chunk_id is required")
            relevance = target.get("relevance")
            _require(relevance in _RELEVANCE, f"{tp}.relevance is invalid")
            direct_count += int(relevance == "direct")
            excerpt = target.get("evidence_excerpt")
            _require(isinstance(excerpt, str) and normalize_answer_span(excerpt), f"{tp}.evidence_excerpt is required")
            digest = target.get("answer_span_hash")
            _require(isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None, f"{tp}.answer_span_hash is invalid")
            _require(digest == answer_span_hash(excerpt), f"{tp}.answer_span_hash does not match evidence_excerpt")
            evidence_scope = target.get("evidence_scope", "child")
            _require(evidence_scope in _EVIDENCE_SCOPES, f"{tp}.evidence_scope is invalid")
            if evidence_scope == "parent_expand_required":
                origin_chunk_id = target.get("origin_chunk_id")
                parent_content_hash = target.get("parent_content_hash")
                _require(
                    isinstance(origin_chunk_id, str) and origin_chunk_id == chunk_id,
                    f"{tp}.origin_chunk_id must equal the reviewed parent chunk_id",
                )
                _require(
                    isinstance(parent_content_hash, str)
                    and _SHA256_RE.fullmatch(parent_content_hash) is not None,
                    f"{tp}.parent_content_hash is invalid",
                )
            _require(isinstance(target.get("review_note"), str) and target["review_note"].strip(), f"{tp}.review_note is required")
            key = (source, chunk_id, digest)
            _require(key not in target_keys, f"{query_id}: duplicate relevant target")
            target_keys.add(key)
        _require(direct_count > 0, f"{query_id}: {outcome} questions need at least one direct target")

    if expected_query_ids is not None:
        expected = set(expected_query_ids)
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        _require(not missing and not extra, f"query coverage mismatch; missing={missing}, extra={extra}")
    return payload


def validate_qrels_against_collection(
    payload: dict[str, Any],
    collection: Any,
    *,
    parent_collection: Any | None = None,
) -> dict[str, Any]:
    """Validate target identity and evidence against a Chroma-like collection.

    Schema validation alone cannot detect a syntactically valid but nonexistent
    chunk ID.  This second-level check is intentionally duck-typed so the core
    contract can be unit-tested without importing Chroma in normal PR tests.
    """

    validate_qrels_overlay(payload)
    expected_collection = payload["index"]["collection"]
    actual_name = getattr(collection, "name", None)
    if actual_name is not None:
        _require(actual_name == expected_collection, f"collection mismatch: expected {expected_collection}, got {actual_name}")

    targets = [target for item in payload["items"] for target in item["relevant_targets"]]
    child_targets = [
        target for target in targets
        if target.get("evidence_scope", "child") == "child"
    ]
    parent_targets = [
        target for target in targets
        if target.get("evidence_scope") == "parent_expand_required"
    ]
    chunk_ids = sorted({target["chunk_id"] for target in child_targets})
    snapshot = (
        collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        if chunk_ids else {"ids": [], "documents": [], "metadatas": []}
    )
    returned_ids = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    _require(len(returned_ids) == len(documents) == len(metadatas), "collection returned misaligned snapshot fields")
    rows = {chunk_id: (document, metadata) for chunk_id, document, metadata in zip(returned_ids, documents, metadatas)}
    missing = sorted(set(chunk_ids) - set(rows))
    _require(not missing, f"qrels reference missing chunk_ids: {missing}")

    for target in child_targets:
        chunk_id = target["chunk_id"]
        document, metadata = rows[chunk_id]
        _require(isinstance(document, str), f"{chunk_id}: collection document is not text")
        _require(isinstance(metadata, dict), f"{chunk_id}: collection metadata is invalid")
        actual_source = metadata.get("source")
        _require(actual_source == target["source"], f"{chunk_id}: source mismatch; qrels={target['source']}, index={actual_source}")
        _require(
            normalize_answer_span(target["evidence_excerpt"]) in normalize_answer_span(document),
            f"{chunk_id}: evidence_excerpt is not contained in the indexed document",
        )
    if parent_targets:
        child_snapshot = collection.get(include=["metadatas"])
        child_metadatas = list(child_snapshot.get("metadatas") or [])
        by_origin: dict[str, list[dict[str, Any]]] = {}
        for metadata in child_metadatas:
            if not isinstance(metadata, dict):
                continue
            origin = str(metadata.get("origin_chunk_id") or "")
            if origin:
                by_origin.setdefault(origin, []).append(metadata)
        for target in parent_targets:
            origin = target["origin_chunk_id"]
            candidates = by_origin.get(origin, [])
            _require(bool(candidates), f"{origin}: no experiment child references reviewed parent")
            _require(
                all(metadata.get("source") == target["source"] for metadata in candidates),
                f"{origin}: child source mismatch for parent expansion",
            )
            _require(
                all(metadata.get("parent_content_hash") == target["parent_content_hash"]
                    for metadata in candidates),
                f"{origin}: child parent_content_hash mismatch",
            )
        if parent_collection is not None:
            parent_ids = sorted({target["origin_chunk_id"] for target in parent_targets})
            parent_snapshot = parent_collection.get(
                ids=parent_ids, include=["documents", "metadatas"],
            )
            parent_rows = {
                chunk_id: (document, metadata)
                for chunk_id, document, metadata in zip(
                    parent_snapshot.get("ids") or [],
                    parent_snapshot.get("documents") or [],
                    parent_snapshot.get("metadatas") or [],
                )
            }
            _require(
                set(parent_ids) <= set(parent_rows),
                f"parent collection missing origin ids: {sorted(set(parent_ids) - set(parent_rows))}",
            )
            for target in parent_targets:
                document, metadata = parent_rows[target["origin_chunk_id"]]
                _require(isinstance(document, str), "parent expansion document is not text")
                _require(isinstance(metadata, dict), "parent expansion metadata is invalid")
                actual_hash = "sha256:" + hashlib.sha256(document.encode("utf-8")).hexdigest()
                _require(
                    actual_hash == target["parent_content_hash"],
                    f"{target['origin_chunk_id']}: reviewed parent hash mismatch",
                )
                _require(
                    metadata.get("source") == target["source"],
                    f"{target['origin_chunk_id']}: reviewed parent source mismatch",
                )
                _require(
                    normalize_answer_span(target["evidence_excerpt"])
                    in normalize_answer_span(document),
                    f"{target['origin_chunk_id']}: full evidence is absent from reviewed parent",
                )
    return payload


def load_qrels_overlay(
    path: str | Path,
    *,
    expected_query_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_qrels_overlay(payload, expected_query_ids=expected_query_ids)
