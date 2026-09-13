# -*- coding: utf-8 -*-
"""Graded, requirement-aware qrels used by colloquial RAG evaluation.

V2 intentionally lives beside :mod:`rag_qrels` instead of replacing it.  The
historical 52-question overlay remains immutable; new training/development and
sealed-blind sets use this stricter contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from rag_qrels import normalize_answer_span


SCHEMA_VERSION = "rag-graded-qrels-v2"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SPLITS = {"train", "dev", "blind"}
# ``implicit_oral`` and ``long_noisy`` were added for the V2-A anchor wave
# (2026-08-26).  The addition is backwards-compatible: every V1 artifact keeps
# validating exactly as before, and V1 files never contain the new styles.
_STYLES = {
    "standard", "natural", "colloquial", "long_context", "negative",
    "implicit_oral", "long_noisy",
}
_REVIEW = {"draft", "needs_adjudication", "approved", "rejected"}
_CASE_KINDS = {"positive", "negative"}


class GradedQrelsValidationError(ValueError):
    """Raised when a V2 qrels artifact violates its fail-closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GradedQrelsValidationError(message)


def evidence_span_hash(text: str) -> str:
    normalized = normalize_answer_span(text)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def index_contract_fingerprint(index: dict[str, Any]) -> str:
    """Hash stable index semantics while excluding timestamps/git dirtiness."""

    stable = {
        "collection": str(index.get("collection") or ""),
        "count": int(index.get("collection_count", index.get("count", 0)) or 0),
        "content": str(index.get("collection_content_hash")
                       or index.get("index_content_fingerprint") or ""),
        "embedding_provider": str(index.get("embedding_provider") or ""),
        "embedding_model": str(index.get("embedding_model") or ""),
        "embedding_dimensions": index.get("embedding_dimensions"),
        "chunker_version": str(index.get("chunker_version") or ""),
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    _require(isinstance(value, list), f"{label} must be a list")
    _require(all(isinstance(item, str) and item.strip() for item in value),
             f"{label} must contain non-empty strings")
    if nonempty:
        _require(bool(value), f"{label} must not be empty")
    return value


def validate_graded_qrels(
    payload: Any,
    *,
    expected_query_ids: Iterable[str] | None = None,
    require_approved: bool = False,
    allowed_splits: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a V2 artifact without consulting retrieval output.

    ``require_approved`` is the release boundary.  Development tools may open
    drafts for review, while any metric used for promotion must set it true.
    """

    _require(isinstance(payload, dict), "qrels root must be an object")
    _require(payload.get("schema_version") == SCHEMA_VERSION,
             "unsupported schema_version")
    _require(isinstance(payload.get("dataset_id"), str)
             and payload["dataset_id"].strip(), "dataset_id is required")
    index = payload.get("index")
    _require(isinstance(index, dict), "index must be an object")
    _require(isinstance(index.get("collection"), str)
             and index["collection"].strip(), "index.collection is required")
    _require(isinstance(index.get("count"), int) and index["count"] >= 0,
             "index.count must be non-negative")
    if index.get("fingerprint"):
        _require(_SHA256_RE.fullmatch(str(index["fingerprint"])) is not None,
                 "index.fingerprint is invalid")
    items = payload.get("items")
    _require(isinstance(items, list), "items must be a list")

    seen: set[str] = set()
    for position, item in enumerate(items):
        prefix = f"items[{position}]"
        _require(isinstance(item, dict), f"{prefix} must be an object")
        query_id = item.get("query_id")
        _require(isinstance(query_id, str) and query_id.strip(),
                 f"{prefix}.query_id is required")
        _require(query_id not in seen, f"duplicate query_id: {query_id}")
        seen.add(query_id)
        _require(isinstance(item.get("anchor_id"), str)
                 and item["anchor_id"].strip(), f"{query_id}: anchor_id is required")
        _require(item.get("split") in _SPLITS, f"{query_id}: invalid split")
        if allowed_splits is not None:
            _require(item["split"] in allowed_splits,
                     f"{query_id}: split is outside this artifact")
        _require(item.get("case_kind") in _CASE_KINDS,
                 f"{query_id}: invalid case_kind")
        _require(item.get("query_style") in _STYLES,
                 f"{query_id}: invalid query_style")
        _require(isinstance(item.get("question"), str) and item["question"].strip(),
                 f"{query_id}: question is required")
        _string_list(item.get("phenomena"), f"{query_id}.phenomena")
        requirements = _string_list(
            item.get("answer_requirements"),
            f"{query_id}.answer_requirements",
            nonempty=item["case_kind"] == "positive",
        )
        _require(len(set(requirements)) == len(requirements),
                 f"{query_id}: duplicate answer requirements")
        review_status = item.get("review_status")
        _require(review_status in _REVIEW, f"{query_id}: invalid review_status")
        if require_approved:
            _require(review_status == "approved",
                     f"{query_id}: release evaluation requires approved rows")
        _require(isinstance(item.get("review_note", ""), str),
                 f"{query_id}: review_note must be text")

        targets = item.get("relevant_targets")
        _require(isinstance(targets, list),
                 f"{query_id}: relevant_targets must be a list")
        if item["case_kind"] == "negative":
            _require(not targets, f"{query_id}: negative rows must not invent targets")
            _require(not requirements,
                     f"{query_id}: negative rows must not claim answer requirements")
        elif review_status != "rejected":
            _require(bool(targets), f"{query_id}: positive row needs evidence targets")

        target_keys: set[tuple[str, str, str]] = set()
        grade3 = 0
        supported: set[str] = set()
        for target_index, target in enumerate(targets):
            tp = f"{query_id}.relevant_targets[{target_index}]"
            _require(isinstance(target, dict), f"{tp} must be an object")
            _require(isinstance(target.get("chunk_id"), str)
                     and target["chunk_id"].strip(), f"{tp}.chunk_id is required")
            _require(isinstance(target.get("source"), str)
                     and target["source"].strip(), f"{tp}.source is required")
            _string_list(target.get("heading_path"), f"{tp}.heading_path")
            grade = target.get("relevance_grade")
            _require(grade in {1, 2, 3}, f"{tp}.relevance_grade is invalid")
            grade3 += int(grade == 3)
            target_requirements = _string_list(
                target.get("supported_requirements"),
                f"{tp}.supported_requirements",
                nonempty=True,
            )
            _require(set(target_requirements) <= set(requirements),
                     f"{tp}: supported requirement is not declared by the question")
            supported.update(target_requirements)
            excerpt = target.get("evidence_excerpt")
            _require(isinstance(excerpt, str) and normalize_answer_span(excerpt),
                     f"{tp}.evidence_excerpt is required")
            digest = target.get("evidence_span_hash")
            _require(isinstance(digest, str)
                     and _SHA256_RE.fullmatch(digest) is not None,
                     f"{tp}.evidence_span_hash is invalid")
            _require(digest == evidence_span_hash(excerpt),
                     f"{tp}.evidence_span_hash does not match evidence_excerpt")
            key = (target["chunk_id"], target["source"], digest)
            _require(key not in target_keys, f"{query_id}: duplicate relevant target")
            target_keys.add(key)
        if item["case_kind"] == "positive" and review_status != "rejected":
            _require(grade3 > 0,
                     f"{query_id}: positive row needs at least one grade-3 target")
            _require(set(requirements) <= supported,
                     f"{query_id}: every answer requirement needs evidence coverage")

        hard_negatives = item.get("hard_negatives")
        _require(isinstance(hard_negatives, list),
                 f"{query_id}.hard_negatives must be a list")
        for negative_index, negative in enumerate(hard_negatives):
            np = f"{query_id}.hard_negatives[{negative_index}]"
            _require(isinstance(negative, dict), f"{np} must be an object")
            _require(isinstance(negative.get("chunk_id"), str)
                     and negative["chunk_id"].strip(), f"{np}.chunk_id is required")
            _require(isinstance(negative.get("reason"), str)
                     and negative["reason"].strip(), f"{np}.reason is required")
            _require(negative["chunk_id"] not in {t["chunk_id"] for t in targets},
                     f"{np}: a relevant target cannot also be a hard negative")

    if expected_query_ids is not None:
        expected = set(expected_query_ids)
        _require(seen == expected,
                 f"query coverage mismatch; missing={sorted(expected - seen)}, "
                 f"extra={sorted(seen - expected)}")
    return payload


def validate_graded_qrels_against_collection(
    payload: dict[str, Any], collection: Any,
) -> dict[str, Any]:
    """Verify stable IDs, source lineage, and exact evidence in the live index."""

    validate_graded_qrels(payload)
    actual_name = getattr(collection, "name", None)
    if actual_name is not None:
        _require(actual_name == payload["index"]["collection"],
                 f"collection mismatch: expected {payload['index']['collection']}, "
                 f"got {actual_name}")
    targets = [
        target for item in payload["items"] for target in item["relevant_targets"]
    ]
    ids = sorted({target["chunk_id"] for target in targets})
    snapshot = (collection.get(ids=ids, include=["documents", "metadatas"])
                if ids else {"ids": [], "documents": [], "metadatas": []})
    rows = {
        str(chunk_id): (document, metadata)
        for chunk_id, document, metadata in zip(
            snapshot.get("ids") or [],
            snapshot.get("documents") or [],
            snapshot.get("metadatas") or [],
        )
    }
    _require(set(ids) <= set(rows),
             f"qrels reference missing chunk_ids: {sorted(set(ids) - set(rows))}")
    for target in targets:
        document, metadata = rows[target["chunk_id"]]
        _require(isinstance(document, str),
                 f"{target['chunk_id']}: indexed document is not text")
        _require(isinstance(metadata, dict),
                 f"{target['chunk_id']}: indexed metadata is invalid")
        _require(str(metadata.get("source") or "") == target["source"],
                 f"{target['chunk_id']}: source mismatch")
        _require(normalize_answer_span(target["evidence_excerpt"])
                 in normalize_answer_span(document),
                 f"{target['chunk_id']}: evidence excerpt is absent")
    return payload


def load_graded_qrels(
    path: str | Path,
    *,
    require_approved: bool = False,
    allowed_splits: set[str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_graded_qrels(
        payload,
        require_approved=require_approved,
        allowed_splits=allowed_splits,
    )
