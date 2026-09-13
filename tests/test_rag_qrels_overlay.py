from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from rag_qrels import (
    QrelsValidationError,
    answer_span_hash,
    load_qrels_overlay,
    validate_qrels_against_collection,
    validate_qrels_overlay,
)


pytestmark = pytest.mark.private_artifact


ROOT = Path(__file__).resolve().parents[1]
BASE_SET = ROOT / "tests" / "rag_bench_paraphrase_set.json"
OVERLAY = ROOT / "docs" / "rag_eval" / "qrels" / "rag_bench_paraphrase_reviewer_a.json"
SCHEMA = ROOT / "tests" / "fixtures" / "rag_qrels_overlay_v1.schema.json"


def _query_ids() -> list[str]:
    return [item["id"] for item in json.loads(BASE_SET.read_text(encoding="utf-8"))["items"]]


def test_reviewer_a_overlay_covers_immutable_52_question_set() -> None:
    payload = load_qrels_overlay(OVERLAY, expected_query_ids=_query_ids())
    assert len(payload["items"]) == 52
    assert any(item["review_outcome"] == "unsupported" for item in payload["items"])
    for item in payload["items"]:
        for target in item["relevant_targets"]:
            assert target["answer_span_hash"] == answer_span_hash(target["evidence_excerpt"])


def test_reviewer_a_overlay_conforms_to_published_json_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    payload = json.loads(OVERLAY.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def test_overlay_rejects_duplicate_query_ids() -> None:
    payload = load_qrels_overlay(OVERLAY)
    broken = copy.deepcopy(payload)
    broken["items"].append(copy.deepcopy(broken["items"][0]))
    with pytest.raises(QrelsValidationError, match="duplicate query_id"):
        validate_qrels_overlay(broken)


@pytest.mark.parametrize("mutation, message", [
    (lambda t: t.__setitem__("relevance", "maybe"), "relevance"),
    (lambda t: t.__setitem__("answer_span_hash", "sha256:" + "0" * 64), "does not match"),
    (lambda t: t.__setitem__("source", "../secret.md"), "path"),
])
def test_overlay_rejects_invalid_targets(mutation, message: str) -> None:
    payload = load_qrels_overlay(OVERLAY)
    broken = copy.deepcopy(payload)
    target = next(item["relevant_targets"][0] for item in broken["items"] if item["relevant_targets"])
    mutation(target)
    with pytest.raises(QrelsValidationError, match=message):
        validate_qrels_overlay(broken)


def test_unsupported_outcome_cannot_carry_a_convenient_false_gold() -> None:
    payload = load_qrels_overlay(OVERLAY)
    broken = copy.deepcopy(payload)
    unsupported = next(item for item in broken["items"] if item["review_outcome"] == "unsupported")
    accepted = next(item for item in broken["items"] if item["review_outcome"] == "accepted")
    unsupported["relevant_targets"] = copy.deepcopy(accepted["relevant_targets"][:1])
    with pytest.raises(QrelsValidationError, match="must not invent targets"):
        validate_qrels_overlay(broken)


class _FakeCollection:
    name = "offerclaw_local_bge_base_zh_768"

    def __init__(self, rows):
        self.rows = rows

    def get(self, *, ids, include):
        present = [(chunk_id, self.rows[chunk_id]) for chunk_id in ids if chunk_id in self.rows]
        return {
            "ids": [chunk_id for chunk_id, _ in present],
            "documents": [row[0] for _, row in present],
            "metadatas": [row[1] for _, row in present],
        }


def _minimal_index_rows(payload):
    excerpts = {}
    sources = {}
    for item in payload["items"]:
        for target in item["relevant_targets"]:
            excerpts.setdefault(target["chunk_id"], []).append(target["evidence_excerpt"])
            sources[target["chunk_id"]] = target["source"]
    return {
        chunk_id: ("\n".join(parts), {"source": sources[chunk_id]})
        for chunk_id, parts in excerpts.items()
    }


def test_collection_level_validation_checks_identity_source_and_span() -> None:
    payload = load_qrels_overlay(OVERLAY)
    rows = _minimal_index_rows(payload)
    validate_qrels_against_collection(payload, _FakeCollection(rows))

    target = next(item["relevant_targets"][0] for item in payload["items"] if item["relevant_targets"])
    missing = dict(rows)
    missing.pop(target["chunk_id"])
    with pytest.raises(QrelsValidationError, match="missing chunk_ids"):
        validate_qrels_against_collection(payload, _FakeCollection(missing))

    wrong_source = dict(rows)
    wrong_source[target["chunk_id"]] = (target["evidence_excerpt"], {"source": "wrong.md"})
    with pytest.raises(QrelsValidationError, match="source mismatch"):
        validate_qrels_against_collection(payload, _FakeCollection(wrong_source))

    wrong_text = dict(rows)
    wrong_text[target["chunk_id"]] = ("unrelated text", {"source": target["source"]})
    with pytest.raises(QrelsValidationError, match="not contained"):
        validate_qrels_against_collection(payload, _FakeCollection(wrong_text))
