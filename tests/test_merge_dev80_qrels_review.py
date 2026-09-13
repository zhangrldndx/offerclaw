# -*- coding: utf-8 -*-
"""Tests for the Dev80 qrels merge decision rules (no model calls)."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_dev80_qrels_review import (  # noqa: E402
    REVISION_ID,
    _apply,
    _decide,
)


def _verification(answerable, supported):
    return {"directly_answerable": answerable,
            "requirement_fully_supported": supported, "reason": "…"}


def _coverage(verdict, missing=()):
    return {"covered_points": ["…"], "missing_points": list(missing),
            "verdict": verdict}


ADDITION = {"kind": "addition"}
RETRACTION = {"kind": "retraction"}


def test_addition_needs_both_passes_to_agree():
    ok = _decide(ADDITION, _verification(True, True), _coverage("fully_covers"))
    assert ok["applied"] is True
    assert ok["action"] == "add_grade3_target"


@pytest.mark.parametrize("verification,coverage", [
    (_verification(True, False), _coverage("fully_covers")),
    (_verification(True, True), _coverage("partially_covers", ["缺一点"])),
    (_verification(False, False), _coverage("does_not_cover", ["全缺"])),
])
def test_addition_rejected_unless_both_passes_confirm(verification, coverage):
    decision = _decide(ADDITION, verification, coverage)
    assert decision["applied"] is False
    assert decision["action"] == "rejected"


def test_retraction_applies_only_on_partial_coverage():
    decision = _decide(RETRACTION, _verification(False, False),
                       _coverage("partially_covers", ["缺机制"]))
    assert decision["applied"] is True
    assert decision["action"] == "retract_negative_add_grade2_target"


def test_retraction_rejected_when_reviewer_says_it_is_actually_gold():
    decision = _decide(RETRACTION, _verification(True, True),
                       _coverage("fully_covers"))
    assert decision["applied"] is False
    assert decision["action"] == "rejected_should_be_grade3"


def test_retraction_rejected_when_reviewer_upholds_the_negative():
    decision = _decide(RETRACTION, _verification(False, False),
                       _coverage("does_not_cover", ["全缺"]))
    assert decision["applied"] is False
    assert decision["action"] == "rejected_negative_upheld"


def _payload(anchor_id="a1", negative_id=None):
    requirement = "答案必须说明机制甲与机制乙。"
    return {"items": [{
        "query_id": f"col-{anchor_id}-{style}",
        "anchor_id": anchor_id,
        "split": "dev",
        "case_kind": "positive",
        "query_style": style,
        "question": "问题？",
        "answer_requirements": [requirement],
        "relevant_targets": [{
            "chunk_id": "gold",
            "source": "doc.md",
            "heading_path": ["正文"],
            "relevance_grade": 3,
            "supported_requirements": [requirement],
            "evidence_excerpt": "机制甲与机制乙的说明。",
            "evidence_span_hash": "sha256:" + "0" * 64,
        }],
        "hard_negatives": ([{"chunk_id": negative_id, "source": "doc.md",
                             "reason": "旧裁决"}] if negative_id else []),
    } for style in ("standard", "natural", "colloquial", "long_context")]}


def _row(chunk_id, grade):
    return {
        "anchor_id": "a1",
        "chunk_id": chunk_id,
        "source": "doc.md",
        "heading_path": ["正文"],
        "relevance_grade": grade,
        "answer_requirements": ["答案必须说明机制甲与机制乙。"],
        "excerpt": "机制甲的说明。",
    }


def test_apply_adds_target_to_every_style_of_the_anchor():
    payload = _payload()
    decision = {"action": "add_grade3_target", "rationale": "r"}
    touched = _apply(payload, _row("equiv", 3), decision, _coverage("fully_covers"))
    assert touched == 4
    for item in payload["items"]:
        grades = {target["chunk_id"]: target["relevance_grade"]
                  for target in item["relevant_targets"]}
        assert grades == {"gold": 3, "equiv": 3}
        assert item["gold_revisions"][0]["revision"] == REVISION_ID


def test_apply_retraction_removes_the_negative_and_records_gaps():
    payload = _payload(negative_id="partial")
    decision = {"action": "retract_negative_add_grade2_target", "rationale": "r"}
    coverage = _coverage("partially_covers", ["缺机制乙"])
    touched = _apply(payload, _row("partial", 2), decision, coverage)
    assert touched == 4
    for item in payload["items"]:
        assert item["hard_negatives"] == []
        added = next(target for target in item["relevant_targets"]
                     if target["chunk_id"] == "partial")
        assert added["relevance_grade"] == 2
        # the schema cannot express partial requirement coverage, so the gap
        # list is what records the truth
        assert added["partial_support"] is True
        assert added["uncovered_requirement_points"] == ["缺机制乙"]


def test_apply_is_idempotent_for_an_already_present_target():
    payload = _payload()
    decision = {"action": "add_grade3_target", "rationale": "r"}
    _apply(payload, _row("equiv", 3), decision, _coverage("fully_covers"))
    again = _apply(payload, _row("equiv", 3), decision, _coverage("fully_covers"))
    assert again == 0
    assert len(payload["items"][0]["relevant_targets"]) == 2


def test_merged_v1_carries_the_revision_and_only_touches_dev():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json"
    trail_path = root / "docs/rag_eval/colloquial/DEV80_QRELS_MERGE_TRAIL_20260826.json"
    if not path.exists() or not trail_path.exists():
        pytest.skip("merged V1 artifacts unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    revisions = {row["revision"] for row in payload.get("revisions", [])}
    assert REVISION_ID in revisions
    revised = [item for item in payload["items"] if item.get("gold_revisions")]
    assert revised, "revision recorded but no row carries it"
    assert {item["split"] for item in revised} == {"dev"}
    trail = json.loads(trail_path.read_text(encoding="utf-8"))
    # every proposal is recorded with both independent passes, accepted or not
    for case in trail["cases"]:
        assert "directly_answerable" in case["verification"]
        assert case["coverage"]["verdict"] in {
            "fully_covers", "partially_covers", "does_not_cover"
        }
