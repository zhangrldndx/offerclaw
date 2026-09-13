# -*- coding: utf-8 -*-
"""Tests for the Dev80 original/reviewed dual-qrels reporter."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_dev80_dual_qrels import (  # noqa: E402
    build_report,
    render_markdown,
    resolve_overlay,
)


class _FakeCollection:
    def __init__(self, rows):
        self._rows = rows

    def get(self, ids=None, include=None):
        selected = [chunk_id for chunk_id in (ids or []) if chunk_id in self._rows]
        return {
            "ids": selected,
            "documents": [self._rows[chunk_id][0] for chunk_id in selected],
            "metadatas": [{"source": self._rows[chunk_id][1]} for chunk_id in selected],
        }


def _case(query_id, anchor_id, style, gold_id):
    return {
        "query_id": query_id,
        "anchor_id": anchor_id,
        "split": "dev",
        "case_kind": "positive",
        "query_style": style,
        "question": f"问题 {query_id}",
        "answer_requirements": [f"要求 {anchor_id}"],
        "relevant_targets": [{
            "chunk_id": gold_id,
            "source": "doc_a.md",
            "heading_path": ["正文"],
            "relevance_grade": 3,
            "supported_requirements": [f"要求 {anchor_id}"],
            "evidence_excerpt": "证据",
            "evidence_span_hash": "sha256:" + "0" * 64,
        }],
        "hard_negatives": [],
    }


def _row(query_id, anchor_id, style, final_ids, gate=True):
    return {
        "query_id": query_id,
        "anchor_id": anchor_id,
        "domain": "backend",
        "query_style": style,
        "dense_chunk_ids": list(final_ids),
        "bm25_chunk_ids": list(final_ids),
        "fusion_chunk_ids": list(final_ids),
        "reranked_chunk_ids": list(final_ids),
        "final_chunk_ids": list(final_ids),
        "gate_decision": gate,
        "latency_ms": 1.0,
        "latency_by_stage": {"total": 1.0},
    }


@pytest.fixture()
def scenario():
    # q1: an equivalent chunk sits at rank 1, gold at rank 2 -> reviewed wins.
    # q2: a partial chunk sits at rank 1, gold at rank 2 -> reviewed must NOT win.
    rows = [
        _row("q1", "a1", "standard", ["equiv", "gold_a"]),
        _row("q2", "a2", "standard", ["partial", "gold_b"]),
    ]
    result = {
        "input": {"dataset_id": "synthetic", "index": {"collection": "c", "count": 1}},
        "configuration": {"retrieval_arm": "compact32"},
        "runs": [{"positive": {"rows": json.loads(json.dumps(rows))}} for _ in range(3)],
    }
    cases = {"items": [
        _case("q1", "a1", "standard", "gold_a"),
        _case("q2", "a2", "standard", "gold_b"),
    ]}
    overlay = {
        "status": "diagnostic_only_not_merged_into_v1",
        "additions": [{
            "anchor_id": "a1", "chunk_id": "equiv", "source": "doc_a.md",
            "heading_path": ["正文"], "relevance_grade": 3,
            "excerpt_start": "完整覆盖", "excerpt_end": "全部要求。",
            "verdict": "equivalent_gold", "reason": "复核确认等价。",
        }],
        "retractions": [{
            "anchor_id": "a2", "chunk_id": "partial", "source": "doc_a.md",
            "heading_path": ["正文"], "from": "v1_hard_negative", "to": "grade2_partial",
            "relevance_grade": 2,
            "excerpt_start": "只覆盖", "excerpt_end": "一部分。",
            "verdict": "partial_support_not_equivalent", "reason": "缺机制细节。",
        }],
    }
    collection = _FakeCollection({
        "equiv": ("这一块完整覆盖了问题的全部要求。后面还有别的内容。", "doc_a.md"),
        "partial": ("这一块只覆盖了要求的一部分。缺少机制细节。", "doc_a.md"),
    })
    return result, cases, overlay, collection


def test_resolve_overlay_hashes_verbatim_slices(scenario):
    _result, _cases, overlay, collection = scenario
    resolved = resolve_overlay(overlay, collection)
    entries = {row["chunk_id"]: row for row in resolved["resolved_entries"]}
    assert entries["equiv"]["evidence_excerpt"] == "完整覆盖了问题的全部要求。"
    assert entries["equiv"]["evidence_span_hash"].startswith("sha256:")
    assert entries["partial"]["kind"] == "retraction"


def test_resolve_overlay_rejects_non_verbatim_marker(scenario):
    _result, _cases, overlay, collection = scenario
    overlay["additions"][0]["excerpt_start"] = "这句话不在原文里"
    with pytest.raises(Exception, match="excerpt_start not found"):
        resolve_overlay(overlay, collection)


def test_grade3_addition_wins_but_grade2_retraction_does_not(scenario):
    result, cases, overlay, collection = scenario
    report = build_report(result, cases, resolve_overlay(overlay, collection))
    original = report["readings"]["original"]["metrics"]["strict_ranking"]["recall@1"]
    reviewed = report["readings"]["reviewed"]["metrics"]["strict_ranking"]["recall@1"]
    assert original["hits"] == 0
    # only q1 (grade-3 equivalent at rank 1) flips; q2's grade-2 chunk must not.
    assert reviewed["hits"] == 1
    assert report["delta"]["wins"] == 1
    assert report["delta"]["losses"] == 0
    still = {row["query_id"] for row in report["delta"]["still_failing_under_reviewed"]}
    assert still == {"q2"}


def test_grade2_does_not_create_sufficient_evidence(scenario):
    result, cases, overlay, collection = scenario
    # Drop the gold from q2's candidate list so only the partial chunk remains.
    for run in result["runs"]:
        for row in run["positive"]["rows"]:
            if row["query_id"] == "q2":
                for key in ("dense_chunk_ids", "bm25_chunk_ids", "fusion_chunk_ids",
                            "reranked_chunk_ids", "final_chunk_ids"):
                    row[key] = ["partial"]
    report = build_report(result, cases, resolve_overlay(overlay, collection))
    failing = {row["query_id"]
               for row in report["delta"]["still_failing_under_reviewed"]}
    assert "q2" in failing
    reviewed = report["readings"]["reviewed"]["metrics"]
    # q1 stays sufficient; q2 must not become sufficient from a partial chunk.
    assert reviewed["context"]["sufficient_evidence@5"]["hits"] == 1


def test_markdown_states_v1_is_unchanged(scenario):
    result, cases, overlay, collection = scenario
    report = build_report(result, cases, resolve_overlay(overlay, collection))
    text = render_markdown(report, hashes={
        "result_path": "r.json", "result_sha256": "sha256:r",
        "cases_path": "c.json", "cases_sha256": "sha256:c",
        "overlay_path": "o.json", "overlay_sha256": "sha256:o",
    })
    assert "V1 金标文件未被修改" in text
    assert "strict R@1" in text


def test_original_reading_reproduces_frozen_metrics():
    """The reviewed reading is only trustworthy if the original one is exact."""

    root = Path(__file__).resolve().parents[1]
    result_path = root / "docs/rag_eval/colloquial/dev80_compact32.json"
    report_path = root / "docs/rag_eval/colloquial/DEV80_DUAL_QRELS_20260826.json"
    if not result_path.exists() or not report_path.exists():
        pytest.skip("frozen Dev80 artifacts unavailable")
    frozen = json.loads(result_path.read_text(encoding="utf-8"))
    stored = frozen["runs"][0]["positive"]["metrics"]
    recomputed = json.loads(report_path.read_text(encoding="utf-8"))
    original = recomputed["readings"]["original"]["metrics"]
    for key in ("strict_ranking", "funnel", "reranker", "context",
                "retriever_and_fusion", "channel_exclusivity"):
        assert json.dumps(stored[key], sort_keys=True) == \
            json.dumps(original[key], sort_keys=True), key
