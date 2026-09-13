# -*- coding: utf-8 -*-
"""Unit tests for the Dev80 ranking-failure audit tool."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_colloquial_ranking_failures import (  # noqa: E402
    CHUNK_ADJUDICATIONS,
    build_audit,
    render_markdown,
)


def _candidate(chunk_id, source, rank, score, heading="正文"):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "source_type": "learning_resource",
        "heading_path": heading,
        "rank": rank,
        "dense_distance": None,
        "bm25_score": None,
        "rrf_score": None,
        "rerank_score": score,
        "channels": [],
    }


def _row(query_id, anchor_id, style, gold_id, reranked, fusion_rank, ranks=None):
    reranked_rank = next(
        (candidate["rank"] for candidate in reranked
         if candidate["chunk_id"] == gold_id),
        0,
    )
    return {
        "query_id": query_id,
        "anchor_id": anchor_id,
        "domain": "llm_app",
        "query_style": style,
        "grade3_chunk_ids": [gold_id],
        "dense_rank": (ranks or {}).get("dense", 2),
        "bm25_rank": (ranks or {}).get("bm25", 1),
        "union_rank": (ranks or {}).get("union", 1),
        "fusion_rank": fusion_rank,
        "reranked_rank": reranked_rank,
        "final_rank": reranked_rank,
        "gate_decision": True,
        "gate_features": {"rerank_top": 0.9, "rerank_margin": 0.01},
        "retrieval_trace": {"reranked_candidates": reranked},
    }


def _case(query_id, anchor_id, style, gold_id, source, hard_negative_id=None):
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
            "source": source,
            "heading_path": ["正文"],
            "relevance_grade": 3,
            "supported_requirements": [f"要求 {anchor_id}"],
            "evidence_excerpt": "证据",
            "evidence_span_hash": "sha256:" + "0" * 64,
        }],
        "hard_negatives": (
            [{"chunk_id": hard_negative_id, "source": source, "reason": "既有负例"}]
            if hard_negative_id else []
        ),
    }


@pytest.fixture()
def synthetic():
    gold = "doc_a_gold"
    wrong_same = "doc_a_wrong"
    wrong_cross = "doc_b_wrong"
    rows = [
        # near miss, same doc, fusion=2, wrong top1 is the V1 hard negative
        _row("col-x1-standard", "x1", "standard", gold, [
            _candidate(wrong_same, "doc_a.md", 1, 0.95),
            _candidate(gold, "doc_a.md", 2, 0.91),
        ], fusion_rank=2),
        # sibling style of the same anchor succeeded -> register shift
        _row("col-x1-natural", "x1", "natural", gold, [
            _candidate(gold, "doc_a.md", 1, 0.97),
            _candidate(wrong_same, "doc_a.md", 2, 0.90),
        ], fusion_rank=1),
        # deep miss, cross doc, fusion=1 -> reranker_harm
        _row("col-x2-standard", "x2", "standard", gold, [
            _candidate(wrong_cross, "doc_b.md", 1, 0.99),
            _candidate("noise_1", "doc_c.md", 2, 0.98),
            _candidate("noise_2", "doc_c.md", 3, 0.97),
            _candidate("noise_3", "doc_c.md", 4, 0.96),
            _candidate(gold, "doc_a.md", 5, 0.95),
        ], fusion_rank=1),
    ]
    result = {
        "input": {"dataset_id": "synthetic", "index": {"collection": "c", "count": 1}},
        "configuration": {"retrieval_arm": "compact32"},
        "runs": [
            {"positive": {"rows": [json.loads(json.dumps(row)) for row in rows]}}
            for _ in range(3)
        ],
    }
    cases = {"items": [
        _case("col-x1-standard", "x1", "standard", gold, "doc_a.md",
              hard_negative_id=wrong_same),
        _case("col-x1-natural", "x1", "natural", gold, "doc_a.md",
              hard_negative_id=wrong_same),
        _case("col-x2-standard", "x2", "standard", gold, "doc_a.md"),
    ]}
    return result, cases


def test_build_audit_classifies_near_and_deep_miss(synthetic):
    result, cases = synthetic
    audit = build_audit(result, cases)
    assert audit["summary"]["failures"] == 2
    by_id = {record["query_id"]: record for record in audit["records"]}

    near = by_id["col-x1-standard"]
    assert "near_miss_rank_2_3" in near["failure_classes"]
    assert "same_document_wrong_section" in near["failure_classes"]
    # sibling natural style ranked 1 -> register shift flagged
    assert "query_document_register_shift" in near["failure_classes"]
    assert near["wrong_top1"]["is_v1_hard_negative"] is True
    assert near["root_cause"] == "query_document_representation"
    assert near["top1_minus_gold_margin"] == pytest.approx(0.04)

    deep = by_id["col-x2-standard"]
    assert "deep_miss_rank_4_10" in deep["failure_classes"]
    assert "cross_document_near_topic" in deep["failure_classes"]
    assert deep["root_cause"] == "reranker_harm"
    assert len(deep["competitors_above_gold"]) == 4


def test_build_audit_flags_cross_run_inconsistency(synthetic):
    result, cases = synthetic
    result["runs"][2]["positive"]["rows"][0]["reranked_rank"] = 3
    audit = build_audit(result, cases)
    assert audit["summary"]["cross_run_inconsistent_query_ids"] == ["col-x1-standard"]
    by_id = {record["query_id"]: record for record in audit["records"]}
    assert by_id["col-x1-standard"]["reranked_rank_consistent_across_runs"] is False


def test_render_markdown_contains_no_training_banner(synthetic):
    result, cases = synthetic
    audit = build_audit(result, cases)
    text = render_markdown(audit, result_sha="sha256:r", cases_sha="sha256:c")
    assert "只诊断，不训练" in text
    assert "col-x2-standard" in text


def test_adjudication_table_covers_review_statuses():
    statuses = {verdict["status"] for verdict in CHUNK_ADJUDICATIONS.values()}
    assert statuses <= {
        "equivalent_evidence_candidate", "questionable_negative",
        "partial_support", "adjudicated_insufficient",
    }
    for verdict in CHUNK_ADJUDICATIONS.values():
        assert verdict["reason"].strip()


def test_cli_is_idempotent_on_frozen_inputs(tmp_path):
    """Running the CLI twice over the frozen artifact produces identical bytes."""

    root = Path(__file__).resolve().parents[1]
    result = root / "docs/rag_eval/colloquial/dev80_compact32.json"
    cases = root / "docs/rag_eval/colloquial/rag_colloquial_dev80_v1.json"
    if not result.exists() or not cases.exists():
        pytest.skip("frozen Dev80 artifacts unavailable")
    outputs = []
    for index in range(2):
        json_path = tmp_path / f"audit_{index}.json"
        md_path = tmp_path / f"audit_{index}.md"
        completed = subprocess.run(
            [sys.executable, str(root / "scripts/audit_colloquial_ranking_failures.py"),
             "--result", str(result), "--cases", str(cases),
             "--output-json", str(json_path), "--output-md", str(md_path)],
            capture_output=True, text=True, check=True,
        )
        assert completed.returncode == 0
        outputs.append((json_path.read_bytes(), md_path.read_bytes()))
    assert outputs[0] == outputs[1]
