# -*- coding: utf-8 -*-
"""Unit tests for Train false-winner mining and reporting (no model loading)."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.mine_colloquial_train_false_winners as mining  # noqa: E402


def _stage_candidate(chunk_id, rank, score, source="doc_a.md"):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "heading_path": "正文",
        "rank": rank,
        "dense_distance": None,
        "bm25_score": None,
        "rrf_score": None,
        "rerank_score": score,
    }


def _case(query_id, anchor_id, style="standard", hard_negative_id=None):
    return {
        "query_id": query_id,
        "anchor_id": anchor_id,
        "split": "train",
        "case_kind": "positive",
        "domain": "llm_app",
        "query_style": style,
        "question": f"问题 {query_id}",
        "review_status": "approved",
        "answer_requirements": ["要求"],
        "relevant_targets": [{
            "chunk_id": "gold",
            "source": "doc_a.md",
            "heading_path": ["正文"],
            "relevance_grade": 3,
            "supported_requirements": ["要求"],
            "evidence_excerpt": "证据",
            "evidence_span_hash": "sha256:" + "0" * 64,
        }],
        "hard_negatives": (
            [{"chunk_id": hard_negative_id, "source": "doc_a.md", "reason": "V1 理由"}]
            if hard_negative_id else []
        ),
    }


def _trace_row(query_id, anchor_id, reranked, style="standard"):
    stages = {
        "dense": reranked,
        "bm25": reranked,
        "fusion": reranked,
        "reranked": reranked,
    }
    return {
        "query_id": query_id,
        "anchor_id": anchor_id,
        "domain": "llm_app",
        "query_style": style,
        "question": f"问题 {query_id}",
        "grade3_chunk_ids": ["gold"],
        "all_relevant_chunk_ids": ["gold"],
        "stages": stages,
        "stage_gold_ranks": {
            stage: mining._gold_rank(candidates, {"gold"})
            for stage, candidates in stages.items()
        },
        "gate_decision": True,
        "latency_ms": 1.0,
    }


def test_load_train_positives_rejects_blind_paths(tmp_path):
    path = tmp_path / "rag_colloquial_blind_v1.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        mining._load_train_positives(path)


def test_load_train_positives_rejects_dev_only_files(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"items": [
        {"split": "dev", "case_kind": "positive", "review_status": "approved"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no positive train rows"):
        mining._load_train_positives(path)


def test_select_candidates_prioritizes_false_winner_and_excludes_relevant():
    case = _case("q1", "a1")
    row = _trace_row("q1", "a1", [
        _stage_candidate("wrong_top1", 1, 0.99),
        _stage_candidate("gold", 2, 0.95),
        _stage_candidate("near_b", 3, 0.94, source="doc_b.md"),
        _stage_candidate("far", 12, 0.10),
    ])
    selected = mining._select_candidates(row, case)
    roles = {candidate["chunk_id"]: candidate["role"] for candidate in selected}
    assert roles["wrong_top1"] == "current_false_winner"
    assert roles["near_b"] == "near_margin_competitor"
    assert "gold" not in roles
    assert "far" not in roles  # outside rank<=3 and margin window
    winner = next(c for c in selected if c["chunk_id"] == "wrong_top1")
    assert winner["relation_to_gold"] == "same_document_wrong_section"
    assert winner["margin_vs_gold"] == pytest.approx(0.04)
    cross = next(c for c in selected if c["chunk_id"] == "near_b")
    assert cross["relation_to_gold"] == "cross_document_near_topic"


def _write_report_inputs(tmp_path, rows, cases, verdicts=None):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps({"items": cases, "index": {"collection": "c", "count": 1}},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    traces_path = tmp_path / "traces.json"
    traces_path.write_text(json.dumps({
        "schema_version": mining.MINING_SCHEMA_VERSION,
        "arm": "compact32",
        "cases_sha256": mining._sha256_file(cases_path),
        "train_cases_sha256": mining.train_cases_fingerprint(
            [row for row in cases
             if row.get("split") == "train" and row.get("case_kind") == "positive"]
        ),
        "index": {"collection": "c", "count": 1},
        "rows": rows,
    }, ensure_ascii=False), encoding="utf-8")
    adjudication_path = tmp_path / "adjudication.json"
    if verdicts is not None:
        adjudication_path.write_text(
            json.dumps({"verdicts": verdicts}, ensure_ascii=False), encoding="utf-8",
        )
    return cases_path, traces_path, adjudication_path


def test_report_defaults_to_needs_adjudication_and_gate(tmp_path):
    rows = [_trace_row("q1", "a1", [
        _stage_candidate("wrong_top1", 1, 0.99),
        _stage_candidate("gold", 2, 0.95),
    ])]
    cases = [_case("q1", "a1")]
    cases_path, traces_path, adjudication_path = _write_report_inputs(
        tmp_path, rows, cases,
    )
    args = mining.parser().parse_args([
        "report", "--traces", str(traces_path), "--cases", str(cases_path),
        "--adjudication", str(adjudication_path),
        "--output-json", str(tmp_path / "report.json"),
        "--output-md", str(tmp_path / "report.md"),
    ])
    mining.report(args)
    artifact = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    summary = artifact["summary"]
    # Without an adjudicated verdict nothing becomes a hard negative.
    assert summary["candidate_status_counts"] == {"needs_adjudication": 1}
    assert summary["hard_negative_pairs"]["total"] == 0
    assert summary["data_quality_gate"]["verdict"] == "insufficient_pairs"
    assert summary["false_winner_groups"] == 1


def test_report_applies_overlay_and_v1_carry(tmp_path):
    rows = [
        _trace_row("q1", "a1", [
            _stage_candidate("wrong_top1", 1, 0.99),
            _stage_candidate("gold", 2, 0.95),
        ]),
        _trace_row("q2", "a2", [
            _stage_candidate("gold", 1, 0.99),
            _stage_candidate("v1neg", 2, 0.90),
        ], style="natural"),
    ]
    cases = [_case("q1", "a1"), _case("q2", "a2", style="natural",
                                      hard_negative_id="v1neg")]
    verdicts = [{
        "anchor_id": "a1", "chunk_id": "wrong_top1",
        "status": "hard_negative", "reason": "复核确认不足以回答。",
    }]
    cases_path, traces_path, adjudication_path = _write_report_inputs(
        tmp_path, rows, cases, verdicts,
    )
    args = mining.parser().parse_args([
        "report", "--traces", str(traces_path), "--cases", str(cases_path),
        "--adjudication", str(adjudication_path),
        "--output-json", str(tmp_path / "report.json"),
        "--output-md", str(tmp_path / "report.md"),
    ])
    mining.report(args)
    artifact = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    summary = artifact["summary"]
    assert summary["candidate_status_counts"]["hard_negative"] == 2
    pairs = summary["hard_negative_pairs"]
    assert pairs["total"] == 2
    assert pairs["base_model_wrong"] == 1   # wrong_top1 above gold
    assert pairs["base_model_correct"] == 1  # v1neg below gold
    by_id = {group["query_id"]: group for group in artifact["groups"]}
    assert by_id["q2"]["group_kind"] == "stability_anchor"
    carried = by_id["q2"]["candidates"][0]
    assert carried["status"] == "hard_negative"
    assert "V1 双审已裁决" in carried["adjudication_reason"]


def test_report_rejects_stale_legacy_traces(tmp_path):
    """Pre-migration artifacts have no Train fingerprint and fall back to the
    whole-file hash, which must still fail closed."""

    rows = [_trace_row("q1", "a1", [_stage_candidate("gold", 1, 0.99)])]
    cases = [_case("q1", "a1")]
    cases_path, traces_path, _ = _write_report_inputs(tmp_path, rows, cases)
    stale = json.loads(traces_path.read_text(encoding="utf-8"))
    stale.pop("train_cases_sha256")
    stale["cases_sha256"] = "sha256:" + "f" * 64
    traces_path.write_text(json.dumps(stale), encoding="utf-8")
    args = mining.parser().parse_args([
        "report", "--traces", str(traces_path), "--cases", str(cases_path),
        "--adjudication", str(tmp_path / "missing.json"),
        "--output-json", str(tmp_path / "report.json"),
        "--output-md", str(tmp_path / "report.md"),
    ])
    with pytest.raises(ValueError, match="different Train cases"):
        mining.report(args)


def test_adjudication_overlay_rejects_unknown_status(tmp_path):
    path = tmp_path / "adjudication.json"
    path.write_text(json.dumps({"verdicts": [{
        "anchor_id": "a1", "chunk_id": "x", "status": "negative", "reason": "r",
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown adjudication status"):
        mining._load_adjudication(path)


def test_train_fingerprint_ignores_dev_only_gold_revisions(tmp_path):
    """A Dev-only gold revision must not invalidate mined Train traces."""

    train = _case("q1", "a1")
    dev = {**_case("q2", "a2"), "split": "dev"}
    before = mining.train_cases_fingerprint([train])
    # revising the Dev row's gold leaves the Train fingerprint untouched
    dev["relevant_targets"].append({
        "chunk_id": "equivalent",
        "source": "doc_a.md",
        "heading_path": ["正文"],
        "relevance_grade": 3,
        "supported_requirements": ["要求"],
        "evidence_excerpt": "另一段等价证据",
        "evidence_span_hash": "sha256:" + "1" * 64,
    })
    assert mining.train_cases_fingerprint([train]) == before
    # but changing a Train row's own gold does invalidate it
    train["relevant_targets"][0]["chunk_id"] = "other_gold"
    assert mining.train_cases_fingerprint([train]) != before


def test_report_rejects_traces_from_different_train_cases(tmp_path):
    rows = [_trace_row("q1", "a1", [_stage_candidate("gold", 1, 0.99)])]
    cases = [_case("q1", "a1")]
    cases_path, traces_path, _ = _write_report_inputs(tmp_path, rows, cases)
    stale = json.loads(traces_path.read_text(encoding="utf-8"))
    stale["train_cases_sha256"] = "sha256:" + "f" * 64
    traces_path.write_text(json.dumps(stale), encoding="utf-8")
    args = mining.parser().parse_args([
        "report", "--traces", str(traces_path), "--cases", str(cases_path),
        "--adjudication", str(tmp_path / "missing.json"),
        "--output-json", str(tmp_path / "report.json"),
        "--output-md", str(tmp_path / "report.md"),
    ])
    with pytest.raises(ValueError, match="different Train cases"):
        mining.report(args)
