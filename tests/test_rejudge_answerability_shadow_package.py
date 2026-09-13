# -*- coding: utf-8 -*-
"""Fail-closed contracts for answerability judge-only package replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dump_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dump_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def _source_package(tmp_path: Path) -> Path:
    package = tmp_path / "source_v3"
    package.mkdir()
    original_question = "Planner负责执行具体子任务而不是设计路径吗？"
    retrieval_question = "Planner 的职责和行动路径设计"
    traffic_id = "traffic-001"
    candidates = [
        {
            "rank": 1,
            "chunk_id": "private-chunk-1",
            "source": "planner.md",
            "source_type": "resource",
            "heading_path": "Planner职责",
            "rerank_score": 0.997,
            "document": "Planner负责设计行动路径，不负责执行具体子任务。",
            "verdict": {
                "grade": 3,
                "relation": "entails",
                "reason": "旧裁判错误标签",
            },
            "action": "answer",
        },
        {
            "rank": 2,
            "chunk_id": "private-chunk-2",
            "source": "agent.md",
            "source_type": "resource",
            "heading_path": "Agent概述",
            "rerank_score": 0.982,
            "document": "Agent系统通常包含规划与执行组件。",
            "verdict": {
                "grade": 1,
                "relation": "not_established",
                "reason": "旧裁判标签",
            },
            "action": "abstain",
        },
    ]
    routes = [{
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "traffic_id": traffic_id,
        "question_sha256": _sha(original_question),
        "traffic_origin": "agent_generated_reference_probe",
        "stratum": "reference_probe_catalog",
        "producer": "reference_probe_llm",
        "transform": "wrong_relation",
        "seed_question_sha256": "",
        "probe_topic_sha256": "a" * 64,
        "probe_type": "wrong_relation",
        "distribution_eligible": False,
        "decision": "answer",
        "resolver_mode": "llm",
        "planner_engine": "structured_llm",
        "planner_version": "llm-first-v4",
        "route_model": "gpt-5.6-terra",
        "routes": [{
            "source": "reference_kb",
            "operation": "search",
            "required": True,
            "source_role": "knowledge",
        }],
        "has_reference_kb": True,
        "planning_ms": 10.0,
        "execution_ms": 20.0,
        "source_status": {
            "reference_kb": {"status": "ok", "count": 2, "latency_ms": 19.0},
        },
        "reference_captures": 1,
    }]
    public = [{
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "traffic_id": traffic_id,
        "traffic_origin": "agent_generated_reference_probe",
        "stratum": "reference_probe_catalog",
        "producer": "reference_probe_llm",
        "producer_model": "gpt-5.6-terra",
        "transform": "wrong_relation",
        "seed_question_sha256": "",
        "probe_topic_sha256": "a" * 64,
        "probe_type": "wrong_relation",
        "distribution_eligible": False,
        "original_question_sha256": _sha(original_question),
        "retrieval_question_sha256": _sha(retrieval_question),
        "reference_capture": 1,
        "features": {
            "rerank_top": 0.997,
            "rerank_margin": 0.015,
            "high_scorers": 2,
        },
        "gate_decision": True,
        "effective_hit": True,
        "routes": routes[0]["routes"],
        "retrieval_profile": "baseline",
        "index_fingerprint": "index-fingerprint",
        "reranker_requested": "bge-reranker-base",
        "reranker_actual": "bge-reranker-base",
        "reranker_status": "ok",
        "planning_ms": 10.0,
        "execution_ms": 20.0,
        "judge_selected": True,
        "sampling_probability": 1.0,
        "sampling_weight": 1.0,
        "sampling_reason": "all",
        "judge_model_requested": "gpt-5.6-terra",
        "judge_schema": "answerability-v3",
        "judge_depth": 2,
        "verdicts": [
            {
                "rank": candidate["rank"],
                "chunk_id_sha256": _sha(candidate["chunk_id"]),
                "rerank_score": candidate["rerank_score"],
                "grade": candidate["verdict"]["grade"],
                "relation": candidate["verdict"]["relation"],
                "action": candidate["action"],
                "available": True,
                "cache_hit": False,
                "latency_ms": 1.0,
            }
            for candidate in candidates
        ],
        "label_available": True,
        "recommended_action": "answer",
        "selected_rank": 1,
        "intervention_needed": False,
        "intervention_type": "no_change",
    }]
    private = [{
        "traffic_id": traffic_id,
        "question": original_question,
        "retrieval_question": retrieval_question,
        "stratum": "reference_probe_catalog",
        "producer": "reference_probe_llm",
        "transform": "wrong_relation",
        "probe_topic_sha256": "a" * 64,
        "probe_type": "wrong_relation",
        "distribution_eligible": False,
        "candidates": candidates,
        "label_available": True,
        "recommended_action": "answer",
        "selected_rank": 1,
        "intervention_needed": False,
        "intervention_type": "no_change",
    }]
    generated = [{
        "traffic_id": traffic_id,
        "question": original_question,
        "traffic_origin": "agent_generated_reference_probe",
        "stratum": "reference_probe_catalog",
        "producer": "reference_probe_llm",
        "producer_model": "gpt-5.6-terra",
        "transform": "wrong_relation",
        "seed_question_sha256": "",
        "probe_topic_sha256": "a" * 64,
        "probe_type": "wrong_relation",
        "distribution_eligible": False,
    }]
    summary = {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "generated_queries": 1,
        "reference_kb_queries": 1,
        "reference_route_rate": 1.0,
        "route_source_counts": {"reference_kb": 1},
        "judge_selected_queries": 1,
        "label_available_queries": 1,
        "intervention_type_counts": {"no_change": 1},
        "traffic_origin_counts": {"agent_generated_reference_probe": 1},
        "stratum_counts": {"reference_probe_catalog": 1},
        "distribution_eligible_queries": 0,
        "traffic_origin": "agent_generated_reference_probe",
        "release_status": "candidate_discovery_only_not_blind",
        "judge_run": {
            "judge_jobs": 2,
            "judge_cache_hits": 0,
            "judge_unavailable": 0,
        },
    }
    _dump_jsonl(package / "route_outcomes.jsonl", routes)
    _dump_jsonl(package / "claude_agent_calibration.jsonl", public)
    _dump_jsonl(package / "private_audit.jsonl", private)
    _dump_json(package / "summary.json", summary)
    (package / "CLAUDE_HANDOFF.md").write_text("old v3 handoff\n", encoding="utf-8")
    _dump_jsonl(package / "generated_queries_private.jsonl", generated)
    required = (
        "route_outcomes.jsonl",
        "claude_agent_calibration.jsonl",
        "private_audit.jsonl",
        "summary.json",
        "CLAUDE_HANDOFF.md",
        "generated_queries_private.jsonl",
    )
    manifest = {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "created_at": "2026-08-28T00:00:00+08:00",
        "traffic_origin": "agent_generated_reference_probe",
        "release_status": "candidate_discovery_only_not_blind",
        "seed_lineage": {"used_by_producer": False, "reason": "probe"},
        "probe_lineage": {"input_kind": "test_probe"},
        "producer": "reference_probe",
        "producer_model": "gpt-5.6-terra",
        "probe_profile": "mixed",
        "judge_mode": "all",
        "judge_control_rate": 0.2,
        "judge_model_requested": "gpt-5.6-terra",
        "judge_schema": "answerability-v3",
        "judge_depth": 2,
        "random_seed": 20260828,
        "git": {"head": "source-head", "dirty": True},
        "files": {
            name: {
                "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest(),
                "bytes": (package / name).stat().st_size,
            }
            for name in required
        },
    }
    _dump_json(package / "manifest.json", manifest)
    return package


def _v4_runner(observed):
    def run(rows, *, depth, workers, model):
        assert depth == 2
        assert workers >= 1
        assert model == "gpt-5.6-terra"
        for row in rows:
            observed.append({
                "question": row["question"],
                "retrieval_question": row["retrieval_question"],
                "judge_question_role": row["judge_question_role"],
            })
            row["verdicts"] = [
                {
                    "rank": 1,
                    "chunk_id": row["candidates_private"][0]["chunk_id"],
                    "rerank_score": row["candidates_private"][0]["rerank_score"],
                    "verdict": {
                        "grade": 3,
                        "relation": "contradicts",
                        "question_form": "polar",
                        "direct_answer": "proposition_false",
                        "premise_status": "refuted",
                        "reason": "资料明确反驳命题",
                    },
                    "action": "correct_premise",
                    "cache_hit": False,
                    "latency_ms": 2.0,
                },
                {
                    "rank": 2,
                    "chunk_id": row["candidates_private"][1]["chunk_id"],
                    "rerank_score": row["candidates_private"][1]["rerank_score"],
                    "verdict": {
                        "grade": 1,
                        "relation": "not_established",
                        "question_form": "polar",
                        "direct_answer": "unknown",
                        "premise_status": "not_established",
                        "reason": "只提到相关主题",
                    },
                    "action": "abstain",
                    "cache_hit": False,
                    "latency_ms": 2.0,
                },
            ]
        return {
            "judge_jobs": len(rows) * depth,
            "judge_cache_hits": 0,
            "judge_unavailable": 0,
        }

    return run


def test_judge_only_replay_is_atomic_private_and_uses_original_question(
    tmp_path, monkeypatch,
):
    import rag_answerability
    from scripts.rejudge_answerability_shadow_package import rejudge_package

    monkeypatch.setattr(rag_answerability, "SCHEMA", "answerability-v4")
    source = _source_package(tmp_path)
    output = tmp_path / "rejudged_v4"
    source_routes = (source / "route_outcomes.jsonl").read_bytes()
    source_generated = (source / "generated_queries_private.jsonl").read_bytes()
    source_manifest = (source / "manifest.json").read_bytes()
    observed = []

    result = rejudge_package(
        source,
        output,
        judge_model="gpt-5.6-terra",
        judge_workers=2,
        judge_runner=_v4_runner(observed),
    )

    assert result["output_dir"] == str(output)
    assert observed == [{
        "question": "Planner负责执行具体子任务而不是设计路径吗？",
        "retrieval_question": "Planner 的职责和行动路径设计",
        "judge_question_role": "original_user_question",
    }]
    assert (output / "route_outcomes.jsonl").read_bytes() == source_routes
    assert (output / "generated_queries_private.jsonl").read_bytes() == source_generated
    assert (source / "manifest.json").read_bytes() == source_manifest

    public = json.loads(
        (output / "claude_agent_calibration.jsonl").read_text(encoding="utf-8")
    )
    assert public["judge_schema"] == "answerability-v4"
    assert public["judge_question_role"] == "original_user_question"
    assert public["intervention_type"] == "correct_premise"
    assert public["verdicts"][0]["question_form"] == "polar"
    assert public["verdicts"][0]["direct_answer"] == "proposition_false"
    assert public["verdicts"][0]["premise_status"] == "refuted"
    assert "reason" not in _all_keys(public)
    assert "Planner负责" not in json.dumps(public, ensure_ascii=False)

    private = json.loads(
        (output / "private_audit.jsonl").read_text(encoding="utf-8")
    )
    assert private["question"].startswith("Planner负责")
    assert private["candidates"][0]["verdict"]["reason"] == "资料明确反驳命题"

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_schema"] == "answerability-v4"
    assert manifest["judge_question_role"] == "original_user_question"
    lineage = manifest["rejudge_lineage"]
    assert lineage["planner_rerun"] is False
    assert lineage["retrieval_rerun"] is False
    assert lineage["source_judge_schema"] == "answerability-v3"
    assert lineage["target_judge_schema"] == "answerability-v4"
    assert lineage["source_git"] == {"head": "source-head", "dirty": True}
    assert len(lineage["source_manifest_sha256"]) == 64
    handoff = (output / "CLAUDE_HANDOFF.md").read_text(encoding="utf-8")
    assert "planner, router, retriever, reranker" in handoff
    assert "original_user_question" in handoff
    assert "stale" in handoff


def test_rejudge_refuses_tampered_source_before_judging(tmp_path, monkeypatch):
    import rag_answerability
    from scripts.rejudge_answerability_shadow_package import (
        RejudgePackageError,
        rejudge_package,
    )

    monkeypatch.setattr(rag_answerability, "SCHEMA", "answerability-v4")
    source = _source_package(tmp_path)
    with (source / "private_audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    called = []

    with pytest.raises(RejudgePackageError, match="byte count changed"):
        rejudge_package(
            source,
            tmp_path / "must_not_exist",
            judge_runner=_v4_runner(called),
        )

    assert called == []
    assert not (tmp_path / "must_not_exist").exists()


def test_rejudge_refuses_original_question_sha_mismatch(tmp_path, monkeypatch):
    import rag_answerability
    from scripts.rejudge_answerability_shadow_package import (
        RejudgePackageError,
        rejudge_package,
    )

    monkeypatch.setattr(rag_answerability, "SCHEMA", "answerability-v4")
    source = _source_package(tmp_path)
    private_path = source / "private_audit.jsonl"
    row = json.loads(private_path.read_text(encoding="utf-8"))
    row["question"] = "被篡改的原始问题？"
    _dump_jsonl(private_path, [row])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][private_path.name] = {
        "sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "bytes": private_path.stat().st_size,
    }
    _dump_json(source / "manifest.json", manifest)

    with pytest.raises(RejudgePackageError, match="private/generated original question"):
        rejudge_package(
            source,
            tmp_path / "must_not_exist",
            judge_runner=_v4_runner([]),
        )

    assert not (tmp_path / "must_not_exist").exists()


def test_rejudge_never_overwrites_an_existing_output(tmp_path, monkeypatch):
    import rag_answerability
    from scripts.rejudge_answerability_shadow_package import (
        RejudgePackageError,
        rejudge_package,
    )

    monkeypatch.setattr(rag_answerability, "SCHEMA", "answerability-v4")
    source = _source_package(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RejudgePackageError, match="already exists"):
        rejudge_package(source, output, judge_runner=_v4_runner([]))

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_shared_judge_runner_uses_original_question_and_retrieval_scope(monkeypatch):
    """Prevent a future replay from judging only the planner-neutralised query."""
    import rag_answerability
    from scripts.run_answerability_shadow_traffic_agent import judge_selected_rows

    observed = []

    def fake_grade(question, chunk, *, model, retrieval_question):
        observed.append((question, retrieval_question, chunk, model))
        return {
            "grade": 3,
            "question_form": "polar",
            "direct_answer": "proposition_false",
            "premise_status": "refuted",
            "relation": "contradicts",
            "reason": "资料明确反驳命题",
        }

    monkeypatch.setattr(rag_answerability, "grade", fake_grade)
    monkeypatch.setattr(rag_answerability, "cache_key", lambda *args, **kwargs: "k")
    monkeypatch.setattr(rag_answerability, "_load_cache", lambda: {})
    monkeypatch.setattr(rag_answerability, "flush_cache", lambda: None)
    rows = [{
        "judge_selected": True,
        "question": "Planner负责执行任务而不是设计路径吗？",
        "retrieval_question": "Planner 的职责和行动路径设计",
        "candidates_private": [{
            "rank": 1,
            "chunk_id": "chunk-1",
            "rerank_score": 0.99,
            "document": "Planner负责设计路径，不负责执行具体任务。",
        }],
        "verdicts": [],
    }]

    result = judge_selected_rows(
        rows, depth=1, workers=1, model="gpt-5.6-terra",
    )

    assert result["judge_jobs"] == 1
    assert observed == [(
        "Planner负责执行任务而不是设计路径吗？",
        "Planner 的职责和行动路径设计",
        "Planner负责设计路径，不负责执行具体任务。",
        "gpt-5.6-terra",
    )]
    assert rows[0]["verdicts"][0]["action"] == "correct_premise"
