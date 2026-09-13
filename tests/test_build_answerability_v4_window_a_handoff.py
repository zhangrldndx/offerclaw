# -*- coding: utf-8 -*-
"""Contracts for the public-only v4 candidate/protocol freeze builder."""

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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _verdict(
    rank: int, *, grade: int, question_form: str, direct_answer: str,
    premise_status: str, relation: str, action: str,
) -> dict:
    return {
        "rank": rank,
        "chunk_id_sha256": _sha(f"chunk-{rank}"),
        "rerank_score": 0.99 - rank / 100,
        "grade": grade,
        "question_form": question_form,
        "direct_answer": direct_answer,
        "premise_status": premise_status,
        "relation": relation,
        "action": action,
        "available": True,
        "cache_hit": False,
        "latency_ms": 1.0,
    }


def _calibration_row(
    traffic_id: str, *, probe_type: str, gate: bool,
    intervention_type: str, recommended_action: str,
    selected_rank: int | None, verdicts: list[dict],
    features: dict | None = None,
) -> dict:
    default_features = {
        "rerank_margin": 0.02,
        "rerank_top": 0.98,
        "high_scorers": 3,
        "gate_score_minus_threshold": 0.13,
    }
    routes = [{
        "source": "reference_kb",
        "operation": "search",
        "required": True,
        "source_role": "knowledge",
    }]
    return {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "traffic_id": traffic_id,
        "traffic_origin": "agent_generated_reference_probe",
        "stratum": "reference_probe",
        "producer": "reference_probe_llm",
        "producer_model": "gpt-5.6-terra",
        "transform": "fixture",
        "seed_question_sha256": "",
        "probe_topic_sha256": _sha(traffic_id),
        "probe_type": probe_type,
        "distribution_eligible": False,
        "original_question_sha256": _sha(f"original-{traffic_id}"),
        "retrieval_question_sha256": _sha(f"retrieval-{traffic_id}"),
        "reference_capture": 1,
        "features": features or default_features,
        "gate_decision": gate,
        "effective_hit": gate,
        "routes": routes,
        "retrieval_profile": "baseline",
        "index_fingerprint": "index-v4",
        "reranker_requested": "BAAI/bge-reranker-base",
        "reranker_actual": "BAAI/bge-reranker-base",
        "reranker_status": "ok",
        "planning_ms": 1.0,
        "execution_ms": 2.0,
        "judge_selected": True,
        "sampling_probability": 1.0,
        "sampling_weight": 1.0,
        "sampling_reason": "all",
        "judge_model_requested": "gpt-5.6-terra",
        "judge_schema": "answerability-v4",
        "judge_question_role": "original_user_question",
        "judge_depth": 2,
        "verdicts": verdicts,
        "label_available": True,
        "recommended_action": recommended_action,
        "selected_rank": selected_rank,
        "intervention_needed": intervention_type != "no_change",
        "intervention_type": intervention_type,
    }


def _package(tmp_path: Path, *, kind: str) -> Path:
    package = tmp_path / f"{kind}_v4"
    package.mkdir()
    answer = _verdict(
        1, grade=3, question_form="open", direct_answer="not_applicable",
        premise_status="none", relation="entails", action="answer",
    )
    abstain = _verdict(
        2, grade=1, question_form="open", direct_answer="not_applicable",
        premise_status="none", relation="not_established", action="abstain",
    )
    if kind == "mixed":
        rows = [
            _calibration_row(
                "mixed-block", probe_type="adjacent_topic", gate=True,
                intervention_type="block_unsupported", recommended_action="abstain",
                selected_rank=None,
                verdicts=[
                    _verdict(
                        1, grade=1, question_form="open",
                        direct_answer="not_applicable", premise_status="none",
                        relation="not_established", action="abstain",
                    ),
                    abstain,
                ],
            ),
            _calibration_row(
                "mixed-ok", probe_type="covered", gate=True,
                intervention_type="no_change", recommended_action="answer",
                selected_rank=1, verdicts=[answer, abstain],
                features={
                    "rerank_margin": 0.002,
                    "rerank_top": 0.995,
                    "high_scorers": 5,
                    "gate_score_minus_threshold": 0.145,
                },
            ),
        ]
        probe_profile = "mixed"
        probe_lineage = {
            "document_text_visible": False,
            "input_kind": "curated_catalog_metadata_only",
        }
    else:
        refute = _verdict(
            1, grade=3, question_form="polar",
            direct_answer="proposition_false", premise_status="refuted",
            relation="contradicts", action="correct_premise",
        )
        rank1_bad = _verdict(
            1, grade=1, question_form="open", direct_answer="not_applicable",
            premise_status="none", relation="not_established", action="abstain",
        )
        rank2_answer = _verdict(
            2, grade=3, question_form="open", direct_answer="not_applicable",
            premise_status="none", relation="entails", action="answer",
        )
        rows = [
            _calibration_row(
                "evidence-positive", probe_type="evidence_positive", gate=True,
                intervention_type="no_change", recommended_action="answer",
                selected_rank=1, verdicts=[answer, abstain],
            ),
            _calibration_row(
                "evidence-wrong", probe_type="evidence_wrong_relation", gate=True,
                intervention_type="correct_premise",
                recommended_action="correct_premise", selected_rank=1,
                verdicts=[refute, abstain],
            ),
            _calibration_row(
                "evidence-rerank", probe_type="evidence_positive", gate=True,
                intervention_type="rerank_evidence", recommended_action="answer",
                selected_rank=2, verdicts=[rank1_bad, rank2_answer],
            ),
        ]
        probe_profile = "evidence_mixed"
        probe_lineage = {
            "document_text_visible": True,
            "evidence_seeded": True,
            "input_kind": "curated_evidence_seeded_synthetic_probe",
        }

    routes = [{
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "traffic_id": row["traffic_id"],
        "question_sha256": row["original_question_sha256"],
        "traffic_origin": "agent_generated_reference_probe",
        "stratum": row["stratum"],
        "producer": row["producer"],
        "transform": row["transform"],
        "seed_question_sha256": "",
        "probe_topic_sha256": row["probe_topic_sha256"],
        "probe_type": row["probe_type"],
        "distribution_eligible": False,
        "decision": "answer",
        "resolver_mode": "llm",
        "planner_engine": "structured_llm",
        "planner_version": "llm-first-v4",
        "route_model": "gpt-5.6-terra",
        "routes": row["routes"],
        "has_reference_kb": True,
        "planning_ms": 1.0,
        "execution_ms": 2.0,
        "reference_captures": 1,
    } for row in rows]
    summary = {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "generated_queries": len(routes),
        "reference_kb_queries": len(rows),
        "reference_route_rate": 1.0,
        "route_source_counts": {"reference_kb": len(rows)},
        "judge_selected_queries": len(rows),
        "label_available_queries": len(rows),
        "intervention_type_counts": dict({}),
        "traffic_origin_counts": {"agent_generated_reference_probe": len(rows)},
        "stratum_counts": {"reference_probe": len(rows)},
        "distribution_eligible_queries": 0,
        "traffic_origin": "agent_generated_reference_probe",
        "release_status": "candidate_discovery_only_not_blind",
        "judge_question_role": "original_user_question",
        "judge_run": {
            "judge_jobs": len(rows) * 2,
            "judge_cache_hits": 0,
            "judge_unavailable": 0,
        },
    }
    _dump_jsonl(package / "route_outcomes.jsonl", routes)
    _dump_jsonl(package / "claude_agent_calibration.jsonl", rows)
    _dump_json(package / "summary.json", summary)
    (package / "CLAUDE_HANDOFF.md").write_text("public v4 handoff\n", encoding="utf-8")
    # Deliberately invalid JSON proves the builder hashes but never parses these.
    (package / "private_audit.jsonl").write_bytes(b"PRIVATE NOT JSON\n")
    (package / "generated_queries_private.jsonl").write_bytes(b"GENERATED NOT JSON\n")

    artifacts = (
        "route_outcomes.jsonl", "claude_agent_calibration.jsonl",
        "private_audit.jsonl", "summary.json", "CLAUDE_HANDOFF.md",
        "generated_queries_private.jsonl",
    )
    manifest = {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "created_at": "2026-08-28T00:00:00+08:00",
        "traffic_origin": "agent_generated_reference_probe",
        "release_status": "candidate_discovery_only_not_blind",
        "seed_lineage": {"used_by_producer": False, "reason": "bounded_enum"},
        "probe_lineage": probe_lineage,
        "producer": "reference_probe",
        "producer_model": "gpt-5.6-terra",
        "probe_profile": probe_profile,
        "judge_mode": "all",
        "judge_control_rate": 0.2,
        "judge_model_requested": "gpt-5.6-terra",
        "judge_schema": "answerability-v4",
        "judge_question_role": "original_user_question",
        "judge_depth": 2,
        "random_seed": 1,
        "git": {"head": "head", "dirty": True},
        "rejudge_lineage": {
            "operation": "judge_only_replay",
            "planner_rerun": False,
            "retrieval_rerun": False,
            "source_judge_schema": "answerability-v3",
            "target_judge_schema": "answerability-v4",
            "judge_question_role": "original_user_question",
        },
        "files": {
            name: {
                "bytes": (package / name).stat().st_size,
                "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest(),
            }
            for name in artifacts
        },
    }
    _dump_json(package / "manifest.json", manifest)
    return package


def test_builder_freezes_protocol_without_parsing_private_files(tmp_path):
    from scripts.build_answerability_v4_window_a_handoff import build_handoff

    mixed = _package(tmp_path, kind="mixed")
    evidence = _package(tmp_path, kind="evidence")
    output = tmp_path / "freeze"

    result = build_handoff(mixed, evidence, output)

    assert set(path.name for path in output.iterdir()) == {
        "INPUT_AUDIT.md",
        "CANDIDATE_SET_FREEZE.md",
        "ORGANIC_WINDOW_A_PREREGISTRATION.md",
        "ORGANIC_WINDOW_A_PROTOCOL.json",
        "CLAUDE_HANDOFF.md",
        "analysis_manifest.json",
    }
    protocol_path = output / "ORGANIC_WINDOW_A_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert result["protocol_sha256"] == protocol_sha
    assert protocol["protocol_id"] == "answerability-organic-v4"
    assert protocol["window_a"]["sampling_mode"] == "accepted_control"
    assert protocol["window_a"]["eligible_requests"] == 200
    reject = protocol["window_a"]["gate_reject_control"]
    assert reject["judge_probability"] == 0.05
    assert reject["sampling"] == "runtime_bernoulli_rng"
    assert "sampling_seed" not in reject
    assert reject["required_row_fields"] == [
        "sampling_probability", "judge_selected",
    ]
    assert protocol["window_a"]["gate_accept"] == {
        "judge_probability": 1.0, "judge_depth": 2,
    }
    assert protocol["feasibility_point_estimates"][
        "conditional_reference_call_rate_formula"
    ] == "gate_true_rule_hits / 200 eligible_reference_requests"
    assert protocol["selection_algorithm"]["eligible_rules"] == ["H1", "H2", "H3"]
    assert protocol["candidate_rules"]["SAT"]["selection_eligible"] is False
    assert protocol["minimum_class_counts"] == {
        "accepted_intervention": 20,
        "accepted_no_change": 20,
        "if_below_either": "underpowered_stop_without_rule_selection",
    }

    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == protocol_sha
    assert manifest["runtime_binding"]["protocol_id"] == "answerability-organic-v4"
    assert manifest["runtime_binding"]["required_value"] == protocol_sha
    assert manifest["inputs"]["mixed_v4"]["private_files_parsed"] is False
    assert manifest["inputs"]["evidence_v4"]["private_files_parsed"] is False
    assert manifest["mixed_accepted_path"]["rules"]["SAT"] == {
        "triggered_intervention": 0,
        "total_intervention": 1,
        "triggered_no_change": 1,
        "total_no_change": 1,
    }
    assert set(manifest["outputs"]) == {
        "INPUT_AUDIT.md", "CANDIDATE_SET_FREEZE.md",
        "ORGANIC_WINDOW_A_PREREGISTRATION.md",
        "ORGANIC_WINDOW_A_PROTOCOL.json", "CLAUDE_HANDOFF.md",
    }
    for name, metadata in manifest["outputs"].items():
        assert metadata["bytes"] == (output / name).stat().st_size
        assert metadata["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    handoff = (output / "CLAUDE_HANDOFF.md").read_text(encoding="utf-8")
    assert "GO_OPEN_WINDOW_A" in handoff
    assert "NO_GO_PROTOCOL" in handoff
    assert "answerability-organic-v4" in handoff
    assert "There is no organic Window A dataset yet" in handoff
    assert "NO_GO_UNDERPOWERED" in handoff
    assert "v3 analysis is stale" in handoff


def test_builder_refuses_tampered_declared_artifact(tmp_path):
    from scripts.build_answerability_v4_window_a_handoff import (
        FreezeBuildError,
        build_handoff,
    )

    mixed = _package(tmp_path, kind="mixed")
    evidence = _package(tmp_path, kind="evidence")
    with (mixed / "private_audit.jsonl").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(FreezeBuildError, match="byte count changed"):
        build_handoff(mixed, evidence, tmp_path / "must_not_exist")

    assert not (tmp_path / "must_not_exist").exists()


def test_builder_refuses_private_field_in_public_calibration(tmp_path):
    from scripts.build_answerability_v4_window_a_handoff import (
        FreezeBuildError,
        build_handoff,
    )

    mixed = _package(tmp_path, kind="mixed")
    evidence = _package(tmp_path, kind="evidence")
    calibration_path = mixed / "claude_agent_calibration.jsonl"
    rows = [json.loads(line) for line in calibration_path.read_text().splitlines()]
    rows[0]["question"] = "raw private question"
    _dump_jsonl(calibration_path, rows)
    manifest_path = mixed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][calibration_path.name] = {
        "bytes": calibration_path.stat().st_size,
        "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    }
    _dump_json(manifest_path, manifest)

    with pytest.raises(FreezeBuildError, match="contains private keys: question"):
        build_handoff(mixed, evidence, tmp_path / "must_not_exist")


def test_builder_refuses_lineage_mismatch_and_existing_output(tmp_path):
    from scripts.build_answerability_v4_window_a_handoff import (
        FreezeBuildError,
        build_handoff,
    )

    mixed = _package(tmp_path, kind="mixed")
    evidence = _package(tmp_path, kind="evidence")
    calibration_path = evidence / "claude_agent_calibration.jsonl"
    rows = [json.loads(line) for line in calibration_path.read_text().splitlines()]
    rows[0]["index_fingerprint"] = "different-index"
    _dump_jsonl(calibration_path, rows)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][calibration_path.name] = {
        "bytes": calibration_path.stat().st_size,
        "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    }
    _dump_json(manifest_path, manifest)

    with pytest.raises(FreezeBuildError, match="do not share one non-empty index_fingerprint"):
        build_handoff(mixed, evidence, tmp_path / "must_not_exist")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FreezeBuildError, match="already exists"):
        build_handoff(mixed, evidence, existing)
