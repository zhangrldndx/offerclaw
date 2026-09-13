# -*- coding: utf-8 -*-
"""Contracts for the frozen Organic Window A Stage 2 analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.private_artifact


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = (
    ROOT / "logs" / "rag_eval" / "answerability_shadow"
    / "claude_reference_probe_v4_freeze_20260828_codex"
    / "ORGANIC_WINDOW_A_PROTOCOL.json"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _answer(rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id_sha256": _sha(f"chunk-{rank}"),
        "rerank_score": 0.99 - rank / 100,
        "verdict": {
            "grade": 3,
            "question_form": "open",
            "direct_answer": "not_applicable",
            "premise_status": "none",
            "relation": "entails",
        },
        "action": "answer",
        "available": True,
        "failure_type": None,
        "latency_ms": 1.0,
    }


def _abstain(rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id_sha256": _sha(f"chunk-{rank}"),
        "rerank_score": 0.99 - rank / 100,
        "verdict": {
            "grade": 1,
            "question_form": "open",
            "direct_answer": "not_applicable",
            "premise_status": "none",
            "relation": "not_established",
        },
        "action": "abstain",
        "available": True,
        "failure_type": None,
        "latency_ms": 1.0,
    }


def _row(
    index: int, *, gate: bool, selected: bool | None = None,
    label: str = "no_change",
) -> dict:
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if selected is None:
        selected = gate
    if label == "block_unsupported":
        verdicts = [_abstain(1), _abstain(2)]
    else:
        verdicts = [_answer(1), _abstain(2)]
    if not selected:
        verdicts = []
    if gate and label == "block_unsupported":
        features = {
            "rerank_margin": 0.02,
            "rerank_top": 0.995,
            "high_scorers": 3,
            "gate_score_minus_threshold": 0.15,
        }
    elif gate:
        features = {
            "rerank_margin": 0.001,
            "rerank_top": 0.995,
            "high_scorers": 5,
            "gate_score_minus_threshold": 0.15,
        }
    else:
        features = {
            "rerank_margin": 0.001,
            "rerank_top": 0.80,
            "high_scorers": 1,
            "gate_score_minus_threshold": -0.10,
        }
    features["gate_decision"] = gate
    return {
        "schema_version": "shadow-answerability-v3",
        "question_sha256": _sha(f"q-{index}"),
        "original_question_sha256": _sha(f"q-{index}"),
        "retrieval_question_sha256": _sha(f"rq-{index}"),
        "features": features,
        "gate_decision": gate,
        "triggered": False,
        "rule_prediction": None,
        "control_sample": bool(selected and not gate),
        "trigger_calibrated": False,
        "judge_selected": selected,
        "sampling_probability": 1.0 if gate else 0.05,
        "sampling_weight": 1.0 if gate else (20.0 if selected else None),
        "sampling_reason": "all_gate_accepts" if gate else "random_gate_reject_control",
        "sampling_mode": "accepted_control",
        "sampling_config_valid": True,
        "sampling_config_errors": [],
        "window_id": "organic_window_a_test",
        "window_phase": "calibration_a",
        "protocol_id": "answerability-organic-v4",
        "protocol_sha256": protocol_sha,
        "frozen_rule_id": None,
        "frozen_rule_sha256": None,
        "judge_question_role": "original_user_question",
        "observed_at_ms": index,
        "retrieval_profile": "baseline",
        "index_fingerprint": "index-v4",
        "reranker_requested": "BAAI/bge-reranker-base",
        "reranker_actual": "BAAI/bge-reranker-base",
        "reranker_status": "ok",
        "route": "reference_kb",
        "traffic_origin": "organic",
        "distribution_eligible": True,
        "judge_depth": 2,
        "queue_status": "enqueued",
        "candidate_count": 2,
        "judge_schema": "answerability-v4",
        "verdicts": verdicts,
        "label_available": selected,
        "top2_complete": selected,
        "judge_started_ms": index,
        "judged_at_ms": index + 1 if selected else None,
        "queue_wait_ms": 0,
        "judge_latency_ms": 1 if selected else None,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_empty_window_reports_collecting_without_creating_data(tmp_path):
    from scripts.analyze_answerability_organic_window import analyze_window

    result = analyze_window(protocol_path=PROTOCOL, log_path=tmp_path / "absent.jsonl")
    assert result["conclusion"] == "COLLECTING"
    assert result["eligible_rows_used"] == 0
    assert result["remaining"] == 200
    assert result["queue_drop_rows"] == 0


def test_complete_window_executes_frozen_tie_break_and_publishes_h1(tmp_path):
    from scripts.analyze_answerability_organic_window import (
        analyze_window,
        publish_result,
    )

    rows = []
    rows.extend(_row(index, gate=True, label="block_unsupported") for index in range(1, 21))
    rows.extend(_row(index, gate=True, label="no_change") for index in range(21, 41))
    rows.extend(_row(index, gate=False, selected=False) for index in range(41, 201))
    log_path = tmp_path / "shadow_answerability.jsonl"
    _write_rows(log_path, rows)

    result = analyze_window(protocol_path=PROTOCOL, log_path=log_path)
    assert result["conclusion"] == "SELECT_RULE"
    assert result["selected_rule"] == "H1"
    assert result["accepted_intervention_rows"] == 20
    assert result["accepted_no_change_rows"] == 20
    assert result["rules"]["H1"]["accepted_intervention_recall"] == 1.0
    assert result["rules"]["H1"]["accepted_no_change_trigger_rate"] == 0.0
    assert result["rules"]["H1"]["conditional_reference_call_rate"] == 0.1

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = tmp_path / "stage2"
    published = publish_result(result=result, protocol=protocol, output_dir=output)
    assert published["selected_rule"] == "H1"
    frozen = json.loads((output / "FROZEN_RULE.json").read_text(encoding="utf-8"))
    assert frozen["rule_id"] == "H1"
    assert frozen["authorization"] == "collect_blind_window_b_only"
    assert set(path.name for path in output.iterdir()) == {
        "WINDOW_A_RESULT.json", "WINDOW_A_RESULT.md", "FROZEN_RULE.json",
        "manifest.json",
    }


def test_underpowered_window_finalizes_without_inventing_a_rule(tmp_path):
    from scripts.analyze_answerability_organic_window import (
        analyze_window,
        publish_result,
    )

    log_path = tmp_path / "shadow_answerability.jsonl"
    _write_rows(
        log_path,
        [_row(index, gate=False, selected=False) for index in range(1, 201)],
    )
    result = analyze_window(protocol_path=PROTOCOL, log_path=log_path)
    assert result["conclusion"] == "NO_GO_UNDERPOWERED"
    assert result["selected_rule"] is None
    output = tmp_path / "underpowered"
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    publish_result(result=result, protocol=protocol, output_dir=output)
    assert not (output / "FROZEN_RULE.json").exists()
    assert "n/a" in (output / "WINDOW_A_RESULT.md").read_text(encoding="utf-8")


def test_drop_ledger_fails_window_completeness(tmp_path):
    from scripts.analyze_answerability_organic_window import analyze_window

    log_path = tmp_path / "shadow_answerability.jsonl"
    _write_rows(log_path, [_row(1, gate=True)])
    _write_rows(log_path.with_name("shadow_answerability.drops.jsonl"), [{
        "schema_version": "shadow-answerability-v3",
        "queue_status": "dropped",
        "question_sha256": _sha("drop"),
    }])
    result = analyze_window(protocol_path=PROTOCOL, log_path=log_path)
    assert result["conclusion"] == "NO_GO_WINDOW_INCOMPLETE"
    assert result["queue_drop_rows"] == 1


def test_refuses_private_judge_reason_and_wrong_protocol_lineage(tmp_path):
    from scripts.analyze_answerability_organic_window import (
        WindowAuditError,
        analyze_window,
    )

    row = _row(1, gate=True)
    row["verdicts"][0]["verdict"]["reason"] = "echoed private text"
    log_path = tmp_path / "private.jsonl"
    _write_rows(log_path, [row])
    with pytest.raises(WindowAuditError, match="private/free-text keys"):
        analyze_window(protocol_path=PROTOCOL, log_path=log_path)

    row = _row(1, gate=True)
    row["protocol_sha256"] = "0" * 64
    _write_rows(log_path, [row])
    with pytest.raises(WindowAuditError, match="protocol_sha256"):
        analyze_window(protocol_path=PROTOCOL, log_path=log_path)
