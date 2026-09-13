# -*- coding: utf-8 -*-
"""Organic Window A/B contracts for the record-only answerability shadow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue

import pytest

import rag_shadow_answerability as shadow


_SHADOW_ENV = {
    "RAG_SHADOW_ANSWERABILITY",
    "RAG_SHADOW_CONTROL_RATE",
    "RAG_SHADOW_FROZEN_RULE_ID",
    "RAG_SHADOW_FROZEN_RULE_SHA256",
    "RAG_SHADOW_GATE_REJECT_CONTROL_RATE",
    "RAG_SHADOW_LOG_PATH",
    "RAG_SHADOW_PROTOCOL_ID",
    "RAG_SHADOW_PROTOCOL_SHA256",
    "RAG_SHADOW_SAMPLING_MODE",
    "RAG_SHADOW_TRIGGER_HIGH_SCORE",
    "RAG_SHADOW_TRIGGER_MARGIN",
    "RAG_SHADOW_WINDOW_ID",
    "RAG_SHADOW_WINDOW_PHASE",
}


class _Candidate:
    def __init__(self, chunk_id: str, score: float, document: str):
        self.chunk_id = chunk_id
        self.rerank_score = score
        self.document = document


class _Trace:
    gate_features = {
        "best_dense_distance": 0.71,
        "strong_threshold": 0.68,
        "rerank_gate_min": 0.85,
        "rerank_top": 0.96,
        "rerank_margin": 0.04,
    }
    gate_decision = True
    retrieval_profile = type("Profile", (), {"name": "baseline"})()
    index_fingerprint = "fp-organic-test"
    reranker_requested = "reranker-requested"
    reranker_actual = "reranker-actual"
    reranker_status = "ok"
    final_candidates = [
        _Candidate("chunk-a", 0.96, "private evidence A"),
        _Candidate("chunk-b", 0.92, "private evidence B"),
    ]
    reranked_candidates = final_candidates
    dense_candidates = [_Candidate("chunk-a", 0.0, "")]
    bm25_candidates = [_Candidate("chunk-b", 0.0, "")]


class _RejectedTrace(_Trace):
    gate_decision = False


@pytest.fixture(autouse=True)
def _clean_shadow_env(monkeypatch):
    for name in _SHADOW_ENV:
        monkeypatch.delenv(name, raising=False)


def _organic_env(monkeypatch, *, phase: str = "calibration_a") -> None:
    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "accepted_control")
    monkeypatch.setenv("RAG_SHADOW_WINDOW_ID", f"organic_{phase}_test")
    monkeypatch.setenv("RAG_SHADOW_WINDOW_PHASE", phase)
    monkeypatch.setenv("RAG_SHADOW_PROTOCOL_ID", "answerability-organic-v4")
    monkeypatch.setenv("RAG_SHADOW_PROTOCOL_SHA256", "a" * 64)
    monkeypatch.setenv("RAG_SHADOW_GATE_REJECT_CONTROL_RATE", "0.25")
    if phase == "blind_b":
        monkeypatch.setenv("RAG_SHADOW_FROZEN_RULE_ID", "rule-h1")
        monkeypatch.setenv("RAG_SHADOW_FROZEN_RULE_SHA256", "b" * 64)


def _capture_job(monkeypatch, question: str, trace, *, sample: float,
                 retrieval_question: str | None = None):
    jobs = queue.Queue(maxsize=8)
    monkeypatch.setattr(shadow, "_QUEUE", jobs)
    monkeypatch.setattr(shadow, "_ensure_worker", lambda: None)
    assert shadow.observe(
        question,
        trace,
        sampler=lambda: sample,
        retrieval_question=retrieval_question,
    ) is None
    return jobs.get_nowait()


def test_window_a_selects_every_gate_accept_and_records_lineage(monkeypatch):
    _organic_env(monkeypatch)
    job = _capture_job(
        monkeypatch,
        "original private question",
        _Trace(),
        sample=0.999,
        retrieval_question="reference route subquery",
    )

    assert job["schema_version"] == "shadow-answerability-v3"
    assert job["judge_selected"] is True
    assert job["sampling_probability"] == 1.0
    assert job["sampling_weight"] == 1.0
    assert job["sampling_reason"] == "all_gate_accepts"
    assert job["window_phase"] == "calibration_a"
    assert job["protocol_sha256"] == "a" * 64
    assert job["judge_question_role"] == "original_user_question"
    assert job["candidate_count"] == 2 and job["judge_depth"] == 2
    assert job["queue_status"] == "enqueued"
    assert job["_question"] == "original private question"
    assert job["_retrieval_question"] == "reference route subquery"
    assert job["original_question_sha256"] == hashlib.sha256(
        b"original private question"
    ).hexdigest()
    assert job["retrieval_question_sha256"] == hashlib.sha256(
        b"reference route subquery"
    ).hexdigest()

    public = {key: value for key, value in job.items() if not key.startswith("_")}
    encoded = json.dumps(public, ensure_ascii=False)
    assert "original private question" not in encoded
    assert "reference route subquery" not in encoded
    assert "private evidence" not in encoded
    assert "chunk-a" not in encoded


@pytest.mark.parametrize(
    "draw,selected,weight",
    [(0.10, True, 4.0), (0.90, False, None)],
)
def test_gate_reject_uses_declared_probability_and_weight(
    monkeypatch, draw, selected, weight,
):
    _organic_env(monkeypatch)
    job = _capture_job(monkeypatch, "private", _RejectedTrace(), sample=draw)

    assert job["judge_selected"] is selected
    assert job["sampling_probability"] == 0.25
    assert job["sampling_weight"] == weight
    assert job["sampling_reason"] == "random_gate_reject_control"
    assert job["control_sample"] is selected
    assert bool(job["_question"]) is selected
    assert bool(job["_candidates"]) is selected


def test_organic_config_is_fail_closed_and_blind_requires_frozen_rule(
    monkeypatch,
):
    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "accepted_control")
    job = _capture_job(monkeypatch, "must not reach judge", _Trace(), sample=0.0)

    assert job["sampling_config_valid"] is False
    assert job["judge_selected"] is False
    assert job["sampling_probability"] == 0.0
    assert job["sampling_reason"] == "config_invalid"
    assert job["_question"] == "" and job["_retrieval_question"] == ""
    assert job["_candidates"] == []

    _organic_env(monkeypatch, phase="blind_b")
    monkeypatch.delenv("RAG_SHADOW_FROZEN_RULE_ID")
    monkeypatch.delenv("RAG_SHADOW_FROZEN_RULE_SHA256")
    config = shadow.sampling_config()
    assert config["sampling_config_valid"] is False
    assert set(config["sampling_config_errors"]) >= {
        "invalid_frozen_rule_id", "invalid_frozen_rule_sha256",
    }


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "not-a-number"])
def test_organic_control_probability_is_strict_not_silently_clamped(
    monkeypatch, value,
):
    _organic_env(monkeypatch)
    monkeypatch.setenv("RAG_SHADOW_GATE_REJECT_CONTROL_RATE", value)
    config = shadow.sampling_config()
    assert config["sampling_config_valid"] is False
    assert "invalid_control_rate" in config["sampling_config_errors"]


def test_nonorganic_rows_cannot_enter_an_organic_window(monkeypatch):
    _organic_env(monkeypatch)
    config = shadow.sampling_config()
    decision = shadow.sampling_decision(
        {"gate_decision": True},
        config,
        triggered=False,
        sampler=lambda: 0.0,
        traffic_origin="agent_generated",
    )
    assert decision == {
        "judge_selected": False,
        "sampling_probability": 0.0,
        "sampling_weight": None,
        "sampling_reason": "ineligible_traffic_origin",
    }
    work_queue = queue.Queue(maxsize=8)
    monkeypatch.setattr(shadow, "_QUEUE", work_queue)
    monkeypatch.setattr(shadow, "_ensure_worker", lambda: None)
    before = shadow.stats()["ineligible_origin"]
    assert shadow.observe(
        "automated private query", _Trace(), traffic_origin="agent_generated",
    ) is None
    assert work_queue.empty()
    assert shadow.stats()["ineligible_origin"] == before + 1


def test_default_mode_keeps_legacy_trigger_sampling(monkeypatch):
    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_MARGIN", "0.05")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_HIGH_SCORE", "0.90")
    job = _capture_job(monkeypatch, "legacy", _Trace(), sample=0.999)

    assert job["schema_version"] == "shadow-answerability-v2"
    assert job["sampling_mode"] == "legacy_trigger_control"
    assert job["triggered"] is True and job["judge_selected"] is True
    assert job["sampling_probability"] == 1.0
    assert job["sampling_reason"] == "trigger_rule"


def test_each_job_binds_its_window_log_path(monkeypatch):
    _organic_env(monkeypatch)
    a = shadow._resolved_log_path(shadow.sampling_config())
    monkeypatch.setenv("RAG_SHADOW_WINDOW_ID", "organic_blind_b_test")
    monkeypatch.setenv("RAG_SHADOW_WINDOW_PHASE", "blind_b")
    monkeypatch.setenv("RAG_SHADOW_FROZEN_RULE_ID", "rule-h1")
    monkeypatch.setenv("RAG_SHADOW_FROZEN_RULE_SHA256", "b" * 64)
    b = shadow._resolved_log_path(shadow.sampling_config())

    assert a != b
    assert a.name == b.name == "shadow_answerability.jsonl"
    assert "organic_calibration_a_test" in str(a)
    assert "organic_blind_b_test" in str(b)


def test_worker_preserves_row_when_one_judge_call_raises(monkeypatch, tmp_path):
    import rag_answerability

    _organic_env(monkeypatch)
    log_path = tmp_path / "window-a.jsonl"
    monkeypatch.setenv("RAG_SHADOW_LOG_PATH", str(log_path))
    calls = []

    def fake_grade(original, chunk, *, retrieval_question=None):
        calls.append((original, retrieval_question, chunk))
        if len(calls) == 2:
            raise RuntimeError("private provider failure")
        return {
            "grade": 3,
            "question_form": "polar",
            "direct_answer": "proposition_true",
            "premise_status": "supported",
            "relation": "entails",
            "reason": "ok",
        }

    monkeypatch.setattr(rag_answerability, "grade", fake_grade)
    monkeypatch.setattr(rag_answerability, "flush_cache", lambda: None)
    work_queue = queue.Queue(maxsize=8)
    monkeypatch.setattr(shadow, "_QUEUE", work_queue)
    monkeypatch.setattr(shadow, "_WORKER", None)

    assert shadow.observe(
        "original secret",
        _Trace(),
        retrieval_question="route secret",
        sampler=lambda: 0.9,
    ) is None
    shadow.drain(2.0)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["judge_question_role"] == "original_user_question"
    assert row["judge_selected"] is True
    assert row["label_available"] is False
    assert row["top2_complete"] is False
    assert len(row["verdicts"]) == 2
    assert row["verdicts"][0]["available"] is True
    assert row["verdicts"][1]["available"] is False
    assert row["verdicts"][1]["failure_type"] == "exception"
    assert calls[0][:2] == ("original secret", "route secret")
    encoded = json.dumps(row, ensure_ascii=False)
    assert "reason" not in row["verdicts"][0]["verdict"]
    assert "original secret" not in encoded
    assert "route secret" not in encoded
    assert "private provider failure" not in encoded

    # Stop this test's daemon before another test replaces the global queue.
    worker = shadow._WORKER
    work_queue.put(None)
    shadow.drain(2.0)
    worker.join(timeout=2.0)


def test_selected_queue_drop_is_counted_and_persisted_without_private_text(
    monkeypatch, tmp_path,
):
    _organic_env(monkeypatch)
    log_path = tmp_path / "window-a.jsonl"
    monkeypatch.setenv("RAG_SHADOW_LOG_PATH", str(log_path))
    full = queue.Queue(maxsize=1)
    full.put_nowait({"occupied": True})
    monkeypatch.setattr(shadow, "_QUEUE", full)
    monkeypatch.setattr(shadow, "_ensure_worker", lambda: None)
    before = shadow.stats()

    assert shadow.observe("private", _Trace(), sampler=lambda: 0.9) is None
    after = shadow.stats()
    assert after["dropped"] == before["dropped"] + 1
    assert after["dropped_selected"] == before["dropped_selected"] + 1
    drop_path = tmp_path / "window-a.drops.jsonl"
    rows = [json.loads(line) for line in drop_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["queue_status"] == "dropped"
    assert rows[0]["judge_selected"] is True
    encoded = json.dumps(rows[0], ensure_ascii=False)
    assert "private" not in encoded
    assert all(not key.startswith("_") for key in rows[0])
