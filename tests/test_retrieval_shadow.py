# -*- coding: utf-8 -*-
"""Bounded candidate retrieval shadow execution and privacy contract."""

from __future__ import annotations

import hashlib
import json
import threading


def _result(profile: str, chunk_id: str, source: str) -> dict:
    return {
        "retrieval_profile": profile,
        "index_fingerprint": f"fp-{profile}",
        "in_kb": True,
        "effective_hit": True,
        "retrieval_trace": {
            "retrieval_profile": {"name": profile},
            "index_fingerprint": f"fp-{profile}",
            "final_top1_chunk_id": chunk_id,
            "final_candidates": [{"chunk_id": chunk_id, "source": source}],
            "gate_decision": True,
            "effective_hit": True,
            "latency_by_stage": {"total": 12.5},
        },
    }


def test_shadow_log_contains_hash_but_never_raw_query(tmp_path):
    import rag_retrieval_shadow as shadow

    log_path = tmp_path / "shadow.jsonl"
    shadow.reset_shadow_state(
        log_path=log_path,
        runner=lambda _job: _result(
            "candidate", "candidate-chunk", "/private/path/candidate.md",
        ),
    )
    query = "我的私密检索问题 unique-secret-9271"
    assert shadow.enqueue_shadow_comparison(
        query, _result("baseline", "baseline-chunk", "baseline.md"), top_k=5,
    )
    assert shadow.flush_shadow_jobs(2)

    raw = log_path.read_text(encoding="utf-8")
    assert query not in raw
    assert "unique-secret-9271" not in raw
    event = json.loads(raw.splitlines()[0])
    assert event["query_sha256"] == hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert set(event) == {
        "timestamp", "query_sha256", "baseline", "candidate", "status", "error",
    }
    assert event["baseline"]["top1_chunk_id"] == "baseline-chunk"
    assert event["candidate"]["profile"] == "candidate"
    assert event["candidate"]["top1_source"] == "candidate.md"
    assert "/private/path" not in raw
    shadow.reset_shadow_state()


def test_shadow_queue_is_bounded_and_drops_without_blocking(tmp_path):
    import rag_retrieval_shadow as shadow

    started = threading.Event()
    release = threading.Event()

    def blocking_runner(_job):
        started.set()
        release.wait(2)
        return _result("candidate", "candidate-chunk", "candidate.md")

    shadow.reset_shadow_state(log_path=tmp_path / "shadow.jsonl", runner=blocking_runner)
    baseline = _result("baseline", "baseline-chunk", "baseline.md")
    assert shadow.enqueue_shadow_comparison("q1", baseline, top_k=5)
    assert started.wait(1)
    # One job is active and exactly two more fit the bounded waiting queue.
    assert shadow.enqueue_shadow_comparison("q2", baseline, top_k=5)
    assert shadow.enqueue_shadow_comparison("q3", baseline, top_k=5)
    assert shadow.enqueue_shadow_comparison("q4", baseline, top_k=5) is False
    assert shadow.shadow_metrics()["dropped"] == 1

    release.set()
    assert shadow.flush_shadow_jobs(2)
    metrics = shadow.shadow_metrics()
    assert metrics["enqueued"] == 3
    assert metrics["completed"] == 3
    shadow.reset_shadow_state()


def test_candidate_exception_isolated_and_error_message_is_redacted(tmp_path):
    import rag_retrieval_shadow as shadow

    query = "private-error-query-441"

    def failing_runner(_job):
        raise ValueError("private-error-query-441 plus document contents")

    log_path = tmp_path / "shadow.jsonl"
    shadow.reset_shadow_state(log_path=log_path, runner=failing_runner)
    accepted = shadow.enqueue_shadow_comparison(
        query, _result("baseline", "baseline-chunk", "baseline.md"), top_k=5,
    )
    assert accepted is True
    assert shadow.flush_shadow_jobs(2)
    assert shadow.shadow_metrics()["errors"] == 1

    raw = log_path.read_text(encoding="utf-8")
    assert query not in raw
    assert "document contents" not in raw
    event = json.loads(raw.splitlines()[0])
    assert event["status"] == "candidate_error"
    assert event["error"] == "ValueError"
    shadow.reset_shadow_state()
