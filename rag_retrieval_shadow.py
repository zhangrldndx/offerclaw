# -*- coding: utf-8 -*-
"""Bounded, privacy-safe candidate retrieval shadow execution.

The synchronous production path always serves the baseline profile.  Only an
explicit ``RAG_RETRIEVAL_PROFILE=candidate_shadow`` request enqueues a candidate
comparison.  One daemon worker and a queue of two jobs bound background work;
on local MPS this worker still shares the reranker inference lock and may consume
background model capacity, which is why shadowing is never enabled implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable


_QUEUE: Queue = Queue(maxsize=2)
_WORKER: threading.Thread | None = None
_START_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_LOG_PATH_OVERRIDE: Path | None = None
_RUNNER_OVERRIDE: Callable[["ShadowJob"], dict] | None = None
_METRICS = {
    "enqueued": 0,
    "dropped": 0,
    "completed": 0,
    "errors": 0,
}


@dataclass(frozen=True)
class ShadowJob:
    query: str
    query_sha256: str
    top_k: int
    source_types: frozenset[str]
    exclude_source_types: frozenset[str]
    metadata_filters: dict[str, frozenset[str]]
    allow_paper_route: bool
    baseline: dict[str, Any]
    runner: Callable[["ShadowJob"], dict] | None = None


def _log_path() -> Path:
    configured = os.environ.get("RAG_CANDIDATE_SHADOW_LOG", "").strip()
    if _LOG_PATH_OVERRIDE is not None:
        return _LOG_PATH_OVERRIDE
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / ".offerclaw" / "rag_candidate_shadow.jsonl"


def _summary(result: dict[str, Any] | None, *, status: str) -> dict[str, Any]:
    result = result or {}
    trace = result.get("retrieval_trace") or {}
    finals = trace.get("final_candidates") or []
    top = finals[0] if finals else {}
    latency = trace.get("latency_by_stage") or {}
    return {
        "profile": str(
            result.get("retrieval_profile")
            or (trace.get("retrieval_profile") or {}).get("name")
            or ""
        ),
        "fingerprint": str(
            result.get("index_fingerprint") or trace.get("index_fingerprint") or ""
        ),
        "top1_chunk_id": str(
            trace.get("final_top1_chunk_id") or top.get("chunk_id") or ""
        ),
        "top1_source": os.path.basename(str(top.get("source") or "")),
        "gate": bool(trace.get("gate_decision", result.get("in_kb", False))),
        "effective_evidence": bool(
            result.get("effective_hit", trace.get("effective_hit", False))
        ),
        "latency_ms": float(latency.get("total", 0.0) or 0.0),
        "status": str(trace.get("reranker_status") or status),
    }


def summarize_baseline(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a baseline result to the only fields allowed in shadow logs."""

    return _summary(result, status="ok")


def _write_event(payload: dict[str, Any]) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
    except Exception:
        # Observability must never affect either the worker or user request.
        return


def _run_production_candidate(job: ShadowJob) -> dict:
    from rag_gate import _retrieve_and_classify
    from rag_retrieval_trace import resolve_candidate_shadow_profile

    return _retrieve_and_classify(
        job.query,
        job.top_k,
        source_types=set(job.source_types),
        exclude_source_types=set(job.exclude_source_types),
        metadata_filters={key: set(values) for key, values in job.metadata_filters.items()},
        allow_paper_route=job.allow_paper_route,
        retrieval_profile=resolve_candidate_shadow_profile(),
        _shadow_internal=True,
    )


def _worker_loop() -> None:
    while True:
        try:
            job = _QUEUE.get()
        except Exception:
            continue
        try:
            runner = job.runner or _run_production_candidate
            candidate = runner(job)
            event = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query_sha256": job.query_sha256,
                "baseline": dict(job.baseline),
                "candidate": _summary(candidate, status="ok"),
                "status": "ok",
                "error": "",
            }
            with _STATE_LOCK:
                _METRICS["completed"] += 1
        except Exception as exc:
            # Exception messages may echo a query or document.  Log only the
            # exception class, never its potentially private message.
            event = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query_sha256": job.query_sha256,
                "baseline": dict(job.baseline),
                "candidate": _summary({}, status="error"),
                "status": "candidate_error",
                "error": type(exc).__name__,
            }
            with _STATE_LOCK:
                _METRICS["errors"] += 1
        finally:
            _write_event(event)
            _QUEUE.task_done()


def _ensure_worker() -> None:
    global _WORKER
    if _WORKER is not None and _WORKER.is_alive():
        return
    with _START_LOCK:
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(
                target=_worker_loop,
                name="rag-candidate-shadow",
                daemon=True,
            )
            _WORKER.start()


def enqueue_shadow_comparison(
    query: str,
    baseline_result: dict[str, Any],
    *,
    top_k: int,
    source_types: set[str] | None = None,
    exclude_source_types: set[str] | None = None,
    metadata_filters: dict[str, set[str]] | None = None,
    allow_paper_route: bool = True,
    runner: Callable[[ShadowJob], dict] | None = None,
) -> bool:
    """Non-blockingly enqueue one candidate comparison.

    ``False`` means the bounded queue was full.  No retry occurs on the user
    thread.  Raw query text is retained only in the in-memory job needed to run
    the candidate and is never serialized.
    """

    _ensure_worker()
    job = ShadowJob(
        query=query,
        query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        top_k=int(top_k),
        source_types=frozenset(source_types or set()),
        exclude_source_types=frozenset(exclude_source_types or set()),
        metadata_filters={
            str(key): frozenset(values)
            for key, values in (metadata_filters or {}).items()
        },
        allow_paper_route=bool(allow_paper_route),
        baseline=summarize_baseline(baseline_result),
        runner=runner or _RUNNER_OVERRIDE,
    )
    try:
        _QUEUE.put_nowait(job)
    except Full:
        with _STATE_LOCK:
            _METRICS["dropped"] += 1
        return False
    with _STATE_LOCK:
        _METRICS["enqueued"] += 1
    return True


def shadow_metrics() -> dict[str, int]:
    with _STATE_LOCK:
        return {**_METRICS, "pending": int(_QUEUE.unfinished_tasks)}


def flush_shadow_jobs(timeout: float = 5.0) -> bool:
    """Wait for tests/shutdown diagnostics; never used by the request path."""

    deadline = time.monotonic() + max(0.0, timeout)
    while _QUEUE.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


def reset_shadow_state(
    *,
    log_path: str | Path | None = None,
    runner: Callable[[ShadowJob], dict] | None = None,
    timeout: float = 5.0,
) -> None:
    """Reset metrics/test overrides after flushing all prior work."""

    global _LOG_PATH_OVERRIDE, _RUNNER_OVERRIDE
    if not flush_shadow_jobs(timeout):
        raise TimeoutError("candidate shadow worker did not flush")
    while True:
        try:
            _QUEUE.get_nowait()
        except Empty:
            break
        else:
            _QUEUE.task_done()
    with _STATE_LOCK:
        for key in _METRICS:
            _METRICS[key] = 0
    _LOG_PATH_OVERRIDE = Path(log_path) if log_path is not None else None
    _RUNNER_OVERRIDE = runner
