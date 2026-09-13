# -*- coding: utf-8 -*-
"""Record-only shadow for the answerability judge.  Never changes an answer.

Purpose: collect the evidence needed to decide whether the judge belongs on the
hot path, and to calibrate a trigger on traffic that has not already been used
to find the problem.  Until that calibration exists this observes; it does not
act.

Three properties the design turns on:

**Nothing it does can change the response.**  ``observe`` returns ``None``,
swallows every exception, and only ever appends to a bounded queue drained by a
background worker.  The current hook runs after the retrieval decision but can
overlap answer generation; queue/provider contention is therefore measured,
not hand-waved away as "zero latency".

**It does not sample only what the trigger likes.**  The backwards-compatible
mode judges triggered rows plus random untriggered controls.  Bounded organic
calibration/release windows use a stronger ``accepted_control`` design: every
request accepted by the existing gate is judged, while rejected requests are
sampled at a declared probability.  The selection probability and inverse
probability weight are recorded on every row.

**The trigger thresholds are not calibrated yet, and say so.**  They are read
from the environment with no defaults that pretend to be tuned; a run whose
config is absent is recorded as ``uncalibrated`` rather than quietly using a
number someone guessed.  Calibrating them on Dev80/V1 is forbidden -- those
sets were used to find the problem and would confirm whatever they were fitted
to.
"""

from __future__ import annotations

import json
import hashlib
import inspect
import os
from pathlib import Path
import queue
import random
import re
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "shadow_answerability.jsonl"
LEGACY_SCHEMA = "shadow-answerability-v2"
ORGANIC_SCHEMA = "shadow-answerability-v3"
# Existing consumers import SCHEMA and expect the legacy record contract.  The
# organic window mode selects ORGANIC_SCHEMA per row without changing that
# backwards-compatible default.
SCHEMA = LEGACY_SCHEMA
QUEUE_MAXSIZE = 64
JUDGE_DEPTH = 2
LEGACY_SAMPLING_MODE = "legacy_trigger_control"
ORGANIC_SAMPLING_MODE = "accepted_control"
JUDGE_QUESTION_ROLE = "original_user_question"
_VALID_WINDOW_PHASES = {"calibration_a", "blind_b"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")

_QUEUE: "queue.Queue[dict[str, Any]] | None" = None
_WORKER: threading.Thread | None = None
_LOCK = threading.Lock()
_STATS = {
    "observed": 0,
    "selected": 0,
    "ineligible_origin": 0,
    "config_invalid": 0,
    "enqueued": 0,
    "dropped": 0,
    "dropped_selected": 0,
    "dropped_unselected": 0,
    "drop_record_failed": 0,
    "judged": 0,
    "judge_failed": 0,
    "judge_calls": 0,
}


def enabled() -> bool:
    return (os.environ.get("RAG_SHADOW_ANSWERABILITY", "") or "").strip() in {
        "1", "true", "yes", "on"}


def _float_env(name: str) -> float | None:
    raw = (os.environ.get(name, "") or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def trigger_config() -> dict[str, Any]:
    """Thresholds for "the reranker cannot separate these", or ``uncalibrated``.

    Deliberately without tuned defaults.  Every candidate number available today
    would have been read off Dev80 or V1, and those two sets were used to *find*
    this failure -- a threshold fitted to them cannot then be evidence about it.
    """
    margin = _float_env("RAG_SHADOW_TRIGGER_MARGIN")
    high = _float_env("RAG_SHADOW_TRIGGER_HIGH_SCORE")
    control = _float_env("RAG_SHADOW_CONTROL_RATE")
    if control is None:
        control = 0.05
    control = min(1.0, max(0.0, control))
    return {
        "margin_below": margin,
        "high_score_at_least": high,
        "calibrated": margin is not None and high is not None,
        "control_rate": control,
    }


def _clean_env(name: str) -> str:
    return (os.environ.get(name, "") or "").strip()


def _organic_control_rate() -> tuple[float, str | None]:
    """Resolve the rejected-gate sampling rate without hiding bad config."""
    raw = _clean_env("RAG_SHADOW_GATE_REJECT_CONTROL_RATE")
    if not raw:
        # Reusing the old name makes a staged rollout less surprising, while
        # the organic mode still validates it strictly instead of clamping.
        raw = _clean_env("RAG_SHADOW_CONTROL_RATE")
    if not raw:
        return 0.0, "missing_control_rate"
    try:
        rate = float(raw)
    except ValueError:
        return 0.0, "invalid_control_rate"
    if not 0.0 < rate <= 1.0:
        return 0.0, "invalid_control_rate"
    return rate, None


def sampling_config() -> dict[str, Any]:
    """Return sampling and lineage config for the current process.

    Missing ``RAG_SHADOW_SAMPLING_MODE`` intentionally retains the v2 trigger
    behaviour.  ``accepted_control`` is fail-closed: it will record features
    but send no private text to the worker unless the window and frozen
    protocol lineage are complete.
    """
    raw_mode = _clean_env("RAG_SHADOW_SAMPLING_MODE")
    mode = raw_mode.lower() if raw_mode else LEGACY_SAMPLING_MODE
    if mode == "legacy":
        mode = LEGACY_SAMPLING_MODE

    window_id = _clean_env("RAG_SHADOW_WINDOW_ID")
    window_phase = _clean_env("RAG_SHADOW_WINDOW_PHASE").lower()
    protocol_id = _clean_env("RAG_SHADOW_PROTOCOL_ID")
    protocol_sha256 = _clean_env("RAG_SHADOW_PROTOCOL_SHA256").lower()
    frozen_rule_id = _clean_env("RAG_SHADOW_FROZEN_RULE_ID")
    frozen_rule_sha256 = _clean_env("RAG_SHADOW_FROZEN_RULE_SHA256").lower()

    errors: list[str] = []
    if mode not in {LEGACY_SAMPLING_MODE, ORGANIC_SAMPLING_MODE}:
        errors.append("invalid_sampling_mode")

    if mode == ORGANIC_SAMPLING_MODE:
        control_rate, control_error = _organic_control_rate()
        if control_error:
            errors.append(control_error)
        if not _SAFE_ID.fullmatch(window_id):
            errors.append("invalid_window_id")
        if window_phase not in _VALID_WINDOW_PHASES:
            errors.append("invalid_window_phase")
        if not _SAFE_ID.fullmatch(protocol_id):
            errors.append("invalid_protocol_id")
        if not _SHA256.fullmatch(protocol_sha256):
            errors.append("invalid_protocol_sha256")
        if window_phase == "blind_b":
            if not _SAFE_ID.fullmatch(frozen_rule_id):
                errors.append("invalid_frozen_rule_id")
            if not _SHA256.fullmatch(frozen_rule_sha256):
                errors.append("invalid_frozen_rule_sha256")
    else:
        control_rate = trigger_config()["control_rate"]

    return {
        "sampling_mode": mode,
        "sampling_config_valid": not errors,
        "sampling_config_errors": errors,
        "gate_reject_control_rate": control_rate,
        "window_id": window_id,
        "window_phase": window_phase,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "frozen_rule_id": frozen_rule_id or None,
        "frozen_rule_sha256": frozen_rule_sha256 or None,
        "judge_question_role": JUDGE_QUESTION_ROLE,
    }


def sampling_decision(
    features: dict[str, Any],
    config: dict[str, Any],
    *,
    triggered: bool,
    sampler=random.random,
    traffic_origin: str = "organic",
) -> dict[str, Any]:
    """Choose judge inclusion without consulting a judge result."""
    mode = config.get("sampling_mode", LEGACY_SAMPLING_MODE)
    if not config.get("sampling_config_valid", False):
        return {
            "judge_selected": False,
            "sampling_probability": 0.0,
            "sampling_weight": None,
            "sampling_reason": "config_invalid",
        }

    if mode == ORGANIC_SAMPLING_MODE:
        if traffic_origin != "organic":
            return {
                "judge_selected": False,
                "sampling_probability": 0.0,
                "sampling_weight": None,
                "sampling_reason": "ineligible_traffic_origin",
            }
        if bool(features.get("gate_decision", False)):
            return {
                "judge_selected": True,
                "sampling_probability": 1.0,
                "sampling_weight": 1.0,
                "sampling_reason": "all_gate_accepts",
            }
        probability = float(config["gate_reject_control_rate"])
        selected = sampler() < probability
        return {
            "judge_selected": selected,
            "sampling_probability": probability,
            "sampling_weight": (1.0 / probability) if selected else None,
            "sampling_reason": "random_gate_reject_control",
        }

    # v2 compatibility: a frozen/experimental trigger is always included and
    # a random fraction of rows it did not flag supplies the control sample.
    if triggered:
        return {
            "judge_selected": True,
            "sampling_probability": 1.0,
            "sampling_weight": 1.0,
            "sampling_reason": "trigger_rule",
        }
    probability = float(config["gate_reject_control_rate"])
    selected = sampler() < probability
    return {
        "judge_selected": selected,
        "sampling_probability": probability,
        "sampling_weight": (
            (1.0 / probability) if selected and probability else None
        ),
        "sampling_reason": "random_untriggered_control",
    }


def _resolved_log_path(config: dict[str, Any]) -> Path:
    """Bind a destination to each job so rotating windows cannot cross-write."""
    override = _clean_env("RAG_SHADOW_LOG_PATH")
    if override:
        return Path(override).expanduser()
    if config.get("sampling_mode") == ORGANIC_SAMPLING_MODE:
        window_id = config.get("window_id") or "config_invalid"
        if not _SAFE_ID.fullmatch(str(window_id)):
            window_id = "config_invalid"
        return (
            ROOT / "logs" / "rag_eval" / "answerability_shadow"
            / str(window_id) / "shadow_answerability.jsonl"
        )
    return LOG_PATH


def _drop_log_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}.drops.jsonl")


def _record_queue_drop(entry: dict[str, Any]) -> None:
    """Persist a privacy-safe loss marker when the judge queue is full.

    A counter kept only in process memory is not enough to certify a frozen
    window after a restart.  This slow path runs only after the bounded queue
    has already rejected a job; it never writes the original question, chunk
    text, model prose, or any other underscore-prefixed private field.
    """
    try:
        log_path = Path(entry.get("_log_path") or LOG_PATH)
        drop_path = _drop_log_path(log_path)
        record = {
            key: value
            for key, value in entry.items()
            if not key.startswith("_")
        }
        record.update({
            "queue_status": "dropped",
            "dropped_at_ms": round(time.time() * 1000),
        })
        drop_path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, drop_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        _STATS["drop_record_failed"] += 1


def trigger_features(trace: Any) -> dict[str, Any]:
    """Judge-independent signals only -- these must be computable without an LLM.

    A trigger that needed the judge to decide whether to call the judge would
    cost exactly what it was meant to save.
    """
    features = getattr(trace, "gate_features", None) or {}
    # Post-rerank fusion/guards may change the winner.  The trigger must describe
    # what the answer path actually sees, not the earlier raw reranker order.
    pool = list(getattr(trace, "final_candidates", None) or
                getattr(trace, "reranked_candidates", None) or [])
    scores = [item.rerank_score for item in pool
              if getattr(item, "rerank_score", None) is not None]
    dense_ids = [item.chunk_id for item in
                 (getattr(trace, "dense_candidates", None) or [])][:5]
    bm25_ids = [item.chunk_id for item in
                (getattr(trace, "bm25_candidates", None) or [])][:5]
    raw_top1, raw_top2 = (scores + [None, None])[:2]
    gate_top = features.get("rerank_top")
    gate_margin = features.get("rerank_margin")
    if gate_top is None:
        gate_top = raw_top1
    if gate_margin is None and raw_top1 is not None and raw_top2 is not None:
        gate_margin = raw_top1 - raw_top2
    gate_min = features.get("rerank_gate_min")
    strong = features.get("strong_threshold")
    best_dense = features.get("best_dense_distance")
    channels_available = bool(dense_ids and bm25_ids)
    return {
        "rerank_top": gate_top,
        "rerank_margin": gate_margin,
        "raw_final_top1": raw_top1,
        "raw_final_margin": (
            raw_top1 - raw_top2
            if raw_top1 is not None and raw_top2 is not None else None
        ),
        # Preserve the score vector so calibration can derive saturation counts
        # without baking an uncalibrated 0.9 cutoff into the only raw dataset.
        "rerank_scores_top5": scores[:5],
        # several candidates simultaneously above a high score = saturation
        "high_scorers": sum(1 for score in scores if score is not None
                            and score >= 0.9),
        # the two channels disagreeing about the winner is a cheap hardness cue
        "channels_available": channels_available,
        "channel_top1_agree": (
            dense_ids[0] == bm25_ids[0] if channels_available else None
        ),
        "channel_top5_overlap": len(set(dense_ids) & set(bm25_ids)),
        "best_dense_distance": best_dense,
        "best_dense_minus_strong": (
            best_dense - strong
            if best_dense is not None and strong is not None else None
        ),
        "gate_score_minus_threshold": (
            gate_top - gate_min
            if gate_top is not None and gate_min is not None else None
        ),
        "gate_anchor_rank": features.get("gate_anchor_rank"),
        "gate_decision": bool(getattr(trace, "gate_decision", False)),
        "pool_size": len(pool),
    }


def should_trigger(features: dict[str, Any], config: dict[str, Any]) -> bool:
    if not config.get("calibrated"):
        return False
    margin = features.get("rerank_margin")
    top = features.get("rerank_top")
    if margin is None or top is None:
        return False
    return (margin < config["margin_below"]
            and top >= config["high_score_at_least"])


def _call_grade_with_question_identity(
    grade_callable,
    original_question: str,
    retrieval_question: str,
    chunk: str,
):
    """Pass route context when the installed judge contract supports it."""
    try:
        parameters = inspect.signature(grade_callable).parameters.values()
        supports_context = any(
            parameter.name == "retrieval_question"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_context = False
    if supports_context:
        return grade_callable(
            original_question,
            chunk,
            retrieval_question=retrieval_question,
        )
    return grade_callable(original_question, chunk)


def _worker_loop() -> None:
    from rag_answerability import SCHEMA as judge_schema
    from rag_answerability import action, flush_cache, grade

    # Bind this worker to the queue it was created for.  Tests and process
    # reconfiguration may replace the module-level queue; looking it up again
    # at ``task_done`` time can acknowledge a different queue.
    work_queue = _QUEUE
    if work_queue is None:
        return
    while True:
        job = work_queue.get()
        if job is None:
            work_queue.task_done()
            return
        try:
            verdicts = []
            candidates = job.get("_candidates", [])
            judge_started_ms = (
                round(time.time() * 1000) if candidates else None
            )
            for rank, chunk_id, rerank_score, text in candidates:
                call_started = time.perf_counter()
                _STATS["judge_calls"] += 1
                failure_type = None
                try:
                    verdict = _call_grade_with_question_identity(
                        grade,
                        job["_question"],
                        job.get("_retrieval_question") or job["_question"],
                        text,
                    )
                    if verdict is None:
                        failure_type = "unavailable"
                except Exception:
                    # Preserve the selected row and its inclusion probability.
                    # Losing it would silently bias a calibration window.
                    verdict = None
                    failure_type = "exception"
                available = verdict is not None
                if not available:
                    _STATS["judge_failed"] += 1
                # Keep only the closed v4 contract in the window log.  The
                # judge's free-form ``reason`` is not a predictor or label and
                # may echo private question/evidence text.
                public_verdict = (
                    {
                        field: verdict.get(field)
                        for field in (
                            "grade",
                            "question_form",
                            "direct_answer",
                            "premise_status",
                            "relation",
                        )
                    }
                    if available else None
                )
                verdicts.append({
                    "rank": rank,
                    "chunk_id_sha256": hashlib.sha256(
                        chunk_id.encode("utf-8")
                    ).hexdigest(),
                    "rerank_score": rerank_score,
                    "verdict": public_verdict,
                    "action": action(verdict),
                    "available": available,
                    "failure_type": failure_type,
                    "latency_ms": round(
                        (time.perf_counter() - call_started) * 1000, 3
                    ),
                })
            if verdicts:
                try:
                    flush_cache()
                except Exception:
                    # Cache durability is useful but cannot erase the window row.
                    pass
            judged_at_ms = round(time.time() * 1000)
            judge_selected = bool(job.get("judge_selected", False))
            candidate_count = int(job.get("candidate_count", len(candidates)))
            label_available = bool(
                judge_selected
                and candidate_count > 0
                and len(verdicts) == candidate_count
                and all(row["available"] for row in verdicts)
            )
            record = {
                **{k: v for k, v in job.items() if not k.startswith("_")},
                "judge_schema": judge_schema,
                "verdicts": verdicts,
                "label_available": label_available,
                "top2_complete": bool(label_available and candidate_count >= 2),
                "judge_started_ms": judge_started_ms,
                "judged_at_ms": judged_at_ms if verdicts else None,
                "queue_wait_ms": max(
                    0,
                    (judge_started_ms or judged_at_ms)
                    - int(job["observed_at_ms"]),
                ),
                "judge_latency_ms": (
                    max(0, judged_at_ms - judge_started_ms)
                    if judge_started_ms is not None else None
                ),
            }
            log_path = Path(job.get("_log_path") or LOG_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if verdicts:
                _STATS["judged"] += 1
        except Exception:
            # A shadow that can raise is not a shadow.  The request is already
            # answered; there is nothing here worth surfacing to a user.
            _STATS["judge_failed"] += 1
        finally:
            work_queue.task_done()


def _ensure_worker() -> None:
    global _QUEUE, _WORKER
    if _QUEUE is None:
        _QUEUE = queue.Queue(maxsize=QUEUE_MAXSIZE)
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(target=_worker_loop, daemon=True,
                                   name="shadow-answerability")
        _WORKER.start()


def observe(question: str, trace: Any, *, route: str = "reference_kb",
            traffic_origin: str = "organic", sampler=random.random,
            retrieval_question: str | None = None) -> None:
    """Record trigger features and maybe queue a judge call.  Always ``None``.

    Callers must not branch on anything this returns, and there is nothing to
    branch on by construction.
    """
    if not enabled():
        return None
    if route != "reference_kb":
        return None
    try:
        _STATS["observed"] += 1
        trigger = trigger_config()
        sample_config = sampling_config()
        if (
            sample_config["sampling_mode"] == ORGANIC_SAMPLING_MODE
            and traffic_origin != "organic"
        ):
            # Automated/replay/test requests must not even share the organic
            # window file.  Recording them as unselected rows there would
            # corrupt the frozen first-N request denominator.
            _STATS["ineligible_origin"] += 1
            return None
        features = trigger_features(trace)
        triggered = should_trigger(features, trigger)
        decision = sampling_decision(
            features,
            sample_config,
            triggered=triggered,
            sampler=sampler,
            traffic_origin=traffic_origin,
        )
        selected = bool(decision["judge_selected"])
        if selected:
            _STATS["selected"] += 1
        if not sample_config["sampling_config_valid"]:
            _STATS["config_invalid"] += 1
        # Compatibility field: in organic windows "control" means a sampled
        # gate reject, not merely an untriggered row.
        if sample_config["sampling_mode"] == ORGANIC_SAMPLING_MODE:
            control = selected and not bool(features["gate_decision"])
        else:
            control = selected and not triggered
        observed_at_ms = round(time.time() * 1000)
        route_question = retrieval_question or question
        entry = {
            "schema_version": (
                ORGANIC_SCHEMA
                if sample_config["sampling_mode"] == ORGANIC_SAMPLING_MODE
                else LEGACY_SCHEMA
            ),
            "question_sha256": hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            "original_question_sha256": hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            "retrieval_question_sha256": hashlib.sha256(
                route_question.encode("utf-8")
            ).hexdigest(),
            "features": features,
            "gate_decision": bool(features["gate_decision"]),
            "triggered": triggered,
            "rule_prediction": (
                None
                if sample_config["sampling_mode"] == ORGANIC_SAMPLING_MODE
                else triggered
            ),
            "control_sample": control,
            "trigger_calibrated": trigger["calibrated"],
            **decision,
            "sampling_mode": sample_config["sampling_mode"],
            "sampling_config_valid": sample_config["sampling_config_valid"],
            "sampling_config_errors": sample_config["sampling_config_errors"],
            "window_id": sample_config["window_id"] or None,
            "window_phase": sample_config["window_phase"] or None,
            "protocol_id": sample_config["protocol_id"] or None,
            "protocol_sha256": sample_config["protocol_sha256"] or None,
            "frozen_rule_id": sample_config["frozen_rule_id"],
            "frozen_rule_sha256": sample_config["frozen_rule_sha256"],
            "judge_question_role": sample_config["judge_question_role"],
            "observed_at_ms": observed_at_ms,
            "retrieval_profile": str(
                getattr(getattr(trace, "retrieval_profile", None), "name", "")
            ),
            "index_fingerprint": str(
                getattr(trace, "index_fingerprint", "") or ""
            ),
            "reranker_requested": str(
                getattr(trace, "reranker_requested", "") or ""
            ),
            "reranker_actual": str(
                getattr(trace, "reranker_actual", "") or ""
            ),
            "reranker_status": str(
                getattr(trace, "reranker_status", "") or ""
            ),
            "route": route,
            "traffic_origin": traffic_origin,
            "distribution_eligible": traffic_origin == "organic",
            "judge_depth": JUDGE_DEPTH,
            "queue_status": "enqueued",
        }
        pool = list(getattr(trace, "final_candidates", None) or
                    getattr(trace, "reranked_candidates", None) or [])[:JUDGE_DEPTH]
        entry["candidate_count"] = len(pool)
        # Private fields exist only in the bounded in-memory job.  They are
        # stripped before JSONL serialization, so raw personal queries and KB
        # text never enter the shadow log.
        entry["_question"] = question if selected else ""
        entry["_retrieval_question"] = route_question if selected else ""
        entry["_candidates"] = (
            [
                (rank, item.chunk_id, getattr(item, "rerank_score", None),
                 getattr(item, "document", "") or "")
                for rank, item in enumerate(pool, 1)
            ]
            if selected else []
        )
        entry["_log_path"] = str(_resolved_log_path(sample_config))
        _ensure_worker()
        try:
            _QUEUE.put_nowait(entry)
            _STATS["enqueued"] += 1
        except queue.Full:
            # Dropping is the correct behaviour under load and is counted, so
            # the release metrics can report it rather than hide it.
            _STATS["dropped"] += 1
            if selected:
                _STATS["dropped_selected"] += 1
            else:
                _STATS["dropped_unselected"] += 1
            _record_queue_drop(entry)
    except Exception:
        pass
    return None


def stats() -> dict[str, int]:
    result = dict(_STATS)
    result["queue_size"] = _QUEUE.qsize() if _QUEUE is not None else 0
    result["queue_unfinished"] = (
        int(getattr(_QUEUE, "unfinished_tasks", 0)) if _QUEUE is not None else 0
    )
    return result


def drain(timeout: float = 30.0) -> None:
    """Test/offline helper: wait for the queue to empty."""
    if _QUEUE is None:
        return
    deadline = time.time() + timeout
    # Queue.join() has no timeout and can hang a shutdown forever if a provider
    # call stalls.  Poll unfinished_tasks instead; production never waits here.
    while getattr(_QUEUE, "unfinished_tasks", 0) and time.time() < deadline:
        time.sleep(0.05)


__all__ = [
    "JUDGE_DEPTH",
    "JUDGE_QUESTION_ROLE",
    "LEGACY_SAMPLING_MODE",
    "ORGANIC_SAMPLING_MODE",
    "QUEUE_MAXSIZE",
    "drain",
    "enabled",
    "observe",
    "sampling_config",
    "sampling_decision",
    "should_trigger",
    "stats",
    "trigger_config",
    "trigger_features",
]
