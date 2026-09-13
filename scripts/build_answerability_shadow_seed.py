#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a privacy-split answerability-trigger seed from historical API traffic.

This is not a blind release set and it is not synthetic traffic.  It extracts
real, previously observed ``/api/stream`` query text from ``logs/api.log``, then
replays each unique query through the *current* baseline retrieval pipeline.
The replay is intentionally labelled as such: current index state and current
retrieval code may differ from what served the historical request.

Outputs are split in two:

* ``claude_calibration.jsonl`` contains numeric trigger features and structured
  judge labels, but no raw query, judge reason, document text, or source path.
* ``private_audit.jsonl`` contains the local evidence needed to audit bad judge
  labels.  It lives under ``logs/`` (gitignored) and must not be published.

Dev80, V1 and Final90 are not read by this script.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "answerability-trigger-seed-v1"
LOGGER_TEXT_LIMIT = 60
KNOWN_AUTOMATION_QUERIES = {"x", "RAG", "什么是RAG", "什么是 RAG"}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip())
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": "", "dirty": None}


def extract_stream_history(api_log: Path, tests_dir: Path) -> tuple[list[dict], dict]:
    """Extract strict historical replay candidates and exclusion counts."""
    events: list[tuple[str, str]] = []
    malformed = 0
    with api_log.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            message = str(payload.get("msg") or "")
            if not message.startswith("stream start q="):
                continue
            try:
                question = ast.literal_eval(message.split("=", 1)[1])
            except (SyntaxError, ValueError):
                malformed += 1
                continue
            if isinstance(question, str):
                events.append((str(payload.get("ts") or ""), question))

    test_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in tests_dir.rglob("*.py")
    )
    by_question: dict[str, list[str]] = {}
    for timestamp, question in events:
        by_question.setdefault(question, []).append(timestamp)

    excluded = Counter()
    selected: list[dict[str, Any]] = []
    for question, timestamps in by_question.items():
        if question in KNOWN_AUTOMATION_QUERIES:
            excluded["known_automation_query"] += 1
            continue
        # The API logger stores req.query[:60].  Exactly 60 characters is
        # ambiguous, so strict replay excludes it rather than inventing a tail.
        if len(question) >= LOGGER_TEXT_LIMIT:
            excluded["possibly_truncated_at_60_chars"] += 1
            continue
        if question in test_text:
            excluded["present_in_test_source"] += 1
            continue
        selected.append({
            "question": question,
            "question_sha256": _sha256_text(question),
            "occurrence_count": len(timestamps),
            "first_seen": min(timestamps),
            "last_seen": max(timestamps),
        })
    selected.sort(key=lambda row: (row["first_seen"], row["question_sha256"]))
    audit = {
        "api_events": len(events),
        "unique_queries": len(by_question),
        "selected_unique_queries": len(selected),
        "selected_weighted_events": sum(row["occurrence_count"] for row in selected),
        "malformed_log_lines": malformed,
        "exclusions_by_unique_query": dict(sorted(excluded.items())),
        "logger_text_limit": LOGGER_TEXT_LIMIT,
    }
    return selected, audit


def _query_label(gate_decision: bool, verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a reviewable query-level target from Top-N chunk verdicts."""
    available = all(row.get("verdict") is not None for row in verdicts)
    if not available:
        return {
            "label_available": False,
            "recommended_action": "judge_unavailable",
            "selected_rank": None,
            "intervention_needed": None,
            "intervention_type": "judge_unavailable",
        }

    usable = [
        row for row in verdicts
        if int((row.get("verdict") or {}).get("grade", -1)) >= 3
        and row.get("action") in {"answer", "correct_premise"}
    ]
    selected = sorted(
        usable,
        key=lambda row: (-int(row["verdict"]["grade"]), int(row["rank"])),
    )[0] if usable else None
    recommended = selected["action"] if selected else "abstain"
    selected_rank = int(selected["rank"]) if selected else None

    if gate_decision:
        if recommended == "abstain":
            kind = "block_unsupported"
        elif recommended == "correct_premise":
            kind = "correct_premise"
        elif selected_rank != 1:
            kind = "rerank_evidence"
        else:
            kind = "no_change"
    else:
        if recommended == "answer":
            kind = "rescue_answer"
        elif recommended == "correct_premise":
            kind = "rescue_correction"
        else:
            kind = "no_change"
    return {
        "label_available": True,
        "recommended_action": recommended,
        "selected_rank": selected_rank,
        "intervention_needed": kind != "no_change",
        "intervention_type": kind,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(float(ordered[index]), 4)


def routing_shadow_coverage(
    public_rows: list[dict[str, Any]], path: Path,
) -> dict[str, Any]:
    """Report whether historical route observations exist for this seed.

    Route observations are diagnostics, never trigger features.  A direct RAG
    replay without route evidence must not be described as route-faithful.
    """
    if not path.is_file():
        return {"available": False, "matched_queries": 0,
                "reference_kb_queries": 0}
    routes: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        routes.setdefault(str(row.get("question_hash") or ""), []).append(row)
    matched = 0
    reference = 0
    for row in public_rows:
        observations = routes.get(row["question_sha256"], [])
        if not observations:
            continue
        matched += 1
        latest = observations[-1]
        selected_routes = set(latest.get("candidate") or []) | set(
            latest.get("legacy") or []
        )
        if "reference_kb.search" in selected_routes:
            reference += 1
    return {
        "available": True,
        "routing_shadow_path": str(path.resolve()),
        "routing_shadow_rows": sum(len(items) for items in routes.values()),
        "matched_queries": matched,
        "reference_kb_queries": reference,
        "unmatched_queries": len(public_rows) - matched,
    }


def build_seed(
    selected: list[dict[str, Any]], *, depth: int, workers: int,
    model: str, features_only: bool,
) -> tuple[list[dict], list[dict], dict]:
    from rag_answerability import action, cache_key, flush_cache, grade, _load_cache
    from rag_gate import _retrieve_and_classify
    from rag_retrieval_trace import RetrievalTrace, resolve_retrieval_profile
    from rag_shadow_answerability import trigger_features

    # Prevent a developer shell from scheduling duplicate live shadow jobs while
    # this controlled replay performs its own explicit labelling.
    os.environ["RAG_SHADOW_ANSWERABILITY"] = "0"
    profile = resolve_retrieval_profile("baseline")
    replay_rows: list[dict[str, Any]] = []
    judge_jobs: list[tuple[int, int, str, str]] = []

    for index, source in enumerate(selected):
        trace = RetrievalTrace(profile)
        started = time.perf_counter()
        result = _retrieve_and_classify(
            source["question"], 5, retrieval_profile=profile,
            _trace=trace, _shadow_internal=True,
        )
        replay_ms = round((time.perf_counter() - started) * 1000, 3)
        candidates = list(trace.final_candidates)[:depth]
        row = {
            **source,
            "features": trigger_features(trace),
            "gate_decision": bool(trace.gate_decision),
            "effective_hit": bool(trace.effective_hit),
            "retrieval_profile": profile.name,
            "index_fingerprint": trace.index_fingerprint,
            "reranker_requested": trace.reranker_requested,
            "reranker_actual": trace.reranker_actual,
            "reranker_status": trace.reranker_status,
            "retrieval_replay_ms": replay_ms,
            "candidates_private": [
                {
                    "rank": rank,
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "source_type": item.source_type,
                    "heading_path": item.heading_path,
                    "rerank_score": item.rerank_score,
                    "document": item.document,
                }
                for rank, item in enumerate(candidates, 1)
            ],
            "verdicts": [],
        }
        replay_rows.append(row)
        if not features_only:
            for rank, item in enumerate(candidates, 1):
                judge_jobs.append((index, rank, source["question"], item.document))
        print(
            f"retrieval {index + 1}/{len(selected)} "
            f"gate={int(bool(result.get('in_kb')))} q={source['question_sha256'][:10]}",
            file=sys.stderr, flush=True,
        )

    cache = _load_cache()

    def judge_one(job: tuple[int, int, str, str]) -> tuple[int, int, dict]:
        row_index, rank, question, document = job
        key = cache_key(model, question, document)
        cache_hit = key in cache
        started = time.perf_counter()
        verdict = grade(question, document, model=model)
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        return row_index, rank, {
            "rank": rank,
            "verdict": verdict,
            "action": action(verdict),
            "cache_hit": cache_hit,
            "latency_ms": elapsed,
        }

    if judge_jobs:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(judge_one, job) for job in judge_jobs]
            for future in as_completed(futures):
                row_index, rank, verdict_row = future.result()
                candidate = replay_rows[row_index]["candidates_private"][rank - 1]
                verdict_row["chunk_id"] = candidate["chunk_id"]
                verdict_row["rerank_score"] = candidate["rerank_score"]
                replay_rows[row_index]["verdicts"].append(verdict_row)
                done += 1
                print(
                    f"judge {done}/{len(judge_jobs)}",
                    file=sys.stderr, flush=True,
                )
        flush_cache()

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for row in replay_rows:
        verdicts = sorted(row["verdicts"], key=lambda item: item["rank"])
        label = (
            _query_label(row["gate_decision"], verdicts)
            if verdicts else {
                "label_available": False,
                "recommended_action": "not_judged",
                "selected_rank": None,
                "intervention_needed": None,
                "intervention_type": "not_judged",
            }
        )
        public_verdicts = [
            {
                "rank": item["rank"],
                "chunk_id_sha256": _sha256_text(item["chunk_id"]),
                "rerank_score": item["rerank_score"],
                "grade": (item["verdict"] or {}).get("grade"),
                "relation": (item["verdict"] or {}).get("relation"),
                "action": item["action"],
                "available": item["verdict"] is not None,
                "cache_hit": item["cache_hit"],
                "latency_ms": item["latency_ms"],
            }
            for item in verdicts
        ]
        public_rows.append({
            "schema_version": SCHEMA,
            "query_id": row["question_sha256"][:16],
            "question_sha256": row["question_sha256"],
            "occurrence_count": row["occurrence_count"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "features": row["features"],
            "gate_decision": row["gate_decision"],
            "effective_hit": row["effective_hit"],
            "retrieval_profile": row["retrieval_profile"],
            "index_fingerprint": row["index_fingerprint"],
            "reranker_requested": row["reranker_requested"],
            "reranker_actual": row["reranker_actual"],
            "reranker_status": row["reranker_status"],
            "retrieval_replay_ms": row["retrieval_replay_ms"],
            "judge_model_requested": model,
            "judge_depth": depth,
            "verdicts": public_verdicts,
            **label,
        })
        private_candidates = []
        verdict_by_rank = {item["rank"]: item for item in verdicts}
        for candidate in row["candidates_private"]:
            verdict_row = verdict_by_rank.get(candidate["rank"], {})
            private_candidates.append({
                **candidate,
                "verdict": verdict_row.get("verdict"),
                "action": verdict_row.get("action"),
            })
        private_rows.append({
            "query_id": row["question_sha256"][:16],
            "question": row["question"],
            "occurrence_count": row["occurrence_count"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "candidates": private_candidates,
            **label,
        })

    label_counts = Counter(row["intervention_type"] for row in public_rows)
    available = [row for row in public_rows if row["label_available"]]
    margins = [
        float(row["features"]["rerank_margin"])
        for row in available if row["features"].get("rerank_margin") is not None
    ]
    judge_latencies = [
        float(verdict["latency_ms"])
        for row in public_rows for verdict in row["verdicts"]
        if not verdict["cache_hit"] and verdict["available"]
    ]
    summary = {
        "schema_version": SCHEMA,
        "unique_queries": len(public_rows),
        "weighted_events": sum(row["occurrence_count"] for row in public_rows),
        "label_available_queries": len(available),
        "judge_calls_planned": len(judge_jobs),
        "judge_unavailable": sum(
            1 for row in public_rows for verdict in row["verdicts"]
            if not verdict["available"]
        ),
        "judge_cache_hits": sum(
            1 for row in public_rows for verdict in row["verdicts"]
            if verdict["cache_hit"]
        ),
        "intervention_type_counts": dict(sorted(label_counts.items())),
        "intervention_needed_queries": sum(
            1 for row in available if row["intervention_needed"]
        ),
        "gate_accept_queries": sum(1 for row in public_rows if row["gate_decision"]),
        "gate_accept_intervention_queries": sum(
            1 for row in available
            if row["gate_decision"] and row["intervention_needed"]
        ),
        "gate_accept_no_change_queries": sum(
            1 for row in available
            if row["gate_decision"] and not row["intervention_needed"]
        ),
        "rerank_margin_p10_p50_p90": {
            "p10": _percentile(margins, 0.10),
            "p50": _percentile(margins, 0.50),
            "p90": _percentile(margins, 0.90),
        },
        "uncached_judge_latency_ms": {
            "count": len(judge_latencies),
            "median": round(statistics.median(judge_latencies), 3)
            if judge_latencies else None,
            "p95": _percentile(judge_latencies, 0.95),
        },
    }
    return public_rows, private_rows, summary


def _handoff_text(
    summary: dict[str, Any], review: dict[str, Any], runs: dict[str, Any],
) -> str:
    route = summary.get("historical_routing_shadow") or {}
    flags = review.get("flags") or []
    return f"""# Claude handoff: answerability trigger seed

This directory is a **historical real-query replay seed**, not live concurrent
traffic and not a blind release set. It contains {summary['unique_queries']}
unique queries ({summary['weighted_events']} weighted historical events), replayed
against the current index and retrieval code.

## Validity gate

Do **not** fit or recommend a production threshold from this package. Direct
replay bypassed the semantic router. Historical routing shadow matched only
{route.get('matched_queries', 0)}/{summary['unique_queries']} queries, and only
{route.get('reference_kb_queries', 0)} matched query actually included
`reference_kb.search`. The present gate accepted only
{summary['gate_accept_queries']} queries: judge labels mark
{summary['gate_accept_intervention_queries']} as interventions and
{summary['gate_accept_no_change_queries']} as no-change. That is not enough
support to estimate accepted-path precision or trigger recall.

`manual_review.json` contains {len(flags)} preliminary label concerns found
during audit. Treat those rows as unresolved until a second reviewer checks the
raw query/evidence in `private_audit.jsonl`.

Use `run_observations.json` for uncached judge availability and latency. The
current JSONL export may show cache hits because the package was rebuilt to
refresh hashes and validity metadata; cached timings are not network latency.

Use `claude_calibration.jsonl` for numeric calibration. It contains no raw query,
judge reason, document text, chunk ID, or source path. `private_audit.jsonl` is local-only
and may be opened solely to audit questionable labels; do not copy or publish it.

Calibration target:

- For the first safety-only rollout, the target is an accepted-path intervention:
  `gate_decision == true` and `intervention_type` in
  (`block_unsupported`, `correct_premise`, `rerank_evidence`). Rejected-path
  `rescue_answer` rows are diagnostics, not permission to weaken the gate.
- Diagnose by `intervention_type`; `correct_premise` and `block_unsupported` are
  higher-risk misses than a simple `rerank_evidence` rescue.
- Candidate features must be limited to the `features` object. Never use judge
  grade/relation, chunk IDs, timestamps, query hashes, or occurrence count as a
  trigger feature. `occurrence_count` is a sample weight only.
- Report a Pareto table of weighted call rate versus weighted intervention recall,
  plus recall by intervention type. Do not emit a production threshold from this
  seed alone.
- Prefer an interpretable rule with at most 2-3 conditions. Show bootstrap
  uncertainty; n={summary['label_available_queries']} labelled unique queries is
  too small for a final release decision.
- Do not read or tune on Dev80, V1, or Final90. After selecting candidate rules,
  freeze them before evaluating a newly collected blind release window.

Before analysis, verify file hashes in `manifest.json` and confirm every row has
the same `index_fingerprint` and `judge_model_requested`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-log", type=Path, default=ROOT / "logs" / "api.log")
    parser.add_argument("--tests-dir", type=Path, default=ROOT / "tests")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--routing-shadow", type=Path,
        default=ROOT / ".offerclaw" / "routing_shadow.jsonl",
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args(argv)

    if args.depth < 1:
        parser.error("--depth must be positive")
    from rag_answerability import resolve_model
    model = resolve_model(args.model)
    if not model and not args.features_only:
        parser.error("judge model could not be resolved")

    selected, selection_audit = extract_stream_history(
        args.api_log.resolve(), args.tests_dir.resolve()
    )
    if args.limit is not None:
        selected = selected[:max(0, args.limit)]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_rows, private_rows, summary = build_seed(
        selected, depth=args.depth, workers=args.workers,
        model=model, features_only=args.features_only,
    )
    summary["historical_routing_shadow"] = routing_shadow_coverage(
        public_rows, args.routing_shadow.resolve()
    )

    public_path = output_dir / "claude_calibration.jsonl"
    private_path = output_dir / "private_audit.jsonl"
    summary_path = output_dir / "summary.json"
    handoff_path = output_dir / "CLAUDE_HANDOFF.md"
    review_path = output_dir / "manual_review.json"
    runs_path = output_dir / "run_observations.json"
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        review = {"schema_version": "answerability-seed-manual-review-v1",
                  "status": "not_started", "flags": []}
        _json_dump(review_path, review)
    try:
        runs = json.loads(runs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        runs = {"schema_version": "answerability-seed-run-observations-v1",
                "runs": []}
        _json_dump(runs_path, runs)
    _jsonl_dump(public_path, public_rows)
    _jsonl_dump(private_path, private_rows)
    _json_dump(summary_path, summary)
    handoff_path.write_text(
        _handoff_text(summary, review, runs), encoding="utf-8"
    )

    fingerprints = sorted({row["index_fingerprint"] for row in public_rows})
    manifest = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "replay_kind": "historical_real_query_text_on_current_index",
        "release_status": "seed_only_not_blind_not_production_threshold",
        "source_log": str(args.api_log.resolve()),
        "selection_contract": {
            "endpoint_log_message": "stream start q=...",
            "exclude_known_automation_queries": sorted(KNOWN_AUTOMATION_QUERIES),
            "exclude_if_present_in_tests_source": True,
            "exclude_length_gte_logger_limit": LOGGER_TEXT_LIMIT,
            "deduplicate_by_exact_query": True,
            "preserve_occurrence_count_as_weight": True,
            "forbidden_sources": ["Dev80", "V1", "Final90"],
        },
        "selection_audit": selection_audit,
        "selected_after_limit": len(selected),
        "judge_depth": args.depth,
        "judge_model_requested": model,
        "features_only": args.features_only,
        "index_fingerprints": fingerprints,
        "git": _git_state(),
        "files": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in (
                public_path, private_path, summary_path, handoff_path,
                review_path, runs_path,
            )
        },
    }
    manifest_path = output_dir / "manifest.json"
    _json_dump(manifest_path, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "selection": selection_audit,
        "summary": summary,
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
