#!/usr/bin/env python3
"""Build/label the private answerability-student-v1 artifact.

Input is a frozen JSONL catalog with one query per line and exactly five
candidates.  The command is a dry-run unless ``--execute`` is passed.  It is
resumable, checks the v4 cache only through exact identity reconstruction, and
caps independent rejudging so a pilot cannot silently become a full run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability import (
    CACHE_PATH, PROMPT_SHA256, flush_cache, grade, resolve_model,
)
from rag_answerability_student_data import (
    QUERY_STYLES, SCHEMA, SPLITS, StudentDataError, audit_pilot,
    audit_split_isolation, public_manifest, reconnect_cached_verdict,
    sha256_text, validate_row,
)
from rag_answerability_student import DEFAULT_MODEL_NAME


CATALOG_SCHEMA = "answerability-student-catalog-v1"
PILOT_MAX_BASE_CALLS = 900
FULL_MAX_BASE_CALLS = 4000
PILOT_MAX_REJUDGE = 300
FULL_MAX_REJUDGE = 800


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    queries = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudentDataError(f"invalid catalog JSON on line {line_no}") from exc
        if row.get("schema_version") != CATALOG_SCHEMA:
            raise StudentDataError("unsupported student catalog schema")
        for field in ("query_id", "anchor_id", "question", "query_style", "split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise StudentDataError(f"catalog row missing {field}")
        if row["query_style"] not in QUERY_STYLES or row["split"] not in SPLITS:
            raise StudentDataError("catalog query_style/split is invalid")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 5:
            raise StudentDataError("every catalog query must contain exactly five candidates")
        seen = set()
        for candidate in candidates:
            for field in ("source_id", "chunk_id", "chunk_text"):
                if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                    raise StudentDataError(f"candidate missing {field}")
            if candidate["chunk_id"] in seen:
                raise StudentDataError("duplicate candidate inside one query")
            seen.add(candidate["chunk_id"])
            rank = candidate.get("current_bge_rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise StudentDataError("candidate current_bge_rank must be positive")
        queries.append(row)
    if not queries:
        raise StudentDataError("catalog is empty")
    return queries


def _audit_catalog(queries: list[dict[str, Any]], *, pilot: bool) -> dict[str, Any]:
    query_ids = [row["query_id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise StudentDataError("catalog contains duplicate query_id")
    anchor_split: dict[str, set[str]] = defaultdict(set)
    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for row in queries:
        anchor_split[row["anchor_id"]].add(row["split"])
        sources_by_split[row["split"]].update(
            candidate["source_id"] for candidate in row["candidates"]
        )
    leaks = [anchor for anchor, splits in anchor_split.items() if len(splits) > 1]
    if leaks:
        raise StudentDataError(f"anchor split leakage in catalog: {leaks[:3]}")
    if sources_by_split["train"] & sources_by_split["validation"]:
        raise StudentDataError("train/validation source leakage in catalog")
    if sources_by_split["blind_b"] & (
        sources_by_split["train"] | sources_by_split["validation"]
        | sources_by_split["blind_a"]
    ):
        raise StudentDataError("blind_b source leakage in catalog")
    candidates = len(queries) * 5
    limit = PILOT_MAX_BASE_CALLS if pilot else FULL_MAX_BASE_CALLS
    if candidates > limit:
        raise StudentDataError(f"catalog requires {candidates} base calls; cap is {limit}")
    if pilot and any(row["split"] in {"blind_a", "blind_b"} for row in queries):
        raise StudentDataError("pilot catalog must not contain blind rows")
    return {
        "queries": len(queries),
        "anchors": len(anchor_split),
        "candidates": candidates,
        "splits": dict(Counter(row["split"] for row in queries)),
        "styles": dict(Counter(row["query_style"] for row in queries)),
        "sources_by_split": {
            split: len(sources_by_split[split]) for split in SPLITS
        },
        "base_call_cap": limit,
        "missing_base_student_scores": sum(
            candidate.get("base_student_score") is None
            for row in queries for candidate in row["candidates"]
        ),
    }


def _score_base_student(queries: list[dict[str, Any]], model_name: str) -> int:
    missing = [
        (row, candidate) for row in queries for candidate in row["candidates"]
        if candidate.get("base_student_score") is None
    ]
    if not missing:
        return 0
    model_path = Path(model_name).expanduser()
    if not model_path.is_dir():
        cached = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_name
        model_path = cached if cached.is_dir() else model_path
    from sentence_transformers import CrossEncoder

    device = os.environ.get("OFFERCLAW_TORCH_DEVICE") or None
    model = CrossEncoder(
        str(model_path) if model_path.is_dir() else model_name,
        device=device, max_length=384,
    )
    pairs = [[row["question"], candidate["chunk_text"]] for row, candidate in missing]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=True)
    for (_row, candidate), score in zip(missing, scores):
        candidate["base_student_score"] = float(score)
    return len(missing)


def _expanded(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    question = row["question"]
    chunk = candidate["chunk_text"]
    return {
        "schema_version": SCHEMA,
        "query_id": row["query_id"],
        "anchor_id": row["anchor_id"],
        "source_id": candidate["source_id"],
        "chunk_id": candidate["chunk_id"],
        "question": question,
        "retrieval_question": row.get("retrieval_question") or question,
        "query_style": row["query_style"],
        "split": row["split"],
        "chunk_text": chunk,
        "chunk_text_sha256": sha256_text(chunk),
        "current_bge_rank": candidate["current_bge_rank"],
        "current_bge_score": candidate.get("current_bge_score"),
        "base_student_score": candidate.get("base_student_score"),
        "known_relevant": bool(candidate.get("known_relevant")),
        "teacher_model": None,
        "teacher_prompt_sha256": PROMPT_SHA256,
        "teacher_votes": [],
        "teacher_grade": None,
        "teacher_relation": None,
        "label_status": "unlabeled",
        "human_review_status": "not_reviewed",
        "provenance": row.get("provenance") or {},
    }


def _load_cache() -> dict[str, Any]:
    try:
        value = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _resume_rows(path: Path, *, teacher_model: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Pilot labels collected before the prompt hash became an explicit
        # column used this exact in-repository prompt.  Backfill only the
        # current hash/model; a future prompt change produces a different hash
        # and therefore cannot reuse these rows silently.
        row.setdefault("teacher_prompt_sha256", PROMPT_SHA256)
        row.setdefault("teacher_model", teacher_model)
        for index, vote in enumerate(row.get("teacher_votes") or []):
            if not isinstance(vote, dict):
                continue
            vote.setdefault(
                "prompt_sha256",
                row["teacher_prompt_sha256"] if index == 0 else PROMPT_SHA256,
            )
            vote.setdefault(
                "audit_reason", "base" if index == 0 else "protocol_comparison",
            )
        validate_row(row)
        rows[(row["query_id"], row["chunk_id"])] = row
    return rows


def _write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _should_rejudge(row: dict[str, Any], group: list[dict[str, Any]]) -> bool:
    if row["teacher_grade"] == 2:
        return True
    if row.get("known_relevant") and row["teacher_grade"] < 3:
        return True
    if row["current_bge_rank"] == 1 and row["teacher_grade"] < 3 and any(
        item["teacher_grade"] == 3 for item in group
    ):
        return True
    return int(sha256_text(row["query_id"] + "\0" + row["chunk_id"])[:8], 16) % 10 == 0


def _apply_vote(row: dict[str, Any], verdict: dict[str, Any], *, status: str) -> None:
    row["teacher_votes"].append({
        **verdict,
        "prompt_sha256": PROMPT_SHA256,
        "audit_reason": "base",
    })
    row["teacher_grade"] = int(verdict["grade"])
    row["teacher_relation"] = verdict["relation"]
    row["label_status"] = status


def build(queries: list[dict[str, Any]], *, output: Path, model: str,
          base_student_model: str, pilot: bool,
          execute: bool, workers: int = 6,
          rejudge_cap_override: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_audit = _audit_catalog(queries, pilot=pilot)
    base_scores_computed = _score_base_student(queries, base_student_model) if execute else 0
    resolved_model = resolve_model(model)
    cache = _load_cache()
    resumed = _resume_rows(output, teacher_model=resolved_model)
    rows = []
    pending = []
    cache_hits = base_calls = 0
    for query in queries:
        for candidate in query["candidates"]:
            row = _expanded(query, candidate)
            row["teacher_model"] = resolved_model
            key = (row["query_id"], row["chunk_id"])
            if key in resumed:
                rows.append(resumed[key])
                continue
            verdict = reconnect_cached_verdict(row, cache, model=resolved_model)
            if verdict is not None:
                _apply_vote(row, verdict, status="cache_exact")
                cache_hits += 1
            else:
                pending.append(row)
            rows.append(row)

    if not execute:
        return rows, {
            "status": "dry_run",
            "catalog": catalog_audit,
            "exact_cache_hits": cache_hits,
            "planned_base_calls": len(pending),
            "resumed_rows": len(resumed),
            "base_student_model": base_student_model,
            "base_scores_computed": 0,
        }

    # Persist schema backfills on resumed rows before any network call.  A
    # provider outage must not leave a successfully migrated private artifact
    # only in memory.
    if resumed:
        _write_private(
            output,
            [item for item in rows if item["teacher_grade"] is not None],
        )

    if pending:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def first_vote(item):
            return item, grade(
                item["question"], item["chunk_text"], model=resolved_model,
                retrieval_question=item["retrieval_question"], use_cache=True,
            )

        failed_first_votes: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(first_vote, row): row for row in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                submitted_row = futures[future]
                try:
                    row, verdict = future.result()
                except Exception as exc:  # keep successful parallel work resumable
                    failed_first_votes.append(
                        f"{submitted_row['query_id']}/{submitted_row['chunk_id']}:"
                        f"{type(exc).__name__}"
                    )
                    verdict = None
                    row = submitted_row
                base_calls += 1
                if verdict is None:
                    if not failed_first_votes or not failed_first_votes[-1].startswith(
                        f"{row['query_id']}/{row['chunk_id']}:"
                    ):
                        failed_first_votes.append(
                            f"{row['query_id']}/{row['chunk_id']}:unavailable"
                        )
                else:
                    _apply_vote(row, verdict, status="teacher_single")
                if completed % 10 == 0 or completed == len(futures):
                    _write_private(
                        output,
                        [item for item in rows if item["teacher_grade"] is not None],
                    )
                if completed % 25 == 0 or completed == len(futures):
                    print(json.dumps({
                        "phase": "base_teacher",
                        "completed": completed,
                        "total": len(futures),
                        "failed": len(failed_first_votes),
                    }), flush=True)
        flush_cache()
        if failed_first_votes:
            raise StudentDataError(
                "teacher base pass incomplete; successful rows were saved for resume; "
                f"failed={len(failed_first_votes)} sample={failed_first_votes[:3]}"
            )

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    candidates = [
        row for group in by_query.values() for row in group
        if _should_rejudge(row, group) and len(row["teacher_votes"]) < 2
    ]
    candidates.sort(key=lambda row: (
        row["teacher_grade"] != 2,
        not row.get("known_relevant"),
        row["current_bge_rank"], row["query_id"], row["chunk_id"],
    ))
    maximum_rejudge_cap = PILOT_MAX_REJUDGE if pilot else FULL_MAX_REJUDGE
    if rejudge_cap_override is None:
        rejudge_cap = maximum_rejudge_cap
    elif not 0 <= rejudge_cap_override <= maximum_rejudge_cap:
        raise StudentDataError(
            f"rejudge cap must be between 0 and {maximum_rejudge_cap}"
        )
    else:
        rejudge_cap = rejudge_cap_override
    rejudge_calls = 0
    selected_rejudges = candidates[:rejudge_cap]
    if selected_rejudges:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def second_vote(item):
            return item, grade(
                item["question"], item["chunk_text"], model=resolved_model,
                retrieval_question=item["retrieval_question"], use_cache=False,
            )

        failed_rejudges: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(second_vote, row): row for row in selected_rejudges}
            for completed, future in enumerate(as_completed(futures), 1):
                submitted_row = futures[future]
                try:
                    row, verdict = future.result()
                except Exception as exc:
                    row, verdict = submitted_row, None
                    failed_rejudges.append(
                        f"{row['query_id']}/{row['chunk_id']}:{type(exc).__name__}"
                    )
                rejudge_calls += 1
                if verdict is not None:
                    previous = row["teacher_votes"][0]
                    row["teacher_votes"].append({
                        **verdict,
                        "prompt_sha256": PROMPT_SHA256,
                        "audit_reason": "high_risk_rejudge",
                    })
                    if (previous.get("grade") == verdict.get("grade")
                            and previous.get("relation") == verdict.get("relation")):
                        row["label_status"] = "teacher_consensus"
                    else:
                        row["label_status"] = "teacher_disagreement"
                elif not failed_rejudges or not failed_rejudges[-1].startswith(
                    f"{row['query_id']}/{row['chunk_id']}:"
                ):
                    failed_rejudges.append(
                        f"{row['query_id']}/{row['chunk_id']}:unavailable"
                    )
                if completed % 10 == 0 or completed == len(futures):
                    _write_private(output, rows)
                if completed % 25 == 0 or completed == len(futures):
                    print(json.dumps({
                        "phase": "teacher_rejudge",
                        "completed": completed,
                        "total": len(futures),
                        "failed": len(failed_rejudges),
                    }), flush=True)
        flush_cache()
        if failed_rejudges:
            raise StudentDataError(
                "teacher rejudge incomplete; successful rows were saved for resume; "
                f"failed={len(failed_rejudges)} sample={failed_rejudges[:3]}"
            )

    for row in rows:
        validate_row(row)
    isolation = audit_split_isolation(rows)
    _write_private(output, rows)
    manifest = public_manifest(rows, private_path=output)
    manifest.update({
        "teacher_model": resolved_model,
        "catalog_audit": catalog_audit,
        "split_isolation": isolation,
        "base_teacher_calls": base_calls,
        "exact_cache_hits": cache_hits,
        "rejudge_calls": rejudge_calls,
        "rejudge_cap": rejudge_cap,
        "total_new_teacher_calls": base_calls + rejudge_calls,
        "base_student_model": base_student_model,
        "base_scores_computed": base_scores_computed,
    })
    if pilot:
        manifest["pilot_audit"] = audit_pilot(rows)
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--base-student-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rejudge-cap", type=int)
    args = parser.parse_args()
    queries = _load_catalog(args.catalog)
    rows, report = build(
        queries, output=args.output, model=args.teacher_model,
        base_student_model=args.base_student_model,
        pilot=args.pilot, execute=args.execute, workers=args.workers,
        rejudge_cap_override=args.rejudge_cap,
    )
    if args.execute:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
