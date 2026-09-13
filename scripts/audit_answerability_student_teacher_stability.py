#!/usr/bin/env python3
"""Run an exact same-prompt, deterministic 10% teacher stability audit.

The Pilot's first 889 rows used the frozen legacy v4 prompt.  A later question-
form hint changed the prompt SHA, so comparing those two protocols is not an
independent-repeat estimate.  This script rejudges 30 rows per Pilot style with
the exact legacy prompt bytes and records per-vote prompt lineage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability import (
    LEGACY_PROMPT_V1, LEGACY_PROMPT_V1_SHA256, MAX_CHUNK_CHARS,
    PROMPT_SHA256, parse_grade, resolve_model,
)
from rag_answerability_student_data import read_jsonl, validate_row
from rag_gate import _chat


PILOT_STYLES = ("natural", "implicit_oral", "long_noisy")
EXPECTED_LEGACY_SHA256 = "b144de3a7e2c2ca8c289235f6dfc879a059e86426c62d2b191c81da2d7bf0d46"


def _write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _migrate_vote_lineage(row: dict[str, Any]) -> None:
    votes = row.get("teacher_votes") or []
    for index, vote in enumerate(votes):
        if not isinstance(vote, dict):
            continue
        vote.setdefault(
            "prompt_sha256",
            row["teacher_prompt_sha256"] if index == 0 else PROMPT_SHA256,
        )
        vote.setdefault(
            "audit_reason", "base" if index == 0 else "protocol_comparison",
        )


def _reconcile_status(row: dict[str, Any]) -> None:
    votes = [vote for vote in row.get("teacher_votes") or [] if isinstance(vote, dict)]
    base = next((vote for vote in votes if vote.get("audit_reason") == "base"), None)
    if base is None:
        return
    same_protocol = [
        vote for vote in votes
        if vote is not base and vote.get("prompt_sha256") == base.get("prompt_sha256")
        and vote.get("audit_reason") in {"random_10pct", "high_risk_rejudge"}
    ]
    if not same_protocol:
        row["label_status"] = "teacher_single"
        return
    if any(
        vote.get("grade") != base.get("grade")
        or vote.get("relation") != base.get("relation")
        for vote in same_protocol
    ):
        row["label_status"] = "teacher_disagreement"
    else:
        row["label_status"] = "teacher_consensus"


def audit(path: Path, *, model: str, workers: int, target_per_style: int) -> dict[str, Any]:
    if LEGACY_PROMPT_V1_SHA256 != EXPECTED_LEGACY_SHA256:
        raise RuntimeError("legacy prompt bytes drifted; refusing incomparable audit")
    rows = read_jsonl(path)
    resolved_model = resolve_model(model)
    for row in rows:
        _migrate_vote_lineage(row)
        _reconcile_status(row)

    selected = []
    existing_by_style = Counter()
    for row in rows:
        legacy_random = any(
            isinstance(vote, dict)
            and vote.get("audit_reason") == "random_10pct"
            and vote.get("prompt_sha256") == LEGACY_PROMPT_V1_SHA256
            for vote in row.get("teacher_votes") or []
        )
        if legacy_random:
            existing_by_style[row["query_style"]] += 1
    for style in PILOT_STYLES:
        needed = max(0, target_per_style - existing_by_style[style])
        candidates = [
            row for row in rows
            if row["query_style"] == style
            and row["teacher_prompt_sha256"] == LEGACY_PROMPT_V1_SHA256
            and not any(
                isinstance(vote, dict)
                and vote.get("audit_reason") == "random_10pct"
                and vote.get("prompt_sha256") == LEGACY_PROMPT_V1_SHA256
                for vote in row.get("teacher_votes") or []
            )
        ]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{row['query_id']}\0{row['chunk_id']}\0stability-v1".encode("utf-8")
        ).hexdigest())
        if len(candidates) < needed:
            raise RuntimeError(f"not enough legacy rows for style {style}")
        selected.extend(candidates[:needed])

    failed: list[str] = []

    def judge(row: dict[str, Any]):
        prompt = LEGACY_PROMPT_V1.format(
            question=row["question"],
            retrieval_question=row.get("retrieval_question") or row["question"],
            chunk=row["chunk_text"][:MAX_CHUNK_CHARS],
        )
        raw = _chat(
            [{"role": "user", "content": prompt}],
            max_tokens=3000, temperature=0.0, model=resolved_model,
        )
        return row, parse_grade(raw, question=row["question"])

    if selected:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(judge, row): row for row in selected}
            for completed, future in enumerate(as_completed(futures), 1):
                submitted = futures[future]
                try:
                    row, verdict = future.result()
                except Exception as exc:
                    failed.append(
                        f"{submitted['query_id']}/{submitted['chunk_id']}:{type(exc).__name__}"
                    )
                    continue
                audit_vote = {
                    **(verdict or {}),
                    "available": verdict is not None,
                    "prompt_sha256": LEGACY_PROMPT_V1_SHA256,
                    "audit_reason": "random_10pct",
                }
                row["teacher_votes"].append(audit_vote)
                _reconcile_status(row)
                if completed % 10 == 0 or completed == len(futures):
                    _write_private(path, rows)
                if completed % 15 == 0 or completed == len(futures):
                    print(json.dumps({
                        "phase": "same_prompt_random_audit",
                        "completed": completed,
                        "total": len(futures),
                        "transport_failures": len(failed),
                    }), flush=True)
        _write_private(path, rows)
    if failed:
        raise RuntimeError(
            "stability audit transport incomplete; successes saved for resume; "
            f"failed={len(failed)} sample={failed[:3]}"
        )

    # Re-read through the public validator after the write.
    rows = read_jsonl(path)
    audited = []
    for row in rows:
        base = next((
            vote for vote in row["teacher_votes"]
            if vote.get("audit_reason") == "base"
            and vote.get("prompt_sha256") == LEGACY_PROMPT_V1_SHA256
        ), None)
        repeated = next((
            vote for vote in row["teacher_votes"]
            if vote.get("audit_reason") == "random_10pct"
            and vote.get("prompt_sha256") == LEGACY_PROMPT_V1_SHA256
        ), None)
        if base is not None and repeated is not None:
            audited.append((row, base, repeated))
    disagreements = sum(
        base.get("grade") != repeated.get("grade")
        for _row, base, repeated in audited
    )
    unavailable = sum(not repeated.get("available", True) for _row, _base, repeated in audited)
    return {
        "schema_version": "answerability-student-teacher-stability-v1",
        "status": "stable" if audited and disagreements / len(audited) <= 0.15 else "unstable",
        "teacher_model": resolved_model,
        "prompt_sha256": LEGACY_PROMPT_V1_SHA256,
        "same_prompt_audit_rows": len(audited),
        "audit_rows_by_style": dict(Counter(row["query_style"] for row, _b, _r in audited)),
        "grade_disagreements": disagreements,
        "grade_disagreement_rate": disagreements / len(audited) if audited else 1.0,
        "contract_unavailable": unavailable,
        "private_dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "private_text_exported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-per-style", type=int, default=30)
    args = parser.parse_args()
    report = audit(
        args.dataset, model=args.teacher_model, workers=args.workers,
        target_per_style=args.target_per_style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
