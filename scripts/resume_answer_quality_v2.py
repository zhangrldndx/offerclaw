#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume only the missing judges of an interrupted answer-quality v2 run.

The frozen generator/evaluator remains ``eval_answer_quality_v2.py``.  This
companion never regenerates an answer and never changes a completed judgment:
it verifies the frozen selection, row order, immutable generation payload and
all existing judgments before requesting only absent judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_answer_quality_v2 as aq


class ResumeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        copied = dict(row)
        copied["judges"] = {}
        clean.append(copied)
    return clean


def validate_partial(
    run_dir: Path, selection_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    run_dir = aq.ensure_repository_external(run_dir)
    raw_path = run_dir / "raw_rows.jsonl"
    if not raw_path.is_file():
        raise ResumeError(f"missing partial raw rows: {raw_path}")
    manifest = aq._read_json(selection_path)
    resolved = aq.resolve_selection(manifest)
    rows = aq._read_jsonl(raw_path)
    if len(rows) != len(resolved):
        raise ResumeError(
            f"generation is incomplete: {len(rows)}/{len(resolved)} rows"
        )
    expected_ids = [item["eval_id"] for item in resolved]
    if [row.get("eval_id") for row in rows] != expected_ids:
        raise ResumeError("partial row identity/order differs from frozen selection")
    if any(not (row.get("generation") or {}).get("available") for row in rows):
        raise ResumeError("one or more frozen generations are unavailable")

    for row in rows:
        for judge_name, result in (row.get("judges") or {}).items():
            if judge_name not in aq.JUDGE_NAMES:
                raise ResumeError(f"unexpected judge in partial: {judge_name}")
            if not result.get("available"):
                continue
            validated = aq.validate_judgment(result.get("judgment") or {}, row)
            expected_scores = aq.score_judgment(row, validated)
            if result.get("scores") != expected_scores:
                raise ResumeError(
                    f"existing judgment score drift: {row['eval_id']} {judge_name}"
                )

    generation_path = run_dir / "generation_rows.jsonl"
    generation_rows = _generation_rows(rows)
    generation_bytes = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in generation_rows
    ).encode("utf-8")
    if generation_path.exists() and generation_path.read_bytes() != generation_bytes:
        raise ResumeError("immutable generation artifact differs from partial rows")
    if not generation_path.exists():
        generation_path.write_bytes(generation_bytes)
    return rows, _sha256(generation_path)


def resume(
    run_dir: Path, selection_path: Path, *, publish_summary: bool = False,
) -> dict[str, Any]:
    run_dir = aq.ensure_repository_external(run_dir)
    rows, generation_sha = validate_partial(run_dir, selection_path)
    raw_path = run_dir / "raw_rows.jsonl"
    for judge_name in aq.JUDGE_NAMES:
        for position, row in enumerate(rows, 1):
            existing = (row.get("judges") or {}).get(judge_name) or {}
            if existing.get("available"):
                continue
            print(
                f"[aqv2-resume] judge={judge_name} {position}/{len(rows)} "
                f"{row['eval_id']}",
                flush=True,
            )
            try:
                result = aq._judge_once(judge_name, row)
                result["scores"] = aq.score_judgment(row, result["judgment"])
            except Exception as exc:
                result = {
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error_sha256": aq._sha256_text(str(exc)),
                }
            row.setdefault("judges", {})[judge_name] = result
            aq._write_jsonl(raw_path, rows)

    raw_sha = _sha256(raw_path)
    summary = aq.summarize_rows(
        rows,
        selection_sha256=_sha256(selection_path),
        raw_sha256=raw_sha,
    )
    summary["run"] = {
        "run_id": run_dir.name,
        "generation_artifact_sha256_before_judging": generation_sha,
        "raw_artifact_bytes": raw_path.stat().st_size,
        "raw_artifact_location": "repository_external",
        "resumed_from_verified_partial": True,
        "resume_tool_sha256": _sha256(Path(__file__)),
    }
    (run_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if publish_summary:
        if summary["status"] != "complete":
            raise ResumeError("judge coverage is incomplete; refusing public summary")
        aq.SUMMARY_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--publish-summary", action="store_true")
    args = parser.parse_args()
    try:
        summary = resume(
            args.run_dir,
            args.selection,
            publish_summary=args.publish_summary,
        )
    except (OSError, ValueError, aq.AnswerQualityV2Error, ResumeError) as exc:
        print(f"[aqv2-resume] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
