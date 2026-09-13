#!/usr/bin/env python3
"""Freeze balanced-mode uncertainty thresholds on validation only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability_student import BALANCED_POLICY_SCHEMA, checkpoint_tree_sha256
from rag_answerability_student_data import StudentDataError, read_jsonl
from scripts.evaluate_answerability_student_blind import _load_model, _predict_group


CONFIDENCE_GRID = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
GAP_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
MAX_TEACHER_CALL_RATE = 0.15


def calibrate(model_dir: Path, dataset: Path, *, device: str | None = None):
    def numeric_or(value, fallback):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

    rows = read_jsonl(dataset)
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise StudentDataError("balanced calibration cannot read blind rows")
    validation = [row for row in rows if row["split"] == "validation"]
    by_query = defaultdict(list)
    for row in validation:
        by_query[row["query_id"]].append(row)
    import torch
    tokenizer, model, actual_device, manifest = _load_model(model_dir, device)
    observations = []
    for query_id in sorted(by_query):
        group = by_query[query_id]
        predictions = _predict_group(tokenizer, model, actual_device, group, torch)
        by_chunk = {item["row"]["chunk_id"]: item for item in predictions}
        student = sorted(
            group,
            key=lambda row: (
                -by_chunk[row["chunk_id"]]["grade"],
                -numeric_or(row.get("current_bge_score"), -row["current_bge_rank"]),
            ),
        )
        teacher = sorted(
            group,
            key=lambda row: (
                -row["teacher_grade"],
                -numeric_or(row.get("current_bge_score"), -row["current_bge_rank"]),
            ),
        )
        base = min(group, key=lambda row: row["current_bge_rank"])
        top = by_chunk[student[0]["chunk_id"]]
        second = by_chunk[student[1]["chunk_id"]] if len(student) > 1 else {"score": -1.0}
        observations.append({
            "base_hit": int(base["teacher_grade"] == 3),
            "student_hit": int(student[0]["teacher_grade"] == 3),
            "teacher_hit": int(teacher[0]["teacher_grade"] == 3),
            "confidence": top["confidence"],
            "gap": top["score"] - second["score"],
        })
    if not observations:
        raise StudentDataError("validation contains no queries")
    base_hits = sum(row["base_hit"] for row in observations)
    student_hits = sum(row["student_hit"] for row in observations)
    teacher_hits = sum(row["teacher_hit"] for row in observations)
    candidates = []
    for confidence in CONFIDENCE_GRID:
        for gap in GAP_GRID:
            fallback = [
                row["confidence"] < confidence or row["gap"] < gap
                for row in observations
            ]
            call_rate = sum(fallback) / len(fallback)
            if call_rate > MAX_TEACHER_CALL_RATE:
                continue
            hybrid_hits = sum(
                row["teacher_hit"] if use_teacher else row["student_hit"]
                for row, use_teacher in zip(observations, fallback)
            )
            teacher_gain = teacher_hits - base_hits
            recovery = (
                (hybrid_hits - base_hits) / teacher_gain if teacher_gain > 0 else 1.0
            )
            candidates.append({
                "min_top_confidence": confidence,
                "min_expected_grade_gap": gap,
                "observed_teacher_call_rate": call_rate,
                "hybrid_r1_hits": hybrid_hits,
                "teacher_gain_recovery": recovery,
            })
    if not candidates:
        return {
            "schema_version": BALANCED_POLICY_SCHEMA,
            "status": "no_balanced_policy_keep_student_and_teacher_modes",
            "frozen_on_split": "validation",
            "student_model_sha256": manifest["checkpoint_sha256"],
            "max_teacher_call_rate": MAX_TEACHER_CALL_RATE,
        }
    selected = max(
        candidates,
        key=lambda row: (
            row["hybrid_r1_hits"], -row["observed_teacher_call_rate"],
            -row["min_top_confidence"], -row["min_expected_grade_gap"],
        ),
    )
    return {
        "schema_version": BALANCED_POLICY_SCHEMA,
        "status": (
            "balanced_candidate" if selected["hybrid_r1_hits"] >= student_hits
            and selected["teacher_gain_recovery"] >= 0.80
            else "no_balanced_policy_keep_student_and_teacher_modes"
        ),
        "frozen_on_split": "validation",
        "student_model_sha256": manifest["checkpoint_sha256"],
        "checkpoint_tree_sha256_verified": checkpoint_tree_sha256(model_dir),
        "min_top_confidence": selected["min_top_confidence"],
        "min_expected_grade_gap": selected["min_expected_grade_gap"],
        "max_teacher_call_rate": MAX_TEACHER_CALL_RATE,
        "observed_teacher_call_rate": selected["observed_teacher_call_rate"],
        "validation_queries": len(observations),
        "base_r1_hits": base_hits,
        "student_r1_hits": student_hits,
        "hybrid_r1_hits": selected["hybrid_r1_hits"],
        "teacher_r1_hits": teacher_hits,
        "teacher_gain_recovery": selected["teacher_gain_recovery"],
        "blind_labels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    args = parser.parse_args()
    report = calibrate(args.model, args.dataset, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
