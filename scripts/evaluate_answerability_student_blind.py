#!/usr/bin/env python3
"""Independent blind evaluator; never imported by the trainer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability_student import checkpoint_tree_sha256
from rag_answerability_student_data import StudentDataError, read_jsonl


def _load_model(model_dir: Path, device_requested: str | None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    manifest = json.loads(
        (model_dir / "student_model_manifest.json").read_text(encoding="utf-8")
    )
    actual = checkpoint_tree_sha256(model_dir)
    if manifest.get("checkpoint_sha256") != actual:
        raise StudentDataError("checkpoint changed after validation freeze")
    if manifest.get("dataset", {}).get("blind_rows_consumed") != 0:
        raise StudentDataError("selected checkpoint manifest reports blind consumption")
    if device_requested:
        device = device_requested
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device, manifest


def _predict_group(tokenizer, model, device, group, torch_module):
    encoded = tokenizer(
        [row["question"] for row in group],
        [row["chunk_text"] for row in group],
        padding=True, truncation=True, max_length=384, return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch_module.inference_mode():
        probabilities = torch_module.softmax(
            model(**encoded).logits.float(), dim=-1,
        ).cpu().tolist()
    predictions = []
    for row, values in zip(group, probabilities):
        predictions.append({
            "row": row,
            "grade": max(range(4), key=lambda index: values[index]),
            "confidence": max(float(value) for value in values),
            "score": sum(index * float(value) for index, value in enumerate(values)),
        })
    return predictions


def _dcg(grades: list[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _query_outcome(group, predictions):
    def numeric_or(value, fallback):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

    base = sorted(
        group,
        key=lambda row: (
            -numeric_or(row.get("base_student_score"), float("-inf")),
            row["current_bge_rank"],
        ),
    )
    prediction_by_chunk = {item["row"]["chunk_id"]: item for item in predictions}
    student = sorted(
        group,
        key=lambda row: (
            -prediction_by_chunk[row["chunk_id"]]["grade"],
            -numeric_or(row.get("current_bge_score"), -row["current_bge_rank"]),
        ),
    )
    has_answer = any(row["teacher_grade"] == 3 for row in group)
    values = {"has_answer": has_answer, "style": group[0]["query_style"],
              "split": group[0]["split"]}
    if has_answer:
        for label, ranked in (("base", base), ("student", student)):
            values[f"{label}_r1"] = int(ranked[0]["teacher_grade"] == 3)
            values[f"{label}_r3"] = int(any(row["teacher_grade"] == 3 for row in ranked[:3]))
            values[f"{label}_r5"] = int(any(row["teacher_grade"] == 3 for row in ranked[:5]))
            grades = [int(row["teacher_grade"]) for row in ranked[:5]]
            ideal = sorted((int(row["teacher_grade"]) for row in group), reverse=True)[:5]
            values[f"{label}_ndcg5"] = _dcg(grades) / _dcg(ideal) if _dcg(ideal) else 0.0
    else:
        top_prediction = prediction_by_chunk[student[0]["chunk_id"]]
        values["student_false_accept"] = int(top_prediction["grade"] == 3)
        baseline_values = {
            row.get("baseline_gate_accept") for row in group
            if row.get("baseline_gate_accept") is not None
        }
        values["base_false_accept"] = int(True in baseline_values) if baseline_values else None
    return values


def _bootstrap_lower(outcomes, *, samples: int = 10000, seed: int = 20260828):
    diffs = [row["student_r1"] - row["base_r1"] for row in outcomes]
    if not diffs:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    means.sort()
    return means[int(samples * 0.05)]


def evaluate(model_dir: Path, blind_path: Path, *, device: str | None = None):
    rows = read_jsonl(blind_path)
    if any(row["split"] not in {"blind_a", "blind_b"} for row in rows):
        raise StudentDataError("blind evaluator accepts blind_a/blind_b rows only")
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    import torch
    tokenizer, model, actual_device, manifest = _load_model(model_dir, device)
    outcomes = []
    latencies = []
    for query_id in sorted(by_query):
        group = by_query[query_id]
        started = time.perf_counter()
        predictions = _predict_group(tokenizer, model, actual_device, group, torch)
        if actual_device == "cuda":
            torch.cuda.synchronize()
        elif actual_device == "mps":
            torch.mps.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        outcomes.append(_query_outcome(group, predictions))
    positive = [row for row in outcomes if row["has_answer"]]
    negative = [row for row in outcomes if not row["has_answer"]]

    def total(field):
        return sum(float(row[field]) for row in positive)

    wins = sum(row["student_r1"] > row["base_r1"] for row in positive)
    losses = sum(row["student_r1"] < row["base_r1"] for row in positive)
    split_net = {
        split: sum(
            row["student_r1"] - row["base_r1"]
            for row in positive if row["split"] == split
        ) for split in ("blind_a", "blind_b")
    }
    style_net = {
        style: sum(
            row["student_r1"] - row["base_r1"]
            for row in positive if row["style"] == style
        ) for style in ("natural", "implicit_oral", "long_noisy")
    }
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)]
    base_false = [row["base_false_accept"] for row in negative if row["base_false_accept"] is not None]
    checks = {
        "positive_n_gte_120": len(positive) >= 120,
        "r1_gain_gte_0_05": (
            (total("student_r1") - total("base_r1")) / len(positive)
            if positive else -1.0
        ) >= 0.05,
        "bootstrap_one_sided_95_lower_gt_0": (_bootstrap_lower(positive) or 0.0) > 0,
        "blind_a_wins_gt_losses": split_net["blind_a"] > 0,
        "blind_b_wins_gt_losses": split_net["blind_b"] > 0,
        "r3_regression_lte_1": total("student_r3") >= total("base_r3") - 1,
        "r5_regression_lte_1": total("student_r5") >= total("base_r5") - 1,
        "ndcg5_not_lower": total("student_ndcg5") >= total("base_ndcg5"),
        "oral_styles_nonnegative": all(value >= 0 for value in style_net.values()),
        "negative_guard_available": len(base_false) == len(negative) and bool(negative),
        "negative_false_accept_not_worse": (
            sum(row["student_false_accept"] for row in negative) <= sum(base_false)
            if len(base_false) == len(negative) and negative else False
        ),
        "mac_student_p95_lte_750ms": actual_device == "mps" and p95 <= 750.0,
    }
    return {
        "schema_version": "answerability-student-blind-result-v1",
        "status": "blind_go" if all(checks.values()) else "blind_no_go_or_incomplete",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "blind_artifact_sha256": __import__("hashlib").sha256(blind_path.read_bytes()).hexdigest(),
        "checks": checks,
        "positive_n": len(positive), "negative_n": len(negative),
        "base_r1": total("base_r1") / len(positive) if positive else 0.0,
        "student_r1": total("student_r1") / len(positive) if positive else 0.0,
        "wins": wins, "losses": losses, "split_net_wins": split_net,
        "style_net_wins": style_net,
        "bootstrap_r1_gain_lower_95_one_sided": _bootstrap_lower(positive),
        "base_false_accepts": sum(base_false) if base_false else None,
        "student_false_accepts": sum(row["student_false_accept"] for row in negative),
        "device": actual_device, "student_query_p95_ms": p95,
        "blind_rows_returned_to_trainer": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--blind", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    args = parser.parse_args()
    report = evaluate(args.model, args.blind, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
