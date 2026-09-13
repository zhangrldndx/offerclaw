#!/usr/bin/env python3
"""Attach frozen pre-train MiniLM scores to a private candidate catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability_student import DEFAULT_MODEL_NAME
from scripts.build_answerability_student_dataset import _load_catalog, _score_base_student


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    queries = _load_catalog(args.catalog)
    scored = _score_base_student(queries, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queries),
        encoding="utf-8",
    )
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass
    print(json.dumps({
        "schema_version": "answerability-student-base-score-v1",
        "rows_scored": scored,
        "queries": len(queries),
        "candidates": sum(len(row["candidates"]) for row in queries),
        "base_student_model": args.model,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
