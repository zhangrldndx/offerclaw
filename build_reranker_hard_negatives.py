#!/usr/bin/env python3
"""Build the gated, development-only Stage-B reranker triples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_hard_negatives import ROOT, build_from_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qrels",
        type=Path,
        default=ROOT / "docs/rag_eval/qrels/rag_bench_paraphrase_adjudicated.json",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="explicitly include questions and production-formatted corpus chunks",
    )
    parser.add_argument(
        "--reconstruct-missing-traces",
        action="store_true",
        help="read-only A0 rerun for legacy result files that lack chunk IDs",
    )
    parser.add_argument(
        "--no-stability-anchors",
        action="store_true",
        help="omit qrels-direct current Top1 anchors (enabled by default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the ID-only selection summary without writing a file",
    )
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")

    artifact = build_from_files(
        qrels_path=args.qrels,
        baseline_path=args.baseline,
        include_text=args.include_text,
        reconstruct_missing=args.reconstruct_missing_traces,
        include_stability_anchors=not args.no_stability_anchors,
    )
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": "validated_not_written" if args.dry_run else "written",
        "output": str(args.output) if args.output else "",
        "contains_text": artifact["contains_text"],
        **artifact["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
