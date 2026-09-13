#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adjudicate mined Train candidates with the same independent review as gold.

Mining surfaces the chunks the production reranker actually puts above the gold
— the highest-value negatives there are.  It also surfaces chunks that beat the
gold *because they answer the question too*, which is why nothing mined may
become a negative on its own.  V1's be02 and V2-A's handbook table-of-contents
both looked like obvious negatives to a human reading quickly and both turned
out to answer the question.

Each unique (anchor, chunk) candidate therefore goes through the shared
independent passes in :mod:`rag_gold_review` and lands in exactly one bucket:

``does_not_cover``      -> ``hard_negative``            (eligible for training)
``partially_covers``    -> ``grade1_partial`` / ``grade2_partial`` (excluded)
``fully_covers``        -> ``equivalent_grade3_candidate``        (excluded,
                            and flagged: the qrels may be missing a gold)

Output is the verdict overlay that ``mine_colloquial_train_false_winners.py
report`` already consumes, so the difficulty report picks it up unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from mine_colloquial_train_false_winners import (  # noqa: E402
    MINING_SCHEMA_VERSION,
    _load_train_positives,
    _select_candidates,
    train_cases_fingerprint,
)
from rag_gold_review import review_batches  # noqa: E402


SCHEMA_VERSION = "colloquial-mined-negative-adjudication-v1"
DEFAULT_TRACES = ROOT / "docs/rag_eval/colloquial/v2a_train_top20_traces_20260826.json"
DEFAULT_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_train_dev_draft.json"
DEFAULT_OUTPUT = (
    ROOT / "docs/rag_eval/colloquial/v2a_train_mined_negative_adjudication_20260826.json"
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def collect_candidates(traces: dict[str, Any],
                       case_by_id: dict[str, dict[str, Any]],
                       documents: dict[str, str]) -> list[dict[str, Any]]:
    """One review row per unique (anchor, chunk), keeping the strongest hit."""

    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in traces["rows"]:
        case = case_by_id[row["query_id"]]
        for candidate in _select_candidates(row, case):
            key = (row["anchor_id"], candidate["chunk_id"])
            margin = candidate["margin_vs_gold"]
            score = (candidate["role"] == "current_false_winner",
                     margin if margin is not None else -99.0)
            existing = best.get(key)
            if existing is not None and existing["_score"] >= score:
                continue
            best[key] = {
                "_score": score,
                "case_id": f"{row['anchor_id']}::{candidate['chunk_id']}",
                "anchor_id": row["anchor_id"],
                "chunk_id": candidate["chunk_id"],
                "source": candidate["source"],
                "heading_path": candidate["heading_path"],
                "role": candidate["role"],
                "relation_to_gold": candidate["relation_to_gold"],
                "reranked_rank": candidate["reranked_rank"],
                "margin_vs_gold": margin,
                "observed_in_query_id": row["query_id"],
                "question": case["question"],
                "answer_requirement": case["answer_requirements"][0],
                "document": documents.get(candidate["chunk_id"], ""),
            }
    rows = [
        {key: value for key, value in row.items() if key != "_score"}
        for _key, row in sorted(best.items())
    ]
    missing = [row["chunk_id"] for row in rows if not row["document"]]
    if missing:
        raise SystemExit(f"candidates missing from the frozen index: {missing[:5]}")
    return rows


def classify(review: dict[str, Any], relation: str) -> dict[str, str]:
    verification = review["verification"]
    coverage = review["coverage"]
    verdict = str(coverage.get("verdict") or "")
    supported = (verification.get("directly_answerable") is True
                 and verification.get("requirement_fully_supported") is True)
    covered = len(coverage.get("covered_points") or [])
    reason = str(verification.get("reason") or "").strip()
    if verdict == "does_not_cover" and not supported:
        status = "hard_negative"
        note = "独立复核判定该块不覆盖答案要求的任何要点，可作训练负例。"
    elif verdict == "fully_covers" and supported:
        status = "equivalent_grade3_candidate"
        note = ("独立复核判定该块可完整回答问题——它不是负例，而是可能缺标的等价金标，"
                "已排除出训练并留待 qrels 复核。")
    else:
        status = "grade2_partial" if covered >= 2 else "grade1_partial"
        note = "独立复核判定该块只覆盖部分答案要求，属部分支持，不得作为负例。"
    if relation == "cross_document_near_topic" and status == "hard_negative":
        note += "（跨文档近主题）"
    return {"status": status, "reason": f"{note} 复核理由：{reason}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--limit", type=int, default=0,
                        help="review only the first N candidates (smoke runs)")
    parser.add_argument(
        "--role", action="append", default=None,
        help=("restrict review to these mining roles; defaults to "
              "current_false_winner because that is the pair the guide calls "
              "highest value. Unreviewed candidates are simply excluded."),
    )
    args = parser.parse_args(argv)

    import chromadb
    from rag_tools import get_collection_name

    traces_path = Path(args.traces).expanduser().resolve()
    cases_path = Path(args.cases).expanduser().resolve()
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    if traces.get("schema_version") != MINING_SCHEMA_VERSION:
        raise SystemExit("unsupported mining artifact")
    _, positives = _load_train_positives(cases_path)
    expected = traces.get("train_cases_sha256")
    if expected is not None and expected != train_cases_fingerprint(positives):
        raise SystemExit("mining traces were built from different Train cases")
    case_by_id = {item["query_id"]: item for item in positives}

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    chunk_ids = sorted({
        candidate["chunk_id"]
        for row in traces["rows"]
        for candidate in _select_candidates(row, case_by_id[row["query_id"]])
    })
    snapshot = collection.get(ids=chunk_ids, include=["documents"])
    documents = {
        str(chunk_id): str(document or "")
        for chunk_id, document in zip(snapshot.get("ids") or [],
                                      snapshot.get("documents") or [])
    }
    rows = collect_candidates(traces, case_by_id, documents)
    roles = set(args.role or ["current_false_winner"])
    skipped = [row for row in rows if row["role"] not in roles]
    rows = [row for row in rows if row["role"] in roles]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[adjudicate] {len(rows)} unique (anchor, chunk) candidates", flush=True)

    output_path = Path(args.output).expanduser().resolve()
    reviews = review_batches(rows, batch_size=args.batch_size,
                             max_tokens=args.max_tokens)
    verdicts: list[dict[str, Any]] = []
    for row in rows:
        review = reviews[row["case_id"]]
        decision = classify(review, row["relation_to_gold"])
        verdicts.append({
            "anchor_id": row["anchor_id"],
            "chunk_id": row["chunk_id"],
            "status": decision["status"],
            "reason": decision["reason"],
            "role": row["role"],
            "relation_to_gold": row["relation_to_gold"],
            "reranked_rank": row["reranked_rank"],
            "margin_vs_gold": row["margin_vs_gold"],
            "observed_in_query_id": row["observed_in_query_id"],
            "verification": review["verification"],
            "coverage": review["coverage"],
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "independent_entailment_plus_adversarial_coverage_review",
        "traces": traces_path.name,
        "traces_sha256": _sha256_file(traces_path),
        "cases": cases_path.name,
        "train_cases_sha256": train_cases_fingerprint(positives),
        "summary": {
            "reviewed_roles": sorted(roles),
            "skipped_unreviewed_candidates": len(skipped),
            "skipped_by_role": dict(sorted(Counter(
                row["role"] for row in skipped
            ).items())),
            "candidates": len(verdicts),
            "by_status": dict(sorted(Counter(
                row["status"] for row in verdicts
            ).items())),
            "by_role": dict(sorted(Counter(row["role"] for row in verdicts).items())),
            "eligible_as_negative": sum(
                row["status"] == "hard_negative" for row in verdicts
            ),
            "flagged_missing_gold": sorted(
                f"{row['anchor_id']}::{row['chunk_id']}" for row in verdicts
                if row["status"] == "equivalent_grade3_candidate"
            ),
        },
        "verdicts": verdicts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **payload["summary"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
