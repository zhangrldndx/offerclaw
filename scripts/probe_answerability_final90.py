#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out confirmation of answerability reranking on the user-authored Final set.

Separate from ``probe_answerability_rerank`` because ``final90`` is judged at
*source* level (``expect_sources``) rather than against chunk-level qrels, and
folding two judging contracts into one script is how a comparison quietly
starts measuring the wrong thing.

Measurement contract: both arms are scored on ``reranked_candidate_ids[0]``.
That is deliberately *not* the harness's own ``r1_hits``, which is computed on
``final_chunk_ids`` -- the post-gate evidence list, from which the gate's
distance cutoff can drop the reranker's winner.  The two definitions differ on
2 of 90 rows here.  Borrowing the harness number for the baseline and computing
the candidate a different way would put that discrepancy straight into the
delta, so this reports its own baseline (71/90) and compares like with like.

This set took no part in designing the judge: no threshold was fitted on it,
and its questions were authored independently.  That matters because the same
day's evidence-gate change looked clean on its development set and was then
destroyed by held-out negatives -- the difference being that the gate carried a
threshold fitted to n=19, while the rule here ("sort by grade, break ties with
the existing reranker score") has nothing to fit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def probe(eval_path: Path, *, set_name: str, depth: int, workers: int,
          limit: int | None, model: str | None) -> dict[str, Any]:
    import chromadb
    from rag_answerability import flush_cache, grade
    from rag_tools import get_collection_name, index_fingerprint

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    entry = next(item for item in payload["sets"] if item["set"] == set_name)
    rows = entry["runs"][0]["rows"]
    if limit:
        rows = rows[:limit]

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    stored = collection.get(include=["documents", "metadatas"])
    text = dict(zip(stored["ids"], stored["documents"]))
    source = {chunk_id: (meta or {}).get("source", "")
              for chunk_id, meta in zip(stored["ids"], stored["metadatas"])}

    def in_expect(chunk_id: str, expect: list[str]) -> bool:
        # ``expect_sources`` are stems; stored sources carry the extension.
        name = source.get(chunk_id, "")
        return any(stem and (name == stem or name.startswith(stem))
                   for stem in expect)

    jobs = []
    for row in rows:
        for chunk_id in (row.get("reranked_candidate_ids") or [])[:depth]:
            jobs.append((row["id"], row["question"], chunk_id))

    graded: dict[tuple[str, str], dict[str, Any] | None] = {}
    done = 0

    def work(job):
        row_id, question, chunk_id = job
        return (row_id, chunk_id), grade(question, text.get(chunk_id, ""),
                                         model=model)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for key, value in executor.map(work, jobs):
            graded[key] = value
            done += 1
            if done % 40 == 0:
                print(f"  graded {done}/{len(jobs)}", file=sys.stderr, flush=True)
    flush_cache()

    findings, unavailable = [], 0
    for row in rows:
        expect = row.get("expect_sources") or []
        candidates = (row.get("reranked_candidate_ids") or [])[:depth]
        scores = (row.get("reranked_scores") or [])[:depth]
        was_correct = bool(candidates) and in_expect(candidates[0], expect)
        scored = []
        for index, chunk_id in enumerate(candidates):
            verdict = graded.get((row["id"], chunk_id))
            if verdict is None:
                unavailable += 1
            scored.append({
                "chunk_id": chunk_id,
                "grade": verdict["grade"] if verdict else -1,
                "rerank_score": float(scores[index]) if index < len(scores) else 0.0,
                "expected": in_expect(chunk_id, expect),
            })
        ordered = sorted(scored, key=lambda e: (-e["grade"], -e["rerank_score"]))
        now_correct = bool(ordered) and ordered[0]["expected"]
        findings.append({
            "id": row["id"],
            "was_correct": was_correct,
            "now_correct": now_correct,
            "outcome": ("rescued" if now_correct and not was_correct else
                        "broken" if was_correct and not now_correct else
                        "unchanged"),
            "expected_grades": [e["grade"] for e in scored if e["expected"]],
            "old_top1_grade": scored[0]["grade"] if scored else None,
        })

    outcomes = Counter(item["outcome"] for item in findings)
    before = sum(1 for item in findings if item["was_correct"])
    after = sum(1 for item in findings if item["now_correct"])
    return {
        "schema_version": "colloquial-answerability-final90-v1",
        "metric": ("top1 of reranked_candidate_ids, both arms; differs from the "
                   "harness r1_hits, which reads the post-gate final_chunk_ids"),
        "set": set_name,
        "eval": eval_path.name,
        "depth": depth,
        "model": model or "",
        "index": index_fingerprint(collection=collection),
        "queries": len(findings),
        "llm_calls": len(jobs),
        "judge_unavailable": unavailable,
        "r1_before": before,
        "r1_after": after,
        "outcomes": dict(outcomes),
        "rescued": sorted(i["id"] for i in findings if i["outcome"] == "rescued"),
        "broken": sorted(i["id"] for i in findings if i["outcome"] == "broken"),
        "expected_grade_distribution": {
            str(g): c for g, c in sorted(Counter(
                g for item in findings for g in item["expected_grades"]).items())},
        "old_top1_grade_distribution": {
            str(g): c for g, c in sorted(Counter(
                item["old_top1_grade"] for item in findings).items())},
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--set", dest="set_name", default="final90")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = probe(Path(args.eval).expanduser().resolve(), set_name=args.set_name,
                   depth=args.depth, workers=args.workers, limit=args.limit,
                   model=args.model)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("set", "queries", "llm_calls", "judge_unavailable",
                       "r1_before", "r1_after", "outcomes",
                       "expected_grade_distribution",
                       "old_top1_grade_distribution", "rescued", "broken")},
                     ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
