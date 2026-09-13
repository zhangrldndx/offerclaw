#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-rank frozen pools by answer-containment instead of topical similarity.

The whole of 2026-08-27 converged on one diagnosis: ``bge-reranker-base``
scores topical relatedness, and every intervention that accepted that score as
the truth -- retraining it, deepening its candidates, re-encoding the query,
breaking its ties, borrowing its confidence as gate evidence -- moved end-to-end
R@1 by zero or made abstention worse.  This scores the same frozen pools with
a judge that is asked the other question, and measures both sides.

Ordering rule, kept as small as it can be: sort by the answerability grade,
break ties with the existing reranker score.  Nothing is tuned; there is no
threshold to fit and therefore nothing to overfit to the development set --
which matters, because fitting a threshold on Dev-New is exactly how the
evidence-gate second path produced a result that held-out sets then destroyed.

Only the top ``--depth`` candidates are judged.  On the Dev-New failures the
gold sits at reranked rank 2-6 in 19 of 20 cases, so depth 6 covers the
population at a fraction of the calls, and rows whose gold is deeper are
reported rather than silently counted as unreachable.
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


def select(rows: list[dict[str, Any]], population: str,
           limit: int | None) -> list[tuple[dict[str, Any], list, set, bool]]:
    chosen = []
    for row in rows:
        pool = row["retrieval_trace"]["reranked_candidates"]
        gold = set(row["all_relevant_chunk_ids"])
        if not pool or not (gold & {item["chunk_id"] for item in pool}):
            continue                     # the reranker never had a chance
        was_correct = pool[0]["chunk_id"] in gold
        if population == "failures" and was_correct:
            continue
        if population == "correct" and not was_correct:
            continue
        chosen.append((row, pool, gold, was_correct))
    return chosen[:limit] if limit else chosen


def probe(eval_path: Path, cases_path: Path, *, depth: int, population: str,
          limit: int | None, workers: int, model: str | None) -> dict[str, Any]:
    import chromadb
    from rag_answerability import flush_cache, grade
    from rag_tools import get_collection_name, index_fingerprint

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    cases = {item["query_id"]: item
             for item in json.loads(cases_path.read_text(encoding="utf-8"))["items"]}
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    stored = collection.get(include=["documents"])
    text = dict(zip(stored["ids"], stored["documents"]))

    chosen = select(payload["runs"][0]["positive"]["rows"], population, limit)
    print(f"{len(chosen)} queries x depth {depth}", file=sys.stderr)

    jobs = []
    for row, pool, _gold, _was in chosen:
        question = cases[row["query_id"]]["question"]
        for item in pool[:depth]:
            jobs.append((row["query_id"], question, item["chunk_id"]))

    graded: dict[tuple[str, str], dict[str, Any] | None] = {}
    done = 0

    def work(job):
        query_id, question, chunk_id = job
        return (query_id, chunk_id), grade(question, text.get(chunk_id, ""),
                                           model=model)

    with ThreadPoolExecutor(max_workers=workers) as pool_executor:
        for key, value in pool_executor.map(work, jobs):
            graded[key] = value
            done += 1
            if done % 20 == 0:
                print(f"  graded {done}/{len(jobs)}", file=sys.stderr, flush=True)
    flush_cache()

    findings, unavailable = [], 0
    for row, pool, gold, was_correct in chosen:
        head = pool[:depth]
        scored = []
        for item in head:
            verdict = graded.get((row["query_id"], item["chunk_id"]))
            if verdict is None:
                unavailable += 1
            scored.append({
                "chunk_id": item["chunk_id"],
                # A judge failure must not read as "grade 0"; -1 sorts it below
                # every real grade and is counted separately.
                "grade": verdict["grade"] if verdict else -1,
                "reason": verdict["reason"] if verdict else "",
                "rerank_score": item.get("rerank_score") or 0.0,
                "is_gold": item["chunk_id"] in gold,
            })
        ordered = sorted(scored, key=lambda entry: (-entry["grade"],
                                                    -entry["rerank_score"]))
        now_correct = bool(ordered) and ordered[0]["is_gold"]
        gold_entry = next((e for e in scored if e["is_gold"]), None)
        winner_entry = scored[0] if scored else None
        findings.append({
            "query_id": row["query_id"],
            "style": row.get("query_style", "?"),
            "was_correct": was_correct,
            "now_correct": now_correct,
            "outcome": ("rescued" if now_correct and not was_correct else
                        "broken" if was_correct and not now_correct else
                        "unchanged"),
            "gold_in_depth": gold_entry is not None,
            "gold_grade": gold_entry["grade"] if gold_entry else None,
            "old_top1_grade": winner_entry["grade"] if winner_entry else None,
            "new_top1": ordered[0]["chunk_id"] if ordered else None,
            "graded": scored,
        })

    outcomes = Counter(item["outcome"] for item in findings)
    gold_grades = Counter(item["gold_grade"] for item in findings
                          if item["gold_in_depth"])
    top1_grades = Counter(item["old_top1_grade"] for item in findings)
    return {
        "schema_version": "colloquial-answerability-rerank-v1",
        "eval": eval_path.name,
        "population": population,
        "depth": depth,
        "model": model or "",
        "index": index_fingerprint(collection=collection),
        "queries": len(findings),
        "llm_calls": len(jobs),
        "judge_unavailable": unavailable,
        "outcomes": dict(outcomes),
        "gold_grade_distribution": {str(k): v for k, v in sorted(gold_grades.items())},
        "reranker_top1_grade_distribution": {str(k): v for k, v
                                             in sorted(top1_grades.items())},
        "rescued": sorted(item["query_id"] for item in findings
                          if item["outcome"] == "rescued"),
        "broken": sorted(item["query_id"] for item in findings
                         if item["outcome"] == "broken"),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--population", choices=("failures", "correct", "all"),
                        default="failures")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = probe(Path(args.eval).expanduser().resolve(),
                   Path(args.cases).expanduser().resolve(),
                   depth=args.depth, population=args.population,
                   limit=args.limit, workers=args.workers, model=args.model)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("population", "queries", "llm_calls", "judge_unavailable",
                       "outcomes", "gold_grade_distribution",
                       "reranker_top1_grade_distribution", "rescued", "broken")},
                     ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
