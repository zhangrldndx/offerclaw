#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does scoring sentence windows instead of whole chunks recover the failures?

The diagnosis this tests, from ``audit_reranker_headroom`` and the synthesis:
the cross-encoder is scoring *topical relatedness*, not *answer containment*.
Its evidence is that 80% of the reranker's losses are neighbouring sections of
the correct document, that the winning chunk covers the question's answer
requirement at 0.07-0.44 against the gold's 0.77-0.93, and that the top scores
sit above 0.998 where the sigmoid has no resolution left.

If that diagnosis is right, there is a cheap consequence worth testing before
anything expensive: when an answer occupies one or two sentences of a long
chunk, scoring the *whole* chunk averages that signal away, while a topically
identical neighbour scores just as high on the shared vocabulary.  Scoring
short windows and taking the maximum should therefore separate them -- using
the same local model, no new dependency, no LLM.

Deliberately diagnostic-first: this reranks the frozen pools of queries that
already failed, which is cheap.  Only if those move is it worth measuring the
harm side (currently-correct queries and the abstention guard), because a
rescoring that helps nothing cannot be worth what it costs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Chinese sentence enders plus newlines; markdown chunks break on both.
_SPLIT = re.compile(r"(?<=[。！？；\n])")
MIN_WINDOW_CHARS = 40


def sentence_windows(text: str, *, window: int, stride: int) -> list[str]:
    """Overlapping sentence windows, with the whole chunk always included.

    Keeping the full chunk in the candidate set means the score can only go up:
    the window maximum is bounded below by the original whole-chunk score, so a
    regression can never come from having *looked* at windows -- only from a
    window scoring higher than the chunk that deserved to win.
    """
    parts = [part.strip() for part in _SPLIT.split(text or "") if part.strip()]
    if not parts:
        return [text or ""]
    windows = []
    for start in range(0, max(1, len(parts) - window + 1), stride):
        joined = "".join(parts[start:start + window])
        if len(joined) >= MIN_WINDOW_CHARS:
            windows.append(joined)
    if not windows:
        windows = ["".join(parts)]
    if text and text not in windows:
        windows.append(text)
    return windows


def probe(eval_path: Path, cases_path: Path, *, window: int, stride: int,
          population: str, limit: int | None) -> dict[str, Any]:
    import chromadb
    from rag_rerank import _load_reranker
    from rag_tools import get_collection_name, index_fingerprint

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    cases = {item["query_id"]: item
             for item in json.loads(cases_path.read_text(encoding="utf-8"))["items"]}
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    stored = collection.get(include=["documents"])
    text = {chunk_id: document
            for chunk_id, document in zip(stored["ids"], stored["documents"])}
    model = _load_reranker()

    rows = payload["runs"][0]["positive"]["rows"]
    selected = []
    for row in rows:
        pool = row["retrieval_trace"]["reranked_candidates"]
        gold = set(row["all_relevant_chunk_ids"])
        if not pool or not (gold & {item["chunk_id"] for item in pool}):
            continue                      # the reranker never had a chance
        was_correct = pool[0]["chunk_id"] in gold
        if population == "failures" and was_correct:
            continue
        if population == "correct" and not was_correct:
            continue
        selected.append((row, pool, gold, was_correct))
    if limit:
        selected = selected[:limit]

    findings = []
    for position, (row, pool, gold, was_correct) in enumerate(selected, start=1):
        question = cases[row["query_id"]]["question"]
        print(f"[span] {position}/{len(selected)} {row['query_id']}",
              file=sys.stderr, flush=True)
        rescored = []
        for item in pool:
            body = text.get(item["chunk_id"], "")
            windows = sentence_windows(body, window=window, stride=stride)
            scores = model.predict([(question, w) for w in windows])
            rescored.append((item["chunk_id"], float(max(scores)),
                             item.get("rerank_score"), len(windows)))
        rescored.sort(key=lambda entry: -entry[1])
        now_correct = rescored[0][0] in gold
        findings.append({
            "query_id": row["query_id"],
            "style": row.get("query_style", "?"),
            "was_correct": was_correct,
            "now_correct": now_correct,
            "outcome": ("rescued" if now_correct and not was_correct else
                        "broken" if was_correct and not now_correct else
                        "unchanged"),
            "span_winner": rescored[0][0],
            "span_top_score": round(rescored[0][1], 6),
            "windows_scored": sum(entry[3] for entry in rescored),
        })

    outcomes = Counter(item["outcome"] for item in findings)
    return {
        "schema_version": "colloquial-span-rerank-probe-v1",
        "eval": eval_path.name,
        "population": population,
        "window_sentences": window,
        "stride": stride,
        "index": index_fingerprint(collection=collection),
        "queries": len(findings),
        "outcomes": dict(outcomes),
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
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = probe(Path(args.eval).expanduser().resolve(),
                   Path(args.cases).expanduser().resolve(),
                   window=args.window, stride=args.stride,
                   population=args.population, limit=args.limit)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("population", "queries", "outcomes", "rescued", "broken")},
                     ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
