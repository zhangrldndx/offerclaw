#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How far is the gold from winning, on the questions the reranker gets wrong?

Three separate rounds this month improved the *candidate pool* and moved
end-to-end R@1 by zero: F3 (reranker fine-tune, 0 wins / 0 losses), pool28
(candidates 67->70, +1/-1), and the query-side LoRA (dense R@1 +4, +1/-1).
Across five end-to-end configurations the R@1 hit sets share a 46-question
common core, which says the reranker's Top-1 choice is close to a fixed point.

Before spending anything else on giving that reranker better candidates, this
asks the prior question: on the queries where the gold *is* in the pool and
still loses, how much would it have to move to win?

The distinction that matters:

``knife_edge``  the gold is within a hair of the winner.  Reachable in
                principle -- though F3 showed that "reachable" and "learnable
                from 72 examples" are different claims.
``far``         the reranker is confidently wrong.  No amount of reordering the
                pool or nudging the scores fixes this; it needs a different
                model or a different scoring input.

Reads frozen evaluation artifacts only -- ``reranked_candidates`` already
carries every pool member's score, so nothing is recomputed and nothing can
drift between this audit and the run it describes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# A cross-encoder emitting sigmoid scores: 0.05 is roughly the width of the
# band inside which the earlier per-case review found the reranker was
# effectively picking arbitrarily among near duplicates.
KNIFE_EDGE = 0.05
CLOSE = 0.30


def classify(gap: float) -> str:
    if gap <= KNIFE_EDGE:
        return "knife_edge"
    if gap <= CLOSE:
        return "close"
    return "far"


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["runs"][0]["positive"]["rows"]
    findings: list[dict[str, Any]] = []
    reachable_but_lost = 0
    for row in rows:
        gold = set(row["all_relevant_chunk_ids"])
        pool = row["retrieval_trace"]["reranked_candidates"]
        if not pool:
            continue
        scored = [(item["chunk_id"], item.get("rerank_score")) for item in pool]
        in_pool = [(chunk_id, score) for chunk_id, score in scored
                   if chunk_id in gold and score is not None]
        if not in_pool:
            continue          # the reranker never had a chance; not its failure
        winner_id, winner_score = scored[0]
        if winner_id in gold:
            continue          # already correct
        reachable_but_lost += 1
        gold_id, gold_score = max(in_pool, key=lambda pair: pair[1])
        gold_rank = next(index for index, (chunk_id, _s)
                         in enumerate(scored, start=1) if chunk_id == gold_id)
        gap = float(winner_score) - float(gold_score)
        findings.append({
            "query_id": row["query_id"],
            "style": row.get("query_style", "?"),
            "gold_rerank_rank": gold_rank,
            "gold_score": round(float(gold_score), 6),
            "winner_score": round(float(winner_score), 6),
            "winner_chunk_id": winner_id,
            "gold_chunk_id": gold_id,
            "gap": round(gap, 6),
            "band": classify(gap),
        })

    bands = Counter(item["band"] for item in findings)
    by_style: dict[str, Counter] = defaultdict(Counter)
    for item in findings:
        by_style[item["style"]][item["band"]] += 1
    gaps = [item["gap"] for item in findings]
    return {
        "schema_version": "colloquial-reranker-headroom-v1",
        "arm": payload["configuration"]["retrieval_arm"],
        "queries": len(rows),
        "gold_in_pool_but_lost": reachable_but_lost,
        "bands": dict(bands),
        "band_thresholds": {"knife_edge": KNIFE_EDGE, "close": CLOSE},
        "gap": {
            "min": round(min(gaps), 6) if gaps else None,
            "median": round(statistics.median(gaps), 6) if gaps else None,
            "max": round(max(gaps), 6) if gaps else None,
        },
        "by_style": {style: dict(counts) for style, counts in sorted(by_style.items())},
        "findings": sorted(findings, key=lambda item: item["gap"]),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"金标在池内却没拿到 Top1：**{report['gold_in_pool_but_lost']}** 道"
        f"（共 {report['queries']} 道正例）",
        "",
        "| 档位 | 判据 | 题数 |",
        "|---|---|---:|",
    ]
    labels = {"knife_edge": f"gap ≤ {KNIFE_EDGE}", "close": f"≤ {CLOSE}",
              "far": f"> {CLOSE}"}
    for band in ("knife_edge", "close", "far"):
        lines.append(f"| `{band}` | {labels[band]} | "
                     f"{report['bands'].get(band, 0)} |")
    gap = report["gap"]
    lines += ["", f"gap 分布：min {gap['min']} / 中位 {gap['median']} / max {gap['max']}",
              "", "| query | 风格 | 金标精排名次 | 金标分 | 冠军分 | gap | 档位 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for item in report["findings"]:
        lines.append(
            f"| `{item['query_id']}` | {item['style']} | {item['gold_rerank_rank']} "
            f"| {item['gold_score']:.4f} | {item['winner_score']:.4f} "
            f"| {item['gap']:.4f} | {item['band']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    reports = {}
    for raw in args.eval:
        path = Path(raw).expanduser().resolve()
        report = audit(json.loads(path.read_text(encoding="utf-8")))
        reports[path.stem] = report
        print(f"\n### {path.stem}\n")
        print(render(report))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": "colloquial-reranker-headroom-v1",
                    "datasets": reports}, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
