#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay the consensus tie-break over frozen traces, before running anything.

Motivation, from ``audit_reranker_headroom``: on Dev80 every one of the eight
"gold was in the pool and lost" cases is decided by a gap of at most 0.048, and
four of them by 0.0006 or less -- with both scores above 0.998, where the
cross-encoder's sigmoid has no resolution left.  The Top-1 there is not a
judgement, it is numerical noise.  A tie-break on an *independent* signal should
therefore be able to recover some of them without touching the reranker.

The traces already carry both channel rankings and every pool member's score,
so the guard is a pure function of data we have.  Sweeping it offline costs
seconds instead of a run per cell, and -- more importantly -- it forces the
harm column to be computed on the same footing as the gain: promoting a
consensus candidate can just as easily displace a gold that was already first.

Fidelity note: this mirrors ``rag_candidate_pool.protect_multichannel_consensus``
deliberately rather than importing it, because that function operates on parallel
document/metadata lists the traces do not preserve.  ``verify`` re-derives the
baseline Top-1 from the trace and refuses to report if it disagrees with the
recorded one, so a divergence between the two implementations shows up as a
failure here rather than as a wrong recommendation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOP_KS = (3, 5, 8, 10, 20)
MARGINS = (0.005, 0.01, 0.03, 0.05, 0.10, 0.15)


def _channel_ids(candidates: list[dict[str, Any]], top_k: int) -> set[str]:
    return {item["chunk_id"] for item in candidates
            if item.get("rank") is not None and int(item["rank"]) <= top_k}


def guard_pick(row: dict[str, Any], top_k: int, margin: float) -> str | None:
    """Return the chunk id the guard would put first, or None if it abstains."""
    trace = row["retrieval_trace"]
    pool = trace["reranked_candidates"]
    if len(pool) < 2:
        return None
    scored = [(item["chunk_id"], item.get("rerank_score")) for item in pool]
    if scored[0][1] is None:
        return None
    consensus = (_channel_ids(trace["dense_candidates"], top_k)
                 & _channel_ids(trace["bm25_candidates"], top_k))
    if not consensus:
        return None
    eligible = [(chunk_id, score) for chunk_id, score in scored
                if chunk_id in consensus and score is not None]
    if not eligible:
        return None
    best_id, best_score = max(eligible, key=lambda pair: pair[1])
    if best_id == scored[0][0]:
        return None                      # consensus candidate is already first
    if float(scored[0][1]) - float(best_score) > margin:
        return None                      # not a tie; leave the reranker alone
    return best_id


def verify(rows: list[dict[str, Any]]) -> None:
    """The recorded Top-1 must equal the trace's own first reranked candidate.

    If it does not, the trace and the metric were produced by different code
    paths and every number below would be describing something else.
    """
    for row in rows:
        pool = row["retrieval_trace"]["reranked_candidates"]
        if not pool or not row.get("final_chunk_ids"):
            continue
        if pool[0]["chunk_id"] != row["final_chunk_ids"][0]:
            raise SystemExit(
                f"{row['query_id']}: trace Top-1 {pool[0]['chunk_id']} does not "
                f"match recorded {row['final_chunk_ids'][0]}"
            )


def simulate(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["runs"][0]["positive"]["rows"]
    verify(rows)
    baseline = {row["query_id"]: row["final_rank"] == 1 for row in rows}
    results = []
    for top_k in TOP_KS:
        for margin in MARGINS:
            gained, lost, fired = [], [], 0
            for row in rows:
                pick = guard_pick(row, top_k, margin)
                if pick is None:
                    continue
                fired += 1
                gold = set(row["all_relevant_chunk_ids"])
                was, now = baseline[row["query_id"]], pick in gold
                if now and not was:
                    gained.append(row["query_id"])
                elif was and not now:
                    lost.append(row["query_id"])
            results.append({
                "top_k": top_k, "margin": margin, "fired": fired,
                "gained": len(gained), "lost": len(lost),
                "net": len(gained) - len(lost),
                "gained_query_ids": gained, "lost_query_ids": lost,
            })
    best = max(results, key=lambda r: (r["net"], -r["lost"], -r["margin"]))
    return {
        "schema_version": "colloquial-consensus-simulation-v1",
        "arm": payload["configuration"]["retrieval_arm"],
        "queries": len(rows),
        "baseline_r1": sum(baseline.values()),
        "grid": results,
        "best": best,
    }


def render(name: str, report: dict[str, Any]) -> str:
    lines = [f"### {name}  （基线 R@1 {report['baseline_r1']}/{report['queries']}）", "",
             "| top_k | margin | 触发 | 救回 | 打坏 | 净 |", "|---:|---:|---:|---:|---:|---:|"]
    for row in report["grid"]:
        mark = "**" if row is report["best"] else ""
        lines.append(f"| {row['top_k']} | {row['margin']} | {row['fired']} "
                     f"| {mark}{row['gained']}{mark} | {row['lost']} "
                     f"| {mark}{row['net']:+d}{mark} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    reports = {}
    for raw in args.eval:
        path = Path(raw).expanduser().resolve()
        report = simulate(json.loads(path.read_text(encoding="utf-8")))
        reports[path.stem] = report
        print(render(path.stem, report))
        print(f"\n最优：top_k={report['best']['top_k']} margin={report['best']['margin']} "
              f"→ 救回 {report['best']['gained_query_ids']} "
              f"打坏 {report['best']['lost_query_ids']}\n")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": "colloquial-consensus-simulation-v1",
                    "datasets": reports}, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
