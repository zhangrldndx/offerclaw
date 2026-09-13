#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarise the candidate-membership sweep: what each arm buys and costs.

The sweep exists to answer one question -- can the golds that the union holds
but RRF@20 drops be recovered, and at what price -- so the report is built
around the funnel stage where they are lost, not around the headline R@1.

Latency is reported from the second row onward.  The first query of a run pays
model load and index warmup (tens of seconds) and would otherwise dominate the
percentiles of an 80-row set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = payload["runs"][0]["positive"]
    metrics, rows = run["metrics"], run["rows"]
    profile = payload["configuration"]["retrieval_profile"]
    warm = [row for row in rows[1:]]
    total = [row["latency_ms"] for row in warm]
    rerank = [row["latency_by_stage"].get("rerank", 0.0) for row in warm]
    funnel, ranking = metrics["funnel"], metrics["strict_ranking"]
    return {
        "arm": profile["name"].replace("colloquial-", ""),
        "pool_size": profile["pool_size"],
        "strategy": profile["candidate_pool_strategy"],
        "exclusive_per_channel": profile.get("exclusive_per_channel"),
        "n": ranking["recall@1"]["n"],
        "union": funnel["union_candidate"]["hits"],
        "rrf": funnel["rrf_candidate"]["hits"],
        "union_preservation": funnel["union_preservation"]["hits"],
        "union_preservation_n": funnel["union_preservation"]["n"],
        "candidate_to_top1": funnel["candidate_to_top1"]["hits"],
        "r1": ranking["recall@1"]["hits"],
        "r3": ranking["recall@3"]["hits"],
        "r5": ranking["recall@5"]["hits"],
        "mrr10": ranking["mrr@10"],
        "effective_evidence": funnel["effective_evidence"]["hits"],
        "total_p50": round(statistics.median(total), 1),
        "total_p95": round(_percentile(total, 0.95), 1),
        "rerank_p50": round(statistics.median(rerank), 1),
        "warm_rows": len(warm),
        # Which queries the union held but the fused pool dropped -- the exact
        # population this sweep is trying to move.  Membership must be tested
        # on chunk ids: ``union_rank``/``fusion_rank`` are ranks inside their
        # own (differently sized) lists and are never None, so comparing them
        # silently reports zero losses for every arm.
        "lost_in_fusion": [
            row["query_id"] for row in rows
            if (set(row["all_relevant_chunk_ids"]) & set(row["union_chunk_ids"]))
            and not (set(row["all_relevant_chunk_ids"]) & set(row["fusion_chunk_ids"]))
        ],
        "r1_hits": [row["query_id"] for row in rows if row["final_rank"] == 1],
    }


def render(entries: list[dict[str, Any]], reference: str) -> str:
    base = next((e for e in entries if e["arm"] == reference), entries[0])
    lines = [
        f"| arm | pool | strategy | union | RRF@pool | 保留率 | R@1 | R@3 | R@5 "
        f"| MRR@10 | total p50 | total p95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        preserve = (f"{entry['union_preservation']}/{entry['union_preservation_n']}")
        mark = lambda cur, ref: (f"{cur}" if cur == ref else f"{cur} ({cur - ref:+d})")
        lines.append(
            f"| `{entry['arm']}` | {entry['pool_size']} | {entry['strategy']}"
            + (f" ×{entry['exclusive_per_channel']}"
               if entry["strategy"] == "retain_channel_exclusives" else "")
            + f" | {mark(entry['union'], base['union'])}"
            f" | {mark(entry['rrf'], base['rrf'])} | {preserve}"
            f" | {mark(entry['r1'], base['r1'])}"
            f" | {mark(entry['r3'], base['r3'])}"
            f" | {mark(entry['r5'], base['r5'])}"
            f" | {entry['mrr10']:.4f}"
            f" | {entry['total_p50']:.0f} ms"
            f" | {entry['total_p95']:.0f} ms |"
        )
    lines += ["", "融合阶段丢失的题（union 有、RRF 池没有）：", ""]
    for entry in entries:
        lost = entry["lost_in_fusion"]
        lines.append(f"- `{entry['arm']}`: {len(lost)} 题"
                     + (f" — {', '.join(lost)}" if lost else ""))
    # A recovered gold that lands at rank 4 is not a win yet; say so explicitly
    # rather than letting the R@1 column imply the arm did nothing.
    lines += ["", "逐题 R@1 相对基线的变化：", ""]
    base_hits = set(base["r1_hits"])
    for entry in entries:
        if entry["arm"] == reference:
            continue
        hits = set(entry["r1_hits"])
        won, lost = sorted(hits - base_hits), sorted(base_hits - hits)
        lines.append(f"- `{entry['arm']}`: +{len(won)} / -{len(lost)}"
                     + (f" | win={won}" if won else "")
                     + (f" | loss={lost}" if lost else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", required=True,
                        help="glob of eval outputs for one dataset")
    parser.add_argument("--reference", default="compact32")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise SystemExit(f"no eval outputs matched {args.glob}")
    entries = [load(path) for path in paths]
    entries.sort(key=lambda e: (e["strategy"], e["pool_size"],
                                e["exclusive_per_channel"] or 0))
    text = render(entries, args.reference)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"schema_version": "colloquial-pool-sweep-v1",
                        "reference": args.reference, "arms": entries},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
