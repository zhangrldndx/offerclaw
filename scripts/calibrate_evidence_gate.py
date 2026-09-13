#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate a second acceptance path for the evidence gate.

The gate currently passes 82% of correct answers to formally phrased questions
and 30-50% of correct answers to colloquial ones, because its primary test is a
dense-distance threshold and colloquial queries sit systematically farther from
their own gold chunk.  The question here is whether a *second* path -- "the main
gate said no, but the reranker is very sure" -- can recover those without
letting anything through that should be refused.

Distribution first, threshold second.  The 2026-08-24 English evidence gate was
calibrated this way (collect the score distribution, then pick the point where
correct acceptances rise and wrong ones do not), and picking a round number
first is how a gate ends up fitted to whichever set happened to be at hand.

Three populations, and all three have to be reported, because a rule tuned on
one of them alone will look better than it is:

``rescuable``   positive, gold is Top1, gate refused -- what we want back.
``wrong_top1``  positive, gold is *not* Top1, gate refused.  Admitting these
                means answering from a chunk that is not the labelled evidence.
                Not automatically a fabrication (the qrels are chunk-level and
                an unlabelled equivalent may exist -- that exact gap was found
                and fixed on Dev80 earlier), so this is reported as a cost to
                weigh, not as a hard failure.
``negative``    no correct answer exists anywhere in the corpus.  Admitting one
                is unambiguously wrong, and is the hard constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]

# Preregistered rule family, kept deliberately small.  With 99 rows an open
# search over feature combinations would fit the noise; these three encode the
# only mechanisms the distribution actually suggests.
RULES: dict[str, Callable[[dict[str, Any], float, float], bool]] = {
    # confidence alone
    "rerank_top": lambda f, tau, _m: _get(f, "rerank_top") >= tau,
    # confidence plus separation from the runner-up: correct rescues showed a
    # much larger margin (median 0.167) than wrong-Top1 rows (0.052)
    "rerank_top_and_margin": lambda f, tau, m: (
        _get(f, "rerank_top") >= tau and _get(f, "rerank_margin") >= m),
    # confidence plus lexical corroboration from the other channel
    "rerank_top_and_bm25": lambda f, tau, _m: (
        _get(f, "rerank_top") >= tau and 0 < _get(f, "bm25_best_rank", 999) <= 5),
}
TAUS = (0.70, 0.75, 0.80, 0.85, 0.87, 0.90, 0.93, 0.95, 0.97)
MARGINS = (0.0, 0.05, 0.10, 0.15)


def _get(features: dict[str, Any], key: str, default: float = -1.0) -> float:
    value = features.get(key)
    return default if value is None else float(value)


def populations(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    run = payload["runs"][0]
    positives, negatives = run["positive"]["rows"], run["negative"]["rows"]
    return {
        "already_accepted": [r for r in positives
                             if r["final_rank"] == 1 and r["gate_decision"]],
        "rescuable": [r for r in positives
                      if r["final_rank"] == 1 and not r["gate_decision"]],
        "wrong_top1_accepted": [r for r in positives
                                if r["final_rank"] != 1 and r["gate_decision"]],
        "wrong_top1": [r for r in positives
                       if r["final_rank"] != 1 and not r["gate_decision"]],
        "negative": [r for r in negatives if not r["gate_decision"]],
        "negative_accepted": [r for r in negatives if r["gate_decision"]],
    }


def _describe(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [_get(r["gate_features"], key) for r in rows]
    values = [v for v in values if v >= 0]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": round(statistics.median(values), 4),
            "min": round(min(values), 4), "max": round(max(values), 4)}


def sweep(pools: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, rule in RULES.items():
        margins = MARGINS if name == "rerank_top_and_margin" else (0.0,)
        for tau in TAUS:
            for margin in margins:
                fired = {
                    key: [r for r in rows if rule(r["gate_features"], tau, margin)]
                    for key, rows in pools.items()
                }
                rescued = fired["rescuable"]
                by_style: dict[str, int] = {}
                for row in rescued:
                    style = row.get("query_style", "?")
                    by_style[style] = by_style.get(style, 0) + 1
                out.append({
                    "rule": name, "tau": tau, "margin": margin,
                    "rescued_correct": len(rescued),
                    "rescued_by_style": by_style,
                    "new_false_accept_negative": len(fired["negative"]),
                    "new_accept_wrong_top1": len(fired["wrong_top1"]),
                    "rescued_query_ids": [r["query_id"] for r in rescued],
                    "false_accept_query_ids": [r["query_id"] for r in fired["negative"]],
                })
    return out


def recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Most rescues subject to zero new false accepts on the negatives.

    Ties break toward fewer wrong-Top1 admissions, then toward the higher
    threshold -- when two settings buy the same thing, take the stricter one.
    """
    safe = [row for row in rows if row["new_false_accept_negative"] == 0]
    if not safe:
        return {"verdict": "no_safe_threshold"}
    best = max(safe, key=lambda row: (row["rescued_correct"],
                                      -row["new_accept_wrong_top1"], row["tau"]))
    return {"verdict": "candidate", **best}


def render(pools: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]],
           choice: dict[str, Any]) -> str:
    lines = ["## 各组的门特征分布", "",
             "| 组 | n | dense 距离 中位 | rerank_top 中位 | rerank_top 最大 | margin 中位 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, group in pools.items():
        if not group:
            lines.append(f"| `{key}` | 0 | — | — | — | — |")
            continue
        dense = _describe(group, "best_dense_distance")
        rerank = _describe(group, "rerank_top")
        margin = _describe(group, "rerank_margin")
        lines.append(
            f"| `{key}` | {len(group)} | {dense.get('median', float('nan')):.3f} "
            f"| {rerank.get('median', float('nan')):.4f} "
            f"| {rerank.get('max', float('nan')):.4f} "
            f"| {margin.get('median', float('nan')):.4f} |")

    lines += ["", "## 阈值扫描（第二通道 = 主门拒 且 满足规则）", "",
              "| 规则 | τ | margin | 救回正确 | **新增负例误纳** | 新增放行错 Top1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        flag = "**" if row["new_false_accept_negative"] else ""
        lines.append(
            f"| `{row['rule']}` | {row['tau']:.2f} | {row['margin']:.2f} "
            f"| {row['rescued_correct']} "
            f"| {flag}{row['new_false_accept_negative']}{flag} "
            f"| {row['new_accept_wrong_top1']} |")
    if choice.get("verdict") == "candidate":
        lines += ["", "## 推荐", "",
                  f"规则 `{choice['rule']}`，τ={choice['tau']}，"
                  f"margin={choice['margin']}：救回 **{choice['rescued_correct']}** 道，"
                  f"负例误纳 **0**，同时放行 {choice['new_accept_wrong_top1']} 道错 Top1。",
                  "", f"按风格：{choice['rescued_by_style']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True,
                        help="eval output carrying both positives and negatives")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    payload = json.loads(
        Path(args.eval).expanduser().resolve().read_text(encoding="utf-8"))
    pools = populations(payload)
    if not pools["negative"] and not pools["negative_accepted"]:
        raise SystemExit(
            "this eval has no negatives; calibrating a gate without them would "
            "measure only how much can be let in, never how much should not be"
        )
    rows = sweep(pools)
    choice = recommend(rows)
    report = {
        "schema_version": "colloquial-gate-calibration-v1",
        "eval": Path(args.eval).name,
        "arm": payload["configuration"]["retrieval_arm"],
        "population_sizes": {key: len(value) for key, value in pools.items()},
        "distributions": {
            key: {feature: _describe(value, feature)
                  for feature in ("best_dense_distance", "rerank_top",
                                  "rerank_margin")}
            for key, value in pools.items()
        },
        "sweep": rows,
        "recommendation": choice,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    text = render(pools, rows, choice)
    print(text)
    if args.report:
        path = Path(args.report).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
