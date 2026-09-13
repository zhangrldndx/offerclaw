#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two ``eval_colloquial_rag`` artifacts on **gate** behaviour.

``eval_colloquial_rag --compare-to`` pairs positive R@1 only.  A gate change
moves a different quantity: which queries get answered from the knowledge base
at all, and whether a negative is wrongly admitted.  Both halves matter and
they are not interchangeable -- a change that buys one true rejection by
refusing four correct answers is a loss that an R@1 diff reports as zero.

Fail-closed on provenance: the two artifacts must come from the same index
fingerprint and the same retrieval arm, otherwise the flips being counted are
not attributable to the gate.  Includes the per-arm sentinel this repo learnt
to require the hard way -- a candidate arm that silently ran as the baseline
reports itself as "no difference", which reads exactly like a negative result.

    scripts/compare_gate_arms.py --baseline B0.json --candidate CAND.json \
        --knob structural_evidence_max --expect 0.32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _rows(artifact: dict, kind: str) -> dict[str, dict]:
    return {row["query_id"]: row for row in artifact["runs"][0][kind]["rows"]}


def _provenance(artifact: dict) -> tuple:
    return (
        artifact["input"]["index"].get("fingerprint"),
        artifact["configuration"].get("retrieval_arm"),
        artifact["configuration"]["retrieval_profile"].get("reranker_model"),
    )


def _sentinel(artifact: dict, knob: str, expected: str) -> str | None:
    """Every row of the candidate arm must show the knob actually in force."""
    observed = {
        str(row["gate_features"].get(knob))
        for kind in ("positive", "negative")
        for row in artifact["runs"][0][kind]["rows"]
    }
    if observed != {expected}:
        return f"{knob} observed {sorted(observed)}, expected {{{expected!r}}}"
    return None


def compare(baseline: dict, candidate: dict) -> dict:
    if _provenance(baseline) != _provenance(candidate):
        raise SystemExit(
            f"provenance mismatch: baseline={_provenance(baseline)} "
            f"candidate={_provenance(candidate)}"
        )
    report: dict = {"index_fingerprint": _provenance(baseline)[0],
                    "retrieval_arm": _provenance(baseline)[1]}
    for kind in ("positive", "negative"):
        before, after = _rows(baseline, kind), _rows(candidate, kind)
        if set(before) != set(after):
            raise SystemExit(f"{kind} query set differs between the two arms")
        flips = []
        for query_id in sorted(before):
            was, now = bool(before[query_id]["gate_decision"]), bool(after[query_id]["gate_decision"])
            if was == now:
                continue
            features = after[query_id]["gate_features"]
            flips.append({
                "query_id": query_id,
                "gate": f"{was} -> {now}",
                "rerank_top": [before[query_id]["gate_features"].get("rerank_top"),
                               features.get("rerank_top")],
                "best_dense_distance": [
                    round(float(before[query_id]["gate_features"]["best_dense_distance"]), 4),
                    round(float(features["best_dense_distance"]), 4)],
                "gate_anchor_rank": features.get("gate_anchor_rank"),
                "skipped_structural_top1": features.get("skipped_structural_top1"),
                # for positives only: was this row ever producing usable evidence?
                "final_rank": before[query_id].get("final_rank"),
                "effective_evidence": [before[query_id].get("effective_evidence"),
                                       after[query_id].get("effective_evidence")],
            })
        summary = {"n": len(before), "flips": flips,
                   "gate_pass": [sum(bool(r["gate_decision"]) for r in before.values()),
                                 sum(bool(r["gate_decision"]) for r in after.values())]}
        if kind == "positive":
            for key in ("final_rank_1", "effective_evidence"):
                summary[key] = [
                    sum((int(r.get("final_rank") or 0) == 1) if key == "final_rank_1"
                        else bool(r.get("effective_evidence")) for r in side.values())
                    for side in (before, after)
                ]
            # ranking must be untouched: this is a gate change, not a reranker change
            summary["top1_changed"] = sorted(
                q for q in before
                if (before[q].get("final_chunk_ids") or [None])[:1]
                != (after[q].get("final_chunk_ids") or [None])[:1]
            )
        else:
            summary["false_accepts"] = [
                sorted(q for q, r in side.items() if r["gate_decision"])
                for side in (before, after)
            ]
        report[kind] = summary
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--knob", help="gate_features key that must be in force")
    parser.add_argument("--expect", help="its expected value in every candidate row")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    if args.knob and args.expect:
        failure = _sentinel(candidate, args.knob, args.expect)
        if failure:
            raise SystemExit(f"SENTINEL FAILED: {failure}")
    report = compare(baseline, candidate)
    report["sentinel"] = (f"{args.knob}={args.expect} verified in every candidate row"
                          if args.knob and args.expect else "not checked")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
