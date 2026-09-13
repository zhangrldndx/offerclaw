#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One abstention guard over every negative set the project has.

Built because of what happened on 2026-08-27: the colloquial evidence-gate
second path was calibrated against 19 freshly-authored colloquial negatives,
passed them with zero new false accepts, and was reported as a clean win.  Two
*other* negative sets -- which nobody thought to run -- then produced three new
false accepts (`adv03`, `adv09`, `nf36`), all of the same shape the 19 could
not express: a topic adjacent to the corpus that the corpus does not cover.

Each set has a blind spot determined by how it was constructed.  Running one of
them and calling it "the guard" is how a loosening ships.  So this runs all of
them, reports per-source, and fails on any regression against a stored
baseline rather than leaving the comparison to whoever reads the output.

**Not every negative is an abstention case** (user ruling, 2026-08-28).  Three
evidence relations, three correct behaviours:

    证据支持前提      -> 正常回答
    证据明确反驳前提  -> 指出前提错误并纠正
    既不支持也不反驳  -> 拒答

``wrong_relation`` questions ("MVCC 是不是用来给向量库去重的") are false-premise,
not unanswerable: when the corpus contains the facts that refute the premise,
answering *"不是，MVCC 是数据库并发控制"* is the right behaviour.  Scoring those
as false accepts would penalise the system for correcting the user and would
train the evidence gate to go silent whenever a premise is false.  Rows whose
expected action is ``correct_premise`` are therefore counted separately, via
``expected_action_overlay.json``; everything else defaults to ``abstain``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OVERLAY_PATH = ROOT / "docs/rag_eval/colloquial/guard/expected_action_overlay.json"


def expected_actions() -> dict[str, str]:
    """Per-id expected action; absent means ``abstain``.

    Only adjudicated rows count.  ``proposals_pending_review`` holds the
    judge's suggestions for datasets this session did not author (V1 is an
    approved frozen artifact) -- those stay visible but do not change scoring
    until a human accepts them.
    """
    if not OVERLAY_PATH.is_file():
        return {}
    payload = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    return {row_id: entry["expected_action"]
            for row_id, entry in (payload.get("adjudicated") or {}).items()}

REFERENCE_PLAN = {
    "decision": "answer",
    "routes": [{"source": "reference_kb", "operation": "search"}],
}

# (source, path, extractor).  Kept explicit rather than globbed: a guard that
# silently loses a source when a file is renamed is worse than no guard.
SOURCES: tuple[tuple[str, str, str], ...] = (
    ("colloquial19", "docs/rag_eval/colloquial/rag_colloquial_v2a_negatives.json",
     "qrels_negative"),
    ("dev80_v1", "docs/rag_eval/colloquial/rag_colloquial_dev80_v1.json",
     "qrels_negative"),
    ("bench_simple", "tests/rag_bench_set.json", "bench_gate_negatives"),
    ("bench_adversarial", "tests/rag_gate_adversarial_negatives.json", "id_q_list"),
    ("final60", "tests/negative_final_set.json", "id_q_items"),
)


def load_source(kind: str, payload: Any, source: str) -> list[dict[str, str]]:
    if kind == "qrels_negative":
        return [{"id": item["query_id"], "q": item["question"], "source": source,
                 "kind": (item.get("phenomena") or ["negative"])[0]}
                for item in payload["items"]
                if item.get("case_kind") == "negative"]
    if kind == "bench_gate_negatives":
        return [{"id": f"{source}-{index + 1}", "q": question, "source": source,
                 "kind": "simple"}
                for index, question in enumerate(payload["gate_negatives"])]
    if kind == "id_q_list":
        return [{"id": item["id"], "q": item["q"], "source": source,
                 "kind": item.get("kind", "adversarial")} for item in payload]
    if kind == "id_q_items":
        return [{"id": item["id"], "q": item["q"], "source": source,
                 "kind": item.get("kind", "unspecified")}
                for item in payload["items"]]
    raise SystemExit(f"unknown extractor {kind!r}")


def collect() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, relative, kind in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing negative source {relative}; the guard is "
                             "incomplete and must not report a pass")
        items = load_source(kind, json.loads(path.read_text(encoding="utf-8")), source)
        for item in items:
            key = item["q"].strip()
            if key in seen:      # the same question in two sets is one question
                continue
            seen.add(key)
            rows.append(item)
    return rows


def run(arm: str) -> dict[str, Any]:
    import chromadb
    from rag_colloquial_profiles import colloquial_profile
    from rag_gate import retrieve_with_trace
    from rag_tools import get_collection_name, index_fingerprint

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    profile = colloquial_profile(arm)
    rows = collect()
    expected = expected_actions()
    results = []
    for position, item in enumerate(rows, start=1):
        print(f"[guard] {position}/{len(rows)} {item['source']}/{item['id']}",
              file=sys.stderr, flush=True)
        trace = retrieve_with_trace(item["q"], REFERENCE_PLAN, profile, top_k=5)
        action = expected.get(item["id"], "abstain")
        results.append({
            **item,
            "expected_action": action,
            "accepted": bool(trace.gate_decision),
            "gate_features": dict(trace.gate_features),
            "top1_chunk_id": (trace.final_candidates[0].chunk_id
                              if trace.final_candidates else None),
        })

    # Only rows whose expected action is abstention can produce a false accept.
    must_abstain = [row for row in results if row["expected_action"] == "abstain"]
    correctable = [row for row in results
                   if row["expected_action"] == "correct_premise"]
    by_source: dict[str, dict[str, Any]] = {}
    for source in {row["source"] for row in results}:
        group = [row for row in must_abstain if row["source"] == source]
        accepted = [row for row in group if row["accepted"]]
        by_source[source] = {
            "n_must_abstain": len(group),
            "n_correctable": sum(1 for row in correctable
                                 if row["source"] == source),
            "false_accepts": len(accepted),
            "reject_rate": round(1 - len(accepted) / len(group), 6) if group else None,
            "false_accept_ids": sorted(row["id"] for row in accepted),
        }
    accepted_all = [row for row in must_abstain if row["accepted"]]
    return {
        "schema_version": "colloquial-negative-guard-v1",
        "arm": arm,
        "index": index_fingerprint(collection=collection),
        "sources": [name for name, _p, _k in SOURCES],
        "n": len(results),
        "n_must_abstain": len(must_abstain),
        "n_correctable": len(correctable),
        # A correctable row the gate accepted is the *desired* outcome, so it
        # is reported as coverage rather than buried as a pass.
        "correctable_reached": sum(1 for row in correctable if row["accepted"]),
        "correctable_ids": sorted(row["id"] for row in correctable),
        "false_accepts": len(accepted_all),
        "false_accept_ids": sorted(f"{row['source']}/{row['id']}"
                                   for row in accepted_all),
        "by_source": dict(sorted(by_source.items())),
        "by_kind": dict(Counter(row["kind"] for row in accepted_all)),
        "rows": results,
        # The knobs in force, so a run can never be mistaken for a different one.
        "sentinel": {
            key: results[0]["gate_features"].get(key) if results else None
            for key in ("colloquial_gate_min", "structural_evidence_max",
                        "strong_threshold", "rerank_gate_min")
        },
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before = set(baseline["false_accept_ids"])
    after = set(candidate["false_accept_ids"])
    return {
        "new_false_accepts": sorted(after - before),
        "fixed_false_accepts": sorted(before - after),
        "regressed": bool(after - before),
    }


def render(report: dict[str, Any], diff: dict[str, Any] | None) -> str:
    lines = [f"负例守卫合集：{report['n']} 条 "
             f"（应拒答 {report['n_must_abstain']} + 应纠正 {report['n_correctable']}），"
             f"误纳 **{report['false_accepts']}**",
             "",
             f"应纠正的 {report['n_correctable']} 条里，门放行了 "
             f"{report['correctable_reached']} 条——**这是期望行为，不是误纳**"
             f"（{', '.join(report['correctable_ids']) or '—'}）",
             "",
             "| 来源 | 应拒答 | 应纠正 | 误纳 | 拒答率 | 误纳 id |",
             "|---|---:|---:|---:|---:|---|"]
    for source, stat in report["by_source"].items():
        rate = ("—" if stat["reject_rate"] is None
                else f"{stat['reject_rate']:.3f}")
        lines.append(f"| `{source}` | {stat['n_must_abstain']} "
                     f"| {stat['n_correctable']} | {stat['false_accepts']} "
                     f"| {rate} "
                     f"| {', '.join(stat['false_accept_ids']) or '—'} |")
    if diff is not None:
        lines += ["", f"对照基线：新增误纳 **{len(diff['new_false_accepts'])}**"
                      f"，修好 {len(diff['fixed_false_accepts'])}"]
        if diff["new_false_accepts"]:
            lines.append(f"- 新增：{diff['new_false_accepts']}")
        if diff["fixed_false_accepts"]:
            lines.append(f"- 修好：{diff['fixed_false_accepts']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="compact32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", help="a previous run to regress against")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args(argv)

    report = run(args.arm)
    diff = None
    if args.baseline:
        baseline = json.loads(
            Path(args.baseline).expanduser().resolve().read_text(encoding="utf-8"))
        diff = compare(baseline, report)
        report["comparison"] = {"baseline": Path(args.baseline).name, **diff}

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(render(report, diff))
    print(f"\nsentinel: {json.dumps(report['sentinel'], ensure_ascii=False)}")
    print(f"wrote {output}")
    if diff and diff["regressed"] and args.fail_on_regression:
        raise SystemExit(f"abstention regressed: {diff['new_false_accepts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
