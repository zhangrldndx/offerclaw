#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build adjudicated colloquial negatives for the V2-A development set.

Dev-New ships 80 positives and zero negatives, so every claim about the
evidence gate on colloquial queries currently has to be guarded with V1's 16
negatives -- all of which are written in formal register.  That is the exact
sampling error this round already documented twice: V1's "colloquial" phrasings
sit at cosine 0.98 from their own standard form, so a guard built from them
cannot detect a gate that has been loosened specifically for colloquial input.

Two subcommands, deliberately separated by a human step:

``probe``
    Run production retrieval on each candidate question against the frozen
    index and write an adjudication worksheet -- top candidates, distances,
    reranker scores and the live gate decision.

``build``
    Emit the graded-qrels negative cases, and *only* for candidates carrying an
    explicit ``unanswerable`` verdict.  A candidate with no verdict, or one
    whose recorded evidence no longer matches the index, is refused rather than
    labelled.  Auto-labelling un-adjudicated candidates as negatives is the one
    thing this file must never do.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same plan the colloquial evaluator uses, so a probe reproduces what the
# harness will later measure rather than a different retrieval path.
REFERENCE_PLAN = {
    "decision": "answer",
    "routes": [{"source": "reference_kb", "operation": "search"}],
}
PROBE_ARM = "compact32"
NEGATIVE_TYPES = (
    "out_of_domain",
    "near_domain_missing",
    "wrong_relation",
    "ambiguous_or_injection",
)
COLLOQUIAL_STYLES = ("natural", "implicit_oral", "long_noisy")
SNIPPET_CHARS = 220


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    seen: set[str] = set()
    for candidate in candidates:
        for key in ("id", "negative_type", "query_style", "question", "intent"):
            if not str(candidate.get(key, "")).strip():
                raise SystemExit(f"{candidate.get('id')}: missing {key}")
        if candidate["negative_type"] not in NEGATIVE_TYPES:
            raise SystemExit(
                f"{candidate['id']}: unknown negative_type "
                f"{candidate['negative_type']!r}"
            )
        # V1 already covers the formal register; a standard-phrased negative
        # here would add nothing this set exists to measure.
        if candidate["query_style"] not in COLLOQUIAL_STYLES:
            raise SystemExit(
                f"{candidate['id']}: query_style must be one of "
                f"{COLLOQUIAL_STYLES}, got {candidate['query_style']!r}"
            )
        if candidate["id"] in seen:
            raise SystemExit(f"duplicate candidate id: {candidate['id']}")
        seen.add(candidate["id"])
    return candidates


def probe(args: argparse.Namespace) -> None:
    import chromadb
    from rag_colloquial_profiles import colloquial_profile
    from rag_gate import retrieve_with_trace
    from rag_tools import get_collection_name, index_fingerprint

    candidates = _load_candidates(Path(args.candidates).expanduser().resolve())
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    index = index_fingerprint(collection=collection)
    profile = colloquial_profile(PROBE_ARM)

    rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        print(f"[probe] {position}/{len(candidates)} {candidate['id']}",
              file=sys.stderr, flush=True)
        trace = retrieve_with_trace(
            candidate["question"], REFERENCE_PLAN, profile, top_k=5,
        )
        top = []
        for rank, item in enumerate(trace.final_candidates[:5], start=1):
            document = getattr(item, "document", "") or ""
            metadata = getattr(item, "metadata", {}) or {}
            top.append({
                "rank": rank,
                "chunk_id": str(item.chunk_id),
                "source": str(metadata.get("source", "")),
                "heading": str(metadata.get("heading", ""))[:120],
                "distance": getattr(item, "distance", None),
                "rerank_score": getattr(item, "rerank_score", None),
                "snippet": document[:SNIPPET_CHARS].replace("\n", " "),
            })
        rows.append({
            **{key: candidate[key] for key in
               ("id", "negative_type", "query_style", "question", "intent")},
            "gate_decision": bool(trace.gate_decision),
            "gate_features": dict(trace.gate_features),
            "top_candidates": top,
        })

    report = {
        "schema_version": "colloquial-negative-probe-v1",
        "arm": PROBE_ARM,
        "index": index,
        "candidates_sha256": hashlib.sha256(
            Path(args.candidates).expanduser().resolve().read_bytes()
        ).hexdigest(),
        "rows": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))
    if args.worksheet:
        worksheet = Path(args.worksheet).expanduser().resolve()
        worksheet.parent.mkdir(parents=True, exist_ok=True)
        worksheet.write_text(render_worksheet(report), encoding="utf-8")
        print(f"worksheet: {worksheet}", file=sys.stderr)
    accepted = sum(1 for row in rows if row["gate_decision"])
    print(json.dumps({
        "output": str(output),
        "candidates": len(rows),
        "gate_would_accept_now": accepted,
    }, ensure_ascii=False, indent=2))


def render_worksheet(report: dict[str, Any]) -> str:
    lines = ["# 口语负例裁决工作表", "",
             f"索引：`{report['index'].get('collection', '?')}` "
             f"{report['index'].get('count', '?')} 块", ""]
    for row in report["rows"]:
        lines += [
            f"## {row['id']}  ({row['negative_type']} / {row['query_style']})",
            "",
            f"**问题**：{row['question']}",
            "",
            f"**作者意图**：{row['intent']}",
            "",
            f"**当前证据门**：{'放行 ⚠️' if row['gate_decision'] else '拒答 ✅'}",
            "",
            "| # | 来源 | 距离 | 精排 | 片段 |",
            "|---:|---|---:|---:|---|",
        ]
        for item in row["top_candidates"]:
            distance = ("—" if item["distance"] is None
                        else f"{float(item['distance']):.3f}")
            rerank = ("—" if item["rerank_score"] is None
                      else f"{float(item['rerank_score']):.4f}")
            snippet = item["snippet"].replace("|", "\\|")
            lines.append(f"| {item['rank']} | {item['source']} | {distance} "
                         f"| {rerank} | {snippet} |")
        lines.append("")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    from rag_qrels_v2 import (
        SCHEMA_VERSION, index_contract_fingerprint, validate_graded_qrels,
    )

    candidates = {row["id"]: row for row in
                  _load_candidates(Path(args.candidates).expanduser().resolve())}
    probe_report = json.loads(
        Path(args.probe).expanduser().resolve().read_text(encoding="utf-8"))
    probed = {row["id"]: row for row in probe_report["rows"]}
    adjudication = json.loads(
        Path(args.adjudication).expanduser().resolve().read_text(encoding="utf-8"))
    verdicts = adjudication["verdicts"]

    items: list[dict[str, Any]] = []
    refused: list[str] = []
    for candidate_id, candidate in candidates.items():
        verdict = verdicts.get(candidate_id)
        if not verdict or verdict.get("verdict") != "unanswerable":
            refused.append(candidate_id)
            continue
        if not str(verdict.get("rationale", "")).strip():
            raise SystemExit(f"{candidate_id}: verdict without a rationale")
        row = probed.get(candidate_id)
        if row is None:
            raise SystemExit(f"{candidate_id}: adjudicated but never probed")
        # The rationale was written against specific chunks.  If retrieval no
        # longer surfaces them, the reasoning behind the verdict is stale and
        # the case must be re-adjudicated rather than shipped.
        reviewed = list(verdict.get("checked_chunk_ids", []))
        surfaced = [item["chunk_id"] for item in row["top_candidates"]]
        missing = [chunk_id for chunk_id in reviewed if chunk_id not in surfaced]
        if missing:
            raise SystemExit(
                f"{candidate_id}: adjudicated against chunks the index no "
                f"longer returns: {missing[:3]}"
            )
        items.append({
            "anchor_id": f"v2a-negative-{candidate['negative_type']}-{candidate_id}",
            "query_id": candidate_id,
            "question": candidate["question"],
            "query_style": candidate["query_style"],
            "case_kind": "negative",
            "domain": "negative",
            "split": "dev",
            "expected_behavior": "abstain_from_kb",
            "phenomena": [candidate["negative_type"], "implicit_or_colloquial"],
            "answer_requirements": [],
            "relevant_targets": [],
            "hard_negatives": [],
            "negative_rationale": verdict["rationale"],
            "adjudication": {
                "method": "evidence_first_manual_adjudication_against_frozen_index",
                "checked_chunk_ids": reviewed,
                "gate_decision_at_adjudication": row["gate_decision"],
            },
            "review_note": adjudication.get("review_note", ""),
            "review_status": "approved",
            "human_review": {"wording_edit": candidate["question"]},
        })

    if refused:
        print(f"refused {len(refused)} un-adjudicated candidates: {refused}",
              file=sys.stderr)
    if not items:
        raise SystemExit("no adjudicated negatives to write")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "rag-colloquial-v2a-negatives-v1",
        "status": "approved",
        "design": {
            "purpose": ("colloquial-register guard for evidence-gate work; "
                        "V1's 16 negatives are all formally phrased"),
            "negative_types": sorted({
                item["phenomena"][0] for item in items}),
            "probe_arm": probe_report["arm"],
        },
        # The probe stores the raw fingerprint dict; qrels artifacts carry the
        # contract view of it, so the two stay comparable across rounds.
        "index": {
            "collection": probe_report["index"]["collection"],
            "count": int(probe_report["index"]["collection_count"]),
            "fingerprint": index_contract_fingerprint(probe_report["index"]),
            "fingerprint_id": probe_report["index"]["fingerprint_id"],
        },
        "items": sorted(items, key=lambda item: item["query_id"]),
    }
    validate_graded_qrels(payload, allowed_splits={"dev"}, require_approved=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(payload))
    print(json.dumps({
        "output": str(output),
        "negatives": len(items),
        "by_type": dict(Counter(item["phenomena"][0] for item in items)),
        "by_style": dict(Counter(item["query_style"] for item in items)),
        "gate_accepts_at_adjudication": sum(
            1 for item in items
            if item["adjudication"]["gate_decision_at_adjudication"]),
    }, ensure_ascii=False, indent=2))


def merge(args: argparse.Namespace) -> None:
    """Emit one runnable Dev-New file: positives + adjudicated negatives.

    The evaluator reads a single ``--cases`` file and splits it by
    ``case_kind``, so the negatives have to travel with the positives.  The
    two source files stay canonical; this output is a view.
    """
    from rag_qrels_v2 import load_graded_qrels, validate_graded_qrels

    positives = load_graded_qrels(Path(args.positives).expanduser().resolve(),
                                  allowed_splits={"dev"})
    negatives = load_graded_qrels(Path(args.negatives).expanduser().resolve(),
                                  allowed_splits={"dev"})
    if positives["index"] != negatives["index"]:
        raise SystemExit(
            "positives and negatives were built against different indexes; "
            "re-probe the negatives before merging"
        )
    collisions = ({item["query_id"] for item in positives["items"]}
                  & {item["query_id"] for item in negatives["items"]})
    if collisions:
        raise SystemExit(f"query_id collision: {sorted(collisions)[:5]}")
    payload = {
        **positives,
        "dataset_id": "rag-colloquial-v2a-dev-with-negatives-v1",
        "items": [*positives["items"], *negatives["items"]],
    }
    validate_graded_qrels(payload, allowed_splits={"dev"}, require_approved=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(payload))
    print(json.dumps({
        "output": str(output),
        "positives": len(positives["items"]),
        "negatives": len(negatives["items"]),
    }, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    probe_parser = sub.add_parser("probe", help="collect evidence for adjudication")
    probe_parser.add_argument("--candidates", required=True)
    probe_parser.add_argument("--output", required=True)
    probe_parser.add_argument("--worksheet")
    probe_parser.set_defaults(func=probe)
    build_parser = sub.add_parser("build", help="emit adjudicated negative cases")
    build_parser.add_argument("--candidates", required=True)
    build_parser.add_argument("--probe", required=True)
    build_parser.add_argument("--adjudication", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.set_defaults(func=build)
    merge_parser = sub.add_parser("merge", help="one runnable Dev-New cases file")
    merge_parser.add_argument("--positives", required=True)
    merge_parser.add_argument("--negatives", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(func=merge)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
