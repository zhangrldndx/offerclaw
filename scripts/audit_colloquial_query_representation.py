#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure what a colloquial question costs *in query representation*.

The colloquial waves are built as a paired design: one anchor, one gold chunk,
four surface forms of the same intent.  That pairing is what makes query
representation measurable at all -- the document side is held byte-identical,
so any per-style difference in where the gold lands is attributable to the
query encoding and nothing else.

Two decompositions are reported, because they imply different fixes:

``gold_similarity``
    cos(query, gold).  If a colloquial form loses the gold, does it lose it
    absolutely, or only relative to a distractor that gained?

``margin``
    cos(query, gold) - cos(query, best non-gold).  Ranking only reads this.
    A uniform contraction of every similarity (which a shorter, vaguer query
    naturally produces) moves ``gold_similarity`` a lot and ``margin`` not at
    all, and is therefore harmless.  Treating the two as interchangeable is
    the standard way to misdiagnose this failure.

Read-only: the collection is queried and its stored vectors are read, never
written.  Nothing here depends on, or can alter, the production index.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deep enough to see where a failed style actually put the gold; production
# only ever looks at the first ``pool_size``.
PROBE_DEPTH = 200
REFERENCE_STYLE = "standard"


def cosine_from_distance(distance: float) -> float:
    """Convert the collection's stored distance to a cosine similarity.

    The production collection carries no ``hnsw:space`` metadata, so it uses
    Chroma's default ``l2``, which reports the *squared* euclidean distance
    (verified against a self-query: an exact match returns 0.0 and the first
    neighbours return ~0.35, not ~0.59).  With unit-norm vectors
    ``d = 2 - 2cos``.  Getting this wrong compresses every similarity toward
    1.0 and would make the colloquial penalty look smaller than it is.
    """
    return 1.0 - distance / 2.0


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload["items"]
            if item.get("case_kind", "positive") == "positive"]


def _gold_ids(item: dict[str, Any]) -> set[str]:
    return {target["chunk_id"] for target in item.get("relevant_targets", [])}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "p10": round(_quantile(values, 0.10), 6),
        "p90": round(_quantile(values, 0.90), 6),
    }


def measure(cases_path: Path) -> dict[str, Any]:
    import chromadb
    from rag_tools import get_collection_name, get_embeddings_batch, index_fingerprint
    from rag_bm25 import _tokenize

    cases = _load_cases(cases_path)
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())

    questions = [item["question"] for item in cases]
    print(f"embedding {len(questions)} questions ...", file=sys.stderr)
    vectors = get_embeddings_batch(questions)

    rows: list[dict[str, Any]] = []
    for item, vector in zip(cases, vectors):
        gold = _gold_ids(item)
        result = collection.query(
            query_embeddings=[vector], n_results=PROBE_DEPTH,
            include=["distances", "documents"],
        )
        ids = result["ids"][0]
        sims = [cosine_from_distance(float(d)) for d in result["distances"][0]]
        gold_rank, gold_sim = None, None
        for rank, (chunk_id, sim) in enumerate(zip(ids, sims), start=1):
            if chunk_id in gold:
                gold_rank, gold_sim = rank, sim
                break
        best_other = next((sim for chunk_id, sim in zip(ids, sims)
                           if chunk_id not in gold), None)
        gold_doc = next((doc for chunk_id, doc in zip(ids, result["documents"][0])
                         if chunk_id in gold), None)
        query_tokens = set(_tokenize(item["question"]))
        gold_tokens = set(_tokenize(gold_doc)) if gold_doc else set()
        rows.append({
            "anchor_id": item["anchor_id"],
            "query_id": item["query_id"],
            "style": item.get("query_style", "standard"),
            "question_chars": len(item["question"]),
            "dense_rank": gold_rank,
            "gold_similarity": round(gold_sim, 6) if gold_sim is not None else None,
            "best_other_similarity": round(best_other, 6) if best_other is not None else None,
            "margin": (round(gold_sim - best_other, 6)
                       if gold_sim is not None and best_other is not None else None),
            # Only meaningful when the gold appeared inside PROBE_DEPTH; the
            # lexical figures below are independent of that.
            "query_tokens": len(query_tokens),
            "lexical_overlap": (round(len(query_tokens & gold_tokens) / len(query_tokens), 6)
                                if query_tokens and gold_tokens else None),
        })

    return {
        "schema_version": "colloquial-query-representation-v1",
        "cases": str(cases_path.relative_to(ROOT)) if cases_path.is_relative_to(ROOT)
                 else str(cases_path),
        "probe_depth": PROBE_DEPTH,
        "index": index_fingerprint(collection=collection),
        "rows": rows,
        "by_style": by_style(rows),
        "paired_vs_standard": paired_vs_standard(rows),
    }


def by_style(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["style"]].append(row)
    out: dict[str, Any] = {}
    for style, group in sorted(grouped.items()):
        ranked = [row for row in group if row["dense_rank"] is not None]
        out[style] = {
            "n": len(group),
            "dense_r1": sum(1 for row in ranked if row["dense_rank"] == 1),
            "dense_r20": sum(1 for row in ranked if row["dense_rank"] <= 20),
            "gold_beyond_probe_depth": len(group) - len(ranked),
            "question_chars": _summarize([row["question_chars"] for row in group]),
            "gold_similarity": _summarize([row["gold_similarity"] for row in ranked]),
            "best_other_similarity": _summarize(
                [row["best_other_similarity"] for row in group
                 if row["best_other_similarity"] is not None]),
            "margin": _summarize([row["margin"] for row in ranked]),
            "lexical_overlap": _summarize(
                [row["lexical_overlap"] for row in group
                 if row["lexical_overlap"] is not None]),
        }
    return out


def paired_vs_standard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Within-anchor deltas.

    Comparing style means across anchors would confound "colloquial is hard"
    with "the anchors that happen to carry colloquial forms are hard".  Every
    anchor here carries all four styles, so the paired delta removes the
    anchor entirely.
    """
    by_anchor: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_anchor[row["anchor_id"]][row["style"]] = row

    out: dict[str, Any] = {}
    styles = {row["style"] for row in rows} - {REFERENCE_STYLE}
    for style in sorted(styles):
        gold_deltas, margin_deltas, other_deltas, lex_deltas = [], [], [], []
        worse, better, same = 0, 0, 0
        for anchor in by_anchor.values():
            ref, alt = anchor.get(REFERENCE_STYLE), anchor.get(style)
            if not ref or not alt:
                continue
            if ref["gold_similarity"] is not None and alt["gold_similarity"] is not None:
                gold_deltas.append(alt["gold_similarity"] - ref["gold_similarity"])
            if ref["margin"] is not None and alt["margin"] is not None:
                margin_deltas.append(alt["margin"] - ref["margin"])
            if (ref["best_other_similarity"] is not None
                    and alt["best_other_similarity"] is not None):
                other_deltas.append(alt["best_other_similarity"]
                                    - ref["best_other_similarity"])
            if ref["lexical_overlap"] is not None and alt["lexical_overlap"] is not None:
                lex_deltas.append(alt["lexical_overlap"] - ref["lexical_overlap"])
            ref_rank = ref["dense_rank"] or PROBE_DEPTH + 1
            alt_rank = alt["dense_rank"] or PROBE_DEPTH + 1
            if alt_rank > ref_rank:
                worse += 1
            elif alt_rank < ref_rank:
                better += 1
            else:
                same += 1
        out[style] = {
            "pairs": worse + better + same,
            "dense_rank_worse_than_standard": worse,
            "dense_rank_better_than_standard": better,
            "dense_rank_tied": same,
            "delta_gold_similarity": _summarize(gold_deltas),
            "delta_best_other_similarity": _summarize(other_deltas),
            "delta_margin": _summarize(margin_deltas),
            "delta_lexical_overlap": _summarize(lex_deltas),
        }
    return out


def render(report: dict[str, Any]) -> str:
    lines = [
        "| style | n | dense R@1 | dense R@20 | cos(q,gold) med | cos(q,best other) med "
        "| margin med | lexical overlap med | chars med |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for style, stat in report["by_style"].items():
        lines.append(
            f"| {style} | {stat['n']} | {stat['dense_r1']} | {stat['dense_r20']} "
            f"| {stat['gold_similarity'].get('median', float('nan')):.4f} "
            f"| {stat['best_other_similarity'].get('median', float('nan')):.4f} "
            f"| {stat['margin'].get('median', float('nan')):+.4f} "
            f"| {stat['lexical_overlap'].get('median', float('nan')):.3f} "
            f"| {stat['question_chars'].get('median', float('nan')):.0f} |"
        )
    lines += ["", f"Paired within-anchor deltas vs `{REFERENCE_STYLE}`:", "",
              "| style | worse | tied | better | Δcos(q,gold) med | Δcos(q,best other) med "
              "| Δmargin med | Δlexical med |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for style, stat in report["paired_vs_standard"].items():
        lines.append(
            f"| {style} | {stat['dense_rank_worse_than_standard']} "
            f"| {stat['dense_rank_tied']} | {stat['dense_rank_better_than_standard']} "
            f"| {stat['delta_gold_similarity'].get('median', float('nan')):+.4f} "
            f"| {stat['delta_best_other_similarity'].get('median', float('nan')):+.4f} "
            f"| {stat['delta_margin'].get('median', float('nan')):+.4f} "
            f"| {stat['delta_lexical_overlap'].get('median', float('nan')):+.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, action="append",
                        help="repeatable; each file is measured separately")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    reports = {}
    for raw in args.cases:
        path = Path(raw).expanduser().resolve()
        report = measure(path)
        reports[path.stem] = report
        print(f"\n### {path.stem}\n", file=sys.stderr)
        print(render(report), file=sys.stderr)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": "colloquial-query-representation-v1",
                    "datasets": reports}, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
