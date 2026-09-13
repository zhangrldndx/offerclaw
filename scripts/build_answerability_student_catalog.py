#!/usr/bin/env python3
"""Mine a 60-anchor/180-query answerability pilot catalog from frozen V2-A.

This is candidate discovery only: it never calls a judge and never copies
``answer_requirements`` into the catalog.  Anchors are prioritised by current
BGE false-winner frequency, capped per source, and split by source before any
labels are generated.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability_student_data import QUERY_STYLES, canonical_sha256
from scripts.build_answerability_student_dataset import CATALOG_SCHEMA


DEFAULT_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_train_dev_draft.json"
DEFAULT_TRACES = ROOT / "docs/rag_eval/colloquial/v2a_train_top20_traces_20260826.json"
PILOT_STYLES = ("natural", "implicit_oral", "long_noisy")


class CatalogBuildError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_source(case: dict[str, Any]) -> str:
    targets = case.get("relevant_targets") or []
    if not targets:
        raise CatalogBuildError(f"case {case.get('query_id')} has no relevant target")
    return str(targets[0].get("source") or "").strip()


def _select_anchors(cases: list[dict[str, Any]], traces: list[dict[str, Any]], *, count: int):
    case_by_query = {row["query_id"]: row for row in cases if row.get("split") == "train"}
    trace_by_query = {row["query_id"]: row for row in traces}
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id, case in case_by_query.items():
        if case.get("query_style") in PILOT_STYLES and query_id in trace_by_query:
            by_anchor[case["anchor_id"]].append(case)
    ranking = []
    for anchor, group in by_anchor.items():
        source = _target_source(group[0])
        false_winners = 0
        for case in group:
            reranked = (trace_by_query[case["query_id"]].get("stages") or {}).get("reranked") or []
            gold = {target["chunk_id"] for target in case.get("relevant_targets") or []}
            false_winners += int(bool(reranked) and reranked[0].get("chunk_id") not in gold)
        ranking.append((false_winners, source, anchor, group))
    ranking.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = []
    source_counts: Counter[str] = Counter()
    max_per_source = max(1, int(count * 0.15))
    for item in ranking:
        if source_counts[item[1]] >= max_per_source:
            continue
        selected.append(item)
        source_counts[item[1]] += 1
        if len(selected) == count:
            break
    # The frozen V2-A draft has only 59 anchors after a strict raw 15% cap.
    # The approved contract caps *effective training weight*, not raw rows, so
    # admit the smallest deterministic overflow and let the pair sampler cap
    # that source's gradient mass.  Do not silently lower source breadth.
    if len(selected) < count:
        selected_keys = {item[2] for item in selected}
        for item in ranking:
            if item[2] in selected_keys:
                continue
            selected.append(item)
            source_counts[item[1]] += 1
            selected_keys.add(item[2])
            if len(selected) == count:
                break
    if len(selected) != count:
        raise CatalogBuildError(
            f"only {len(selected)} anchors survive the 15% source cap; need {count}"
        )
    return selected, trace_by_query


def _validation_sources(selected, target_anchors: int) -> set[str]:
    anchors_by_source: dict[str, int] = Counter(item[1] for item in selected)
    # Pick whole, well-populated sources so validation can still contain five
    # hard candidates without borrowing text from a training source.
    ordered = sorted(anchors_by_source, key=lambda source: (
        -anchors_by_source[source], hashlib.sha256(source.encode()).hexdigest(),
    ))
    chosen: set[str] = set()
    count = 0
    for source in ordered:
        if count >= target_anchors:
            break
        chosen.add(source)
        count += anchors_by_source[source]
    return chosen


def _collection_documents(chunk_ids: list[str] | None = None) -> dict[str, tuple[str, dict[str, Any]]]:
    from day1_api_starter import load_local_env
    load_local_env()
    import chromadb
    from rag_tools import get_collection_name

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection(get_collection_name())
    if chunk_ids:
        result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    else:
        result = collection.get(include=["documents", "metadatas"])
    return {
        chunk_id: (document, metadata or {})
        for chunk_id, document, metadata in zip(
            result.get("ids") or [], result.get("documents") or [],
            result.get("metadatas") or [],
        )
    }


def _candidate_ids(case: dict[str, Any], trace: dict[str, Any], *,
                   allowed_sources: set[str],
                   fallback_by_source: dict[str, list[str]]) -> list[tuple[str, bool]]:
    stages = trace.get("stages") or {}
    reranked = stages.get("reranked") or []
    expanded_pool = []
    for stage in ("reranked", "fusion", "dense", "bm25"):
        expanded_pool.extend(stages.get(stage) or [])
    gold = [
        target["chunk_id"] for target in case.get("relevant_targets") or []
        if target.get("source") in allowed_sources
    ]
    hard = [
        item["chunk_id"] for item in case.get("hard_negatives") or []
        if item.get("source") in allowed_sources
    ]
    gold_sources = {target.get("source") for target in case.get("relevant_targets") or []}
    cross = [
        item["chunk_id"] for item in expanded_pool
        if item.get("source") in allowed_sources and item.get("source") not in gold_sources
    ]
    reranked = [item for item in reranked if item.get("source") in allowed_sources]
    expanded_pool = [
        item for item in expanded_pool if item.get("source") in allowed_sources
    ]
    ordered = []
    if reranked:
        ordered.append((reranked[0]["chunk_id"], reranked[0]["chunk_id"] in gold))
    ordered.extend((chunk_id, True) for chunk_id in gold)
    ordered.extend((chunk_id, False) for chunk_id in hard[:1])
    ordered.extend((chunk_id, False) for chunk_id in cross[:1])
    ordered.extend((item["chunk_id"], item["chunk_id"] in gold) for item in reranked[:10])
    ordered.extend((item["chunk_id"], item["chunk_id"] in gold) for item in expanded_pool)
    fallback_ids = [
        chunk_id for source in sorted(allowed_sources)
        for chunk_id in fallback_by_source.get(source, [])
    ]
    fallback_ids.sort(key=lambda chunk_id: hashlib.sha256(
        f"{case['query_id']}:{chunk_id}".encode("utf-8")
    ).hexdigest())
    ordered.extend((chunk_id, chunk_id in gold) for chunk_id in fallback_ids)
    deduped = []
    seen = set()
    for chunk_id, relevant in ordered:
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            deduped.append((chunk_id, relevant))
        if len(deduped) == 5:
            break
    if len(deduped) != 5:
        raise CatalogBuildError(f"cannot build five candidates for {case['query_id']}")
    return deduped


def build(cases_path: Path, traces_path: Path, *, anchors: int = 60):
    cases_payload = _load(cases_path)
    traces_payload = _load(traces_path)
    selected, trace_by_query = _select_anchors(
        cases_payload["items"], traces_payload["rows"], count=anchors,
    )
    validation_sources = _validation_sources(selected, target_anchors=15)
    documents = _collection_documents()
    fallback_by_source: dict[str, list[str]] = defaultdict(list)
    for chunk_id, (_document, metadata) in documents.items():
        source = str(metadata.get("source") or "")
        if source:
            fallback_by_source[source].append(chunk_id)
    query_specs = []
    all_ids = set()
    for _false_winners, source, _anchor, group in selected:
        split = "validation" if source in validation_sources else "train"
        for case in sorted(group, key=lambda row: PILOT_STYLES.index(row["query_style"])):
            trace = trace_by_query[case["query_id"]]
            allowed_sources = (
                validation_sources if split == "validation"
                else {_target_source(item[3][0]) for item in selected if item[1] not in validation_sources}
            )
            candidates = _candidate_ids(
                case, trace, allowed_sources=allowed_sources,
                fallback_by_source=fallback_by_source,
            )
            all_ids.update(chunk_id for chunk_id, _ in candidates)
            query_specs.append((case, trace, split, candidates))
    if len(query_specs) != anchors * 3:
        raise CatalogBuildError(f"expected {anchors * 3} queries, got {len(query_specs)}")
    missing = sorted(all_ids - set(documents))
    if missing:
        raise CatalogBuildError(f"{len(missing)} selected chunks are missing from current index")
    rows = []
    for case, trace, split, candidates in query_specs:
        reranked = {
            item["chunk_id"]: item
            for item in (trace.get("stages") or {}).get("reranked") or []
        }
        expanded = []
        for fallback_rank, (chunk_id, relevant) in enumerate(candidates, 1):
            if chunk_id not in documents:
                raise CatalogBuildError(f"chunk {chunk_id} is missing from current index")
            document, metadata = documents[chunk_id]
            trace_item = reranked.get(chunk_id) or {}
            expanded.append({
                "chunk_id": chunk_id,
                "source_id": str(
                    trace_item.get("source") or metadata.get("source") or "unknown"
                ),
                "chunk_text": document,
                "current_bge_rank": int(trace_item.get("rank") or (20 + fallback_rank)),
                "current_bge_score": trace_item.get("rerank_score"),
                "base_student_score": None,
                "known_relevant": bool(relevant),
            })
        rows.append({
            "schema_version": CATALOG_SCHEMA,
            "query_id": case["query_id"],
            "anchor_id": case["anchor_id"],
            "question": case["question"],
            "retrieval_question": case["question"],
            "query_style": case["query_style"],
            "split": split,
            "candidates": expanded,
            "provenance": {
                "dataset": cases_payload.get("dataset_id"),
                "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                "trace_sha256": hashlib.sha256(traces_path.read_bytes()).hexdigest(),
                "candidate_method": "current_top1_gold_same_source_cross_source_top6",
                "answer_requirements_visible_to_teacher": False,
            },
        })
    report = {
        "schema_version": "answerability-student-catalog-build-v1",
        "anchors": len({row["anchor_id"] for row in rows}),
        "queries": len(rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "sources": len({candidate["source_id"] for row in rows for candidate in row["candidates"]}),
        "splits": dict(Counter(row["split"] for row in rows)),
        "styles": dict(Counter(row["query_style"] for row in rows)),
        "catalog_sha256": canonical_sha256(rows),
        "answer_requirements_exported": False,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=60)
    args = parser.parse_args()
    rows, report = build(args.cases, args.traces, anchors=args.anchors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
