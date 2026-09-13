#!/usr/bin/env python3
"""Refine only Train candidates after the answerability Pilot data gate fails.

The first hard-mining pass is label-blind.  Once its development labels exist,
this pass may use them to find Train anchors that still have no *real*
MiniLM inversion.  Validation is byte-for-byte untouched.  For selected Train
queries it replaces low-value candidates with locally scored chunks from
sources missing from the corrective-pair coverage.  It never calls a teacher.

The output remains a 180-query/five-candidate catalog, so the next resumable
dataset build reuses unchanged labels and calls the teacher only for replaced
candidates (plus any previously unavailable rows).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability_student import DEFAULT_MODEL_NAME
from rag_answerability_student_data import build_strong_pairs, read_jsonl
from scripts.build_answerability_student_catalog import _collection_documents
from scripts.build_answerability_student_dataset import _load_catalog
from scripts.remine_answerability_student_catalog import _load_cross_encoder


def _predict(model, is_onnx: bool, pairs: list[list[str]]) -> list[float]:
    if not pairs:
        return []
    if is_onnx:
        values = model.predict(pairs, batch_size=32)
    else:
        values = model.predict(pairs, batch_size=8, show_progress_bar=True)
    return [float(value) for value in values]


def _candidate_from_document(
    chunk_id: str, text: str, source: str, score: float,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_id": source,
        "chunk_text": text,
        "known_relevant": False,
        "current_bge_rank": 1000,
        "current_bge_score": None,
        "base_student_score": score,
    }


def refine(
    catalog_path: Path,
    dataset_path: Path,
    model_name: str,
    *,
    max_anchors: int = 8,
    replacements_per_query: int = 2,
    include_corrective_anchors: bool = False,
    restrict_pool_to_missing_sources: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = _load_catalog(catalog_path)
    rows = read_jsonl(dataset_path)
    pairs = build_strong_pairs(rows)
    corrective = [pair for pair in pairs if pair["corrective"]]
    corrective_anchors = {pair["anchor_id"] for pair in corrective}
    corrective_sources = {
        source for pair in corrective
        for source in (pair["positive"]["source_id"], pair["negative"]["source_id"])
    }
    train_sources = {
        candidate["source_id"]
        for query in queries if query["split"] == "train"
        for candidate in query["candidates"]
    }
    missing_sources = train_sources - corrective_sources
    if not missing_sources:
        return queries, {
            "schema_version": "answerability-student-pilot-refine-v1",
            "status": "no_refinement_needed",
            "teacher_calls": 0,
        }

    row_by_key = {(row["query_id"], row["chunk_id"]): row for row in rows}
    documents = _collection_documents()
    pool_sources = missing_sources if restrict_pool_to_missing_sources else train_sources
    document_pool = [
        (chunk_id, text, str(metadata.get("source") or ""))
        for chunk_id, (text, metadata) in documents.items()
        if str(metadata.get("source") or "") in pool_sources
    ]
    model, is_onnx = _load_cross_encoder(model_name)

    # One plan per anchor is enough for the Pilot coverage gate.  Candidate
    # selection uses only Train labels and the frozen, untrained local model.
    eligible_by_anchor: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for query in queries:
        if query["split"] != "train" or (
            not include_corrective_anchors
            and query["anchor_id"] in corrective_anchors
        ):
            continue
        labeled = [
            row_by_key.get((query["query_id"], candidate["chunk_id"]))
            for candidate in query["candidates"]
        ]
        if any(row is None for row in labeled):
            continue
        highs = [
            row for row in labeled
            if row["teacher_grade"] == 3
            and row.get("label_status") != "teacher_disagreement"
        ]
        if not highs:
            continue
        # A lower-scored true answer gives the widest honest opportunity for
        # finding a current false-winner; it is never relabelled or fabricated.
        high = min(highs, key=lambda row: float(row["base_student_score"]))
        eligible_by_anchor[query["anchor_id"]].append((query, high))

    # Score one frozen wording per anchor, selected only by the base student's
    # score on an already labelled grade-3 answer.  This bounds local work and
    # cannot look at Validation or future teacher labels.
    anchor_queries = [
        min(items, key=lambda item: float(item[1]["base_student_score"]))
        for _anchor, items in sorted(eligible_by_anchor.items())
    ]
    plans: list[dict[str, Any]] = []
    for query, high in anchor_queries:
        existing = {candidate["chunk_id"] for candidate in query["candidates"]}
        available = [item for item in document_pool if item[0] not in existing]
        scores = _predict(
            model, is_onnx,
            [[query["question"], text] for _chunk, text, _source in available],
        )
        options = [
            _candidate_from_document(chunk_id, text, source, score)
            for (chunk_id, text, source), score in zip(available, scores)
            if score >= float(high["base_student_score"])
        ]
        options.sort(
            key=lambda item: (
                item["source_id"] in missing_sources,
                item["base_student_score"],
            ),
            reverse=True,
        )
        selected = []
        seen_sources = set()
        for option in options:
            if option["source_id"] not in seen_sources:
                selected.append(option)
                seen_sources.add(option["source_id"])
            if len(selected) >= replacements_per_query:
                break
        if len(selected) < replacements_per_query:
            for option in options:
                if option["chunk_id"] not in {item["chunk_id"] for item in selected}:
                    selected.append(option)
                if len(selected) >= replacements_per_query:
                    break
        if selected:
            plans.append({
                "query": query,
                "high": high,
                "options": selected,
                "sources": (
                    {item["source_id"] for item in selected} | {high["source_id"]}
                ) & missing_sources,
                "max_margin": max(
                    item["base_student_score"] - float(high["base_student_score"])
                    for item in selected
                ),
            })

    chosen = []
    remaining = list(plans)
    covered_new_sources: set[str] = set()
    used_anchors: set[str] = set()
    while remaining and len(chosen) < max_anchors:
        eligible = [
            plan for plan in remaining
            if plan["query"]["anchor_id"] not in used_anchors
        ]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda plan: (
                len(plan["sources"] - covered_new_sources),
                len(plan["options"]),
                plan["max_margin"],
                plan["query"]["query_id"],
            ),
        )
        chosen.append(best)
        used_anchors.add(best["query"]["anchor_id"])
        covered_new_sources.update(best["sources"])
        remaining.remove(best)

    by_query = {plan["query"]["query_id"]: plan for plan in chosen}
    output = []
    replacements = 0
    for original in queries:
        plan = by_query.get(original["query_id"])
        if plan is None:
            output.append(original)
            continue
        replacement_count = min(len(plan["options"]), replacements_per_query)
        high_chunk = plan["high"]["chunk_id"]
        incumbent = min(original["candidates"], key=lambda item: item["current_bge_rank"])
        protected = {high_chunk, incumbent["chunk_id"]}
        removable = sorted(
            [candidate for candidate in original["candidates"] if candidate["chunk_id"] not in protected],
            key=lambda candidate: (
                row_by_key[(original["query_id"], candidate["chunk_id"])]["teacher_grade"] == 3,
                (
                    float(candidate["base_student_score"])
                    if isinstance(candidate.get("base_student_score"), (int, float))
                    and not isinstance(candidate.get("base_student_score"), bool)
                    else float("-inf")
                ),
            ),
        )
        remove_ids = {item["chunk_id"] for item in removable[:replacement_count]}
        kept = [item for item in original["candidates"] if item["chunk_id"] not in remove_ids]
        candidates = kept + plan["options"][:replacement_count]
        if len(candidates) != 5 or len({item["chunk_id"] for item in candidates}) != 5:
            raise ValueError(f"refinement broke five-candidate contract: {original['query_id']}")
        updated = dict(original)
        updated["candidates"] = candidates
        updated["provenance"] = {
            **(original.get("provenance") or {}),
            "pilot_refine": "train_label_guided_local_minilm_v1",
            "pilot_refine_teacher_calls": 0,
            "validation_modified": False,
        }
        output.append(updated)
        replacements += replacement_count

    report = {
        "schema_version": "answerability-student-pilot-refine-v1",
        "status": "refined_train_only" if replacements else "no_viable_refinement",
        "teacher_calls": 0,
        "input_queries": len(queries),
        "output_queries": len(output),
        "candidates_per_query": 5,
        "validation_rows_modified": 0,
        "corrective_anchors_before": len(corrective_anchors),
        "corrective_sources_before": len(corrective_sources),
        "missing_sources_before": len(missing_sources),
        "target_anchors": len(used_anchors),
        "target_sources": len(covered_new_sources),
        "candidate_replacements_requiring_teacher": replacements,
        "include_corrective_anchors": include_corrective_anchors,
        "restrict_pool_to_missing_sources": restrict_pool_to_missing_sources,
        "base_model": model_name,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-anchors", type=int, default=8)
    parser.add_argument("--replacements-per-query", type=int, default=2)
    parser.add_argument("--include-corrective-anchors", action="store_true")
    parser.add_argument("--restrict-pool-to-missing-sources", action="store_true")
    args = parser.parse_args()
    rows, report = refine(
        args.catalog, args.dataset, args.model,
        max_anchors=args.max_anchors,
        replacements_per_query=args.replacements_per_query,
        include_corrective_anchors=args.include_corrective_anchors,
        restrict_pool_to_missing_sources=args.restrict_pool_to_missing_sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
