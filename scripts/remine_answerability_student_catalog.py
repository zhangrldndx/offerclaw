#!/usr/bin/env python3
"""Re-mine the five-candidate pilot head using only local MiniLM scores.

The first deterministic catalog may have enough inverted pairs but too little
source coverage.  This pass scores a wider frozen candidate pool, then keeps
the BGE incumbent, gold, hardest same-source, hardest cross-source and one
source-diversifying inverted candidate.  No teacher labels are consumed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_answerability_student import DEFAULT_MODEL_NAME
from scripts.build_answerability_student_catalog import (
    DEFAULT_CASES, DEFAULT_TRACES, _collection_documents,
)
from scripts.build_answerability_student_dataset import _load_catalog


def _load_cross_encoder(model_name: str):
    path = Path(model_name).expanduser()
    cached = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_name
    resolved = cached if cached.is_dir() else path if path.is_dir() else model_name
    resolved_path = Path(resolved) if isinstance(resolved, (str, Path)) else Path()
    if resolved_path.is_dir() and (resolved_path / "onnx/model_qint8_arm64.onnx").is_file():
        from rag_rerank_onnx import OnnxCrossEncoder
        model = OnnxCrossEncoder(str(resolved_path))
        model.max_seq_length = 384
        return model, True
    from sentence_transformers import CrossEncoder
    return CrossEncoder(
        str(resolved), device=os.environ.get("OFFERCLAW_TORCH_DEVICE") or None,
        max_length=384,
    ), False


def _pool(query, case, trace, documents, allowed_sources):
    gold = {target["chunk_id"] for target in case.get("relevant_targets") or []}
    trace_items = {}
    stage_limits = {"reranked": 12, "fusion": 5, "dense": 5, "bm25": 5}
    for stage, limit in stage_limits.items():
        for item in ((trace.get("stages") or {}).get(stage) or [])[:limit]:
            trace_items.setdefault(item["chunk_id"], item)
    ids = set(gold)
    ids.update(candidate["chunk_id"] for candidate in query["candidates"])
    ids.update(
        item["chunk_id"] for item in case.get("hard_negatives") or []
        if item.get("source") in allowed_sources
    )
    ids.update(
        chunk_id for chunk_id, item in trace_items.items()
        if item.get("source") in allowed_sources
    )
    # Add a deterministic local neighbourhood from every allowed source.  It
    # is scored before selection, so arbitrary filler cannot masquerade as a
    # hard pair merely because it was sampled.
    by_source = defaultdict(list)
    for chunk_id, (_text, metadata) in documents.items():
        source = str(metadata.get("source") or "")
        if source in allowed_sources:
            by_source[source].append(chunk_id)
    gold_sources = {
        target.get("source") for target in case.get("relevant_targets") or []
    }
    extra_sources = sorted(
        allowed_sources - gold_sources,
        key=lambda source: hashlib.sha256(
            f"{query['query_id']}:{source}".encode("utf-8")
        ).hexdigest(),
    )[:4]
    for source in sorted(gold_sources | set(extra_sources)):
        ranked = sorted(by_source[source], key=lambda chunk_id: hashlib.sha256(
            f"{query['query_id']}:{chunk_id}".encode("utf-8")
        ).hexdigest())
        ids.update(ranked[:2])
    output = []
    for chunk_id in ids:
        if chunk_id not in documents:
            continue
        text, metadata = documents[chunk_id]
        item = trace_items.get(chunk_id) or {}
        source = str(item.get("source") or metadata.get("source") or "")
        if source not in allowed_sources:
            continue
        output.append({
            "chunk_id": chunk_id, "source_id": source, "chunk_text": text,
            "known_relevant": chunk_id in gold,
            "current_bge_rank": int(item.get("rank") or 1000),
            "current_bge_score": item.get("rerank_score"),
        })
    return output


def remine(catalog_path: Path, cases_path: Path, traces_path: Path, model_name: str):
    queries = _load_catalog(catalog_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["items"]
    traces = json.loads(traces_path.read_text(encoding="utf-8"))["rows"]
    case_by_query = {row["query_id"]: row for row in cases}
    trace_by_query = {row["query_id"]: row for row in traces}
    documents = _collection_documents()
    sources_by_split = {
        split: {
            candidate["source_id"] for row in queries if row["split"] == split
            for candidate in row["candidates"]
        } for split in ("train", "validation")
    }
    pools = []
    flat = []
    for query in queries:
        pool = _pool(
            query, case_by_query[query["query_id"]], trace_by_query[query["query_id"]],
            documents, sources_by_split[query["split"]],
        )
        pools.append(pool)
        flat.extend((query, candidate) for candidate in pool)
    model, is_onnx = _load_cross_encoder(model_name)
    pairs_to_score = [
        [query["question"], candidate["chunk_text"]] for query, candidate in flat
    ]
    if is_onnx:
        scores = model.predict(pairs_to_score, batch_size=32)
    else:
        scores = model.predict(
            pairs_to_score, batch_size=8, show_progress_bar=True,
        )
    for (_query, candidate), score in zip(flat, scores):
        candidate["base_student_score"] = float(score)

    # First choose the fixed four roles, then use the fifth slot to maximise
    # source coverage among candidates that actually outrank gold.
    preliminary = []
    inverted_options = []
    for query, pool in zip(queries, pools):
        golds = [item for item in pool if item["known_relevant"]]
        if not golds:
            raise ValueError(f"missing gold in pool for {query['query_id']}")
        gold = max(golds, key=lambda item: item["base_student_score"])
        incumbent = min(pool, key=lambda item: item["current_bge_rank"])
        same = [
            item for item in pool if not item["known_relevant"]
            and item["source_id"] == gold["source_id"]
        ]
        cross = [
            item for item in pool if not item["known_relevant"]
            and item["source_id"] != gold["source_id"]
        ]
        chosen = [incumbent, gold]
        if same:
            chosen.append(max(same, key=lambda item: item["base_student_score"]))
        if cross:
            chosen.append(max(cross, key=lambda item: item["base_student_score"]))
        deduped = []
        seen = set()
        for item in chosen:
            if item["chunk_id"] not in seen:
                seen.add(item["chunk_id"])
                deduped.append(item)
        options = sorted(
            [
                item for item in pool if item["chunk_id"] not in seen
                and not item["known_relevant"]
                and item["base_student_score"] >= gold["base_student_score"]
            ],
            key=lambda item: -item["base_student_score"],
        )
        preliminary.append((query, deduped, pool))
        inverted_options.append(options)

    covered = set()
    for _query, chosen, unused_pool in preliminary:
        gold_score = max(
            item["base_student_score"] for item in chosen if item["known_relevant"]
        )
        for item in chosen:
            if not item["known_relevant"] and item["base_student_score"] >= gold_score:
                covered.add(item["source_id"])
                covered.update(
                    gold["source_id"] for gold in chosen if gold["known_relevant"]
                )
    output = []
    for (query, chosen, pool), options in zip(preliminary, inverted_options):
        while len(chosen) < 5:
            diverse = next((item for item in options if item["source_id"] not in covered
                            and item["chunk_id"] not in {row["chunk_id"] for row in chosen}), None)
            remaining = [
                item for item in pool
                if item["chunk_id"] not in {row["chunk_id"] for row in chosen}
            ]
            pick = diverse or (max(remaining, key=lambda item: item["base_student_score"])
                               if remaining else None)
            if pick is None:
                raise ValueError(f"cannot fill five candidates for {query['query_id']}")
            chosen.append(pick)
            covered.add(pick["source_id"])
        query = dict(query)
        query["candidates"] = chosen[:5]
        query["provenance"] = {
            **(query.get("provenance") or {}),
            "hard_remine": "local_minilm_wide_pool_v1",
            "hard_remine_teacher_calls": 0,
        }
        output.append(query)

    inverted = anchors = sources = set()
    inverted_count = total = 0
    anchor_set = set()
    source_set = set()
    for query in output:
        pos = [item for item in query["candidates"] if item["known_relevant"]]
        neg = [item for item in query["candidates"] if not item["known_relevant"]]
        for high in pos:
            for low in neg:
                total += 1
                if high["base_student_score"] <= low["base_student_score"]:
                    inverted_count += 1
                    anchor_set.add(query["anchor_id"])
                    source_set.update((high["source_id"], low["source_id"]))
    report = {
        "schema_version": "answerability-student-hard-remine-v1",
        "queries": len(output), "candidates": len(output) * 5,
        "wide_pool_pairs_scored": len(flat),
        "provisional_pairs": total, "provisional_inverted": inverted_count,
        "provisional_inversion_rate": inverted_count / total if total else 0.0,
        "anchors_with_inversion": len(anchor_set),
        "sources_in_inversions": len(source_set),
        "teacher_calls": 0,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    rows, report = remine(args.catalog, args.cases, args.traces, args.model)
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
