#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify reviewer-approved direct evidence missing from Candidate@20.

This is a read-only Stage-D diagnostic.  It never rebuilds or mutates a
collection and never calls the cross encoder.  The output separates channel
depth/fusion loss from current-chunk visibility and metadata problems before a
new chunking experiment is allowed to run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SET = ROOT / "tests" / "rag_bench_paraphrase_set.json"
DEFAULT_QRELS = ROOT / "docs" / "rag_eval" / "qrels" / "rag_bench_paraphrase_adjudicated.json"
DEFAULT_STAGE_C = ROOT / "docs" / "rag_eval" / "r1_upgrade" / "stage_c_candidate_pools.json"
DEFAULT_TOKENIZER = (
    Path.home() / ".cache" / "modelscope" / "hub" / "models"
    / "BAAI" / "bge-base-zh-v1.5"
)


def _baseline_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report = next(
        item for item in payload["strategies"]
        if item["strategy"] == "baseline_rrf20"
    )
    return {row["id"]: row for row in report["rows"]}


def _rank(candidates, target_ids: set[str]) -> int:
    return next(
        (index for index, candidate in enumerate(candidates, start=1)
         if candidate.chunk_id in target_ids),
        0,
    )


def _token_offsets(tokenizer, document: str, excerpt: str) -> tuple[int, int] | None:
    position = document.find(excerpt)
    if position < 0:
        probe = excerpt.strip()[:80]
        position = document.find(probe) if probe else -1
    if position < 0:
        return None
    start = len(tokenizer.encode(document[:position], add_special_tokens=False))
    end = len(tokenizer.encode(
        document[:position + len(excerpt)], add_special_tokens=False,
    ))
    return start, end


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb
    from transformers import AutoTokenizer

    from eval_candidate_pools import (
        _allowed_reference_meta, _load_items, _load_qrels,
        read_candidate_channels,
    )
    from rag_tools import get_collection_name, index_fingerprint

    items = _load_items(args.set)
    item_by_id = {str(item["id"]): item for item in items}
    stage_c = json.loads(args.stage_c.read_text(encoding="utf-8"))
    baseline = _baseline_rows(stage_c)
    missing_ids = sorted(
        query_id for query_id, row in baseline.items()
        if not row["direct_candidate_hit"]
    )
    missing_items = [item_by_id[query_id] for query_id in missing_ids]

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    direct_qrels, qrels_payload = _load_qrels(
        args.qrels, [str(item["id"]) for item in items], collection,
    )
    qrels_by_id = {item["query_id"]: item for item in qrels_payload["items"]}
    channels = read_candidate_channels(
        missing_items, collection, channel_depth=args.channel_depth,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True,
    )

    target_ids = list(dict.fromkeys(
        target["chunk_id"]
        for query_id in missing_ids
        for target in qrels_by_id[query_id]["relevant_targets"]
        if target["relevance"] == "direct"
    ))
    fetched = collection.get(
        ids=target_ids, include=["documents", "metadatas"],
    )
    target_content = {
        chunk_id: (document, metadata or {})
        for chunk_id, document, metadata in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"],
        )
    }

    rows = []
    for query_id in missing_ids:
        dense, bm25 = channels[query_id]
        approved = [
            target for target in qrels_by_id[query_id]["relevant_targets"]
            if target["relevance"] == "direct"
        ]
        approved_ids = {target["chunk_id"] for target in approved}
        dense_rank = _rank(dense, approved_ids)
        bm25_rank = _rank(bm25, approved_ids)
        targets = []
        query_flags: set[str] = set()
        for target in approved:
            document, metadata = target_content[target["chunk_id"]]
            tokens = len(tokenizer.encode(document, add_special_tokens=False))
            offsets = _token_offsets(
                tokenizer, document, str(target.get("evidence_excerpt") or ""),
            )
            heading_path = [str(value) for value in target.get("heading_path") or []]
            indexed_title = str(metadata.get("title") or "")
            generic_title = indexed_title in {"", "正文", "正文采集", "内容", "__header__"}
            metadata_allowed = _allowed_reference_meta(metadata)
            flags = []
            if not metadata_allowed:
                flags.append("metadata_filter_excluded")
            if tokens > args.visible_tokens:
                flags.append("chunk_exceeds_visible_window")
            if offsets and offsets[1] > args.visible_tokens:
                flags.append("answer_span_beyond_visible_window")
            if generic_title or (len(heading_path) > 1 and indexed_title not in heading_path[-2:]):
                flags.append("breadcrumb_not_indexed")
            if not offsets:
                flags.append("evidence_excerpt_not_located")
            query_flags.update(flags)
            targets.append({
                "chunk_id": target["chunk_id"],
                "source": target["source"],
                "heading_path": heading_path,
                "indexed_title": indexed_title,
                "document_chars": len(document),
                "document_tokens": tokens,
                "answer_token_offsets": list(offsets) if offsets else None,
                "metadata_allowed": metadata_allowed,
                "flags": flags,
            })
        if dense_rank > 20 or bm25_rank > 20:
            query_flags.add("present_deeper_than_20")
        if not dense_rank and not bm25_rank:
            query_flags.add("absent_from_both_channels_at_depth")
        rows.append({
            "id": query_id,
            "question": item_by_id[query_id]["q"],
            "dense_rank_at_depth": dense_rank,
            "bm25_rank_at_depth": bm25_rank,
            "channel_depth": args.channel_depth,
            "flags": sorted(query_flags),
            "targets": targets,
        })

    counts = Counter(flag for row in rows for flag in row["flags"])
    structural = {
        row["id"] for row in rows
        if any(flag in row["flags"] for flag in (
            "chunk_exceeds_visible_window",
            "answer_span_beyond_visible_window",
            "breadcrumb_not_indexed",
        ))
    }
    return {
        "schema_version": "candidate-miss-audit-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "read_only": True,
        "set": str(args.set),
        "qrels": str(args.qrels),
        "index": index_fingerprint(cache_ttl=0),
        "candidate_depth": args.channel_depth,
        "visible_tokens": args.visible_tokens,
        "missing_query_count": len(rows),
        "structural_chunking_candidate_count": len(structural),
        "structural_chunking_candidate_ids": sorted(structural),
        "flag_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--stage-c", type=Path, default=DEFAULT_STAGE_C)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--channel-depth", type=int, default=100)
    parser.add_argument("--visible-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.channel_depth < 20 or args.visible_tokens < 64:
        parser.error("channel depth must be >=20 and visible tokens >=64")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "missing_query_count": result["missing_query_count"],
        "structural_chunking_candidate_count": result[
            "structural_chunking_candidate_count"
        ],
        "flag_counts": result["flag_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
