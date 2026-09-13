#!/usr/bin/env python3
"""Export approved Train/Dev qrels as leak-safe CrossEncoder triples.

The private Blind80 file is deliberately unsupported.  This exporter accepts
only the public ``train`` and ``dev`` splits, resolves chunk text from the
frozen local Chroma index, and formats documents exactly like the selected
production-mirror reranker arm.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "colloquial-reranker-training-v1"
DEFAULT_CASES = (
    ROOT / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json"
)


class ColloquialTrainingBuildError(ValueError):
    """Raised when an export would violate the frozen-data contract."""


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_public_cases(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    lowered = str(resolved).lower()
    if "private_eval" in lowered or "blind" in resolved.name.lower():
        raise ColloquialTrainingBuildError("private Blind data is forbidden")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ColloquialTrainingBuildError("cases file contains no items")
    splits = {str(item.get("split") or "") for item in items}
    if not splits <= {"train", "dev"} or not {"train", "dev"} <= splits:
        raise ColloquialTrainingBuildError(
            f"training export requires exactly public train/dev splits, got {sorted(splits)}"
        )
    if any(item.get("review_status") != "approved" for item in items):
        raise ColloquialTrainingBuildError("all training cases must be approved")
    return payload


def _load_chunks(collection: Any, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
    snapshot = collection.get(
        ids=sorted(chunk_ids), include=["documents", "metadatas"],
    )
    result = {
        str(chunk_id): {
            "document": str(document or ""),
            "metadata": dict(metadata or {}),
        }
        for chunk_id, document, metadata in zip(
            snapshot.get("ids") or [],
            snapshot.get("documents") or [],
            snapshot.get("metadatas") or [],
        )
    }
    missing = sorted(chunk_ids - set(result))
    if missing:
        raise ColloquialTrainingBuildError(
            f"frozen index is missing {len(missing)} referenced chunks: {missing[:5]}"
        )
    return result


def compact32_scoring_text(document: str, metadata: dict[str, Any]) -> str:
    """Use the exact compact32 formatter of the current winning Dev80 arm."""

    from rag_gate import _rerank_pair_documents

    values = _rerank_pair_documents(
        [document], [metadata], use_breadcrumb=False, prefix_mode="compact32",
    )
    if not values or len(values) != 1:
        raise ColloquialTrainingBuildError("compact32 formatter returned no text")
    return values[0]


def _chunk_ref(
    chunk_id: str,
    chunks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = chunks[chunk_id]
    metadata = row["metadata"]
    source = Path(str(metadata.get("source") or "")).name
    heading = str(
        metadata.get("heading_path") or metadata.get("section_path")
        or metadata.get("title") or ""
    )
    return {
        "chunk_id": chunk_id,
        "source": source,
        "heading_path": heading,
        "document_hash": _sha256_text(row["document"]),
        "scoring_text": compact32_scoring_text(row["document"], metadata),
    }


def _negative_kind(positive: dict[str, Any], negative: dict[str, Any]) -> str:
    if positive["source"] and positive["source"] == negative["source"]:
        return "same_document_wrong_section"
    return "adjudicated_hard_negative"


def build_training_artifact(
    *,
    cases_path: str | Path,
    collection: Any,
) -> dict[str, Any]:
    path = Path(cases_path).expanduser().resolve()
    payload = _load_public_cases(path)
    positive_items = [
        item for item in payload["items"] if item.get("case_kind") == "positive"
    ]
    if not positive_items:
        raise ColloquialTrainingBuildError("no positive cases available for reranker training")

    referenced: set[str] = set()
    for item in positive_items:
        referenced.update(
            str(target["chunk_id"])
            for target in item.get("relevant_targets") or []
            if int(target.get("relevance_grade") or 0) == 3
        )
        referenced.update(
            str(negative["chunk_id"])
            for negative in item.get("hard_negatives") or []
        )
    chunks = _load_chunks(collection, referenced)

    triples: list[dict[str, Any]] = []
    exclusions = Counter()
    for item in positive_items:
        positives = [
            target for target in item.get("relevant_targets") or []
            if int(target.get("relevance_grade") or 0) == 3
        ]
        negatives = list(item.get("hard_negatives") or [])
        if not positives:
            exclusions["no_grade3_positive"] += 1
            continue
        if not negatives:
            # Unjudged retrieval candidates are not silently treated as
            # negatives.  False-negative labels are more damaging than a few
            # fewer training pairs on this small audited corpus.
            exclusions["no_adjudicated_hard_negative"] += 1
            continue
        for positive_target in positives:
            positive = _chunk_ref(str(positive_target["chunk_id"]), chunks)
            for negative_target in negatives:
                negative = _chunk_ref(str(negative_target["chunk_id"]), chunks)
                if positive["chunk_id"] == negative["chunk_id"]:
                    raise ColloquialTrainingBuildError(
                        f"{item['query_id']}: positive equals hard negative"
                    )
                triples.append({
                    "query_id": str(item["query_id"]),
                    "anchor_id": str(item["anchor_id"]),
                    "split": str(item["split"]),
                    "query_style": str(item.get("query_style") or ""),
                    "query": str(item["question"]),
                    "negative_kind": _negative_kind(positive, negative),
                    "positive": positive,
                    "negative": negative,
                })

    train_anchors = {
        row["anchor_id"] for row in triples if row["split"] == "train"
    }
    dev_anchors = {row["anchor_id"] for row in triples if row["split"] == "dev"}
    overlap = sorted(train_anchors & dev_anchors)
    if overlap:
        raise ColloquialTrainingBuildError(
            f"anchor leakage between train/dev: {overlap[:5]}"
        )
    if not any(row["split"] == "train" for row in triples):
        raise ColloquialTrainingBuildError("training split produced no triples")
    if not any(row["split"] == "dev" for row in triples):
        raise ColloquialTrainingBuildError("development split produced no triples")

    index = dict(payload.get("index") or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "contains_text": True,
        "private_blind_set": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_set": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name),
        "source_dataset_id": payload.get("dataset_id"),
        "provenance": {
            "cases_sha256": _sha256_file(path),
            "index": index,
            "scoring_text_formatter": "rag_gate._rerank_pair_documents(prefix_mode=compact32)",
            "blind_set_used": False,
        },
        "split_contract": {
            "method": "preassigned_source_group_and_anchor_split",
            "train_anchor_ids": sorted(train_anchors),
            "dev_anchor_ids": sorted(dev_anchors),
        },
        "summary": {
            "triple_count": len(triples),
            "train_triple_count": sum(row["split"] == "train" for row in triples),
            "dev_triple_count": sum(row["split"] == "dev" for row in triples),
            "train_anchor_count": len(train_anchors),
            "dev_anchor_count": len(dev_anchors),
            "by_style": dict(sorted(Counter(row["query_style"] for row in triples).items())),
            "by_negative_kind": dict(sorted(Counter(
                row["negative_kind"] for row in triples
            ).items())),
            "excluded_cases": dict(sorted(exclusions.items())),
        },
        "triples": triples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import chromadb
    from rag_tools import get_collection_name

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    artifact = build_training_artifact(
        cases_path=args.cases, collection=collection,
    )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "schema_version": artifact["schema_version"],
        "source_dataset_id": artifact["source_dataset_id"],
        **artifact["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
