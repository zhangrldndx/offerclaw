#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build, edit, inspect, and validate OfferClaw's 400 colloquial RAG cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_colloquial_dataset import (  # noqa: E402
    apply_wording_edits,
    build_draft_dataset,
    parse_wording_batches,
    review_batch_markdown,
    review_summary,
    select_reviewed_split,
    split_public_private,
)
from rag_qrels_v2 import load_graded_qrels, validate_graded_qrels  # noqa: E402


PUBLIC_PATH = ROOT / "docs" / "rag_eval" / "colloquial" / "rag_colloquial_train_dev_v1.json"
MANIFEST_PATH = ROOT / "docs" / "rag_eval" / "colloquial" / "rag_colloquial_blind_v1.manifest.json"


def _canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_new_or_force(path: Path, data: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _private_root(value: str | None) -> Path:
    return Path(value or "~/.offerclaw/private_eval").expanduser().resolve()


def build(args: argparse.Namespace) -> None:
    import chromadb
    from rag_tools import get_collection_name, index_fingerprint

    private_root = _private_root(args.private_root)
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    index = index_fingerprint(collection=collection)
    payload = build_draft_dataset(ROOT, collection, index)
    public, blind = split_public_private(payload)
    blind_path = private_root / "rag_colloquial_blind_v1.json"
    public_bytes = _canonical_bytes(public)
    blind_bytes = _canonical_bytes(blind)
    _write_new_or_force(PUBLIC_PATH, public_bytes, force=args.force)
    _write_new_or_force(blind_path, blind_bytes, force=args.force)

    # Review batches are all private because the eight 50-row files include
    # sealed-blind questions.  The public train/dev JSON remains available in
    # the repository for normal development after review.
    all_items = sorted(payload["items"], key=lambda item: hashlib.sha256(
        item["query_id"].encode("utf-8")
    ).hexdigest())
    batch_dir = private_root / "rag_colloquial_review_batches_v1"
    for batch_index in range(8):
        batch = all_items[batch_index * 50:(batch_index + 1) * 50]
        batch_path = batch_dir / f"batch_{batch_index + 1:02d}_of_08.md"
        _write_new_or_force(
            batch_path,
            review_batch_markdown(batch, batch_index + 1).encode("utf-8"),
            force=args.force,
        )

    manifest = {
        "schema_version": "rag-colloquial-blind-manifest-v1",
        "dataset_id": blind["dataset_id"],
        "private_path_hint": "$OFFERCLAW_PRIVATE_EVAL_ROOT/rag_colloquial_blind_v1.json",
        "sha256": hashlib.sha256(blind_bytes).hexdigest(),
        "rows": len(blind["items"]),
        "positive_rows": sum(item["case_kind"] == "positive" for item in blind["items"]),
        "negative_rows": sum(item["case_kind"] == "negative" for item in blind["items"]),
        "index": blind["index"],
        "status": "draft_not_release_eligible",
        "questions_or_answers_committed": False,
    }
    _write_new_or_force(MANIFEST_PATH, _canonical_bytes(manifest), force=args.force)
    print(json.dumps({
        "public": str(PUBLIC_PATH),
        "private_blind": str(blind_path),
        "review_batches": str(batch_dir),
        "manifest": str(MANIFEST_PATH),
        "summary": review_summary(payload),
    }, ensure_ascii=False, indent=2))


def validate(args: argparse.Namespace) -> None:
    public = load_graded_qrels(PUBLIC_PATH, allowed_splits={"train", "dev"})
    private_root = _private_root(args.private_root)
    blind_path = private_root / "rag_colloquial_blind_v1.json"
    blind = load_graded_qrels(blind_path, allowed_splits={"blind"})
    combined = {**public, "items": [*public["items"], *blind["items"]]}
    combined["dataset_id"] = "rag-colloquial-400-v1-validation-view"
    validate_graded_qrels(combined, require_approved=args.require_approved)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(blind_path.read_bytes()).hexdigest()
    if actual_sha != manifest["sha256"]:
        raise SystemExit("private blind hash does not match repository manifest")
    print(json.dumps(review_summary(combined), ensure_ascii=False, indent=2))


def summary(args: argparse.Namespace) -> None:
    payload = load_graded_qrels(Path(args.path), require_approved=False)
    print(json.dumps(review_summary(payload), ensure_ascii=False, indent=2))


def apply_wording(args: argparse.Namespace) -> None:
    private_root = _private_root(args.private_root)
    blind_path = private_root / "rag_colloquial_blind_v1.json"
    batch_dir = private_root / "rag_colloquial_review_batches_v1"
    batch_paths = sorted(batch_dir.glob("batch_*_of_08.md"))
    if len(batch_paths) != 8:
        raise SystemExit(f"expected 8 review batches, found {len(batch_paths)}")
    questions = parse_wording_batches(batch_paths)
    if len(questions) != 400:
        raise SystemExit(f"wording coverage drift: expected 400 rows, found {len(questions)}")
    public = load_graded_qrels(PUBLIC_PATH, allowed_splits={"train", "dev"})
    blind = load_graded_qrels(blind_path, allowed_splits={"blind"})
    combined = {**public, "dataset_id": "rag-colloquial-400-v1-review-view",
                "items": [*public["items"], *blind["items"]]}
    combined, counts = apply_wording_edits(combined, questions)
    updated_public, updated_blind = split_public_private(combined)
    public_bytes = _canonical_bytes(updated_public)
    blind_bytes = _canonical_bytes(updated_blind)
    PUBLIC_PATH.write_bytes(public_bytes)
    blind_path.write_bytes(blind_bytes)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(blind_bytes).hexdigest()
    manifest["status"] = (
        "approved_sealed_blind_wording_applied"
        if all(item["review_status"] == "approved" for item in updated_blind["items"])
        else "gold_incomplete_not_release_eligible"
    )
    MANIFEST_PATH.write_bytes(_canonical_bytes(manifest))
    print(json.dumps({
        "wording_edits": counts,
        "summary": review_summary(combined),
        "blind_manifest_status": manifest["status"],
    }, ensure_ascii=False, indent=2))


def export_split(args: argparse.Namespace) -> None:
    """Export an explicit Train or Dev artifact after review."""

    public = load_graded_qrels(PUBLIC_PATH, allowed_splits={"train", "dev"})
    selected = select_reviewed_split(
        public,
        args.split,
        require_approved=not args.allow_draft,
    )
    output = Path(args.output).expanduser().resolve()
    _write_new_or_force(output, _canonical_bytes(selected), force=args.force)
    print(json.dumps({
        "output": str(output),
        "split": args.split,
        "release_eligible": not args.allow_draft,
        "summary": review_summary(selected),
    }, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build-draft")
    build_parser.add_argument("--private-root")
    build_parser.add_argument("--force", action="store_true")
    build_parser.set_defaults(func=build)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--private-root")
    validate_parser.add_argument("--require-approved", action="store_true")
    validate_parser.set_defaults(func=validate)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("path")
    summary_parser.set_defaults(func=summary)
    apply_parser = sub.add_parser(
        "apply-wording",
        help="apply only the user-edited question lines from all eight batches",
    )
    apply_parser.add_argument("--private-root")
    apply_parser.set_defaults(func=apply_wording)
    export_parser = sub.add_parser("export-split")
    export_parser.add_argument("--split", choices=("train", "dev"), required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--allow-draft", action="store_true")
    export_parser.add_argument("--force", action="store_true")
    export_parser.set_defaults(func=export_split)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
