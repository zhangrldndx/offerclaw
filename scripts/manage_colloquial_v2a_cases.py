#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build, validate, and word-edit OfferClaw's V2-A colloquial anchor wave.

Anchor specs (including sealed anchors) live outside the repository in the
private staging directory; the repository stores only the public Train/Dev-New
draft, the sealed manifest, and this tooling.  ``build`` is idempotent: the
same specs and index produce byte-identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_colloquial_v2a import (  # noqa: E402
    apply_v2a_wording,
    build_v2a_dataset,
    collection_rows,
    compute_v1_exclusions,
    parse_v2a_wording,
    split_v2a_public_sealed,
    v2a_review_batches,
    v2a_summary,
)
from rag_qrels_v2 import load_graded_qrels, validate_graded_qrels  # noqa: E402


PUBLIC_PATH = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_train_dev_draft.json"
MANIFEST_PATH = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_sealed.manifest.json"


def _canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _private_root(value: str | None) -> Path:
    return Path(value or "~/.offerclaw/private_eval").expanduser().resolve()


def _spec_paths(specs_dir: Path) -> list[Path]:
    paths = sorted(specs_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no anchor spec files found in {specs_dir}")
    return paths


def _load_specs(specs_dir: Path) -> list[dict]:
    specs: list[dict] = []
    for path in _spec_paths(specs_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        anchors = payload.get("anchors") if isinstance(payload, dict) else payload
        if not isinstance(anchors, list):
            raise SystemExit(f"{path}: spec file must contain an 'anchors' list")
        specs.extend(anchors)
    return specs


def _manifest(sealed_bytes: bytes, sealed: dict, status: str) -> dict:
    return {
        "schema_version": "rag-colloquial-v2a-sealed-manifest-v1",
        "dataset_id": sealed["dataset_id"],
        "private_path_hint": "$OFFERCLAW_PRIVATE_EVAL_ROOT/rag_colloquial_v2a_sealed.json",
        "sha256": hashlib.sha256(sealed_bytes).hexdigest(),
        "rows": len(sealed["items"]),
        "anchors": len({item["anchor_id"] for item in sealed["items"]}),
        "index": sealed["index"],
        "status": status,
        "sealed_note": sealed.get("sealed_note", ""),
        "questions_or_answers_committed": False,
    }


def build(args: argparse.Namespace) -> None:
    import chromadb
    from rag_tools import get_collection_name, index_fingerprint

    # ``build`` regenerates questions from the specs, which still hold the
    # pre-review wording.  Running it after apply-wording would silently throw
    # away every human edit, so an approved dataset has to be unlocked first.
    if PUBLIC_PATH.exists() and not args.force:
        current = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
        if any(item.get("review_status") == "approved"
               for item in current.get("items", [])):
            raise SystemExit(
                f"{PUBLIC_PATH.name} already carries approved (human-edited) "
                "wording; rebuilding would discard it. Pass --force if that is "
                "really what you want."
            )
    private_root = _private_root(args.private_root)
    specs_dir = Path(args.specs_dir).expanduser().resolve() if args.specs_dir \
        else private_root / "rag_colloquial_v2a_specs"
    specs = _load_specs(specs_dir)
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    rows = collection_rows(collection)
    exclusions = compute_v1_exclusions(ROOT, rows)
    index = index_fingerprint(collection=collection)
    targets = None
    if args.split_targets:
        train, dev, sealed_count = (int(part) for part in args.split_targets.split(","))
        targets = {"train": train, "dev": dev, "blind": sealed_count}
    payload = build_v2a_dataset(specs, rows, exclusions, index, split_targets=targets)
    public, sealed = split_v2a_public_sealed(payload)

    sealed_path = private_root / "rag_colloquial_v2a_sealed.json"
    batch_dir = private_root / "rag_colloquial_v2a_review_batches"
    public_bytes = _canonical_bytes(public)
    sealed_bytes = _canonical_bytes(sealed)
    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.write_bytes(public_bytes)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_bytes(sealed_bytes)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches = v2a_review_batches(payload, batch_size=args.batch_size)
    for index_number, content in enumerate(batches, start=1):
        (batch_dir / f"batch_{index_number:02d}_of_{len(batches):02d}.md").write_text(
            content, encoding="utf-8",
        )
    MANIFEST_PATH.write_bytes(_canonical_bytes(
        _manifest(sealed_bytes, sealed, "draft_awaiting_user_wording")
    ))
    print(json.dumps({
        "public": str(PUBLIC_PATH),
        "sealed_private": str(sealed_path),
        "review_batches": str(batch_dir),
        "manifest": str(MANIFEST_PATH),
        "batches": len(batches),
        "summary": v2a_summary(payload),
    }, ensure_ascii=False, indent=2))


def validate(args: argparse.Namespace) -> None:
    public = load_graded_qrels(PUBLIC_PATH, allowed_splits={"train", "dev"})
    private_root = _private_root(args.private_root)
    sealed_path = private_root / "rag_colloquial_v2a_sealed.json"
    sealed = load_graded_qrels(sealed_path, allowed_splits={"blind"})
    combined = {**public, "dataset_id": "rag-colloquial-v2a-validation-view",
                "items": [*public["items"], *sealed["items"]]}
    validate_graded_qrels(combined, require_approved=args.require_approved)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    if actual != manifest["sha256"]:
        raise SystemExit("sealed file hash does not match repository manifest")
    print(json.dumps(v2a_summary(combined), ensure_ascii=False, indent=2))


def apply_wording(args: argparse.Namespace) -> None:
    private_root = _private_root(args.private_root)
    sealed_path = private_root / "rag_colloquial_v2a_sealed.json"
    # The batches are the one artifact a human edits by hand, so they must be
    # allowed to live wherever that human keeps them.
    batch_dir = (Path(args.batch_dir).expanduser().resolve() if args.batch_dir
                 else private_root / "rag_colloquial_v2a_review_batches")
    batch_paths = sorted(batch_dir.glob("batch_*.md"))
    if not batch_paths:
        raise SystemExit(f"no review batches found in {batch_dir}")
    questions = parse_v2a_wording(batch_paths)
    public = load_graded_qrels(PUBLIC_PATH, allowed_splits={"train", "dev"})
    sealed = load_graded_qrels(sealed_path, allowed_splits={"blind"})
    combined = {**public, "dataset_id": "rag-colloquial-v2a-wording-view",
                "items": [*public["items"], *sealed["items"]]}
    combined, counts = apply_v2a_wording(combined, questions)
    from rag_colloquial_v2a import split_v2a_public_sealed as _split

    updated_public, updated_sealed = _split(combined)
    updated_public["dataset_id"] = "rag-colloquial-v2a-train-dev-v1"
    updated_public["status"] = "approved"
    updated_sealed["dataset_id"] = "rag-colloquial-v2a-sealed-v1"
    updated_sealed["status"] = "approved"
    public_bytes = _canonical_bytes(updated_public)
    sealed_bytes = _canonical_bytes(updated_sealed)
    PUBLIC_PATH.write_bytes(public_bytes)
    sealed_path.write_bytes(sealed_bytes)
    MANIFEST_PATH.write_bytes(_canonical_bytes(
        _manifest(sealed_bytes, updated_sealed, "approved_sealed_wording_applied")
    ))
    print(json.dumps({
        "wording_edits": counts,
        "summary": v2a_summary(combined),
    }, ensure_ascii=False, indent=2))


def export_split(args: argparse.Namespace) -> None:
    """Export one public split explicitly, never by positional truncation."""

    from rag_colloquial_v2a import V2A_SPLIT_TARGETS, V2A_STYLES

    public = load_graded_qrels(PUBLIC_PATH, allowed_splits={"train", "dev"})
    items = [item for item in public["items"] if item["split"] == args.split]
    expected = V2A_SPLIT_TARGETS[args.split] * len(V2A_STYLES)
    if len(items) != expected:
        raise SystemExit(
            f"{args.split} split drift: expected {expected} rows, found {len(items)}"
        )
    selected = {
        **public,
        "dataset_id": f"rag-colloquial-v2a-{args.split}-v1",
        "status": "approved" if not args.allow_draft else "diagnostic_draft",
        "items": items,
    }
    validate_graded_qrels(selected, allowed_splits={args.split},
                          require_approved=not args.allow_draft)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(selected))
    print(json.dumps({"output": str(output), "split": args.split,
                      "summary": v2a_summary(selected)},
                     ensure_ascii=False, indent=2))


def summary(args: argparse.Namespace) -> None:
    payload = load_graded_qrels(Path(args.path), require_approved=False)
    print(json.dumps(v2a_summary(payload), ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="build drafts from private anchor specs")
    build_parser.add_argument("--private-root")
    build_parser.add_argument("--force", action="store_true",
                              help="rebuild even if approved wording exists")
    build_parser.add_argument("--specs-dir")
    build_parser.add_argument("--batch-size", type=int, default=30)
    build_parser.add_argument(
        "--split-targets",
        help="train,dev,sealed anchor counts (default 80,20,20)",
    )
    build_parser.set_defaults(func=build)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--private-root")
    validate_parser.add_argument("--require-approved", action="store_true")
    validate_parser.set_defaults(func=validate)
    apply_parser = sub.add_parser(
        "apply-wording",
        help="apply only the user-edited question lines from review batches",
    )
    apply_parser.add_argument("--private-root")
    apply_parser.add_argument("--batch-dir",
                              help="where the edited review batches live")
    apply_parser.set_defaults(func=apply_wording)
    export_parser = sub.add_parser("export-split")
    export_parser.add_argument("--split", choices=("train", "dev"), required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--allow-draft", action="store_true")
    export_parser.set_defaults(func=export_split)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("path")
    summary_parser.set_defaults(func=summary)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
