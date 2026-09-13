#!/usr/bin/env python3
"""Verify that a copied OfferClaw workspace is the frozen fine-tune handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
# Each handoff freezes its own inputs, so the manifest is a parameter: the
# F1/F2 manifest stays as that round's record while F3 checks its own.
DEFAULT_MANIFEST = (
    ROOT / "docs/rag_eval/colloquial/RERANKER_FINETUNE_HANDOFF_MANIFEST.json"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = _sha256(path)
        checked.append(relative)
        if actual != expected:
            errors.append(f"hash mismatch: {relative}: {actual} != {expected}")

    # Manifests and schemas are safe; the actual repository-external Blind80
    # question file must never travel to the training computer.
    leaked = [
        path for path in ROOT.rglob("*.json")
        if "private_eval" in str(path).lower()
        or ("rag_colloquial_blind_v1" in path.name.lower()
            and "manifest" not in path.name.lower())
    ]
    if leaked:
        errors.append("private Blind data present inside workspace: " + ", ".join(
            str(path.relative_to(ROOT)) for path in leaked
        ))

    try:
        import chromadb
        from rag_qrels_v2 import index_contract_fingerprint
        from rag_tools import get_collection_name, index_fingerprint

        collection = chromadb.PersistentClient(
            path=str(ROOT / "chroma_db")
        ).get_collection(get_collection_name())
        live = index_fingerprint(collection=collection)
        actual_count = collection.count()
        actual_fingerprint = index_contract_fingerprint(live)
        expected_index = manifest["index"]
        if actual_count != expected_index["count"]:
            errors.append(
                f"index count mismatch: {actual_count} != {expected_index['count']}"
            )
        if actual_fingerprint != expected_index["fingerprint"]:
            errors.append(
                "index fingerprint mismatch: "
                f"{actual_fingerprint} != {expected_index['fingerprint']}"
            )
    except Exception as exc:
        errors.append(f"cannot validate Chroma index: {exc}")

    report = {
        "status": "ok" if not errors else "failed",
        "manifest": manifest_path.name,
        "workspace": str(ROOT),
        "checked_file_count": len(checked),
        "private_blind_present": bool(leaked),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
