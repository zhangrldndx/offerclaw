#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze a handoff's inputs into a manifest ``check_reranker_handoff`` can verify.

Each round of fine-tuning freezes a different set of code and data, so the
manifest is generated per round rather than edited by hand: a hand-maintained
hash list drifts silently, and a drifting list is worse than none because it
still reports ``ok``.
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(files: list[str], *, handoff: str, note: str) -> dict:
    import chromadb
    from rag_qrels_v2 import index_contract_fingerprint
    from rag_tools import get_collection_name, index_fingerprint

    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze missing files: {missing}")
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    live = index_fingerprint(collection=collection)
    return {
        "schema_version": "colloquial-reranker-handoff-manifest-v1",
        "handoff": handoff,
        "note": note,
        "files": {name: _sha256(ROOT / name) for name in sorted(files)},
        "index": {
            "collection": collection.name,
            "count": collection.count(),
            "fingerprint": index_contract_fingerprint(live),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file", action="append", required=True,
                        dest="files", help="repo-relative path to freeze")
    args = parser.parse_args(argv)

    manifest = build(args.files, handoff=args.handoff, note=args.note)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "files": len(manifest["files"]),
        "index": manifest["index"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
