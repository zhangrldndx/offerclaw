#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补齐 Chroma 派生索引的 source_type/owner_scope，不重新计算 embedding。

默认 dry-run；显式 ``--apply`` 才更新。Markdown 仍是事实源，本脚本可重复执行。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

from rag_ingest import _frontmatter_entity_fields  # noqa: E402
from rag_source_policy import infer_owner_scope, normalized_source_type  # noqa: E402
from rag_tools import get_collection_name  # noqa: E402


def _declared_by_basename() -> dict[str, dict[str, str]]:
    candidates: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if any(part in {".git", ".claude", ".venv", "node_modules"} for part in rel.parts):
            continue
        try:
            fields = _frontmatter_entity_fields(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if fields.get("source_type") or fields.get("owner_scope"):
            candidates.setdefault(path.name, []).append((rel, fields))
    result: dict[str, dict[str, str]] = {}
    for name, rows in candidates.items():
        rows.sort(key=lambda row: (0 if row[0].parts[:1] == ("knowledge_base",) else 1,
                                   len(row[0].parts), str(row[0])))
        result[name] = rows[0][1]
    return result


def run(collection_name: str, apply: bool = False, batch_size: int = 500) -> dict:
    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(collection_name)
    raw = collection.get(include=["metadatas"])
    ids = raw.get("ids") or []
    metas = raw.get("metadatas") or []
    declared = _declared_by_basename()
    changed_ids: list[str] = []
    changed_metas: list[dict] = []
    transitions: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()

    for rid, original in zip(ids, metas):
        meta = dict(original or {})
        source = str(meta.get("source") or "")
        frontmatter = declared.get(Path(source).name, {})
        old_type = str(meta.get("source_type") or "doc")
        new_type = normalized_source_type(frontmatter.get("source_type") or old_type, source)
        new_owner = infer_owner_scope(new_type, source, frontmatter.get("owner_scope") or "")
        owner_counts[new_owner] += 1
        updated = dict(meta)
        updated.update({"source_type": new_type, "owner_scope": new_owner})
        if updated != meta:
            changed_ids.append(str(rid))
            changed_metas.append(updated)
            transitions[f"{old_type}->{new_type}/{new_owner}"] += 1

    if apply:
        for start in range(0, len(changed_ids), batch_size):
            collection.update(
                ids=changed_ids[start:start + batch_size],
                metadatas=changed_metas[start:start + batch_size],
            )
    return {
        "collection": collection_name,
        "total": len(ids),
        "changed": len(changed_ids),
        "applied": bool(apply),
        "owner_counts": dict(owner_counts),
        "transitions": dict(transitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=get_collection_name())
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import json
    print(json.dumps(run(args.collection, apply=args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

