#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick blind anchor chunks for Final v2, and prove they are blind.

Final v2 only means something if no part of it was seen while the candidate was
being tuned.  "Blind" here is a property of the *anchors*, not of the wording:
reusing a chunk that Dev-New or held-out already probes would let a question
inherit whatever the tuning already fitted to that chunk's neighbourhood.

So this walks every existing evaluation set, collects every chunk id and source
file that any of them uses as gold, and samples anchors from what is left.  It
deliberately never calls the answerability judge -- the judge is the thing under
test, and a set whose anchors it helped choose is not independent of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Chunk-level gold appears under several key names across the sets' vintages.
GOLD_KEYS = ("grade3_chunk_ids", "all_relevant_chunk_ids", "relevant_targets",
             "direct_target_chunk_ids", "target_chunk_ids", "gold_chunk_ids",
             "chunk_ids", "expect_sources", "expected_sources", "anchor_id")


def _walk(node, chunk_ids: set, sources: set) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in GOLD_KEYS:
                for item in (value if isinstance(value, list) else [value]):
                    if isinstance(item, str):
                        (sources if item.endswith(".md") else chunk_ids).add(item)
                    elif isinstance(item, dict):
                        _walk(item, chunk_ids, sources)
            else:
                _walk(value, chunk_ids, sources)
    elif isinstance(node, list):
        for item in node:
            _walk(item, chunk_ids, sources)


def used_material(skip_dir: str = "final_v2") -> tuple[set, set]:
    chunk_ids: set = set()
    sources: set = set()
    roots = [ROOT / "docs" / "rag_eval", ROOT / "tests"]
    for base in roots:
        for path in base.rglob("*.json"):
            if skip_dir and skip_dir in path.parts:
                continue                       # the set being built is not prior art
            try:
                _walk(json.loads(path.read_text(encoding="utf-8")), chunk_ids, sources)
            except Exception:
                continue
    return chunk_ids, sources


def _harvest_chunk_ids(node, out: set) -> None:
    """Collect every ``chunk_id`` string, wherever it nests.

    The generic walk keys on gold-label field names and deliberately ignores
    retrieval traces, so a *dataset's own* anchors and near-twin previews --
    which live under plain ``chunk_id`` keys -- slip through it.  For files that
    are prior art in their entirety (a previous blind set), everything they
    mention was read while building them, so all of it is excluded.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "chunk_id" and isinstance(value, str):
                out.add(value)
            else:
                _harvest_chunk_ids(value, out)
    elif isinstance(node, list):
        for item in node:
            _harvest_chunk_ids(item, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=int, default=40)
    parser.add_argument("--min-chars", type=int, default=320)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--output", default="docs/rag_eval/final_v2/ANCHOR_CANDIDATES.json")
    parser.add_argument("--skip-dir", default="final_v2",
                        help="正在建的集所在目录名(不算 prior art)")
    parser.add_argument("--prior-art", nargs="*", default=[],
                        help="整文件算已见材料的 JSON(前一代盲集及其锚点候选)")
    args = parser.parse_args()

    import chromadb
    from rag_tools import get_collection_name

    used_chunks, used_sources = used_material(skip_dir=args.skip_dir)
    for prior in args.prior_art:
        harvested: set = set()
        _harvest_chunk_ids(json.loads(Path(prior).read_text(encoding="utf-8")), harvested)
        used_chunks |= harvested
        print(f"[anchors] prior art {prior}: +{len(harvested)} chunk")
    print(f"[anchors] 已被现有评测集用过: {len(used_chunks)} 个 chunk / {len(used_sources)} 个来源文件")

    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name())
    got = collection.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    print(f"[anchors] 索引共 {len(ids)} 块")

    pool = []
    for chunk_id, document, meta in zip(ids, docs, metas):
        meta = meta or {}
        source = meta.get("source", "")
        if chunk_id in used_chunks or source in used_sources:
            continue
        if not (args.min_chars <= len(document or "") <= args.max_chars):
            continue
        if (meta.get("source_type") or "") == "paper":
            continue                            # paper domain is a separate track
        # Route-scope eligibility (added 2026-08-30 after the Final v3 audit):
        # the reference route excludes these source_types by construction, so a
        # question whose gold lives there is unanswerable through the route the
        # evaluation forces -- 12/80 of Final v3's positives were authored that
        # way and scored as retrieval failures.  An anchor selector for a
        # reference-route evaluation must only offer chunks that route serves.
        _ROUTE_EXCLUDED = {"application", "application_jd", "experience", "jd",
                           "log", "paper", "profile", "project_context",
                           "resume", "resume_rule", "story", "system",
                           "verification"}
        if (meta.get("source_type") or "") in _ROUTE_EXCLUDED:
            continue
        if (meta.get("owner_scope") or "") not in ("", "curated"):
            continue
        # A table of contents, an acknowledgements page or a block that is
        # nothing but figure transcription carries no fact to ask about.  A
        # question authored from one of those is a question about the corpus's
        # formatting, not about its content.
        title = meta.get("title", "") or ""
        if any(word in title for word in ("致谢", "目录", "参考文献", "索引", "封面")):
            continue
        prose = re.sub(r"\[图[:：].*?\]", "", document or "", flags=re.S)
        prose = re.sub(r"^\s*##.*$", "", prose, flags=re.M)
        if len(prose.strip()) < 250:
            continue
        pool.append({"chunk_id": chunk_id, "source": source,
                     "title": meta.get("title", ""),
                     "source_type": meta.get("source_type", ""),
                     "chars": len(document), "document": document})
    print(f"[anchors] 可选盲锚点: {len(pool)} 块，来自 {len({r['source'] for r in pool})} 个文件")

    # Deterministic, source-spread sampling: round-robin over sources, ordered by
    # a hash so the choice does not track anything about the corpus order.
    by_source: dict[str, list] = {}
    for row in sorted(pool, key=lambda r: hashlib.sha256(r["chunk_id"].encode()).hexdigest()):
        by_source.setdefault(row["source"], []).append(row)
    order = sorted(by_source, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    picked, cursor = [], 0
    while len(picked) < args.anchors and any(by_source.values()):
        source = order[cursor % len(order)]
        cursor += 1
        if by_source[source]:
            picked.append(by_source[source].pop(0))

    payload = {
        "schema_version": "final-v2-anchor-candidates-v1",
        "selection": {
            "n_anchors": len(picked),
            "excluded_chunk_ids": len(used_chunks),
            "excluded_sources": len(used_sources),
            "min_chars": args.min_chars, "max_chars": args.max_chars,
            "paper_domain_excluded": True,
            "judge_used": False,
        },
        "anchors": picked,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[anchors] wrote {out.relative_to(ROOT)} — {len(picked)} 个锚点，"
          f"{len({r['source'] for r in picked})} 个来源")


if __name__ == "__main__":
    main()
