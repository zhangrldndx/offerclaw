#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a shadow collection embedded with bge-large-zh-v1.5.

Probe infrastructure only: the production index, its embedding model and every
distance calibration stay untouched.  If the probe passes, the landing shape is
an *auxiliary retrieval channel* (query both collections, fuse by rank, recompute
distances in base-zh space) -- the proven dual-collection pattern, not a swap.

Same chunk ids, same documents, same default L2 metric as the main collection
(a cosine shadow once cost a 2x scale mismatch; pinned here).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHADOW = "kb_shadow_large_zh_v1"
MODEL = "BAAI/bge-large-zh-v1.5"


def main() -> None:
    import os
    os.environ.setdefault("OFFERCLAW_EMBED_MAX_SEQ", "512")
    os.environ.setdefault("OFFERCLAW_TORCH_DEVICE", "cpu")
    import chromadb
    from rag_tools import _local_embed, get_collection_name

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    main_col = client.get_collection(get_collection_name())
    got = main_col.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    print(f"[shadow] 主库 {len(ids)} 块", flush=True)

    try:
        client.delete_collection(SHADOW)
    except Exception:
        pass
    shadow = client.create_collection(SHADOW)   # default metric = L2, matches main

    t0 = time.time()
    BATCH = 64
    for start in range(0, len(ids), BATCH):
        chunk_docs = docs[start:start + BATCH]
        embeddings = _local_embed(chunk_docs, MODEL)
        shadow.add(ids=ids[start:start + BATCH], documents=chunk_docs,
                   metadatas=metas[start:start + BATCH], embeddings=embeddings)
        if (start // BATCH) % 8 == 0:
            done = start + len(chunk_docs)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[shadow] {done}/{len(ids)}  {rate:.0f} 块/s", flush=True)
    print(f"[shadow] 完成 {shadow.count()} 块, 用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
