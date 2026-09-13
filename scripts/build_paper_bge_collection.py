# -*- coding: utf-8 -*-
"""从 e5 论文集合克隆同一批 chunk,用主库同款 BGE 重新编码建 kb_paper_bge_v1。

为什么克隆而不是重新 ingest(报告 §2.2 单变量核对):
  重新 ingest 会套上 2026-08-10 的分块卫生规则(硬上限/引文过滤),chunk 文本就变了,
  于是"BGE vs E5"里混入"新分块 vs 旧分块"两个变量。克隆保证两个集合
  **文档集合完全一致、0 个 chunk 文本不同**,唯一变量=Dense Embedding 模型。

用法(必须显式给主库同款 embedding 配置,防血缘污染):
  OFFERCLAW_TORCH_DEVICE=cpu OFFERCLAW_EMBED_PREFIX= \
  .venv/bin/python scripts/build_paper_bge_collection.py
"""
from __future__ import annotations

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SRC = os.environ.get("SRC_COLLECTION", "kb_paper_e5_v1")
DST = os.environ.get("DST_COLLECTION", "kb_paper_bge_v1")


def main() -> int:
    import chromadb
    from rag_lang import detect_language
    from rag_tools import embed_profile, get_embeddings_batch

    if os.environ.get("OFFERCLAW_EMBED_PREFIX"):
        print("[中止] OFFERCLAW_EMBED_PREFIX 非空——BGE 侧不加 e5 前缀,否则两库不可比",
              file=sys.stderr)
        return 2

    client = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db"))
    src = client.get_collection(SRC)
    got = src.get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    ids = got.get("ids") or []
    print(f"[源] {SRC}: {len(docs)} 块")
    if not docs:
        print("[中止] 源集合为空", file=sys.stderr)
        return 2

    try:                                    # 重建语义:先删后建,避免半旧半新
        client.delete_collection(DST)
        print(f"[重建] 已删除既有 {DST}")
    except Exception:
        pass
    # 距离空间必须与主库一致(报告 §18 审计项):主库 metadata=None → Chroma 默认 l2,
    # 归一化向量下 L2² = 2-2cos = 2×cosine_distance。建成 cosine 会让配额通道的距离
    # **只有主池的一半**,门控 cutoff 与 Phase 5 标定全部失真(首建时实测踩到,故显式钉死)。
    src_space = (src.metadata or {}).get("hnsw:space", "l2")
    dst = client.create_collection(DST, metadata={"hnsw:space": src_space})
    print(f"[空间] 与源集合对齐:hnsw:space={src_space}")

    prof = embed_profile()
    new_metas = []
    for d, m in zip(docs, metas):
        mm = dict(m or {})
        mm["document_language"] = detect_language(d)   # 报告 §9.2 语言分区
        mm["embed_profile"] = prof                     # 血缘签名(防静默毁库)
        new_metas.append(mm)

    print(f"[编码] BGE 编码 {len(docs)} 块(profile={prof})…")
    vecs = get_embeddings_batch(docs)
    dst.add(ids=ids, documents=docs, embeddings=vecs, metadatas=new_metas)

    from collections import Counter
    langs = Counter(m["document_language"] for m in new_metas)
    print(f"[完成] {DST}: {dst.count()} 块;语言分布 {dict(langs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
