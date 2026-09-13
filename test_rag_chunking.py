#!/usr/bin/env python3
"""
OfferClaw RAG Quickstart — 单文档分块 + 向量化 + 检索测试
目标：验证 ChromaDB + LangChain 分片器能正确处理 user_profile.md

说明：这是一个**可独立运行的 quickstart 演示脚本**（不是 pytest 用例）。
所有副作用都收在 ``main()`` 里，并由 ``if __name__ == "__main__"`` 守卫，
以免 ``pytest`` 在收集阶段 import 本文件时执行入库逻辑、打断整个测试收集。
运行：``python test_rag_chunking.py``
"""

import hashlib
import os
import struct
import sys

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.md")


def simple_hash_embedding(text: str, dim: int = 384) -> list[float]:
    """用 SHA256 哈希生成固定维度伪向量，仅用于验证 ChromaDB 入库/检索流程。

    必须用「无符号整数」解包再归一化：``struct.unpack('f')`` 把任意哈希字节当
    IEEE-754 float32 会有约 95% 概率落到 NaN/Inf 位模式，被 ChromaDB 拒绝。
    整数解包恒为有限值。生产环境用真实 Embedding API 替换本函数。
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    extended = b""
    while len(extended) < dim * 4:
        h = hashlib.sha256(h).digest()
        extended += h
    ints = struct.unpack(f"{dim}I", extended[: dim * 4])  # 无符号 32-bit，恒有限
    mn, mx = min(ints), max(ints)
    if mx == mn:
        return [0.5] * dim
    span = mx - mn
    return [(v - mn) / span for v in ints]


def main() -> None:
    # ─── Step 1: 读取源文档 ───
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"[1] 读取 user_profile.md: {len(raw_text)} 字符, {raw_text.count(chr(10))} 行")

    # ─── Step 2: 用 LangChain 分块 ───
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        print("[!] langchain-text-splitters 未安装，尝试安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "langchain-text-splitters", "-q"])
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n----", "\n\n", "\n",
                    "。", "！", "？", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_text(raw_text)
    print(f"[2] 分块完成: {len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        lines = chunk.count("\n") + 1
        print(f"    Block {i+1:02d}: {len(chunk):4d} 字符, {lines:3d} 行 | "
              f"前 60 字: {chunk[:60].replace(chr(10), ' ')}...")

    # ─── Step 3: ChromaDB 本地入库 ───
    import chromadb

    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db_test")
    if os.path.exists(db_dir):
        import shutil
        shutil.rmtree(db_dir)
    client = chromadb.PersistentClient(path=db_dir)
    try:
        client.delete_collection("profile_test")
    except Exception:
        pass
    collection = client.create_collection(name="profile_test")

    # ─── Step 4: 用简单哈希模拟 embedding 入库（不依赖 API）───
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"block_{i+1:02d}"],
            embeddings=[simple_hash_embedding(chunk)],
            documents=[chunk],
            metadatas=[{"block_index": i + 1, "char_len": len(chunk)}],
        )
    print(f"\n[3] ChromaDB 入库完成: {collection.count()} 条记录")
    print(f"    数据库目录: {db_dir}")

    # ─── Step 5: 检索测试 ───
    test_queries = [
        "求职方向是什么？", "Python 技能水平如何？",
        "项目经历有哪些？", "可接受哪些城市？",
    ]
    print(f"\n[4] 检索测试:")
    for query in test_queries:
        results = collection.query(
            query_embeddings=[simple_hash_embedding(query)], n_results=2)
        print(f"\n  查询: 「{query}」")
        for j in range(len(results["ids"][0])):
            doc_id = results["ids"][0][j]
            distance = results["distances"][0][j]
            doc_preview = results["documents"][0][j][:100].replace("\n", " ")
            print(f"    [{doc_id}] distance={distance:.4f} | {doc_preview}...")

    print(f"\n✅ Quickstart 全部通过！")
    print(f"   - LangChain 分块: {len(chunks)} 块")
    print(f"   - ChromaDB 入库: {collection.count()} 条")
    print(f"   - 检索测试: {len(test_queries)} 组查询均返回结果")
    print(f"\n下一步：接入真实 Embedding API（智谱 glm-4-embed）替换简单哈希向量。")


if __name__ == "__main__":
    main()
