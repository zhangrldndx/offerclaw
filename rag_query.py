# -*- coding: utf-8 -*-
"""
OfferClaw · RAG Query 脚本

功能：
1. 从 ChromaDB 检索相关文档片段
2. 构造 Prompt 调用 LLM 回答
3. 支持交互模式和单次查询模式

用法：
  python rag_query.py "我的求职方向是什么？"     # 单次查询
  python rag_query.py                              # 交互模式
  python rag_query.py --top-k 3 "Python技能"       # 指定 top-k
"""

import os
import sys
import argparse

import chromadb

from rag_tools import (
    get_embedding,
    chat_with_llm,
    describe_embedding_config,
    fake_embedding,
    get_collection_name,
    has_embedding_api_key,
)

# =====================================================
# 配置
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = get_collection_name()
DEFAULT_TOP_K = 3

# RAG Prompt 模板
RAG_PROMPT_TEMPLATE = """你是 OfferClaw，一位求职作战助手。
请基于以下检索到的片段回答用户的问题。如果片段中没有相关容，请如实告知用户"没有找到相关信息"。

【检索片段】
{context}

【用户问题】
{question}

请给出结构化的回答，并注明信息来源（如"user_profile.md §2"）。"""


def retrieve(query: str, collection, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    检索相关文档片段。
    返回 list[dict]，包含 document、source、score。
    """
    # 获取查询向量
    if not has_embedding_api_key():
        # 伪向量（与 ingest 阶段一致）
        query_embedding = fake_embedding(query)
    else:
        query_embedding = get_embedding(query)

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "document": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "title": results["metadatas"][0][i].get("title", ""),
            "distance": results["distances"][0][i],
        })

    return docs


def answer_with_llm(query: str, docs: list[dict]) -> str:
    """
    用检索到的片段构造 Prompt，调用 LLM 回答。
    """
    context_parts = []
    for i, doc in enumerate(docs):
        source_tag = f"[片段{i+1}] 来源: {doc['source']}"
        if doc["title"]:
            source_tag += f" / {doc['title']}"
        context_parts.append(f"{source_tag}\n{doc['document']}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

    messages = [
        {"role": "system", "content": "你是 OfferClaw，一位严谨专业的求职作战助手。"},
        {"role": "user", "content": prompt},
    ]

    return chat_with_llm(messages)


def main():
    parser = argparse.ArgumentParser(description="OfferClaw RAG Query")
    parser.add_argument("query", nargs="?", help="查询问题（不指定则进入交互模式）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="检索返回数量")
    parser.add_argument("--no-llm", action="store_true", help="只检索不调 LLM")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Chroma collection 名称（默认按 embedding provider/model 自动选择）",
    )
    args = parser.parse_args()

    # 检查数据库
    if not os.path.exists(DB_DIR):
        print(f"[ERROR] 数据库目录不存在: {DB_DIR}")
        print("请先运行: python rag_ingest.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"[ERROR] Collection 不存在: {args.collection}")
        print("请先运行: python rag_ingest.py")
        sys.exit(1)

    print(f"[DB] 当前记录数: {collection.count()}")
    print(f"[Embedding] {describe_embedding_config()}")
    print(f"[API] {'已配置' if has_embedding_api_key() else '⚠️ 未配置（伪向量模式）'}")
    print()

    if args.query:
        # 单次查询
        run_query(args.query, collection, args.top_k, args.no_llm)
    else:
        # 交互模式
        print("OfferClaw RAG 交互模式（输入 'quit' 退出）")
        print("-" * 40)
        while True:
            query = input("\n> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not query:
                continue
            run_query(query, collection, args.top_k, args.no_llm)


def run_query(query: str, collection, top_k: int, no_llm: bool):
    print(f"\n[QUERY] {query!r}")

    docs = retrieve(query, collection, top_k)

    if not docs:
        print("  未检索到相关片段。")
        return

    print(f"  检索到 {len(docs)} 条相关片段：")
    for i, doc in enumerate(docs):
        source = doc["source"]
        title = doc["title"]
        preview = doc["document"][:80].replace("\n", " ")
        print(f"    [{i+1}] {source} / {title}")
        print(f"        {preview}...")

    if no_llm:
        return

    print(f"\n  调用 LLM 生成回答...")
    answer = answer_with_llm(query, docs)
    print(f"\n  [ANSWER]\n{answer}")


if __name__ == "__main__":
    main()
