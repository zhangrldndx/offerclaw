# -*- coding: utf-8 -*-
"""
OfferClaw · RAG Agent 主入口

把 RAG 检索 + LLM 问答 + 工具调用链路整合为一个完整的 Agent。
支持交互模式和单次查询模式。

架构：
  用户提问
    ↓
  先检索 ChromaDB（获取相关文档片段）
    ↓
  把检索结果注入 System Prompt
    ↓
  LLM 判断：需要调工具？ → 工具执行 → 整合回答
              ↓ 不需要 → 直接回答
    ↓
  返回用户

用法：
  python rag_agent.py "我本周的 RAG 学习进度怎么样？"  # 单次查询
  python rag_agent.py                                 # 交互模式
  python rag_agent.py --no-retrieval "你好"           # 不调检索（纯对话）
"""

import os
import sys
import json
import argparse
import time
from typing import Optional

import chromadb

from rag_tools import (
    get_embedding,
    get_embeddings_batch,
    chat_with_llm,
    get_collection_name,
    LLM_MODEL,
)

# =====================================================
# 配置
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = get_collection_name()
DEFAULT_TOP_K = 5

# 系统 Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """你是 OfferClaw，一位严谨专业的求职作战助手，运行在本地 OpenClaw 环境中。

你的使命是陪伴用户走完"画像 → 匹配 → 规划 → 执行 → 复盘"的完整求职闭环。

## 当前可用的知识库片段
以下是从 OfferClaw 知识库中检索到的相关内容，可能与用户的问题相关。
请基于这些片段回答。如果片段中没有相关内容，请如实告知"没有找到相关信息"，
**不要编造事实**。

【检索片段】
{retrieval_context}

## 行为原则
1. 渐进式画像：允许用户从不完整简历起步，缺失项标【待补充】，禁止脑补
2. 可解释优先：任何结论必须给出理由，并指向具体文件/章节
3. 不编造事实：岗位信息、薪资、门槛都不许编
4. 三档结论：岗位匹配只输出"当前适合投递 / 当前暂不建议投递 / 中长期可转向"
5. 结构化留痕：涉及画像/计划/日志变更时，主动提示写入哪个文件

## 输出风格
- 严谨、直接、少空话
- 优先使用分点、分节
- 先给结论与关键建议，再补充背景"""

# 无检索时的系统 Prompt
SYSTEM_PROMPT_NO_RETRIEVAL = """你是 OfferClaw，一位严谨专业的求职作战助手，运行在本地 OpenClaw 环境中。

你的使命是陪伴用户走完"画像 → 匹配 → 规划 → 执行 → 复盘"的完整求职闭环。

## 行为原则
1. 渐进式画像：允许用户从不完整简历起步，缺失项标【待补充】，禁止脑补
2. 可解释优先：任何结论必须给出理由，并指向具体文件/章节
3. 不编造事实：岗位信息、薪资、门槛都不许编
4. 三档结论：岗位匹配只输出"当前适合投递 / 当前暂不建议投递 / 中长期可转向"
5. 结构化留痕：涉及画像/计划/日志变更时，主动提示写入哪个文件

## 输出风格
- 严谨、直接、少空话
- 优先使用分点、分节
- 先给结论与关键建议，再补充背景"""


# =====================================================
# RAG Agent 类
# =====================================================

class RAGAgent:
    """
    OfferClaw RAG Agent
    整合 RAG 检索 + LLM 问答 + 工具调用的完整 Agent。

    v0.6.3 起 chat completion 走 OpenAI 兼容代理（gpt-5.4 + medium effort）。
    Embeddings 由 rag_tools.py 的 EMBEDDING_PROVIDER 配置决定。
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        model: str | None = None,
    ):
        from day1_api_starter import get_llm_config

        cfg = get_llm_config()
        self.collection_name = collection_name
        self.top_k = top_k
        self.model = model or cfg["model"]
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.conversation_history = []  # [{"role": ..., "content": ...}]
        self.tools = self._build_tools()

        # 初始化 ChromaDB
        self._init_db()

    def _init_db(self):
        """初始化 ChromaDB 连接"""
        if not os.path.exists(DB_DIR):
            print(f"[WARN] 数据库目录不存在: {DB_DIR}")
            print("[INFO] RAG 检索功能不可用，将使用纯对话模式")
            return

        self.client = chromadb.PersistentClient(path=DB_DIR)

        try:
            self.collection = self.client.get_collection(self.collection_name)
            from rag_tools import get_collection_sqlite_stats
            count = get_collection_sqlite_stats(DB_DIR, self.collection_name)[0]
            print(f"[DB] Collection 已加载: {count} 条记录")
        except Exception:
            print(f"[WARN] Collection 不存在: {self.collection_name}")
            print("[INFO] 请先运行: python rag_ingest.py")
            self.collection = None

    def _build_tools(self) -> list[dict]:
        """构建工具注册表（OpenAI 兼容格式）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前本地时间",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "从 OfferClaw 知识库检索相关文档片段",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "检索关键词或问题",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回的片段数量，默认 5",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用"""
        if tool_name == "get_current_time":
            import datetime
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if tool_name == "search_docs":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", self.top_k)
            docs = self._retrieve(query, top_k)
            if not docs:
                return "未检索到相关文档片段。"
            parts = []
            for i, doc in enumerate(docs):
                source = doc.get("source", "unknown")
                title = doc.get("title", "")
                preview = doc["document"][:200]
                parts.append(f"[片段{i+1}] 来源: {source} / {title}\n{preview}")
            return "\n\n---\n\n".join(parts)

        return f"未知工具: {tool_name}"

    def _retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """检索相关文档片段"""
        if self.collection is None:
            return []

        if top_k is None:
            top_k = self.top_k

        # 获取查询向量
        query_embedding = self._get_query_embedding(query)

        # 检索
        results = self.collection.query(
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

    def _get_query_embedding(self, query: str) -> list[float]:
        """获取查询的 embedding 向量"""
        return get_embedding(query)

    def _build_system_prompt(self, user_query: str, use_retrieval: bool = True) -> str:
        """构造 System Prompt"""
        if not use_retrieval or self.collection is None:
            return SYSTEM_PROMPT_NO_RETRIEVAL

        # 先检索
        docs = self._retrieve(user_query, self.top_k)

        if not docs:
            return SYSTEM_PROMPT_TEMPLATE.format(retrieval_context="（未检索到相关片段）")

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.get("source", "unknown")
            title = doc.get("title", "")
            tag = f"[片段{i+1}] 来源: {source}"
            if title:
                tag += f" / {title}"
            context_parts.append(f"{tag}\n{doc['document']}")

        context = "\n\n---\n\n".join(context_parts)
        return SYSTEM_PROMPT_TEMPLATE.format(retrieval_context=context)

    def chat(
        self,
        user_message: str,
        use_retrieval: bool = True,
        max_tool_iterations: int = 5,
    ) -> str:
        """
        完整的对话链路：检索 → System Prompt → LLM → 工具调用循环 → 回答
        
        参数：
            user_message: 用户输入
            use_retrieval: 是否启用 RAG 检索
            max_tool_iterations: 最大工具调用迭代次数
        
        返回：
            LLM 的最终回答
        """
        # 记录用户消息
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # 构造 System Prompt（含检索结果）
        system_prompt = self._build_system_prompt(user_message, use_retrieval)

        # 构造消息列表
        messages = [
            {"role": "system", "content": system_prompt},
        ] + self.conversation_history[-10:]  # 保留最近 10 条对话历史

        # 调用 LLM
        for iteration in range(max_tool_iterations):
            response = self._call_llm(messages)

            # 检查是否有工具调用
            tool_calls = response.get("tool_calls")
            if not tool_calls:
                # 没有工具调用，直接返回回答
                assistant_msg = response.get("content", "")
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_msg,
                })
                return assistant_msg

            # 有工具调用，执行工具
            messages.append(response)

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_function = tc.get("function", {})
                tc_name = tc_function.get("name", "")
                tc_args_str = tc_function.get("arguments", "{}")

                try:
                    tc_args = json.loads(tc_args_str)
                except json.JSONDecodeError:
                    tc_args = {}

                print(f"  [TOOL] 调用 {tc_name}({tc_args})")
                tool_result = self._execute_tool(tc_name, tc_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

        # 超过最大迭代次数，最后一次调用
        response = self._call_llm(messages)
        assistant_msg = response.get("content", "")
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_msg,
        })
        return assistant_msg

    def _call_llm(self, messages: list[dict]) -> dict:
        """调用 LLM（支持工具调用）

        v0.6.3 起：默认走 OpenAI 兼容代理 (gpt-5.4 + medium effort)。
        ``chat_with_llm_raw`` 内部根据 ``get_llm_config()`` 选 bearer 方式。
        """
        from day1_api_starter import build_zhipu_jwt, get_llm_config

        cfg = get_llm_config()
        if cfg["is_zhipu"]:
            bearer = build_zhipu_jwt(cfg["api_key"])
        else:
            bearer = cfg["api_key"]

        resp_data: dict = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
        }
        if cfg["reasoning_effort"]:
            resp_data["reasoning_effort"] = cfg["reasoning_effort"]

        resp = chat_with_llm_raw(bearer, resp_data)
        
        # 解析响应
        choices = resp.get("choices", [])
        if not choices:
            return {"content": "LLM 返回为空"}
        
        message = choices[0].get("message", {})
        return message

    def reset(self):
        """清空对话历史"""
        self.conversation_history = []
        print("[INFO] 对话历史已清空")


# =====================================================
# 直接 HTTP 调用（绕过 rag_tools.py 的 chat_with_llm）
# =====================================================

def chat_with_llm_raw(token: str, body: dict) -> dict:
    """直接调用 chat completion endpoint，支持工具调用。

    v0.6.3 起 endpoint 从 ``get_llm_config()`` 读，默认是 OpenAI 兼容代理。
    ``token`` 由调用方根据 provider 选好（OpenAI = raw key, Zhipu = JWT）。
    """
    import requests

    from day1_api_starter import get_llm_config

    cfg = get_llm_config()
    endpoint = f"{cfg['api_base']}/chat/completions"

    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=cfg["timeout"],
    )
    resp.raise_for_status()
    return resp.json()


# =====================================================
# 主入口
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="OfferClaw RAG Agent")
    parser.add_argument("query", nargs="?", help="查询问题（不指定则进入交互模式）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="检索返回数量")
    parser.add_argument("--no-retrieval", action="store_true", help="不调检索（纯对话）")
    parser.add_argument("--reset", action="store_true", help="清空对话历史")
    args = parser.parse_args()

    print("=" * 60)
    print("OfferClaw RAG Agent V1")
    print(f"检索: {'启用' if not args.no_retrieval else '禁用'}")
    print(f"Top-K: {args.top_k}")
    from day1_api_starter import API_KEY_ENV  # noqa: PLC0415
    print(f"API Key ({API_KEY_ENV}): {'已配置' if os.environ.get(API_KEY_ENV) else '未配置'}")
    print("=" * 60)
    print()

    agent = RAGAgent(top_k=args.top_k)

    if args.reset:
        agent.reset()
        if not args.query:
            return

    if args.query:
        # 单次查询
        print(f"\n[User] {args.query}")
        answer = agent.chat(args.query, use_retrieval=not args.no_retrieval)
        print(f"\n[OfferClaw] {answer}")
    else:
        # 交互模式
        print("OfferClaw RAG 交互模式（输入 'quit' 退出，'reset' 清空历史）")
        print("-" * 50)

        while True:
            try:
                query = input("\n[你] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            if query.lower() == "reset":
                agent.reset()
                continue

            answer = agent.chat(query, use_retrieval=not args.no_retrieval)
            print(f"\n[OfferClaw] {answer}")


if __name__ == "__main__":
    main()
