# -*- coding: utf-8 -*-
"""
OfferClaw · LangGraph 工作流编排

用 LangGraph 重构 RAG Agent 的"检索 → Prompt 注入 → LLM → 工具调用"链路。
从手动编排升级为声明式状态机。

架构：
  __start__ → retrieve → build_prompt → call_llm → {有工具调用?}
      ↓ 是                                    ↓ 否
    execute_tools → call_llm (循环)         → __end__
    
覆盖常见 AI 应用 JD 中的端到端 LLM 工作流与调用链路编排职责

用法：
  python rag_graph.py "我的求职方向是什么"       # 单次查询
  python rag_graph.py                              # 交互模式
  python rag_graph.py --no-retrieval "你好"       # 纯对话（不经检索节点）
"""

import os
import sys
import json
import argparse
import datetime
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rag_tools import (
    get_embedding,
    describe_embedding_config,
    get_collection_name,
    has_embedding_api_key,
)

HAS_API_KEY = has_embedding_api_key()

import chromadb

# =====================================================
# 配置
# =====================================================

DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = get_collection_name()
DEFAULT_TOP_K = 5


# =====================================================
# State 定义（TypedDict）
# =====================================================

class AgentState(TypedDict):
    """
    LangGraph 状态定义。
    每个节点读取/修改这个状态。
    """
    # 用户输入
    query: str
    # 是否启用检索
    use_retrieval: bool
    # 检索到的文档片段
    retrieved_docs: list[dict]
    # 对话消息列表（OpenAI 格式，支持 add_messages 注解）
    messages: Annotated[list, add_messages]
    # 最终回答
    final_answer: str
    # 工具调用次数（用于监控）
    tool_call_count: int
    # 错误信息
    error: str


# =====================================================
# 工具注册表
# =====================================================

def tool_get_current_time(arguments: dict) -> str:
    """获取当前本地时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_search_docs(collection, arguments: dict, top_k: int = DEFAULT_TOP_K) -> str:
    """从 OfferClaw 知识库检索相关文档片段"""
    if collection is None:
        return "知识库未连接。请先运行: python rag_ingest.py"

    query = arguments.get("query", "")

    query_embedding = get_embedding(query)

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs = []
    for i in range(len(results["ids"][0])):
        source = results["metadatas"][0][i].get("source", "unknown")
        title = results["metadatas"][0][i].get("title", "")
        doc = results["documents"][0][i]
        docs.append(f"[片段{i+1}] 来源: {source}{' / ' + title if title else ''}\n{doc[:300]}")

    if not docs:
        return "未检索到相关文档片段。"

    return "\n\n---\n\n".join(docs)


# 工具定义（OpenAI 兼容格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前本地时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "top_k": {"type": "integer", "description": "返回的片段数量，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_current_time": lambda args: tool_get_current_time(args),
    "search_docs": None,  # 需要 collection，在节点中动态绑定
}


# =====================================================
# System Prompt 模板
# =====================================================

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
# 工作流节点（Nodes）
# =====================================================

def node_retrieve(state: AgentState, collection=None) -> dict:
    """
    节点 1：检索
    根据用户问题检索 ChromaDB，结果存入 state['retrieved_docs']。
    """
    query = state["query"]
    use_retrieval = state.get("use_retrieval", True)

    if not use_retrieval or collection is None:
        return {"retrieved_docs": [], "tool_call_count": 0}

    query_embedding = get_embedding(query)

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=DEFAULT_TOP_K,
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "document": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "title": results["metadatas"][0][i].get("title", ""),
            "distance": results["distances"][0][i],
        })

    return {"retrieved_docs": docs, "tool_call_count": 0}


def node_build_prompt(state: AgentState) -> dict:
    """
    节点 2：构造 System Prompt
    把检索结果注入 Prompt，追加到消息列表。
    """
    docs = state.get("retrieved_docs", [])

    if docs:
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.get("source", "unknown")
            title = doc.get("title", "")
            tag = f"[片段{i+1}] 来源: {source}"
            if title:
                tag += f" / {title}"
            context_parts.append(f"{tag}\n{doc['document']}")
        context = "\n\n---\n\n".join(context_parts)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(retrieval_context=context)
    else:
        system_content = SYSTEM_PROMPT_NO_RETRIEVAL

    # 构造消息列表：system prompt + 用户问题 + 历史对话
    query = state["query"]
    messages = [{"role": "system", "content": system_content}] + state.get("messages", [])
    messages.append({"role": "user", "content": query})

    return {"messages": messages}


def _msg_to_dict(m):
    """把 LangChain BaseMessage 或 dict 统一转回 OpenAI dict 格式。"""
    if isinstance(m, dict):
        return m
    role_map = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
    role = role_map.get(getattr(m, "type", ""), "user")
    d = {"role": role, "content": getattr(m, "content", "") or ""}
    ak = getattr(m, "additional_kwargs", {}) or {}
    tc = getattr(m, "tool_calls", None) or ak.get("tool_calls")
    if tc:
        d["tool_calls"] = tc
    if role == "tool":
        d["tool_call_id"] = getattr(m, "tool_call_id", "") or ak.get("tool_call_id", "")
        d["name"] = getattr(m, "name", "") or ak.get("name", "")
    return d


def node_call_llm(state: AgentState) -> dict:
    """
    节点 3：调用 LLM
    传入 tools，LLM 可能返回工具调用。

    v0.6.3 起走 OpenAI 兼容代理（gpt-5.4 + medium effort）。
    """
    from day1_api_starter import build_zhipu_jwt, get_llm_config

    cfg = get_llm_config()
    bearer = build_zhipu_jwt(cfg["api_key"]) if cfg["is_zhipu"] else cfg["api_key"]

    messages = [_msg_to_dict(m) for m in state["messages"]]

    body: dict = {
        "model": cfg["model"],
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if cfg["reasoning_effort"]:
        body["reasoning_effort"] = cfg["reasoning_effort"]

    import requests
    resp = requests.post(
        f"{cfg['api_base']}/chat/completions",
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=cfg["timeout"],
    )
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        return {
            "messages": [{"role": "assistant", "content": "LLM 返回为空"}],
            "final_answer": "LLM 返回为空",
        }

    message = choices[0].get("message", {})
    
    # 追加到消息列表
    new_messages = [message]

    # 检查是否有工具调用
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return {"messages": new_messages, "tool_call_count": len(tool_calls)}
    else:
        # 没有工具调用，直接输出回答
        content = message.get("content", "")
        return {"messages": new_messages, "final_answer": content}


def node_execute_tools(state: AgentState, collection=None) -> dict:
    """
    节点 4：执行工具调用
    根据 LLM 的 tool_calls 执行对应工具，结果追加到消息列表。
    """
    messages = state["messages"]
    last_message = _msg_to_dict(messages[-1])
    tool_calls = last_message.get("tool_calls", [])

    if not tool_calls:
        return {"messages": [], "tool_call_count": 0}

    tool_messages = []
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        tc_function = tc.get("function", {})
        tc_name = tc_function.get("name", "")
        tc_args_str = tc_function.get("arguments", "{}")

        try:
            tc_args = json.loads(tc_args_str)
        except json.JSONDecodeError:
            tc_args = {}

        # 执行工具
        if tc_name == "get_current_time":
            result = tool_get_current_time(tc_args)
        elif tc_name == "search_docs":
            result = tool_search_docs(collection, tc_args)
        else:
            result = f"未知工具: {tc_name}"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": tc_name,
            "content": result,
        })

    return {"messages": tool_messages, "tool_call_count": len(tool_calls)}


def should_continue_to_tools(state: AgentState) -> str:
    """
    条件边：检查最后一条消息是否有工具调用。
    有 → 执行工具；无 → 结束。
    """
    messages = state["messages"]
    if not messages:
        return END

    last_message = _msg_to_dict(messages[-1])
    tool_calls = last_message.get("tool_calls")

    if tool_calls:
        return "execute_tools"
    else:
        return END


# =====================================================
# 构建图（Graph）
# =====================================================

def build_graph(collection=None):
    """
    构建 LangGraph 工作流图。
    
    节点：
      retrieve → build_prompt → call_llm → {条件边} → execute_tools → call_llm (循环)
                                                                    → END
    
    边：
      __start__ → retrieve
      retrieve → build_prompt
      build_prompt → call_llm
      call_llm → {条件边: 有工具?} → execute_tools → call_llm (循环)
                                            → END
    """
    # 用闭包绑定 collection
    def retrieve_node(state):
        return node_retrieve(state, collection)

    def execute_tools_node(state):
        return node_execute_tools(state, collection)

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("build_prompt", node_build_prompt)
    workflow.add_node("call_llm", node_call_llm)
    workflow.add_node("execute_tools", execute_tools_node)

    # 添加边
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "build_prompt")
    workflow.add_edge("build_prompt", "call_llm")

    # 条件边
    workflow.add_conditional_edges(
        "call_llm",
        should_continue_to_tools,
        {
            "execute_tools": "execute_tools",
            END: END,
        },
    )

    # 工具执行后回到 LLM
    workflow.add_edge("execute_tools", "call_llm")

    # 编译
    return workflow.compile()


# =====================================================
# LangGraph Studio / `langgraph dev` 入口
# =====================================================

def make_graph():
    """`langgraph dev` / LangGraph Studio 的图工厂（零参）。

    为什么单独包一层而不是直接指向 ``build_graph``：langgraph-cli 对**带参数**的
    工厂会塞一个 ``RunnableConfig`` 进第一个位置参数，会把 ``collection`` 位顶掉
    （collection 变成 config dict，invoke 时 ``collection.query`` 直接炸）。这里做成
    无参工厂：内部自己把真实 chroma collection 绑好——库在就用（Studio 里能真跑检索），
    库不在就退化成 collection=None 的纯对话图（可视化照样出）。
    """
    collection = None
    try:
        if os.path.exists(DB_DIR):
            client = chromadb.PersistentClient(path=DB_DIR)
            collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = None
    return build_graph(collection)


# =====================================================
# 主入口
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="OfferClaw LangGraph Agent")
    parser.add_argument("query", nargs="?", help="查询问题（不指定则进入交互模式）")
    parser.add_argument("--no-retrieval", action="store_true", help="不调检索（纯对话）")
    args = parser.parse_args()

    print("=" * 60)
    print("OfferClaw LangGraph Agent V1")
    print(f"检索: {'启用' if not args.no_retrieval else '禁用'}")
    print(f"架构: LangGraph 状态机（retrieve → build_prompt → call_llm → tools）")
    print(f"Embedding: {describe_embedding_config()}")
    print(f"API Key: {'已配置' if HAS_API_KEY else '[MISS] 未配置'}")
    print("=" * 60)
    print()

    # 初始化 ChromaDB
    collection = None
    if os.path.exists(DB_DIR):
        client = chromadb.PersistentClient(path=DB_DIR)
        try:
            collection = client.get_collection(COLLECTION_NAME)
            print(f"[DB] Collection 已加载: {collection.count()} 条记录")
        except Exception:
            print(f"[WARN] Collection 不存在: {COLLECTION_NAME}")
    else:
        print(f"[WARN] 数据库目录不存在，RAG 检索不可用")

    # 构建图
    app = build_graph(collection)

    if args.query:
        # 单次查询
        print(f"[USER] {args.query}")
        print(f"\n[GRAPH] 开始执行工作流...")

        initial_state = {
            "query": args.query,
            "use_retrieval": not args.no_retrieval,
            "retrieved_docs": [],
            "messages": [],
            "final_answer": "",
            "tool_call_count": 0,
            "error": "",
        }

        # 执行图
        final_state = app.invoke(initial_state)

        answer = final_state.get("final_answer", "")
        doc_count = len(final_state.get("retrieved_docs", []))
        tool_count = final_state.get("tool_call_count", 0)

        print(f"\n[GRAPH] 执行完成")
        print(f"  检索到 {doc_count} 条片段")
        print(f"  工具调用 {tool_count} 次")
        print(f"\n[BOT] OfferClaw: {answer}")
    else:
        # 交互模式
        print("OfferClaw LangGraph 交互模式（输入 'quit' 退出）")
        print("-" * 50)
        
        conversation_history = []

        while True:
            try:
                query = input("\n[YOU] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            print(f"\n[GRAPH] 开始执行工作流...")

            initial_state = {
                "query": query,
                "use_retrieval": not args.no_retrieval,
                "retrieved_docs": [],
                "messages": conversation_history[-10:],  # 保留最近 10 条
                "final_answer": "",
                "tool_call_count": 0,
                "error": "",
            }

            final_state = app.invoke(initial_state)

            answer = final_state.get("final_answer", "")
            doc_count = len(final_state.get("retrieved_docs", []))
            tool_count = final_state.get("tool_call_count", 0)

            # 更新对话历史
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})

            print(f"\n[GRAPH] 完成 | 检索 {doc_count} 条 | 工具 {tool_count} 次")
            print(f"\n[BOT] OfferClaw: {answer}")


if __name__ == "__main__":
    main()
