# -*- coding: utf-8 -*-
"""
OfferClaw · Agent Demo · 最小可运行版

目的：
    演示 "用户对话 → LLM 决定是否调工具 → 工具执行 → LLM 整合结果 → 回复"
    的最小 Agent 闭环。是 OfferClaw V1 代码实现层的核心交付物，也是简历上
    的第一个 AI 项目雏形。

功能边界：
    - 跨会话多轮对话（启动时从 memory.json 恢复历史，退出时保存）
    - 4 个内置工具：get_current_time / calculator / echo / simple_profile_lookup
    - 走智谱 GLM-4-Flash 的 OpenAI 兼容 tools 参数做 function calling
    - 命令行交互，输入 "quit" 或 Ctrl+C 退出
    - 命令：/clear 清空 memory，/save 立即落盘，/history 看条数
    - 每轮最多 5 次 tool call 迭代（防死循环保险丝）

明确不做（下一版再说）：
    - RAG / 向量检索 / 知识库
    - LangGraph / LlamaIndex 框架化
    - Web UI 集成（已在 rag_api.py 实现）

使用：
    1. 确认 .env.local 里已有 OPENAI_API_KEY（v0.6.3 起，day1 已配好）
    2. 无需 pip install 新依赖（requests 已装）
    3. python agent_demo.py
    4. 提问示例：
       - "现在几点了？"
       - "帮我算 1234 * 5678"
       - "echo 一下：hello offerclaw"
       - "现在几点，然后算一下从现在到 2026-05-01 还有多少天" （测多轮 tool call）
"""

import os
import sys
import json
import time
import hmac
import hashlib
import requests

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA

# =====================================================
# Provider 配置 —— 单点配置在 day1_api_starter.py
# =====================================================
# v0.6.3 起：chat completions 走 OpenAI 兼容代理（gpt-5.4 + medium effort）。
# 所有 provider 切换在 .env.local + day1_api_starter.get_llm_config()
# 一处完成，这里只复用。
from day1_api_starter import (  # noqa: E402
    API_KEY_ENV,
    build_zhipu_jwt,
    get_llm_config,
    load_local_env,
)


# =====================================================
# System Prompt
# =====================================================

SYSTEM_PROMPT = """你是 OfferClaw Agent Demo，一个带工具调用能力的最小演示助手。

你有 4 个工具可用：
  - get_current_time：获取当前时间
  - calculator：计算数学表达式
  - echo：原样回显文本
  - simple_profile_lookup：从 user_profile.md 抽取章节内容（回答关于用户画像的问题时必须调用）

规则：
1. 当用户的问题需要用到这些能力时，**主动调用对应工具，不要猜答案**
2. 用户问到自己的画像信息（专业 / 技能 / 项目 / 实习 / 时间安排 等），**必须调 simple_profile_lookup，不要凭对话历史猜**
3. 如果问题不需要工具（比如闲聊 / 解释概念），直接回答
4. 可以在一轮里调用多个工具（比如先取时间再计算）
5. 回答简洁直接，不要空话
6. 工具的返回结果会作为 tool 角色的消息给你，你需要根据它组织最终回复"""


# =====================================================
# 跨会话 memory（V2 新增）
# =====================================================

MEMORY_PATH = "memory.json"


def load_memory(path: str = MEMORY_PATH) -> list:
    """从 JSON 文件恢复 conversation history。

    返回的 messages 列表不含 system prompt（system 由当前进程注入，
    保证每次启动都用最新版 prompt）。
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict) and m.get("role") != "system"]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_memory(messages: list, path: str = MEMORY_PATH) -> None:
    """把 conversation history 落盘（剔除 system prompt，避免污染）。"""
    payload = [m for m in messages if m.get("role") != "system"]
    try:
        from memory_layers import _atomic_write_json
        _atomic_write_json(path, payload)  # A5: 原子写防半截文件
    except OSError as e:
        print(f"[WARN] memory 保存失败：{e}")


# =====================================================
# LLM 调用
# =====================================================

def call_llm(messages, api_key, tools=None):
    """发送一次 chat completion 请求，返回解析后的 JSON dict。

    messages : list[dict]   —— 对话历史，符合 OpenAI chat 格式
    api_key  : str          —— Bearer 用的 API Key（Zhipu 路径会自动 JWT 化）
    tools    : list[dict]   —— 可选的 tools schema
    """
    cfg = get_llm_config()
    if cfg["is_zhipu"]:
        bearer_token = build_zhipu_jwt(api_key)
    else:
        bearer_token = api_key

    url = f"{cfg['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": cfg["model"],
        "messages": messages,
        # Agent 需要工具调用决策稳定，temperature 调低
        "temperature": 0.2,
    }
    if cfg["reasoning_effort"]:
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    if tools:
        payload["tools"] = tools

    response = requests.post(url, headers=headers, json=payload, timeout=cfg["timeout"])
    response.raise_for_status()
    return response.json()


# =====================================================
# 工具调用执行
# =====================================================

def execute_tool_call(tool_call: dict) -> str:
    """执行 LLM 请求的一次 tool call，返回工具结果的字符串。

    tool_call 结构示例：
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "1234 * 5678"}'
            }
        }
    """
    fn_info = tool_call.get("function", {})
    name = fn_info.get("name")
    args_json = fn_info.get("arguments", "")

    # arguments 字段是 JSON 字符串，需要反序列化
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        return f"[工具参数解析失败] 原始字符串：{args_json}"

    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"[未知工具] {name}"

    try:
        return fn(**args)
    except TypeError as e:
        return f"[工具参数不匹配] {name}: {e}"
    except Exception as e:
        return f"[工具执行出错] {type(e).__name__}: {e}"


# =====================================================
# Agent 主循环（单轮对话内的 tool call 循环）
# =====================================================

MAX_TOOL_ITERATIONS = 5  # 保险丝：防 LLM 无限循环调工具


def run_agent_turn(messages: list, api_key: str) -> str:
    """运行一轮 Agent 对话。

    假设 messages 里已经有用户最新的 user 消息，本函数负责：
      1. 调 LLM 拿到 assistant 响应
      2. 如果 assistant 请求工具调用，逐个执行、把 tool 结果追加到 messages
      3. 再次调 LLM，直到 assistant 不再请求工具（给出最终回复）
      4. 在迭代超过 MAX_TOOL_ITERATIONS 时强制退出

    返回：LLM 的最终文本回复
    副作用：messages 会被 in-place 修改（追加 assistant 和 tool 消息）
    """
    for iteration in range(MAX_TOOL_ITERATIONS):
        data = call_llm(messages, api_key, tools=TOOLS_SCHEMA)
        msg = data["choices"][0]["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # LLM 没请求工具 → 已经给出最终回复
            return msg.get("content", "") or ""

        # 有工具调用 → 逐个执行
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name, tc_id = fn.get("name"), tc.get("id")
            if not name or not tc_id:
                # A4: 畸形 tool_call（缺 name/id）→ 占位 tool 回复，保证与 assistant.tool_calls 数量配对，
                # 不留「有 tool_calls 却无 tool 回复」的孤儿消息污染下一轮请求体（原裸下标会抛 KeyError）
                messages.append({"role": "tool",
                                 "tool_call_id": tc_id or f"malformed_{iteration}",
                                 "content": "[malformed tool_call skipped]"})
                continue
            result = execute_tool_call(tc)
            print(f"  [tool:{name}] {result}")
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
        # 继续循环：让 LLM 看到工具结果后给最终回复

    # A4: 截断显式标记，让上游能区分「模型主动结束」与「被强制掐断到上限」
    return "[max_iterations_reached] 达到最大工具调用迭代次数，强制终止"


# =====================================================
# 主入口
# =====================================================

def main():
    # Windows 控制台 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 加载密钥
    load_local_env()
    cfg = get_llm_config()
    api_key = cfg["api_key"]
    if not api_key:
        print(f"[ERROR] 未检测到环境变量 {API_KEY_ENV}")
        print(f"请确认 .env.local 里有一行：{API_KEY_ENV}=你的密钥")
        sys.exit(1)

    print("=" * 60)
    print("OfferClaw · Agent Demo · V2（含 profile 查询 + 跨会话 memory）")
    print(f"Model    : {cfg['model']} (effort={cfg['reasoning_effort'] or '-'})")
    print(f"Tools    : {', '.join(TOOL_FUNCTIONS.keys())}")
    print(f"Memory   : {MEMORY_PATH}（启动恢复 / 退出保存）")
    print("命令     : /clear 清空 memory · /save 立即保存 · /history 看条数 · quit 退出")
    print("=" * 60)
    print()

    # 对话历史初始化：system prompt + 从 memory.json 恢复
    history = load_memory()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    if history:
        print(f"[memory] 已恢复 {len(history)} 条历史消息")
        print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n保存 memory 后退出...")
            save_memory(messages)
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            save_memory(messages)
            print("已保存 memory，退出。")
            break
        if user_input == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_memory(messages)
            print("[memory] 已清空")
            continue
        if user_input == "/save":
            save_memory(messages)
            print(f"[memory] 已保存（{len(messages) - 1} 条非 system 消息）")
            continue
        if user_input == "/history":
            non_system = [m for m in messages if m.get("role") != "system"]
            print(f"[memory] 当前共 {len(non_system)} 条非 system 消息")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = run_agent_turn(messages, api_key)
            print(f"Agent: {reply}")
            print()
            # 每轮成功后落盘一次（防进程崩溃丢历史）
            save_memory(messages)
        except requests.HTTPError as e:
            print(f"[HTTP ERROR] {e}")
            print(f"响应体：{e.response.text if e.response is not None else 'N/A'}")
            messages.pop()
        except requests.Timeout:
            print("[TIMEOUT] 请求超过 30 秒未响应")
            messages.pop()
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            messages.pop()


if __name__ == "__main__":
    main()
