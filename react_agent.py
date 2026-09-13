# -*- coding: utf-8 -*-
"""react_agent.py — OfferClaw 极简 ReAct Agent（V4 §3）

让上层调用方用一句自然语言驱动 `tools_registry.REGISTRY` 里的 tool。

两种模式：
1. ``mode='deterministic'``（默认）：纯关键词路由——把用户消息按规则匹配到
   单个 tool，**不调 LLM**。这让 CI、doctor、单测无 KEY 也能跑通整条调用链，
   也用作 LLM 模式的 fallback。
2. ``mode='llm'``：把 ``REGISTRY.to_openai_schemas()`` 喂给智谱 GLM-4-Flash，
   走 OpenAI 兼容的 function calling，最多 ``max_steps`` 轮工具循环。

返回结构（统一）：
    {
        "answer": str,           # 最终给用户的话
        "tool_calls": [          # 调用过的工具列表
            {"name": "match_jd", "args": {...}, "result": {...}},
            ...
        ],
        "mode": "deterministic" | "llm",
        "steps": int,            # 实际工具循环轮数
        "errors": [str, ...],
    }
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools_registry import REGISTRY, ToolRegistry


# =====================================================
# Deterministic router（无 LLM）
# =====================================================

# 关键词 → tool 名称（顺序敏感：先匹配 specific 再 fallback）
_ROUTING_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(投递记录|投了哪些|applications|investments)", re.I), "list_applications"),
    (re.compile(r"(今天.*做|today|今日建议)", re.I), "today_advice"),
    (re.compile(r"(完整跑|career[\s_]?flow|端到端|8\s*节点)", re.I), "career_flow"),
    (re.compile(r"(抽取|提取|结构化).*jd", re.I), "extract_jd"),
    (re.compile(r"(简历|resume|骨架)", re.I), "resume_skeleton"),
    (re.compile(r"(匹配|match|结论|缺口|gap)", re.I), "match_jd"),
]


def _heuristic_pick_tool(msg: str) -> str | None:
    for pat, name in _ROUTING_RULES:
        if pat.search(msg):
            return name
    return None


def _extract_jd_from_message(msg: str) -> str:
    """如果消息里夹带了 JD，把它挑出来。
    粗略规则：长度 ≥ 30 的连续段，或带"岗位/职位/技术要求"标识。"""
    if any(k in msg for k in ("岗位名称", "职位名称", "技术要求", "工作地点")):
        return msg
    return msg if len(msg) >= 30 else ""


def _deterministic_step(msg: str, registry: ToolRegistry) -> dict:
    """单步路由：挑一个 tool 调一次。返回与 LLM mode 同构的输出。"""
    tool_name = _heuristic_pick_tool(msg)
    tool_calls: list[dict] = []
    errors: list[str] = []

    if not tool_name:
        return {
            "answer": "（deterministic mode）未识别出意图。"
                      "可问：匹配 / 简历 / 今日建议 / 投递记录 / 抽取 JD / 完整流程。",
            "tool_calls": [],
            "mode": "deterministic",
            "steps": 0,
            "errors": ["no_route_matched"],
        }

    # 组装参数
    args: dict[str, Any] = {}
    if tool_name in ("match_jd", "resume_skeleton", "career_flow"):
        jd = _extract_jd_from_message(msg)
        if not jd:
            return {
                "answer": f"识别到工具 {tool_name}，但消息里未发现可识别的 JD 文本。",
                "tool_calls": [],
                "mode": "deterministic",
                "steps": 0,
                "errors": ["no_jd_in_message"],
            }
        args["jd_text"] = jd
    elif tool_name == "extract_jd":
        args["raw"] = msg

    tool = registry.get(tool_name)
    result = tool.call(**args)
    tool_calls.append({"name": tool_name, "args": args, "result": result})

    answer = _summarize_result(tool_name, result)
    if "error" in result:
        errors.append(f"{tool_name}: {result['error']}")

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "mode": "deterministic",
        "steps": 1,
        "errors": errors,
    }


def _summarize_result(tool_name: str, result: dict) -> str:
    """把工具结果压成给用户的一句话。"""
    if "error" in result:
        return f"工具 {tool_name} 执行失败：{result['error']}"

    if tool_name == "match_jd":
        return (f"匹配结论：**{result.get('status')}**（方向={result.get('direction')}）"
                f"，共 {result.get('gap_count', 0)} 项缺口。")
    if tool_name == "today_advice":
        return f"今日建议：{result.get('headline', '')}"
    if tool_name == "list_applications":
        return f"applications.md 共 {result.get('total', 0)} 条真实投递记录。"
    if tool_name == "extract_jd":
        return (f"抽取出：公司={result.get('company') or '未识别'}，"
                f"岗位={result.get('title') or '未识别'}，"
                f"地点={result.get('location') or '未识别'}，"
                f"技能命中 {len(result.get('skills_detected', []))} 项。")
    if tool_name == "resume_skeleton":
        return (f"简历骨架已生成，命中 {len(result.get('keywords_hit', []))} 个关键词，"
                f"其中需要强调：{result.get('must_emphasize', [])[:3]}。")
    if tool_name == "career_flow":
        return (f"CareerFlow 跑完：结论={result.get('status')} · "
                f"路径={result.get('route_taken')} · "
                f"决策={result.get('application_decision')}。")
    return json.dumps(result, ensure_ascii=False)[:200]


# =====================================================
# LLM mode（OpenAI 兼容 function calling）
# =====================================================

_SYSTEM_PROMPT = """你是 OfferClaw 的求职作战助手。
你被允许调用下面的 tools 来回答用户问题。
原则：
1. 工具就是你能"做事"的全部方式，不要凭空编造结果。
2. 选 tool 后必须真的调一次，不要只描述意图。
3. 任何一个 tool 调完就给用户一句结论；不需要堆叠多步。"""


# ── [L5] 内循环 token 预算 + 工具结果智能截断 ──────────────────
# 动机：单轮 ReAct 在 messages 上无界累积、工具结果硬截 2000——名义 max_steps 越高越易在第 N 轮
# LLM 调用时才爆窗（已浪费），且硬尾截会把结构化关键字段（如 error）切掉。这里加预算预检 + 智能截断。
import os as _os

REACT_MAX_CTX_CHARS = int(_os.environ.get("REACT_MAX_CTX_CHARS", "24000"))
REACT_TOOL_RESULT_LIMIT = int(_os.environ.get("REACT_TOOL_RESULT_LIMIT", "2000"))


def _messages_chars(messages: list) -> int:
    """粗估当前 messages 上下文体积（字符≈token 量级，作内循环预算闸）。"""
    return sum(len(str(m.get("content") or "")) for m in messages)


def _truncate_tool_result(result, limit: int = REACT_TOOL_RESULT_LIMIT) -> str:
    """工具结果序列化 + 智能截断：超 limit 时 dict 优先保留关键字段（error/status/summary…），
    避免硬尾截把结构化关键信息切掉；关键字段也超或非 dict → 退回尾截。"""
    full = json.dumps(result, ensure_ascii=False)
    if len(full) <= limit:
        return full
    if isinstance(result, dict):
        KEY = ("error", "status", "ok", "code", "msg", "message", "summary",
               "title", "name", "count", "total", "reason")
        head = {k: result[k] for k in KEY if k in result}
        out = json.dumps({"_truncated": True, "_orig_chars": len(full), **head},
                         ensure_ascii=False)
        if len(out) <= limit:
            return out
    marker = "…[截断]"                                       # 预留 marker 长度，保证返回 ≤ limit
    return full[:max(0, limit - len(marker))] + marker


def _llm_step(msg: str, registry: ToolRegistry, max_steps: int = 3) -> dict:
    """LLM 模式：失败时无缝降级到 deterministic.

    v0.6.3 起走 OpenAI 兼容代理 (gpt-5.4 + medium effort)。
    实际通路集中在 ``day1_api_starter.get_llm_config()``。
    """
    # Local import to avoid a circular at module load if day1 itself
    # ever imports react_agent helpers.
    from day1_api_starter import API_KEY_ENV, build_zhipu_jwt, get_llm_config

    cfg = get_llm_config()
    api_key = cfg["api_key"]
    if not api_key:
        out = _deterministic_step(msg, registry)
        out["errors"].append(f"no_{API_KEY_ENV.lower()}:fallback_to_deterministic")
        return out

    bearer = build_zhipu_jwt(api_key) if cfg["is_zhipu"] else api_key

    import requests as _r
    tool_calls_log: list[dict] = []
    errors: list[str] = []
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": msg},
    ]
    tools_schema = registry.to_openai_schemas()

    final_answer = ""
    completed = False  # A4: 模型是否主动给出最终答复（用于区分 max_steps 掐断 vs 自然结束）
    for step in range(max_steps):
        if _messages_chars(messages) > REACT_MAX_CTX_CHARS:   # L5: 内循环 token 预算，超则提前止损
            errors.append("ctx_budget_exhausted")
            break
        try:
            payload: dict = {
                "model": cfg["model"],
                "messages": messages,
                "tools": tools_schema,
                "tool_choice": "auto",
                "temperature": 0.2,
            }
            if cfg["reasoning_effort"]:
                payload["reasoning_effort"] = cfg["reasoning_effort"]
            resp = _r.post(
                f"{cfg['api_base']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {bearer}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=cfg["timeout"],
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]["message"]
        except Exception as e:  # pragma: no cover
            errors.append(f"llm_call_failed:{e}")
            break

        if choice.get("tool_calls"):
            messages.append(choice)
            for call in choice["tool_calls"]:
                fn = call.get("function") or {}
                fn_name = fn.get("name")
                if not fn_name:  # A4: 畸形 tool_call（缺 function.name）→ 占位回复保配对 + 标记，不抛 KeyError
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "content": "[malformed tool_call skipped]"})
                    errors.append("malformed_tool_call")
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    tool = registry.get(fn_name)
                    result = tool.call(**args)
                except KeyError:
                    result = {"error": f"未知 tool：{fn_name}"}
                tool_calls_log.append({
                    "name": fn_name, "args": args, "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": _truncate_tool_result(result),   # L5: 智能截断（dict 保关键字段）
                })
            continue  # 让模型基于 tool 结果再说一句话

        # 没有 tool_calls → 模型给最终答复
        final_answer = (choice.get("content") or "").strip()
        completed = True
        break

    # A4: 跑满 max_steps 仍未主动收敛（且非 LLM 调用失败）→ 显式标记被掐断，区分于「模型主动结束」
    if not completed and not any(e.startswith("llm_call_failed") for e in errors):
        errors.append("max_steps_reached")

    return {
        "answer": final_answer or "（LLM 未给出答复）",
        "tool_calls": tool_calls_log,
        "mode": "llm",
        "steps": len(tool_calls_log),
        "errors": errors,
    }


# =====================================================
# 顶层入口
# =====================================================

def run(message: str, *, mode: str = "deterministic",
        registry: ToolRegistry | None = None,
        max_steps: int = 3) -> dict:
    """对一条用户消息跑一次 ReAct loop。

    mode='deterministic'：纯关键词路由，无 LLM 依赖。
    mode='llm'：function calling 工具循环，失败自动降级。
    """
    reg = registry or REGISTRY
    if mode == "llm":
        return _llm_step(message, reg, max_steps=max_steps)
    return _deterministic_step(message, reg)


if __name__ == "__main__":
    samples = [
        "今天该做什么？",
        "帮我匹配一下：岗位名称：AI Agent 应用开发实习生\n工作地点：上海\n技术要求：Python / LangGraph / RAG / FastAPI",
        "把这段 JD 抽成结构化：岗位名称：Java 后端开发实习生 公司：阿里 工作地点：杭州",
        "我投了哪些岗位？",
    ]
    for s in samples:
        out = run(s)
        print(f"\n>>> {s}")
        print(json.dumps(out, ensure_ascii=False, indent=2))
