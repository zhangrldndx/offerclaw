# -*- coding: utf-8 -*-
"""
OfferClaw · Agent Demo · 工具模块

本模块定义 Agent Demo 可用的工具函数、工具注册表和 OpenAI 兼容的 schema。

设计原则：
    - 纯函数实现，无状态，不依赖外部资源
    - schema 描述与函数定义一一对应，避免漂移
    - 新增工具时，同时修改 TOOL_FUNCTIONS 和 TOOLS_SCHEMA 两处

已有工具：
    - get_current_time：获取当前本地时间
    - calculator：安全数学表达式求值（AST 解析，不用 eval）
    - echo：原样回显，用于测试工具调用链路
    - simple_profile_lookup：按章节标题从 user_profile.md 抽取片段
"""

import ast
import datetime
import operator
import os
import re


# =====================================================
# 工具 1：获取当前时间
# =====================================================

def tool_get_current_time() -> str:
    """返回当前本地时间的字符串表示。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =====================================================
# 工具 2：安全计算器（AST 解析，不用 eval）
# =====================================================

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node):
    """递归求值 AST 节点。

    只允许数字常量、二元运算符、一元运算符，拒绝函数调用、
    属性访问、名字引用等潜在危险操作。
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型：{type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        val = _safe_eval_node(node.operand)
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(val)
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def tool_calculator(expression: str) -> str:
    """对一个简单数学表达式求值。

    支持：+ - * / ** % 和括号。
    禁止：函数调用（如 eval / __import__）、属性访问、名字引用。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"计算出错：{type(e).__name__}: {e}"


# =====================================================
# 工具 3：Echo（原样回显，最小测试工具）
# =====================================================

def tool_echo(text: str) -> str:
    """原样回显一段文本，前缀加 'echo: '。用于验证工具调用链路。"""
    return f"echo: {text}"


# =====================================================
# 工具 4：用户画像查询（V2 新增）
# =====================================================

PROFILE_PATH = "user_profile.md"
_PROFILE_CACHE = {"mtime": None, "sections": None}


def _parse_profile_sections(text: str) -> dict:
    """把 user_profile.md 拆成 {section_title: section_body} 的 dict。

    识别规则：以 "## " 开头的行作为章节分隔；标题取 "## " 之后的整行。
    例如 "## 1. 基础信息" → key 既是 "1. 基础信息" 也建别名 "1" / "§1" / "基础信息"
    便于 LLM 用任意写法查询。
    """
    sections = {}
    current_title = None
    buf = []

    def _flush():
        if current_title is not None:
            body = "\n".join(buf).strip()
            sections[current_title] = body
            # 别名：尝试拆出编号和标题
            m = re.match(r"^(\d+)\.\s*(.+)$", current_title)
            if m:
                num, name = m.group(1), m.group(2).strip()
                sections.setdefault(num, body)
                sections.setdefault(f"§{num}", body)
                sections.setdefault(name, body)

    for line in text.splitlines():
        if line.startswith("## "):
            _flush()
            current_title = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    _flush()
    return sections


def _load_profile_sections() -> dict:
    """带 mtime 缓存的 profile 解析，避免每次工具调用都重读文件。"""
    if not os.path.exists(PROFILE_PATH):
        return {}
    mtime = os.path.getmtime(PROFILE_PATH)
    if _PROFILE_CACHE["mtime"] != mtime:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            _PROFILE_CACHE["sections"] = _parse_profile_sections(f.read())
            _PROFILE_CACHE["mtime"] = mtime
    return _PROFILE_CACHE["sections"] or {}


def tool_simple_profile_lookup(section: str = "", max_chars: int = 1200) -> str:
    """从 user_profile.md 抽取一个章节的内容。

    section : str
        章节标识。支持多种写法：
        - "1" / "§1"：按编号
        - "基础信息" / "1. 基础信息"：按标题（精确或包含匹配）
        - 空字符串：返回所有章节标题列表
    max_chars : int
        返回内容最大字符数，超出截断（防止单次工具结果过大）

    返回：
        命中 → 章节正文
        未命中 → "未找到匹配章节" + 可用章节列表
        文件不存在 → 错误说明
    """
    sections = _load_profile_sections()
    if not sections:
        return f"[错误] 未找到 {PROFILE_PATH}（必须在当前工作目录运行）"

    if not section.strip():
        titles = sorted({k for k in sections.keys() if re.match(r"^\d+\.", k)})
        return "可查询的章节：\n" + "\n".join(f"  - {t}" for t in titles)

    key = section.strip().lstrip("§").strip()
    if key in sections:
        body = sections[key]
        return body[:max_chars] + ("\n...[已截断]" if len(body) > max_chars else "")

    # 兜底：包含匹配
    for k, v in sections.items():
        if key in k:
            return v[:max_chars] + ("\n...[已截断]" if len(v) > max_chars else "")

    titles = sorted({k for k in sections.keys() if re.match(r"^\d+\.", k)})
    return (
        f"未找到匹配章节：'{section}'\n"
        f"可用章节：\n" + "\n".join(f"  - {t}" for t in titles)
    )


# =====================================================
# 工具注册表
# =====================================================

# name -> python function（运行时通过名字路由）
TOOL_FUNCTIONS = {
    "get_current_time": tool_get_current_time,
    "calculator": tool_calculator,
    "echo": tool_echo,
    "simple_profile_lookup": tool_simple_profile_lookup,
}

# OpenAI 兼容的 tools schema（传给 LLM 让它知道有哪些工具、怎么调）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前本地时间（精确到秒）。无参数。",
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
            "name": "calculator",
            "description": "对一个简单数学表达式求值。支持加减乘除、幂、取余和括号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要求值的数学表达式，例如 '1234 * 5678' 或 '(3+4)*5'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "原样回显一段文本。用于测试工具调用链路是否贯通。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要回显的文本内容",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simple_profile_lookup",
            "description": (
                "从 user_profile.md 抽取一个章节的内容，用于回答关于用户画像"
                "（基础信息 / 求职方向 / 技能清单 / 项目经历 / 实习经历 / 可投入时间 等）的问题。"
                "用户问到自己的任何画像信息时优先调用本工具，而不是凭记忆作答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": (
                            "章节标识。支持：编号（如 '1' 或 '§1'）/ 标题（如 '基础信息' "
                            "或 '1. 基础信息'）/ 空字符串（返回所有可用章节列表）"
                        ),
                    },
                },
                "required": ["section"],
            },
        },
    },
]
