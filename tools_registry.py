# -*- coding: utf-8 -*-
"""tools_registry.py — OfferClaw Tool Ecology（V4 §3）

把 ``match_job / job_discovery / career_agent / resume_builder`` 等能力以
**OpenAI 兼容的 function calling schema** 暴露成 ``Tool``，再用一个
极简注册表收口。所有上层调度（ReAct agent / 未来的 MCP server / 第三方
脚本）都从同一个 ``REGISTRY`` 拿工具，避免每个调度方各自写一遍 schema。

设计原则：
1. **不引入新依赖**：仅复用项目既有模块。
2. **schema 与执行函数捆绑在一处**：避免漂移。
3. **失败可恢复**：任何 tool 抛异常都包成 ``{"error": str}``，
   ReAct loop 不会被卡住。
4. **零 LLM 依赖**：每个 tool 本体都是确定性逻辑（match_jd 的 LLM 版用
   ``resume_build`` 另一个 tool 表达），方便 CI / 单测全程不读 KEY。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# =====================================================
# Tool 数据结构
# =====================================================

@dataclass
class Tool:
    """与 OpenAI / 智谱 function calling 兼容的 Tool 描述。"""
    name: str
    description: str
    parameters: dict  # JSON Schema, OpenAI 风格
    fn: Callable[..., Any]

    def to_openai_schema(self) -> dict:
        """生成 ``tools=[...]`` 数组里的单个 schema 元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, **kwargs) -> dict:
        """统一调用：失败包成 {"error": ...}，永远返回 dict。
        A7: 加墙钟预算（``TOOL_TIMEOUT`` 默认 30s）——工具死循环/阻塞 IO 不再无限期挂死 ReAct loop。"""
        import os
        import concurrent.futures
        timeout = float(os.environ.get("TOOL_TIMEOUT", "30"))

        def _run():
            out = self.fn(**kwargs)
            return out if isinstance(out, dict) else {"result": out}

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(_run).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {"error": "tool_timeout", "tool": self.name, "timeout_s": timeout}
        except TypeError as e:
            return {"error": f"参数不匹配：{e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        finally:
            ex.shutdown(wait=False)  # 不等卡住线程，避免反被阻塞（Python 同步硬超时的已知代价：线程泄漏）


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"重复注册 tool：{tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"未注册 tool：{name}")
        return self.tools[name]

    def list_names(self) -> list[str]:
        return sorted(self.tools.keys())

    def to_openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tools.values()]


# =====================================================
# Tool 实现
# =====================================================

def _tool_match_jd(jd_text: str, jd_title: str = "未命名 JD") -> dict:
    """对一段 JD 跑规则版三档结论 + 缺口（不调 LLM）。"""
    from match_job import run_match
    from jd_parser import analyze_jd
    from profile_loader import load_profile

    profile = load_profile()
    analysis = analyze_jd(jd_text).model_dump(mode="json")
    report = run_match(
        profile, jd_text, jd_title=jd_title, jd_analysis=analysis,
    )
    return {
        "status": report.conclusion,
        "direction": report.direction,
        "gap_count": sum(len(v) for v in report.gap_list.values()
                         if isinstance(v, list)),
        "gap_list": report.gap_list,
        "reason": report.conclusion_reason,
    }


def _tool_extract_jd(raw: str) -> dict:
    """把 JD 原文抽成 ``{company, title, location, skills_detected, ...}``。"""
    from job_discovery import extract_jd
    out = extract_jd(raw)
    # 控制返回体积，确保 ReAct loop 上下文不爆
    if "duties" in out:
        out["duties"] = out["duties"][:300]
    if "requirements" in out:
        out["requirements"] = out["requirements"][:300]
    if "jd_text" in out:
        out.pop("jd_text", None)
    return out


def _tool_resume_skeleton(jd_text: str) -> dict:
    """JD-aware 简历骨架（不调 LLM）。复用 career_flow.resume_node 的逻辑。"""
    from jd_parser import analyze_jd
    from profile_loader import load_profile

    profile = load_profile()
    keywords = analyze_jd(jd_text, mode="deterministic").keyword_names()
    must = [
        k for k in keywords
        if any(k.lower() in s.lower() for s in
               profile.get("熟练技能", []) + profile.get("会用技能", []))
    ]
    return {
        "headline": (
            f"{profile.get('学历', '')} · {profile.get('专业', '')} · "
            f"目标方向 {(profile.get('方向优先级') or ['未指定'])[0]}"
        ),
        "keywords_hit": keywords,
        "must_emphasize": must,
        "mode": "skeleton",
    }


def _tool_today_advice() -> dict:
    """聚合投递池 + 日志 + 状态机的"今天最该做什么"。"""
    from career_agent import get_today_advice
    return dict(get_today_advice())


def _tool_list_applications(limit: int = 10) -> dict:
    """读 applications.md 拿真实投递记录（过滤 ``[DEMO]``）。"""
    from career_agent import parse_applications, _read, APPLICATIONS_PATH
    rows = parse_applications(_read(APPLICATIONS_PATH))
    return {"total": len(rows), "items": rows[:limit]}


def _tool_career_flow(jd_text: str, jd_title: str = "未命名 JD") -> dict:
    """跑完整 CareerFlow（条件路由版，10 个业务节点）。

    此前写"8 节点"，那是 UI 折叠后的可见阶段数；图里另有 jd_analyze 与 critic。
    """
    from career_flow import run_career_flow_routed
    out = run_career_flow_routed(jd_text, jd_title=jd_title, skip_llm=True)
    # 精简返回体，去掉冗长 trace
    return {
        "status": out.get("match_report", {}).get("status"),
        "direction": out.get("match_report", {}).get("direction"),
        "gap_total": out.get("gaps", {}).get("total"),
        "route_taken": out.get("route_taken"),
        "today_headline": out.get("today_advice", {}).get("headline"),
        "resume_keywords": out.get("resume_skeleton", {}).get("keywords_hit", []),
        "application_decision": out.get("application_suggestion", {}).get("decision"),
        "confirm_pending": len(out.get("requires_confirmation", [])),
        "errors": out.get("errors", []),
    }


# =====================================================
# 默认注册表
# =====================================================

def _build_default_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool(
        name="match_jd",
        description="对一段 JD 跑规则版三档结论 + 缺口（不调 LLM）。"
                    "返回 status / direction / gap_count / gap_list。",
        parameters={
            "type": "object",
            "properties": {
                "jd_text": {"type": "string", "description": "JD 全文"},
                "jd_title": {"type": "string", "description": "JD 标题（可选）"},
            },
            "required": ["jd_text"],
        },
        fn=_tool_match_jd,
    ))
    r.register(Tool(
        name="extract_jd",
        description="把 JD 原文抽成结构化字段：公司 / 岗位 / 地点 / 技能命中等。",
        parameters={
            "type": "object",
            "properties": {
                "raw": {"type": "string", "description": "JD 原始文本"},
            },
            "required": ["raw"],
        },
        fn=_tool_extract_jd,
    ))
    r.register(Tool(
        name="resume_skeleton",
        description="基于用户画像 + JD 关键词生成简历段落骨架（不调 LLM）。",
        parameters={
            "type": "object",
            "properties": {
                "jd_text": {"type": "string", "description": "JD 全文"},
            },
            "required": ["jd_text"],
        },
        fn=_tool_resume_skeleton,
    ))
    r.register(Tool(
        name="today_advice",
        description="聚合 applications.md / daily_log.md 给出"
                    "「今天最该做什么」。无参数。",
        parameters={"type": "object", "properties": {}},
        fn=_tool_today_advice,
    ))
    r.register(Tool(
        name="list_applications",
        description="列出 applications.md 中的真实投递记录（自动过滤 DEMO）。",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
            },
        },
        fn=_tool_list_applications,
    ))
    r.register(Tool(
        name="career_flow",
        description="对一份 JD 跑完整 CareerFlow（条件路由版，10 个业务节点），"
                    "返回 status / direction / gap_total / today_headline / "
                    "resume_keywords / application_decision。",
        parameters={
            "type": "object",
            "properties": {
                "jd_text": {"type": "string"},
                "jd_title": {"type": "string"},
            },
            "required": ["jd_text"],
        },
        fn=_tool_career_flow,
    ))
    return r


REGISTRY: ToolRegistry = _build_default_registry()


if __name__ == "__main__":
    import json
    print(json.dumps(REGISTRY.to_openai_schemas(), ensure_ascii=False, indent=2))
