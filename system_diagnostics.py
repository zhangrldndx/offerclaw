# -*- coding: utf-8 -*-
"""顶部问答的只读诊断能力。

本模块只检查事实源、审批状态、关联完整性和检索运行配置，不修改文件、不重建
索引，也不调用 LLM。诊断是一种稳定服务模式，不为每个失败说法创建新工具。
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_REGISTRY_VERSION = "system-diagnostics-v1"

_FAILURE_RE = re.compile(
    r"为什么.{0,24}(?:没找到|找不到|没有结果|没结果|不能|无法|失败|异常|没反应|"
    r"不对|错误|未覆盖|变成通用|没有证据)|"
    r"(?:检索|路由|知识库|问答|关联|匹配|计划).{0,18}(?:失败|异常|不对|错误|没反应|"
    r"没有结果|未覆盖)|"
    r"(?:为什么).{0,24}(?:没有|没能|未能|尚未).{0,8}"
    r"(?:被)?(?:检索|读取|关联|更新|识别|加入).{0,6}(?:出来|到|成功)?|"
    r"(?:哪里|哪一步).{0,12}(?:出错|有问题)|为什么这次"
)
_CONCEPT_WHY_RE = re.compile(
    r"为什么(?:要|需要|应该|能够|可以|会).{0,20}(?:学习|使用|采用|设计|实现|检索|重排)"
)


def _normalise(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(text or "").strip().lower())


def _topics(q: str) -> list[str]:
    topics: list[str] = []
    if any(x in q for x in ("项目", "个人记忆")):
        topics.append("project")
    if any(x in q for x in ("jd", "职位描述", "岗位要求", "关联")):
        topics.append("application_jd")
    if any(x in q for x in ("学习计划", "纳入计划", "计划目标", "计划")):
        topics.append("plan")
    if any(x in q for x in ("画像", "掌握", "能力证据")):
        topics.append("profile")
    if any(x in q for x in ("学习记录", "执行记录", "复盘", "历史记录")):
        topics.append("reflection")
    if any(x in q for x in (
            "知识库", "资料", "论文", "检索", "路由", "问答", "通用回答", "未覆盖", "证据")):
        topics.append("routing")
    return list(dict.fromkeys(topics)) or ["routing"]


def detect_diagnostic_request(question: str) -> dict[str, Any]:
    """识别明确的系统/数据失败诊断；普通“为什么”知识题不得进入此路由。"""
    q = _normalise(question)
    if not q or _CONCEPT_WHY_RE.search(q) and not any(
            x in q for x in ("没有结果", "没找到", "不对", "失败", "异常", "未覆盖")):
        return {"matched": False, "topics": [], "confidence": 0.0}
    if not _FAILURE_RE.search(q):
        return {"matched": False, "topics": [], "confidence": 0.0}
    return {
        "matched": True,
        "topics": _topics(q),
        "confidence": 0.97,
        "reason": "识别到 OfferClaw 检索结果、数据关联或功能前置条件诊断",
        "registry_version": DIAGNOSTIC_REGISTRY_VERSION,
    }


def _pending_project_count() -> int:
    pending = BASE_DIR / "knowledge_base" / "_pending"
    if not pending.exists():
        return 0
    count = 0
    for path in pending.rglob("*.md"):
        if "_archived" in path.parts:
            continue
        try:
            head = path.read_text(encoding="utf-8")[:1600]
        except OSError:
            continue
        if re.search(r"^source_type:\s*[\"']?project_context", head, re.M | re.I):
            count += 1
    return count


def _project_diagnostic() -> list[str]:
    try:
        from project_memory import list_project_catalog
        projects = list_project_catalog()
    except Exception:
        projects = []
    pending = _pending_project_count()
    lines = [f"- 已确认、可被个人项目路由读取：{len(projects)} 个。"]
    if projects:
        lines.append("- 当前目录：" + "、".join(
            str(item.get("name") or item.get("project_id") or "未命名") for item in projects[:8]
        ))
    lines.append(f"- 待审批的个人项目候选：{pending} 个；待审批内容不会作为个人事实回答。")
    if not projects and pending:
        lines.append("- 最可能原因：项目仍在待确认区，尚未批准进入个人项目事实源。")
    elif not projects:
        lines.append("- 最可能原因：当前没有已确认项目材料，或事实源文件不可读。")
    return lines


def _application_diagnostic(include_plan: bool) -> list[str]:
    try:
        from applications_store import application_fact_views
        applications = application_fact_views()
    except Exception:
        applications = []
    linked = [item for item in applications if item.get("jd_version_id")]
    matched = [item for item in linked if item.get("match_id")]
    lines = [
        f"- 投递记录：{len(applications)} 条；已关联活动 JD：{len(linked)} 条；已有匹配快照：{len(matched)} 条。"
    ]
    missing_jd = [f"{x.get('company')}·{x.get('position')}" for x in applications
                  if not x.get("jd_version_id")]
    if missing_jd:
        lines.append("- JD 未关联：" + "、".join(missing_jd[:8]))
    if include_plan:
        try:
            from application_jd_store import plan_targets
            targets = plan_targets()
        except Exception:
            targets = {"included": [], "excluded": []}
        lines.append(f"- 当前纳入 JD 驱动计划：{len(targets.get('included') or [])} 条。")
        reasons: dict[str, int] = {}
        for item in targets.get("excluded") or []:
            reason = str(item.get("excluded_reason") or "未记录原因")
            reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            lines.append("- 排除原因：" + "；".join(f"{key} {value} 条" for key, value in reasons.items()))
    return lines


def _profile_reflection_diagnostic() -> list[str]:
    profile = BASE_DIR / "user_profile.md"
    daily = BASE_DIR / "daily_log.md"
    summaries = BASE_DIR / "summaries"
    reflection_count = len(list(summaries.glob("*.md"))) if summaries.exists() else 0
    try:
        from profile_store import list_suggestions
        pending = len(list_suggestions("pending"))
    except Exception:
        pending = 0
    return [
        f"- 正式画像：{'可读' if profile.exists() else '缺失'}；待确认画像建议：{pending} 条。",
        f"- 每日执行文件：{'可读' if daily.exists() else '缺失'}；完整复盘文件：{reflection_count} 份。",
        "- 完成学习或复盘只形成能力证据；用户未批准前不会自动写成“已掌握”。",
    ]


def _routing_diagnostic() -> list[str]:
    try:
        from rag_tools import has_embedding_api_key, index_fingerprint
        fingerprint = index_fingerprint()
        embedding = "可用" if has_embedding_api_key() else "未配置远程密钥/使用本地配置"
    except Exception:
        fingerprint, embedding = {}, "状态读取失败"
    count = fingerprint.get("collection_count")
    count_label = "不可用" if count is None else str(count)
    return [
        "- 当前路由模式：semantic_v4；所有自然语言均由一次结构化 LLM 规划，"
        "失败时保守澄清。",
        f"- Embedding：{embedding}；主知识库 collection："
        f"{fingerprint.get('collection') or '未读取'}；索引块数：{count_label}。",
        "- 个人事实路由无证据时禁止用通用知识替代；只有 Explain 模式允许明确标注后降级。",
    ]


def render_diagnostics(topics: list[str]) -> str:
    """返回可审计的本地诊断摘要；不会执行修复。"""
    selected = list(dict.fromkeys(topic for topic in topics if topic)) or ["routing"]
    lines = [
        "### OfferClaw 只读诊断",
        "",
        "本次只检查数据和运行状态，不会修改记录、重建索引或替你执行卡片操作。",
    ]
    if "project" in selected:
        lines.extend(["", "#### 个人项目", *_project_diagnostic()])
    if "application_jd" in selected or "plan" in selected:
        lines.extend(["", "#### 投递、JD 与计划前置条件",
                      *_application_diagnostic("plan" in selected)])
    if "profile" in selected or "reflection" in selected:
        lines.extend(["", "#### 画像与长期复盘", *_profile_reflection_diagnostic()])
    if "routing" in selected:
        lines.extend(["", "#### 检索运行状态", *_routing_diagnostic()])
    lines.extend([
        "",
        "下一步请在对应功能卡片补齐或确认数据；顶部问答保持只读，不会自动修复。",
    ])
    return "\n".join(lines)
