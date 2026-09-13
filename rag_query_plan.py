# -*- coding: utf-8 -*-
"""OfferClaw 顶部问答规划器。

生产自由文本统一经过版本化的结构化语义规划；规则规划和原型排序仅供离线评测。
本模块不执行检索或写操作，模型输出还要经过路由及动作能力注册表校验。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
import re
import time
from typing import Any, Callable

from action_capabilities import capability_topics
from product_help import detect_product_help_request, is_explicit_knowledge_content_lookup
from rag_route_registry import (READ_SOURCE_REGISTRY, SERVICE_MODES, SERVICE_REGISTRY,
                                SOURCE_ROLES, get_route_definition,
                                rank_route_candidates, registry_summary,
                                route_output_contracts)
from semantic_query_planner import (
    SEMANTIC_PLANNER_VERSION, SEMANTIC_QUERY_SCHEMA_VERSION,
    SemanticPlanningOutcome, generic_operation_for_route, plan_semantically,
    semantic_personal_scope, semantic_task_answer_objects,
    semantic_task_output_contracts, semantic_task_route_roles,
)
from system_diagnostics import detect_diagnostic_request


ROUTE_OPERATIONS: dict[str, set[str]] = {}
for _definition in READ_SOURCE_REGISTRY:
    ROUTE_OPERATIONS.setdefault(_definition.source, set()).add(_definition.operation)

PERSONAL_ROUTES = frozenset(
    definition.source for definition in READ_SOURCE_REGISTRY if definition.personal
)


@dataclass(frozen=True)
class RouteSpec:
    source: str
    operation: str
    subquery: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    required: bool = False
    depends_on: tuple[str, ...] = ()
    bindings: dict[str, str] = field(default_factory=dict)
    route_id: str = ""
    source_role: str = "answer_source"
    selection_reason: str = ""
    locked: bool = False


@dataclass
class RequestFrame:
    """与具体数据源解耦的统一只读请求框架。

    ``personal_scope`` 回答“是否在问用户本人”，``domains`` 回答“问哪类信息”，
    ``tasks`` 回答“希望怎样组织答案”。先形成框架再选择路由，可避免把“以前的工作”
    之类弱表达误当成普通知识问题。
    """

    service_mode: str = "explain"  # guide | recall | explain | advise | diagnose
    interaction_kind: str = "query"  # query | command | how_to
    turn_relation: str = "standalone"  # standalone | continuation | correction
    relation_target_turn_id: str = ""  # server-authored; never accepted from the model
    action_request: dict[str, Any] = field(default_factory=dict)
    requested_action: str = ""      # 仅识别越界写请求，不代表顶部问答会执行
    personal_scope: str = "none"  # none | current | history | artifact | application
    domains: list[str] = field(default_factory=list)
    time_scope: dict[str, str] = field(default_factory=lambda: {
        "kind": "none", "date_from": "", "date_to": "",
    })
    tasks: list[str] = field(default_factory=list)
    answer_object: str = ""
    answer_objects: list[str] = field(default_factory=list)
    output_contracts: list[dict[str, str]] = field(default_factory=list)
    capability_ids: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    source_roles: list[dict[str, str]] = field(default_factory=list)
    usage_contexts: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    unresolved_references: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    deterministic_signals: list[str] = field(default_factory=list)
    routing_assurance: str = "degraded"
    decision_reasons: list[str] = field(default_factory=list)
    context_resolution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    planner_mode: str
    routes: list[RouteSpec]
    intent_frame: RequestFrame = field(default_factory=RequestFrame)
    entities: dict[str, Any] = field(default_factory=dict)
    answer_sections: list[str] = field(default_factory=list)
    reason: str = ""
    decision: str = "answer"  # answer | clarify | fallback
    resolver_mode: str = "llm"  # structured_read | semantic | llm | fallback
    candidate_routes: list[dict[str, Any]] = field(default_factory=list)
    clarification: str = ""
    confidence_factors: dict[str, Any] = field(default_factory=dict)
    planner_engine: str = "rules"
    planner_version: str = SEMANTIC_PLANNER_VERSION
    schema_version: str = "query-plan-v1"
    route_model: str = ""
    repair_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "planner_mode": self.planner_mode,
            "routes": [asdict(r) for r in self.routes],
            "intent_frame": self.intent_frame.to_dict(),
            "entities": self.entities,
            "answer_sections": self.answer_sections,
            "reason": self.reason,
            "decision": self.decision,
            "resolver_mode": self.resolver_mode,
            "candidate_routes": self.candidate_routes,
            "clarification": self.clarification,
            "confidence_factors": self.confidence_factors,
            "planner_engine": self.planner_engine,
            "planner_version": self.planner_version,
            "schema_version": self.schema_version,
            "route_model": self.route_model,
            "repair_used": self.repair_used,
            "fallback_reason": self.fallback_reason,
        }


_APPLICATION_HINTS = (
    "投了", "投过", "投哪", "哪几家", "岗位进度", "投递状态", "投递记录", "已投递",
    "投递进度", "投递池", "求职进度", "官网投", "待投", "还没投", "还需要投", "准备投", "面试进度",
    "offer 状态", "offer状态", "拿到 offer", "收到 offer",
    "被拒", "拒绝", "失败企业", "挂了", "没通过", "官网链接", "投出去",
    "投递哪些", "要投递哪些", "准备投哪些", "近期投递", "最近要投", "已经投递",
)
_EXPERIENCE_HINTS = (
    "经验总结", "我的经验", "投递经验", "失败企业经验", "亲历", "面经", "笔试经验", "面试经验",
    "失败原因", "被拒原因", "复盘", "教训", "面试中总结", "之前面试",
)
_REFLECTION_HINTS = (
    "学习复盘", "每日复盘", "日复盘", "周复盘", "执行记录", "执行情况", "历史学习",
    "以前学", "曾经学", "什么时候学", "什么时候做", "哪里卡住", "卡在哪里", "反复卡",
    "为什么建议", "减少任务", "降低任务", "任务量", "计划偏离", "偏离度", "学习阻碍",
    "复盘记录", "学习记录", "掌握了吗", "掌握了么", "掌握没有", "实践证据", "能力证据",
)
_HISTORY_HINTS = (
    "之前", "此前", "前面", "过去", "以往", "曾经", "以前", "一路以来",
    "前一段时间", "过往", "历史上我", "到目前为止",
)
_HISTORICAL_ACTION_HINTS = (
    "做过", "做了", "所做", "完成", "推进", "实践", "练习", "学习", "学过", "工作",
    "经历", "尝试", "卡住", "遇到", "处理过", "实现过", "复盘",
)
_SUMMARY_HINTS = ("总结", "回顾", "梳理", "盘点", "概括")
_CURRENT_HINTS = ("现在", "目前", "当前", "今天", "本周", "最近")
_ANAPHORA_HINTS = (
    "刚才", "前面那个", "上面那个", "这些内容", "这个部分", "上述内容", "那一点",
    "这个呢", "那个呢", "这些呢", "上述呢", "前者", "后者", "这些公司", "这些企业",
    "这些岗位", "这些经历", "上述公司", "上述企业", "上述岗位",
)
_PROFILE_HINTS = (
    "我的画像", "我的情况", "我的状态", "我的记录", "我目前", "我现在",
    "学习计划", "当前计划", "计划进度", "最近学", "最近做", "学习留痕",
    "能力缺口", "缺少什么能力", "欠缺什么", "还缺什么", "我的能力", "下一步学习", "学习建议",
    "求职方向", "日志", "本周", "今天", "昨天", "学了什么", "做了什么",
)
_PROJECT_HINTS = (
    "我的项目", "项目材料", "项目内容", "项目经历", "项目记忆",
    "我做的项目", "offerclaw 项目", "offerclaw项目",
)
_RESUME_HINTS = (
    "我的简历", "简历格式", "简历模板", "简历写法", "写作指导",
    "按我的格式", "按什么格式", "项目经历怎么写", "简历规则",
)
_REFERENCE_HINTS = (
    "什么是", "解释", "概念", "原理", "技术应用", "技术知识", "如何实现",
    "参考资料", "学习资料", "结合资料", "结合我提供", "资料说明", "博客", "教程", "知识库",
    "普及", "介绍一下", "学习路径", "注意事项", "注意什么", "需要注意", "要注意",
    "避坑", "岗位要求", "任职要求", "核心能力", "需要哪些能力", "需要什么能力",
    "岗位职责", "招聘流程", "面试流程",
)
_PAPER_HINTS = ("论文", "文献", "paper", "学术研究", "研究证据", "原论文")
_NEXT_HINTS = ("下一步", "接下来", "还需要", "待办", "动作", "要去")
_PENDING_HINTS = (
    "还没投", "还需要投", "待投", "准备投", "官网投", "官网（投递", "需要去",
    "要投递", "投递哪些", "准备投哪些", "近期投递", "最近要投",
)
_APPLIED_HINTS = ("投了哪", "投过哪", "投了几", "投过几", "投过", "已经投", "已经投递", "已投递", "投出去", "哪几家")
_FAILED_HINTS = ("失败企业", "失败公司", "失败投递", "失败的投递", "被拒", "拒绝", "挂了", "没通过")
_ADVICE_HINTS = (
    "注意事项", "注意什么", "需要注意", "要注意", "怎么准备", "如何准备", "准备方法",
    "避坑", "后续怎么跟进", "如何跟进", "投递建议", "面试建议",
)
_ROLE_KNOWLEDGE_HINTS = (
    "岗位要求", "任职要求", "核心能力", "需要哪些能力", "需要什么能力", "所需技术",
    "技能要求", "岗位职责", "招聘流程", "面试流程", "会问什么", "考察什么",
)
_APPLICATION_ANAPHORA = (
    "该企业", "该公司", "该岗位", "这些企业", "这些公司", "这些岗位", "上述企业",
    "上述公司", "上述岗位", "对应企业", "对应公司", "对应岗位", "它们", "这些经历",
)
_APPLICATION_JD_HINTS = (
    "目标jd", "目标 jd", "投递jd", "投递 jd", "对应jd", "对应 jd", "当时的jd", "当时 jd",
    "绑定jd", "绑定 jd", "投递绑定的jd", "投递绑定的 jd",
    "jd要求", "jd 要求", "jd职责", "jd 职责", "职位描述", "岗位原文", "岗位版本",
    "为什么学", "为什么要学", "对应哪个岗位", "哪些岗位要求",
)


def _normalise(question: str) -> str:
    q = (question or "").strip().lower()
    q = re.sub(r"[\s\u3000]+", " ", q)
    synonym_rules = (
        (r"已经(?:被)?审批", "已审批"),
        (r"经过审批", "已审批"),
        (r"已经(?:被)?批准", "已批准"),
        (r"尚未投递", "还没投"),
        (r"未投递", "还没投"),
    )
    for pattern, replacement in synonym_rules:
        q = re.sub(pattern, replacement, q)
    return q


def _strip_request_scaffolding(q: str) -> str:
    """移除礼貌/界面话语，避免“告诉我、请说明”被误判为业务操作。"""
    patterns = (
        r"^请(?:直接)?(?:回答|说明)[：:,，\s]*",
        r"^我想确认一下[：:,，\s]*",
        r"^麻烦帮我(?:看一下|看看|确认)[：:,，\s]*",
        r"^请帮我(?:核对|确认|看看)[：:,，\s]*",
        r"^有个问题[：:,，\s]*",
        r"^我需要了解[：:,，\s]*",
        r"^能否告诉我[：:,，\s]*",
        r"^请按现有信息回答[：:,，\s]*",
    )
    out = q
    for pattern in patterns:
        out = re.sub(pattern, "", out, count=1)
    return out.strip()


def _contains_any(q: str, hints: tuple[str, ...]) -> bool:
    return any(h.lower() in q for h in hints)


def _is_project_inventory_question(q: str, domains: list[str], tasks: list[str]) -> bool:
    """识别“项目有哪些”；投递/简历可以只是用途，不能反客为主成为事实源。"""
    if "project" not in domains or "list" not in tasks:
        return False
    if any(x in q for x in ("项目岗位", "项目岗", "公司项目", "企业项目")):
        return False
    if any(x in q for x in (
            "量化结果", "技术栈", "用了哪些", "如何实现", "架构", "难点", "指标",
            "功能", "贡献", "代码", "实现细节", "项目内容")):
        return False
    if ("投递" in q or "投了" in q or "投过" in q) and any(
            x in q for x in ("公司", "企业", "岗位", "状态", "进度", "记录", "哪几家")):
        return False
    return bool(re.search(
        r"(?:哪些|哪几个|有什么|列出|列一下).{0,14}(?:开源|个人|已审批|已确认)?项目|"
        r"项目(?:有)?(?:哪些|哪几个|有什么)|"
        r"(?:做过的|可用的?|能用的?|用于.{0,8}的?|开源的?).{0,14}项目.{0,3}(?:有哪些|有什么)|"
        r"项目(?:清单|列表|目录)|项目记忆(?:中)?(?:有哪些|有什么)\s*[？?。]*$",
        q,
    ))


def _has_project_choice_predicate(q: str) -> bool:
    """识别“为某次求职选择哪个项目”的决策谓词。

    这里判断的是语义角色，不把“项目”“投递”等领域词本身当成路由依据。
    历史事实询问（例如“当时用了哪个项目”）也必须与面向未来的适配决策分开。
    """
    if re.search(
        r"(?:已经|曾经|当时|之前|上次).{0,8}"
        r"(?:用了|使用了|采用了|选了|选择了|主推了|放了).{0,8}"
        r"(?:哪个|哪一个|什么|项目)",
        q,
    ):
        return False
    return bool(
        any(x in q for x in (
            "推荐", "主推项目", "选择项目", "选用项目", "项目匹配", "项目适配",
            "使用哪个", "采用哪个", "用哪个", "拿哪个", "选哪个", "选择哪个",
            "项目更合适", "项目更适合", "更合适", "更适合", "更匹配", "更有说服力",
        ))
        or re.search(
            r"(?:应该|应当|该|可以|需要|要|建议|推荐).{0,10}"
            r"(?:使用|采用|选(?:择|用)?|主推|拿|放|展示|用).{0,6}"
            r"(?:哪个|哪一个|什么)?(?:已有|个人|开源)?项目",
            q,
        )
        or re.search(
            r"(?:使用|采用|选(?:择|用)?|主推|拿|放|展示|用).{0,5}"
            r"(?:哪个|哪一个|什么)(?:已有|个人|开源)?项目",
            q,
        )
        or re.search(
            r"(?:哪个|哪一个|什么)(?:已有|个人|开源)?项目.{0,10}"
            r"(?:合适|适合|匹配|有说服力|作为.{0,4}证据)",
            q,
        )
        or re.search(
            r"(?:offerclaw|localflow).{0,12}(?:还是|或).{0,12}"
            r"(?:offerclaw|localflow).{0,10}(?:合适|适合|匹配|选|用)",
            q,
        )
    )


def build_intent_frame(question: str, context: list[str] | None = None) -> RequestFrame:
    """把自然语言归一化成与存储实现无关的意图框架；纯本地、无数据读取。"""
    q = _normalise(question)
    task_q = _strip_request_scaffolding(q)
    context = [str(x).strip()[:500] for x in (context or []) if str(x).strip()][-3:]
    unresolved = [hint for hint in _ANAPHORA_HINTS if hint in q]
    context_q = _normalise("；".join(context)) if unresolved and context else ""
    semantic_q = f"{context_q}；{q}" if context_q else q
    semantic_core = f"{context_q}；{task_q}" if context_q else task_q

    content_lookup = is_explicit_knowledge_content_lookup(task_q)
    diagnostic = {} if content_lookup else detect_diagnostic_request(task_q)
    if diagnostic.get("matched"):
        return RequestFrame(
            service_mode="diagnose",
            personal_scope="current" if any(x in task_q for x in (
                "我的", "这条投递", "项目", "画像", "学习记录", "计划")) else "none",
            domains=["system_diagnostics"],
            time_scope={"kind": "none", "date_from": "", "date_to": ""},
            tasks=["diagnose"],
            answer_object="system_diagnostic",
            answer_objects=["system_diagnostic"],
            operations=["diagnose"],
            filters={
                "topics": list(diagnostic.get("topics") or ["routing"]),
                "registry_version": str(diagnostic.get("registry_version") or ""),
            },
            confidence=float(diagnostic.get("confidence") or 0.97),
            unresolved_references=unresolved,
            signals=["diagnostic_request"],
            deterministic_signals=["explicit_diagnostic_request"],
        )

    product_help_context = _normalise("；".join(context)) if context else ""
    product_help = detect_product_help_request(task_q, context=product_help_context)
    if product_help.get("matched"):
        topics = list(product_help.get("topics") or [])
        confidence = float(product_help.get("confidence") or 0.94)
        request_kind = str(product_help.get("request_kind") or "guide")
        return RequestFrame(
            service_mode="guide",
            requested_action=str(product_help.get("action") or "locate"),
            personal_scope="none",
            domains=["product_help"],
            time_scope={"kind": "none", "date_from": "", "date_to": ""},
            tasks=["help"],
            answer_object="system_capability",
            answer_objects=["system_capability"],
            operations=["guide"],
            usage_contexts=[],
            filters={
                "topics": topics,
                "action": str(product_help.get("action") or "locate"),
                "request_kind": request_kind,
                "registry_version": str(product_help.get("registry_version") or ""),
            },
            confidence=confidence,
            unresolved_references=unresolved,
            signals=["product_help_correction" if request_kind == "correction" else "product_help_request"],
            deterministic_signals=[
                "explicit_product_correction"
                if request_kind == "correction" else "explicit_product_operation"
            ],
        )

    explicit_personal = (any(h in task_q for h in ("我", "我的", "本人", "自己", "帮我", "为我"))
                         or bool(context_q and any(h in context_q for h in ("我", "我的", "本人", "自己"))))
    history = _contains_any(q, _HISTORY_HINTS) or bool(context_q and _contains_any(context_q, _HISTORY_HINTS))
    historical_action = (_contains_any(q, _HISTORICAL_ACTION_HINTS)
                         or bool(context_q and _contains_any(context_q, _HISTORICAL_ACTION_HINTS)))
    current = _contains_any(q, _CURRENT_HINTS) or bool(context_q and _contains_any(context_q, _CURRENT_HINTS))
    implicit_personal_history = bool(
        not explicit_personal and history and historical_action
        and any(x in q for x in ("哪一天", "哪次", "记录", "做过什么", "完成过什么"))
    )
    implicit_personal_current = bool(
        not explicit_personal
        and any(x in q for x in (
            "当前计划", "学习计划", "计划进行到", "今天的计划", "今日计划", "画像状态",
        ))
    )
    definitional_question = (
        any(x in q for x in ("什么是", "是什么"))
        and not any(x in q for x in (
            "当前", "现在", "接下来", "下一步", "我的", "这条", "这些",
        ))
    )
    general_method = (
        any(x in q for x in ("一般", "通常", "应届生", "方法"))
        or definitional_question
    )
    implicit_application_current = bool(
        not explicit_personal and not general_method
        and any(x in q for x in (
            "投", "申请", "应聘", "求职", "投递", "岗位", "公司", "企业",
        ))
        and any(x in q for x in (
            "投递", "投过", "准备投", "需要投", "投简历", "待投", "官网投递",
            "投递记录", "投递状态", "求职进度", "已拒绝", "被拒", "面试中",
            "等待反馈", "主动放弃", "offer",
        ))
        and any(x in q for x in (
            "当前", "现在", "接下来", "还有哪些", "哪些", "哪几", "下一步",
            "进度", "状态", "待投", "已投", "被拒", "失败",
        ))
    )
    implicit_personal_artifact = bool(
        not explicit_personal and not general_method
        and any(x in q for x in (
            "项目材料", "offerclaw项目", "offerclaw 项目", "项目的架构记录",
            "简历模板", "简历规则", "写作指导", "亲历材料", "经验总结为空",
            "之前面试", "亲历", "失败原因", "能力缺口",
        ))
    )
    implicit_application_project_fit = bool(
        any(x in semantic_q for x in ("投递", "申请", "应聘", "岗位", "职位", "jd", "投"))
        and any(x in semantic_q for x in ("项目", "offerclaw", "localflow"))
        and _has_project_choice_predicate(semantic_q)
    )
    personal = (explicit_personal or implicit_personal_history
                or implicit_personal_current or implicit_personal_artifact
                or implicit_application_current or implicit_application_project_fit)
    personal_history_negated = bool(re.search(
        r"(?:没|没有|从未|未曾).{0,8}(?:学|做|接触|实践|经历|记录)", q
    ))
    personal_forbidden = bool(re.search(
        r"(?:不要|不用|无需|不需要|不查|别查|别)[^，,。；;？！?]{0,14}"
        r"(?:我的|个人|记录|画像|经历|复盘|项目|简历|投递)", q
    ))
    application_source_negated = bool(re.search(
        r"(?:不是|不要|不用|别)[^，,。；;？！?]{0,10}(?:面试|投递|笔试|求职)"
        r"[，,。；;？！?].{0,12}(?:学习|执行|练习|复盘)", q
    ))
    learning_source_negated = bool(re.search(
        r"(?:不是|不要|不用|别)[^，,。；;？！?]{0,10}(?:学习|执行|练习|复盘|历史记录)"
        r"[，,。；;？！?].{0,12}(?:解释|介绍|概念|原理)", q
    ))

    domains: list[str] = []
    application_terms = (
        "投递", "投了", "投过", "准备投", "申请", "应聘", "企业", "公司", "岗位", "职位",
        "面试", "笔试", "被拒", "拒绝",
    )
    if any(x in semantic_q for x in application_terms) or re.search(r"\boffer\b", semantic_q):
        domains.append("application")
    if implicit_application_project_fit and "application" not in domains:
        domains.append("application")
    if "jd" in semantic_q or any(x in semantic_q for x in ("职位描述", "任职要求", "岗位要求")):
        domains.append("application_jd" if personal else "reference")
    if any(x in semantic_q for x in ("项目", "localflow", "offerclaw")):
        domains.append("project" if personal else "reference")
    resume_specific = any(x in semantic_q for x in (
        "简历格式", "简历模板", "简历规则", "简历写法", "写作指导",
        "项目经历怎么写", "按什么格式", "按我的格式",
    ))
    if "简历" in semantic_q and ("application" not in domains or resume_specific):
        domains.append("resume" if personal else "reference")
    learning_terms = (
        "学了", "学习了", "学过", "学到", "练习", "执行", "完成", "卡住",
        "学习工作", "学习经历", "学习留痕", "学习情况", "做过的工作", "做了什么",
    )
    learning_context = (
        any(x in semantic_q for x in learning_terms)
        or ("学习" in semantic_q and (history or current)
            and any(x in semantic_q for x in (
                "记录", "经历", "过程", "内容", "进展", "做", "完成", "复盘",
            )))
    )
    generic_history_context = (
        ("复盘" in semantic_q or "工作" in semantic_q or "经历" in semantic_q)
        and not any(x in semantic_q for x in application_terms)
        and not any(x in semantic_q for x in ("项目经历", "工作经历怎么写", "简历"))
    )
    if learning_context or generic_history_context:
        domains.append("learning_execution" if personal else "reference")
    if (personal and (
            "画像" in semantic_q
            or any(x in semantic_q for x in (
                "我的能力", "我缺少", "我欠缺", "我还缺", "掌握", "我会什么", "会什么技术",
                "能力缺口", "缺少什么能力", "缺什么能力", "欠缺什么能力",
                "求职方向", "目标方向", "技能摘要",
            )))):
        domains.append("profile")
    if personal and ("计划" in semantic_q or any(x in semantic_q for x in (
            "学习建议", "下一步学习", "学习安排",
        ))):
        domains.append("plan")
    if _contains_any(semantic_core, _PAPER_HINTS):
        domains.append("paper")
    if _contains_any(semantic_core, _REFERENCE_HINTS):
        domains.append("reference")
    explicit_past_action = any(x in semantic_q for x in (
        "做过", "完成过", "推进过", "实践过", "练习过", "学过", "经历过",
        "尝试过", "卡住过", "遇到过", "处理过", "实现过",
    ))
    if (personal and (history or explicit_past_action)
            and not any(x in domains for x in ("application", "application_jd", "project", "resume"))
            and not general_method):
        domains.append("learning_execution")
    if application_source_negated:
        domains = [x for x in domains if x not in ("application", "application_jd")]
    if learning_source_negated:
        domains = [x for x in domains if x != "learning_execution"]

    tasks: list[str] = []
    summary_q = task_q.replace("经验总结", "").replace("总结材料", "")
    if _contains_any(summary_q, _SUMMARY_HINTS):
        tasks.append("reflect" if "复盘" in task_q else "summarize")
    if any(x in task_q for x in ("哪些", "哪几", "列出", "有什么")):
        tasks.append("list")
    if any(x in task_q for x in ("解释", "什么是", "原理", "介绍", "普及", "说明")):
        tasks.append("explain")
    if any(x in task_q for x in ("比较", "对比", "区别", "变化")):
        tasks.append("compare")
    recommend_intent = (
        any(x in task_q for x in (
            "下一步", "怎么做", "如何改", "怎么办", "推荐", "使用哪个",
            "选哪个", "选择哪个", "选择项目", "选用项目", "主推",
            "哪个更适合", "哪个最适合", "更适合",
            "更合适", "更匹配",
        ))
        or _has_project_choice_predicate(task_q)
        or ("建议" in task_q and not any(x in task_q for x in ("为什么建议", "建议的原因")))
    )
    if recommend_intent:
        tasks.append("recommend")
    if any(x in task_q for x in ("是否", "判断", "评估", "掌握")):
        tasks.append("evaluate")
    if ("list" in tasks and "summarize" in tasks
            and not any(x in task_q for x in ("并总结", "总结并", "同时总结"))):
        tasks = [task for task in tasks if task != "summarize"]
    if ("list" in tasks and "recommend" in tasks
            and any(x in task_q for x in ("下一步动作", "投递待办", "保存的下一步"))):
        tasks = [task for task in tasks if task != "recommend"]
    if not tasks:
        tasks.append("search")

    # 这是一类固定、可审计的关系检索，而不是普通项目搜索：先定位投递，再沿
    # application_id → jd_version_id → match_id 取得比较标准，最后选择项目证据。
    application_project_fit = bool(
        personal
        and "application" in domains
        and "project" in domains
        and (
            "recommend" in tasks
            or any(x in task_q for x in (
                "项目匹配", "项目适配", "主推项目", "项目更合适", "项目更适合",
            ))
        )
        and any(x in task_q for x in ("岗位", "职位", "投递", "申请", "应聘", "jd", "投"))
    )
    if application_project_fit and "application_jd" not in domains:
        domains.append("application_jd")

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
    if dates:
        time_scope = {"kind": "explicit_range", "date_from": min(dates), "date_to": max(dates)}
    elif history and historical_action and not personal_history_negated:
        time_scope = {"kind": "history", "date_from": "", "date_to": ""}
    elif "今天" in q or "昨天" in q or "本周" in q or "上周" in q:
        time_scope = {"kind": "today" if "今天" in q else "recent", "date_from": "", "date_to": ""}
    elif "最近" in q or "近期" in q or "这段时间" in q:
        time_scope = {"kind": "recent", "date_from": "", "date_to": ""}
    else:
        time_scope = {"kind": "none", "date_from": "", "date_to": ""}

    project_inventory = personal and _is_project_inventory_question(q, domains, tasks)
    if (project_inventory and "知识库" in semantic_core
            and not any(x in semantic_core for x in (
                "参考资料", "学习资料", "结合资料", "教程", "博客", "论文", "文献",
            ))):
        domains = [domain for domain in domains if domain != "reference"]
    explicit_application_fact = any(x in semantic_q for x in (
        "投了哪", "投过哪", "哪几家", "投递记录", "投递状态", "岗位进度",
        "准备投哪些公司", "准备投哪些岗位", "已投递公司", "被拒企业", "失败投递",
        "这些公司下一步", "这些企业下一步", "这些岗位下一步",
    ))
    experience_central = personal and any(x in q for x in (
        "我的经验", "经验总结", "投递经验", "面试经验", "笔试经验", "失败原因", "被拒原因",
    ))
    explicit_reference_material = (
        any(x in semantic_core for x in (
            "参考资料", "学习资料", "结合资料", "结合我提供",
            "资料说明", "教程", "博客", "普通学习资料",
        ))
        # “知识库里我审批过的项目”中，知识库是个人事实的存放位置，
        # 不是另一个通用资料答案源。
        or ("知识库" in semantic_core and not project_inventory)
    )
    if application_project_fit and not explicit_reference_material:
        # “按 JD 说明推荐依据”中的“说明”是回答形式，不是通用知识来源请求。
        reference_hit = False
    jd_needs_explanation = (
        "application_jd" in domains
        and any(x in semantic_core for x in ("技术概念", "解释技术", "概念是什么", "原理是什么"))
    )
    resume_writing = (
        "resume" in domains
        and any(x in semantic_q for x in ("怎么写", "怎样写", "如何写", "写进简历"))
        and not any(x in semantic_q for x in ("我的项目", "项目材料", "项目内容", "我的项目记录"))
    )
    answer_objects: list[str] = []
    if application_project_fit:
        answer_objects.append("project_fit")
        if any(x in semantic_core for x in ("缺口", "缺少", "欠缺", "短板")):
            answer_objects.append("profile_gap")
    if project_inventory:
        answer_objects.append("project")
    application_as_filter = bool(
        explicit_application_fact
        and experience_central
        and not any(x in semantic_core for x in ("哪些", "哪几", "状态", "进度", "列出"))
    )
    project_as_filter = bool(
        explicit_application_fact
        and any(x in semantic_core for x in ("项目相关岗位", "项目相关的岗位", "开源项目相关"))
    )
    if (explicit_application_fact and not project_inventory and not application_as_filter
            and not application_project_fit):
        answer_objects.append("application")
    if experience_central:
        answer_objects.append("application_experience")
    if (personal and "learning_execution" in domains) or (personal and any(
            x in semantic_core for x in ("历史证据", "能力证据", "实践证据"))):
        answer_objects.append("reflection")
    if personal and "profile" in domains and any(x in semantic_q for x in (
            "画像", "能力", "掌握", "会什么", "缺口", "缺少", "欠缺",
            "求职方向", "目标方向", "技能摘要")):
        answer_objects.append("profile")
    if personal and "plan" in domains and "计划" in semantic_q:
        answer_objects.append("plan")
    if personal and "application_jd" in domains and any(x in semantic_q for x in (
            "jd", "职位描述", "岗位要求", "任职要求", "岗位原文")):
        answer_objects.append("application_jd")
    if personal and "resume" in domains:
        answer_objects.append("resume_rule")
    if (personal and "project" in domains and not project_inventory
            and not resume_writing and not project_as_filter and not application_project_fit):
        answer_objects.append("project_detail")
    if personal and "application" in domains and not answer_objects:
        answer_objects.append("application")
    if "paper" in domains:
        answer_objects.append("paper_knowledge")
    if ("reference" in domains and (explicit_reference_material or jd_needs_explanation
                                    or "paper" not in domains and "application_jd" not in domains)) \
            or (not domains and not personal):
        answer_objects.append("reference_knowledge")
    if (not personal and "application" in domains
            and not any(x in domains for x in ("application_jd", "paper"))):
        answer_objects.append("reference_knowledge")

    # 复合问题保留多个答案中心；用途词不因此升级为答案来源。
    if experience_central and "application_experience" not in answer_objects:
        answer_objects.append("application_experience")
    if personal and "profile" in domains and any(x in tasks for x in ("evaluate", "recommend")):
        answer_objects.append("profile")
    if ("reference" in domains and (explicit_reference_material or jd_needs_explanation)
            and any(x in tasks for x in ("explain", "recommend", "compare"))):
        answer_objects.append("reference_knowledge")
    if (personal and "application" in domains and "application" in answer_objects
            and any(x in semantic_core for x in _ADVICE_HINTS + _ROLE_KNOWLEDGE_HINTS)):
        answer_objects.append("reference_knowledge")
    if "paper" in domains:
        answer_objects.append("paper_knowledge")
    answer_objects = list(dict.fromkeys(answer_objects))
    answer_object = answer_objects[0] if answer_objects else ""

    operation_map = {
        "list": "list", "compare": "compare", "summarize": "summarize",
        "reflect": "summarize", "recommend": "advise", "explain": "search",
        "evaluate": "lookup", "search": "search",
    }
    operations = list(dict.fromkeys(operation_map.get(task, "search") for task in tasks))
    if ("advise" in operations and "application" in domains and "下一步" in semantic_core
            and not any(x in semantic_core for x in ("建议", "怎么办", "怎么做", "如何"))):
        operations = ["lookup" if operation == "advise" else operation for operation in operations]
    usage_contexts: list[str] = []
    if application_project_fit:
        usage_contexts.append("application")
    if project_inventory and any(x in q for x in ("投递", "求职", "应聘")):
        usage_contexts.append("application")
    if project_inventory and "简历" in q:
        usage_contexts.append("resume")

    filters: dict[str, Any] = {}
    if application_project_fit:
        filters["relation_intent"] = "application_project_fit"
        # 只把问题文本作为本地实体匹配输入；规划模型不会接触投递文件。
        filters["application_entity_query"] = task_q[:300]
    if "开源" in q:
        filters["project_type"] = "open_source"
    if any(x in q for x in ("已审批", "审批过", "已批准", "确认过", "已确认")):
        filters["review_status"] = "approved"
    statuses = [s for s in ("已评估", "准备投递", "不投递", "已投递", "等待反馈",
                            "面试中", "已 offer", "已拒绝", "主动放弃") if s in q]
    if statuses:
        filters["statuses"] = statuses

    if personal_forbidden or (personal_history_negated and "reference" in domains
                              and not any(x in domains for x in (
                                  "application", "project", "resume", "profile", "plan",
                              ))):
        personal_scope = "none"
    elif project_inventory:
        personal_scope = "artifact"
    elif personal and "application" in domains:
        personal_scope = "application"
    elif personal and time_scope["kind"] == "history":
        personal_scope = "history"
    elif personal and any(x in domains for x in ("project", "resume")):
        personal_scope = "artifact"
    elif personal:
        personal_scope = ("history" if history or historical_action or implicit_personal_artifact
                          else "current")
    else:
        personal_scope = "none"

    signals = []
    if explicit_personal:
        signals.append("personal_pronoun")
    if implicit_personal_history:
        signals.append("implicit_personal_history")
    if implicit_personal_current:
        signals.append("implicit_personal_current")
    if implicit_personal_artifact:
        signals.append("implicit_personal_artifact")
    if implicit_application_current:
        signals.append("implicit_application_current")
    if implicit_application_project_fit:
        signals.append("implicit_application_project_fit")
    if history:
        signals.append("history_expression")
    if historical_action:
        signals.append("historical_action")
    if personal_forbidden:
        signals.append("personal_source_negated")
    if application_source_negated:
        signals.append("application_source_negated")
    if learning_source_negated:
        signals.append("learning_source_negated")
    if personal_history_negated:
        signals.append("history_fact_negated")
    if project_inventory:
        signals.append("project_inventory")
    if application_project_fit:
        signals.append("application_project_fit")
    if unresolved:
        signals.append("anaphora")
    deterministic_signals: list[str] = []
    if dates or any(x in q for x in ("今天", "昨天", "本周", "上周")):
        deterministic_signals.append("explicit_time")
    if statuses:
        deterministic_signals.append("explicit_status")
    if project_inventory:
        deterministic_signals.append("explicit_catalog_operation")
    if explicit_application_fact:
        deterministic_signals.append("explicit_application_fact")
    if (personal_scope == "application" and "application" in answer_objects
            and set(operations) <= {"list", "lookup"}):
        deterministic_signals.append("structured_application_fact")
    if application_project_fit:
        deterministic_signals.append("explicit_relation_workflow")
    if re.search(r"\b(?:app|jdv|jd|match)_[a-z0-9_-]+\b", q):
        deterministic_signals.append("explicit_id")

    completeness = sum(bool(x) for x in (answer_objects, operations, domains, personal_scope)) / 4
    confidence = 0.55 + 0.35 * completeness
    if deterministic_signals:
        confidence += 0.06
    if len(answer_objects) > 1:
        confidence -= 0.08
    confidence = round(max(0.0, min(0.99, confidence)), 3)
    if unresolved and not context:
        confidence = min(confidence, 0.45)
    if personal and not domains:
        confidence = min(confidence, 0.72)
    service_mode = "explain"
    if personal_scope != "none":
        service_mode = "advise" if (
            "advise" in operations
            or application_project_fit
            or (len(answer_objects) > 1 and any(
                obj in answer_objects for obj in ("reference_knowledge", "paper_knowledge")))
        ) else "recall"

    return RequestFrame(
        service_mode=service_mode,
        personal_scope=personal_scope,
        domains=list(dict.fromkeys(domains)),
        time_scope=time_scope,
        tasks=list(dict.fromkeys(tasks)),
        answer_object=answer_object,
        answer_objects=answer_objects,
        operations=operations,
        source_roles=[],
        usage_contexts=usage_contexts,
        filters=filters,
        confidence=confidence,
        unresolved_references=unresolved,
        signals=signals,
        deterministic_signals=deterministic_signals,
    )


def _route(source: str, operation: str, question: str, *,
           required: bool = False, filters: dict | None = None,
           depends_on: tuple[str, ...] = (),
           bindings: dict[str, str] | None = None,
           source_role: str = "", selection_reason: str = "",
           locked: bool = False) -> RouteSpec:
    definition = get_route_definition(source, operation)
    role = source_role or (definition.source_role if definition else "answer_source")
    route_id = "route_" + hashlib.sha256(
        f"{source}.{operation}|{question[:300]}".encode("utf-8")
    ).hexdigest()[:12]
    return RouteSpec(source=source, operation=operation, subquery=question,
                     filters=filters or {}, required=required,
                     depends_on=depends_on, bindings=bindings or {},
                     route_id=route_id, source_role=role,
                     selection_reason=selection_reason, locked=locked)


def _dedupe(routes: list[RouteSpec]) -> list[RouteSpec]:
    out: list[RouteSpec] = []
    positions: dict[tuple[str, str], int] = {}
    for route in routes:
        if route.source not in ROUTE_OPERATIONS:
            continue
        if route.operation not in ROUTE_OPERATIONS[route.source]:
            continue
        key = (route.source, route.operation)
        if key in positions:
            i = positions[key]
            old = out[i]
            out[i] = RouteSpec(
                source=old.source,
                operation=old.operation,
                subquery=old.subquery or route.subquery,
                filters={**old.filters, **route.filters},
                required=old.required or route.required,
                depends_on=tuple(dict.fromkeys(old.depends_on + route.depends_on)),
                bindings={**old.bindings, **route.bindings},
                route_id=old.route_id or route.route_id,
                source_role=(old.source_role if old.source_role != "supporting_context"
                             else route.source_role),
                selection_reason="；".join(filter(None, (
                    old.selection_reason, route.selection_reason))),
                locked=old.locked or route.locked,
            )
            continue
        positions[key] = len(out)
        out.append(route)
    return out


def _matching_subquery(question: str, hints: tuple[str, ...]) -> str:
    """选出承载某个子诉求的分句，避免所有路线都拿整句做向量查询。"""
    clauses = [part.strip(" ，,。；;？！?\t\n") for part in re.split(
        r"(?<=[。；;？！?])|(?:同时|并且|另外|然后|再结合)", question or ""
    ) if part.strip(" ，,。；;？！?\t\n")]
    matched = [clause for clause in clauses if _contains_any(_normalise(clause), hints)]
    return "；".join(matched) if matched else (question or "").strip()


def _entities(q: str) -> dict[str, Any]:
    statuses = [s for s in ("已评估", "准备投递", "不投递", "已投递", "等待反馈",
                            "面试中", "已 offer", "已拒绝", "主动放弃") if s.lower() in q]
    time_range = ""
    for hint in ("今天", "昨天", "本周", "上周", "最近", "目前", "当前"):
        if hint in q:
            time_range = hint
            break
    return {"companies": [], "positions": [], "statuses": statuses,
            "time_range": time_range, "technical_topic": ""}


def _answer_sections(routes: list[RouteSpec]) -> list[str]:
    sources = {r.source for r in routes}
    operations = {(r.source, r.operation) for r in routes}
    intents = {str(r.filters.get("intent", "")) for r in routes}
    sections: list[str] = []
    if "product_help" in sources:
        sections.append("OfferClaw 功能入口与确认边界")
    if "system_diagnostics" in sources:
        sections.append("数据完整性与检索运行诊断")
    if "application_state" in sources:
        sections.append("实时投递事实")
    if "application_jd" in sources:
        sections.append("绑定 JD 要求与版本依据")
    if ("application_jd", "get_match_snapshot") in operations:
        sections.append("已确认匹配结论与能力缺口")
    if "application_advice" in intents:
        sections.append("按企业与岗位说明注意事项")
    if "role_requirements" in intents:
        sections.append("对应岗位要求与准备重点")
    if "application_experience" in sources:
        sections.append("个人经验依据")
    if "reflection_memory" in sources:
        sections.append("每日执行与长期复盘依据")
    if ("profile_plan", "get_gaps") in operations:
        sections.append("记录中的能力缺口")
    if any(source == "profile_plan" and operation in {"get_profile", "get_plan", "get_recent_log"}
           for source, operation in operations):
        sections.append("当前画像、计划与近期执行")
    if "profile_plan" in sources and sources & {"reference_kb", "paper_kb"}:
        sections.append("基于记录的能力推断")
    if sources & {"project_memory", "resume_rules"}:
        sections.append("个人材料依据")
    if ("project_memory", "rank_for_application") in operations:
        sections.append("项目适配比较与推荐")
    if sources & {"reference_kb", "paper_kb"}:
        sections.append("参考资料与概念补充")
    if "profile_plan" in sources or "application_experience" in sources or "reflection_memory" in sources:
        sections.append("下一步建议")
    return sections or ["回答"]


def _is_history_overview_question(q: str, frame: RequestFrame) -> bool:
    """判断是否需要读取完整历史，而不是只检索几个相似片段。"""
    if frame.time_scope.get("kind") != "history" and frame.personal_scope != "history":
        return False
    if "explain" in frame.tasks:
        return False
    if any(h in q for h in ("什么时候", "何时", "哪一天", "哪里卡", "为什么", "哪次")):
        return False
    broad_objects = (
        "所做的工作", "做过的工作", "学习工作", "学习经历", "学习过程", "过往经历",
        "以前做过什么", "之前做了什么", "完成过什么", "练习过什么", "一路以来",
        "执行记录", "历史记录", "之前的进展", "过去的进展",
    )
    return (_contains_any(q, _SUMMARY_HINTS)
            and any(x in q for x in ("工作", "学习", "经历", "过程"))) \
        or any(x in q for x in broad_objects)


def _route_answer_objects(route: RouteSpec) -> set[str]:
    definition = get_route_definition(route.source, route.operation)
    return set(definition.answer_objects if definition else ())


def _decorate_rule_routes(question: str, frame: RequestFrame,
                          routes: list[RouteSpec]) -> list[RouteSpec]:
    """区分可被语义裁决替换的软路由与必须保留的确定性/显式路由。"""
    q = _normalise(question)
    primary = set(frame.answer_objects or ([frame.answer_object] if frame.answer_object else []))
    explicit_reference = any(x in q for x in (
        "参考资料", "学习资料", "知识库", "结合资料", "结合我提供", "教程", "博客",
    ))
    explicit_experience = any(x in q for x in (
        "我的经验", "经验总结", "投递经验", "面试经验", "笔试经验", "失败原因", "被拒原因",
    ))
    out: list[RouteSpec] = []
    source_roles: list[dict[str, str]] = []
    for route in routes:
        objects = _route_answer_objects(route)
        role = route.source_role
        if route.source in {"reference_kb", "paper_kb"} and frame.personal_scope != "none":
            role = "supporting_context"
        if route.source == "project_memory" and "application" in primary:
            role = "filter_source"
        if route.operation == "get_profile_evidence":
            role = "validation_source"
        # 显式的筛选/验证来源不能因为与答案对象同域，就被重新提升为回答主体。
        # 典型场景：上一轮先圈定“准备投递的公司”，本轮再问“这些公司的下一步”。
        if objects & primary and role not in {"filter_source", "validation_source"}:
            role = "answer_source" if role != "validation_source" else role

        locked = route.locked
        if route.source == "application_state" and (
                "explicit_application_fact" in frame.deterministic_signals
                or "explicit_status" in frame.deterministic_signals
                or "structured_application_fact" in frame.deterministic_signals
                or "explicit_id" in frame.deterministic_signals):
            locked = True
        if route.source == "application_jd" and "explicit_id" in frame.deterministic_signals:
            locked = True
        if route.source == "project_memory" and route.operation in {"list_catalog", "list_approved"}:
            locked = "explicit_catalog_operation" in frame.deterministic_signals
        if route.source == "reflection_memory" and route.operation == "get_by_date":
            locked = "explicit_time" in frame.deterministic_signals
        if route.source == "paper_kb" and _contains_any(q, _PAPER_HINTS):
            locked = True
        if route.source == "general_fallback" and route.filters.get("internal_sources_forbidden"):
            locked = True

        required = route.required
        if route.source == "application_experience" and explicit_experience:
            required = True
        if route.source == "reference_kb" and explicit_reference:
            required = True
        if route.source == "paper_kb" and _contains_any(q, _PAPER_HINTS):
            required = True
        reason = route.selection_reason or (
            "确定性事实/显式来源" if locked or required else "规则生成的软候选"
        )
        decorated = RouteSpec(
            route.source, route.operation, route.subquery, route.filters, required,
            route.depends_on, route.bindings, route.route_id,
            role, reason, locked,
        )
        out.append(decorated)
        source_roles.append({
            "route": f"{route.source}.{route.operation}", "role": role,
        })
    frame.source_roles = source_roles
    return _dedupe(out)


def rule_plan_query(question: str, *, context: list[str] | None = None) -> QueryPlan:
    """纯本地、多标签规则规划。此函数不得调用 LLM 或读取个人数据。"""
    q = _normalise(question)
    frame = build_intent_frame(question, context=context)
    context_text = "；".join(
        str(x).strip() for x in (context or [])[-3:] if str(x).strip()
    )
    semantic_q = (_normalise(f"{context_text}；{question}")
                  if frame.unresolved_references and context_text else q)
    request_q = _strip_request_scaffolding(q)
    semantic_core = (_normalise(f"{context_text}；{request_q}")
                     if frame.unresolved_references and context_text else request_q)
    routes: list[RouteSpec] = []
    reasons: list[str] = []

    if "product_help" in frame.domains:
        route = _route(
            "product_help", "guide", question,
            filters=dict(frame.filters), required=True,
            source_role="answer_source",
            selection_reason="系统操作请求必须返回确定性入口与只读边界",
            locked=True,
        )
        frame.source_roles = [{"route": "product_help.guide", "role": "answer_source"}]
        return QueryPlan(
            planner_mode="rule",
            routes=[route],
            intent_frame=frame,
            entities=_entities(q),
            answer_sections=_answer_sections([route]),
            reason="识别到 OfferClaw 功能入口或数据管理操作请求",
            resolver_mode="hard_rule",
            confidence_factors={
                "frame_completeness": frame.confidence,
                "hard_route_count": 1,
                "route_count": 1,
            },
        )

    if "system_diagnostics" in frame.domains:
        route = _route(
            "system_diagnostics", "inspect", question,
            filters=dict(frame.filters), required=True,
            source_role="answer_source",
            selection_reason="明确的失败诊断只读取本地运行状态与数据完整性",
            locked=True,
        )
        frame.source_roles = [{"route": "system_diagnostics.inspect", "role": "answer_source"}]
        return QueryPlan(
            planner_mode="rule", routes=[route], intent_frame=frame,
            entities=_entities(q), answer_sections=_answer_sections([route]),
            reason="识别到 OfferClaw 检索或数据前置条件诊断",
            resolver_mode="hard_rule",
            confidence_factors={
                "frame_completeness": frame.confidence,
                "hard_route_count": 1, "route_count": 1,
            },
        )

    personal_marker = frame.personal_scope != "none"
    personal_forbidden = "personal_source_negated" in frame.signals
    history_fact_negated = "history_fact_negated" in frame.signals
    reference_forbidden = bool(re.search(
        r"(?:不要|不用|无需|不需要|别)[^，,。；;？！?]{0,14}"
        r"(?:通用(?:知识|资料)?|知识库|参考资料|学习资料|一般资料)", q
    ))
    strong_app_hit = _contains_any(semantic_q, _APPLICATION_HINTS)
    # 投递/JD 只有在语义框架已确认为个人事实时才读取 applications.md。
    # “企业官网投递通常有哪些注意事项”虽然含有相同领域词，但答案
    # 中心是通用方法，不应读取用户投递状态。
    app_hit = personal_marker and (strong_app_hit or "投递" in semantic_q)
    exp_hit = _contains_any(semantic_q, _EXPERIENCE_HINTS)
    reflection_hit = _contains_any(semantic_q, _REFLECTION_HINTS)
    if (personal_marker and not history_fact_negated
            and "learning_execution" in frame.domains
            and (not any(x in frame.domains for x in ("project", "resume"))
                 or any(x in q for x in ("学习", "执行", "复盘", "记录", "练习", "卡住")))):
        reflection_hit = True
    if personal_marker and (
        any(h in q for h in ("掌握", "能力证据", "实践证据"))
        or (any(h in q for h in ("今天", "昨天", "本周", "上周", "最近"))
            and any(h in q for h in ("做了什么", "学了什么", "完成了什么", "执行了什么")))
        or (any(h in q for h in ("以前", "曾经", "什么时候", "何时", "过去"))
            and any(h in q for h in ("学", "做", "完成", "卡", "阻碍", "复盘", "建议", "练习", "实践")))
    ):
        reflection_hit = True
    profile_hit = (
        personal_marker and (
            "profile" in frame.domains or "plan" in frame.domains
            or (_contains_any(semantic_q, _PROFILE_HINTS)
                and not any(x in frame.domains for x in (
                    "application", "application_jd", "project", "resume",
                )))
        )
    )
    project_hit = _contains_any(semantic_core, _PROJECT_HINTS) or (
        personal_marker and "project" in frame.domains
    )
    resume_hit = _contains_any(semantic_core, _RESUME_HINTS) or (
        personal_marker and "resume" in frame.domains
    )
    paper_hit = _contains_any(semantic_core, _PAPER_HINTS)
    reference_hit = _contains_any(semantic_core, _REFERENCE_HINTS)
    if "reference" in frame.domains:
        reference_hit = True
    # “简历格式/项目经验”在没有个人指代时属于通用知识，而不是用户私人材料。
    if project_hit and not personal_marker:
        project_hit = False
        reference_hit = True
    if resume_hit and not personal_marker:
        resume_hit = False
        reference_hit = True
    if (resume_hit and project_hit
            and any(x in q for x in ("模板没有项目段", "简历模板没有项目段", "项目段时"))):
        project_hit = False
    if (resume_hit and project_hit
            and any(x in request_q for x in ("怎么写", "怎样写", "如何写", "写进简历"))
            and not any(x in request_q for x in ("我的项目", "项目材料", "项目内容", "我的项目记录"))):
        project_hit = False
    advice_hit = _contains_any(request_q, _ADVICE_HINTS)
    role_knowledge_hit = _contains_any(request_q, _ROLE_KNOWLEDGE_HINTS)
    application_anaphora = _contains_any(request_q, _APPLICATION_ANAPHORA)
    application_jd_hit = (_contains_any(semantic_core, _APPLICATION_JD_HINTS)
                          or (personal_marker and "application_jd" in frame.domains
                              and "jd" in semantic_core))
    application_project_fit = "application_project_fit" in frame.signals
    if application_project_fit:
        # 关系工作流的来源集合由契约固定，不能因为用户没有显式说“JD”而漏掉
        # 真正的比较标准。
        app_hit = True
        application_jd_hit = True
        project_hit = True
        # 本题缺口必须来自该 application_id 的匹配快照，不能混入全部计划目标
        # 的聚合缺口视图。
        profile_hit = False
    if application_jd_hit and personal_marker:
        app_hit = True

    explicit_reference_material = any(x in semantic_core for x in (
        "参考资料", "学习资料", "知识库", "结合资料", "结合我提供", "资料说明", "教程", "博客", "普通学习资料",
    ))
    if (paper_hit and not explicit_reference_material
            and "技术应用" not in semantic_core):
        reference_hit = False
    if (application_jd_hit and not explicit_reference_material
            and not any(x in semantic_core for x in ("技术概念", "解释技术", "概念是什么", "原理是什么"))):
        reference_hit = False
    if (application_jd_hit and frame.answer_object == "application_jd"
            and not any(x in request_q for x in (
                "投了哪", "投过哪", "哪几家", "投递状态", "投递记录", "准备投哪些",
            ))):
        app_hit = False
    if ("learning_execution" in frame.domains and "application" not in frame.domains
            and not any(x in semantic_core for x in (
                "经验总结", "我的经验", "投递经验", "面试经验", "笔试经验", "亲历", "面经",
            ))):
        exp_hit = False

    project_inventory = "project_inventory" in frame.signals
    explicit_application_fact = any(x in q for x in (
        "公司", "企业", "岗位", "状态", "进度", "投递记录", "哪几家",
        "投了哪", "投过哪", "准备投哪些公司", "准备投哪些岗位",
    ))
    if project_inventory and not explicit_application_fact:
        app_hit = False
    if (project_inventory and "知识库" in semantic_core
            and not any(x in semantic_core for x in (
                "参考资料", "学习资料", "结合资料", "教程", "博客", "论文", "文献",
            ))):
        # 此时“知识库”是个人项目的位置限定，不是通用资料支持源。
        reference_hit = False

    if personal_forbidden:
        app_hit = exp_hit = reflection_hit = profile_hit = project_hit = resume_hit = False
    if reference_forbidden:
        reference_hit = False
    if "application_source_negated" in frame.signals:
        app_hit = False
        exp_hit = False
    if "learning_source_negated" in frame.signals:
        reflection_hit = False
        if "application" not in frame.domains:
            exp_hit = False

    # 没有个人指代时，“面试经验/学习复盘/简历格式”是在问通用方法，不能读取私人域。
    if not personal_marker:
        if exp_hit or reflection_hit or profile_hit:
            reference_hit = True
        exp_hit = False
        reflection_hit = False
        profile_hit = False

    technical_tokens = [
        token for token in re.findall(r"[a-z][a-z0-9_-]{1,}", q)
        if token not in {"offerclaw", "localflow"}
    ]
    explanatory_personal_mix = (
        personal_marker
        and any(x in frame.domains for x in ("learning_execution", "project", "resume"))
        and (any(x in request_q for x in ("解释", "普及", "原理", "概念", "作用"))
             or (any(x in request_q for x in ("介绍", "说明"))
                 and (bool(technical_tokens) or any(x in request_q for x in (
                     "混合检索", "向量检索", "重排", "知识图谱", "大模型", "智能体",
                 )))))
    )
    if explanatory_personal_mix and not reference_forbidden:
        reference_hit = True

    application_reflection_context = any(h in q for h in (
        "投递", "企业", "公司", "岗位", "面试", "笔试", "被拒", "拒绝", "失败企业", "面经",
    ))
    specific_experience = any(h in q for h in (
        "经验总结", "我的经验", "投递经验", "失败原因", "被拒原因", "亲历", "面经",
    ))
    if specific_experience and not any(x in q for x in (
            "学习复盘", "学习经历", "执行记录", "学习记录")):
        reflection_hit = False
    # “复盘”本身不是投递经验专属词；没有投递上下文时归入学习/执行长期记忆。
    if (personal_marker and "learning_source_negated" not in frame.signals
            and "复盘" in q and not application_reflection_context and not specific_experience):
        exp_hit = False
        reflection_hit = True
    if (reflection_hit and not application_project_fit
            and any(h in q for h in ("掌握", "能力证据", "实践证据"))):
        profile_hit = True
    if (reflection_hit and "profile" not in frame.domains and "plan" not in frame.domains
            and not any(h in q for h in ("缺口", "能力", "画像", "掌握"))):
        profile_hit = False
    if (application_project_fit and not any(h in q for h in (
            "学习记录", "执行记录", "历史记录", "复盘记录", "学习复盘",
            "实践记录", "以前学", "曾经学", "过去学",
        ))):
        # “把项目作为能力证据”描述的是项目在投递中的用途，不等价于要求读取
        # 学习复盘或正式画像；比较依据仍由活动 JD 与匹配快照提供。
        reflection_hit = False
        profile_hit = False

    # “我的投递经验/投递复盘”主要查经验语义，不因词面出现“投递”顺带拉全量状态。
    if exp_hit and not strong_app_hit and not any(h in q for h in ("状态", "进度", "下一步", "哪几")):
        app_hit = False

    # 明确否定的来源不能因为关键词出现而被读取。
    if re.search(r"(不要|不用|无需|不需要|没问).{0,12}(投递|个人记录|我的状态)", q):
        app_hit = False
    if re.search(r"(不要|不用|无需|不需要|不是).{0,12}(我的项目|项目材料|项目记忆)", q):
        project_hit = False
    if re.search(r"(不要|不用|无需|不需要).{0,12}(我的简历|简历格式|简历模板|简历规则)", q):
        resume_hit = False
    experience_forbidden = bool(re.search(
        r"(不要|不用|无需|不需要).{0,12}(我的经验|经验总结|面经|亲历|个人经验)", q
    ))
    if experience_forbidden:
        exp_hit = False
    if re.search(r"(不要|不用|无需|不需要).{0,8}(论文|文献)", q):
        paper_hit = False

    advice_negated = bool(re.search(
        r"(不要|不用|无需|不需要|别).{0,10}(注意事项|建议|准备方法|跟进)", q
    ))
    if advice_negated:
        advice_hit = False

    # “失败企业/被拒原因”既需要确定性企业集合，也需要对应亲历经验。
    if exp_hit and _contains_any(q, _FAILED_HINTS):
        app_hit = True

    # 复合问题的后半句常用“该企业/这些岗位”引用前半句的投递事实。此时不是把
    # 两组关键词简单并列，而是建立 application_state -> 语义检索的实体绑定。
    application_advice = app_hit and advice_hit
    application_role_knowledge = app_hit and role_knowledge_hit
    application_followup = application_advice or application_role_knowledge
    if application_advice and not experience_forbidden:
        exp_hit = True
    if application_followup:
        reference_hit = True
    followup_hints = _ADVICE_HINTS + _ROLE_KNOWLEDGE_HINTS + _APPLICATION_ANAPHORA
    followup_subquery = _matching_subquery(question, followup_hints)
    state_binding = ("application_state",) if application_followup else ()
    binding_map = ({"companies": "application_state.companies",
                    "positions": "application_state.positions"}
                   if state_binding else {})

    if app_hit:
        operation_specs: list[tuple[str, str]] = []
        # 指代问题需要保留上一轮圈定对象的事实操作，但它只负责筛选本轮答案，
        # 不能和本轮真正请求的操作混成两个回答中心。
        for operation, hints in (
            ("list_ever_applied", _APPLIED_HINTS),
            ("list_pending_submission", _PENDING_HINTS),
            ("list_failed", _FAILED_HINTS),
        ):
            if _contains_any(semantic_q, hints):
                role = (
                    "filter_source"
                    if frame.unresolved_references and context_text
                    and not _contains_any(q, hints)
                    else "answer_source"
                )
                operation_specs.append((operation, role))
        if application_project_fit:
            # 项目适配按实体定位投递，不应因“准备投/投过”等叙述词附带状态过滤；
            # 状态只是目标记录的属性，关系链始终从当前事实视图开始。
            operation_specs = [("list_current", "filter_source")]
        else:
            if _contains_any(q, _NEXT_HINTS):
                operation_specs.append(("get_next_actions", "answer_source"))
            if not operation_specs:
                operation_specs.append(("list_current", "answer_source"))
        application_filters = ({
            "intent": "application_project_fit",
            "entity_query": str(frame.filters.get("application_entity_query") or question),
            "strict_entity_filter": True,
        } if application_project_fit else {})
        routes.extend(_route(
            "application_state", operation, question, required=True,
            filters=application_filters,
            source_role=("filter_source" if application_project_fit else role),
            selection_reason=(
                "为项目适配关系图精确定位目标投递"
                if application_project_fit else "由上一轮确定当前指代对象"
                if role == "filter_source" else "当前问题明确要求的投递事实"
            ),
            locked=application_project_fit,
        ) for operation, role in operation_specs)
        reasons.append("识别到个人投递事实或动作")

    if application_jd_hit or application_followup:
        operation = "compare_versions" if any(x in q for x in (
            "版本变化", "版本对比", "更新了什么", "新旧版本", "比较版本", "对比版本",
        )) else "search_bound_jd"
        routes.append(_route(
            "application_jd", operation,
            followup_subquery if application_followup else question,
            required=bool(application_role_knowledge or application_project_fit),
            filters={"intent": ("application_project_fit" if application_project_fit
                                else "bound_jd_evidence")},
            depends_on=("application_state",) if app_hit else (),
            bindings=(
                {"application_ids": "application_state.application_ids",
                 "jd_version_ids": "application_state.jd_version_ids"}
                if application_project_fit else binding_map if app_hit else {}
            ),
            source_role=("validation_source" if application_project_fit else ""),
            selection_reason=("活动 JD 是项目适配的验证标准"
                              if application_project_fit else ""),
            locked=application_project_fit,
        ))
        if application_project_fit:
            routes.append(_route(
                "application_jd", "get_match_snapshot", question,
                required=True,
                filters={"intent": "application_project_fit"},
                # 同一 source 内先读取 JD 再读取 match，由执行器顺序保证；DAG 只
                # 表示跨 source 依赖，避免制造 application_jd 自环。
                depends_on=("application_state",),
                bindings={
                    "application_ids": "application_state.application_ids",
                    "jd_version_ids": "application_state.jd_version_ids",
                    "match_ids": "application_state.match_ids",
                },
                source_role="validation_source",
                selection_reason="读取该投递已确认的匹配结论与缺口",
                locked=True,
            ))
        reasons.append("识别到投递绑定 JD 的要求或溯源意图")

    if exp_hit:
        filters = {"scope": "failed"} if _contains_any(q, _FAILED_HINTS) else {}
        if application_advice:
            filters["intent"] = "application_advice"
            filters["entity_binding"] = "selected_applications"
        exp_depends = (("application_state",)
                       if app_hit and (application_followup or _contains_any(q, _FAILED_HINTS))
                       else ())
        exp_bindings = ({"companies": "application_state.companies",
                         "positions": "application_state.positions"}
                        if exp_depends else {})
        routes.append(_route(
            "application_experience", "search",
            followup_subquery if application_followup else question,
            required=False, filters=filters, depends_on=exp_depends,
            bindings=exp_bindings,
        ))
        reasons.append("识别到个人投递经验/复盘")

    if profile_hit:
        if ("profile" in frame.answer_objects
                and any(h in q for h in ("画像", "掌握", "会什么", "正式能力", "当前能力"))):
            routes.append(_route("profile_plan", "get_profile", question))
        if any(h in q for h in ("缺口", "缺少", "欠缺", "学习建议", "下一步学习")):
            routes.append(_route("profile_plan", "get_gaps", question))
        if (any(h in q for h in ("计划", "下一步学习", "学习建议"))
                or ("今天" in q and any(h in q for h in ("该做", "要做", "任务", "安排")))):
            routes.append(_route("profile_plan", "get_plan", question))
        if any(h in q for h in ("最近", "留痕", "日志", "昨天", "做了", "学了")):
            routes.append(_route("profile_plan", "get_recent_log", question))
        if not any(r.source == "profile_plan" for r in routes):
            routes.append(_route("profile_plan", "get_profile", question))
        reasons.append("识别到个人画像/计划/缺口/留痕")

    if reflection_hit:
        if any(h in q for h in ("掌握", "能力证据", "实践证据")):
            operation = "get_profile_evidence"
        elif any(h in q for h in ("今天", "昨天", "本周", "上周")) or re.search(
                r"\d{4}-\d{2}-\d{2}", q):
            operation = "get_by_date"
        elif any(h in q for h in ("最近", "近期", "这段时间")):
            operation = "get_recent"
        elif _is_history_overview_question(q, frame):
            operation = "get_history_overview"
        else:
            operation = "search_topic"
        routes.append(_route("reflection_memory", operation, question))
        reasons.append("识别到每日执行或长期学习复盘")

    if project_hit:
        operation = ("rank_for_application" if application_project_fit
                     else "list_approved" if project_inventory and any(
            x in q for x in ("已审批", "审批过", "已批准", "确认过", "已确认")
        ) else "list_catalog" if project_inventory else "search")
        filters: dict[str, Any] = {}
        if application_project_fit:
            filters["intent"] = "application_project_fit"
        if project_inventory:
            filters = {
                "open_source_only": "开源" in q,
                "usage_context": ",".join(frame.usage_contexts),
            }
            if any(x in q for x in ("简历上用过", "简历中用过", "写进过简历", "曾用于简历")):
                filters["resume_confirmed_only"] = True
        routes.append(_route(
            "project_memory", operation, question, filters=filters,
            required=application_project_fit,
            depends_on=(
                ("application_state", "application_jd") if application_project_fit
                else ("application_state",) if app_hit and application_anaphora else ()
            ),
            bindings=(
                {"application_ids": "application_state.application_ids",
                 "jd_version_ids": "application_jd.jd_version_ids",
                 "gap_evidence": "application_jd.match_gaps"}
                if application_project_fit else binding_map
                if app_hit and application_anaphora else {}
            ),
            source_role="answer_source",
            selection_reason=("依据活动 JD 与匹配缺口比较已确认个人项目"
                              if application_project_fit else ""),
            locked=application_project_fit,
        ))
        reasons.append("识别到个人项目清单" if project_inventory else "识别到个人项目材料")

    if resume_hit:
        routes.append(_route(
            "resume_rules", "search", question,
            depends_on=("application_state",) if app_hit and application_anaphora else (),
            bindings=binding_map if app_hit and application_anaphora else {},
        ))
        reasons.append("识别到个人简历规则")

    if paper_hit:
        routes.append(_route("paper_kb", "search", question))
        reasons.append("识别到显式论文/文献意图")

    # 状态问法中的“给我信息”不等于知识检索；只有真实的解释/资料词才加参考库。
    if application_project_fit and not explicit_reference_material:
        reference_hit = False
    if reference_hit:
        filters = {}
        if application_advice:
            filters = {"intent": "application_advice",
                       "entity_binding": "selected_applications"}
        elif application_role_knowledge:
            filters = {"intent": "role_requirements",
                       "entity_binding": "selected_applications"}
        routes.append(_route(
            "reference_kb", "search",
            followup_subquery if application_followup else question,
            filters=filters, depends_on=state_binding, bindings=binding_map,
        ))
        reasons.append("识别到概念解释或用户参考资料意图")

    # Personal Intent Guard：确认在问用户历史时，不能因为漏掉一个关键词就静默
    # 退化成通用知识回答。没有内部证据时应明确报告“没有记录”。
    if (not personal_forbidden and not history_fact_negated
            and frame.personal_scope == "history"
            and ("learning_execution" in frame.domains or not frame.domains)
            and not any(r.source in PERSONAL_ROUTES for r in routes)):
        operation = ("get_history_overview" if _is_history_overview_question(q, frame)
                     else "search_topic")
        routes.append(_route(
            "reflection_memory", operation, question, required=True,
            filters={"guard": "personal_history"},
        ))
        reasons.append("个人历史意图保护：禁止退化为通用回答")

    if not routes and reference_forbidden:
        routes.append(_route(
            "general_fallback", "answer", question,
            filters={"internal_sources_forbidden": True},
        ))
        reasons.append("用户明确排除内部资料，使用通用回答")
    elif not routes:
        # 顶部框的默认职责仍是知识库问答，未知问题先检索再决定是否通用兜底。
        routes.append(_route("reference_kb", "search", question))
        reasons.append("未命中个人状态规则，默认查询策展知识库")

    routes = _decorate_rule_routes(question, frame, _dedupe(routes))
    hard_count = sum(route.locked for route in routes)
    return QueryPlan(
        planner_mode="rule",
        routes=routes,
        intent_frame=frame,
        entities=_entities(q),
        answer_sections=_answer_sections(routes),
        reason="；".join(reasons),
        resolver_mode="hard_rule" if hard_count == len(routes) else "fallback",
        confidence_factors={
            "frame_completeness": frame.confidence,
            "hard_route_count": hard_count,
            "route_count": len(routes),
        },
    )


def _semantic_base_plan(question: str) -> QueryPlan:
    """Create a source-neutral shell for the structured semantic planner.

    This deliberately does not call ``rule_plan_query``.  The intelligent
    route model must decide the answer object and operation itself; only
    deterministic entity extraction is retained for later validation.
    """
    return QueryPlan(
        planner_mode="pending", routes=[], intent_frame=RequestFrame(
            service_mode="explain", personal_scope="none", confidence=0.0,
            signals=["await_structured_semantic_plan"],
        ),
        entities=_entities(_normalise(question)), answer_sections=[],
        reason="非硬问题等待结构化语义规划", resolver_mode="llm",
        confidence_factors={"hard_route_count": 0, "route_count": 0},
        planner_engine="structured_llm",
        planner_version=SEMANTIC_PLANNER_VERSION,
        schema_version=SEMANTIC_QUERY_SCHEMA_VERSION,
    )


def plan_structured_read(source: str, operation: str, *,
                         filters: dict[str, Any] | None = None,
                         subquery: str = "") -> QueryPlan:
    """Build the only zero-LLM v4 fast path from trusted structured input.

    Human free text must use :func:`plan_query`. Callers of this function have
    already selected a registry route through a typed system control.
    """
    definition = get_route_definition(source, operation)
    if definition is None or not definition.planner_visible:
        raise ValueError(f"unknown structured read route: {source}.{operation}")
    generic_operation = generic_operation_for_route(definition.key)
    service_mode = (
        "guide" if source == "product_help" else
        "diagnose" if source == "system_diagnostics" else
        "recall" if definition.personal else "explain"
    )
    contracts = [item.to_dict() for item in route_output_contracts(source, operation)]
    answer_objects = list(definition.answer_objects)
    route = _route(
        source, operation, subquery, filters=dict(filters or {}),
        required=True, locked=True,
        selection_reason="trusted structured read request",
    )
    frame = RequestFrame(
        service_mode=service_mode, interaction_kind="query",
        personal_scope="current" if definition.personal else "none",
        domains=[source], tasks=[generic_operation],
        answer_object=answer_objects[0] if answer_objects else "",
        answer_objects=answer_objects, output_contracts=contracts,
        operations=[generic_operation],
        source_roles=[{"route": definition.key, "role": "answer_source"}],
        signals=["structured_system_request"],
        deterministic_signals=["structured_system_request"],
        routing_assurance="deterministic",
        decision_reasons=["trusted_structured_read_bypassed_llm"],
    )
    plan = QueryPlan(
        planner_mode="structured", routes=[route], intent_frame=frame,
        answer_sections=_answer_sections([route]),
        reason="系统生成的结构化只读请求", resolver_mode="hard_rule",
        confidence_factors={"llm_calls": 0, "schema_validated": True},
        planner_engine="structured_read",
        planner_version=SEMANTIC_PLANNER_VERSION,
        schema_version=SEMANTIC_QUERY_SCHEMA_VERSION,
    )
    errors = _plan_validation_errors(plan)
    if errors:
        raise ValueError("invalid structured read: " + ",".join(errors))
    return plan


def _extract_json(text: str) -> dict | None:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
    candidate = fenced.group(1) if fenced else ""
    if not candidate:
        start, end = raw.find("{"), raw.rfind("}")
        candidate = raw[start:end + 1] if start >= 0 and end > start else ""
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _validated_llm_routes(data: dict, question: str, *,
                          allowed_keys: set[str] | None = None) -> list[RouteSpec]:
    routes: list[RouteSpec] = []
    for item in data.get("routes", []) if isinstance(data.get("routes"), list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        operation = str(item.get("operation", ""))
        if source not in ROUTE_OPERATIONS or operation not in ROUTE_OPERATIONS[source]:
            continue
        if allowed_keys is not None and f"{source}.{operation}" not in allowed_keys:
            continue
        filters = item.get("filters") if isinstance(item.get("filters"), dict) else {}
        if any(str(key).lower() in {"path", "file", "collection", "collection_name"}
               for key in filters):
            continue
        depends_on = tuple(
            dependency for dependency in (item.get("depends_on") or [])
            if isinstance(dependency, str) and dependency in ROUTE_OPERATIONS
            and dependency != source
        )
        bindings = item.get("bindings") if isinstance(item.get("bindings"), dict) else {}
        source_role = str(item.get("source_role") or "answer_source")
        if source_role not in SOURCE_ROLES:
            continue
        routes.append(_route(
            source, operation, str(item.get("subquery") or question)[:500],
            filters={str(k)[:50]: str(v)[:200] for k, v in filters.items()},
            required=bool(item.get("required", False)), depends_on=depends_on,
            bindings={str(k)[:50]: str(v)[:120] for k, v in bindings.items()},
            source_role=source_role, selection_reason="LLM 在候选白名单内裁决",
        ))
    return _dedupe(routes)


def _default_llm_call(planner_payload: str) -> str | None:
    # 延迟导入，避免 rag_gate -> rag_query_plan -> rag_gate 的模块初始化环。
    from rag_gate import _chat

    messages = [
        {"role": "system", "content": (
            "你是 OfferClaw 的受约束 Route Arbiter，不是检索 Agent。"
            "输入只含问题、语义框架和白名单候选，不含个人资料正文。"
            "只能选择候选中给出的 source/operation；不得编造工具、文件或 collection。"
            "如果候选指向不同答案中心且问题无法消歧，decision 必须为 clarify。"
            "只输出 JSON："
            '{"decision":"answer|clarify|fallback","routes":[{'
            '"source":"...","operation":"...","subquery":"...","filters":{},'
            '"source_role":"answer_source|supporting_context|filter_source|validation_source",'
            '"required":false,"depends_on":[],"bindings":{}}],'
            '"clarification":"...","reason":"..."}。'
        )},
        {"role": "user", "content": planner_payload[:12000]},
    ]
    route_model = os.environ.get("RAG_ROUTE_MODEL", "").strip() or None
    return _chat(messages, max_tokens=650, temperature=0.0, model=route_model)


def _call_with_timeout(call: Callable[[str], str | None], question: str,
                       timeout_seconds: float) -> str | None:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-planner")
    future = pool.submit(call, question)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeout:
        future.cancel()
        return None
    except Exception:
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _routing_profile() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "version": "routing-profile-v1",
        "semantic_accept_score": 0.70,
        "semantic_margin": 0.08,
        "conflict_margin": 0.10,
        "top_k_per_task": 3,
        "max_routes": 8,
    }
    path = os.path.join(os.path.dirname(__file__), "config", "rag_routing_profile.json")
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            defaults.update({key: loaded[key] for key in defaults if key in loaded})
    except Exception:
        pass
    env_map = {
        "semantic_accept_score": "RAG_ROUTE_ACCEPT_SCORE",
        "semantic_margin": "RAG_ROUTE_MARGIN",
        "conflict_margin": "RAG_ROUTE_CONFLICT_MARGIN",
        "top_k_per_task": "RAG_ROUTE_TOP_K",
        "max_routes": "RAG_ROUTE_MAX_ROUTES",
    }
    for key, env_name in env_map.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            continue
        try:
            defaults[key] = int(value) if key in {"top_k_per_task", "max_routes"} else float(value)
        except ValueError:
            continue
    return defaults


def _plan_validation_errors(plan: QueryPlan, *,
                            must_keep: set[str] | None = None) -> list[str]:
    profile = _routing_profile()
    errors: list[str] = []
    if plan.decision not in {"answer", "clarify", "fallback"}:
        errors.append("invalid_decision")
    if plan.intent_frame.service_mode not in SERVICE_MODES:
        errors.append("invalid_service_mode")
    else:
        allowed_operations = SERVICE_REGISTRY[plan.intent_frame.service_mode]["operations"]
        if any(operation not in allowed_operations for operation in plan.intent_frame.operations):
            errors.append("operation_not_allowed_for_service")
    if len(plan.routes) > int(profile["max_routes"]):
        errors.append("too_many_routes")
    keys = {f"{route.source}.{route.operation}" for route in plan.routes}
    if must_keep and not must_keep <= keys:
        errors.append("required_route_removed")
    sources = {route.source for route in plan.routes}
    known_answer_objects = {
        answer_object for definition in READ_SOURCE_REGISTRY
        for answer_object in definition.answer_objects
    }
    answer_definitions = [
        get_route_definition(route.source, route.operation)
        for route in plan.routes if route.source_role == "answer_source"
    ]
    answer_definitions = [item for item in answer_definitions if item is not None]
    if plan.decision == "answer" and not answer_definitions:
        errors.append("missing_answer_source")
    if (plan.intent_frame.answer_object in known_answer_objects and answer_definitions
            and not any(plan.intent_frame.answer_object in item.answer_objects
                        for item in answer_definitions)):
        errors.append("answer_object_route_mismatch")
    registered_contracts = {
        item.key for route in plan.routes if route.source_role == "answer_source"
        for item in route_output_contracts(route.source, route.operation)
    }
    frame_contracts = {
        f"{item.get('entity_type')}.{item.get('projection')}"
        for item in plan.intent_frame.output_contracts
        if item.get("entity_type") and item.get("projection")
    }
    if frame_contracts and frame_contracts != registered_contracts:
        errors.append("output_contract_route_mismatch")
    graph: dict[str, set[str]] = {source: set() for source in sources}

    def contains_storage_target(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower() in {
                    "path", "file", "file_path", "collection", "collection_name",
                } or contains_storage_target(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_storage_target(item) for item in value)
        return False

    for route in plan.routes:
        if route.source not in ROUTE_OPERATIONS:
            errors.append(f"unknown_source:{route.source}")
            continue
        if route.operation not in ROUTE_OPERATIONS[route.source]:
            errors.append(f"unknown_operation:{route.source}.{route.operation}")
        if route.source_role not in SOURCE_ROLES:
            errors.append(f"invalid_source_role:{route.source_role}")
        if contains_storage_target(route.filters):
            errors.append("arbitrary_storage_target")
        for dependency in route.depends_on:
            if dependency not in sources or dependency == route.source:
                errors.append(f"invalid_dependency:{route.source}->{dependency}")
            else:
                graph.setdefault(route.source, set()).add(dependency)
        definition = get_route_definition(route.source, route.operation)
        if definition and definition.personal and plan.intent_frame.personal_scope == "none":
            errors.append(f"personal_scope_violation:{definition.key}")
        if definition:
            for entity in definition.required_entities:
                if entity == "application":
                    available = (
                        route.source == "application_state"
                        or "application_state" in route.depends_on
                        or bool(route.filters.get("application_id"))
                        or bool(route.filters.get("application_ids"))
                        or bool(route.filters.get("stable_id"))
                    )
                elif entity == "application_jd":
                    available = (
                        route.source == "application_jd"
                        or "application_jd" in route.depends_on
                        or bool(route.filters.get("jd_version_id"))
                        or bool(route.filters.get("jd_version_ids"))
                    )
                elif entity == "time_scope":
                    available = (
                        plan.intent_frame.time_scope.get("kind") != "none"
                        or any(route.filters.get(key)
                               for key in ("date", "date_from", "date_to", "time_scope"))
                        or any(token in route.subquery for token in (
                            "今天", "昨天", "本周", "上周", "最近", "近期",
                        ))
                    )
                else:
                    available = False
                if not available:
                    errors.append(f"missing_required_entity:{definition.key}:{entity}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(source: str) -> bool:
        if source in visiting:
            return True
        if source in visited:
            return False
        visiting.add(source)
        cycle = any(visit(dep) for dep in graph.get(source, ()))
        visiting.remove(source)
        visited.add(source)
        return cycle

    if any(visit(source) for source in graph):
        errors.append("dependency_cycle")
    if plan.decision == "answer" and not plan.routes:
        errors.append("empty_answer_plan")
    return list(dict.fromkeys(errors))


def _candidate_dicts(candidates: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        item = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
        item["score"] = round(float(item.get("score", 0.0)), 6)
        out.append(item)
    return out


def _semantic_route_from_candidate(candidate: dict[str, Any], question: str,
                                   base_routes: list[RouteSpec]) -> RouteSpec:
    key = str(candidate.get("key") or f"{candidate.get('source')}.{candidate.get('operation')}")
    existing = next((route for route in base_routes
                     if f"{route.source}.{route.operation}" == key), None)
    if existing:
        return RouteSpec(
            existing.source, existing.operation, existing.subquery, existing.filters,
            existing.required, existing.depends_on, existing.bindings, existing.route_id,
            str(candidate.get("source_role") or existing.source_role),
            f"语义原型候选 score={candidate.get('score', 0):.3f}", existing.locked,
        )
    return _route(
        str(candidate.get("source")), str(candidate.get("operation")), question,
        source_role=str(candidate.get("source_role") or "answer_source"),
        selection_reason=f"语义原型候选 score={candidate.get('score', 0):.3f}",
    )


def _compound_query(question: str, frame: RequestFrame) -> bool:
    markers = ("以及", "同时", "结合", "分别", "并且", "还要", "然后", "再根据", "给出下一步")
    return (len(frame.answer_objects) > 1 or len(frame.operations) > 1
            or any(marker in question for marker in markers))


def _clarification(candidates: list[dict[str, Any]], frame: RequestFrame) -> str:
    objects = []
    for candidate in candidates[:2]:
        obj = str(candidate.get("answer_object") or "").strip()
        if obj and obj not in objects:
            objects.append(obj)
    labels = {
        "application": "投递记录", "application_experience": "投递经验",
        "project": "个人项目清单", "project_detail": "个人项目内容",
        "reflection": "学习执行与复盘", "profile": "当前个人画像",
        "reference_knowledge": "通用学习资料", "resume_rule": "个人简历规则",
        "application_jd": "投递绑定 JD", "paper_knowledge": "论文资料",
    }
    if len(objects) >= 2:
        return f"你希望我主要查询“{labels.get(objects[0], objects[0])}”，还是“{labels.get(objects[1], objects[1])}”？"
    if frame.unresolved_references:
        return "你提到的“这个/前面内容”具体指哪一项？请补充对象后我再检索。"
    return "这个问题可能对应不同的个人记录来源。请说明你希望查询的主要对象。"


def _planner_payload(question: str, context: list[str], frame: RequestFrame,
                     candidates: list[dict[str, Any]], must_keep: set[str]) -> str:
    candidate_keys = [str(item.get("key")) for item in candidates]
    payload = {
        "question": question[:1200],
        "recent_user_questions": context[-3:],
        "intent_frame": frame.to_dict(),
        "candidate_routes": candidates,
        "route_definitions": registry_summary(candidate_keys),
        "must_keep_routes": sorted(must_keep),
        "constraints": {
            "max_routes": _routing_profile()["max_routes"],
            "different_answer_centers_require_clarification": True,
            "personal_content_available": False,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _apply_llm_arbiter(base: QueryPlan, question: str, context: list[str],
                       candidates: list[dict[str, Any]], *,
                       llm_call: Callable[[str], str | None] | None) -> QueryPlan:
    must_keep_routes = [route for route in base.routes if route.locked or route.required]
    must_keep = {f"{route.source}.{route.operation}" for route in must_keep_routes}
    allowed = {str(item.get("key")) for item in candidates} | {
        f"{route.source}.{route.operation}" for route in base.routes
    }
    payload = _planner_payload(question, context, base.intent_frame, candidates, must_keep)
    text = _call_with_timeout(
        llm_call or _default_llm_call, payload,
        float(os.environ.get("RAG_PLANNER_TIMEOUT_SECONDS", "5")),
    )
    data = _extract_json(text or "")
    if not data:
        base.planner_mode = "fallback"
        base.resolver_mode = "fallback"
        base.reason += "；语义裁决超时或输出非法，回退至已校验规则计划"
        base.confidence_factors["llm_status"] = "invalid_or_timeout"
        return base
    decision = str(data.get("decision") or "answer")
    if decision == "clarify":
        base.planner_mode = "llm"
        base.resolver_mode = "llm"
        base.decision = "clarify"
        base.routes = must_keep_routes
        base.clarification = str(data.get("clarification") or _clarification(candidates, base.intent_frame))[:300]
        base.reason += "；LLM 判定答案中心存在歧义"
        base.answer_sections = []
        base.confidence_factors["llm_status"] = "clarify"
        return base
    selected = _validated_llm_routes(data, question, allowed_keys=allowed)
    selected_keys = {f"{route.source}.{route.operation}" for route in selected}
    final_routes = _dedupe(must_keep_routes + [
        route for route in selected
        if f"{route.source}.{route.operation}" not in must_keep
    ])
    plan = QueryPlan(
        planner_mode="llm", routes=final_routes, intent_frame=base.intent_frame,
        entities=base.entities, answer_sections=_answer_sections(final_routes),
        reason=base.reason + "；LLM 在语义候选白名单内完成裁决",
        decision="answer" if decision != "fallback" else "fallback",
        resolver_mode="llm", candidate_routes=candidates,
        confidence_factors={**base.confidence_factors, "llm_status": "valid",
                            "llm_selected": sorted(selected_keys)},
    )
    errors = _plan_validation_errors(plan, must_keep=must_keep)
    if errors:
        base.planner_mode = "fallback"
        base.resolver_mode = "fallback"
        base.reason += "；LLM 计划校验失败，使用规则计划"
        base.confidence_factors.update({"llm_status": "validation_failed",
                                        "validation_errors": errors})
        return base
    return plan


def _hybrid_plan(question: str, context: list[str], base: QueryPlan, *,
                 llm_call: Callable[[str], str | None] | None,
                 semantic_ranker: Callable[..., Any] | None = None,
                 force_llm: bool = False) -> QueryPlan:
    profile = _routing_profile()
    locked = [route for route in base.routes if route.locked]
    soft = [route for route in base.routes if not route.locked]
    if locked and not soft and not base.intent_frame.unresolved_references:
        base.resolver_mode = "hard_rule"
        base.confidence_factors.update({"embedding_calls": 0, "llm_calls": 0})
        return base
    started = time.perf_counter()
    try:
        ranker = semantic_ranker or rank_route_candidates
        ranked = ranker(question, base.intent_frame,
                        top_k=int(profile["top_k_per_task"]))
        if isinstance(ranked, tuple):
            raw_candidates, semantic_meta = ranked
        else:
            raw_candidates, semantic_meta = ranked, {}
        candidates = _candidate_dicts(list(raw_candidates))
    except Exception as exc:
        base.planner_mode = "fallback"
        base.resolver_mode = "fallback"
        base.reason += "；语义候选不可用，回退至规则计划"
        base.confidence_factors.update({"semantic_status": "error",
                                        "semantic_error": type(exc).__name__,
                                        "embedding_calls": 1, "llm_calls": 0})
        return base
    base.candidate_routes = candidates
    base.confidence_factors.update({
        "semantic_status": "ok", "semantic_profile": semantic_meta,
        "semantic_ms": round((time.perf_counter() - started) * 1000, 1),
        "embedding_calls": 1,
    })
    if not candidates:
        base.planner_mode = "fallback"
        base.resolver_mode = "fallback"
        base.reason += "；语义候选为空，回退至规则计划"
        base.confidence_factors["llm_calls"] = 0
        return base

    top = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin = float(top["score"]) - float(runner_up["score"]) if runner_up else 1.0
    frame_objects = set(base.intent_frame.answer_objects)
    clear_single_frame = bool(
        len(frame_objects) == 1 and base.intent_frame.confidence >= 0.8
        and not (base.intent_frame.unresolved_references and not context)
    )
    conflict = bool(
        runner_up
        and top.get("answer_object") != runner_up.get("answer_object")
        and margin < float(profile["conflict_margin"])
        and not clear_single_frame
    )
    compound = _compound_query(question, base.intent_frame)
    base.confidence_factors.update({
        "semantic_top_score": top["score"], "semantic_margin": round(margin, 6),
        "answer_center_conflict": conflict, "compound_query": compound,
    })
    base_validation_errors = _plan_validation_errors(base)
    base_has_specific_source = (
        base.intent_frame.service_mode == "explain"
        or any(route.source not in {"reference_kb", "general_fallback"}
               for route in base.routes)
    )
    coherent_base = bool(
        not force_llm
        and base.routes
        and base_has_specific_source
        and not base_validation_errors
        and not compound
        and not (base.intent_frame.unresolved_references and not context)
        and base.intent_frame.confidence >= 0.65
    )
    if coherent_base:
        # 语义层是高精度规则计划的校验器，不是第二个互相覆盖的
        # 路由器。当 RequestFrame 完整、答案中心单一且计划通过契约校验
        # 时，保留整个基础计划，避免 Top-1 候选删掉互补来源。
        base.planner_mode = "semantic"
        base.resolver_mode = "semantic"
        base.reason += "；语义原型已校验该单一答案中心，保留已验证计划"
        base.confidence_factors.update({
            "semantic_guard": "preserve_validated_plan", "llm_calls": 0,
        })
        return base
    if conflict and not compound:
        base.planner_mode = "semantic"
        base.resolver_mode = "semantic"
        base.decision = "clarify"
        base.clarification = _clarification(candidates, base.intent_frame)
        base.routes = locked
        base.answer_sections = []
        base.reason += "；语义候选指向不同答案中心，先澄清而不执行并集"
        base.confidence_factors["llm_calls"] = 0
        return base

    ambiguous = bool(
        force_llm or compound
        or (base.intent_frame.unresolved_references and not context)
        or float(top["score"]) < float(profile["semantic_accept_score"])
        or margin < float(profile["semantic_margin"])
        or (soft and f"{soft[0].source}.{soft[0].operation}" != top.get("key"))
    )
    if ambiguous and os.environ.get("RAG_SCOPED_LLM_PLANNER", "1") == "1":
        base.confidence_factors["llm_calls"] = 1
        return _apply_llm_arbiter(base, question, context, candidates, llm_call=llm_call)

    if ambiguous:
        base.planner_mode = "fallback"
        base.resolver_mode = "fallback"
        base.reason += "；语义候选需要裁决但 LLM 已关闭，保守使用规则计划"
        base.confidence_factors["llm_calls"] = 0
        return base

    selected = _semantic_route_from_candidate(top, question, base.routes)
    required = [
        route for route in base.routes
        if route.locked or route.required
        or route.source_role in {"filter_source", "validation_source"}
    ]
    routes = _dedupe(required + [selected])
    plan = QueryPlan(
        planner_mode="semantic", routes=routes, intent_frame=base.intent_frame,
        entities=base.entities, answer_sections=_answer_sections(routes),
        reason=base.reason + "；高置信语义原型选择",
        decision="answer", resolver_mode="semantic", candidate_routes=candidates,
        confidence_factors={**base.confidence_factors, "llm_calls": 0},
    )
    must_keep = {f"{route.source}.{route.operation}" for route in required}
    errors = _plan_validation_errors(plan, must_keep=must_keep)
    if errors:
        base.planner_mode = "fallback"
        base.resolver_mode = "fallback"
        base.reason += "；语义计划校验失败，使用规则计划"
        base.confidence_factors.update({"validation_errors": errors, "llm_calls": 0})
        return base
    return plan


def _request_frame_personal_scope(value: str) -> str:
    return "none" if value == "none" else ("artifact" if value == "mixed" else "current")


def _semantic_outcome_telemetry(outcome: SemanticPlanningOutcome) -> dict[str, Any]:
    meta = outcome.meta
    return {
        "route_gateway": meta.gateway,
        "route_reasoning_effort": meta.reasoning_effort,
        "planner_cache_hit": outcome.cache_hit,
        "planner_queue_ms": meta.planner_queue_ms,
        "planner_provider_ms": meta.planner_provider_ms,
        "planner_wall_ms": meta.planner_wall_ms,
        "planner_deadline_ms": meta.planner_deadline_ms,
        "planner_timeout_stage": meta.planner_timeout_stage,
        "planner_late_response": meta.planner_late_response,
        "planner_circuit_state": meta.planner_circuit_state,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "planner_elapsed_ms": outcome.elapsed_ms,
        "prompt_chars": outcome.prompt_chars,
        "structured_output_mode": meta.structured_output_mode,
        "json_capability": getattr(meta, "json_capability", "not_requested"),
    }


def _plan_from_semantic(question: str, base: QueryPlan,
                        outcome: SemanticPlanningOutcome, *,
                        context_resolution: dict[str, Any] | None = None) -> QueryPlan | None:
    semantic = outcome.plan
    if semantic is None:
        return None
    task_roles = {task.task_id: semantic_task_route_roles(task) for task in semantic.tasks}
    task_objects = {
        task.task_id: semantic_task_answer_objects(task) for task in semantic.tasks
    }
    task_contracts = {
        task.task_id: semantic_task_output_contracts(task) for task in semantic.tasks
    }
    task_operations = {
        task.task_id: list(dict.fromkeys(
            generic_operation_for_route(key) for key in task.answer_routes
        )) for task in semantic.tasks
    }
    all_objects = list(dict.fromkeys(
        value for task in semantic.tasks for value in task_objects[task.task_id]
    ))
    all_operations = list(dict.fromkeys(
        value for task in semantic.tasks for value in task_operations[task.task_id]
    ))
    all_contracts = list({
        f"{value.get('entity_type')}.{value.get('projection')}": value
        for task in semantic.tasks for value in task_contracts[task.task_id]
    }.values())
    action_request = (
        semantic.action_request.model_dump() if semantic.action_request is not None else {}
    )
    context_resolution = dict(context_resolution or {})
    relation = semantic.turn_relation
    explicit_reference_statuses = {
        "resolved", "ambiguous", "missing_antecedent", "missing_entity_id",
        "legacy_client_context",
    }
    relation_reasons: list[str] = []
    if (relation == "standalone"
            and context_resolution.get("status") in explicit_reference_statuses):
        relation = "continuation"
        relation_reasons.append("explicit_reference_implies_continuation")
    relation_target_turn_id = ""
    if relation in {"continuation", "correction"}:
        if context_resolution.get("source") == "server":
            relation_target_turn_id = str(context_resolution.get("turn_id") or "")
        if context_resolution.get("status") == "candidate":
            context_resolution = {
                **context_resolution,
                "status": "relation_resolved",
                "reason": f"latest_successful_turn_used_for_{relation}",
            }
    elif context_resolution.get("status") == "candidate":
        context_resolution = {}
    if (action_request and semantic.interaction_kind in {"command", "how_to"}
            and semantic.service_mode == "guide"
            and context_resolution.get("status") in {
                "ambiguous", "missing_antecedent", "missing_entity_id",
            }):
        context_resolution = {
            **context_resolution,
            "original_status": context_resolution.get("status"),
            "status": "not_required_for_guide",
            "reason": "guide_only_does_not_bind_or_execute_entity",
        }
    if (action_request and not action_request.get("target_refs")
            and context_resolution.get("status") in {"resolved", "relation_resolved"}
            and context_resolution.get("entity_type") == action_request.get("target_type")):
        action_request["target_refs"] = list(context_resolution.get("target_refs") or [])
    frame = RequestFrame(
        service_mode=semantic.service_mode,
        interaction_kind=semantic.interaction_kind,
        turn_relation=relation,
        relation_target_turn_id=relation_target_turn_id,
        action_request=action_request,
        requested_action=str(action_request.get("verb") or ""),
        personal_scope=_request_frame_personal_scope(semantic_personal_scope(semantic)),
        domains=list(dict.fromkeys(
            key.split(".", 1)[0] for task in semantic.tasks
            for key in task_roles[task.task_id]
        )),
        time_scope=dict(base.intent_frame.time_scope),
        tasks=all_operations,
        answer_object=(all_objects[0] if all_objects else ""),
        answer_objects=all_objects,
        output_contracts=all_contracts,
        capability_ids=list(semantic.capability_ids),
        operations=all_operations,
        source_roles=[],
        usage_contexts=list(base.intent_frame.usage_contexts),
        filters=dict(base.intent_frame.filters),
        confidence=0.0,
        unresolved_references=list(base.intent_frame.unresolved_references),
        signals=["llm_semantic_plan"],
        deterministic_signals=list(base.intent_frame.deterministic_signals),
        routing_assurance="schema_validated",
        decision_reasons=[
            "structured_v4_contract_validated",
            *relation_reasons,
            *( ["top_chat_write_is_guide_only"]
               if semantic.interaction_kind == "command" else [] ),
        ],
        context_resolution=context_resolution,
    )
    task_sources = {
        task.task_id: tuple(dict.fromkeys(
            key.split(".", 1)[0] for key in task_roles[task.task_id]
        ))
        for task in semantic.tasks
    }
    all_sources = {
        source for sources in task_sources.values() for source in sources
    }

    def registry_dependencies(source: str, operation: str) -> tuple[str, ...]:
        definition = get_route_definition(source, operation)
        if definition is None:
            return ()
        inferred: list[str] = []
        if "application" in definition.required_entities:
            if "application_state" in all_sources and source != "application_state":
                inferred.append("application_state")
        if "application_jd" in definition.required_entities:
            if "application_jd" in all_sources and source != "application_jd":
                inferred.append("application_jd")
        return tuple(inferred)

    routes: list[RouteSpec] = []
    source_roles: list[dict[str, str]] = []
    for task in semantic.tasks:
        role_by_key = task_roles[task.task_id]
        dependencies = tuple(dict.fromkeys(
            source for dependency in task.depends_on
            for source in task_sources.get(dependency, ())
        ))
        for key, role in role_by_key.items():
            source, operation = key.split(".", 1)
            definition = get_route_definition(source, operation)
            if definition is None:
                return None
            filters = {
                str(k)[:80]: v for k, v in task.filters.items()
                if str(k).lower() not in {
                    "path", "file", "file_path", "collection", "collection_name",
                }
            }
            if source == "product_help":
                filters["capability_ids"] = list(semantic.capability_ids)
                filters["topics"] = capability_topics(semantic.capability_ids)
                filters["action_request"] = action_request
                filters["action"] = str(action_request.get("verb") or "locate")
            elif context_resolution.get("status") in {"resolved", "relation_resolved"}:
                refs = list(context_resolution.get("target_refs") or [])
                app_ids = [item.get("entity_id") for item in refs
                           if item.get("entity_type") == "application"]
                jd_ids = [item.get("entity_id") for item in refs
                          if item.get("entity_type") == "application_jd"]
                if app_ids and source == "application_state":
                    filters["application_ids"] = app_ids
                    filters["strict_entity_filter"] = True
                if jd_ids and source == "application_jd":
                    filters["jd_version_ids"] = jd_ids
            route_dependencies = tuple(dict.fromkeys((
                *(dep for dep in dependencies if dep != source),
                *registry_dependencies(source, operation),
            )))
            routes.append(_route(
                source, operation, task.subquery,
                required=role == "answer_source", filters=filters,
                depends_on=route_dependencies,
                source_role=role,
                selection_reason=f"结构化语义规划 task={task.task_id}",
                locked=(source == "product_help"),
            ))
            source_roles.append({"route": key, "role": role})
    frame.source_roles = source_roles
    completeness_parts = [
        bool(semantic.tasks),
        all(task.subquery for task in semantic.tasks),
        all(task.answer_routes for task in semantic.tasks),
        all(task_objects[task.task_id] for task in semantic.tasks),
    ]
    frame.confidence = 0.0
    candidate_routes = [{
        "task_id": task.task_id,
        "answer_objects": task_objects[task.task_id],
        "output_contracts": task_contracts[task.task_id],
        "operations": task_operations[task.task_id],
        "answer_routes": list(task.answer_routes),
        "context_routes": [item.model_dump() for item in task.context_routes],
    } for task in semantic.tasks]
    plan = QueryPlan(
        planner_mode="llm", routes=_dedupe(routes), intent_frame=frame,
        entities=base.entities, answer_sections=_answer_sections(routes),
        reason="非硬问题由一次受约束结构化语义规划生成",
        decision=semantic.decision, resolver_mode="llm",
        candidate_routes=candidate_routes,
        clarification=semantic.clarification,
        confidence_factors={
            "schema_validated": True,
            "entity_coverage": bool(base.entities),
            "route_consistency": True,
            "llm_calls": outcome.meta.calls,
            **_semantic_outcome_telemetry(outcome),
        },
        planner_engine="structured_llm",
        planner_version=SEMANTIC_PLANNER_VERSION,
        schema_version=SEMANTIC_QUERY_SCHEMA_VERSION,
        route_model=outcome.meta.model,
        repair_used=outcome.meta.repair_used,
    )
    if semantic.decision == "clarify":
        plan.routes = []
        plan.answer_sections = []
    errors = _plan_validation_errors(plan)
    if "dependency_cycle" in errors:
        # The model may express a cyclic task DAG even though the selected read
        # routes are sound. Rebuild only the execution edges from the registry's
        # entity contracts; never invent or remove a data source.
        plan.routes = [replace(
            route,
            depends_on=registry_dependencies(route.source, route.operation),
        ) for route in plan.routes]
        repaired_errors = _plan_validation_errors(plan)
        if "dependency_cycle" not in repaired_errors:
            errors = repaired_errors
            plan.repair_used = True
            plan.confidence_factors["dependency_cycle_repaired"] = True
    if errors:
        outcome.meta.errors.extend(errors)
        return None
    return plan


def _semantic_failure_fallback(question: str, base: QueryPlan, *,
                               semantic_ranker: Callable[..., Any] | None,
                               reason: str) -> QueryPlan:
    """Fail closed when the structured planner is unavailable or invalid.

    Prototype similarity is an offline evaluation signal only. It can never
    authorize a source or turn a write command into a read in production.
    """
    clarification = (
        "我还不能可靠判断你希望查询哪一类信息。请说明主要想了解："
        "OfferClaw 使用方法、你的个人记录，还是通用求职/技术知识？"
    )
    base.intent_frame.routing_assurance = "degraded"
    base.intent_frame.decision_reasons = ["structured_planner_unavailable"]
    return QueryPlan(
        planner_mode="fallback", routes=[], intent_frame=base.intent_frame,
        entities=base.entities, answer_sections=[], reason="语义规划不可用，准确性优先并保守澄清",
        decision="clarify", resolver_mode="fallback", candidate_routes=[],
        clarification=clarification,
        confidence_factors={
            "prototype_execution_allowed": False,
            "llm_calls": 1,
        },
        planner_engine="structured_llm_unavailable",
        planner_version=SEMANTIC_PLANNER_VERSION,
        schema_version=SEMANTIC_QUERY_SCHEMA_VERSION,
        fallback_reason=reason,
    )


def _intelligent_plan(question: str, context: list[str], base: QueryPlan, *,
                      llm_call: Callable[[str], str | None] | None,
                      semantic_ranker: Callable[..., Any] | None,
                      context_resolution: dict[str, Any] | None = None) -> QueryPlan:
    outcome = plan_semantically(
        question, context, context_resolution=context_resolution,
        legacy_text_call=llm_call,
        timeout_seconds=float(os.environ.get("RAG_ROUTE_TIMEOUT_SECONDS", "60") or 60),
        lane="online",
    )
    plan = _plan_from_semantic(
        question, base, outcome, context_resolution=context_resolution,
    )
    if plan is not None:
        return plan
    reason = ",".join(outcome.meta.errors) or "invalid_semantic_plan"
    fallback = _semantic_failure_fallback(
        question, base, semantic_ranker=semantic_ranker, reason=reason,
    )
    fallback.route_model = outcome.meta.model
    fallback.repair_used = outcome.meta.repair_used
    fallback.confidence_factors.update({
        "structured_call_errors": list(outcome.meta.errors),
        "llm_calls": outcome.meta.calls,
        **_semantic_outcome_telemetry(outcome),
    })
    return fallback


def _context_clarification(question: str, resolution: dict[str, Any]) -> QueryPlan:
    reason = str(resolution.get("reason") or "unresolved_conversation_reference")
    if reason == "singular_reference_has_multiple_entities":
        clarification = "上一轮包含多个对象。请说明你指的是哪一条投递、岗位或项目。"
    else:
        clarification = "我找不到可可靠绑定的上一轮对象。请直接说明具体投递、岗位、项目或记录。"
    frame = RequestFrame(
        interaction_kind="query", turn_relation="continuation", service_mode="recall",
        relation_target_turn_id=str(resolution.get("turn_id") or ""),
        unresolved_references=[question], routing_assurance="ambiguous",
        decision_reasons=[reason], context_resolution=resolution,
        signals=["server_context_clarification"],
    )
    return QueryPlan(
        planner_mode="v4", routes=[], intent_frame=frame,
        reason="服务端会话上下文无法唯一绑定", decision="clarify",
        resolver_mode="semantic", clarification=clarification,
        confidence_factors={"llm_calls": 0, "context_gate": reason},
        planner_engine="context_gate",
        planner_version=SEMANTIC_PLANNER_VERSION,
        schema_version=SEMANTIC_QUERY_SCHEMA_VERSION,
    )


def _bind_relation_candidate(plan: QueryPlan,
                             resolution: dict[str, Any]) -> QueryPlan:
    """Bind a server-authored prior turn after the model selects a relation."""
    frame = plan.intent_frame
    if (plan.decision != "answer" or frame.turn_relation == "standalone"
            or resolution.get("status") != "candidate"):
        return plan
    relation = frame.turn_relation
    frame.relation_target_turn_id = str(resolution.get("turn_id") or "")
    frame.context_resolution = {
        **resolution,
        "status": "relation_resolved",
        "reason": f"latest_successful_turn_used_for_{relation}",
    }
    refs = list(resolution.get("target_refs") or [])
    action = frame.action_request
    if (action and not action.get("target_refs")
            and resolution.get("entity_type") == action.get("target_type")):
        action["target_refs"] = refs
    app_ids = [item.get("entity_id") for item in refs
               if item.get("entity_type") == "application"]
    jd_ids = [item.get("entity_id") for item in refs
              if item.get("entity_type") == "application_jd"]
    for route in plan.routes:
        if app_ids and route.source == "application_state":
            route.filters["application_ids"] = app_ids
            route.filters["strict_entity_filter"] = True
        if jd_ids and route.source == "application_jd":
            route.filters["jd_version_ids"] = jd_ids
    return plan


def plan_query(question: str, *, context: list[str] | None = None,
               conversation_id: str = "",
               llm_call: Callable[[str], str | None] | None = None,
               semantic_ranker: Callable[..., Any] | None = None) -> QueryPlan:
    """Plan free-text input through the single production v4 semantic path.

    ``rule_plan_query`` remains available only as an explicit offline baseline.
    Environment variables cannot downgrade user traffic to that implementation.
    """
    context = [str(x).strip()[:500] for x in (context or []) if str(x).strip()][-1:]
    from conversation_context import resolve_conversation_context
    resolution = resolve_conversation_context(
        question, conversation_id, legacy_context=context,
    ).to_dict()
    planner_resolution = (
        None if resolution.get("status") in {"none", "candidate"} else resolution
    )
    semantic = _intelligent_plan(
        question, context, _semantic_base_plan(question), llm_call=llm_call,
        semantic_ranker=semantic_ranker,
        context_resolution=planner_resolution,
    )
    semantic = _bind_relation_candidate(semantic, resolution)
    if (resolution.get("status") in {
            "ambiguous", "missing_antecedent", "missing_entity_id",
        } and semantic.decision == "answer"
            and semantic.intent_frame.interaction_kind not in {"command", "how_to"}):
        return _context_clarification(question, resolution)
    return semantic


def plan_has_personal_source(plan: QueryPlan) -> bool:
    return any(route.source in PERSONAL_ROUTES for route in plan.routes)


def plan_is_reference_only(plan: QueryPlan) -> bool:
    return (bool(plan.routes)
            and plan.intent_frame.service_mode == "explain"
            and plan.intent_frame.personal_scope == "none"
            and all(r.source == "reference_kb" for r in plan.routes))
