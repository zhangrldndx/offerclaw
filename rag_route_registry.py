# -*- coding: utf-8 -*-
"""OfferClaw read-source registry and offline prototype diagnostics.

The online v3 planner selects typed read routes directly. Prototype vectors in
this module are retained only for offline evaluation and never authorize an
online route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable


# v8(2026-08-31):"本系统工程事实"类归属修复(UI_JOURNEY_FINDINGS 发现 1)。
# reference_kb 正面认领配置/阈值/指标/选型类例句,project_memory 以对比例句
# 推开系统事实;升版号使原型缓存与规划器提示词缓存一并重建。
REGISTRY_VERSION = "route-registry-v9"
SERVICE_REGISTRY_VERSION = "service-registry-v1"
SOURCE_ROLES = {
    "answer_source", "supporting_context", "filter_source", "validation_source",
}

# 顶部问答先判断用户需要哪类只读服务，再选择下方数据源。服务模式与数据源
# 分开登记，避免把“项目/投递/学习”等用途词继续扩展成新路由。
SERVICE_REGISTRY: dict[str, dict[str, Any]] = {
    "guide": {
        "operations": {"guide"}, "allow_general_fallback": False,
        "description": "解释真实功能入口、前置条件、限制与确认边界。",
    },
    "recall": {
        "operations": {"list", "lookup", "search", "compare", "summarize"},
        "allow_general_fallback": False,
        "description": "读取用户已经保存并确认的个人事实或历史。",
    },
    "explain": {
        "operations": {"list", "lookup", "search", "compare", "summarize", "explain", "advise"},
        "allow_general_fallback": True,
        "description": "用策展资料、论文或明确标注的模型常识解释专业知识。",
    },
    "advise": {
        "operations": {"list", "lookup", "search", "compare", "summarize", "advise"},
        "allow_general_fallback": False,
        "description": "组合个人证据与参考资料生成只读个性化建议。",
    },
    "diagnose": {
        "operations": {"diagnose"}, "allow_general_fallback": False,
        "description": "解释无结果、未关联、未审批、过期或来源不可用。",
    },
}
SERVICE_MODES = frozenset(SERVICE_REGISTRY)


@dataclass(frozen=True)
class RouteDefinition:
    source: str
    operation: str
    answer_objects: tuple[str, ...]
    source_role: str
    description: str
    prototypes: tuple[str, ...]
    contrast_prototypes: tuple[str, ...] = ()
    personal: bool = False
    required_entities: tuple[str, ...] = ()
    planner_visible: bool = True

    @property
    def key(self) -> str:
        return f"{self.source}.{self.operation}"


@dataclass(frozen=True)
class OutputContract:
    entity_type: str
    projection: str

    @property
    def key(self) -> str:
        return f"{self.entity_type}.{self.projection}"

    def to_dict(self) -> dict[str, str]:
        return {"entity_type": self.entity_type, "projection": self.projection}


@dataclass(frozen=True)
class RouteCandidate:
    source: str
    operation: str
    answer_object: str
    source_role: str
    score: float
    semantic_score: float
    frame_bonus: float
    description: str

    @property
    def key(self) -> str:
        return f"{self.source}.{self.operation}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"key": self.key}


def _d(source: str, operation: str, objects: tuple[str, ...], role: str,
       description: str, prototypes: tuple[str, ...], *,
       contrasts: tuple[str, ...] = (), personal: bool = False,
       entities: tuple[str, ...] = (), planner_visible: bool = True) -> RouteDefinition:
    return RouteDefinition(source, operation, objects, role, description, prototypes,
                           contrasts, personal, entities, planner_visible)


READ_SOURCE_REGISTRY: tuple[RouteDefinition, ...] = (
    _d("product_help", "guide", ("system_capability",), "answer_source",
       "确定性说明 OfferClaw 的功能入口、审批流程和只读边界。",
       ("怎么把新项目加入OfferClaw", "在哪里给投递关联JD", "系统能帮我修改画像吗"),
       contrasts=("我的项目有哪些", "解释如何写项目经历", "我刚才添加了什么")),
    _d("system_diagnostics", "inspect", ("system_diagnostic",), "answer_source",
       "只读检查个人数据完整性、卡片前置条件和检索运行状态。",
       ("为什么没有检索结果", "为什么没找到我的项目", "为什么这条投递不能进入学习计划"),
       contrasts=("为什么RAG需要重排", "解释为什么要写项目经历")),
    _d("application_state", "list_current", ("application",), "answer_source",
       "读取当前投递记录、企业、岗位和状态。",
       ("我目前有哪些投递记录", "现在各岗位进展到哪里", "当前求职投递状态"),
       contrasts=("我有哪些个人项目", "解释如何投递简历"), personal=True),
    _d("application_state", "list_ever_applied", ("application",), "answer_source",
       "确定性列出已经正式投递过的企业或岗位。",
       ("我投过哪几家公司", "已经投递了哪些岗位", "列出我正式投出去的申请"),
       contrasts=("可用于投递的项目有哪些",), personal=True),
    _d("application_state", "list_pending_submission", ("application",), "answer_source",
       "确定性列出准备投递、尚待官网提交的记录。",
       ("接下来还要投哪些公司", "哪些岗位准备投递但还没提交", "待官网投递清单"),
       contrasts=("适合投递的项目有哪些",), personal=True),
    _d("application_state", "list_failed", ("application",), "answer_source",
       "确定性列出状态为已拒绝的投递。",
       ("哪些公司拒绝了我", "我失败的投递有哪些", "列出已经被拒的岗位"),
       personal=True),
    _d("application_state", "get_next_actions", ("application",), "answer_source",
       "读取投递记录中已保存的下一步动作。",
       ("每条投递下一步要做什么", "接下来需要完成哪些投递动作", "投递待办"),
       personal=True),
    _d("application_experience", "search", ("application_experience",), "answer_source",
       "检索个人笔试、面试、失败复盘和经验总结。",
       ("根据我的面试经验总结问题", "以前被拒的原因是什么", "检索个人投递复盘"),
       contrasts=("总结我的学习复盘",), personal=True),
    _d("application_jd", "get_bound_jd", ("application_jd",), "answer_source",
       "按投递 ID 精确读取当前绑定 JD。",
       ("这条投递绑定的JD是什么", "查看当前岗位完整职位描述", "读取投递对应JD"),
       personal=True, entities=("application",)),
    _d("application_jd", "search_bound_jd", ("application_jd",), "answer_source",
       "在目标投递绑定的 JD 中检索职责或任职要求。",
       ("这些岗位的JD要求哪些能力", "当时职位描述里怎么写的", "对应岗位要求是什么"),
       contrasts=("一般岗位需要什么能力",), personal=True,
       entities=("application",)),
    _d("application_jd", "get_match_snapshot", ("profile_gap",), "validation_source",
       "按投递和活动 JD 精确读取已确认匹配快照、能力缺口与建议。",
       ("这条投递的匹配缺口是什么", "这个JD对应的能力短板", "岗位匹配快照里记录了哪些缺口"),
       contrasts=("我整体还缺少什么能力",), personal=True, entities=("application",)),
    _d("application_jd", "compare_versions", ("application_jd",), "answer_source",
       "比较同一投递的 JD 历史版本。",
       ("这个JD版本有什么变化", "比较岗位描述新旧版本", "JD更新了哪些要求"),
       personal=True, entities=("application",)),
    _d("profile_plan", "get_profile", ("profile",), "answer_source",
       "读取用户确认的当前个人画像和能力状态。",
       ("我现在掌握哪些技能", "查看我的正式画像", "目前我的能力情况"),
       personal=True),
    _d("profile_plan", "get_plan", ("plan",), "answer_source",
       "读取当前学习计划。",
       ("我当前学习计划是什么", "今天计划要做什么", "查看现在的学习安排"),
       personal=True),
    _d("profile_plan", "get_gaps", ("profile_gap",), "answer_source",
       "读取已确认能力缺口。",
       ("我还缺少什么能力", "当前有哪些技能缺口", "岗位匹配暴露了哪些短板"),
       personal=True),
    _d("profile_plan", "get_recent_log", ("recent_execution",), "answer_source",
       "读取近期执行留痕摘要。",
       ("最近几天我做了什么", "查看近期学习留痕", "最近执行情况"),
       personal=True, planner_visible=False),
    _d("reflection_memory", "get_recent", ("reflection",), "answer_source",
       "读取最近的每日执行和复盘记录。",
       ("最近我学习了什么", "这段时间的执行复盘", "近期学习经历"),
       contrasts=("介绍一种学习方法",), personal=True),
    _d("reflection_memory", "get_by_date", ("reflection",), "answer_source",
       "按明确日期或时间范围读取执行和复盘。",
       ("昨天我完成了什么", "上周学习复盘", "查询某一天的执行记录"),
       personal=True, entities=("time_scope",)),
    _d("reflection_memory", "search_topic", ("reflection",), "answer_source",
       "按技能、阻碍或原因检索长期复盘记忆。",
       ("我以前在哪里卡住BM25", "什么时候实践过LangGraph", "过去为什么总是延期"),
       contrasts=("解释BM25原理",), personal=True),
    _d("reflection_memory", "get_profile_evidence", ("profile_evidence",), "validation_source",
       "检索支持或反驳当前能力判断的历史实践证据。",
       ("根据学习记录判断我是否掌握LangGraph", "这项能力有什么实践证据", "画像里的技能有何依据"),
       personal=True),
    _d("reflection_memory", "get_history_overview", ("reflection",), "answer_source",
       "汇总完整有效学习经历和工作进展。",
       ("总结我以前所做的学习工作", "回顾一路以来的学习经历", "梳理过去完成过的工作"),
       contrasts=("总结一般学习方法",), personal=True),
    _d("project_memory", "list_catalog", ("project",), "answer_source",
       "确定性列出已确认可用的个人项目目录。",
       ("我有哪些个人项目", "可用于投递的开源项目有哪些", "列出当前项目清单"),
       contrasts=("我投了哪些项目相关岗位", "项目有哪些量化结果"), personal=True),
    _d("project_memory", "list_approved", ("project",), "answer_source",
       "确定性列出用户已审批的个人项目材料。",
       ("有哪些已审批项目材料", "知识库中确认过的个人项目", "列出批准入库的项目"),
       personal=True),
    _d("project_memory", "search", ("project_detail",), "answer_source",
       "检索已审批的个人项目材料:正文、架构、实现或量化结果。",
       ("我的项目用了哪些技术", "OfferClaw项目有什么量化结果", "检索项目实现细节"),
       # 资料/文档里的技术配置与阈值属 reference_kb,不是个人项目材料——
       # UI 实测这类问题曾被本路由截胡后 0 证据拒答(发现 1)
       contrasts=("我的项目有哪些", "资料里的检索阈值等配置是多少"), personal=True),
    _d("project_memory", "rank_for_application", ("project_fit",), "answer_source",
       "依据一条投递的活动 JD 和匹配缺口，对已确认个人项目做适配比较。",
       ("这个岗位推荐我使用哪个项目", "哪个个人项目最适合这份JD", "投递这个职位应该主推什么项目"),
       contrasts=("我有哪些项目", "这个项目如何实现"), personal=True,
       entities=("application", "application_jd")),
    _d("resume_rules", "search", ("resume_rule",), "answer_source",
       "检索用户确认的简历模板、写作规则和示例。",
       ("按我的简历模板怎么写", "我的简历格式规则", "检索已确认简历范例"),
       personal=True),
    _d("reference_kb", "search", ("reference_knowledge",), "answer_source",
       "检索用户上传的学习资料、博客、策展文档与工程技术文档"
       "(含配置、阈值、评测指标、组件选型等技术事实)。",
       ("根据参考资料解释RAG", "知识库里如何介绍混合检索", "结合教程说明技术概念",
        # "资料中的技术事实"类锚点(配置阈值/组件选型/评测指标/默认开关四侧面):
        # 这一类问题此前在分类学里无锚可依,全被个人记录路截胡(发现 1)。
        # 措辞取**使用者视角**(问自己资料里讲了什么),不取"问系统自身设定"
        # ——后者不是真实用法(2026-08-31 用户方法论纠偏)
        "资料里拒答的距离阈值是怎么定的", "笔记里精排用的是什么模型",
        "文档里的评测指标是多少", "教程里这个功能默认开还是关"),
       contrasts=("根据我的学习记录", "我的项目有哪些", "我的项目有什么量化结果")),
    _d("paper_kb", "search", ("paper_knowledge",), "answer_source",
       "检索论文和研究文献，仅用于显式论文意图。",
       ("结合论文解释ReAct", "有哪些研究文献支持这个结论", "查找原论文证据")),
)

_BY_KEY = {item.key: item for item in READ_SOURCE_REGISTRY}

_OUTPUT_PROJECTIONS: dict[str, tuple[OutputContract, ...]] = {
    "product_help.guide": (OutputContract("system_capability", "instructions"),),
    "system_diagnostics.inspect": (OutputContract("system", "diagnostic"),),
    "application_state.list_current": (OutputContract("application", "summary"),),
    "application_state.list_ever_applied": (OutputContract("application", "status"),),
    "application_state.list_pending_submission": (OutputContract("application", "status"),),
    "application_state.list_failed": (OutputContract("application", "status"),),
    "application_state.get_next_actions": (OutputContract("application", "next_action"),),
    "application_experience.search": (OutputContract("application", "experience"),),
    "application_jd.get_bound_jd": (OutputContract("application_jd", "full_text"),),
    "application_jd.search_bound_jd": (OutputContract("application_jd", "requirements"),),
    "application_jd.get_match_snapshot": (OutputContract("application", "match_gap"),),
    "application_jd.compare_versions": (OutputContract("application_jd", "version_diff"),),
    "profile_plan.get_profile": (OutputContract("profile", "summary"),),
    "profile_plan.get_plan": (OutputContract("plan", "summary"),),
    "profile_plan.get_gaps": (OutputContract("profile", "gap"),),
    "profile_plan.get_recent_log": (OutputContract("daily_log", "recent"),),
    "reflection_memory.get_recent": (OutputContract("reflection", "recent"),),
    "reflection_memory.get_by_date": (OutputContract("reflection", "date_range"),),
    "reflection_memory.search_topic": (OutputContract("reflection", "topic"),),
    "reflection_memory.get_profile_evidence": (OutputContract("profile", "evidence"),),
    "reflection_memory.get_history_overview": (OutputContract("reflection", "history"),),
    "project_memory.list_catalog": (OutputContract("project", "summary"),),
    "project_memory.list_approved": (OutputContract("project", "approved_material"),),
    "project_memory.search": (OutputContract("project", "detail"),),
    "project_memory.rank_for_application": (OutputContract("project", "application_fit"),),
    "resume_rules.search": (OutputContract("resume", "rule"),),
    "reference_kb.search": (OutputContract("knowledge", "reference"),),
    "paper_kb.search": (OutputContract("knowledge", "paper"),),
}


def route_output_contracts(source: str, operation: str) -> tuple[OutputContract, ...]:
    """Return the typed entity/projection contract for an executable route."""
    key = f"{source}.{operation}"
    explicit = _OUTPUT_PROJECTIONS.get(key)
    if explicit is not None:
        return explicit
    definition = _BY_KEY.get(key)
    return tuple(
        OutputContract(answer_object, "summary")
        for answer_object in (definition.answer_objects if definition else ())
    )

# A route may require an entity produced by an earlier task.  This mapping is
# deliberately domain-contract based (never question-keyword based) and is
# shared by prompt construction and code validation.
ENTITY_PRODUCER_ROUTES: dict[str, tuple[str, ...]] = {
    "application": tuple(
        item.key for item in READ_SOURCE_REGISTRY
        if item.source == "application_state"
    ),
    "application_jd": tuple(
        item.key for item in READ_SOURCE_REGISTRY
        if item.source == "application_jd"
    ),
}


def route_produces_entity(route_key: str, entity: str) -> bool:
    return route_key in ENTITY_PRODUCER_ROUTES.get(entity, ())


def plannable_route_definitions() -> tuple[RouteDefinition, ...]:
    """Routes exposed to a semantic planner.

    Hidden routes remain executable for old checkpoints and deterministic
    compatibility paths, but a new model-authored plan cannot select them.
    """
    return tuple(item for item in READ_SOURCE_REGISTRY if item.planner_visible)


def get_route_definition(source: str, operation: str) -> RouteDefinition | None:
    return _BY_KEY.get(f"{source}.{operation}")


def registry_summary(keys: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = set(keys or ())
    rows = (READ_SOURCE_REGISTRY if not selected else
            tuple(d for d in READ_SOURCE_REGISTRY if d.key in selected))
    return [{
        "source": d.source, "operation": d.operation,
        "answer_objects": list(d.answer_objects), "source_role": d.source_role,
        "description": d.description, "required_entities": list(d.required_entities),
        "output_contracts": [item.to_dict() for item in route_output_contracts(
            d.source, d.operation
        )],
    } for d in rows]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _profile_key() -> tuple[str, dict[str, Any]]:
    from rag_tools import get_embedding_config
    cfg = get_embedding_config()
    stable = {k: cfg.get(k) for k in ("provider", "model", "dimensions")}
    raw = json.dumps({"registry": REGISTRY_VERSION, **stable}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], stable


def _cache_path(profile_key: str) -> Path:
    base = Path(__file__).resolve().parent / ".offerclaw" / "cache"
    return base / f"route_prototypes_{profile_key}.json"


def _prototype_texts() -> tuple[list[str], list[tuple[str, str]]]:
    texts: list[str] = []
    owners: list[tuple[str, str]] = []
    for definition in plannable_route_definitions():
        for text in definition.prototypes:
            texts.append(text)
            owners.append((definition.key, "positive"))
        for text in definition.contrast_prototypes:
            texts.append(text)
            owners.append((definition.key, "contrast"))
    return texts, owners


def _load_or_build_vectors(
    embedder: Callable[[list[str]], list[list[float]]],
    *, persist: bool,
) -> tuple[dict[str, dict[str, list[list[float]]]], dict[str, Any]]:
    profile_key, config = _profile_key()
    path = _cache_path(profile_key)
    texts, owners = _prototype_texts()
    # 例句全文哈希进缓存有效性判定:只比版号的话,改例句忘升版 = 静默用
    # 旧向量,分类学修复形同未部署(2026-08-31 修复发现 1 时挖出的潜伏雷)。
    texts_sha = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    if persist and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("registry_version") == REGISTRY_VERSION
                    and data.get("prototype_sha256") == texts_sha
                    and isinstance(data.get("vectors"), dict)):
                return data["vectors"], {"profile_key": profile_key, "cache": "hit", **config}
        except Exception:
            pass
    encoded = embedder(texts)
    if len(encoded) != len(texts):
        raise ValueError("route prototype embedding count mismatch")
    vectors: dict[str, dict[str, list[list[float]]]] = {}
    for (key, role), vector in zip(owners, encoded):
        vectors.setdefault(key, {"positive": [], "contrast": []})[role].append(vector)
    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps({
            "registry_version": REGISTRY_VERSION,
            "prototype_sha256": texts_sha,
            "embedding": config,
            "vectors": vectors,
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    return vectors, {"profile_key": profile_key, "cache": "miss", **config}


_PROTOTYPE_RUNTIME: dict[str, tuple[
    dict[str, dict[str, list[list[float]]]], dict[str, Any]
]] = {}


def route_prototypes_ready() -> bool:
    """Return whether the current embedding profile was explicitly prewarmed."""
    try:
        profile_key, _ = _profile_key()
    except Exception:
        return False
    return profile_key in _PROTOTYPE_RUNTIME


def clear_route_prototype_runtime() -> None:
    """Testing/admin hook; persistent derivative caches are not removed."""
    _PROTOTYPE_RUNTIME.clear()


def rank_route_candidates(question: str, frame: Any, *, top_k: int = 3,
                          embedder: Callable[[list[str]], list[list[float]]] | None = None,
                          ) -> tuple[list[RouteCandidate], dict[str, Any]]:
    """Return prototype similarity for offline evaluation only.

    ``frame`` remains in the one-release evaluation API so old reports can be
    reproduced, but it cannot add hand-authored score bonuses.
    """
    if embedder is None:
        from rag_tools import get_embeddings_batch
        embedder = lambda texts: get_embeddings_batch(texts, max_retries=1, throttle=0)
    profile_key, _ = _profile_key()
    runtime = _PROTOTYPE_RUNTIME.get(profile_key)
    if runtime is None:
        raise RuntimeError("route_prototypes_not_ready")
    vectors, stored_meta = runtime
    meta = {**stored_meta, "runtime": "ready"}
    query_vector = embedder([question])[0]
    candidates: list[RouteCandidate] = []
    for definition in plannable_route_definitions():
        bucket = vectors.get(definition.key) or {}
        positives = [_cosine(query_vector, v) for v in bucket.get("positive", [])]
        contrasts = [_cosine(query_vector, v) for v in bucket.get("contrast", [])]
        semantic = max(positives, default=0.0)
        contrast = max(contrasts, default=0.0)
        contrast_penalty = max(0.0, contrast - semantic)
        score = max(-1.0, min(1.0, semantic - contrast_penalty))
        answer_object = definition.answer_objects[0]
        candidates.append(RouteCandidate(
            definition.source, definition.operation, answer_object,
            definition.source_role, round(score, 6), round(semantic, 6),
            round(-contrast_penalty, 6), definition.description,
        ))
    candidates.sort(key=lambda item: (-item.score, item.key))
    return candidates[:max(1, top_k)], meta


def prebuild_route_prototype_cache(
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, Any]:
    """显式预热原型缓存，避免首个非硬路由问题承担模型加载与批量编码。"""
    if embedder is None:
        from rag_tools import get_embeddings_batch
        embedder = lambda texts: get_embeddings_batch(texts, max_retries=1, throttle=0)
    vectors, meta = _load_or_build_vectors(embedder, persist=True)
    profile_key = str(meta.get("profile_key") or "")
    if profile_key:
        _PROTOTYPE_RUNTIME[profile_key] = (vectors, dict(meta))
    return {**meta, "route_count": len(vectors), "registry_version": REGISTRY_VERSION}
