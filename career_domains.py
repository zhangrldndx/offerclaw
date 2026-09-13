# -*- coding: utf-8 -*-
"""通用职业域模板与确定性匹配辅助。

设计目标：
1. 保留 ``match_job`` 既有 AI/后端规则与输出，旧规则无法识别时才进入本模块；
2. 用户画像里的 ``方向优先级`` 是第一事实源，模板只负责同义归一，不覆盖用户选择；
3. 模板不做学历/年限等硬门槛裁决，LLM 也不参与本模块；
4. 未知职业仍可通过目标方向与 JD 的直接文本命中工作，不要求穷举全行业词表。
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "career_domain_templates.json")
# JD 正常只有数百到数千字。限制职业域规则扫描窗口，避免异常超长输入让
# ``域数 × 关键词数 × 文本长度`` 的确定性匹配无意义放大；硬门槛解析仍由
# match_job 读取原文，不受此窗口影响。
MAX_DOMAIN_TEXT_CHARS = 20_000

_ROLE_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("backend", ("后端", "服务端", "java开发", "go开发", "python开发")),
    ("frontend", ("前端", "web前端", "javascript", "typescript")),
    ("fullstack", ("全栈", "fullstack", "full-stack")),
    ("mobile", ("客户端", "安卓", "android", "ios", "鸿蒙")),
    ("qa", ("测试开发", "测试工程师", "质量工程")),
    ("devops", ("devops", "sre", "运维开发", "云平台")),
    ("data_engineering", ("数据工程", "数据仓库", "etl", "spark")),
    ("llm_application", ("ai应用", "大模型应用", "llm应用", "agent", "智能体", "rag", "prompt", "workflow")),
    ("model_training", ("大模型训练", "预训练", "后训练", "模型微调", "强化学习")),
    ("computer_vision", ("计算机视觉", "视觉算法", "图像算法", "多模态")),
    ("nlp", ("自然语言处理", "nlp", "语音算法")),
    ("recommendation", ("推荐算法", "推荐系统", "召回排序")),
    ("chip_design", ("芯片设计", "数字ic", "模拟ic", "fpga", "asic", "验证工程师")),
    ("biomedical_research", ("生物信息", "药物研发", "医药科研", "临床研究", "计算生物")),
)

_FAMILY_DOMAIN = {
    "backend": "software_engineering", "frontend": "software_engineering",
    "fullstack": "software_engineering", "mobile": "software_engineering",
    "qa": "software_engineering", "devops": "software_engineering",
    "data_engineering": "software_engineering",
    "llm_application": "ai_engineering", "model_training": "ai_engineering",
    "computer_vision": "ai_engineering", "nlp": "ai_engineering",
    "recommendation": "ai_engineering", "chip_design": "chip_hardware",
    "biomedical_research": "biomedical_research",
}


def _normalize(text: object) -> str:
    """用于短语命中的轻量归一：小写并移除空白/常见分隔符。"""
    return re.sub(r"[\s·•_\-/|（）()【】\[\]，,。.:：;；]+", "", str(text or "").lower())


def _bounded(text: object) -> str:
    return str(text or "")[:MAX_DOMAIN_TEXT_CHARS]


@lru_cache(maxsize=1)
def load_domain_templates() -> tuple[dict[str, Any], ...]:
    """读取并校验模板；失败时返回空集合，让旧匹配链路安全降级。"""
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        domains = payload.get("domains") or []
        required = {"id", "name", "title_aliases", "competency_keywords",
                    "major_keywords", "related_domains"}
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in domains:
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError("career domain template schema invalid")
            domain_id = str(item["id"]).strip()
            if not domain_id or domain_id in seen:
                raise ValueError("career domain id empty or duplicated")
            seen.add(domain_id)
            out.append(item)
        return tuple(out)
    except Exception:
        return ()


def _domain_by_id(domain_id: str | None) -> dict[str, Any] | None:
    if not domain_id:
        return None
    return next((d for d in load_domain_templates() if d["id"] == domain_id), None)


@lru_cache(maxsize=256)
def _detect_domain_bounded(bounded: str) -> str | None:
    """对已经截断的文本识别职业域；缓存避免同一 JD 在方向/专业/能力层重复扫描。"""
    norm = _normalize(bounded)
    head = _normalize(bounded[:240])
    if not norm:
        return None
    scored: list[tuple[int, str]] = []
    for domain in load_domain_templates():
        score = 0
        for alias in domain["title_aliases"]:
            key = _normalize(alias)
            if key and key in head:
                score += 8
            elif key and key in norm:
                score += 4
        for kw in domain["competency_keywords"]:
            key = _normalize(kw)
            if key and key in norm:
                score += 1
        if score:
            scored.append((score, domain["id"]))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def detect_domain(text: str) -> str | None:
    """从职位文本确定性识别职业域。

    标题/前 240 字中的职位别名权重最高，正文能力词只作弱证据；最高分并列时
    返回 ``None``，宁可信息不足也不武断分类。
    """
    return _detect_domain_bounded(_bounded(text))


def _llm_domain_hint(text: str) -> str | None:
    """让 LLM 只在封闭职业域中做“别名翻译”，不得发明岗位门槛。

    结果必须给出 JD 原文中的短证据且置信度 >= 0.8；否则丢弃。任何异常都
    返回 None，由确定性规则继续保守处理。
    """
    try:
        from day1_api_starter import get_llm_config
        from plan_gen import call_llm_plain

        cfg = get_llm_config() or {}
        api_key = cfg.get("api_key")
        if not api_key:
            return None
        allowed = [{"id": d["id"], "name": d["name"]} for d in load_domain_templates()]
        messages = [
            {"role": "system", "content":
             "你只负责把不规范岗位叫法翻译到给定职业域，不判断候选人是否合适，"
             "不新增技能或硬门槛。只输出 JSON："
             '{"domain_id":"...","confidence":0.0,"evidence":"JD原文短句"}。'},
            {"role": "user", "content":
             f"允许的职业域：{json.dumps(allowed, ensure_ascii=False)}\nJD：\n{_bounded(text)}"},
        ]
        raw = call_llm_plain(messages, api_key, max_tokens=180) or ""
        left, right = raw.find("{"), raw.rfind("}")
        if left < 0 or right < left:
            return None
        data = json.loads(raw[left:right + 1])
        domain_id = str(data.get("domain_id") or "")
        confidence = float(data.get("confidence") or 0)
        evidence = str(data.get("evidence") or "").strip()
        if confidence < 0.8 or not evidence or evidence not in text:
            return None
        return domain_id if _domain_by_id(domain_id) else None
    except Exception:
        return None


def resolve_domain(text: str, *, allow_llm: bool | None = None) -> str | None:
    """规则优先；模板未识别时才可选用 LLM 做封闭集合岗位别名归一。"""
    domain_id = detect_domain(text)
    if domain_id:
        return domain_id
    if allow_llm is None:
        allow_llm = os.environ.get("ROLE_NORMALIZE_LLM", "0") == "1"
    return _llm_domain_hint(text) if allow_llm else None


def detect_role_family(text: str) -> str | None:
    """识别职业域内部的主要工作族，避免把前端/后端/AI 统称为“技术岗”。"""
    norm = _normalize(_bounded(text))
    hits: list[tuple[int, str]] = []
    for family, phrases in _ROLE_FAMILIES:
        score = sum(3 if _normalize(p) in _normalize(_bounded(text)[:240]) else 1
                    for p in phrases if _normalize(p) and _normalize(p) in norm)
        if score:
            hits.append((score, family))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


def clear_domain_caches() -> None:
    """供模板热更新/测试隔离使用；正常进程启动后无需调用。"""
    _detect_domain_bounded.cache_clear()
    load_domain_templates.cache_clear()


def _target_domain(target: str) -> str | None:
    """目标方向通常很短，先做别名直命中，再退回通用检测。"""
    target_norm = _normalize(target)
    for domain in load_domain_templates():
        for alias in domain["title_aliases"]:
            alias_norm = _normalize(alias)
            if alias_norm and (alias_norm in target_norm or target_norm in alias_norm):
                return domain["id"]
    detected = detect_domain(target)
    if detected:
        return detected
    return _FAMILY_DOMAIN.get(detect_role_family(target) or "")


def generic_direction_alignment(profile: dict, jd: str) -> dict[str, str]:
    """旧规则未覆盖时，按用户目标方向判断 JD 是主方向/派生方向/不考虑。"""
    targets = [str(x).strip() for x in (profile.get("方向优先级") or []) if str(x).strip()]
    if not targets:
        return {"direction": "不考虑", "status": "未命中",
                "reason": "profile §2 目标方向未填，通用规则无法判断方向一致度"}

    jd_norm = _normalize(_bounded(jd))
    direct = [t for t in targets if _normalize(t) and _normalize(t) in jd_norm]
    if direct:
        return {"direction": "主方向", "status": "命中",
                "reason": f"JD 直接命中 profile §2 目标方向 {direct}"}

    jd_domain = resolve_domain(jd)
    target_pairs = [(t, _target_domain(t)) for t in targets]
    same = [t for t, domain_id in target_pairs if domain_id and domain_id == jd_domain]
    if same and jd_domain:
        domain = _domain_by_id(jd_domain)
        jd_family = detect_role_family(jd)
        target_families = [(t, detect_role_family(t)) for t in same]
        exact_family = [t for t, family in target_families if family and family == jd_family]
        conflicting_family = [t for t, family in target_families
                              if family and jd_family and family != jd_family]
        if exact_family:
            return {"direction": "主方向", "status": "命中",
                    "reason": f"JD 归入「{domain['name']}/{jd_family}」，与 profile §2 目标方向 {exact_family} 一致"}
        if conflicting_family:
            return {"direction": "派生方向", "status": "部分命中",
                    "reason": f"JD 与目标同属「{domain['name']}」，但工作族 {jd_family} 与 {conflicting_family} 不同"}
        return {"direction": "主方向", "status": "命中",
                "reason": f"JD 归入「{domain['name']}」职业域，与 profile §2 目标方向 {same} 一致"}

    related: list[str] = []
    if jd_domain:
        for target, target_domain in target_pairs:
            domain = _domain_by_id(target_domain)
            if (target_domain == "ai_engineering" and jd_domain == "software_engineering"
                    and not any(k in _normalize(jd) for k in ("ai", "大模型", "llm", "agent", "智能体", "rag"))):
                # AI 用户并未自动选择所有传统软件岗。只有 JD 本身带 AI 工作内容，
                # 或画像另有软件方向，才把软件研发当成相邻方向。
                continue
            if domain and jd_domain in (domain.get("related_domains") or []):
                related.append(target)
    if related:
        jd_name = (_domain_by_id(jd_domain) or {}).get("name", jd_domain)
        return {"direction": "派生方向", "status": "部分命中",
                "reason": f"JD 归入「{jd_name}」职业域，是 profile §2 目标方向 {related} 的相邻方向"}

    return {"direction": "不考虑", "status": "未命中",
            "reason": f"JD 与 profile §2 目标方向 {targets} 未建立可解释的直接或职业域关联"}


def major_matches_domain(user_major: str, jd: str) -> tuple[bool, str]:
    """仅为“相关专业”提供职业域模板证据，不替代原文硬门槛。"""
    domain_id = detect_domain(jd)
    domain = _domain_by_id(domain_id)
    if not domain:
        return False, ""
    hits = [m for m in domain["major_keywords"] if m and m in (user_major or "")]
    if not hits:
        return False, ""
    return True, f"{user_major} 命中「{domain['name']}」职业域相关专业模板 {hits}"


def general_competency_hits(jd: str) -> list[str]:
    """抽取非技术职业域中 JD 原文明确出现的能力词；不做同义臆测。"""
    domain_id = detect_domain(jd)
    domain = _domain_by_id(domain_id)
    if not domain or domain_id in {"software_engineering", "ai_engineering"}:
        return []
    lower = _bounded(jd).lower()
    return [kw for kw in domain["competency_keywords"] if kw.lower() in lower]
