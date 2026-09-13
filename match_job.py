# -*- coding: utf-8 -*-
"""
OfferClaw · 岗位匹配最小规则版 (match_job.py)

作用：
    V1 比赛版的最小可运行岗位匹配脚本。与 job_match_prompt.md 共用同一套
    输出契约（来自 target_rules.md §6），保证 Prompt 版与 Python 版逻辑一致。

原则：
    - 只做规则式判断，禁止玄学综合打分
    - 三层逻辑：硬门槛 → 软性维度 → 三档结论 + 缺口清单
    - 无数据库、无爬虫、无复杂评分模型、无第三方依赖

使用：
    直接运行：
        python match_job.py
    输出会打印一份完整匹配报告（DEMO_PROFILE + DEMO_JD）。
    测试别的 JD：修改本文件底部的 DEMO_JD 字符串即可。
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from jd_parser import term_in_text
from domain_status import HardGateStatusCode, SoftConditionStatusCode, requirement_status_code


# =====================================================
# 常量：与 target_rules.md §3 / §4 一一对应
# =====================================================

# 学历数值：越大越高
EDU_LEVEL = {"专科": 1, "本科": 2, "学士": 2, "硕士": 3, "博士": 4}

# OfferClaw 主方向关键词（profile §2 主方向）
MAIN_DIRECTION_KWS = [
    "agent", "llm", "大模型", "prompt", "workflow",
    "ai 应用", "智能体", "ai应用",
]

# 派生方向关键词（profile §2 派生方向）
SUB_DIRECTION_KWS = ["python 后端", "数据处理", "算法落地"]

# AI 应用友好专业（当 JD 写"相关专业"时的扩展集合）
AI_FRIENDLY_MAJORS = [
    "计算机", "软件", "通信", "电子", "信息",
    "人工智能", "数据", "自动化",
]

# 常见城市列表（用于地域识别，MVP 版手写，后续可扩）
COMMON_CITIES = [
    "北京", "上海", "深圳", "广州", "杭州",
    "成都", "南京", "武汉", "西安", "苏州", "天津",
]

# 质性经验要求关键词（JD 不给具体 N 年，但显式要求项目/实习/工程经验）
# 当 JD 出现这类关键词 且 profile §4/§7 为空时，判硬门槛 ✗
# 合成回归来源："在校期间有相关项目经验"——旧版仅匹配 "N 年" 模式会漏掉
QUALITATIVE_EXP_KEYWORDS = [
    "项目经验", "相关经验", "落地经验", "实战经验",
    "开发经验", "工程经验", "应用经验", "相关项目",
    "项目落地", "应用落地",
]

# 出现在关键词后方 30 字符内的"软化词"——会把"质性要求"降级为"加分项"而非硬要求
# 例如 "有项目经验者优先" 的 "优先" 命中软化词 → 判加分项，不触发硬门槛 ✗
SOFTEN_HINTS = ["优先", "加分", "bonus", "更佳", "为佳", "者优先"]

_AI_PROFILE_HINTS = tuple(k.lower() for k in (
    MAIN_DIRECTION_KWS + [
        "人工智能", "机器学习", "深度学习", "大模型", "算法研究", "模型训练",
        "自然语言处理", "计算机视觉", "推荐算法",
    ]
))


def _profile_uses_legacy_tech_scope(profile: dict) -> bool:
    """当前用户是否明确把 AI/大模型作为目标方向。

    “技术岗”不是“AI 岗”的同义词。后端/前端/全栈用户看到正文里提到 LLM，
    不能因此自动被归为 AI 主方向；只有画像本身明确选择 AI/大模型方向，才启用
    旧的 AI 关键词快速分支。画像未填时保留历史 fallback，避免旧调用崩溃。
    """
    targets = [str(x).strip().lower() for x in (profile.get("方向优先级") or []) if str(x).strip()]
    if not targets:
        return True
    joined = " ".join(targets)
    return any(h in joined for h in _AI_PROFILE_HINTS)


_GATE_MARKERS = {
    "education": ("学历", "本科", "学士", "硕士", "博士", "学位"),
    "major": ("专业", "学科", "方向"),
    "experience": ("经验", "经历", "项目", "实习", "年"),
    "language": ("英语", "英文", "english", "cet", "托福", "雅思", "日语", "语言"),
    "tech": tuple(),
}

_EXPLICIT_NO_REQUIREMENT = {
    "education": (
        r"(?:学历|学位)(?:要求)?\s*[:：]?\s*(?:不限|不作要求|无要求)", r"不限\s*(?:学历|学位)",
        r"不要求\s*(?:学历|学位)",
    ),
    "major": (
        r"(?:专业|学科)(?:要求)?\s*[:：]?\s*(?:不限|不作要求|无要求)", r"不限\s*(?:专业|学科)",
        r"不要求\s*(?:专业|学科)",
    ),
    "experience": (
        r"(?:经验|经历)(?:要求)?\s*[:：]?\s*(?:不限|不作要求|无要求)", r"(?:不要求|无需)\s*(?:相关|工作|项目|实习)?(?:经验|经历)",
        r"(?:没有|无)(?:相关|工作|项目|实习)?(?:经验|经历)(?:也可|亦可|可以)",
    ),
    "language": (
        r"(?:语言|英语|英文|证书)(?:要求)?\s*[:：]?\s*(?:不限|不作要求|无要求)",
        r"(?:不要求|无需)\s*(?:英语|英文|语言|(?:英语|语言)?证书|cet)",
    ),
}


def _has_explicit_no_requirement(text: str, gate: str) -> bool:
    return any(re.search(pattern, text or "", re.I)
               for pattern in _EXPLICIT_NO_REQUIREMENT.get(gate, ()))


def _analysis_gate_text(jd: str, jd_analysis: dict | None, gate: str) -> str:
    """Return only evidence that may legally drive one hard gate.

    An LLM may label and group requirements, but preferred/alternative/context
    wording must never silently become a hard rejection.  Explicitly negative
    wording is retained because it proves that a gate is *not* required.
    """
    if not jd_analysis:
        return jd
    pieces: list[str] = []
    markers = _GATE_MARKERS.get(gate, ())
    for item in jd_analysis.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        evidence = "\n".join(
            str(span.get("text") or "") for span in item.get("evidence_spans") or []
            if isinstance(span, dict)
        ).strip() or str(item.get("text") or "").strip()
        if not evidence:
            continue
        lower = evidence.lower()
        relevant = gate == "tech" or any(marker.lower() in lower for marker in markers)
        if not relevant:
            continue
        modality = str(item.get("modality") or "unknown")
        if _has_explicit_no_requirement(evidence, gate):
            pieces.append(evidence)
        elif modality == "required":
            pieces.append(evidence)
    if gate == "location":
        location = str(jd_analysis.get("location") or "").strip()
        if location:
            pieces.append(location)
        # Remote support is a structured work-mode fact rather than a skill
        # requirement, so preserve only explicit, non-negated source wording.
        for match in re.finditer(r"[^。；;\n]{0,24}(?:全程远程|支持远程|可远程|远程办公|远程实习|工作地点\s*[:：]\s*远程|remote)[^。；;\n]{0,24}", jd, re.I):
            value = match.group(0).strip()
            if not re.search(r"(?:不|无法|不可|禁止)\s*(?:支持)?远程|no\s+remote", value, re.I):
                pieces.append(value)
    elif gate == "tech":
        title = str(jd_analysis.get("title") or "").strip()
        if title:
            pieces.append(title)
    return "\n".join(dict.fromkeys(piece for piece in pieces if piece))


# =====================================================
# 数据结构
# =====================================================

@dataclass
class CheckResult:
    """单项检查结果。
    status 取值：
      - 硬门槛：'✓' / '✗' / '?'
      - 软性维度：'命中' / '部分命中' / '未命中' / '?'
    reason 必须引用 profile 章节或说明"信息不足"。
    """
    name: str
    status: str
    reason: str
    status_code: str = ""

    def __post_init__(self) -> None:
        code = requirement_status_code(self.status)
        if self.status_code and self.status_code != code:
            raise ValueError("status 与 status_code 冲突")
        self.status_code = code


@dataclass
class MatchReport:
    jd_title: str
    direction: str
    hard_gate: List[CheckResult] = field(default_factory=list)
    soft_dims: List[CheckResult] = field(default_factory=list)
    conclusion: str = ""
    conclusion_reason: str = ""
    gap_list: Dict[str, List[str]] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    requirement_analysis: Dict[str, object] = field(default_factory=dict)


# =====================================================
# 硬门槛检查（6 项；编程语言兼容性单列为软性维度，避免多语言 JD 错杀）
# =====================================================

def check_education(profile: dict, jd: str) -> CheckResult:
    """学历：JD 要求 vs profile §1 学历层次。"""
    user_edu = profile.get("学历")
    user_level = EDU_LEVEL.get(user_edu, 0)

    if _has_explicit_no_requirement(jd, "education"):
        return CheckResult("学历", "✓", "JD 明确不限学历")

    # 取 JD 中最低可接受学历（最宽松的那一档）
    if "本科" in jd or "学士" in jd:
        required, req_name = 2, "本科"
    elif "硕士" in jd:
        required, req_name = 3, "硕士"
    elif "博士" in jd:
        required, req_name = 4, "博士"
    else:
        return CheckResult("学历", "?", "JD 未明确学历要求")

    if user_level == 0:
        return CheckResult("学历", "?", "profile §1 学历层次未填")
    if user_level >= required:
        return CheckResult(
            "学历", "✓",
            f"用户 {user_edu} ≥ JD 要求 {req_name}（profile §1）"
        )
    return CheckResult(
        "学历", "✗",
        f"用户 {user_edu} 低于 JD 要求 {req_name}（profile §1）"
    )


def check_major(profile: dict, jd: str) -> CheckResult:
    """专业：JD 列出的专业范围 vs profile §1 专业。"""
    user_major = profile.get("专业") or ""
    if not user_major:
        return CheckResult("专业", "?", "profile §1 专业未填")

    if _has_explicit_no_requirement(jd, "major"):
        return CheckResult("专业", "✓", "JD 明确不限专业")

    # 用户专业字面量直接出现
    if user_major in jd:
        return CheckResult(
            "专业", "✓",
            f"JD 中直接提及 {user_major}（profile §1）"
        )

    # “计算机科学”应能满足“计算机/软件”等明确枚举；这里仍要求
    # 用户专业与 JD 原文共享受控专业词，不做跨职业域推断。
    shared_major_terms = [
        term for term in AI_FRIENDLY_MAJORS
        if len(term) >= 2 and term in user_major and term in jd
    ]
    if shared_major_terms:
        return CheckResult(
            "专业", "✓",
            f"用户专业与 JD 明确专业范围共享 {shared_major_terms}（profile §1）"
        )

    # JD 说"相关专业"，且用户专业属于 AI 友好专业集合
    if "相关专业" in jd or "相关方向" in jd:
        domain_id = None
        try:
            from career_domains import resolve_domain
            domain_id = resolve_domain(jd, allow_llm=False)
        except Exception:
            pass
        # AI 友好专业白名单只服务 AI/软件或尚未识别的历史技术 JD；不能拿它
        # 让通信专业自动通过医药科研等完全不同职业域。
        if domain_id in {None, "ai_engineering", "software_engineering"}:
            for m in AI_FRIENDLY_MAJORS:
                if m in user_major:
                    label = "AI 友好相关专业" if domain_id in {None, "ai_engineering"} else "软件研发相关专业"
                    return CheckResult(
                        "专业", "✓",
                        f"{user_major} 属于 {label}（profile §1 + JD '相关专业'）"
                    )
        # 兼容扩展：旧 AI 白名单未覆盖时，才使用通用职业域模板解释
        # “相关专业”。模板只提供可解释的专业关联，不自行发明 JD 门槛。
        try:
            from career_domains import major_matches_domain
            matched, reason = major_matches_domain(user_major, jd)
            if matched:
                return CheckResult("专业", "✓", f"{reason}（profile §1 + JD '相关专业'）")
        except Exception:
            pass

    # JD 没明确专业要求 → 默认 AI 友好专业通过（避免对所有 AI/技术岗都报"信息不足"）
    if not any(tag in jd for tag in ("专业", "Major", "学科")):
        for m in AI_FRIENDLY_MAJORS:
            if m in user_major:
                return CheckResult(
                    "专业", "✓",
                    f"{user_major} 属于 AI 友好专业（JD 未限制专业，按相关专业默认通过）"
                )
        # JD 没有提出专业要求时，非 AI 专业也不应被固定白名单变相限制。
        # 放在旧分支之后，保证现有技术 Persona 的解释文本逐字不变。
        return CheckResult(
            "专业", "✓",
            f"JD 未提出专业限制，用户专业 {user_major} 不构成硬门槛"
        )

    return CheckResult(
        "专业", "?",
        f"JD 未明确是否接受 {user_major}"
    )


def _scan_qualitative_exp(jd: str) -> Optional[str]:
    """扫描 JD 是否存在'真正的硬门槛型'质性经验要求。

    判定规则：
    - 关键词命中 QUALITATIVE_EXP_KEYWORDS 列表
    - 且关键词后方 30 字符窗口内不包含 SOFTEN_HINTS（优先/加分等）
    - 同时满足以上两条 → 视为硬门槛

    返回：命中的关键词字符串；未命中返回 None。
    """
    for kw in QUALITATIVE_EXP_KEYWORDS:
        idx = jd.find(kw)
        if idx == -1:
            continue
        window = jd[idx: idx + len(kw) + 30]
        if any(s in window for s in SOFTEN_HINTS):
            continue  # 命中软化词 → 判为加分项，不算硬门槛
        return kw
    return None


def check_experience(profile: dict, jd: str) -> CheckResult:
    """经验要求：JD 要求 vs profile §4 / §7。

    判断优先级（从高到低）：
      1. 实习岗 + 显式宽松词（不强制 / 不要求） → ✓
      2. 质性经验硬要求（如"相关项目经验"且非加分项）+ profile §4/§7 为空 → ✗
      3. "N 年" 年限匹配 → 按年限与 profile 对比判定
      4. 无年限 + 实习岗 → ✓（宽松默认）
      5. 其他 → ?

    V1 合成回归暴露旧版仅匹配"N 年"模式的盲区，
    导致"在校期间有相关项目经验"被误判为 ✓。本版本通过分支 (2) 修复。
    """
    user_proj = profile.get("项目数量", 0) or 0
    user_intern = profile.get("实习数量", 0) or 0
    user_exp = user_proj + user_intern

    if _has_explicit_no_requirement(jd, "experience"):
        return CheckResult("经验", "✓", "JD 明确不要求相关经验")

    # (1) 实习岗 + 显式宽松词 → 直接 ✓
    if "实习" in jd and ("不强制" in jd or "不要求" in jd):
        return CheckResult("经验", "✓", "JD 明确实习不强制经验要求")

    # (2) 质性经验硬要求（JD 写"相关项目经验"等且非加分项）
    qual_hit = _scan_qualitative_exp(jd)
    if qual_hit and user_exp == 0:
        return CheckResult(
            "经验", "✗",
            f"JD 任职要求显式包含'{qual_hit}'（非加分项），"
            f"profile §4/§7 均为空，无可证明的相关经验"
        )

    # (3) "N 年" 年限匹配
    m = re.search(r"(\d+)\s*年", jd)
    if m:
        required = int(m.group(1))
        if required == 0:
            return CheckResult("经验", "✓", "JD 显式无经验年限要求")
        if user_exp == 0:
            return CheckResult(
                "经验", "✗",
                f"JD 要求 {required} 年经验，profile §4/§7 暂无可证明经验"
            )
        return CheckResult(
            "经验", "?",
            f"JD 要求 {required} 年，用户实际年限需人工核对"
        )

    # (4) 无显式年限 + 实习岗 → 宽松默认 ✓
    if "实习" in jd:
        return CheckResult(
            "经验", "✓",
            "实习岗默认无硬性年限要求（JD 未列质性经验硬要求）"
        )

    # (5) 其他情况
    return CheckResult("经验", "?", "JD 未明确经验要求")


def check_language(profile: dict, jd: str) -> CheckResult:
    """语言：JD 要求 vs profile §9 英语读写 自评。"""
    if _has_explicit_no_requirement(jd, "language"):
        return CheckResult("语言", "✓", "JD 明确不设置语言/证书硬门槛")
    english_req = any(k in jd for k in [
        "英语", "english", "cet", "托福", "雅思"
    ])
    if not english_req:
        return CheckResult("语言", "✓", "JD 无特殊语言要求")

    user_en = profile.get("英语自评")
    if user_en is None:
        return CheckResult("语言", "?", "profile §9 英语读写 自评未填")

    try:
        lvl = int(user_en)
    except (TypeError, ValueError):
        return CheckResult("语言", "?", "profile §9 英语自评格式异常")

    if lvl >= 3:
        return CheckResult(
            "语言", "✓",
            f"英语自评 {lvl}/5 可覆盖 JD 的阅读/沟通要求（profile §9）"
        )
    return CheckResult(
        "语言", "✗",
        f"英语自评 {lvl}/5 不足以覆盖 JD 要求（profile §9）"
    )


def check_location(profile: dict, jd: str) -> CheckResult:
    """地域：JD 所在地 vs profile §1 所在地 / 可接受工作地域。"""
    user_areas = profile.get("可接受地域") or []
    # 固定城市表只用于画像为空时兜底；优先把用户自己的可接受地域加入扫描，
    # 否则无锡/南通等真实偏好会因不在手写热门城市表里被误判“JD 未说明地点”。
    city_candidates = list(dict.fromkeys(
        COMMON_CITIES + [str(x) for x in user_areas if str(x) and str(x) != "远程"]
    ))
    supports_remote = bool(re.search(r"(?:远程|remote)", jd, re.I))
    rejects_remote = bool(re.search(r"(?:不|无法|不可|禁止)\s*(?:支持)?远程|no\s+remote", jd, re.I))
    accepts_remote = any(str(area).strip().lower() in {"远程", "remote"} for area in user_areas)
    if supports_remote and not rejects_remote and accepts_remote:
        return CheckResult("地域", "✓", "JD 明确支持远程")

    jd_cities = [c for c in city_candidates if c in jd]

    if not jd_cities:
        return CheckResult("地域", "?", "JD 未明确工作地点")

    if not user_areas:
        return CheckResult(
            "地域", "?",
            f"profile §1 可接受工作地域未填（JD 地点：{jd_cities}）"
        )

    for c in jd_cities:
        if c in user_areas:
            return CheckResult(
                "地域", "✓",
                f"JD 地点 {c} 在用户可接受地域内（profile §1）"
            )
    return CheckResult(
        "地域", "✗",
        f"JD 地点 {jd_cities} 不在用户可接受地域 {user_areas}（profile §1）"
    )


def check_tech_mainline(profile: dict, jd: str) -> CheckResult:
    """技术主线：JD 是否强制要求用户明确不做的技术栈。"""
    banned = profile.get("明确不做") or []
    if not banned:
        if "明确不做" in profile:
            return CheckResult("技术主线", "✓", "profile §2 未设置排除方向，不存在已知技术主线冲突")
        return CheckResult("技术主线", "?", "profile §2 明确不做方向字段缺失")

    jd_lower = jd.lower()
    try:
        from job_requirements import analyze_programming_languages, normalize_programming_language
        language_req = analyze_programming_languages(jd)
        banned_languages = {
            lang for lang in (normalize_programming_language(x) for x in banned) if lang
        }
        required = set(language_req.hard_required)
        if language_req.policy == "single_required" and required & banned_languages:
            hit = sorted(required & banned_languages)
            return CheckResult(
                "技术主线", "✗",
                f"JD 将 {hit} 作为单一/主要编程语言，与 profile §2 明确不做方向冲突；"
                f"证据：{language_req.evidence[0]}"
            )
        if language_req.policy == "multi_required_explicit" and required & banned_languages:
            hit = sorted(required & banned_languages)
            return CheckResult(
                "技术主线", "✗",
                f"JD 明确要求同时掌握 {list(language_req.hard_required)}，其中 {hit} 与 profile §2 冲突"
            )
        # flexible_pool 即使含被排除语言，也不能判硬冲突：用户可用池中其他语言满足。
    except Exception:
        pass
    for b in banned:
        # 取关键词的头部（例：'java 后端' → 'java'）
        key = b.split()[0].lower() if b else ""
        if not key:
            continue
        if key not in jd_lower:
            continue
        # 只有当 JD 把它列为强要求时才算未命中
        strong_hints = [
            f"精通 {key}", f"熟练 {key}", f"{key} 为主",
            f"{key} 开发", f"{key}开发", f"资深 {key}",
        ]
        if any(h in jd_lower for h in strong_hints):
            return CheckResult(
                "技术主线", "✗",
                f"JD 强制 {b} 主线，与 profile §2 明确不做方向冲突"
            )
    return CheckResult(
        "技术主线", "✓",
        "JD 未强制要求用户排除的技术主线"
    )


def soft_programming_language(profile: dict, jd: str) -> CheckResult:
    """编程语言兼容性：多语言列表默认命中任一即可，缺失时避免直接错杀。"""
    try:
        from job_requirements import analyze_programming_languages, profile_programming_languages
        req = analyze_programming_languages(jd)
        owned = profile_programming_languages(profile)
    except Exception:
        return CheckResult("编程语言兼容", "部分命中", "语言要求解析失败，需人工核对")

    if req.policy == "none":
        return CheckResult("编程语言兼容", "部分命中", "JD 未明确编程语言要求")
    if req.policy == "context_only":
        return CheckResult(
            "编程语言兼容", "部分命中",
            f"JD 仅描述团队技术栈 {list(req.languages)}，不作为候选人硬门槛"
        )
    hits = sorted(owned & set(req.languages))
    if req.policy == "flexible_pool":
        if hits:
            return CheckResult(
                "编程语言兼容", "命中",
                f"JD 的多语言候选池为 {list(req.languages)}，用户命中 {hits}；不要求全部掌握"
            )
        return CheckResult(
            "编程语言兼容", "部分命中",
            f"用户未直接命中 JD 候选语言池 {list(req.languages)}；原文未证明全部必需，作为可迁移差距而非投递否决"
        )
    if req.policy == "preferred":
        status = "命中" if hits else "部分命中"
        return CheckResult(
            "编程语言兼容", status,
            f"JD 将 {list(req.languages)} 作为优先/加分项；用户命中 {hits or '无'}"
        )
    required = set(req.hard_required)
    required_hits = sorted(owned & required)
    if req.policy == "single_required":
        if required_hits:
            return CheckResult("编程语言兼容", "命中", f"用户具备 JD 重点语言 {required_hits}")
        return CheckResult(
            "编程语言兼容", "未命中",
            f"JD 仅列或重点要求 {list(required)}，用户画像未记录；需结合框架生态和原文人工确认"
        )
    if required.issubset(owned):
        return CheckResult("编程语言兼容", "命中", f"用户满足 JD 明确的多语言要求 {sorted(required)}")
    if required_hits:
        return CheckResult(
            "编程语言兼容", "部分命中",
            f"JD 明确要求 {sorted(required)}，用户仅命中 {required_hits}"
        )
    return CheckResult(
        "编程语言兼容", "未命中",
        f"JD 明确要求同时掌握 {sorted(required)}，用户画像均未记录"
    )


# =====================================================
# 软性维度分析（6 项）
# =====================================================

def _semantic_dimension(
    jd_analysis: dict | None,
    semantic_alignment: dict | None,
    *,
    kinds: set[str],
    evidence_types: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Reduce validated relations to one soft-dimension status.

    This is categorical on purpose. A preferred item cannot compensate for an
    unsupported required item, and no hand-tuned relation weights are needed.
    """
    if not semantic_alignment or semantic_alignment.get("status") != "completed":
        return "unavailable", [], {}
    requirements = [
        item for item in (jd_analysis or {}).get("requirements") or []
        if isinstance(item, dict)
        and str(item.get("kind") or "") in kinds
        and str(item.get("modality") or "unknown") != "context"
    ]
    if not requirements:
        return "unavailable", [], {}
    rows = {
        str(item.get("requirement_id") or ""): item
        for item in semantic_alignment.get("alignments") or []
        if isinstance(item, dict) and item.get("requirement_id")
    }
    mandatory = [item for item in requirements
                 if str(item.get("modality") or "") in {"required", "alternative"}]
    strong_ids: set[str] = set()
    partial_ids: set[str] = set()
    hits: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        row = rows.get(requirement_id) or {}
        row_types = {str(value) for value in row.get("evidence_types") or []}
        if evidence_types is not None and not (row_types & evidence_types):
            continue
        relation = str(row.get("relation") or "")
        if relation in {"direct", "transferable"}:
            strong_ids.add(requirement_id)
            hits.append(row)
        elif relation == "partial":
            partial_ids.add(requirement_id)
            hits.append(row)
    mandatory_ids = {str(item.get("requirement_id") or "") for item in mandatory}
    positive_ids = strong_ids | partial_ids
    if mandatory_ids:
        status = "met" if mandatory_ids.issubset(strong_ids) else (
            "partial" if positive_ids else "unmet"
        )
    else:
        all_ids = {str(item.get("requirement_id") or "") for item in requirements}
        status = "met" if all_ids and all_ids.issubset(strong_ids) else (
            "partial" if positive_ids else "unmet"
        )
    stats = {
        "total": len(requirements),
        "mandatory": len(mandatory_ids),
        "strong": len(strong_ids),
        "partial": len(partial_ids),
        "unsupported": len(requirements) - len(positive_ids),
        "mandatory_strong": len(mandatory_ids & strong_ids),
    }
    return status, hits, stats


def _semantic_reason(prefix: str, stats: dict[str, int],
                     hits: list[dict[str, Any]]) -> str:
    examples = []
    for row in hits[:3]:
        requirement = str(row.get("requirement_text") or "")[:36]
        sources = ", ".join(str(value) for value in row.get("evidence_sources") or [])
        examples.append(f"“{requirement}” ← {sources or '画像证据'}")
    detail = "；".join(examples)
    counts = (
        f"必选 {stats.get('mandatory', 0)} 项中直接/可迁移证据 "
        f"{stats.get('mandatory_strong', 0)} 项；全部要求另有部分证据 "
        f"{stats.get('partial', 0)} 项、无证据 {stats.get('unsupported', 0)} 项"
    )
    return f"{prefix}{counts}" + (f"：{detail}" if detail else "")


def soft_skill_overlap(profile: dict, jd: str,
                       jd_analysis: dict | None = None,
                       semantic_alignment: dict | None = None) -> CheckResult:
    """技能重叠：JD 要求技能 vs profile §3 技能清单。"""
    declared = (profile.get("熟练技能") or []) + (profile.get("会用技能") or [])
    evidence = profile.get("技能证据") or []
    skills = list(dict.fromkeys(declared + evidence))
    if not skills and not (
        semantic_alignment and semantic_alignment.get("status") == "completed"
    ):
        return CheckResult("技能重叠", "部分命中", "profile §3/§4 技能证据为空，暂不能判断重叠度")

    keyword_items = list((jd_analysis or {}).get("keywords") or [])
    if keyword_items:
        weighted_hits: dict[str, float] = {}
        total_weight = 0.0
        requirement_by_id = {
            str(item.get("requirement_id") or ""): item
            for item in (jd_analysis or {}).get("requirements") or []
            if isinstance(item, dict) and item.get("requirement_id")
        }
        for item in keyword_items:
            if not isinstance(item, dict):
                continue
            requirement_ids = [str(value) for value in item.get("requirement_ids") or []]
            related = [requirement_by_id[value] for value in requirement_ids
                       if value in requirement_by_id]
            # Education/location/experience metadata are valid JD facts but
            # not skills. Keep unlinked legacy terms only when explicitly
            # classified as technology.
            if related and not any(
                str(value.get("kind") or "") in {"skill", "nice_to_have"}
                for value in related
            ):
                continue
            if not related and str(item.get("category") or "") != "technology":
                continue
            canonical = str(item.get("canonical_name") or "").strip()
            forms = [str(value).strip() for value in (item.get("surface_forms") or [])]
            try:
                importance = max(0.0, min(1.0, float(item.get("importance") or 0.0)))
            except (TypeError, ValueError):
                importance = 0.0
            candidates = list(dict.fromkeys(value for value in [canonical, *forms] if value))
            if not candidates:
                continue
            total_weight += importance
            for skill in skills:
                normalized = str(skill).strip()
                if normalized and any(
                    term_in_text(candidate, normalized) or term_in_text(normalized, candidate)
                    for candidate in candidates
                ):
                    weighted_hits[str(skill)] = max(
                        weighted_hits.get(str(skill), 0.0), importance,
                    )
                    break
        hit = sorted(weighted_hits, key=lambda value: weighted_hits[value], reverse=True)
        hit_weight = sum(weighted_hits.values())
        coverage = hit_weight / total_weight if total_weight else 0.0
    else:
        hit = [s for s in skills if term_in_text(jd, str(s))]
        hit_weight = float(len(hit))
        total_weight = float(max(1, len(skills)))
        coverage = hit_weight / total_weight
    semantic_status, semantic_hits, semantic_stats = _semantic_dimension(
        jd_analysis, semantic_alignment,
        kinds={"skill", "nice_to_have", "responsibility"},
    )
    semantic_completed = semantic_status != "unavailable"
    if semantic_status == "met":
        return CheckResult(
            "技能重叠", "命中",
            _semantic_reason("经要求-证据对齐，", semantic_stats, semantic_hits),
        )
    if hit and coverage >= 0.6 and hit_weight >= 0.5:
        return CheckResult(
            "技能重叠", "命中",
            f"JD 命中 {hit}（重要度加权覆盖 {coverage:.0%}；profile §3/§4）"
        )
    if semantic_status == "partial":
        return CheckResult(
            "技能重叠", "部分命中",
            _semantic_reason("经要求-证据对齐，", semantic_stats, semantic_hits),
        )
    if hit:
        return CheckResult(
            "技能重叠", "部分命中",
            f"JD 命中 {hit}，但重要度加权覆盖仅 {coverage:.0%}（profile §3）"
        )
    if semantic_completed:
        return CheckResult(
            "技能重叠", "未命中",
            "语义对齐未找到可引用的画像证据；不是仅因关键词写法不同而判定",
        )
    return CheckResult("技能重叠", "未命中", "profile §3 中的技能在 JD 中均未出现")


def soft_direction(profile: dict, jd: str) -> CheckResult:
    """方向一致度：JD 所属方向 vs profile §2 目标方向。"""
    jd_lower = jd.lower()
    main_hit = [k for k in MAIN_DIRECTION_KWS if k in jd_lower]
    legacy_scope = _profile_uses_legacy_tech_scope(profile)
    # 先按画像目标做职业域/工作族匹配。JD 正文出现 LLM 只是岗位属性，
    # 不能覆盖用户实际选择的前端、后端、芯片、医药等方向。
    try:
        from career_domains import generic_direction_alignment
        result = generic_direction_alignment(profile, jd)
        if result["direction"] != "不考虑" or not legacy_scope:
            return CheckResult("方向一致度", result["status"], result["reason"])
    except Exception:
        pass
    # 模板无法识别时，才允许“画像明确选择 AI + JD AI 关键词”兜底。
    if main_hit and legacy_scope:
        return CheckResult(
            "方向一致度", "命中",
            f"JD 出现 AI 主方向关键词 {main_hit}，且 profile §2 明确选择 AI 方向"
        )
    sub_hit = [k for k in SUB_DIRECTION_KWS if k in jd_lower]
    if sub_hit and legacy_scope:
        return CheckResult(
            "方向一致度", "部分命中",
            f"JD 出现派生方向关键词 {sub_hit}（profile §2 AI 方向的相邻工作）"
        )
    return CheckResult("方向一致度", "未命中", "JD 与 profile §2 求职方向不一致")


def soft_project_fit(profile: dict, jd: str,
                     jd_analysis: dict | None = None,
                     semantic_alignment: dict | None = None) -> CheckResult:
    """项目契合：profile §4 项目经历 / §7 实习。
    契约：只允许 '命中 / 部分命中 / 未命中' 三值（target_rules.md §4）。
    """
    has_proj = (profile.get("项目数量", 0) or 0) > 0
    has_intern = (profile.get("实习数量", 0) or 0) > 0
    if not has_proj and not has_intern:
        return CheckResult(
            "项目契合", "未命中",
            "profile §4/§7 当前无项目或实习可对标"
        )
    project_skills = profile.get("项目技能") or []
    lexical_project_hits: list[str] = []
    if project_skills:
        jd_lower = jd.lower()
        hits = [skill for skill in project_skills if str(skill).lower() in jd_lower]
        if len(hits) >= 2:
            return CheckResult(
                "项目契合", "命中",
                f"JD 命中项目经历中的能力证据 {hits}（profile §4）"
            )
        lexical_project_hits = hits
    semantic_status, semantic_hits, semantic_stats = _semantic_dimension(
        jd_analysis, semantic_alignment,
        kinds={"skill", "experience", "responsibility", "nice_to_have"},
        evidence_types={"project", "work", "research", "competition"},
    )
    semantic_completed = semantic_status != "unavailable"
    if semantic_status == "met":
        return CheckResult(
            "项目契合", "命中",
            _semantic_reason("项目/经历与 JD 要求对齐，", semantic_stats, semantic_hits),
        )
    if semantic_status == "partial":
        return CheckResult(
            "项目契合", "部分命中",
            _semantic_reason("项目/经历仅覆盖部分 JD 要求，", semantic_stats, semantic_hits),
        )
    if semantic_completed:
        return CheckResult(
            "项目契合", "未命中",
            "已有项目或经历，但语义对齐未找到能支撑当前 JD 要求的可引用证据",
        )
    if lexical_project_hits:
        return CheckResult(
            "项目契合", "部分命中",
            f"JD 命中 1 项项目能力证据 {lexical_project_hits}（profile §4）",
        )
    # 有项目或实习但 MVP 不做语义契合度判断，保守给部分命中
    return CheckResult(
        "项目契合", "部分命中",
        "profile §4/§7 存在经历条目，MVP 版暂不做语义匹配，保守估计为部分命中"
    )


def soft_work_mode(profile: dict, jd: str) -> CheckResult:
    """工作性质：JD vs profile §2 工作性质偏好。
    契约：只允许 '命中 / 部分命中 / 未命中' 三值。
    """
    pref = profile.get("工作性质偏好")
    target_type = str(profile.get("目标岗位类型") or "").strip()
    if target_type and "不限" not in target_type:
        desired_tokens = [x for x in ("日常实习", "暑期实习", "实习", "校招", "全职", "社招")
                          if x in target_type]
        if desired_tokens and any(x in jd for x in desired_tokens):
            return CheckResult(
                "工作性质", "命中",
                f"JD 工作类型命中 profile §2 目标岗位类型 {target_type}"
            )
        if desired_tokens and any(x in target_type for x in ("优先", "倾向")):
            return CheckResult(
                "工作性质", "部分命中",
                f"JD 未直接命中优先岗位类型 {target_type}，但该表述不是排他条件"
            )
        if desired_tokens:
            return CheckResult(
                "工作性质", "未命中",
                f"JD 与 profile §2 目标岗位类型 {target_type} 不一致"
            )
    if not pref:
        return CheckResult(
            "工作性质", "部分命中",
            "profile §2 工作性质偏好未填，保守估计为部分命中"
        )
    # 用户明确'不限'时直接命中
    if "不限" in str(pref):
        return CheckResult(
            "工作性质", "命中",
            "profile §2 工作性质偏好为'不限'，JD 任意工作性质均可接受"
        )
    if str(pref) in jd:
        return CheckResult(
            "工作性质", "命中",
            f"JD 符合偏好：{pref}（profile §2）"
        )
    return CheckResult(
        "工作性质", "未命中",
        f"JD 与偏好 {pref} 不一致（profile §2）"
    )


def soft_industry_preference(profile: dict, jd: str) -> CheckResult:
    """行业偏好只作软排序；除非用户另列“明确不做”，否则不能否决岗位。"""
    pref = str(profile.get("行业偏好") or "").strip()
    if not pref or "不限" in pref:
        return CheckResult("行业偏好", "部分命中", "profile §2 行业偏好未限定")
    tokens = [x.strip().lower() for x in re.split(r"[/、，,;；\s]+", pref) if len(x.strip()) >= 2]
    jd_lower = jd.lower()
    hits = [x for x in tokens if x in jd_lower]
    # “AI大模型相关”等复合表述需要拆出稳定概念，不要求 JD 逐字复现整句。
    if "ai" in pref.lower() or "大模型" in pref:
        if any(x in jd_lower for x in ("ai", "大模型", "llm", "agent", "人工智能")):
            hits.append("AI/大模型")
    if hits:
        return CheckResult("行业偏好", "命中", f"JD 命中 profile §2 行业偏好 {sorted(set(hits))}")
    return CheckResult(
        "行业偏好", "部分命中",
        f"JD 未直接体现行业偏好“{pref}”；偏好不是硬门槛，不据此否决投递"
    )


def soft_salary(profile: dict, jd: str) -> CheckResult:
    """薪资：MVP 版不做区间解析。
    契约：只允许 '命中 / 部分命中 / 未命中' 三值。
    MVP 版对薪资一律保守给部分命中，理由中区分'未填'与'未解析'两种情况。
    """
    if not profile.get("期望薪资"):
        return CheckResult(
            "薪资", "部分命中",
            "profile §2 期望薪资未填，MVP 版无法判断匹配度，保守估计为部分命中"
        )
    return CheckResult(
        "薪资", "部分命中",
        "MVP 版暂不做薪资区间解析，需人工核对，保守估计为部分命中"
    )


def soft_location(profile: dict, jd: str) -> CheckResult:
    """地域匹配：复用硬门槛地域检查的结论，映射到软性三值契约。
    硬门槛的 '?' 在软性层映射为 '部分命中'（保守策略）。
    """
    r = check_location(profile, jd)
    status_map = {
        HardGateStatusCode.MET.value: "命中",
        HardGateStatusCode.UNMET.value: "未命中",
        HardGateStatusCode.UNKNOWN.value: "部分命中",
    }
    return CheckResult(
        "地域匹配",
        status_map.get(r.status_code, "部分命中"),
        r.reason,
    )


# =====================================================
# 三档结论与缺口清单
# =====================================================

def decide(
    hard: List[CheckResult],
    soft: List[CheckResult],
    *,
    direction: str = "",
    has_target_direction: bool = True,
) -> tuple:
    """返回 (conclusion, one_line_reason)。
    规则（target_rules.md §5）：
      - 任一硬门槛 ✗ → 当前暂不建议投递
      - 硬门槛 ? ≤ 1 且软性命中 ≥ 3 → 当前适合投递
      - 软性未命中 ≥ 3 → 当前暂不建议投递
      - 其他 → 中长期可转向
    """
    hard_fail = [r for r in hard if r.status_code == HardGateStatusCode.UNMET.value]
    hard_unknown = sum(1 for r in hard if r.status_code == HardGateStatusCode.UNKNOWN.value)
    if hard_fail:
        return (
            "当前暂不建议投递",
            f"硬门槛存在 {len(hard_fail)} 项未命中："
            + "；".join(r.name for r in hard_fail),
        )

    if has_target_direction and direction == "不考虑":
        return (
            "当前暂不建议投递",
            "JD 与用户明确的目标方向未建立直接或相邻关系；不是能力不足，而是求职诉求不一致",
        )

    # 新增的语言/行业字段先作为解释与风险提示，不给旧阈值“凑命中数”。
    # 否则仅命中 Python 或 AI 字样就可能把相邻岗位错误升级成适合投递。
    decision_soft = [r for r in soft if r.name not in {"编程语言兼容", "行业偏好"}]
    soft_hit = sum(1 for r in decision_soft if r.status_code == SoftConditionStatusCode.MET.value)
    soft_fail = sum(1 for r in decision_soft if r.status_code == SoftConditionStatusCode.UNMET.value)

    language_fail = any(
        r.name == "编程语言兼容" and r.status_code == SoftConditionStatusCode.UNMET.value for r in soft
    )
    if direction == "派生方向" or language_fail:
        reason = (
            "岗位属于用户目标的相邻方向"
            if direction == "派生方向"
            else "JD 的重点编程语言尚未在画像中得到证明"
        )
        return (
            "中长期可转向",
            f"{reason}；保留投递机会，但不直接判为当前高度匹配",
        )

    explicit_experience_unknown = any(
        r.name == "经验"
        and r.status_code == HardGateStatusCode.UNKNOWN.value
        and "JD 要求" in r.reason
        for r in hard
    )
    if explicit_experience_unknown:
        return (
            "中长期可转向",
            "JD 有明确经验年限，但画像缺少可核验年限；补齐证据前不判为当前适合投递",
        )

    if hard_unknown <= 1 and soft_hit >= 3:
        return (
            "当前适合投递",
            f"硬门槛无未命中（信息不足 {hard_unknown} 项），"
            f"软性维度命中 {soft_hit} 项",
        )

    if soft_fail >= 3:
        return (
            "当前暂不建议投递",
            f"软性维度未命中达 {soft_fail} 项，差距较大",
        )

    return (
        "中长期可转向",
        f"硬门槛信息不足 {hard_unknown} 项 / 软性命中 {soft_hit} 项，"
        "建议补齐画像或积累项目后再评估",
    )


def build_gap_list(
    hard: List[CheckResult],
    soft: List[CheckResult],
    profile: dict,
) -> Dict[str, List[str]]:
    gaps = {
        "硬门槛缺口": [],
        "技能缺口": [],
        "经历缺口": [],
    }
    for r in hard:
        if r.status_code == HardGateStatusCode.UNMET.value:
            gaps["硬门槛缺口"].append(f"{r.name}：{r.reason}")
        elif r.status_code == HardGateStatusCode.UNKNOWN.value:
            gaps["硬门槛缺口"].append(f"[信息不足] {r.name}：{r.reason}")

    for r in soft:
        if r.status_code in {SoftConditionStatusCode.UNMET.value,
                             SoftConditionStatusCode.PARTIAL.value}:
            if r.name in {"技能重叠", "编程语言兼容"}:
                gaps["技能缺口"].append(r.reason)
            elif r.name == "项目契合":
                gaps["经历缺口"].append(r.reason)

    # 画像级硬事实：§4 项目为空直接进经历缺口
    if (profile.get("项目数量", 0) or 0) == 0:
        gaps["经历缺口"].append(
            "profile §4 项目经历为空，需尽快形成 1-2 个可投递项目"
        )
    return gaps


def judge_direction(profile: dict, jd: str) -> str:
    jd_lower = jd.lower()
    legacy_scope = _profile_uses_legacy_tech_scope(profile)
    try:
        from career_domains import generic_direction_alignment
        result = generic_direction_alignment(profile, jd)["direction"]
        if result != "不考虑" or not legacy_scope:
            return result
    except Exception:
        pass
    if legacy_scope and any(k in jd_lower for k in MAIN_DIRECTION_KWS):
        return "主方向"
    if legacy_scope and any(k in jd_lower for k in SUB_DIRECTION_KWS):
        return "派生方向"
    return "不考虑"


def build_suggestions(
    report: MatchReport,
    profile: dict,
) -> List[str]:
    """最多 3 条，每条带主线标签。"""
    out = []
    if report.conclusion == "当前适合投递":
        out.append(
            "[投递准备] 准备简历 / 项目概述 / 自我介绍，按缺口清单做最后补强"
        )
    elif report.conclusion == "中长期可转向":
        out.append(
            "[补项目] 将缺口清单转成 daily_log.md 核心任务，"
            "1-2 周内补齐关键经历后再评估"
        )
    else:
        out.append(
            "[岗位调研] 暂不投递，优先修正硬门槛或改换目标方向"
        )

    # 画像级固定建议
    if (profile.get("项目数量", 0) or 0) == 0:
        out.append(
            "[补项目] 在 2026-05-01 前形成至少 1 个可投递项目（profile §4）"
        )

    # 如果技能缺口非空，追加一条补技能
    if report.gap_list.get("技能缺口"):
        out.append(
            "[补技能] 针对技能缺口安排当周学习任务，"
            "每项缺口对应 1 个可交付小产出"
        )

    return out[:3]


# =====================================================
# 主流程
# =====================================================

def run_match(
    profile: dict,
    jd_text: str,
    jd_title: str = "未命名 JD",
    jd_analysis: dict | None = None,
    semantic_alignment: dict | None = None,
) -> MatchReport:
    jd = jd_text  # 保持原文大小写做人读展示
    hard = [
        check_education(profile, _analysis_gate_text(jd, jd_analysis, "education")),
        check_major(profile, _analysis_gate_text(jd, jd_analysis, "major")),
        check_experience(profile, _analysis_gate_text(jd, jd_analysis, "experience")),
        check_language(profile, _analysis_gate_text(jd, jd_analysis, "language")),
        check_location(profile, _analysis_gate_text(jd, jd_analysis, "location")),
        check_tech_mainline(profile, _analysis_gate_text(jd, jd_analysis, "tech")),
    ]
    soft = [
        soft_skill_overlap(profile, jd, jd_analysis, semantic_alignment),
        soft_programming_language(profile, jd),
        soft_direction(profile, jd),
        soft_project_fit(profile, jd, jd_analysis, semantic_alignment),
        soft_work_mode(profile, jd),
        soft_industry_preference(profile, jd),
        soft_salary(profile, jd),
        soft_location(profile, jd),
    ]
    direction = judge_direction(profile, jd)
    requirement_analysis: Dict[str, object] = {}
    try:
        from career_domains import detect_role_family, resolve_domain
        from job_requirements import analyze_programming_languages
        requirement_analysis = {
            "career_domain": resolve_domain(jd),
            "role_family": detect_role_family(jd),
            "programming_languages": analyze_programming_languages(jd).as_dict(),
            "user_targets": list(profile.get("方向优先级") or []),
            "target_job_type": profile.get("目标岗位类型") or "不限",
            "industry_preference": profile.get("行业偏好") or "不限",
            "jd_analysis": {
                "schema_version": (jd_analysis or {}).get("schema_version", ""),
                "source": (jd_analysis or {}).get("source", ""),
                "requirements": list((jd_analysis or {}).get("requirements") or []),
                "keywords": list((jd_analysis or {}).get("keywords") or []),
                "warnings": list((jd_analysis or {}).get("warnings") or []),
            },
            "semantic_alignment": semantic_alignment or {
                "status": "not_requested", "source": "deterministic",
                "alignments": [], "warnings": [],
            },
        }
    except Exception:
        requirement_analysis = {"career_domain": None, "role_family": None}
    report = MatchReport(
        jd_title=jd_title,
        direction=direction,
        hard_gate=hard,
        soft_dims=soft,
        requirement_analysis=requirement_analysis,
    )
    report.conclusion, report.conclusion_reason = decide(
        hard,
        soft,
        direction=direction,
        has_target_direction=bool(profile.get("方向优先级")),
    )
    report.gap_list = build_gap_list(hard, soft, profile)
    report.suggestions = build_suggestions(report, profile)
    return report


def format_report(report: MatchReport) -> str:
    """输出结构严格对齐 target_rules.md §6 与 job_match_prompt.md 第 8 步。"""
    lines = []
    lines.append(f"岗位：{report.jd_title}")
    lines.append(f"方向判定：{report.direction}")
    lines.append("")
    lines.append("硬门槛：")
    for r in report.hard_gate:
        lines.append(f"  - {r.name}：{r.status}（{r.reason}）")
    lines.append("")
    lines.append("软性维度：")
    for r in report.soft_dims:
        lines.append(f"  - {r.name}：{r.status}（{r.reason}）")
    lines.append("")
    lines.append(f"结论：{report.conclusion}")
    lines.append(f"一句话理由：{report.conclusion_reason}")
    lines.append("")
    lines.append("缺口清单（喂给路线规划模块）：")
    for category, items in report.gap_list.items():
        lines.append(f"  - {category}：")
        if not items:
            lines.append("      * （无）")
        else:
            for it in items:
                lines.append(f"      * {it}")
    lines.append("")
    lines.append("下一步建议（最多 3 条，每条带主线标签）：")
    for s in report.suggestions:
        lines.append(f"  - {s}")
    return "\n".join(lines)


# =====================================================
# 演示数据（可替换）
# =====================================================

# 纯合成演示画像，不对应任何真实用户。
DEMO_PROFILE = {
    # §1
    "学历": "本科",
    "专业": "软件工程",
    "所在地": "杭州",
    "可接受地域": ["杭州", "上海", "远程"],
    # §2
    "方向优先级": [
        "AI 应用开发",
        "Python 后端",
    ],
    # 只放能被 check_tech_mainline 的 b.split()[0].lower() 机制使用的干净 token。
    # profile §2 的第二条 "和目前大模型主线出入大的" 是 meta 规则，
    # 无法转成单 token，在 MVP 规则版里不纳入机读列表，留在 user_profile.md 作人读。
    "明确不做": ["嵌入式"],
    "工作性质偏好": "不限",
    "期望薪资": "面议",
    # §3
    "熟练技能": ["Python"],
    "会用技能": ["FastAPI", "SQL"],
    # §4 / §7
    "项目数量": 2,
    "实习数量": 1,
    # §9
    "英语自评": 3,
}


DEMO_JD = """
岗位名称：AI 应用开发工程师（实习）
公司：示例公司
工作地点：北京
学历要求：本科及以上
专业要求：计算机、软件、通信、电子信息等相关专业
经验要求：有 LLM / Agent / Prompt 相关项目经验者优先，实习生不强制要求年限
语言要求：能阅读英文技术文档
技术要求：
  - 熟悉 Python
  - 了解 Prompt 工程 / LangChain / Agent 工作流
  - 加分：有开源项目或个人作品集
工作性质：驻场
"""


def main():
    # Windows 控制台默认 GBK 无法输出 ✓ / ✗，这里显式切到 UTF-8。
    # Python 3.7+ 支持 reconfigure；失败则静默回退（例如被重定向到管道时）。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    report = run_match(
        DEMO_PROFILE,
        DEMO_JD,
        jd_title="AI 应用开发工程师（实习）",
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
