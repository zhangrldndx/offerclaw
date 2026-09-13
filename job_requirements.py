# -*- coding: utf-8 -*-
"""从 JD 原文抽取可审计的岗位要求。

本模块只做确定性文本解析，不调用 LLM，也不修改用户画像。当前先解决最容易
误伤候选人的“编程语言列表”问题：多语言并列默认是候选池，只有原文明确要求
同时掌握时才判多语言硬要求；单一语言或明确主语言则标成重点要求。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


_LANG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("C++", re.compile(r"(?<![A-Za-z0-9])c\s*\+\s*\+(?![A-Za-z0-9])", re.I)),
    ("C#", re.compile(r"(?<![A-Za-z0-9])c\s*#(?![A-Za-z0-9])", re.I)),
    ("C", re.compile(r"(?<![A-Za-z0-9])C(?:语言)?(?![A-Za-z0-9+#])")),
    ("Python", re.compile(r"(?<![A-Za-z0-9])python(?![A-Za-z0-9])", re.I)),
    ("Java", re.compile(r"(?<![A-Za-z0-9])java(?!script)(?![A-Za-z0-9])", re.I)),
    ("JavaScript", re.compile(r"(?<![A-Za-z0-9])(?:javascript|js)(?![A-Za-z0-9])", re.I)),
    ("TypeScript", re.compile(r"(?<![A-Za-z0-9])(?:typescript|ts)(?![A-Za-z0-9])", re.I)),
    ("Go", re.compile(r"(?<![A-Za-z0-9])(?:golang|Go)(?![A-Za-z0-9])")),
    ("Rust", re.compile(r"(?<![A-Za-z0-9])rust(?![A-Za-z0-9])", re.I)),
    ("Kotlin", re.compile(r"(?<![A-Za-z0-9])kotlin(?![A-Za-z0-9])", re.I)),
    ("Swift", re.compile(r"(?<![A-Za-z0-9])swift(?![A-Za-z0-9])", re.I)),
    ("PHP", re.compile(r"(?<![A-Za-z0-9])php(?![A-Za-z0-9])", re.I)),
    ("Ruby", re.compile(r"(?<![A-Za-z0-9])ruby(?![A-Za-z0-9])", re.I)),
    ("Scala", re.compile(r"(?<![A-Za-z0-9])scala(?![A-Za-z0-9])", re.I)),
    ("MATLAB", re.compile(r"(?<![A-Za-z0-9])matlab(?![A-Za-z0-9])", re.I)),
    ("R", re.compile(r"(?<![A-Za-z0-9])R(?:语言)?(?![A-Za-z0-9])")),
)

_ONE_OF = ("至少一门", "至少一种", "至少掌握一门", "至少掌握一种", "一种以上", "一种或多种",
           "掌握一种或多种", "任意一种", "任一", "其中一种")
_PREFERRED = ("优先", "加分", "bonus", "更佳", "为佳")
_MULTI_REQUIRED = ("同时掌握", "同时精通", "均需掌握", "均要求", "都要掌握", "缺一不可")
_SINGLE_STRONG = ("必须", "精通", "熟练掌握", "主要开发语言", "为主", "开发经验", "核心技术栈")
_CONTEXT_ONLY = ("团队使用", "技术栈包括", "项目使用", "目前使用")


@dataclass(frozen=True)
class ProgrammingLanguageRequirement:
    policy: str = "none"
    languages: tuple[str, ...] = ()
    hard_required: tuple[str, ...] = ()
    preferred: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    ambiguous: bool = False

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "languages": list(self.languages),
            "hard_required": list(self.hard_required),
            "preferred": list(self.preferred),
            "evidence": list(self.evidence),
            "ambiguous": self.ambiguous,
        }


def normalize_programming_language(value: object) -> str | None:
    """把画像技能或“不做方向”中的语言写法归一到同一名称。"""
    text = str(value or "").strip()
    for name, pattern in _LANG_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _languages_in(text: str) -> list[str]:
    return [name for name, pattern in _LANG_PATTERNS if pattern.search(text or "")]


def analyze_programming_languages(jd_text: str) -> ProgrammingLanguageRequirement:
    """解析 JD 中的编程语言要求，默认高召回、避免把语言列表误判为 all-of。"""
    lines = [line.strip(" \t-•") for line in (jd_text or "").splitlines() if line.strip()]
    evidence = [line for line in lines if _languages_in(line)]
    languages: list[str] = []
    for line in evidence:
        for language in _languages_in(line):
            if language not in languages:
                languages.append(language)
    if not languages:
        return ProgrammingLanguageRequirement()

    joined = "\n".join(evidence)
    preferred: list[str] = []
    strong: list[str] = []
    for line in evidence:
        lower = line.lower()
        for language, pattern in _LANG_PATTERNS:
            for match in pattern.finditer(line):
                # 同一行可能同时出现“Java 为主，Python 加分”，因此必须按语言
                # 附近的局部短语分类，不能把整行一刀切成 preferred。
                left = max(lower.rfind(sep, 0, match.start()) for sep in ("，", ",", "；", ";")) + 1
                right_candidates = [lower.find(sep, match.end()) for sep in ("，", ",", "；", ";")]
                right_candidates = [i for i in right_candidates if i >= 0]
                right = min(right_candidates) if right_candidates else len(lower)
                local = lower[left:right]
                if any(h.lower() in local for h in _SINGLE_STRONG) and language not in strong:
                    strong.append(language)
                if any(h.lower() in local for h in _PREFERRED) and language not in preferred:
                    preferred.append(language)

    # “至少一门/一种以上”是最明确的 one-of 证据。
    if any(h in joined for h in _ONE_OF):
        return ProgrammingLanguageRequirement(
            policy="flexible_pool", languages=tuple(languages), preferred=tuple(preferred),
            evidence=tuple(evidence),
        )

    # 只有明确出现“同时/均需/缺一不可”等措辞，才允许升级为多语言硬要求。
    if len(languages) >= 2 and any(h in joined for h in _MULTI_REQUIRED):
        return ProgrammingLanguageRequirement(
            policy="multi_required_explicit", languages=tuple(languages),
            hard_required=tuple(languages), preferred=tuple(preferred), evidence=tuple(evidence),
        )

    # 强要求优先于同一行稍后出现的“加分”；其余被局部加分词修饰的语言剔除。
    non_preferred = [language for language in languages if language in strong or language not in preferred]

    if not non_preferred:
        return ProgrammingLanguageRequirement(
            policy="preferred", languages=tuple(languages), preferred=tuple(preferred or languages),
            evidence=tuple(evidence),
        )

    # JD 只列一种语言，或明确“X 为主/必须/多年经验”，将它标成重点要求。
    # 是否因此不投仍由画像诉求与总体匹配决定，不由本解析器直接裁决。
    if all(any(h.lower() in line.lower() for h in _CONTEXT_ONLY) for line in evidence):
        return ProgrammingLanguageRequirement(
            policy="context_only", languages=tuple(languages), preferred=tuple(preferred),
            evidence=tuple(evidence),
        )

    if len(non_preferred) == 1:
        language = non_preferred[0]
        return ProgrammingLanguageRequirement(
            policy="single_required", languages=tuple(languages), hard_required=(language,),
            preferred=tuple(preferred), evidence=tuple(evidence),
            ambiguous=not any(h.lower() in joined.lower() for h in _SINGLE_STRONG),
        )

    # “Python/C++”“Java、Python、Go”等普通并列一律按候选池处理；即便原文
    # 写“熟练掌握”，没有双语言强制证据也不能当成 all-of。
    return ProgrammingLanguageRequirement(
        policy="flexible_pool", languages=tuple(languages), preferred=tuple(preferred),
        evidence=tuple(evidence), ambiguous=not any(h in joined for h in _ONE_OF),
    )


def profile_programming_languages(profile: dict) -> set[str]:
    out: set[str] = set()
    for value in (profile.get("熟练技能") or []) + (profile.get("会用技能") or []):
        language = normalize_programming_language(value)
        if language:
            out.add(language)
    return out
