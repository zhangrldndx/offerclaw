"""profile_loader.py — OfferClaw 状态真实化（产品级 Agent 化指导 §4）

把 ``user_profile.md`` 解析成 ``match_job.run_match`` 能直接吃的 dict，
让 ``/api/match`` 等核心链路不再依赖 ``match_job.DEMO_PROFILE``。

**设计原则**

1. 不引入第三方 Markdown 解析库，纯正则 + 行扫描。
2. 字段缺失时回退到纯合成的 ``profiles/p1_demo_ai.json`` 对应键，
   再缺失时给安全默认（空 list / 0 / "不限"），保证下游 ``run_match``
   永远能拿到完整的 13 个键，不抛 KeyError。
3. 输出 dict 的键名和类型完全对齐 ``match_job.DEMO_PROFILE``，
   方便老调用方一行替换。
4. 默认带轻量缓存（按文件 mtime 失效），避免每次 ``/api/match``
   都重新读盘 + 正则。
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_MD = os.path.join(BASE_DIR, "user_profile.md")
PROFILE_JSON_FALLBACK = os.path.join(BASE_DIR, "profiles", "p1_demo_ai.json")

REQUIRED_KEYS = [
    "学历",
    "专业",
    "所在地",
    "可接受地域",
    "方向优先级",
    "目标岗位类型",
    "行业偏好",
    "明确不做",
    "工作性质偏好",
    "期望薪资",
    "熟练技能",
    "会用技能",
    "技能证据",
    "项目技能",
    "项目数量",
    "实习数量",
    "英语自评",
]

_SAFE_DEFAULT: dict[str, Any] = {
    "学历": "本科",
    "专业": "",
    "所在地": "",
    "可接受地域": [],
    "方向优先级": [],
    "目标岗位类型": "不限",
    "行业偏好": "不限",
    "明确不做": [],
    "工作性质偏好": "不限",
    "期望薪资": "面议",
    "熟练技能": [],
    "会用技能": [],
    "技能证据": [],
    "项目技能": [],
    "项目数量": 0,
    "实习数量": 0,
    "英语自评": 1,
}

# ------- 简易缓存 -------
_CACHE: dict[str, Any] = {
    "mtime": None, "revision": None, "path": None, "data": None, "evidence": None,
}


# =====================================================
# 字段提取（每个函数独立、可单测）
# =====================================================

def _line_value(text: str, label: str) -> str | None:
    """匹配形如 ``- 学历层次：硕士（在读）`` / ``学历层次：硕士``，返回冒号后内容。"""
    pat = re.compile(rf"^\s*-?\s*{re.escape(label)}\s*[:：]\s*(.+?)\s*$", re.MULTILINE)
    m = pat.search(text)
    return m.group(1).strip() if m else None


def _strip_paren(s: str) -> str:
    """去掉中英文括号及其中内容，例如 ``硕士（在读）`` → ``硕士``。"""
    return re.sub(r"[（(].*?[)）]", "", s).strip()


def _split_locations(s: str) -> list[str]:
    parts = re.split(r"[/、，,;；\s]+", s)
    return [p for p in (x.strip() for x in parts) if p]


def parse_education(text: str) -> str:
    raw = _line_value(text, "学历层次") or _line_value(text, "学历")
    if not raw:
        return _SAFE_DEFAULT["学历"]
    cleaned = _strip_paren(raw)
    for k in ["博士", "硕士", "本科", "大专", "高中"]:
        if k in cleaned:
            return k
    return cleaned or _SAFE_DEFAULT["学历"]


def parse_major(text: str) -> str:
    return (_line_value(text, "专业") or "").strip() or _SAFE_DEFAULT["专业"]


def parse_location(text: str) -> str:
    return (_line_value(text, "所在地") or "").strip() or _SAFE_DEFAULT["所在地"]


def parse_acceptable_locations(text: str) -> list[str]:
    raw = _line_value(text, "可接受工作地域") or _line_value(text, "可接受地域")
    return _split_locations(raw) if raw else []


def parse_directions(text: str) -> list[str]:
    """解析 ``§2 目标方向`` 子项下的有序列表（``1. xxx`` / ``2. xxx``）。

    ``目标方向`` 不是 ``##`` 顶级头而是 §2 内部 bullet，需要先抽 §2 再抽子块。
    """
    sec = _section(text, r"求职方向与偏好") or _section(text, r"求职方向")
    if not sec:
        return []
    m = re.search(r"^-\s*目标方向[^\n:：]*[：:]\s*\n([\s\S]*?)(?=^-\s|\Z)", sec, re.MULTILINE)
    block = m.group(1) if m else sec
    return [
        m.group(1).strip()
        for m in re.finditer(r"^\s*\d+[.、)]\s*(.+?)\s*$", block, re.MULTILINE)
    ]


def parse_target_job_type(text: str) -> str:
    return (_line_value(text, "目标岗位类型") or _SAFE_DEFAULT["目标岗位类型"]).strip()


def parse_industry_preference(text: str) -> str:
    return (_line_value(text, "行业偏好") or _SAFE_DEFAULT["行业偏好"]).strip()


def parse_explicit_not(text: str) -> list[str]:
    """``§2`` 中 ``- 明确不做的方向：`` 子项，取每条第一个 Latin token，小写化。

    user_profile.md 里它不是 ``##`` 顶级头，而是 §2 内部的二级 bullet，
    所以这里专门做"父 bullet 起，到下一个同级 bullet 止"的子块抽取。
    """
    sec = _section(text, r"求职方向与偏好") or _section(text, r"求职方向")
    if not sec:
        return []
    m = re.search(r"^-\s*明确不做的方向[：:]\s*\n([\s\S]*?)(?=^-\s|\Z)", sec, re.MULTILINE)
    if not m:
        return []
    sub = m.group(1)
    out: list[str] = []
    for line in re.finditer(r"^\s*-\s*([A-Za-z][A-Za-z0-9+#.\-]*)", sub, re.MULTILINE):
        tok = line.group(1).strip().lower()
        if tok and tok not in out and tok != "meta":
            out.append(tok)
    return out


def parse_work_mode(text: str) -> str:
    return (_line_value(text, "工作性质偏好") or _SAFE_DEFAULT["工作性质偏好"]).strip()


def parse_salary(text: str) -> str:
    return (_line_value(text, "期望薪资区间") or _line_value(text, "期望薪资") or _SAFE_DEFAULT["期望薪资"]).strip()


def _skills_block(text: str) -> str:
    """抽 §3 ``技能清单`` 中 ``编程语言`` 子块。"""
    sec = _section(text, r"技能清单")
    if not sec:
        return ""
    m = re.search(r"编程语言[：:]([\s\S]*?)(?:\n\s*-\s*工具|\n\s*-\s*AI|\n##|\Z)", sec)
    return m.group(1) if m else sec


def parse_skills_proficient(text: str) -> list[str]:
    block = _skills_block(text)
    m = re.search(r"熟练[：:]\s*(.+)", block)
    if not m:
        return []
    return _split_skills(m.group(1))


def parse_skills_familiar(text: str) -> list[str]:
    block = _skills_block(text)
    m = re.search(r"会用[：:]\s*(.+)", block)
    if not m:
        return []
    return _split_skills(m.group(1))


def _split_skills(s: str) -> list[str]:
    s = _strip_paren(s)
    parts = re.split(r"[、,，/;；\s]+", s)
    return [p for p in (x.strip() for x in parts) if p]


def _scan_known_capabilities(text: str) -> list[str]:
    """从画像原文抽取明确出现的能力词，不把它们升级为“熟练”。"""
    vocabulary: list[str] = []
    try:
        from context_budget import keywords_from
        vocabulary.extend(keywords_from(text or "", top=120))
    except Exception:
        pass
    try:
        from career_domains import load_domain_templates
        for domain in load_domain_templates():
            vocabulary.extend(domain.get("competency_keywords") or [])
    except Exception:
        pass
    out: list[str] = []
    lower = (text or "").lower()
    seen: set[str] = set()
    for kw in vocabulary:
        key = str(kw or "").lower()
        if not key or key in seen:
            continue
        start = 0
        supported = False
        while True:
            idx = lower.find(key, start)
            if idx < 0:
                break
            before = lower[max(0, idx - 18):idx]
            if not any(neg in before for neg in ("无", "没有", "未使用", "不使用", "无需", "不依赖", "未接触")):
                supported = True
                break
            start = idx + len(key)
        if supported:
            display = kw
            if str(kw).isascii() and str(kw).islower() and len(str(kw)) <= 10:
                variants = re.findall(re.escape(str(kw)), text or "", re.I)
                upper = next((value for value in variants if value.isupper()), "")
                if upper:
                    display = upper
            out.append(display)
            seen.add(key)
    return out


def parse_skill_evidence(text: str) -> list[str]:
    """读取 §3 + §4 的显式能力证据，补足旧 loader 只读编程语言的盲区。"""
    return _scan_known_capabilities(
        (_section(text, r"技能清单") or "") + "\n" + (_section(text, r"项目经历") or "")
    )


def parse_project_skills(text: str) -> list[str]:
    return _scan_known_capabilities(_section(text, r"项目经历") or "")


def parse_project_count(text: str) -> int:
    """统计 §4 中 ``项目 N`` 出现次数（去重）。"""
    sec = _section(text, r"项目经历")
    if not sec:
        return 0
    nums = set(int(m.group(1)) for m in re.finditer(r"^\s*-\s*项目\s*(\d+)", sec, re.MULTILINE))
    return len(nums)


def parse_intern_count(text: str) -> int:
    """统计 §7 中 ``实习 N`` 且非『待补充』的实习经历数。"""
    sec = _section(text, r"实习\s*/\s*工作经历")
    if not sec:
        return 0
    n = 0
    for m in re.finditer(r"^\s*-\s*实习\s*\d+[：:]\s*(.+?)\s*$", sec, re.MULTILINE):
        if "待补充" not in m.group(1):
            n += 1
    return n


def parse_english(text: str) -> int:
    """从 §9 自评表读 ``英语读写`` 行的分数；找不到则看『英语自评』行。"""
    sec = _section(text, r"当前能力自评")
    if sec:
        m = re.search(r"英语[读写]?[读写]?\s*\|\s*(\d+)", sec)
        if m:
            return int(m.group(1))
    raw = _line_value(text, "英语自评")
    if raw:
        m = re.search(r"\d+", raw)
        if m:
            return int(m.group(0))
    return _SAFE_DEFAULT["英语自评"]


def _evidence_id(source_ref: str, content: str) -> str:
    raw = f"{source_ref}|{content}".encode("utf-8")
    return f"ev_{hashlib.sha256(raw).hexdigest()[:20]}"


def _evidence_item(evidence_type: str, source_ref: str, content: str) -> dict[str, str]:
    value = str(content or "").strip()[:5000]
    return {
        "evidence_id": _evidence_id(source_ref, value),
        "evidence_type": evidence_type,
        "source_ref": source_ref,
        "text": value,
    }


def _is_positive_evidence(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized or "待补充" in normalized:
        return False
    # Aspirations and recorded gaps must never be sent to the semantic matcher
    # as proof that a capability already exists.
    if re.search(r"(?:想学|计划学习|尚未|还没|未掌握|缺少|不足|不会)", normalized):
        return False
    return True


def parse_evidence_catalog(text: str) -> list[dict[str, str]]:
    """Preserve profile facts that semantic matching can cite.

    The legacy loader reduced projects to a count and a vocabulary list.  This
    catalog keeps the source wording while excluding future plans, gaps and
    placeholders.  It is an internal field; existing profile contracts remain
    unchanged.
    """
    items: list[dict[str, str]] = []

    skill_section = _section(text, r"技能清单")
    for index, raw in enumerate(skill_section.splitlines()):
        value = re.sub(r"^\s*[-*]\s*", "", raw).strip()
        if not _is_positive_evidence(value):
            continue
        if value.startswith(">") or value in {"编程语言：", "工具与框架：", "AI / LLM 相关技能："}:
            continue
        items.append(_evidence_item(
            "declared_skill", f"user_profile.md#section-3-line-{index + 1}", value,
        ))

    project_section = _section(text, r"项目经历")
    project_matches = list(re.finditer(r"^\s*-\s*项目\s*(\d+)\s*$", project_section, re.M))
    for index, match in enumerate(project_matches):
        end = project_matches[index + 1].start() if index + 1 < len(project_matches) else len(project_section)
        block = project_section[match.start():end].strip()
        block = "\n".join(
            line for line in block.splitlines()
            if not line.lstrip().startswith(">") and "待补充" not in line
        ).strip()
        # A completed project may legitimately describe a problem as
        # "性能不足". Do not discard the whole evidence block because of that
        # wording; quote/future notes and placeholders were removed per line.
        substantive = re.sub(r"^\s*-\s*项目\s*\d+\s*$", "", block, flags=re.M).strip()
        if len(substantive) >= 12:
            items.append(_evidence_item(
                "project", f"user_profile.md#section-4-project-{match.group(1)}", block,
            ))

    section_specs = (
        (r"竞赛\s*/\s*获奖经历", "competition", "5"),
        (r"科研\s*/\s*论文\s*/\s*专利", "research", "6"),
        (r"实习\s*/\s*工作经历", "work", "7"),
    )
    for header, evidence_type, section_number in section_specs:
        section = _section(text, header)
        for index, raw in enumerate(section.splitlines()):
            value = re.sub(r"^\s*[-*]\s*", "", raw).strip()
            if not _is_positive_evidence(value) or value.startswith(">"):
                continue
            items.append(_evidence_item(
                evidence_type,
                f"user_profile.md#section-{section_number}-line-{index + 1}",
                value,
            ))

    return items


# =====================================================
# 工具
# =====================================================

def _section(text: str, header_pat: str) -> str:
    """抽取 ``## N. <header_pat>...`` 到下一个 ``## `` 之间的内容。"""
    m = re.search(rf"^##\s*\d+\.[^\n]*{header_pat}[^\n]*\n([\s\S]*?)(?=^##\s|\Z)", text, re.MULTILINE)
    return m.group(1) if m else ""


# =====================================================
# 主入口
# =====================================================

def load_profile(path: str | None = None, *, use_cache: bool = True,
                 include_evidence: bool = False) -> dict[str, Any]:
    """从 ``user_profile.md`` 解析出 ``run_match`` 能用的 profile dict。

    流程：
      1. 优先解析 Markdown；缺字段就回退到 ``profiles/p1_demo_ai.json``；
      2. 再缺就用 ``_SAFE_DEFAULT`` 兜底；
      3. 保证返回值一定包含 ``REQUIRED_KEYS`` 全部 13 个键。
    """
    md_path = path or PROFILE_MD
    structured = None
    if path is None:
        try:
            from profile_review import ProfileRepository
            structured = ProfileRepository(profile_path=md_path).current()
        except Exception:
            structured = None

    if use_cache and _CACHE["data"] is not None and _CACHE["path"] == md_path:
        try:
            mtime = os.path.getmtime(md_path) if os.path.exists(md_path) else None
            current_revision = structured.get("revision") if structured else None
            if (current_revision == _CACHE["revision"] and
                    (structured is not None or mtime == _CACHE["mtime"])):
                cached = dict(_CACHE["data"])
                if include_evidence:
                    cached["_evidence_catalog"] = list(_CACHE.get("evidence") or [])
                return cached
        except OSError:
            pass

    text = str(structured.get("content_md") or "") if structured else ""
    if not text and os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

    parsed: dict[str, Any] = {
        "学历": parse_education(text) if text else "",
        "专业": parse_major(text) if text else "",
        "所在地": parse_location(text) if text else "",
        "可接受地域": parse_acceptable_locations(text) if text else [],
        "方向优先级": parse_directions(text) if text else [],
        "目标岗位类型": parse_target_job_type(text) if text else "不限",
        "行业偏好": parse_industry_preference(text) if text else "不限",
        "明确不做": parse_explicit_not(text) if text else [],
        "工作性质偏好": parse_work_mode(text) if text else "不限",
        "期望薪资": parse_salary(text) if text else "",
        "熟练技能": parse_skills_proficient(text) if text else [],
        "会用技能": parse_skills_familiar(text) if text else [],
        "技能证据": parse_skill_evidence(text) if text else [],
        "项目技能": parse_project_skills(text) if text else [],
        "项目数量": parse_project_count(text) if text else 0,
        "实习数量": parse_intern_count(text) if text else 0,
        "英语自评": parse_english(text) if text else _SAFE_DEFAULT["英语自评"],
    }

    fallback: dict[str, Any] = {}
    if os.path.exists(PROFILE_JSON_FALLBACK):
        try:
            with open(PROFILE_JSON_FALLBACK, "r", encoding="utf-8") as f:
                fallback = json.load(f)
        except (OSError, json.JSONDecodeError):
            fallback = {}

    out: dict[str, Any] = {}
    list_keys = {"可接受地域", "方向优先级", "明确不做", "熟练技能", "会用技能", "技能证据", "项目技能"}
    for key in REQUIRED_KEYS:
        v = parsed.get(key)
        # text 非空时 list 字段保留解析结果（即使为空），不再被 fallback 覆盖，
        # 否则用户从 user_profile.md 删掉某项，行为不会变化（状态驱动名存实亡）。
        is_list_with_text = key in list_keys and bool(text)
        if _is_empty(v) and not is_list_with_text:
            v = fallback.get(key)
        if _is_empty(v) and not is_list_with_text:
            v = _SAFE_DEFAULT[key]
        if v is None:
            v = _SAFE_DEFAULT[key]
        out[key] = v

    out["_source"] = "profile.sqlite3" if structured else ("user_profile.md" if text else (
        "profiles/p1_demo_ai.json" if fallback else "safe_default"
    ))
    evidence_catalog = parse_evidence_catalog(text) if text else []

    if use_cache and os.path.exists(md_path):
        _CACHE["mtime"] = os.path.getmtime(md_path)
        _CACHE["revision"] = structured.get("revision") if structured else None
        _CACHE["path"] = md_path
        _CACHE["data"] = dict(out)
        _CACHE["evidence"] = list(evidence_catalog)

    if include_evidence:
        out["_evidence_catalog"] = evidence_catalog
    return out


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, str, dict)) and len(v) == 0:
        return True
    return False


def reset_cache() -> None:
    """测试 / 状态变更后强制下次重新解析。"""
    _CACHE["mtime"] = None
    _CACHE["revision"] = None
    _CACHE["path"] = None
    _CACHE["data"] = None
    _CACHE["evidence"] = None


# =====================================================
# Schema validation（V4 §5）
# =====================================================

PROFILE_SCHEMA_PATH = os.path.join(BASE_DIR, "profile_schema.json")
_SCHEMA_CACHE: dict[str, Any] = {"data": None, "mtime": None}


def load_schema() -> dict:
    """读 profile_schema.json，带 mtime 缓存。"""
    if _SCHEMA_CACHE["data"] is not None:
        try:
            if os.path.getmtime(PROFILE_SCHEMA_PATH) == _SCHEMA_CACHE["mtime"]:
                return _SCHEMA_CACHE["data"]
        except OSError:
            pass
    with open(PROFILE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _SCHEMA_CACHE["data"] = data
    try:
        _SCHEMA_CACHE["mtime"] = os.path.getmtime(PROFILE_SCHEMA_PATH)
    except OSError:
        pass
    return data


def validate_profile(p: dict) -> tuple[bool, list[str]]:
    """校验 profile dict 是否符合 schema。

    Returns ``(ok, errors)``：errors 是字符串列表（友好可读），
    可以直接落到日志或返回给前端。
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return True, ["jsonschema 未安装，校验被跳过"]
    schema = load_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for err in validator.iter_errors(p):
        path = ".".join(str(x) for x in err.absolute_path) or "<root>"
        errors.append(f"[{path}] {err.message}")
    return (len(errors) == 0), errors


if __name__ == "__main__":
    import json as _j
    p = load_profile()
    print(_j.dumps(p, ensure_ascii=False, indent=2))
    ok, errs = validate_profile(p)
    print(f"\nschema valid: {ok}")
    if errs:
        for e in errs:
            print(f"  - {e}")
