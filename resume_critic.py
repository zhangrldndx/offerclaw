# -*- coding: utf-8 -*-
"""Resume hard checks and the independent v2 Resume Critic Agent.

``critic_report`` remains the deterministic compatibility API used by the
lightweight project-section tool. ``run_resume_critic_agent`` is the actual
LLM reviewer in the full Resume Review workflow. Hard rules retain veto power;
the Critic supplies the semantic judgment that keyword and blacklist checks
cannot provide.
"""
from __future__ import annotations

import json
import os
import re

from jd_parser import term_in_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 简历里绝不该出现的编造（与 resume_builder 的"不允许编造"同源）
_FABRICATION_BLACKLIST = [
    "上千人", "线上服务", "LinkedIn", "智联", "Boss 直聘", "Boss直聘",
    "自动投递", "自动跟进", "React", "Vue", "登录系统",
]
# "数据库":项目确无关系型库,但"向量数据库/ChromaDB"合法 → 只拦裸"数据库"
_DB_FABRICATION = re.compile(r"(?<!向量)数据库")

# 覆盖率 needs_fix 阈值:**provisional**,由 E1（P4）标定后再收紧,先给保守下限
_MIN_COVERAGE = 0.15


def _current_metrics() -> dict:
    try:
        with open(os.path.join(BASE_DIR, "metrics.json"), encoding="utf-8") as f:
            return json.load(f).get("current", {})
    except Exception:
        return {}


def keyword_coverage(resume_md: str, jd_keywords: list) -> dict:
    """腿1:按 JDAnalysis importance 计算覆盖；旧字符串列表仍兼容。"""
    text = resume_md or ""
    hit, miss = [], []
    hit_weight = 0.0
    total_weight = 0.0
    for item in (jd_keywords or []):
        if isinstance(item, dict):
            name = str(item.get("canonical_name") or "").strip()
            forms = [str(value).strip() for value in (item.get("surface_forms") or [])
                     if str(value).strip()]
            candidates = list(dict.fromkeys([name, *forms]))
            try:
                weight = max(0.0, min(1.0, float(item.get("importance", 1.0))))
            except (TypeError, ValueError):
                weight = 1.0
        else:
            name = str(item or "").strip()
            candidates = [name]
            weight = 1.0
        if not name:
            continue
        matched = any(term_in_text(text, value) for value in candidates if value)
        (hit if matched else miss).append(name)
        total_weight += weight
        if matched:
            hit_weight += weight
    total = len(hit) + len(miss)
    return {"hit": hit, "miss": miss,
            "coverage": (hit_weight / total_weight) if total_weight else 1.0,
            "weighted_hit": round(hit_weight, 6),
            "weighted_total": round(total_weight, 6),
            "item_count": total}


def _stale_number_flags(resume_md: str) -> list:
    """腿3a:chunks/pytest/路由 数字与 metrics.json 真值对账,不符即 flag。

    严格 scope 到这三类（蓝图 §6.6）,避免误伤简历里合法的任意量化数字抬高误报率。
    """
    m = _current_metrics()
    flags = []
    checks = [
        (r"(\d[\d,]*)\s*chunks?", "chunks", str(m.get("chunks", ""))),
        (r"pytest\s*(\d+)", "pytest", str(m.get("pytest", ""))),
        (r"(\d+)\s*(?:条)?\s*路由", "routes", str(m.get("routes", ""))),
    ]
    for pat, name, truth in checks:
        for mt in re.finditer(pat, resume_md or "", re.I):
            got = mt.group(1).replace(",", "")
            if truth and got != truth:
                flags.append({"kind": "stale_number", "field": name,
                              "found": got, "truth": truth})
    return flags


def _blacklist_flags(resume_md: str) -> list:
    text = resume_md or ""
    flags = [{"kind": "fabrication", "term": t}
             for t in _FABRICATION_BLACKLIST if t in text]
    if _DB_FABRICATION.search(text):
        flags.append({"kind": "fabrication", "term": "数据库(裸)"})
    return flags


def fabrication_flags(resume_md: str) -> list:
    """腿3:黑名单 + 数字对账（纯代码硬拦）。"""
    return _blacklist_flags(resume_md) + _stale_number_flags(resume_md)


def _profile_to_evidence(profile: dict) -> str:
    """把画像 dict 序列化成证据文本（供腿2语义核对"简历是否有画像支撑"）。"""
    parts = []
    for k, v in (profile or {}).items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, (list, tuple)):
            v = "、".join(str(x) for x in v)
        parts.append(f"{k}：{v}")
    return "\n".join(parts)[:4000]


def _semantic_fabrication_check(resume_md: str, evidence: str):
    """腿2（LLM,默认关）:独立判断简历里哪些具体主张缺画像证据支撑。

    **结构上独立于 Writer**:只喂 {简历 + 画像证据},不喂 Writer 的推理/事实清单,让审查"裸眼"找茬。
    LLM 只 flag,不参与硬 verdict（硬拦交代码,见蓝图 §6.4）。返回 flag 列表;无 key/失败 → None 降级。
    """
    try:
        from day1_api_starter import get_llm_config
        from plan_gen import call_llm_plain
        api_key = (get_llm_config() or {}).get("api_key")
        if not api_key:
            return None
        messages = [
            {"role": "system", "content":
                "你是独立的简历审核员。只依据【候选人证据】判断【简历】里是否有无证据支撑或明显夸大的"
                "具体主张。逐条列出有问题的原句（每行一句、不解释）;全部有支撑则只回 NONE。"},
            {"role": "user", "content": f"【候选人证据】\n{evidence}\n\n【简历】\n{resume_md}"},
        ]
        out = call_llm_plain(messages, api_key, max_tokens=800) or ""
        if out.strip().upper().replace(".", "") in ("NONE", ""):
            return []
        flags = []
        for line in out.splitlines():
            s = line.strip("-•* \t")
            if s and s.upper() != "NONE":
                flags.append({"kind": "semantic_overclaim", "claim": s[:120]})
        return flags[:10]
    except Exception:
        return None


def critic_report(resume_md: str, jd_text: str = "", profile: dict = None,
                  material: str = "", *, jd_keywords: list = None,
                  use_llm: bool = False) -> dict:
    """独立审查主入口。``profile`` 统一为 dict（蓝图 §6.6）。

    返回契约:``keyword_coverage`` / ``fabrication_flags`` / ``semantic_flags`` / ``verdict``。
    verdict ∈ {pass, needs_fix, reject},**代码聚合**:编造硬拦=reject（LLM 不得干预）;
    覆盖太低 **或** LLM 标了语义问题 → needs_fix（建议改写,软动作）。
    """
    kws = jd_keywords
    if kws is None:
        from jd_parser import analyze_jd
        kws = analyze_jd(jd_text, mode="deterministic").model_dump(mode="json")["keywords"]
    cov = keyword_coverage(resume_md, kws)
    fab = fabrication_flags(resume_md)

    semantic = None                       # 腿2(LLM):默认关 → None 降级
    if use_llm:
        evidence = _profile_to_evidence(profile or {})
        if material:
            evidence += "\n\n【已确认项目材料】\n" + str(material)[:5000]
        semantic = _semantic_fabrication_check(resume_md, evidence)

    if fab:
        verdict = "reject"                # 编造是硬边界,代码直接打回(LLM 不参与此判定)
    elif cov["coverage"] < _MIN_COVERAGE or semantic:
        verdict = "needs_fix"             # 覆盖过低 或 LLM 标了语义问题 → 建议改写
    else:
        verdict = "pass"
    return {"keyword_coverage": cov, "fabrication_flags": fab,
            "semantic_flags": semantic, "verdict": verdict}


def build_resume_critic_messages(*, resume_spec: dict, resume_md: str,
                                 context: dict, hard_report: dict,
                                 review_contract: dict,
                                 previous_report: dict | None = None,
                                 final_review: bool = False) -> list[dict]:
    """Build the independent Resume Critic Agent prompt.

    Deterministic findings are evidence supplied to the agent. They are not the
    agent itself and cannot be overridden by its qualitative verdict.
    """
    import json
    from review_protocol import CriticOutput, evidence_registry

    evidence = [{"evidence_ref": key, "content": value[:5000]}
                for key, value in evidence_registry(context).items()]
    artifact_scope = str(context.get("resume_scope") or "full_resume")
    scope_rule = (
        "这是完整简历：检查章节取舍、整体说服力与 JD 针对性。"
        if artifact_scope == "full_resume" else
        "这是单个项目经历段：检查项目事实、职责边界、技术取舍和成果表达；"
        "若未绑定 JD，只审查通用表达，不得要求虚构岗位针对性。"
    )
    system = (
        "你是 OfferClaw Resume Critic Agent，与 Resume Agent 相互独立。"
        "你只负责评审，不得重写简历正文，也不得修改用户记忆。"
        + scope_rule +
        "请判断事实是否有证据、绑定 JD 时是否真正回应 JD、关键词是否自然、内容是否清楚具体，"
        "以及每项用户要求是否满足。hard_validation 中的 error 拥有否决权，你不得判为通过。"
        "首次评审可以细化已有 criterion 的 acceptance，也可以增加以 jd_ 开头的 JD 标准；"
        "不得删除、降级或替换固定标准，不得新增伪造的用户要求。"
        "最终复审必须沿用冻结标准；新的轻微风格建议不得阻塞通过，除非修改引入了明显回归。"
        "revision_brief 只写修改目标、位置和验收条件，不要给出一份替代简历。"
        "只返回严格 JSON。"
    )
    payload = {
        "review_phase": "final" if final_review else "initial",
        "artifact_scope": artifact_scope,
        "target": {
            "application": context.get("application") or {},
            "jd_text": str((context.get("jd") or {}).get("jd_text") or "")[:7000],
            "jd_analysis": context.get("jd_analysis") or {},
        },
        "resume_spec": resume_spec,
        "resume_markdown": resume_md,
        "candidate_evidence": evidence,
        "hard_validation": hard_report,
        "review_contract": review_contract,
        "previous_review": previous_report,
        "output_schema": CriticOutput.model_json_schema(),
    }
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def run_resume_critic_agent(*, resume_spec: dict, resume_md: str,
                            context: dict, hard_report: dict,
                            review_contract: dict, artifact_revision: int,
                            caller, previous_report: dict | None = None,
                            final_review: bool = False) -> dict:
    """Invoke and validate one real Resume Critic Agent reasoning step."""
    from review_protocol import finalize_review, parse_critic_output

    raw = caller(build_resume_critic_messages(
        resume_spec=resume_spec, resume_md=resume_md, context=context,
        hard_report=hard_report, review_contract=review_contract,
        previous_report=previous_report, final_review=final_review,
    ), max_tokens=3000)
    output = parse_critic_output(raw)
    kind = ("resume_project" if str(context.get("resume_scope") or "") == "project_section"
            else "resume")
    return finalize_review(
        kind=kind, artifact_revision=artifact_revision, output=output,
        base_contract=review_contract, hard_report=hard_report,
        final_review=final_review, previous_report=previous_report,
    )
