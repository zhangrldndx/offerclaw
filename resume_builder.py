# -*- coding: utf-8 -*-
"""Resume authoring for OfferClaw.

The legacy functions build a lightweight JD-specific project section for the
workshop UI. The v2 Review workflow uses ``build_resume_agent_messages`` and
``parse_resume_agent_output``: one Resume Agent owns the complete structured
resume across initial drafting and revisions, while the separate Critic Agent
reviews it in ``resume_critic.py``.
"""

import json
import os
from typing import Dict

from plan_gen import call_llm_plain  # 复用智谱 chat 通道


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_current_metrics() -> dict:
    """从 metrics.json 单一真源读当前指标——避免"项目事实清单"里的数字再次漂移。

    历史 bug：此处曾硬编码 118 chunks / pytest 37 / 15 路由 / Recall@5 0.96 等远古值，
    导致简历生成器长期给 LLM 喂过期事实（系统性引用旧数据）。改为运行时读 metrics.json
    的当前口径；读失败时回退到当前真值（而非旧值），保证 fail 也不倒退。
    """
    fallback = {"chunks": "3340", "pytest": "442", "routes": "46", "R@1": "86",
                "Recall@5": "0.98", "MRR": "0.905", "eval_set": "100",
                "realworld_set": "52", "realworld_R1": "48", "doctor_ok": "12"}
    try:
        with open(os.path.join(BASE_DIR, "metrics.json"), encoding="utf-8") as f:
            cur = json.load(f).get("current", {})
        return {k: str(cur.get(k, v)) for k, v in fallback.items()}
    except Exception:
        return fallback


def _build_project_facts() -> str:
    m = _load_current_metrics()
    return f"""\
========== OfferClaw 项目事实清单（必须严格基于这些事实，不得新增功能/数字）==========

定位：本地长期运行的求职作战 AI Agent（不是 to-C 产品 / 不是 SaaS）

技术栈（严格命名）：
- Python / FastAPI / Pydantic
- LLM 网关（统一 chat + tools 通道，含容错重试与降级）
- Embedding：bge-base-zh-v1.5（768 维，本地）
- 向量库：ChromaDB（本地持久化，{m['chunks']} chunks）
- 工作流：LangGraph（state machine）
- 检索：混合召回（BM25 + 向量）+ RRF 融合 + bge-reranker 精排 + 双证据门控
- 评估：自建 {m['eval_set']} 题 RAG 评估集 + {m['realworld_set']} 题 held-out 口语集（自评口径，非公开 benchmark）

核心能力（已落地的）：
1. Prompt 契约层：source_policy / target_rules / plan_prompt / summary_prompt
2. 用户画像层：user_profile.md + /api/profile（已去硬编码）
3. 双通路岗位匹配：规则 (match_job.py) + LLM (plan_gen.py)
4. 4 周路线规划 / 每日复盘 / 周复盘
5. RAG + ChromaDB（{m['chunks']} chunks，同分布 R@1 = {m['R@1']}% / Recall@5 = {m['Recall@5']} / MRR = {m['MRR']}；held-out 真实口径 R@1 = {m['realworld_R1']}%）
6. LangGraph 工作流编排
7. FastAPI 服务（{m['routes']} 路由，含 SSE 流式 + MCP Server）
8. 6 卡片本地求职控制台（/ui）
9. 顶层 Orchestrator（career_agent.py），状态机驱动"今日建议"
10. 半自动 JD 抽取（job_discovery.py，规则 + 关键字命中，禁自动爬虫）

工程健康：
- pytest {m['pytest']} 通过 + 47 故障注入测试
- doctor.py {m['doctor_ok']} 项检查 + verify_docs 口径门禁
- 跨天/跨会话执行追踪 + 断点续跑

不允许编造的内容：
- 不得说"用户上千人"或"线上服务"
- 不得说"对接了 LinkedIn / 智联 / Boss" 等爬虫
- 不得说"自动投递 / 自动跟进"
- 不得说用了 React / Vue / 数据库 / 登录系统
"""


_PROJECT_FACTS = _build_project_facts()


def build_messages(jd_summary: str, profile: str,
                   jd_analysis: dict | None = None) -> list:
    # [L2] 硬截 profile[:3000] 会无声丢后段章节（科研/竞赛等）→ 改为按 JD 相关性挑章节 + 5000 字预算
    from context_budget import select_relevant_sections
    if not jd_analysis:
        from jd_parser import analyze_jd
        jd_analysis = analyze_jd(jd_summary).model_dump(mode="json")
    keyword_names = [
        str(item.get("canonical_name") or "").strip()
        for item in (jd_analysis.get("keywords") or []) if isinstance(item, dict)
    ]
    profile_view = select_relevant_sections(
        profile, keywords=[value for value in keyword_names if value], max_chars=5000,
        always_keep=("基础信息", "元信息", "教育", "技能", "方向"))
    analysis_view = {
        "schema_version": jd_analysis.get("schema_version", ""),
        "requirements": [
            {key: item.get(key) for key in ("text", "kind", "modality", "priority")}
            for item in (jd_analysis.get("requirements") or []) if isinstance(item, dict)
        ],
        "keywords": keyword_names,
    }
    system = (
        "你是 OfferClaw 的 简历定制 助手。\n"
        "你只能基于下面给出的【项目事实清单】来重写项目段，不得编造新功能。\n"
        "你的任务：根据这份 JD，重排和强调对该岗位最相关的能力，生成简历项目段。\n\n"
        + _PROJECT_FACTS +
        "\n========== 候选人画像（user_profile.md · JD 相关章节）==========\n"
        + profile_view
    )
    user = (
        "请根据下面这份 JD 关键信息，输出针对此 JD 的 OfferClaw 项目段。\n"
        "输出格式（严格 markdown）：\n\n"
        "## 简历 bullet 版（3-5 条，每条 ≤ 50 字，命中 JD 关键词）\n"
        "- ...\n\n"
        "## 简历段落版（一段话 ≤ 300 字，含量化指标 + 技术栈）\n"
        "...\n\n"
        "## 命中分析\n"
        "- 命中的 JD 关键词：...\n"
        "- 主动强调的项目能力：...\n"
        "- 刻意弱化的能力（与该 JD 无关）：...\n\n"
        "========== JD 关键信息 ==========\n"
        f"{jd_summary}\n\n"
        "========== 统一 JDAnalysis（CareerFlow / Resume / Critic 共用）==========\n"
        f"{json.dumps(analysis_view, ensure_ascii=False)}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_resume_agent_messages(context: dict, *, base_resume_md: str,
                                requirements: list[dict] | None = None,
                                current_spec: dict | None = None,
                                review_report: dict | None = None) -> list[dict]:
    """Build the v2 Resume Agent prompt for both drafting and revision.

    The Resume Agent owns the artifact. It receives the Critic's findings when
    revising, but never receives authority to decide whether its own output
    passes review.
    """
    from review_protocol import ResumeSpec, evidence_registry

    registry = evidence_registry(context)
    evidence = [{"evidence_ref": key, "content": value[:5000]}
                for key, value in registry.items()]
    app = context.get("application") or {}
    artifact_scope = str(context.get("resume_scope") or "full_resume")
    schema = ResumeSpec.model_json_schema()
    mode = "revision" if current_spec else "initial_draft"
    scope_instruction = (
        "输出完整、可直接使用的简历，至少包含 section_id=summary、skills、project；"
        "证据中存在教育信息时还必须包含 education。"
        if artifact_scope == "full_resume" else
        "只输出一段可直接放入简历的项目经历，必须使用 section_id=project；"
        "不要生成求职摘要、技能清单、教育经历或整份简历。"
    )
    system = (
        "你是 OfferClaw Resume Agent，是简历产物的唯一作者。"
        "你负责选择、组织和改写内容，但无权评价自己是否通过，也不得输出评审结论。"
        "只能使用 evidence_bundle 中的事实；每个 paragraph 或 bullet 只表达一个主要主张，"
        "并填写稳定 claim_id 与有效 evidence_refs。"
        "不得把 JD 要求写成候选人已经具备的经历，不得编造数字、组织、用户量或线上效果。"
        + scope_instruction +
        "artifact_scope 必须与输入的 output_scope 完全一致。"
        "不要输出命中分析、解释、评语或 Markdown。"
        "修改模式下保留不受要求影响的稳定 block_id、claim_id 和 evidence_refs，只修改必要章节。"
        "只返回严格 JSON，结构必须符合给定 JSON Schema。"
    )
    payload = {
        "mode": mode,
        "output_scope": artifact_scope,
        "target": {
            "application_id": app.get("application_id", ""),
            "company": app.get("company", ""),
            "position": app.get("position", ""),
            "jd_version_id": app.get("jd_version_id", ""),
            "jd_text": str((context.get("jd") or {}).get("jd_text") or "")[:7000],
            "jd_analysis": context.get("jd_analysis") or {},
        },
        "evidence_bundle": evidence,
        "active_memory": context.get("memory_context") or {},
        "user_requirements": requirements or [],
        "base_resume_for_structure_only": base_resume_md[:9000],
        "current_resume_spec": current_spec,
        "critic_review": review_report,
        "output_schema": schema,
    }
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def parse_resume_agent_output(raw: str, *, context: dict,
                              previous_spec: dict | None = None) -> dict:
    """Parse a Resume Agent response into the canonical ResumeSpec."""
    import re
    from review_protocol import ResumeSpec

    text = str(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Resume Agent 没有返回结构化 ResumeSpec")
    parsed = ResumeSpec.model_validate(json.loads(text[start:end + 1]))
    if previous_spec:
        old_blocks = [block
                      for section in ResumeSpec.model_validate(previous_spec).sections
                      for block in section.blocks]
        old_ids = {block.block_id for block in old_blocks}
        old_claim_ids = {block.claim_id for block in old_blocks if block.claim_id}
        new_ids = {block.block_id for section in parsed.sections for block in section.blocks}
        new_claim_ids = {block.claim_id for section in parsed.sections
                         for block in section.blocks if block.claim_id}
        if old_ids and not (old_ids & new_ids):
            raise ValueError("Resume Agent 修改时丢失了全部稳定 block_id")
        if old_claim_ids and not (old_claim_ids & new_claim_ids):
            raise ValueError("Resume Agent 修改时丢失了全部稳定 claim_id")
    return parsed.model_dump(mode="json")


def build_resume_for_jd(jd_summary: str, profile_path: str = None,
                        jd_analysis: dict | None = None) -> Dict[str, str]:
    """生成 JD 定制简历项目段。"""
    from day1_api_starter import get_llm_config

    cfg = get_llm_config()
    api_key = cfg["api_key"]
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} 未配置（.env.local 或环境变量）")
    profile = _read(profile_path or os.path.join(BASE_DIR, "user_profile.md"))
    messages = build_messages(
        jd_summary=jd_summary, profile=profile, jd_analysis=jd_analysis,
    )
    try:
        md = call_llm_plain(messages, api_key, max_tokens=2000)
    except Exception as e:  # A2: LLM 失败返回结构化 error 而非崩，调用方可优雅呈现
        from day1_api_starter import llm_error_detail
        return {"status": "error", "error": llm_error_detail(e), "jd_summary_chars": len(jd_summary)}
    return {"resume_md": md, "jd_summary_chars": len(jd_summary)}


# =====================================================================
# Phase 5：Resume Builder 完整化 —— Markdown 简历草稿（无 LLM 也能跑）
# =====================================================================

def _grab_section(text: str, header_pattern: str, max_chars: int = 1500) -> str:
    """从 markdown 中按"## 头"截取一段，截到下一个同级标题。"""
    import re
    m = re.search(rf"(^|\n)#{{1,3}}\s*{header_pattern}[^\n]*\n([\s\S]*?)(?=\n#{{1,3}}\s|\Z)",
                  text, re.IGNORECASE)
    if not m:
        return ""
    body = m.group(2).strip()
    return body[:max_chars]


def build_skill_section(profile: dict) -> str:
    skilled = profile.get("熟练技能") or []
    used = profile.get("会用技能") or []
    out = ["## 技能栏"]
    if skilled:
        out.append("- **熟练**：" + " · ".join(skilled))
    if used:
        out.append("- **会用**：" + " · ".join(used))
    if not skilled and not used:
        out.append("- *待补充*")
    return "\n".join(out)


def build_summary_section(profile: dict) -> str:
    name = profile.get("姓名") or "候选人"
    school = profile.get("学校") or ""
    major = profile.get("专业") or ""
    grad = profile.get("毕业时间") or ""
    direction = "/".join(profile.get("方向优先级") or []) or "AI / LLM 应用方向"
    cities = "/".join(profile.get("可接受地域") or []) or ""
    line = f"{name}，{school}{major}（{grad}）。求职方向：{direction}"
    if cities:
        line += f"，可工作地：{cities}"
    return f"## 求职摘要\n{line}。"


def build_project_section(text_project_status: str, text_one_pager: str) -> str:
    """从 PROJECT_STATUS.md / docs/project_one_pager.md 提取 OfferClaw 项目段。"""
    section = _grab_section(text_one_pager, r"(项目|核心能力|Project)", max_chars=1200)
    if not section:
        section = _grab_section(text_project_status, r"(已完成|当前进度|核心能力|项目)", max_chars=1200)
    body = section or "*未在 PROJECT_STATUS.md / project_one_pager.md 中找到可用段落，请先填写。*"
    return "## 项目经历 — OfferClaw\n" + body


def build_competition_section(text_profile: str) -> str:
    section = _grab_section(text_profile, r"(竞赛|比赛|Competition)")
    return "## 竞赛经历\n" + (section or "*暂无 — 在 user_profile.md 增加 `## 竞赛` 章节即可被识别。*")


def build_research_section(text_profile: str) -> str:
    section = _grab_section(text_profile, r"(科研|Research|论文|Publication)")
    return "## 科研经历\n" + (section or "*暂无 — 在 user_profile.md 增加 `## 科研` 章节即可被识别。*")


def build_jd_tailored_section(jd_summary: str = "") -> str:
    """JD 定制项目段：skip_llm=True 时返回结构骨架。"""
    if not jd_summary.strip():
        return "## JD 定制项目段（占位）\n*未提供 JD，跳过定制段。可调用 `/api/resume/build` 走 LLM 生成。*"
    return ("## JD 定制项目段（骨架）\n"
            "- 候选 JD 关键词命中：参见 `/api/match`\n"
            "- 建议用 `/api/resume/build`（需当前 LLM API Key）生成 STAR 段落。\n"
            f"- JD 摘要长度：{len(jd_summary)} 字符。")


def build_resume_markdown(jd_text: str = "", profile_path: str | None = None,
                          skip_llm: bool = True) -> dict:
    """聚合生成一份完整 Markdown 简历草稿（默认无 LLM）。"""
    from profile_loader import load_profile
    profile = load_profile(profile_path)
    text_profile = _read(profile_path or os.path.join(BASE_DIR, "user_profile.md"))
    text_status = _read(os.path.join(BASE_DIR, "PROJECT_STATUS.md"))
    text_one_pager = _read(os.path.join(BASE_DIR, "docs", "project_one_pager.md"))

    sections = [
        build_summary_section(profile),
        build_skill_section(profile),
        build_project_section(text_status, text_one_pager),
        build_competition_section(text_profile),
        build_research_section(text_profile),
        build_jd_tailored_section(jd_text),
    ]
    md = "\n\n".join(sections) + "\n"

    out = {
        "resume_md": md,
        "sections": ["summary", "skills", "project", "competition", "research", "jd_tailored"],
        "skip_llm": skip_llm,
        "jd_chars": len(jd_text or ""),
    }
    if not skip_llm and jd_text.strip():
        try:
            tailored = build_resume_for_jd(jd_text, profile_path=profile_path)
            out["resume_md"] = md + "\n\n## JD 定制项目段（LLM 生成）\n" + tailored.get("resume_md", "")
            out["llm_used"] = True
        except Exception as e:
            out["llm_error"] = str(e)
            out["llm_used"] = False
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "markdown":
        print(build_resume_markdown(skip_llm=True)["resume_md"])
    else:
        jd = sys.stdin.read() if not sys.stdin.isatty() else "公司：测试\n岗位：LLM Agent 实习\n要求：Python / RAG / FastAPI / LangGraph"
        print(json.dumps(build_resume_for_jd(jd), ensure_ascii=False, indent=2))
