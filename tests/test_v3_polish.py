# -*- coding: utf-8 -*-
"""4 项 V3 二阶优化的回归测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- Fix 1: application_suggestion.reason 顶层字段 ----------
def test_application_suggestion_has_top_level_reason():
    from career_flow import run_career_flow
    out = run_career_flow(
        "岗位：AI Agent 实习生\n要求：Python / LangGraph / RAG / 大模型",
        jd_title="测试岗", skip_llm=True,
    )
    sug = out["application_suggestion"]
    assert "reason" in sug and sug["reason"], (
        f"application_suggestion 缺少顶层 reason 字段：{sug}"
    )
    assert "匹配结论" in sug["reason"]


# ---------- Fix 2: Feishu/段落式 JD 兜底 (location/title/job_type) ----------
def test_extract_jd_handles_feishu_paragraph_format():
    from job_discovery import extract_jd
    feishu_like = """大模型应用开发实习生
上海｜实习｜数字技术 - 软件研发
本科及以上 / 工作年限不限

职位描述
基于 RAG 的知识增强问答系统；端到端 LLM 应用工作流。

职位要求
熟悉 Python / LangGraph / FastAPI；了解 LoRA / Embedding。
"""
    out = extract_jd(feishu_like)
    assert out["location"] == "上海", f"location 兜底失败: {out['location']!r}"
    assert out["job_type"] == "实习", f"job_type 兜底失败: {out['job_type']!r}"
    assert "大模型" in out["title"], (
        f"title 兜底失败: {out['title']!r}"
    )


def test_extract_jd_handles_ats_section_fields_and_newline_location():
    from job_discovery import extract_jd
    ats_text = """首页/职位列表/职位详情
【2027】AI算法研发工程师-【引力计划】人才专项
产品技术中心
工作地点
北京
职位描述
岗位职责：负责 Agent 与 RAG 应用开发。
任职要求：熟悉 Python。
公司信息
搜狐
"""
    out = extract_jd(ats_text)
    assert out["title"] == "【2027】AI算法研发工程师-【引力计划】人才专项"
    assert out["company"] == "搜狐"
    assert out["location"] == "北京"


# ---------- Fix 3: JD 没明确专业要求 → AI 友好专业默认通过 ----------
def test_match_major_passes_when_jd_has_no_major_constraint():
    from match_job import check_major
    profile = {"专业": "软件工程"}
    jd_no_major = (
        "岗位：AI 应用开发实习生\n职责：基于 RAG 构建知识问答\n"
        "要求：Python，LangGraph，FastAPI，了解 LoRA 与 Embedding\n"
        "硕士及以上学历"
    )
    res = check_major(profile, jd_no_major)
    assert res.status == "✓", (
        f"专业应判通过（JD 无专业限制 + 通信属 AI 友好）"
        f"，实际 {res.status} / {res.reason}"
    )


# ---------- Fix 4: applications.md DEMO 行不应污染 today_advice ----------
def test_today_advice_skips_demo_rows():
    from career_agent import get_today_advice, parse_applications, _read, APPLICATIONS_PATH
    apps_md = _read(APPLICATIONS_PATH)
    rows = parse_applications(apps_md)
    for r in rows:
        for v in r.values():
            assert "[DEMO]" not in v, (
                f"parse_applications 不应返回 [DEMO] 行：{r}"
            )
    # 单独跑 today（不带本次 JD 上下文，仅看是否会带出 [DEMO]）
    adv = get_today_advice()
    assert "[DEMO]" not in (adv.get("headline") or ""), (
        f"today_advice headline 不应包含 [DEMO]：{adv}"
    )
