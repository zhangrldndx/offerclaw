# -*- coding: utf-8 -*-
"""
OfferClaw · 核心逻辑回归测试

覆盖：
    1. match_job: DEMO 用例输出三档结论
    2. match_job: 缺口清单 dict 结构正确
    3. match_job: format_report 不崩
    4. rag_tools: Markdown 分块切片基本性质
    5. rag_tools: 太短的内容被过滤
    6. rag_tools: JWT 生成（需 API Key）
    7. pipeline: gaps_to_text 序列化正确
    8. summary_tool: extract_date_block 抓块逻辑
    9. summary_tool: extract_date_block 缺失日期返回空
"""
import os
import sys
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


# ---------- match_job ----------

def test_match_demo_runs():
    from match_job import run_match, DEMO_PROFILE, DEMO_JD
    report = run_match(DEMO_PROFILE, DEMO_JD, jd_title="UnitTest")
    assert report.conclusion in ("当前适合投递", "当前暂不建议投递", "中长期可转向")
    assert report.jd_title == "UnitTest"
    assert isinstance(report.hard_gate, list) and len(report.hard_gate) >= 5


def test_match_gap_list_structure():
    from match_job import run_match, DEMO_PROFILE, DEMO_JD
    report = run_match(DEMO_PROFILE, DEMO_JD, jd_title="UnitTest")
    assert isinstance(report.gap_list, dict)
    assert all(isinstance(v, list) for v in report.gap_list.values())


def test_match_format_report_no_crash():
    from match_job import run_match, format_report, DEMO_PROFILE, DEMO_JD
    text = format_report(run_match(DEMO_PROFILE, DEMO_JD, jd_title="X"))
    assert "结论：" in text and "缺口清单" in text


# ---------- rag_tools ----------

def test_split_markdown_basic():
    from rag_tools import split_markdown_document
    md = "# 标题\n" + ("内容段落。" * 200)
    chunks = split_markdown_document(md)
    assert len(chunks) >= 1
    assert all("text" in c and "metadata" in c for c in chunks)
    assert all("char_len" in c["metadata"] for c in chunks)


def test_split_markdown_filters_tiny():
    from rag_tools import split_markdown_document
    chunks = split_markdown_document("# T\n短")
    assert len(chunks) == 0


@pytest.mark.skipif(not os.environ.get("ZHIPU_API_KEY"), reason="无 API Key")
def test_jwt_token_generates():
    from rag_tools import generate_zhipu_token
    token = generate_zhipu_token()
    assert isinstance(token, str) and token.count(".") == 2


# ---------- pipeline ----------

def test_pipeline_gaps_to_text():
    from pipeline import gaps_to_text

    class Fake:
        gap_list = {"硬门槛缺口": ["地域不符"], "技能缺口": []}

    out = gaps_to_text(Fake())
    assert "## 硬门槛缺口" in out and "地域不符" in out
    assert "（无）" in out


# ---------- summary_tool ----------

def test_extract_date_block():
    from summary_tool import extract_date_block
    log = (
        "## 2026-04-25 · 测试块\n内容A\n更多A\n\n"
        "## 2026-04-26 · 另一天\n内容B\n"
    )
    b = extract_date_block(log, "2026-04-25")
    assert "内容A" in b and "内容B" not in b


def test_extract_date_block_missing():
    from summary_tool import extract_date_block
    assert extract_date_block("空内容", "2026-04-25") == ""
