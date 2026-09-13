"""产品级 Agent 化指导 §7-§10 验收测试。

覆盖：
- Phase 4: query_builder + rank_candidates (job_discovery)
- Phase 5: build_resume_markdown (resume_builder)
- Phase 6: rag_ingest 配置含 verification source_type
- Phase 7: CareerFlow 能回答 4 个真实使用核心问题
- 新端点：/api/jd/queries, /api/jd/rank, /api/resume/markdown, /ui/console
"""

from __future__ import annotations

import os
import json
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _public_demo_profile():
    path = os.path.join(ROOT, "profiles", "p1_demo_ai.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


JD_AI = (
    "岗位名称：大模型应用开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n专业要求：计算机、人工智能\n经验要求：实习\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)
JD_JAVA = (
    "岗位名称：Java 后端开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n经验要求：实习\n"
    "技术要求：精通 Java / SpringBoot / MySQL\n工作性质：日常实习\n"
)


# ============== Phase 4 ==============

def test_phase4_build_search_queries():
    from job_discovery import build_search_queries
    profile = _public_demo_profile()
    profile["目标岗位类型"] = "校招"
    qs = build_search_queries(profile)
    assert 1 <= len(qs) <= 6
    assert all(isinstance(q, str) and q.strip() for q in qs)
    assert any("实习" in q or "校招" in q for q in qs), qs


def test_phase4_rank_candidates_orders_by_status():
    from job_discovery import rank_candidates
    profile = _public_demo_profile()
    out = rank_candidates([
        {"title": "Java 后端", "jd_text": JD_JAVA},
        {"title": "AI 应用开发", "jd_text": JD_AI},
    ], profile=profile)
    assert len(out) == 2
    assert out[0]["score"] >= out[1]["score"]
    titles = [r["title"] for r in out]
    assert titles[0] == "AI 应用开发", f"AI JD 应排在 Java 之前：{out}"


def test_phase4_rank_candidates_analyzes_once_and_passes_contract(monkeypatch):
    from types import SimpleNamespace

    import jd_parser
    import match_job
    from job_discovery import rank_candidates

    analyzed = []
    passed = []

    class FakeAnalysis:
        def __init__(self, text):
            self.text = text

        def model_dump(self, mode="json"):
            return {"input_hash": self.text, "schema_version": "test",
                    "keywords": [], "requirements": []}

    monkeypatch.setattr(jd_parser, "analyze_jd", lambda text: (
        analyzed.append(text) or FakeAnalysis(text)
    ))
    monkeypatch.setattr(match_job, "run_match", lambda *_a, **kwargs: (
        passed.append(kwargs.get("jd_analysis")) or SimpleNamespace(
            conclusion="中长期可转向", direction="主方向",
            gap_list={}, conclusion_reason="ok",
        )
    ))
    out = rank_candidates([
        {"title": "A", "jd_text": "JD-A"},
        {"title": "B", "jd_text": "JD-B"},
    ], profile={})
    assert len(out) == 2
    assert analyzed == ["JD-A", "JD-B"]
    assert [item["input_hash"] for item in passed] == ["JD-A", "JD-B"]


# ============== Phase 5 ==============

def test_phase5_build_resume_markdown_skip_llm(monkeypatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    from resume_builder import build_resume_markdown
    out = build_resume_markdown(
        jd_text="",
        profile_path=os.path.join(ROOT, "tests", "fixtures", "missing-profile.md"),
        skip_llm=True,
    )
    md = out["resume_md"]
    for must in ("## 求职摘要", "## 技能栏", "## 项目经历", "## 竞赛经历", "## 科研经历"):
        assert must in md, f"简历草稿缺：{must}"
    assert out["sections"] == ["summary", "skills", "project", "competition", "research", "jd_tailored"]
    assert out["skip_llm"] is True


# ============== Phase 6 ==============

def test_phase6_rag_ingest_has_verification_source_type():
    from rag_ingest import DEFAULT_FILES
    paths = {p for p, _ in DEFAULT_FILES}
    types = {t for _, t in DEFAULT_FILES}
    assert "docs/verification_report.md" in paths
    assert "jd_candidates.md" not in paths
    for needed in ("profile", "log", "application", "story",
                   "resume", "verification"):
        assert needed in types, f"DEFAULT_FILES 缺 source_type={needed}"
    assert "jd" not in types


# ============== Phase 7：4 个核心问题 ==============

def test_phase7_careerflow_answers_4_core_questions():
    """指导文档 §10.3 验收：CareerFlow 能回答 4 个真实使用问题。"""
    from career_flow import run_career_flow
    out = run_career_flow(JD_AI, jd_title="Phase7 验收 AI", skip_llm=True)

    # Q1: 过去一周做了什么？ → today_advice 应能引用 daily_log / applications
    today = out["today_advice"]
    assert today and (today.get("headline") or today.get("detail") or today.get("next_action"))

    # Q2: 哪个岗位现在最值得投？ → match_report 给三档结论 + direction
    mr = out["match_report"]
    assert mr["status"] in {
        "当前适合投递", "中长期可转向",
        "信息不足，建议补充后再判断", "当前暂不建议投递"}
    assert mr["direction"]

    # Q3: 我今天最该做什么？ → today_advice.headline + 计划首周
    plan = out["plan_outline"]
    assert plan and len(plan) >= 1

    # Q4: 我的简历还缺哪一段？ → gaps + resume_skeleton
    gaps = out["gaps"]
    resume = out["resume_skeleton"]
    assert isinstance(gaps, dict)
    assert resume["mode"] == "skeleton"


# ============== 新端点 e2e ==============

@pytest.fixture(scope="module")
def client():
    from rag_api import app
    return TestClient(app)


def test_endpoint_jd_queries(client):
    r = client.get("/api/jd/queries")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["queries"], list) and len(body["queries"]) >= 1
    assert isinstance(body["profile_cities"], list)


def test_endpoint_jd_rank(client):
    r = client.post("/api/jd/rank", json={"candidates": [
        {"title": "Java 后端", "jd_text": JD_JAVA},
        {"title": "AI 应用开发", "jd_text": JD_AI},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    assert body["ranked"][0]["title"] == "AI 应用开发"


def test_endpoint_resume_markdown(client):
    r = client.post("/api/resume/markdown", json={"jd_text": "", "skip_llm": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "## 求职摘要" in body["resume_md"]
    assert body["skip_llm"] is True
    assert body["llm_used"] is False
    assert len(body["sections"]) == 6


def test_endpoint_ui_console(client):
    r = client.get("/ui/console")
    assert r.status_code == 200
    assert "CareerFlow 流程".encode("utf-8") in r.content
    assert b"/api/flow/run" in r.content
    assert b'href="/ui?rev=latest"' in r.content
    assert b'href="/ui"' not in r.content
    assert b"Swagger" not in r.content
    assert b"Health" not in r.content
