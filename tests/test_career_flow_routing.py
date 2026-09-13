# -*- coding: utf-8 -*-
"""career_flow.run_career_flow_routed 的条件分支测试（V4 §2 Agentic Graph）。

验证三档结论触发三条不同的后续路径 + JD 过短直接 END。
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytestmark = pytest.mark.usefixtures("synthetic_career_profile")


JD_SUITABLE = (
    "岗位名称：AI Agent 应用开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n专业要求：计算机/AI\n经验要求：实习\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)

JD_STRETCH = (
    "岗位名称：数据分析实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科\n专业要求：不限\n经验要求：实习\n"
    "技术要求：Python / SQL / 数据处理 / 数据可视化\n工作性质：实习\n"
)

JD_NOT_RECOMMENDED = (
    "岗位名称：Java 后端开发实习\n公司：示例\n工作地点：上海\n"
    "学历要求：本科\n经验要求：实习\n"
    "技术要求：精通 Java / SpringBoot / MySQL\n工作性质：日常实习\n"
)

JD_TOO_SHORT = "招人"


def _nodes_in_trace(out: dict) -> set:
    return {t["node"] for t in out.get("trace", [])}


# --------- 1) 适合 → 全路径 ---------

def test_routed_suitable_runs_full_path():
    from career_flow import run_career_flow_routed
    out = run_career_flow_routed(JD_SUITABLE, jd_title="AI Agent 实习",
                                 skip_llm=True)
    assert "适合" in out["match_report"]["status"], out["match_report"]
    nodes = _nodes_in_trace(out)
    # full path: 8 业务节点全跑 + router 节点至少 3 个
    assert nodes >= {"profile", "job_input", "match", "gap",
                     "plan", "today", "resume", "application_suggest"}, nodes
    assert out.get("route_taken", "").startswith("suitable")
    # 适合应该写入 requires_confirmation
    assert any("applications.md" in c.get("target_file", "")
               for c in out.get("requires_confirmation", []))


# --------- 2) 中长期 → 跳过 resume + application_suggest ---------

def test_routed_stretch_skips_resume_and_application_suggest():
    from career_flow import run_career_flow_routed
    out = run_career_flow_routed(JD_STRETCH, jd_title="数据分析实习",
                                 skip_llm=True)
    status = out["match_report"]["status"]
    assert "中长期" in status, f"期望中长期，实际 {status}"
    nodes = _nodes_in_trace(out)
    # 必须跑 gap+plan+today
    assert {"gap", "plan", "today"}.issubset(nodes)
    # 必须 NOT 跑 resume / application_suggest
    assert "resume" not in nodes, "stretch 路径不该跑 resume"
    assert "application_suggest" not in nodes, "stretch 路径不该跑 application_suggest"
    assert out.get("route_taken", "").startswith("stretch")
    # 没有 application_suggest 跑过 → 不应有 requires_confirmation 投递 patch
    assert all("applications.md" not in c.get("target_file", "")
               for c in out.get("requires_confirmation", []))


# --------- 3) 暂不建议 → 跳过 plan/today/resume，直接 application_suggest ---------

def test_routed_not_recommended_only_runs_gap_and_application_suggest():
    from career_flow import run_career_flow_routed
    out = run_career_flow_routed(JD_NOT_RECOMMENDED, jd_title="Java 后端",
                                 skip_llm=True)
    status = out["match_report"]["status"]
    assert "暂不建议" in status, f"期望暂不建议，实际 {status}"
    nodes = _nodes_in_trace(out)
    assert {"match", "gap", "application_suggest"}.issubset(nodes)
    # NOT 跑 plan / today / resume
    assert "plan" not in nodes
    assert "today" not in nodes
    assert "resume" not in nodes
    assert out.get("route_taken", "").startswith("not_recommended")
    # 决策必须是"暂不建议加入"
    assert "暂不建议" in out["application_suggestion"]["decision"]


# --------- 4) JD 过短 → 立刻 END ---------

def test_routed_too_short_jd_ends_after_job_input():
    from career_flow import run_career_flow_routed
    out = run_career_flow_routed(JD_TOO_SHORT, jd_title="bad", skip_llm=True)
    nodes = _nodes_in_trace(out)
    # match 及之后都不应跑
    assert "match" not in nodes
    assert "gap" not in nodes
    assert "plan" not in nodes
    assert "resume" not in nodes
    assert out.get("route_taken", "").startswith("jd_too_short")
    # 必须留下 job_input 的错误
    assert any(e.get("node") == "job_input" and "过短" in e.get("message", "")
               for e in out.get("errors", []))


# --------- 5) 路由器不能破坏旧的 linear flow ---------

def test_original_linear_flow_still_works():
    """V3 的 run_career_flow 必须保持不变（兼容旧测试）。"""
    from career_flow import run_career_flow
    out = run_career_flow(JD_SUITABLE, jd_title="AI Agent 实习", skip_llm=True)
    nodes = {t["node"] for t in out.get("trace", [])}
    assert nodes >= {"profile", "job_input", "match", "gap",
                     "plan", "today", "resume", "application_suggest"}
    # linear flow 没有 route_taken 字段
    assert "route_taken" not in out


# --------- 6) routed flow 主流程禁止落盘 ---------

def test_routed_writes_nothing_to_disk(tmp_path, monkeypatch):
    import applications_store
    from career_flow import run_career_flow_routed
    import profile_loader

    apps = str(tmp_path / "applications.md")
    profile = str(tmp_path / "user_profile.md")
    for path in (apps, profile):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("synthetic sentinel\n")
    monkeypatch.setattr(applications_store, "APPLICATIONS_PATH", apps)
    monkeypatch.setattr(profile_loader, "PROFILE_MD", profile)
    before = (os.path.getmtime(apps), os.path.getmtime(profile))
    run_career_flow_routed(JD_SUITABLE, skip_llm=True)
    after = (os.path.getmtime(apps), os.path.getmtime(profile))
    assert before == after, "routed CareerFlow 不应直接写状态文件"
