"""产品级 Agent 化指导 §5 验收：CareerFlow 编排测试。"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytestmark = pytest.mark.usefixtures("synthetic_career_profile")


JD_AI = (
    "岗位名称：大模型应用开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n专业要求：计算机、人工智能\n经验要求：实习\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)

JD_JAVA = (
    "岗位名称：Java 后端开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n专业要求：计算机/软件\n经验要求：实习\n"
    "技术要求：精通 Java / SpringBoot / MySQL\n工作性质：日常实习\n"
)


# --------- run_career_flow 单元 ---------

def test_career_flow_runs_all_8_nodes_skip_llm():
    from career_flow import run_career_flow
    out = run_career_flow(JD_AI, jd_title="UnitTest AI", skip_llm=True)
    # 8 个节点都至少 trace 一次
    nodes = {t["node"] for t in out.get("trace", [])}
    assert nodes >= {"profile", "job_input", "match", "gap", "plan",
                     "today", "resume", "application_suggest"}, nodes
    # 每个关键产物都存在
    assert out["match_report"]["status"]
    assert out["match_report"]["direction"]
    assert out["plan_outline"] and len(out["plan_outline"]) == 4
    assert out["today_advice"]["headline"]
    assert out["resume_skeleton"]["mode"] == "skeleton"
    assert "decision" in out["application_suggestion"]


def test_career_flow_state_drives_decision_via_profile_loader():
    """同一个 Java JD：当 profile 'java' 在 明确不做 时，application_suggestion 不应推荐加入。"""
    from career_flow import run_career_flow
    out = run_career_flow(JD_JAVA, jd_title="Java JD")
    decision = out["application_suggestion"]["decision"]
    assert "暂不建议" in decision, f"应该不推荐 Java，但得到：{decision}"
    # 因为不推荐，requires_confirmation 不应该追加 patch
    assert all("applications.md" not in c.get("target_file", "")
               for c in out["requires_confirmation"]) or out["requires_confirmation"] == []


def test_career_flow_skip_llm_no_zhipu_key_required(monkeypatch):
    """skip_llm=True 时全程不读 ZHIPU_API_KEY。"""
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    from career_flow import run_career_flow
    out = run_career_flow(JD_AI, skip_llm=True)
    assert out["match_report"]["status"]
    # resume 节点 mode 必须是 skeleton（没动 LLM）
    assert out["resume_skeleton"]["mode"] == "skeleton"
    assert not any(e["node"] == "resume" for e in out.get("errors", []))


def test_career_flow_writes_nothing_to_disk(tmp_path, monkeypatch):
    """write 规则：CareerFlow 主流程禁止落盘。"""
    import os as _os
    import applications_store
    from career_flow import run_career_flow
    import profile_loader

    apps = str(tmp_path / "applications.md")
    profile = str(tmp_path / "user_profile.md")
    for path in (apps, profile):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("synthetic sentinel\n")
    monkeypatch.setattr(applications_store, "APPLICATIONS_PATH", apps)
    monkeypatch.setattr(profile_loader, "PROFILE_MD", profile)
    before = (_os.path.getmtime(apps), _os.path.getmtime(profile))
    run_career_flow(JD_AI, skip_llm=True)
    after = (_os.path.getmtime(apps), _os.path.getmtime(profile))
    assert before == after, "CareerFlow 不应直接写状态文件，应只产 patch"


# --------- /api/flow/run 端到端 ---------

def test_api_flow_run_endpoint_returns_full_state():
    from rag_api import app
    client = TestClient(app)
    resp = client.post("/api/flow/run", json={"jd_text": JD_AI, "skip_llm": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("match_report", "gaps", "plan_outline", "today_advice",
                "resume_skeleton", "application_suggestion",
                "requires_confirmation", "trace"):
        assert key in body, f"FlowRunResponse 缺字段：{key}"
    assert body["match_report"]["status"]
    assert len(body["plan_outline"]) == 4
    assert len(body["trace"]) >= 8
