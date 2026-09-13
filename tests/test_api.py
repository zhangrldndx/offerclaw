# -*- coding: utf-8 -*-
"""
tests/test_api.py — FastAPI 路由的离线 / 联网双层测试

策略：
1. 全部用 fastapi.testclient.TestClient（不起 uvicorn，跑得快）
2. 不依赖 LLM 的接口（健康、画像、根、信息、reset）→ 必跑
3. 依赖 LLM / Embedding 的接口（query、search、match、stream）→ 用 monkeypatch 打桩；
   仅在显式 export OFFERCLAW_E2E=1 时跑真调用
"""
from __future__ import annotations
import os
import sys
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rag_api  # noqa: E402

client = TestClient(rag_api.app)


# ---------- 1. 离线类（必跑） ----------

def test_root_redirects_to_ui():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui"


def test_ui_returns_html():
    r = client.get("/ui")
    assert r.status_code == 200
    assert "OfferClaw" in r.text
    assert "<html" in r.text.lower()
    assert "no-store" in r.headers.get("cache-control", "")


def test_ui_sidebar_careerflow_status_rail():
    r = client.get("/ui?rev=test")
    assert r.status_code == 200
    assert 'aria-label="CareerFlow 状态"' in r.text
    assert 'href="/ui/console?rev=latest"' in r.text
    assert 'data-flow-step="profile"' in r.text
    assert 'data-flow-step="application_suggest"' in r.text
    assert 'class="mobile-nav"' not in r.text


def test_ui_bare_path_redirects_to_versioned_html():
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].startswith("/ui?rev=")
    assert "no-store" in r.headers.get("cache-control", "")


def test_ui_console_bare_path_redirects_to_versioned_html():
    r = client.get("/ui/console", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].startswith("/ui/console?rev=")
    assert "no-store" in r.headers.get("cache-control", "")


def test_api_info():
    r = client.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "OfferClaw API"
    assert "/health" in body["endpoints"]["GET /health"] or body["endpoints"].get("GET /health")


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "degraded")
    assert "collection_records" in body
    assert isinstance(body["collection_records"], int)
    assert body["agent_workflows"]["status"] == "ready"
    assert body["agent_workflows"]["checkpoint"] == "sqlite"


def test_agent_start_reports_missing_checkpoint_before_opening_stream(monkeypatch):
    import career_multi_agent

    monkeypatch.setattr(career_multi_agent, "agent_runtime_health", lambda: {
        "status": "unavailable",
        "checkpoint": "sqlite",
        "dependency": "langgraph-checkpoint-sqlite==3.1.1",
        "detail": "simulated missing dependency",
    })
    response = client.post("/api/agent/flows/start", json={
        "task": "resume",
        "application_id": "app_missing",
        "jd_version_id": "jdv_missing",
    })
    assert response.status_code == 503
    assert "langgraph-checkpoint-sqlite==3.1.1" in response.json()["detail"]


def test_legacy_project_stream_delegates_to_reviewed_resume_workflow(monkeypatch):
    import career_multi_agent

    calls = []
    monkeypatch.setattr(career_multi_agent, "start_agent_flow", lambda **kwargs: (
        calls.append(kwargs) or {
            "status": "waiting_approval",
            "thread_id": "agent_project_review",
            "artifact": {
                "kind": "resume_project", "status": "ready_for_approval",
                "content_md": "# 项目经历\n\n- 经审查的项目内容", "saved_path": "",
            },
            "interrupt": {"type": "approval_required"},
        }
    ))

    response = client.post("/api/resume/project/stream", json={
        "text": "这是足够长的真实项目素材，包含 Python、RAG、FastAPI、检索评测和人工审批流程。" * 2,
        "project_name": "OfferClaw",
        "jd_text": "不应作为临时定制依据",
        "stage_memory": True,
    })

    assert response.status_code == 200
    body = response.text
    assert '"deprecated_endpoint": true' in body
    assert '"resume_scope": "project_section"' in body
    assert '"type": "artifact"' in body
    assert '"type": "interrupt"' in body
    assert calls[0]["task"] == "resume"
    assert calls[0]["resume_scope"] == "project_section"
    assert calls[0]["project_name"] == "OfferClaw"
    assert "jd_text" not in calls[0]


def test_profile_no_hardcoded_name():
    r = client.get("/api/profile")
    if r.status_code == 404:
        pytest.skip("user_profile.md missing")
    assert r.status_code == 200
    body = r.json()
    assert "name" in body
    assert "direction" in body and isinstance(body["direction"], list)
    # 不应再硬编码英文名
    assert body["name"] != "示例用户" or body["updated_at"] != "2026-04-21", \
        "rag_api should parse user_profile.md, not return hardcoded values"


def test_reset_returns_ok():
    r = client.post("/api/reset")
    assert r.status_code == 200


def test_missing_private_profile_has_empty_suggestions(tmp_path, monkeypatch):
    import profile_store

    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setattr(profile_store, "PROFILE_PATH", tmp_path / "missing_profile.md")
    r = client.get("/api/profile/suggestions?status=pending")
    assert r.status_code == 200
    assert r.json()["suggestions"] == []


def test_missing_daily_log_has_empty_state(tmp_path, monkeypatch):
    import summary_tool

    monkeypatch.setattr(summary_tool, "DAILY_LOG_PATH", str(tmp_path / "missing_daily_log.md"))
    r = client.get("/api/daily")
    assert r.status_code == 200
    assert r.json() == {"today_log": "", "recent_summary": "", "recent_days": 7}


def test_daily_attachment_pdf_upload_and_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_api, "DAILY_ATTACHMENT_DIR", str(tmp_path))
    r = client.post("/api/daily/attachments", json={
        "files": [{
            "name": "学习截图.pdf",
            "content_type": "application/pdf",
            "data_base64": "JVBERi0xLjQK",
        }]
    })
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["files"][0]["url"].startswith("/daily_attachments/")
    assert body["files"][0]["markdown"].endswith(")")

    fetched = client.get(body["files"][0]["url"])
    assert fetched.status_code == 200
    assert fetched.content.startswith(b"%PDF")


def test_daily_attachment_rejects_non_image_or_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_api, "DAILY_ATTACHMENT_DIR", str(tmp_path))
    r = client.post("/api/daily/attachments", json={
        "files": [{
            "name": "notes.txt",
            "content_type": "text/plain",
            "data_base64": "aGVsbG8=",
        }]
    })
    assert r.status_code == 400


def test_match_with_clearly_unsuitable_jd():
    """硬否决：Java 后端 → 应返回 暂不 / 不适合 类结论；不依赖 LLM。"""
    jd = "Java 高级开发工程师\n精通 Spring / MyBatis / Java 主线\n北京"
    r = client.post("/api/match", json={"jd_text": jd})
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "summary" in body
    # match_job 是规则版，结论字段应可读
    assert isinstance(body["status"], str) and len(body["status"]) > 0


# ---------- 2. 联网/LLM 类（默认跳过；OFFERCLAW_E2E=1 时跑） ----------

E2E = os.environ.get("OFFERCLAW_E2E") == "1"


@pytest.mark.skipif(not E2E, reason="set OFFERCLAW_E2E=1 to run LLM e2e tests")
def test_query_e2e():
    r = client.post("/api/query", json={"query": "OfferClaw 主方向是什么？", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["answer"]) > 10


@pytest.mark.skipif(not E2E, reason="set OFFERCLAW_E2E=1 to run LLM e2e tests")
def test_search_e2e():
    r = client.post("/api/search", json={"query": "硬否决规则", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body or "matches" in body or isinstance(body, dict)


@pytest.mark.skipif(not E2E, reason="set OFFERCLAW_E2E=1 to run LLM e2e tests")
def test_stream_e2e():
    """SSE：能至少收到 meta 和一条 delta。"""
    with client.stream("POST", "/api/stream",
                       json={"query": "OfferClaw 主方向是什么？", "top_k": 3,
                             "use_retrieval": True}) as resp:
        assert resp.status_code == 200
        seen_meta = False
        seen_delta = False
        for chunk in resp.iter_text():
            # 统一 wire 格式(见 rag_api._sse_event):每条 data: {"type": ...};
            # 旧断言 "event: meta" 是格式统一前的陈钉,只在 E2E 开跑时才会暴露
            # (2026-08-31 真实测试实录:UI 四次流式全正常,本钉独红)。
            if '"type": "meta"' in chunk:
                seen_meta = True
            if '"type": "delta"' in chunk:
                seen_delta = True
            if seen_meta and seen_delta:
                break
        assert seen_meta, "no meta event received"
        assert seen_delta, "no delta token received"
