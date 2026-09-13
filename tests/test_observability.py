# -*- coding: utf-8 -*-
"""observability.py 测试（V4 §4 Trace + 重放）。

每个 test 用独立 monkey-patched TRACE_DIR 避免污染真实 logs/traces。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


JD_AI = (
    "岗位名称：AI Agent 应用开发实习生\n公司：示例\n工作地点：上海\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)


@pytest.fixture
def tmp_trace_dir(tmp_path, monkeypatch):
    """每个用例用独立的 trace 目录，避免污染真实 logs/traces。"""
    d = tmp_path / "traces"
    d.mkdir()
    import observability as obs
    monkeypatch.setattr(obs, "TRACE_DIR", str(d))
    return str(d)


# --------- TraceWriter ---------

def test_trace_writer_creates_jsonl_file(tmp_trace_dir):
    from observability import TraceWriter

    with TraceWriter() as tw:
        tw.record({"event": "test", "k": 1})
        tw.record({"event": "test", "k": 2})
        path = tw.path
        trace_id = tw.trace_id

    assert os.path.exists(path)
    lines = [json.loads(ln) for ln in
             open(path, "r", encoding="utf-8") if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        assert ln["trace_id"] == trace_id
        assert ln["event"] == "test"
        assert "ts_iso" in ln
    assert lines[0]["k"] == 1
    assert lines[1]["k"] == 2


def test_trace_writer_after_close_raises(tmp_trace_dir):
    from observability import TraceWriter
    tw = TraceWriter()
    with pytest.raises(RuntimeError):
        tw.record({"x": 1})  # 没 __enter__


def test_trace_writer_explicit_id_round_trip(tmp_trace_dir):
    from observability import TraceWriter, read_trace
    fixed = "20990101_000000_abcdef"
    with TraceWriter(fixed) as tw:
        tw.record({"event": "x"})
    out = read_trace(fixed)
    assert out["trace_id"] == fixed
    assert out["n_events"] == 1
    assert out["events"][0]["event"] == "x"


# --------- read_trace / list_traces ---------

def test_read_trace_missing_raises(tmp_trace_dir):
    from observability import read_trace
    with pytest.raises(FileNotFoundError):
        read_trace("nonexistent_trace_id")


def test_list_traces_orders_by_mtime_desc(tmp_trace_dir):
    import time
    from observability import TraceWriter, list_traces

    ids = []
    for i in range(3):
        with TraceWriter() as tw:
            tw.record({"event": "start", "jd_title": f"jd_{i}"})
        ids.append(tw.trace_id)
        time.sleep(0.05)  # 确保 mtime 拉开

    items = list_traces(limit=10)
    assert len(items) == 3
    # 最新的在最前
    assert items[0]["trace_id"] == ids[-1]
    assert items[0]["jd_title"] == "jd_2"
    assert items[-1]["trace_id"] == ids[0]


def test_list_traces_empty_when_no_dir(tmp_path, monkeypatch):
    import observability as obs
    monkeypatch.setattr(obs, "TRACE_DIR", str(tmp_path / "nope"))
    assert obs.list_traces() == []


# --------- trace_career_flow 集成 ---------

def test_trace_career_flow_writes_node_events(tmp_trace_dir):
    from observability import trace_career_flow, read_trace
    out = trace_career_flow(JD_AI, jd_title="AI 实习")
    trace_id = out["_trace_id"]
    assert trace_id

    tr = read_trace(trace_id)
    events = tr["events"]
    # 第一条 event 必须是 start
    assert events[0]["event"] == "start"
    assert events[0]["jd_title"] == "AI 实习"
    # 最后一条必须是 end
    assert events[-1]["event"] == "end"
    assert "summary" in events[-1]
    # 中间必须有 node 事件
    node_events = [e for e in events if e["event"] == "node"]
    assert len(node_events) >= 6  # 至少 profile/job_input/match/gap/plan/...
    nodes_seen = {e["node"] for e in node_events}
    assert "profile" in nodes_seen
    assert "match" in nodes_seen


def test_trace_career_flow_for_not_recommended_path(tmp_trace_dir):
    """暂不建议路径下，summary.route_taken 必须反映分支。"""
    from observability import trace_career_flow, read_trace
    jd_java = (
        "岗位名称：Java 后端开发实习\n工作地点：上海\n"
        "技术要求：精通 Java / SpringBoot / MySQL\n工作性质：日常实习\n"
    )
    out = trace_career_flow(jd_java, jd_title="Java")
    tr = read_trace(out["_trace_id"])
    end = tr["events"][-1]
    assert end["event"] == "end"
    assert end["summary"]["route_taken"].startswith("not_recommended")


def test_trace_career_flow_for_too_short_jd(tmp_trace_dir):
    """过短 JD 也要有 trace（事件至少有 start/node/end），且记录到 jd_too_short。"""
    from observability import trace_career_flow, read_trace
    out = trace_career_flow("招人", jd_title="bad")
    tr = read_trace(out["_trace_id"])
    end = tr["events"][-1]
    assert end["event"] == "end"
    assert end["summary"]["route_taken"].startswith("jd_too_short")


# --------- API endpoints ---------

def test_api_trace_list_endpoint(tmp_trace_dir):
    from fastapi.testclient import TestClient
    from observability import TraceWriter
    from rag_api import app

    with TraceWriter() as tw:
        tw.record({"event": "start", "jd_title": "via_api"})

    client = TestClient(app)
    resp = client.get("/api/trace?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    titles = [i["jd_title"] for i in body["items"]]
    assert "via_api" in titles


def test_api_trace_detail_endpoint(tmp_trace_dir):
    from fastapi.testclient import TestClient
    from observability import TraceWriter
    from rag_api import app

    with TraceWriter() as tw:
        tw.record({"event": "start", "jd_title": "detail_check"})
        tw.record({"event": "node", "node": "test"})
        tid = tw.trace_id

    client = TestClient(app)
    resp = client.get(f"/api/trace/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == tid
    assert body["n_events"] == 2
    assert body["events"][0]["jd_title"] == "detail_check"


def test_api_trace_detail_404(tmp_trace_dir):
    from fastapi.testclient import TestClient
    from rag_api import app

    client = TestClient(app)
    resp = client.get("/api/trace/no_such_trace")
    assert resp.status_code == 404


def test_api_flow_run_traced_returns_trace_id(tmp_trace_dir):
    from fastapi.testclient import TestClient
    from rag_api import app

    client = TestClient(app)
    resp = client.post("/api/flow/run_traced", json={
        "jd_text": JD_AI, "jd_title": "traced", "skip_llm": True,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # trace 数组的最后一个元素应带 _trace_id
    last = body["trace"][-1]
    assert "_trace_id" in last
    tid = last["_trace_id"]
    # 立即去查这条 trace
    detail = client.get(f"/api/trace/{tid}")
    assert detail.status_code == 200
    assert detail.json()["n_events"] >= 5
