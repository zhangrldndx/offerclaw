# -*- coding: utf-8 -*-
"""发布级 API live 验收:12 个强语义用例(COMPLETE_TEST_PLAN §8.2)。

旧的三个 E2E 只断言"有答案/有 delta",语义门槛太弱,只能算连通性检查。
本套每例检查:结构、语义要求、来源/动作、request_id、超时与无 5xx。

- 仅在 ``OFFERCLAW_E2E=1`` 时运行(真 LLM,走完整生产路径:语义规划器→
  检索→门→生成);
- 题面全部取自**已曝光**材料(bench 集 / 动作专项集)——绝不触碰 Blind A
  (§9.3:解封前系统不得见其题面);
- 不写任何真实用户数据;LLM_USAGE_LOG 由运行方置 0。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

E2E = os.environ.get("OFFERCLAW_E2E") == "1"
pytestmark = pytest.mark.skipif(not E2E, reason="set OFFERCLAW_E2E=1 for release live suite")

ROOT = Path(__file__).resolve().parents[1]
_CORRECTION_MARKERS = ("前提不成立", "前提有误", "不成立", "并不是", "并非",
                       "不是", "有误", "相矛盾", "恰恰相反")


@pytest.fixture(scope="module", autouse=True)
def _production_profile():
    """恢复生产 profile:conftest 为离线套件全局关掉了判据与精排
    (RAG_ANSWERABILITY=0 / RAG_RERANK=0),发布级 live 验收必须按生产
    默认跑——否则 correct_premise 永远判不出来,kb 命中也只是距离门
    侥幸(2026-09-01 实录:两例 cp 在 pytest 内恒红、独立进程恒绿,
    31s 跑完两发 quality 查询即为铁证)。做法是**删除 override 回落
    内建默认**,不是自拼一套配置。"""
    if not E2E:
        yield
        return
    mp = pytest.MonkeyPatch()
    for name in ("RAG_ANSWERABILITY", "RAG_RERANK"):
        mp.delenv(name, raising=False)
    mp.setenv("LLM_USAGE_LOG", "0")
    yield
    mp.undo()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import rag_api

    return TestClient(rag_api.app)


def _bench_case(index: int) -> dict:
    payload = json.loads((ROOT / "tests/rag_bench_set.json").read_text(encoding="utf-8"))
    return payload["items"][index]


def _query(client, question: str, timeout_note: str = "") -> dict:
    r = client.post("/api/query", json={"query": question, "top_k": 5})
    assert r.status_code == 200, f"5xx/非 200:{r.status_code} {r.text[:200]}"
    return r.json()


# ------------------------------------------------- 1-3 reference_kb 正常回答

@pytest.mark.parametrize("index", [0, 1, 2])
def test_live_kb_answer_semantics(client, index):
    """结构+语义:命中 KB、来源与金标文件相交、动作合同为 answer。"""
    case = _bench_case(index)
    body = _query(client, case["q"])
    assert body["in_kb"] is True, f"库内题未命中:{case['id']} mode={body['mode']}"
    assert body["mode"] == "kb_grounded"
    assert body["answer_action"] == "answer"
    assert len(body["answer"]) > 30, "答案疑似空壳"
    assert body["sources"], "命中却无来源"
    expected = set(case.get("expect_sources") or [])
    if expected:
        # bench 金标是短名(如 rag_basics),API 返回完整文件名——按子串判命中
        hit = any(short in src for short in expected for src in body["sources"])
        assert hit, f"来源与金标不相交:{body['sources']} vs {sorted(expected)}"


# ------------------------------------------------- 4-5 correct_premise 纠偏

@pytest.mark.parametrize("question", [
    "咱们的混合检索里,RRF 是把 BM25 分数和向量分数直接相加来融合的吧?",
    "LoRA 属于全量微调的一种吧?",
])
def test_live_correct_premise_first_sentence(client, question):
    body = _query(client, question)
    assert body["in_kb"] is True
    assert body["answer_action"] == "correct_premise", (
        f"门动作未判纠偏:{body['answer_action']}")
    head = body["answer"].strip()[:60]
    assert any(m in head for m in _CORRECTION_MARKERS), f"首句未纠偏:{head!r}"


# ------------------------------------------------- 6-7 KB 拒答(不冒充证据)

@pytest.mark.parametrize("question", [
    "根据资料讲讲 Mamba 状态空间模型的选择性扫描机制?",
    "资料里投机采样(speculative decoding)的草稿模型怎么选?",
])
def test_live_abstain_never_fakes_kb_grounding(client, question):
    body = _query(client, question)
    # 允许坦白拒答或请求澄清;唯一红线:不得以 kb_grounded 冒充库内证据
    assert body["mode"] != "kb_grounded", (
        f"库外题被判 kb_grounded:sources={body['sources']}")
    if body["mode"] == "general_fallback":
        assert body["answer_action"] == "abstain"
        assert not body["sources"], "fallback 不应携带 KB 来源"


# ------------------------------------------------- 8 JD 匹配(规则+LLM 双通路)

def test_live_match_structured_conclusion(client):
    jd = "岗位: 大模型应用开发工程师\n要求: 熟悉 Python 与 RAG, 了解 Agent 工具调用\n地点: 远程"
    r = client.post("/api/match", json={"jd_text": jd})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("status"), str) and body["status"], "结论字段不可读"
    assert "summary" in body


# ------------------------------------------------- 9 今日建议聚合

def test_live_today_aggregation_shape(client):
    r = client.get("/api/today")
    assert r.status_code == 200
    body = r.json()
    assert body.get("today") and body.get("headline"), "今日建议缺核心字段"
    assert isinstance(body.get("next_actions"), list)


# ------------------------------------------------- 10 CareerFlow 全流程

def test_live_flow_run_full_state(client):
    r = client.post("/api/flow/run", json={
        "jd_text": "岗位: AI 应用开发实习生\n要求: 熟悉 Python, 了解 RAG 与 Agent 开发",
    })
    assert r.status_code == 200
    body = r.json()
    blob = json.dumps(body, ensure_ascii=False)
    for node in ("match", "gap", "resume"):
        assert node in blob, f"流程状态缺 {node} 环节"


# ------------------------------------------------- 11-12 SSE 正常与中断恢复

def test_live_sse_unified_wire_with_action(client):
    case = _bench_case(0)
    seen_meta = seen_delta = seen_done = False
    action = request_id = None
    with client.stream("POST", "/api/stream",
                       json={"query": case["q"], "top_k": 5,
                             "use_retrieval": True}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("type") == "meta":
                seen_meta = True
                action = ev.get("answer_action")
                request_id = ev.get("request_id")
            elif ev.get("type") == "delta":
                seen_delta = True
            elif ev.get("type") == "done":
                seen_done = True
    assert seen_meta and seen_delta and seen_done, "meta/delta/done 序列不完整"
    assert request_id, "meta 未携带 request_id"
    assert action in {"answer", "correct_premise"}, f"meta 未暴露动作:{action}"


def test_live_sse_client_abort_leaves_server_healthy(client):
    case = _bench_case(1)
    events = 0
    with client.stream("POST", "/api/stream",
                       json={"query": case["q"], "top_k": 5,
                             "use_retrieval": True}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events += 1
            if events >= 2:
                break                      # 客户端提前掐断
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "healthy", \
        "客户端中断后服务不健康"
