# -*- coding: utf-8 -*-
"""实时状态 RAG 快路径与中文流式编码回归。"""

import json


def test_state_stream_skips_vector_retrieval(monkeypatch):
    import rag_gate
    import rag_query_plan

    def must_not_retrieve(*args, **kwargs):
        raise AssertionError("个人投递问题不应启动向量检索/reranker")

    monkeypatch.setattr(rag_gate, "_retrieve_and_classify", must_not_retrieve)
    monkeypatch.setattr(
        rag_query_plan, "plan_query",
        lambda question, **_: rag_query_plan.rule_plan_query(question),
    )
    import applications_store
    monkeypatch.setattr(applications_store, "list_applications", lambda: [{
        "日期": "2026-08-21", "公司": "华为", "岗位": "AI 应用工程师",
        "来源": "官网", "地点": "南京", "当前状态": "准备投递",
        "下一步动作": "面试准备", "备注": "08-21 准备投递",
    }])

    events = list(rag_gate.gated_query_stream("为我统计目前准备投递的公司"))
    stage_events = [e for e in events if e["type"] == "stage"]
    payload_events = [e for e in events if e["type"] != "stage"]
    assert [e["type"] for e in payload_events] == ["meta", "delta", "done"]
    assert stage_events[0]["stage"] == "planning" and stage_events[0]["status"] == "active"
    assert any(e["stage"] == "retrieval" and e["status"] == "completed"
               for e in stage_events)
    assert any(e["stage"] == "evidence" and e["status"] == "completed"
               for e in stage_events)
    assert payload_events[0]["mode"] == "state_grounded"
    assert payload_events[0]["matched_by"] == "multi_route"
    assert "华为" in payload_events[1]["text"] and "面试准备" in payload_events[1]["text"]


def test_live_state_includes_full_application_and_experience(monkeypatch):
    import applications_store
    import gap_store
    import plan_gen
    import rag_gate
    import summary_tool

    monkeypatch.setattr(plan_gen, "summarize_plan_for_automation", lambda: {"has_plan": False})
    monkeypatch.setattr(summary_tool, "extract_recent_blocks", lambda *args, **kwargs: "")
    monkeypatch.setattr(gap_store, "summary", lambda: {})
    monkeypatch.setattr(applications_store, "list_applications", lambda: [{
        "日期": "2026-08-21", "公司": "华为计算产品线", "岗位": "AI应用工程师",
        "来源": "官网", "地点": "南京", "匹配结论": "适合", "样本定位": "Agent工程",
        "当前状态": "准备投递", "下一步动作": "面试前准备", "备注": "简历 v2",
    }])
    monkeypatch.setattr(applications_store, "list_experiences", lambda: [{
        "company": "华为计算产品线", "position": "AI应用工程师", "stage": "机考通过",
        "date": "2026-08-21", "summary": "两轮技术面需要准备手撕与八股。",
    }])

    block = rag_gate._live_state_block(max_chars=5000)
    for expected in ("面试前准备", "官网", "南京", "简历 v2", "机考通过", "手撕与八股"):
        assert expected in block


def test_chat_stream_forces_utf8_when_sse_omits_charset(monkeypatch):
    import day1_api_starter
    import rag_gate

    class FakeResponse:
        encoding = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, decode_unicode=False):
            payload = "data: " + json.dumps({
                "choices": [{"delta": {"content": "目前准备投递华为"}}]
            }, ensure_ascii=False)
            raw = payload.encode("utf-8")
            yield raw.decode(self.encoding or "latin-1") if decode_unicode else raw

    response = FakeResponse()
    monkeypatch.setattr(day1_api_starter, "load_local_env", lambda: None)
    monkeypatch.setattr(day1_api_starter, "get_llm_config", lambda: {
        "api_key": "test", "is_zhipu": False, "model": "test-model",
        "api_base": "https://example.invalid/v1",
    })
    monkeypatch.setattr(day1_api_starter, "chat_completion", lambda *args, **kwargs: response)

    assert list(rag_gate._chat_stream([{"role": "user", "content": "x"}])) == ["目前准备投递华为"]
    assert response.encoding == "utf-8"
