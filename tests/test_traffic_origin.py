# -*- coding: utf-8 -*-

import asyncio

from traffic_origin import (
    current_traffic_origin,
    normalize_traffic_origin,
    reset_traffic_origin,
    set_traffic_origin,
)


def test_normal_browser_request_defaults_to_organic():
    assert normalize_traffic_origin(None) == "organic"
    assert normalize_traffic_origin("") == "organic"


def test_automated_origins_are_explicit_and_unknown_values_fail_safe():
    assert normalize_traffic_origin("agent_generated") == "agent_generated"
    assert normalize_traffic_origin("historical_replay") == "historical_replay"
    assert normalize_traffic_origin("not-a-real-origin") == "unclassified"


def test_request_origin_context_is_reset_after_use():
    assert current_traffic_origin() == "organic"


def test_query_thread_preserves_declared_agent_origin(monkeypatch):
    import rag_api
    import rag_gate

    seen = []

    def fake_query(*_args, **_kwargs):
        seen.append(current_traffic_origin())
        return {"answer": "ok", "retrieval_count": 0, "in_kb": False}

    monkeypatch.setattr(rag_gate, "gated_query", fake_query)
    token = set_traffic_origin("agent_generated")
    try:
        asyncio.run(rag_api.rag_query(rag_api.QueryRequest(query="test")))
    finally:
        reset_traffic_origin(token)
    assert seen == ["agent_generated"]


def test_stream_generator_preserves_declared_test_origin(monkeypatch):
    import rag_api
    import rag_gate

    seen = []

    def fake_stream(*_args, **_kwargs):
        seen.append(current_traffic_origin())
        yield {"type": "done"}

    monkeypatch.setattr(rag_gate, "gated_query_stream", fake_stream)

    async def consume():
        response = await rag_api.rag_stream(rag_api.QueryRequest(query="test"))
        return [chunk async for chunk in response.body_iterator]

    token = set_traffic_origin("test")
    try:
        chunks = asyncio.run(consume())
    finally:
        reset_traffic_origin(token)
    assert seen == ["test"]
    assert chunks
    token = set_traffic_origin("test")
    try:
        assert current_traffic_origin() == "test"
    finally:
        reset_traffic_origin(token)
    assert current_traffic_origin() == "organic"
