# -*- coding: utf-8 -*-
"""Structured GPT runtime tests; all provider calls are injected/mocked."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time

import pytest
from pydantic import BaseModel, ConfigDict

import structured_llm as sl
import day1_api_starter as d1


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


@pytest.fixture(autouse=True)
def _fresh_runtime(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LLM_ONLINE_WORKERS", "2")
    monkeypatch.setenv("STRUCTURED_LLM_ONLINE_QUEUE", "4")
    monkeypatch.setenv("STRUCTURED_LLM_SHADOW_WORKERS", "1")
    monkeypatch.setenv("STRUCTURED_LLM_SHADOW_QUEUE", "2")
    monkeypatch.setenv("STRUCTURED_LLM_QUEUE_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "0")
    sl._reset_structured_runtime_for_tests()
    yield
    sl._reset_structured_runtime_for_tests()


def _messages():
    return [{"role": "user", "content": "return one integer"}]


def test_repository_default_model_is_gpt_56_terra():
    assert d1.DEFAULT_MODEL == "gpt-5.6-terra"


def test_total_deadline_is_shared_with_repair(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LLM_REPAIR_MIN_REMAINING_SECONDS", "0.01")
    calls = 0

    def caller(_messages_arg, _max_tokens, _temperature, _model):
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(0.02)
            return "not-json"
        time.sleep(0.20)
        return '{"value":1}'

    started = time.monotonic()
    value, meta = sl.call_structured(
        _Answer, _messages(), caller=caller, timeout_seconds=0.09,
        repair=True, repair_min_remaining_seconds=0.01,
    )
    elapsed = time.monotonic() - started
    assert value is None
    assert meta.calls == 2 and meta.repair_used is True
    assert "repair_timeout_or_model_unavailable" in meta.errors
    assert elapsed < 0.15, "repair must not receive a second full timeout budget"


def test_repair_is_skipped_without_minimum_remaining_budget():
    calls = 0

    def caller(_messages_arg, _max_tokens, _temperature, _model):
        nonlocal calls
        calls += 1
        return "not-json"

    value, meta = sl.call_structured(
        _Answer, _messages(), caller=caller, timeout_seconds=0.10, repair=True,
    )
    assert value is None and calls == 1
    assert meta.repair_used is False
    assert "repair_skipped_insufficient_budget" in meta.errors


def test_executor_is_bounded_and_queue_wait_fails_fast(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LLM_ONLINE_WORKERS", "1")
    monkeypatch.setenv("STRUCTURED_LLM_ONLINE_QUEUE", "1")
    sl._reset_structured_runtime_for_tests()
    release = Event()
    entered = Event()
    lock = Lock()
    active = 0
    max_active = 0

    def caller(_messages_arg, _max_tokens, _temperature, _model):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        entered.set()
        release.wait(0.25)
        with lock:
            active -= 1
        return '{"value":1}'

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(
            sl.call_structured, _Answer, _messages(), caller=caller,
            timeout_seconds=0.30, queue_timeout_seconds=0.01,
        )
        assert entered.wait(0.1)
        rest = [pool.submit(
            sl.call_structured, _Answer, _messages(), caller=caller,
            timeout_seconds=0.20, queue_timeout_seconds=0.01,
        ) for _ in range(3)]
        time.sleep(0.04)
        release.set()
        outcomes = [first.result(), *(item.result() for item in rest)]

    assert max_active == 1
    assert any("queue_timeout" in meta.errors for _value, meta in outcomes)


def test_late_provider_response_is_counted():
    def caller(_messages_arg, _max_tokens, _temperature, _model):
        time.sleep(0.06)
        return '{"value":1}'

    value, meta = sl.call_structured(
        _Answer, _messages(), caller=caller, timeout_seconds=0.02,
        queue_timeout_seconds=0.01,
    )
    assert value is None and meta.planner_late_response is True
    time.sleep(0.07)
    assert sl.structured_runtime_stats()["online_late_responses"] >= 1


def test_circuit_opens_after_two_bad_windows_and_half_open_recovers(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_WINDOW", "2")
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_FAILURE_RATIO", "0.5")
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_BAD_WINDOWS", "2")
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_OPEN_SECONDS", "30")
    sl._reset_structured_runtime_for_tests()

    def failing(_messages_arg, _max_tokens, _temperature, _model):
        raise RuntimeError("provider down")

    for _ in range(4):
        value, meta = sl.call_structured(
            _Answer, _messages(), caller=failing, timeout_seconds=0.1,
            repair=False,
        )
        assert value is None and "provider_error" in meta.errors
    value, blocked = sl.call_structured(
        _Answer, _messages(), caller=failing, timeout_seconds=0.1,
        repair=False,
    )
    assert value is None and blocked.calls == 0
    assert "circuit_open" in blocked.errors

    sl._runtime().online_breaker._open_until = time.monotonic() - 0.01
    value, recovered = sl.call_structured(
        _Answer, _messages(),
        caller=lambda *_args: '{"value":7}', timeout_seconds=0.1,
    )
    assert value and value.value == 7
    assert recovered.planner_circuit_state == "half_open"
    assert sl.structured_runtime_stats()["circuit_state"] == "closed"


def test_gpt_json_mode_and_low_reasoning_are_sent_to_provider(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gpt-gateway.example/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6")
    monkeypatch.setenv("STRUCTURED_LLM_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("STRUCTURED_LLM_JSON_CAPABILITY", "verified")
    monkeypatch.setenv("STRUCTURED_LLM_REASONING_EFFORT", "low")

    def fake_chat(_messages_arg, **kwargs):
        captured.update(kwargs)
        kwargs["meta"].update({
            "model": "gpt-5.6",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        })
        return '{"value":9}'

    monkeypatch.setattr("rag_gate._chat", fake_chat)
    value, meta = sl.call_structured(
        _Answer, _messages(), model="gpt-5.6", timeout_seconds=0.2,
    )
    assert value and value.value == 9
    assert captured["extra_payload"] == {
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }
    assert meta.model == "gpt-5.6"
    assert meta.reasoning_effort == "low"
    assert meta.structured_output_mode == "json_object"
    assert meta.prompt_tokens == 12 and meta.completion_tokens == 4


def test_structured_direct_fallback_flag_cannot_bypass_global_switch(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "0")
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "not-used")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "deepseek-chat")
    monkeypatch.setenv("STRUCTURED_LLM_USE_FALLBACK_ENDPOINT", "1")
    primary_calls = 0

    def fake_primary(_messages_arg, **_kwargs):
        nonlocal primary_calls
        primary_calls += 1
        return '{"value":5}'

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("fallback chat_completion must stay unreachable")

    monkeypatch.setattr("rag_gate._chat", fake_primary)
    monkeypatch.setattr(d1, "chat_completion", forbidden_fallback)
    value, _meta = sl.call_structured(
        _Answer, _messages(), model="gpt-5.6", timeout_seconds=0.2,
        response_format="prompt_only",
    )
    assert value and value.value == 5 and primary_calls == 1


def test_shadow_lane_does_not_wait_for_online_worker(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LLM_ONLINE_WORKERS", "1")
    monkeypatch.setenv("STRUCTURED_LLM_ONLINE_QUEUE", "0")
    monkeypatch.setenv("STRUCTURED_LLM_SHADOW_WORKERS", "1")
    monkeypatch.setenv("STRUCTURED_LLM_SHADOW_QUEUE", "0")
    sl._reset_structured_runtime_for_tests()
    release = Event()
    entered = Event()

    def blocked(_messages_arg, _max_tokens, _temperature, _model):
        entered.set()
        release.wait(0.2)
        return '{"value":1}'

    with ThreadPoolExecutor(max_workers=1) as pool:
        online = pool.submit(
            sl.call_structured, _Answer, _messages(), caller=blocked,
            timeout_seconds=0.3,
        )
        assert entered.wait(0.1)
        shadow_value, _meta = sl.call_structured(
            _Answer, _messages(), caller=lambda *_args: '{"value":2}',
            timeout_seconds=0.1, lane="shadow",
        )
        release.set()
        online.result()
    assert shadow_value and shadow_value.value == 2


def test_unknown_json_capability_blocks_online_default_call(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6")
    monkeypatch.setenv("STRUCTURED_LLM_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("STRUCTURED_LLM_JSON_CAPABILITY", "unknown")
    monkeypatch.setenv("STRUCTURED_LLM_CAPABILITY_CACHE", str(tmp_path / "missing.json"))
    invoked = False

    def must_not_call(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        return '{"value":1}'

    monkeypatch.setattr(sl, "_default_caller", must_not_call)
    value, meta = sl.call_structured(
        _Answer, _messages(), model="gpt-5.6", timeout_seconds=0.1,
    )
    assert value is None and invoked is False
    assert "json_object_capability_unverified" in meta.errors
    assert meta.planner_timeout_stage == "capability_gate"


def test_capability_probe_is_dry_by_default_and_caches_public_result(monkeypatch, tmp_path):
    cache = tmp_path / "capabilities.json"
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gpt-gateway.example/v1")
    monkeypatch.setenv("STRUCTURED_LLM_CAPABILITY_CACHE", str(cache))
    monkeypatch.setenv("STRUCTURED_LLM_JSON_CAPABILITY", "unknown")
    calls = 0

    def fake_default(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return '{"probe":"ok"}'

    monkeypatch.setattr(sl, "_default_caller", fake_default)
    inspected = sl.probe_json_object_capability(execute=False)
    assert inspected["executed"] is False and calls == 0 and not cache.exists()

    probed = sl.probe_json_object_capability(execute=True, timeout_seconds=0.1)
    assert probed["status"] == "verified" and calls == 1 and cache.exists()
    assert sl._json_capability(
        "https://gpt-gateway.example/v1", "gpt-5.6"
    ) == "verified"


def test_shadow_failures_do_not_open_online_circuit(monkeypatch):
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_WINDOW", "2")
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_FAILURE_RATIO", "0.5")
    monkeypatch.setenv("STRUCTURED_LLM_CIRCUIT_BAD_WINDOWS", "1")
    sl._reset_structured_runtime_for_tests()

    def failing(*_args):
        raise RuntimeError("shadow provider down")

    for _ in range(2):
        sl.call_structured(
            _Answer, _messages(), caller=failing, timeout_seconds=0.1,
            repair=False, lane="shadow",
        )
    stats = sl.structured_runtime_stats()
    assert stats["shadow_circuit_state"] == "open"
    assert stats["online_circuit_state"] == "closed"


def test_compact_schema_instructions_avoid_repeating_full_schema():
    seen: list[list[dict[str, str]]] = []

    def caller(messages, _max_tokens, _temperature, _model):
        seen.append(messages)
        return '{"value":7}'

    value, meta = sl.call_structured(
        _Answer, _messages(), caller=caller, timeout_seconds=0.1,
        schema_instructions='JSON shape: {"value": integer}. No extra fields.',
    )
    assert value and value.value == 7 and not meta.errors
    assert seen[-1][-1]["content"] == (
        'JSON shape: {"value": integer}. No extra fields.'
    )
    assert "$defs" not in seen[-1][-1]["content"]
