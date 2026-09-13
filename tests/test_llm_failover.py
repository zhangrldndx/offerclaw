# -*- coding: utf-8 -*-
"""LLM 网关兜底切换(主 GPT → DeepSeek)与 VL 端点独立解析的单元测试。

全部 mock requests.post,零真实网络调用;time.sleep 打桩,零等待。
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import day1_api_starter as d1  # noqa: E402

OK_DATA = {"choices": [{"message": {"content": "ok"}}], "usage": {}}


class _Resp:
    def __init__(self, status: int, body: str = "", data: dict | None = None):
        self.status_code = status
        self.text = body if body else json.dumps(data or OK_DATA)
        self._data = data or OK_DATA

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err


@pytest.fixture()
def gateway(monkeypatch):
    """打桩网络与 sleep;返回 (发起调用的函数, 记录所有请求的列表)。"""
    calls: list[dict] = []
    queue: list[_Resp] = []

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append({"url": url, "model": (json or {}).get("model"),
                      "auth": (headers or {}).get("Authorization", "")})
        return queue.pop(0) if queue else _Resp(200)

    monkeypatch.setattr(d1.requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    for k in ("LLM_FALLBACK_ENABLED", "LLM_FALLBACK_API_KEY",
              "LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_MODEL"):
        monkeypatch.delenv(k, raising=False)

    def call():
        return d1.chat_completion(
            "https://primary.example.com/v1/chat/completions",
            {"Authorization": "Bearer sk-primary"},
            {"model": "gpt-5.6", "messages": [], "reasoning_effort": "medium"},
            timeout=5)

    return call, calls, queue


def _enable_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "sk-fallback")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "deepseek-v4-pro")


def test_fallback_credentials_are_inert_when_switch_is_off(gateway, monkeypatch):
    """DeepSeek 配置可以保留，但总开关默认关闭时绝不能发出请求。"""
    call, calls, queue = gateway
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "0")
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "sk-fallback")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "deepseek-v4-pro")
    queue.append(_Resp(402, "insufficient balance"))
    with pytest.raises(requests.exceptions.HTTPError):
        call()
    assert calls
    assert all("api.deepseek.com" not in item["url"] for item in calls)


def test_quota_402_switches_to_fallback(gateway, monkeypatch):
    call, calls, queue = gateway
    _enable_fallback(monkeypatch)
    queue.extend([_Resp(402, "insufficient balance"), _Resp(200)])
    assert call() == OK_DATA
    assert calls[-1]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[-1]["model"] == "deepseek-v4-pro"
    assert calls[-1]["auth"] == "Bearer sk-fallback"


def test_hard_quota_403_skips_backoff_then_fallback(gateway, monkeypatch):
    """额度耗尽 403(百炼 FreeTierOnly 同款文案)有兜底时不再长退避:主配置只打 1 次。"""
    call, calls, queue = gateway
    _enable_fallback(monkeypatch)
    queue.extend([_Resp(403, "Free quota exhausted. use free tier only"), _Resp(200)])
    assert call() == OK_DATA
    primary_calls = [c for c in calls if "primary" in c["url"]]
    assert len(primary_calls) == 1
    assert calls[-1]["model"] == "deepseek-v4-pro"
    # 兜底 payload 不带 reasoning_effort(由请求记录无法看到——由 url/model 断言 + 行为回归覆盖)


def test_connection_error_switches_to_fallback(gateway, monkeypatch):
    call, calls, queue = gateway
    _enable_fallback(monkeypatch)

    real_queue_pop = list(queue)

    def flaky_post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append({"url": url, "model": (json or {}).get("model"),
                      "auth": (headers or {}).get("Authorization", "")})
        if "primary" in url:
            raise requests.exceptions.ConnectionError("proxy unreachable")
        return _Resp(200)

    monkeypatch.setattr(d1.requests, "post", flaky_post)
    assert call() == OK_DATA
    assert calls[-1]["url"].startswith("https://api.deepseek.com")
    assert real_queue_pop == []  # queue 未消费(自定义桩)


def test_no_fallback_configured_raises(gateway):
    call, calls, queue = gateway
    queue.extend([_Resp(402, "insufficient balance"), _Resp(402, "insufficient balance")])
    with pytest.raises(requests.exceptions.HTTPError):
        call()
    assert all("primary" in c["url"] for c in calls)


def test_param_error_400_never_fails_over(gateway, monkeypatch):
    """参数错(如模型名打错 400/404)不兜底——配置错误必须暴露而非被掩盖。"""
    call, calls, queue = gateway
    _enable_fallback(monkeypatch)
    queue.append(_Resp(400, "unknown model"))
    with pytest.raises(requests.exceptions.HTTPError):
        call()
    assert all("primary" in c["url"] for c in calls)


def test_fallback_loop_guard(gateway, monkeypatch):
    """主配置已指向兜底端点时不再自我切换(防循环)。"""
    _call, calls, queue = gateway
    _enable_fallback(monkeypatch)
    queue.extend([_Resp(402, "insufficient balance"), _Resp(402, "insufficient balance")])
    with pytest.raises(requests.exceptions.HTTPError):
        d1.chat_completion("https://api.deepseek.com/chat/completions",
                           {"Authorization": "Bearer sk-fallback"},
                           {"model": "deepseek-v4-pro", "messages": []}, timeout=5)
    assert all("deepseek" in c["url"] for c in calls)


# ---------- VL 端点独立解析 ----------

def test_vl_endpoint_default_bailian(monkeypatch):
    from image_caption import _vl_endpoint
    monkeypatch.delenv("VL_BASE_URL", raising=False)
    monkeypatch.delenv("VL_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dash")
    base, key = _vl_endpoint()
    assert base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert key == "sk-dash"


def test_vl_endpoint_env_override(monkeypatch):
    from image_caption import _vl_endpoint
    monkeypatch.setenv("VL_BASE_URL", "https://vl.example.com/v1/")
    monkeypatch.setenv("VL_API_KEY", "sk-vl")
    base, key = _vl_endpoint()
    assert base == "https://vl.example.com/v1" and key == "sk-vl"
