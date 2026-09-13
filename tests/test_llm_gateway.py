"""A1 统一 LLM 网关 chat_completion 的故障注入测试。

量化目标：注入 429/5xx/超时/连接错误时网关靠重试把端到端成功率从『首次失败即崩』拉到接近 100%；
而 4xx（参数/认证错）立即抛、不浪费重试。time.sleep 被 patch 掉避免真等退避。
"""
import os
import sys

import pytest
import requests
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from day1_api_starter import chat_completion


def _resp(status, content="ok"):
    r = MagicMock()
    r.status_code = status
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(response=r)
    else:
        r.raise_for_status.return_value = None
        r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)  # 跳过退避真等
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")


def _patch_posts(monkeypatch, responses):
    calls = []
    it = iter(responses)

    def fake_post(*a, **k):
        calls.append(1)
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr("day1_api_starter.requests.post", fake_post)
    return calls


def test_retries_5xx_then_success(monkeypatch):
    calls = _patch_posts(monkeypatch, [_resp(503), _resp(503), _resp(200, "good")])
    out = chat_completion("u", {}, {}, timeout=5)
    assert out["choices"][0]["message"]["content"] == "good"
    assert len(calls) == 3  # 前两次 503 重试，第三次成功


def test_retries_429_then_success(monkeypatch):
    calls = _patch_posts(monkeypatch, [_resp(429), _resp(200)])
    assert chat_completion("u", {}, {}, timeout=5)["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2


def test_retries_timeout_then_success(monkeypatch):
    calls = _patch_posts(monkeypatch, [requests.exceptions.Timeout(), _resp(200)])
    assert chat_completion("u", {}, {}, timeout=5)["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2


def test_retries_connection_error_then_success(monkeypatch):
    calls = _patch_posts(monkeypatch, [requests.exceptions.ConnectionError(), _resp(200)])
    assert chat_completion("u", {}, {}, timeout=5)
    assert len(calls) == 2


def test_no_retry_on_4xx(monkeypatch):
    """4xx（如 400 参数错 / 401 认证错）立即抛，不重试——重试也不会变对。"""
    calls = _patch_posts(monkeypatch, [_resp(400), _resp(200)])
    with pytest.raises(requests.exceptions.HTTPError):
        chat_completion("u", {}, {}, timeout=5)
    assert len(calls) == 1  # 只调一次，没浪费重试


def test_exhausts_retries_then_raises(monkeypatch):
    calls = _patch_posts(monkeypatch, [_resp(503)] * 4)
    with pytest.raises(requests.exceptions.HTTPError):
        chat_completion("u", {}, {}, timeout=5)
    assert len(calls) == 4  # 用尽 LLM_MAX_RETRIES=4 次


def test_zero_retries_means_one_attempt_instead_of_raise_none(monkeypatch):
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    calls = _patch_posts(monkeypatch, [_resp(200, "once")])
    out = chat_completion("u", {}, {}, timeout=5, max_retries=0)
    assert out["choices"][0]["message"]["content"] == "once"
    assert len(calls) == 1


def test_stream_returns_response(monkeypatch):
    """stream=True 返回原始 Response 供逐行读，而非解析 JSON。"""
    r = _resp(200)
    _patch_posts(monkeypatch, [r])
    assert chat_completion("u", {}, {}, timeout=5, stream=True) is r


# ── A2 防御式响应解析 + 错误详情 ─────────────────────────────

def test_extract_content_empty_choices_raises():
    """空 choices / error 对象 → 抛 LLMResponseError（而非裸 IndexError/KeyError 崩 CLI）。"""
    from day1_api_starter import extract_content, LLMResponseError
    for bad in ({"choices": []}, {"error": {"message": "blocked"}}, {}):
        with pytest.raises(LLMResponseError):
            extract_content(bad)


def test_extract_content_normal():
    from day1_api_starter import extract_content
    assert extract_content({"choices": [{"message": {"content": "hi"}}]}) == "hi"
    assert extract_content({"choices": [{"message": {}}]}) == ""  # 无 content → 空串不报错


def test_llm_error_detail_http_carries_proxy_body():
    """HTTPError 把代理响应体带出来（裸抛会丢失这条关键排障信息）。"""
    from day1_api_starter import llm_error_detail
    r = MagicMock(); r.status_code = 403; r.text = "proxy blocked: content policy"
    e = requests.exceptions.HTTPError(response=r)
    msg = llm_error_detail(e)
    assert "403" in msg and "proxy blocked" in msg


def test_llm_error_detail_timeout_no_response():
    from day1_api_starter import llm_error_detail
    assert "Timeout" in llm_error_detail(requests.exceptions.Timeout("timed out"))


def test_gateway_empty_choices_then_extract_raises(monkeypatch):
    """端到端：网关返回空 choices（代理拦截）→ extract_content 抛 LLMResponseError。"""
    from day1_api_starter import extract_content, LLMResponseError
    _patch_posts(monkeypatch, [_resp(200, content=None)])
    # _resp(200, content=None) 仍是合法 choices；这里单独构造空 choices 响应
    bad = MagicMock(); bad.status_code = 200
    bad.raise_for_status.return_value = None
    bad.json.return_value = {"choices": [], "error": {"message": "policy"}}
    monkeypatch.setattr("day1_api_starter.requests.post", lambda *a, **k: bad)
    data = chat_completion("u", {}, {}, timeout=5)
    with pytest.raises(LLMResponseError):
        extract_content(data)
