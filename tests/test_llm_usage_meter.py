# -*- coding: utf-8 -*-
"""P5 网关旁路计量单测:记账正确 + 永不干扰主链路 + 开关可关。"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import day1_api_starter as gw


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def test_usage_line_written(tmp_path, monkeypatch):
    # 把 logs 目录指到临时位置:通过改 __file__ 所在目录不可行,改用真实 logs + 唯一断言?
    # 更稳:monkeypatch os.path.join 太宽。选择直接调私有函数并临时重定向 BASE_DIR 逻辑——
    # _log_llm_usage 用模块文件位置拼 logs/,这里 monkeypatch dirname 返回 tmp。
    monkeypatch.setattr(gw.os.path, "dirname", lambda p: str(tmp_path))
    payload = {"model": "test-model"}
    data = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    gw._log_llm_usage(payload, data, 123)
    rows = _read_lines(tmp_path / "logs" / "llm_usage.jsonl")
    assert rows[-1]["model"] == "test-model"
    assert rows[-1]["prompt_tokens"] == 10 and rows[-1]["elapsed_ms"] == 123


def test_no_usage_no_line(tmp_path, monkeypatch):
    monkeypatch.setattr(gw.os.path, "dirname", lambda p: str(tmp_path))
    gw._log_llm_usage({"model": "m"}, {"choices": []}, 1)   # 无 usage 字段
    assert not (tmp_path / "logs" / "llm_usage.jsonl").exists()


def test_switch_off(tmp_path, monkeypatch):
    monkeypatch.setattr(gw.os.path, "dirname", lambda p: str(tmp_path))
    monkeypatch.setenv("LLM_USAGE_LOG", "0")
    gw._log_llm_usage({"model": "m"}, {"usage": {"total_tokens": 1}}, 1)
    assert not (tmp_path / "logs" / "llm_usage.jsonl").exists()


def test_never_raises_on_garbage(monkeypatch):
    # 记账内部任何异常都必须被吞掉——传入各种畸形输入不允许抛
    gw._log_llm_usage(None, None, 0)
    gw._log_llm_usage({}, {"usage": "not-a-dict"}, 0)
    gw._log_llm_usage({"model": object()}, {"usage": {"prompt_tokens": object()}}, 0)
