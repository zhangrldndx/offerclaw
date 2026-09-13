# -*- coding: utf-8 -*-
"""Fast/Quality 运行模式开关(COMPLETE_TEST_PLAN §10.1)。

计划原文:"若代码尚未提供显式模式,本阶段先实现统一 mode 配置;未设置仍保持
当前默认行为,**禁止通过测试脚本私自拼环境变量冒充产品模式**。"

契约(类级,不是散落的 env 组合):
  - RAG_MODE 未设置        → 一切旋钮维持生产默认(quality),行为零变化;
  - RAG_MODE=fast          → 判据重排 off + 答案含量门 off + HyDE 双通道 off
                             (本地召回+BM25+精排+零 LLM 门,§10.1 的 fast 定义);
  - RAG_MODE=quality       → 与未设置等价;
  - 显式单项 env(如 RAG_ANSWERABILITY_GATE=1)**永远压过**模式预设——
    模式只补缺省,不抢显式决定;
  - 未知模式值 → 按 quality 处理(fail-safe 到现状,不悄悄变行为)。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RAG_MODE", "RAG_ANSWERABILITY", "RAG_ANSWERABILITY_MODE",
                 "RAG_ANSWERABILITY_GATE", "RAG_HYDE_CHANNELS"):
        monkeypatch.delenv(name, raising=False)


def test_unset_mode_changes_nothing(monkeypatch):
    import rag_answerability
    from rag_gate import _answerability_gate
    from rag_mode import mode_name
    from rag_retrieval_trace import _env_bool

    assert mode_name() == "quality"
    assert _answerability_gate() is True
    assert rag_answerability.mode() == "teacher"
    assert _env_bool("RAG_HYDE_CHANNELS", True) is True


def test_fast_mode_maps_to_the_verified_knobs(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "fast")
    import rag_answerability
    from rag_gate import _answerability_gate
    from rag_retrieval_trace import _env_bool

    assert _answerability_gate() is False
    assert rag_answerability.mode() == "off"
    assert _env_bool("RAG_HYDE_CHANNELS", True) is False


def test_quality_mode_equals_unset(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "quality")
    import rag_answerability
    from rag_gate import _answerability_gate

    assert _answerability_gate() is True
    assert rag_answerability.mode() == "teacher"


def test_explicit_knob_beats_mode_preset(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "fast")
    monkeypatch.setenv("RAG_ANSWERABILITY_GATE", "1")
    from rag_gate import _answerability_gate

    assert _answerability_gate() is True, "显式单项 env 必须压过模式预设"


def test_unknown_mode_fails_safe_to_quality(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "turbo")
    from rag_gate import _answerability_gate
    from rag_mode import mode_name

    assert mode_name() == "quality"
    assert _answerability_gate() is True
