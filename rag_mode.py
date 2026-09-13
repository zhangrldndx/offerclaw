# -*- coding: utf-8 -*-
"""Fast/Quality 运行模式(COMPLETE_TEST_PLAN §10.1 的"统一 mode 配置")。

一个显式产品开关 ``RAG_MODE``,映射到**既有的、已单独验证过的**回退旋钮——
不发明新行为,只给"这一组旋钮"起一个用户级名字:

  - ``quality``(缺省) 生产默认:判据重排(teacher) + 答案含量门(三票) +
                        HyDE 双通道。Final v4 背书的那套。
  - ``fast``            本地召回 + BM25 + 精排 + 零 LLM 门:判据/门/HyDE
                        双通道全关。面试演示与普通交互档,数值口径见
                        docs/rag_eval/release_v5/LATENCY_SUMMARY.json。

规则(类级):
  1. 显式单项 env(如 ``RAG_ANSWERABILITY_GATE=1``)永远压过模式预设——
     模式只补缺省,不抢显式决定;
  2. 未设置/未知值一律按 quality(fail-safe 到现状,行为零变化);
  3. 各旋钮读点统一经 :func:`mode_env` 取值,新增旋钮想进模式就登记进
     ``PRESETS``,不再散落地拼 env(计划明令禁止的做法)。
"""

from __future__ import annotations

import os

PRESETS: dict[str, dict[str, str]] = {
    # quality = 生产默认,不覆盖任何旋钮(缺省即是)。
    "quality": {},
    # fast 的三个旋钮都是文档在案的既有回退开关(metrics.json default_profile_note)。
    "fast": {
        "RAG_ANSWERABILITY": "0",
        "RAG_ANSWERABILITY_GATE": "0",
        "RAG_HYDE_CHANNELS": "0",
    },
}


def mode_name() -> str:
    raw = os.environ.get("RAG_MODE", "").strip().lower()
    return raw if raw in PRESETS else "quality"


def mode_env(name: str) -> str | None:
    """旋钮取值:显式 env 优先;缺省时由当前模式预设补;都没有返回 None。

    返回 None 表示"维持该旋钮自己的内建默认"——调用方原有的默认分支
    一行不改,行为等价性由 tests/test_rag_mode.py 钉死。
    """
    explicit = os.environ.get(name)
    if explicit is not None and explicit.strip():
        return explicit
    return PRESETS[mode_name()].get(name)
