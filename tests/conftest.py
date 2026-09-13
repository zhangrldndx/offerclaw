# -*- coding: utf-8 -*-
"""pytest 全局配置。

测试默认关闭 rerank（RAG_RERANK=0）：避免在未下载 reranker 模型的环境（CI/新机器）
触发 ~1GB 模型下载、拖慢检索类测试。rerank 自身逻辑由 tests/test_rag_rerank.py 用
monkeypatch 显式覆盖测试；生产运行（rag_api / offerclaw_cli）仍默认 RAG_RERANK=1 开启。
"""
import copy
import os

import pytest

os.environ.setdefault("RAG_RERANK", "0")

# A developer may be collecting a real organic answerability window through
# .env.local while running the test suite.  Tests must never inherit that
# production-local switch and write fixtures into the live window.  Individual
# shadow tests opt in explicitly with monkeypatch.
# 2026-08-29 默认翻转后判据在生产默认开;测试套件必须密闭(不打真 LLM),
# 所以这里显式关。测判据的用例用 monkeypatch.setenv("RAG_ANSWERABILITY","1") 自己开。
os.environ.setdefault("RAG_ANSWERABILITY", "0")
os.environ.setdefault("RAG_SHADOW_ANSWERABILITY", "0")
os.environ.setdefault("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")


def pytest_collection_modifyitems(config, items):
    """Keep ignored evaluation evidence opt-in so public CI matches a fresh clone."""
    if os.environ.get("OFFERCLAW_PRIVATE_EVAL") == "1":
        return
    marker = pytest.mark.skip(
        reason="requires ignored private evaluation artifacts; set OFFERCLAW_PRIVATE_EVAL=1",
    )
    for item in items:
        if item.get_closest_marker("private_artifact"):
            item.add_marker(marker)


@pytest.fixture
def synthetic_career_profile(monkeypatch):
    """Patch CareerFlow with a fictional profile instead of local user state."""
    import career_flow

    profile = {
        "学历": "硕士",
        "专业": "人工智能",
        "所在地": "杭州",
        "可接受地域": ["上海", "杭州", "远程"],
        "方向优先级": ["AI 应用开发", "Python 后端"],
        "目标岗位类型": "实习",
        "行业偏好": "不限",
        "明确不做": ["Java", "C++", "前端", "嵌入式"],
        "工作性质偏好": "实习",
        "期望薪资": "面议",
        "熟练技能": ["Python", "LangGraph", "RAG", "FastAPI", "Embedding", "Agent"],
        "会用技能": ["SQL", "数据处理"],
        "技能证据": ["使用合成数据完成 RAG 回归项目"],
        "项目技能": ["Python", "LangGraph", "RAG", "FastAPI", "Embedding", "Agent"],
        "项目数量": 2,
        "实习数量": 0,
        "英语自评": 2,
        "_source": "synthetic_test_fixture",
    }
    monkeypatch.setattr(
        career_flow,
        "load_profile",
        lambda **_kwargs: copy.deepcopy(profile),
    )
    return profile
