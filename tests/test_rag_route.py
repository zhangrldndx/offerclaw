"""Round 4 元数据/文件名感知路由单元测试（纯函数）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_route import audience_intent, apply_audience_routing, route_enabled

_P05 = {"source": "llm_app_intro_05_zero_foundation_path.md"}
_P06 = {"source": "llm_app_intro_06_backend_transition_path.md"}
_P07 = {"source": "llm_app_intro_07_algorithm_transition_path.md"}


def test_intent_basic():
    assert audience_intent("算法岗怎么转大模型应用开发") == "algorithm"
    assert audience_intent("后端工程师怎么转型") == "backend"
    assert audience_intent("零基础怎么入门") == "zero_foundation"
    assert audience_intent("大模型应用开发岗位主要做什么") == ""


def test_intent_long_word_priority():
    # "算法工程师" 与 "算法" 同映射 algorithm，长词优先不应抛错或误判
    assert audience_intent("算法工程师转大模型") == "algorithm"


def test_audience_mention_without_path_request_does_not_route():
    assert audience_intent("应用岗与算法岗的学历门槛有什么不同") == ""
    docs = ["correct-fact", "algorithm-path"]
    metas = [
        {"source": "llm_app_intro_04_entry_qa.md"},
        _P07,
    ]
    routed, _, _ = apply_audience_routing(
        "应用岗与算法岗的学历门槛有什么不同",
        docs,
        metas,
        [0.1, 0.2],
    )
    assert routed == docs


def test_routing_promotes_match_demotes_competitor():
    docs = ["d5", "d6", "d7", "other"]
    metas = [_P05, _P06, _P07, {"source": "some_other_doc.md"}]
    dists = [0.1, 0.2, 0.3, 0.15]
    nd, nm, _ = apply_audience_routing("算法岗怎么转", docs, metas, dists)
    assert nm[0]["source"].endswith("07_algorithm_transition_path.md")  # 匹配人群提最前
    assert set(nd[-2:]) == {"d5", "d6"}                                  # 竞争人群降最后
    assert "other" in nd                                                # 非 career 文件保留


def test_routing_no_intent_keeps_order():
    docs, metas, dists = ["a", "b"], [{"source": "x"}, {"source": "y"}], [0.1, 0.2]
    assert apply_audience_routing("没有人群词的问题", docs, metas, dists)[0] == ["a", "b"]


def test_routing_non_career_untouched():
    # intent 命中，但候选都不是 career 路径文件 → 不动（避免误伤 algorithm 域等）
    docs = ["a", "b"]
    metas = [{"source": "llm_algorithm_basic_03.md"}, {"source": "backend_api_design.md"}]
    nd, _, _ = apply_audience_routing("算法岗", docs, metas, [0.1, 0.2])
    assert nd == ["a", "b"]


def test_route_disabled(monkeypatch):
    monkeypatch.setenv("RAG_ROUTE", "0")
    assert not route_enabled()
    nd, _, _ = apply_audience_routing("算法岗", ["a", "b"], [_P07, _P05], [0.1, 0.2])
    assert nd == ["a", "b"]   # 关闭后不重排
