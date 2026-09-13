"""A6 结构化 JSON 输出 repair 重试 + 解析埋点测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summary_tool import extract_json_with_repair

GOOD = '复盘正文……\n```json\n{"deviation_score": 30, "main_tag": "RAG"}\n```\n结尾'
BAD = '一段完全没有结构化块的复盘文本，模型没按要求输出'


def test_first_pass_ok_skips_repair():
    called = []
    out = extract_json_with_repair(GOOD, repair_fn=lambda: called.append(1))
    assert out["deviation_score"] == 30
    assert not called                                   # 首轮成功 → 不触发 repair


def test_repair_recovers_missing_json():
    out = extract_json_with_repair(BAD, repair_fn=lambda: GOOD)   # 重发带 json
    assert out["deviation_score"] == 30                 # repair 抽到结构化字段


def test_repair_still_missing_returns_empty():
    assert extract_json_with_repair(BAD, repair_fn=lambda: "还是没有 json") == {}


def test_no_repair_fn_returns_empty():
    assert extract_json_with_repair(BAD) == {}


def test_repair_fn_exception_handled():
    def boom():
        raise RuntimeError("llm down")
    assert extract_json_with_repair(BAD, repair_fn=boom) == {}    # 不冒泡


def test_repair_lifts_extraction_rate():
    """量化：首轮无 json 时，无 repair 提取失败(0%)，有 repair 提取成功(100%)。"""
    assert extract_json_with_repair(BAD) == {}                        # baseline: miss
    assert extract_json_with_repair(BAD, repair_fn=lambda: GOOD)      # repaired: hit
