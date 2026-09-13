# -*- coding: utf-8 -*-
"""[L6] 反思二阶 verifier + episodic 回流 —— 故障注入式量化测试。

注入场景：LLM 给「格式完美但逻辑矛盾」的复盘（deviation_score=0 却列 5 项未完成）。
- 改造前：A6 的 repair 只保证能抽出 JSON，矛盾内容原样落库。
- 改造后：二阶 verifier 校验 score 与未完成比例自洽，不一致 → 以确定性比例值纠正 + 记 _verifier。
- 另：episodic 反思事件回流到 plan_gen（此前只写不读）。
量化：矛盾反思的「拦截率」（被 verifier 标记并纠正）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summary_tool import verify_reflection_consistency, build_structured_reflection
from memory_layers import EpisodicMemory, record_reflection, recent_reflection_lessons


# ── 二阶 verifier 纯函数 ─────────────────────────────────────

def test_consistent_reflection_passes():
    r = {"deviation_score": 80, "completed": ["a"], "incomplete": ["x", "y", "z", "w"]}
    v = verify_reflection_consistency(r)
    assert v["consistent"] and v["issues"] == [] and v["deviation_score"] == 80


def test_contradiction_zero_score_with_incomplete():
    r = {"deviation_score": 0, "completed": ["a"], "incomplete": ["x", "y", "z", "w", "v"]}
    v = verify_reflection_consistency(r)
    assert not v["consistent"] and v["issues"]
    assert v["deviation_score"] == 83          # 纠正为确定性比例值 5/6≈83（不再是骗人的 0）


def test_contradiction_high_score_zero_incomplete():
    r = {"deviation_score": 80, "completed": ["a", "b"], "incomplete": []}
    v = verify_reflection_consistency(r)
    assert not v["consistent"] and v["deviation_score"] == 0   # 零未完成 → 偏离应为 0


# ── build_structured_reflection 应用 verifier ────────────────

def test_build_reflection_corrects_contradiction():
    summary_text = ('```json\n'
                    '{"deviation_score": 0, "completed": ["做了A"], '
                    '"incomplete": ["x", "y", "z", "w", "v"]}\n```')
    refl = build_structured_reflection("## 2026-06-22\n内容", "2026-06-22", summary_text)
    assert refl["deviation_score"] != 0          # 矛盾被纠正（原 LLM 给 0）
    assert refl["deviation_score"] == 83
    assert refl.get("_verifier") and not refl["_verifier"]["consistent"]   # 被标记


def test_build_reflection_consistent_no_flag():
    summary_text = ('```json\n'
                    '{"deviation_score": 80, "completed": ["A"], '
                    '"incomplete": ["x", "y", "z", "w"]}\n```')
    refl = build_structured_reflection("## d\n", "2026-06-22", summary_text)
    assert refl["deviation_score"] == 80 and "_verifier" not in refl   # 一致 → 不标记


# ── episodic 回流 ───────────────────────────────────────────

def test_recent_reflection_lessons(tmp_path):
    epi = EpisodicMemory(base_dir=str(tmp_path / "mem"))
    record_reflection(epi, {"date": "2026-06-20", "deviation_score": 60,
                            "incomplete": ["复习 Transformer", "刷题"]})
    record_reflection(epi, {"date": "2026-06-21", "deviation_score": 0, "incomplete": []})
    lessons = recent_reflection_lessons(epi, n=3)
    assert len(lessons) == 1                      # 只回流有未完成项的复盘
    assert "2026-06-20" in lessons[0] and "Transformer" in lessons[0]


def test_corrected_value_flows_to_episodic_and_lessons(tmp_path):
    """端到端：矛盾反思被 verifier 纠正后，纠正值（而非原错值 0）应真正流入 episodic→回流教训。"""
    summary_text = ('```json\n{"deviation_score": 0, "completed": ["A"], '
                    '"incomplete": ["x", "y", "z", "w", "v"]}\n```')
    refl = build_structured_reflection("## d\n", "2026-06-22", summary_text)
    assert refl["deviation_score"] == 83          # verifier 已纠正
    epi = EpisodicMemory(base_dir=str(tmp_path / "mem"))
    record_reflection(epi, refl)                  # 落库的是纠正后的反思
    lessons = recent_reflection_lessons(epi, n=3)
    assert lessons and "偏离度 83" in lessons[0]    # 回流看到的是纠正值 83，不是骗人的 0
