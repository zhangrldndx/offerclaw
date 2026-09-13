# -*- coding: utf-8 -*-
"""test_reflection_loop.py — P2：复盘→调整闭环的纯离线单元测试。

覆盖 memory_layers 的 reflection 记录、distill 规则沉淀、调整读取，
以及 summary_tool 的结构化复盘抽取（确定性部分，不调 LLM）。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import memory_layers as ml


def _fresh(tmp_path):
    epi = ml.EpisodicMemory(base_dir=str(tmp_path))
    sem = ml.SemanticMemory(base_dir=str(tmp_path))
    return epi, sem


def test_record_reflection_roundtrip(tmp_path):
    epi, _sem = _fresh(tmp_path)
    ml.record_reflection(epi, {
        "date": "2026-06-02", "main_tag": "补技能",
        "deviation_score": 30, "completed": ["看完RAG教程"],
        "incomplete": ["刷题2道"], "blockers": ["文档太少"],
        "next_day_suggestion": "继续RAG",
    })
    refl = [e for e in epi.all() if e.get("kind") == "reflection"]
    assert len(refl) == 1
    assert refl[0]["main_tag"] == "补技能"
    assert refl[0]["deviation_score"] == 30


def test_distill_high_deviation_streak(tmp_path):
    epi, sem = _fresh(tmp_path)
    for d, score in [("2026-05-31", 60), ("2026-06-01", 55), ("2026-06-02", 70)]:
        ml.record_reflection(epi, {"date": d, "deviation_score": score})
    out = ml.distill_reflections_to_semantic(epi, sem, streak=3)
    patterns = [r["pattern"] for r in out["rules"]]
    assert "high_deviation_streak" in patterns
    # 生效调整可被读出
    adj = ml.get_active_adjustments(sem)
    assert any("偏离度" in a for a in adj)


def test_distill_no_streak_when_low_deviation(tmp_path):
    epi, sem = _fresh(tmp_path)
    for d, score in [("2026-05-31", 60), ("2026-06-01", 20), ("2026-06-02", 70)]:
        ml.record_reflection(epi, {"date": d, "deviation_score": score})
    out = ml.distill_reflections_to_semantic(epi, sem, streak=3)
    patterns = [r["pattern"] for r in out["rules"]]
    assert "high_deviation_streak" not in patterns


def test_distill_recurring_incomplete(tmp_path):
    epi, sem = _fresh(tmp_path)
    for d in ["2026-05-31", "2026-06-01", "2026-06-02"]:
        ml.record_reflection(epi, {
            "date": d, "deviation_score": 20,
            "incomplete": ["刷 LeetCode 2 题"],
        })
    out = ml.distill_reflections_to_semantic(epi, sem, streak=3)
    patterns = [r["pattern"] for r in out["rules"]]
    assert any(p.startswith("recurring_incomplete:") for p in patterns)
    adj = ml.get_active_adjustments(sem)
    assert any("LeetCode" in a or "未完成" in a for a in adj)


def test_distill_dedup_same_task_within_one_day(tmp_path):
    """同一天 incomplete 里重复同类任务，只计 1 次，不应误触发。"""
    epi, sem = _fresh(tmp_path)
    ml.record_reflection(epi, {
        "date": "2026-06-02", "deviation_score": 10,
        "incomplete": ["刷题", "刷题", "刷题"],
    })
    out = ml.distill_reflections_to_semantic(epi, sem, streak=3)
    assert all(not p.startswith("recurring_incomplete:") for p in [r["pattern"] for r in out["rules"]])


def test_get_active_adjustments_empty_when_none(tmp_path):
    _epi, sem = _fresh(tmp_path)
    assert ml.get_active_adjustments(sem) == []


# ---- summary_tool 结构化复盘抽取（确定性部分，不调 LLM）----

import summary_tool as st


def test_extract_llm_json_from_fenced_block():
    text = (
        "复盘正文……\n\n```json\n"
        '{"deviation_score": 40, "incomplete": ["刷题"], "completed": ["看RAG"]}\n'
        "```\n"
    )
    out = st._extract_llm_json(text)
    assert out["deviation_score"] == 40
    assert out["incomplete"] == ["刷题"]


def test_extract_llm_json_empty_when_absent():
    assert st._extract_llm_json("没有结构化块的纯文本复盘") == {}


def test_build_structured_reflection_estimates_deviation_without_llm():
    """无 LLM JSON 时，deviation_score 按未完成/总数比例估算。"""
    block = (
        "## 2026-06-02\n"
        "### 今日主线标签：补技能\n"
        "### 实际完成\n- 看完 RAG 教程\n- 写了分块代码\n"
        "### 未完成\n- 刷 LeetCode 2 题\n"
    )
    refl = st.build_structured_reflection(block, "2026-06-02", "纯文本无json")
    assert refl["main_tag"] == "补技能"
    assert len(refl["completed"]) == 2
    assert len(refl["incomplete"]) == 1
    # 1 未完成 / 3 总 ≈ 33
    assert 30 <= refl["deviation_score"] <= 36


def test_build_structured_reflection_prefers_llm_json():
    block = "## 2026-06-02\n### 今日主线标签：补项目\n"
    summary = '```json\n{"deviation_score": 80, "incomplete": ["A","B"], "blockers": ["卡在环境"]}\n```'
    refl = st.build_structured_reflection(block, "2026-06-02", summary)
    assert refl["deviation_score"] == 80
    assert refl["blockers"] == ["卡在环境"]


def test_record_and_distill_writes_memory(tmp_path, monkeypatch):
    """record_and_distill 应把复盘写进 memory 并能产出调整（用临时目录隔离）。"""
    import memory_layers
    monkeypatch.setattr(memory_layers, "BASE_DIR_DEFAULT", str(tmp_path))
    # 连续 3 天高偏离 → 应沉淀 high_deviation_streak
    for d, s in [("2026-05-31", 60), ("2026-06-01", 60), ("2026-06-02", 60)]:
        st.record_and_distill({"date": d, "deviation_score": s})
    last = st.record_and_distill({"date": "2026-06-02", "deviation_score": 60})
    # 注意：record_and_distill 内部新建 Memory 实例，会读 BASE_DIR_DEFAULT
    assert last["ok"] is True


# ---- P3：微信结构化留痕解析（cmd_log 的纯函数）----

import offerclaw_cli as oc


def test_parse_structured_log_full():
    content = "主线:补技能\n完成:看完RAG教程;写了分块代码\n未完成:刷题2道\n笔记:文档偏少"
    tag, done, todo, notes = oc._parse_structured_log(content)
    assert tag == "补技能"
    assert done == ["看完RAG教程", "写了分块代码"]
    assert todo == ["刷题2道"]
    assert "文档偏少" in notes


def test_parse_structured_log_english_prefixes():
    content = "tag: 补项目\ndone: A; B\ntodo: C"
    tag, done, todo, notes = oc._parse_structured_log(content)
    assert tag == "补项目"
    assert done == ["A", "B"]
    assert todo == ["C"]


def test_parse_structured_log_accepts_multiple_fields_on_one_line():
    content = "完成:验证微信附件预览;笔记:不得自动写入"
    tag, done, todo, notes = oc._parse_structured_log(content)
    assert tag == ""
    assert done == ["验证微信附件预览"]
    assert todo == []
    assert notes == "不得自动写入"


def test_parse_structured_log_freeform_fallback():
    """无前缀的纯文本整段进 notes，不报错。"""
    tag, done, todo, notes = oc._parse_structured_log("今天就是随便记一句")
    assert tag == ""
    assert done == [] and todo == []
    assert notes == "今天就是随便记一句"


# ---- 计划微信摘要抽取 ----

def test_extract_weekly_themes():
    plan = "前言\nWeek 1 (06-03 → 06-09) 主题：RAG 基础\n日任务\nWeek 2 (06-10 → 06-16) 主题：Agent\n"
    themes = oc._extract_weekly_themes(plan)
    assert len(themes) == 2
    assert "RAG 基础" in themes[0]


def test_default_gaps_from_profile_nonempty():
    g = oc._default_gaps_from_profile()
    assert "RAG" in g and "Agent" in g
