# -*- coding: utf-8 -*-
"""缺口信息库（gap_store）+ 每日执行自动分析（analyze_incomplete）回归。

覆盖用户定义的长期养成流程：
- JD 设为目标 → 缺口跨 JD 合并去重、持久累积；
- 计划缺口来源优先级：显式 > 缺口库 > 画像默认；
- 留痕提交后由系统对照"今日计划"判定未完成（不由用户填写）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gap_store  # noqa: E402
from summary_tool import analyze_incomplete  # noqa: E402


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(gap_store, "STORE_PATH", str(tmp_path / "gap_store.json"))
    return gap_store


# ---------------- gap_store ----------------

def test_add_target_extracts_title_and_accumulates(tmp_store):
    r = tmp_store.add_target(
        "岗位名称：大模型应用开发实习生\n公司：A公司\n要求：…",
        {"技能缺口": ["缺少 RAG 工程化实战", "Python 工程能力需提升"],
         "经历缺口": ["缺少端到端大模型项目"]})
    assert r["status"] == "ok"
    assert r["title"] == "大模型应用开发实习生" and r["company"] == "A公司"
    assert r["added_gaps"] == 3 and r["merged_gap_count"] == 3


def test_add_second_jd_merges_and_dedupes(tmp_store):
    tmp_store.add_target("岗位名称：X", {"技能缺口": ["缺少 RAG 工程化实战"]})
    r2 = tmp_store.add_target(
        "岗位名称：Y",
        {"技能缺口": ["缺少 RAG 工程化实战经验",   # 前缀包含 → 判重
                   "缺少 Agent 工具调用经历"],  # 新
         "硬门槛缺口": ["需要 3 段实习经历"]})    # 新
    assert r2["duplicate_gaps"] == 1 and r2["added_gaps"] == 2
    assert r2["total_targets"] == 2 and r2["merged_gap_count"] == 3


def test_same_jd_added_twice_merges_not_duplicates(tmp_store):
    """累加原则（目标级）：同一 JD 重复设为目标 → 合并进原条目，目标数不变。"""
    r1 = tmp_store.add_target("岗位名称：大模型应用开发工程师\n公司：G司",
                              {"技能缺口": ["缺少 RAG 实战"]})
    assert r1["action"] == "added" and r1["total_targets"] == 1
    r2 = tmp_store.add_target("岗位名称：大模型应用开发工程师\n公司：G司",
                              {"技能缺口": ["缺少 RAG 实战", "缺少 Agent 编排经历"]})
    assert r2["action"] == "merged"
    assert r2["total_targets"] == 1              # 不新增目标
    assert r2["duplicate_gaps"] == 1 and r2["added_gaps"] == 1   # 缺口仍按累加去重
    assert r2["merged_gap_count"] == 2


def test_unnamed_jd_dedup_by_content_fingerprint(tmp_store):
    """无法抽出岗位名的 JD（未命名岗位）按内容指纹判同。"""
    jd = "与大模型算法工程师紧密合作，负责智能体应用建设与落地。"
    r1 = tmp_store.add_target(jd, {"技能缺口": ["缺少工具调用经历"]})
    r2 = tmp_store.add_target(jd, {"技能缺口": ["缺少工具调用经历"]})
    assert r1["action"] == "added" and r2["action"] == "merged"
    assert r2["total_targets"] == 1


def test_different_jds_both_added(tmp_store):
    tmp_store.add_target("岗位名称：A岗\n公司：甲", {"技能缺口": ["缺少 RAG 实战"]})
    r2 = tmp_store.add_target("岗位名称：B岗\n公司：乙", {"技能缺口": ["缺少微调经验"]})
    assert r2["action"] == "added" and r2["total_targets"] == 2


def test_merged_text_compatible_with_plan_gen(tmp_store):
    tmp_store.add_target("岗位名称：X",
                         {"技能缺口": ["缺少 RAG 工程化实战"],
                          "经历缺口": ["缺少端到端大模型项目"]})
    text = tmp_store.merged_gaps_text()
    assert "技能缺口：" in text and "- 缺少 RAG 工程化实战" in text
    from plan_gen import _split_gap_queries
    qs = _split_gap_queries(text)
    assert len(qs) == 2  # 分类标题不进 query，条目逐条进


def test_empty_store_summary(tmp_store):
    assert tmp_store.merged_gaps_text() == ""
    s = tmp_store.summary()
    assert s["total_targets"] == 0 and s["merged_gap_count"] == 0


# ---------------- analyze_incomplete ----------------

def test_analyze_all_incomplete_when_no_done():
    planned = ["推进本周主线（第2周）：RAG 全链路实战", "向本周交付推进：集成 RAG recipe"]
    assert analyze_incomplete([], planned) == planned


def test_analyze_detects_done_by_tech_keyword():
    planned = ["推进本周主线（第2周）：RAG 全链路实战落地",
               "向本周交付推进：在 LocalFlow 集成 recipe"]
    done = ["今天跑通了 RAG 检索链路并写了笔记"]
    inc = analyze_incomplete(done, planned)
    assert len(inc) == 1 and "LocalFlow" in inc[0]   # RAG 项已覆盖，LocalFlow 项未做


def test_analyze_detects_done_by_chinese_overlap():
    planned = ["整理面试故事素材库"]
    done = ["把面试故事素材整理了一半"]
    assert analyze_incomplete(done, planned) == []


def test_analyze_no_false_complete_on_generic_words():
    """只有「推进/学习/完成」这类框架词重叠不算完成。"""
    planned = ["向本周交付推进：撰写执行管道设计复盘"]
    done = ["今天推进了一些别的学习"]
    inc = analyze_incomplete(done, planned)
    assert len(inc) == 1


def test_analyze_empty_planned():
    assert analyze_incomplete(["做了点事"], []) == []


# ---------------- today_plan 字段 ----------------

def test_today_advice_exposes_today_plan(monkeypatch):
    import datetime
    import plan_gen
    import career_agent
    PLAN = ("计划周期：2026-06-03 → 2026-06-30\n"
            "Week 1 (06-03 → 06-09) 主题：夯实基础\n  交付物：完成入门教程\n"
            "Week 2 (06-10 → 06-16) 主题：RAG 实战\n  交付物：集成 recipe\n")
    monkeypatch.setattr(plan_gen, "load_latest_plan",
                        lambda: {"content": PLAN, "filename": "p.md", "mtime": 0,
                                 "edited_by_user": False})

    class _D(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 12)
    monkeypatch.setattr(career_agent.datetime, "date", _D)
    adv = career_agent.get_today_advice()
    tp = adv["today_plan"]
    assert any("第2周" in t and "RAG" in t for t in tp)
    assert any("recipe" in t for t in tp)
