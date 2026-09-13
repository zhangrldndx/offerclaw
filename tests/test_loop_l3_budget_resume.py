# -*- coding: utf-8 -*-
"""[L3] 外循环预算 + checkpoint/resume —— 故障注入式量化测试。

注入场景：① 给 CareerFlow 一个过小预算（max_steps）；② 模拟跑到一半「崩溃」（只剩 checkpoint）。
- 改造前：graph.invoke 无预算（长任务无法限本）、state 仅在内存（崩了全丢，只能重跑全流程）。
- 改造后：预算超限 → 下游节点优雅降级跳过（记 skipped_budget_exhausted）；每节点 checkpoint 落盘，
  resume_career_flow 从最后检查点 replay 续跑到完成。
量化：预算超限的"中止成功率"（下游被跳过且可观测）、崩溃后 resume 的"续跑成功率"（完成度恢复）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import career_flow
from career_flow import (make_budget, _budget_exhausted, run_career_flow_routed,
                         resume_career_flow, load_checkpoint)

pytestmark = pytest.mark.usefixtures("synthetic_career_profile")

_JD = (
    "岗位名称：大模型应用开发实习生\n公司：示例\n工作地点：上海\n"
    "学历要求：本科及以上\n专业要求：计算机、人工智能\n经验要求：实习\n"
    "技术要求：Python / LangGraph / RAG / FastAPI / Embedding\n工作性质：实习\n"
)


# ── 预算上下文纯函数 ─────────────────────────────────────────

def test_budget_unlimited_when_none():
    assert _budget_exhausted(None) is False
    assert _budget_exhausted(make_budget()) is False        # 两维都 None → 不限


def test_budget_steps_exhaust():
    b = make_budget(max_steps=2)
    assert _budget_exhausted(b) is False
    b["steps"] = 2
    assert _budget_exhausted(b) is True and "max_steps" in b["reason"]


def test_budget_wall_exhaust():
    b = make_budget(max_wall_s=0.0)                          # 0 秒预算 → 立即超限
    assert _budget_exhausted(b) is True and "max_wall_s" in b["reason"]


# ── 故障注入：预算超限优雅降级 ───────────────────────────────

def test_budget_exhaustion_degrades_downstream():
    # max_steps=4：profile/job_input/jd_analyze/match 后预算到顶。
    out = run_career_flow_routed(_JD, jd_title="预算测试", budget=make_budget(max_steps=4))
    assert out.get("match_report")
    assert not out.get("plan_outline")                      # plan 在预算后 → 被跳过
    skipped = [t for t in out.get("trace", []) if t.get("action") == "skipped_budget_exhausted"]
    assert skipped, "应有节点因预算超限被跳过"


# ── 故障注入：崩溃后 checkpoint resume 续跑 ──────────────────

def test_checkpoint_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(career_flow, "CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    run_id = "run_test_001"
    # 模拟"跑到一半崩溃"：小预算只跑前几个节点，但每个执行过的节点已落 checkpoint
    partial = run_career_flow_routed(_JD, jd_title="续跑测试",
                                     budget=make_budget(max_steps=4), run_id=run_id)
    assert not partial.get("plan_outline")                  # 确实没跑完
    ck = load_checkpoint(run_id)
    assert ck and ck["run_id"] == run_id                    # checkpoint 落盘成功
    assert ck["node"] in {"match", "gap", "job_input", "jd_analyze"}
    # 崩溃恢复：从最后检查点续跑到完成
    resumed = resume_career_flow(run_id)
    assert resumed.get("plan_outline")                      # 续跑后完成（plan 节点产出恢复）
    assert any(t.get("node") == "resume" for t in resumed.get("trace", []))


def test_resume_without_checkpoint_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(career_flow, "CHECKPOINT_DIR", str(tmp_path / "empty"))
    try:
        resume_career_flow("does_not_exist")
        assert False, "应抛 FileNotFoundError"
    except FileNotFoundError:
        pass


def test_normal_run_unaffected():
    # 不传 budget/run_id → 行为与改造前一致（完整跑完）
    out = run_career_flow_routed(_JD, jd_title="常规")
    assert out.get("match_report") and out.get("trace")
