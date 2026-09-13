# -*- coding: utf-8 -*-
"""eval_match.py 的回归与契约测试。

不依赖 LLM：保证任何环境（含无 ZHIPU_API_KEY）都能跑。
LLM-as-judge 仅在 KEY 存在时做一次 smoke。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_eval_set_schema_valid():
    from eval_match import load_eval_set, STATUSES, DIRECTIONS

    es = load_eval_set()
    assert es["schema_version"] == "1.0"
    assert isinstance(es["items"], list) and len(es["items"]) >= 8
    seen_ids = set()
    for it in es["items"]:
        for k in ("id", "profile_id", "jd_text",
                  "expected_status", "expected_direction"):
            assert k in it, f"item {it.get('id')} 缺字段 {k}"
        assert it["id"] not in seen_ids, f"重复 id：{it['id']}"
        seen_ids.add(it["id"])
        assert it["expected_status"] in STATUSES
        assert it["expected_direction"] in DIRECTIONS


def test_eval_set_profile_fixtures_exist():
    from eval_match import load_eval_set, load_profile_fixture

    for it in load_eval_set()["items"]:
        prof = load_profile_fixture(it["profile_id"])
        # 必须有 match_job 期望的关键字段
        for k in ("学历", "可接受地域", "明确不做"):
            assert k in prof


def test_baseline_runs_and_meets_threshold():
    """确定性 baseline 必须跑通且整体 status 准确率 ≥ 70%。"""
    from eval_match import run_baseline, load_eval_set

    out = run_baseline(load_eval_set()["items"])
    s = out["summary"]
    assert s["n"] >= 8
    assert s["status_acc"] >= 0.70, (
        f"status 准确率 {s['status_acc']:.2%} 低于 70% 阈值，"
        f"混淆：{s['confusion']}"
    )
    assert s["direction_acc"] >= 0.80, (
        f"direction 准确率 {s['direction_acc']:.2%} 低于 80% 阈值"
    )


def test_baseline_each_bucket_has_samples():
    """三档结论每档都至少有 1 个样本，否则评估覆盖率不够。"""
    from eval_match import run_baseline, load_eval_set, STATUSES

    out = run_baseline(load_eval_set()["items"])
    per = out["summary"]["per_status"]
    for s in STATUSES:
        assert per[s]["n"] >= 1, f"分桶 {s} 样本数=0，需要补 fixture"


def test_results_contain_actual_and_gap_total():
    from eval_match import run_baseline, load_eval_set

    out = run_baseline(load_eval_set()["items"])
    for r in out["results"]:
        assert "actual_status" in r
        assert "actual_direction" in r
        assert isinstance(r["gap_total"], int)


def test_format_report_renders():
    from eval_match import run_baseline, load_eval_set, format_report

    out = run_baseline(load_eval_set()["items"])
    md = format_report(out)
    assert "## match_job 评估报告" in md
    assert "status 准确率" in md
    assert "混淆矩阵" in md


@pytest.mark.skipif(
    not os.environ.get("ZHIPU_API_KEY"),
    reason="需要 ZHIPU_API_KEY 才能跑 LLM-as-judge",
)
def test_llm_judge_smoke():
    """KEY 在时跑 1 条 LLM judge，验证响应结构。"""
    from eval_match import llm_judge_one, load_profile_fixture, load_eval_set

    it = load_eval_set()["items"][0]
    profile = load_profile_fixture(it["profile_id"])
    baseline = {
        "actual_status": it["expected_status"],
        "actual_direction": it["expected_direction"],
        "gap_total": 0,
    }
    out = llm_judge_one(profile, it["jd_text"], baseline)
    assert isinstance(out, dict)
    # 失败也只标 error，不抛
    if "error" not in out:
        assert isinstance(out.get("score"), int)
        assert 1 <= out["score"] <= 5
