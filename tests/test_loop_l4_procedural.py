# -*- coding: utf-8 -*-
"""[L4] Procedural 记忆激活（SOP 闭环）—— 量化测试。

baseline（改造前）：ProceduralMemory 类已建好却**零写零读**（架空）→ proc.list() 恒空。
改造后：CareerFlow 经验 → career_flow_run 事件 → 确定性沉淀方向级 SOP → 规划时按方向查询注入。
量化：SOP 命中数 0 → N（同一方向累计 ≥min_support 次「适合」后沉淀出可复用 SOP）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory_layers
from memory_layers import (EpisodicMemory, ProceduralMemory, record_career_flow_run,
                           distill_procedural_sops, get_active_sops, record_sop_outcome)


def _mem(tmp_path):
    base = str(tmp_path / "mem")
    return EpisodicMemory(base_dir=base), ProceduralMemory(base_dir=base)


def test_procedural_empty_baseline(tmp_path):
    _epi, proc = _mem(tmp_path)
    assert proc.list() == []                    # 架空基线：从未被写过


def test_distill_creates_direction_sop(tmp_path):
    epi, proc = _mem(tmp_path)
    for _ in range(2):                          # 同方向 2 次「适合」
        record_career_flow_run(epi, jd_title="x", status="当前适合投递", direction="主方向")
    res = distill_procedural_sops(epi, proc, min_support=2)
    assert "主方向" in res["directions"]
    sops = proc.list()
    assert len(sops) == 1 and sops[0]["trigger"] == "direction=主方向"   # 0 → 1（命中数提升）


def test_min_support_not_met(tmp_path):
    epi, proc = _mem(tmp_path)
    record_career_flow_run(epi, jd_title="x", status="当前适合投递", direction="主方向")
    distill_procedural_sops(epi, proc, min_support=2)
    assert proc.list() == []                    # 仅 1 次「适合」未达阈值 → 不沉淀


def test_non_fit_not_counted(tmp_path):
    epi, proc = _mem(tmp_path)
    for _ in range(3):
        record_career_flow_run(epi, jd_title="x", status="暂不建议投递", direction="副方向")
    distill_procedural_sops(epi, proc, min_support=2)
    assert proc.list() == []                    # 「不适合」不计入 fit


def test_get_active_sops_context_filter(tmp_path):
    epi, proc = _mem(tmp_path)
    for _ in range(2):
        record_career_flow_run(epi, jd_title="x", status="适合", direction="主方向")
    distill_procedural_sops(epi, proc, min_support=2)
    # 匹配结论只产生候选；必须有独立执行结果才能激活。
    for index, day in enumerate(("2026-06-01", "2026-06-02", "2026-06-02")):
        evidence = epi.append({"kind": "sop_execution", "result": "success", "business_date": day})
        record_sop_outcome(proc, "apply_direction:主方向", evidence["event_id"], "success", day)
    assert get_active_sops(proc, context="当前方向：主方向，AI 应用")      # 命中
    assert get_active_sops(proc, context="完全无关的上下文") == []        # 不命中
    assert get_active_sops(proc, context="") == []                       # 空上下文不注入专用 SOP


def test_learn_from_flow_wiring(tmp_path, monkeypatch):
    # career_flow._learn_from_flow 应写 career_flow_run 事件并（达阈值后）沉淀 SOP
    monkeypatch.setattr(memory_layers, "BASE_DIR_DEFAULT", str(tmp_path / "mem"))
    import career_flow
    for _ in range(2):
        career_flow._learn_from_flow({"jd_title": "T",
                                      "match_report": {"status": "当前适合投递", "direction": "主方向"}})
    epi = EpisodicMemory()                       # 用 monkeypatch 后的默认 base_dir
    runs = [e for e in epi.all() if e.get("kind") == "career_flow_run"]
    assert len(runs) == 2                        # 经验确实写入了 episodic
    sops = ProceduralMemory().list()
    assert sops and sops[0]["lifecycle"] == "candidate"


def test_learn_from_flow_no_direction_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_layers, "BASE_DIR_DEFAULT", str(tmp_path / "mem2"))
    import career_flow
    career_flow._learn_from_flow({"jd_title": "T", "match_report": {}})   # 无 direction
    assert [e for e in EpisodicMemory().all() if e.get("kind") == "career_flow_run"] == []
