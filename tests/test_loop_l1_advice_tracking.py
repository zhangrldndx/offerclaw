# -*- coding: utf-8 -*-
"""[L1] 外循环执行追踪 checkpoint —— 故障注入式量化测试。

注入场景：连续 N 天唤醒、状态冻结（同一面试中投递、无新日志）。
- 改造前行为：每天给完全相同的 headline，无任何「已重复」感知（cold start 健忘）。
- 改造后行为：连续重复达阈值 → execution_tracking.escalated=True + 升级动作，建议不再原样重复。
量化指标：重复建议「未被感知」的天数应被阈值封顶，而非 = N。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import career_agent
from career_agent import compute_execution_tracking, _advice_signature, get_today_advice
from memory_layers import SemanticMemory


# ── 纯函数：执行追踪逻辑 ─────────────────────────────────────

def test_first_advice_no_repeat():
    t, esc, rec = compute_execution_tracking("sigA", "建议X", "2026-06-22", {}, progressed=True)
    assert t["repeat_count"] == 0 and t["escalated"] is False and esc is None
    assert rec["signature"] == "sigA"


def test_same_signature_not_progressed_accumulates(monkeypatch):
    # 2026-08-08 修正:原版 4 次调用共用同一天日期,恰把「刷新页面就 +1」的 bug 固化为预期。
    # "连续 N 天"必须以日期推进为准,故逐日推进模拟。
    monkeypatch.setattr(career_agent, "ADVICE_ESCALATE_AFTER", 2)
    rec = {}
    counts = []
    for day in ("2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25"):
        sig = "sigA"
        # 模拟「同建议、无推进」：progressed 由 signature 是否变化决定
        progressed = (not rec) or (sig != rec.get("signature"))
        t, esc, rec = compute_execution_tracking(sig, "建议X", day, rec, progressed)
        counts.append((t["repeat_count"], t["escalated"]))
    # 0,1,2(→escalated),3(escalated)
    assert [c[0] for c in counts] == [0, 1, 2, 3]
    assert [c[1] for c in counts] == [False, False, True, True]


def test_same_day_refresh_is_idempotent(monkeypatch):
    """同一天内反复唤醒(刷新页面)不得累加——bug 场景:刷 N 次涨到'连续 N 天'。"""
    monkeypatch.setattr(career_agent, "ADVICE_ESCALATE_AFTER", 2)
    # 先跨两天累到 1
    _t, _e, rec = compute_execution_tracking("sigA", "X", "2026-06-22", {}, progressed=True)
    t, _e, rec = compute_execution_tracking("sigA", "X", "2026-06-23", rec, progressed=False)
    assert t["repeat_count"] == 1
    for _ in range(5):   # 同日刷 5 次
        t, _e, rec = compute_execution_tracking("sigA", "X", "2026-06-23", rec, progressed=False)
    assert t["repeat_count"] == 1          # 纹丝不动
    # 次日再唤醒才 +1
    t, _e, rec = compute_execution_tracking("sigA", "X", "2026-06-24", rec, progressed=False)
    assert t["repeat_count"] == 2


def test_progress_resets_repeat(monkeypatch):
    monkeypatch.setattr(career_agent, "ADVICE_ESCALATE_AFTER", 2)
    last = {"signature": "sigA", "headline": "X", "date": "2026-06-20", "repeat_count": 5}
    t, esc, rec = compute_execution_tracking("sigA", "X", "2026-06-22", last, progressed=True)
    assert t["repeat_count"] == 0 and t["escalated"] is False and esc is None


def test_signature_change_resets(monkeypatch):
    monkeypatch.setattr(career_agent, "ADVICE_ESCALATE_AFTER", 2)
    last = {"signature": "sigA", "headline": "X", "date": "2026-06-20", "repeat_count": 5}
    t, esc, rec = compute_execution_tracking("sigB", "Y", "2026-06-22", last, progressed=False)
    assert t["repeat_count"] == 0 and t["escalated"] is False


def test_signature_strips_date():
    assert _advice_signature("【ExampleCo · 实习】准备面试 2026-06-22") == _advice_signature("【ExampleCo · 实习】准备面试 2026-06-23")


# ── 故障注入：连续唤醒、状态冻结 ─────────────────────────────

_APPS_FROZEN = """# 投递池

| 日期 | 公司 | 岗位 | 来源 | 当前状态 | 下一步动作 |
|---|---|---|---|---|---|
| 2026-06-01 | ExampleCo | 大模型应用实习 | synthetic | 面试中 | 刷题 |
"""

_APPS_CHANGED = """# 投递池

| 日期 | 公司 | 岗位 | 来源 | 当前状态 | 下一步动作 |
|---|---|---|---|---|---|
| 2026-06-01 | 字节 | Agent 实习 | official | 准备投递 | 定稿简历 |
"""


def _setup(tmp_path, monkeypatch, apps_md):
    apps = tmp_path / "applications.md"
    log = tmp_path / "daily_log.md"
    apps.write_text(apps_md, encoding="utf-8")
    log.write_text("", encoding="utf-8")          # 冻结：无日志推进
    monkeypatch.setattr(career_agent, "APPLICATIONS_PATH", str(apps))
    monkeypatch.setattr(career_agent, "DAILY_LOG_PATH", str(log))
    monkeypatch.setattr(career_agent, "ADVICE_ESCALATE_AFTER", 2)
    return apps


def test_frozen_state_escalates_and_breaks_blind_repeat(tmp_path, monkeypatch):
    apps = _setup(tmp_path, monkeypatch, _APPS_FROZEN)
    sem = SemanticMemory(base_dir=str(tmp_path / "mem"))

    headlines, escalated_flags, repeat_counts, blind_unacked = [], [], [], 0
    prev_headline = None
    # 2026-08-08 修正:原版 5 次共用同一天,同日幂等修复后必须逐日推进才是"连续 5 天"
    for day in ("2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26"):
        adv = get_today_advice(sem=sem, today=day)
        et = adv["execution_tracking"]
        headlines.append(adv["headline"])
        escalated_flags.append(et["escalated"])
        repeat_counts.append(et["repeat_count"])
        # 「盲目重复」= 与上次同 headline、未升级、未推进
        if prev_headline == adv["headline"] and not et["escalated"] and not et["progressed"]:
            blind_unacked += 1
        prev_headline = adv["headline"]

    # headline 本体稳定（同一面试中投递）
    assert headlines[0].startswith("【ExampleCo")
    # repeat_count 持续累加不归零（证明「记得到哪了」，封顶行为可见）
    assert repeat_counts == [0, 1, 2, 3, 4]
    # 阈值后开始升级感知，且升级状态持续（非一次性）
    assert escalated_flags == [False, False, True, True, True]
    # 升级时注入了【执行追踪】动作
    adv_last = get_today_advice(sem=sem, today="2026-06-22")
    assert any("【执行追踪】" in a for a in adv_last["next_actions"])
    # 量化：盲目未感知的重复天数被阈值封顶（=2），而非 5（改造前会是 4）
    assert blind_unacked <= 2


def test_state_change_resets_tracking(tmp_path, monkeypatch):
    apps = _setup(tmp_path, monkeypatch, _APPS_FROZEN)
    sem = SemanticMemory(base_dir=str(tmp_path / "mem"))
    # 2026-08-08 修正:累积必须逐日推进(同日重复唤醒已幂等)
    for day in ("2026-06-22", "2026-06-23", "2026-06-24"):
        get_today_advice(sem=sem, today=day)
    assert get_today_advice(sem=sem, today="2026-06-25")["execution_tracking"]["escalated"] is True
    # 状态变化（换了投递）→ 重复计数归零、不再升级
    apps.write_text(_APPS_CHANGED, encoding="utf-8")
    et = get_today_advice(sem=sem, today="2026-06-25")["execution_tracking"]
    assert et["repeat_count"] == 0 and et["escalated"] is False


def test_same_day_double_wakeup_via_advice(tmp_path, monkeypatch):
    """端到端复现 bug 场景:同一天两次 GET /api/today(刷新页面)计数必须不变。"""
    _setup(tmp_path, monkeypatch, _APPS_FROZEN)
    sem = SemanticMemory(base_dir=str(tmp_path / "mem"))
    get_today_advice(sem=sem, today="2026-06-22")
    a1 = get_today_advice(sem=sem, today="2026-06-23")
    a2 = get_today_advice(sem=sem, today="2026-06-23")   # 同日刷新
    assert a1["execution_tracking"]["repeat_count"] == 1
    assert a2["execution_tracking"]["repeat_count"] == 1
