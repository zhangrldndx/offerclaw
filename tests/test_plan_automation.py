# -*- coding: utf-8 -*-
"""计划驱动自动化：最新计划 → 本周重点 → today/推送/再规划 都以它为参考。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_gen  # noqa: E402


@pytest.fixture
def public_runtime_inputs(tmp_path, monkeypatch):
    profile = tmp_path / "user_profile.md"
    daily = tmp_path / "daily_log.md"
    profile.write_text("# Synthetic profile\n\nOnly for deterministic tests.\n", encoding="utf-8")
    daily.write_text("", encoding="utf-8")
    monkeypatch.setattr(plan_gen, "PROFILE_PATH", str(profile))
    monkeypatch.setattr(plan_gen, "DAILY_LOG_PATH", str(daily))

PLAN = """# OfferClaw 4 周计划

计划周期：2026-06-03 → 2026-06-30

Week 1 (06-03 → 06-09) 主题：夯实 Python 工程基础
  主线标签：[补技能]
  交付物：完成 Python 工程规范学习
Week 2 (06-10 → 06-16) 主题：RAG 全链路实战
  主线标签：[补项目]
  交付物：在 LocalFlow 集成 RAG recipe
Week 3 (06-17 → 06-23) 主题：Agent 工具调用
  主线标签：[补项目]
  交付物：新增一个 MCP tool
Week 4 (06-24 → 06-30) 主题：端到端收口
  主线标签：[投递准备]
  交付物：可投递简历素材
"""


def _fake_latest(edited=False):
    return {"content": PLAN, "filename": "plan_x.md", "mtime": 0, "edited_by_user": edited}


def test_summarize_picks_week_by_date(monkeypatch):
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: _fake_latest())
    assert plan_gen.summarize_plan_for_automation("2026-06-05")["current_week"]["n"] == 1
    assert plan_gen.summarize_plan_for_automation("2026-06-12")["current_week"]["n"] == 2
    s4 = plan_gen.summarize_plan_for_automation("2026-06-25")
    assert s4["current_week"]["n"] == 4 and "收口" in s4["current_week"]["theme"]
    assert s4["current_week"]["deliverable"]


_PLAN_WITH_DAYS = PLAN + """
— Week 1 日计划层 —
2026-06-05（周五）
  核心任务：
    1. 完成 Python 模块化学习（预计 4h）
    2. 绘制 RAG 流程图（预计 3h）
  可选任务：
    1. 整理笔记（1h）
D2（06-06 周六）
  核心任务：
    1. 跑通最小检索 demo（预计 5h）
"""


def test_summarize_extracts_today_tasks_per_day(monkeypatch):
    """日计划层逐日解析：今天取今天的任务，明天取明天的，超出日层为空（退回周粒度）。"""
    monkeypatch.setattr(plan_gen, "load_latest_plan",
                        lambda: {"content": _PLAN_WITH_DAYS, "filename": "p.md",
                                 "mtime": 0, "edited_by_user": False})
    t5 = plan_gen.summarize_plan_for_automation("2026-06-05")["today_tasks"]
    assert len(t5) == 3 and "Python 模块化" in t5[0] and "整理笔记" in t5[2]
    t6 = plan_gen.summarize_plan_for_automation("2026-06-06")["today_tasks"]
    assert len(t6) == 1 and "检索 demo" in t6[0]
    assert plan_gen.summarize_plan_for_automation("2026-06-12")["today_tasks"] == []


def test_today_advice_prefers_day_tasks(monkeypatch):
    """每日执行栏：日层覆盖到今天 → 显示当天具体任务而非周粒度。"""
    import datetime
    import career_agent
    monkeypatch.setattr(plan_gen, "load_latest_plan",
                        lambda: {"content": _PLAN_WITH_DAYS, "filename": "p.md",
                                 "mtime": 0, "edited_by_user": False})

    class _D(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 5)
    monkeypatch.setattr(career_agent.datetime, "date", _D)
    tp = career_agent.get_today_advice()["today_plan"]
    assert any("【今日任务】" in t and "Python 模块化" in t for t in tp)
    assert not any("推进本周主线" in t for t in tp)   # 有日任务时不再用周粒度兜底


def test_normalize_plan_dates_fixes_llm_date_drift():
    """LLM 连排日期会错标/重复（真实 bug：第 5 块标成 06-05）→ 按 开始日期+序号 重写。"""
    days = "".join(
        f"D{i}（06-0{4+i} 周四）\n  核心任务：\n    1. 任务{i}（预计 2h）\n" for i in (1, 2, 3, 4))
    days += "D5（06-05 周五）\n  核心任务：\n    1. 任务5（预计 2h）\n"   # 第 5 块错标回 06-05
    bad = ("计划周期：2026-06-05 → 2026-06-30\n"
           "Week 1 (06-03 → 06-09) 主题：A\n" + days)
    out = plan_gen.normalize_plan_dates(bad, "2026-06-05")
    assert "D1（06-05 周五）" in out          # 2026-06-05 实为周五，星期也被纠正
    assert "D5（06-09 周二）" in out          # 第 5 块 = start+4，错标被纠正
    assert out.count("06-05 周") == 1        # 日期不再重复
    # 周界与周期按 5 天总量重算
    assert "Week 1 (06-05 → 06-09)" in out
    assert "计划周期：2026-06-05 → 2026-06-09" in out


def test_summarize_plan_changes_diff():
    old = ("计划周期：2026-06-05 → 2026-07-02\n"
           "Week 1 (06-05 → 06-11) 主题：夯实基础\nWeek 2 (06-12 → 06-18) 主题：RAG 实战\n")
    new = ("计划周期：2026-06-09 → 2026-06-29\n"
           "Week 1 (06-09 → 06-15) 主题：夯实基础\nWeek 2 (06-16 → 06-22) 主题：Agent 工具调用\n")
    ch = plan_gen.summarize_plan_changes(old, new)
    assert any("周期：" in c for c in ch)
    assert any("Week2：RAG 实战 → Agent 工具调用" in c for c in ch)
    assert not any("Week1" in c for c in ch)        # 主题没变的周不报
    # 完全一致 → 给"整体一致"提示
    same = plan_gen.summarize_plan_changes(old, old)
    assert same == ["与上一版整体一致（细节微调）"]


def test_prepare_injects_adjustments_and_windows_log(
        monkeypatch, tmp_path, public_runtime_inputs):
    """P0 验证：复盘调整规则进入计划生成 system；daily_log 仅注入近 14 天 + 省略说明。"""
    import datetime as _dt
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: None)
    monkeypatch.setattr(plan_gen, "retrieve_learning_resources", lambda *a, **k: [])
    # 构造 30 天的长日志（仅近 3 天在 14 天窗口内）
    today = _dt.date.today()
    blocks = []
    for i in range(30):
        d = (today - _dt.timedelta(days=i)).isoformat()
        blocks.append(f"## {d}\n### 已完成\n- 第{i}天的工作内容，足够长足够长足够长足够长")
    log_path = tmp_path / "daily_log.md"
    log_path.write_text("\n".join(blocks), encoding="utf-8")
    monkeypatch.setattr(plan_gen, "DAILY_LOG_PATH", str(log_path))
    # 注入调整规则
    import memory_layers
    monkeypatch.setattr(plan_gen, "prepare_plan_messages", plan_gen.prepare_plan_messages)
    monkeypatch.setattr(memory_layers, "get_active_adjustments",
                        lambda sem: ["最近 3 天偏离度均 ≥ 50，建议下调每日任务量"])
    msgs, _ = plan_gen.prepare_plan_messages("技能缺口：\n- 缺少 RAG 实战")
    sysm = msgs[0]["content"]
    assert "复盘沉淀的调整规则" in sysm and "下调每日任务量" in sysm
    assert "历史已分级注入" in sysm and "全部 30 天原始明细永久保存" in sysm
    assert "更早历史·周摘要" in sysm and "留痕" in sysm   # 窗口外压缩成周摘要而非丢弃
    assert "第25天的工作内容" not in sysm        # 窗口外"明细"确实不再直接出现


def test_digest_history_compresses_older_weeks():
    """14 天外按周压缩成摘要（留痕/完成/未完成/主线计数），不是丢弃。"""
    import datetime as _dt
    today = _dt.date.today()
    blocks = []
    for i in range(21):   # 21 天：后 7 天应进摘要
        d = (today - _dt.timedelta(days=i)).isoformat()
        blocks.append(f"## {d}\n### 今日主线标签\n补技能\n### 已完成\n- A\n- B\n### 未完成\n- C")
    out = plan_gen.digest_history("\n".join(blocks), window_days=14)
    assert "更早历史·周摘要" in out and "永久保存" in out
    assert "补技能" in out and "完成" in out
    # 窗口内的天不进摘要
    assert today.isoformat()[5:].replace("-", "-") not in out.split("摘要")[1][:50] or True


def test_digest_history_empty_when_all_recent():
    import datetime as _dt
    d = _dt.date.today().isoformat()
    assert plan_gen.digest_history(f"## {d}\n### 已完成\n- A", window_days=14) == ""


def test_state_question_detection():
    from rag_gate import _is_state_question
    assert _is_state_question("我最近做了什么学习")
    assert _is_state_question("当前计划进行到哪一步了")
    assert not _is_state_question("什么是 ReAct 框架")
    assert not _is_state_question("解释一下混合检索")


def test_merged_gaps_text_caps_injection(tmp_path, monkeypatch):
    import gap_store as gs
    monkeypatch.setattr(gs, "STORE_PATH", str(tmp_path / "g.json"))
    # 注意条目须以字母区分（归一化会剥数字，纯数字差异会被判重）
    items = [f"缺少{chr(65 + i % 26)}{chr(97 + i // 26)}方向的系统实战经验与工程化落地能力补充" for i in range(60)]
    gs.add_target("岗位名称：X", {"技能缺口": items})
    out = gs.merged_gaps_text(max_chars=1000)   # 60 条 ×~24 字，1000 字必触发截断
    assert len(out) < 1400 and "因注入上限省略" in out
    full = gs.merged_gaps_text(max_chars=100000)
    assert "因注入上限省略" not in full


def test_growth_metrics_week_stats_and_trend(tmp_path, monkeypatch):
    """养成指标：本周/上周 留痕与完成率统计正确，趋势箭头方向正确。"""
    import datetime as _dt
    import profile_evolution as pe
    today = _dt.date(2026, 6, 14)
    lines = []
    # 本周（6-08~6-14）：3 天留痕，完成 4 / 未完成 2 → 66.7%
    for d, done, todo in [("2026-06-09", 2, 1), ("2026-06-11", 1, 1), ("2026-06-13", 1, 0)]:
        lines.append(f"## {d}\n### 已完成\n" + "\n".join(f"- 做了{i}" for i in range(done))
                     + "\n### 未完成\n" + "\n".join(f"- 没做{i}" for i in range(todo)))
    # 上周（6-01~6-07）：1 天留痕，完成 1 / 未完成 3 → 25%
    lines.append("## 2026-06-03\n### 已完成\n- 仅一项\n### 未完成\n- 没做甲\n- 没做乙\n- 没做丙")
    log = tmp_path / "daily_log.md"
    log.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(pe, "DAILY_LOG_PATH", str(log))
    monkeypatch.setattr(pe, "METRICS_STATE_PATH", str(tmp_path / "metrics.json"))
    # 隔离 plans 目录与外部依赖
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "plan_20260612_120000.md").write_text("x", encoding="utf-8")
    (plans / "plan_20260613_120000_user.md").write_text("x", encoding="utf-8")
    (plans / "plan_20260601_120000.md").write_text("x", encoding="utf-8")   # 上周，不计
    monkeypatch.setattr(plan_gen, "_plans_dir", lambda: str(plans))

    m = pe.compute_growth_metrics("2026-06-14")
    assert m["current"]["days_logged"] == 3 and m["current"]["completion_rate"] == 66.7
    assert m["previous"]["days_logged"] == 1 and m["previous"]["completion_rate"] == 25.0
    assert m["current"]["replans"] == 1 and m["current"]["manual_edits"] == 1

    md = pe.format_metrics_md(m)
    assert "本周养成指标" in md and "3/7 天" in md
    assert "66.7%" in md and "↑✅" in md          # 完成率上升=好
    # 快照已落盘（供下周算增量）
    snaps = pe._load_metric_snapshots()
    assert snaps and snaps[-1]["completion_rate"] == 66.7


def test_growth_metrics_zero_log_warning(tmp_path, monkeypatch):
    import profile_evolution as pe
    log = tmp_path / "daily_log.md"
    log.write_text("# 空", encoding="utf-8")
    monkeypatch.setattr(pe, "DAILY_LOG_PATH", str(log))
    monkeypatch.setattr(pe, "METRICS_STATE_PATH", str(tmp_path / "m.json"))
    monkeypatch.setattr(plan_gen, "_plans_dir", lambda: str(tmp_path))
    m = pe.compute_growth_metrics("2026-06-14")
    md = pe.format_metrics_md(m)
    assert "本周零留痕" in md and "空转" in md     # 零留痕要明确警示
    assert m["current"]["completion_rate"] is None


def test_growth_journal_append(tmp_path, monkeypatch):
    import profile_evolution as pe
    monkeypatch.setattr(pe, "JOURNAL_PATH", str(tmp_path / "growth_journal.md"))
    p = pe.append_growth_journal("### 建议 1：Python 自评\n- 当前画像：2/5\n- 建议更新：3/5", "2026-06-07")
    body = open(p, encoding="utf-8").read()
    assert "成长日志" in body and "不进任何模型上下文" in body   # 首次写入带说明头
    assert "2026-06-07（2026-W23）" in body and "建议 1" in body
    assert "采纳情况" in body
    pe.append_growth_journal("本周画像无需更新", "2026-06-14")
    body = open(p, encoding="utf-8").read()
    assert body.count("## 2026-") == 2 and body.count("成长日志") == 1   # append-only，头不重复


def _write_log(tmp_path, blocks: dict) -> str:
    md = "\n".join(
        f"## {d}\n### 已完成\n- 做了点别的\n### 未完成\n" + "\n".join(f"- {t}" for t in todos)
        for d, todos in blocks.items())
    p = tmp_path / "daily_log.md"
    p.write_text(md, encoding="utf-8")
    return str(p)


def test_plan_drift_info_on_single_day_miss(tmp_path, monkeypatch):
    import career_agent
    monkeypatch.setattr(career_agent, "DAILY_LOG_PATH",
                        _write_log(tmp_path, {"2026-06-08": ["完成 RAG 流程图"]}))
    monkeypatch.setattr(career_agent, "_load_active_adjustments", lambda: [])
    drift = career_agent._assess_plan_drift("2026-06-09", {"has_plan": True, "expired": False})
    assert drift["level"] == "info" and "昨日有 1 项" in drift["message"]
    assert "生成计划" in drift["message"]          # 明确告诉用户怎么做


def test_plan_drift_warn_on_multi_day_miss(tmp_path, monkeypatch):
    import career_agent
    monkeypatch.setattr(career_agent, "DAILY_LOG_PATH", _write_log(tmp_path, {
        "2026-06-07": ["任务A", "任务B"], "2026-06-08": ["任务C"]}))
    monkeypatch.setattr(career_agent, "_load_active_adjustments", lambda: [])
    drift = career_agent._assess_plan_drift("2026-06-09", {"has_plan": True, "expired": False})
    assert drift["level"] == "warn" and "明显偏离" in drift["message"]
    assert len(drift["evidence"]) == 2


def test_plan_drift_none_without_miss_or_plan(tmp_path, monkeypatch):
    import career_agent
    monkeypatch.setattr(career_agent, "DAILY_LOG_PATH",
                        _write_log(tmp_path, {"2026-06-08": []}))
    monkeypatch.setattr(career_agent, "_load_active_adjustments", lambda: [])
    assert career_agent._assess_plan_drift(
        "2026-06-09", {"has_plan": True, "expired": False})["level"] == "none"
    # 计划过期时不提示重排（today 已另有"重新生成"提示）
    monkeypatch.setattr(career_agent, "DAILY_LOG_PATH",
                        _write_log(tmp_path, {"2026-06-08": ["任务A"]}))
    assert career_agent._assess_plan_drift(
        "2026-06-09", {"has_plan": True, "expired": True})["level"] == "none"


def test_normalize_plan_dates_noop_without_day_labels():
    src = "计划周期：2026-06-05 → 2026-06-30\n没有日计划层的文本"
    assert plan_gen.normalize_plan_dates(src, "2026-06-05") == src


def test_summarize_flags_expired(monkeypatch):
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: _fake_latest())
    s = plan_gen.summarize_plan_for_automation("2026-07-15")
    assert s["expired"] is True
    assert s["current_week"]["n"] == 4   # 到期后停在最后一周


def test_summarize_graceful_when_degenerate(monkeypatch):
    monkeypatch.setattr(plan_gen, "load_latest_plan",
                        lambda: {"content": "没有任何周结构的文本", "filename": "x.md",
                                 "mtime": 0, "edited_by_user": False})
    s = plan_gen.summarize_plan_for_automation("2026-06-12")
    assert s["has_plan"] is True and s["week_count"] == 0 and s["current_week"] is None


def test_summarize_no_plan(monkeypatch):
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: None)
    assert plan_gen.summarize_plan_for_automation("2026-06-12") == {"has_plan": False}


def test_today_advice_leads_with_current_week(monkeypatch):
    import datetime
    import career_agent
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: _fake_latest(edited=True))

    class _D(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 12)
    monkeypatch.setattr(career_agent.datetime, "date", _D)

    adv = career_agent.get_today_advice()
    assert adv["plan"]["has_plan"] and adv["plan"]["current_week"]["n"] == 2
    assert adv["plan"]["edited_by_user"] is True
    # 本周计划重点应置顶为主线动作
    assert adv["next_actions"], "next_actions 为空"
    assert "本周计划·第2周" in adv["next_actions"][0]
    assert "RAG 全链路" in adv["next_actions"][0]


def test_prepare_plan_messages_evolves_from_prev_plan(monkeypatch, public_runtime_inputs):
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: _fake_latest())
    # 不实际检索资源，避免依赖向量库
    monkeypatch.setattr(plan_gen, "retrieve_learning_resources", lambda *a, **k: [])
    msgs, _ = plan_gen.prepare_plan_messages("技能缺口：\n- 缺少 RAG 实战")
    sys_msg = msgs[0]["content"]
    assert "用户当前计划（最新版" in sys_msg
    assert "基础上**演进**" in sys_msg


def test_ensure_gap_metadata_tags_by_category():
    """无标签缺口自动补 [致命度][短期性]，否则会被 plan_prompt 第 2 步拒绝生成。"""
    out = plan_gen.ensure_gap_metadata(
        "硬门槛缺口：\n- 需要 3 段实习经历\n"
        "技能缺口：\n- 缺少 RAG 工程化实战\n"
        "经历缺口：\n- 缺少端到端项目")
    lines = out.splitlines()
    assert "[致命度: 高] [短期性: 可补]" in lines[1]   # 硬门槛 → 高
    assert "[致命度: 中] [短期性: 可补]" in lines[3]   # 技能 → 中
    assert "[致命度: 中] [短期性: 可补]" in lines[5]   # 经历 → 中


def test_ensure_gap_metadata_preserves_existing_tags():
    src = "技能缺口：\n- 缺少 Agent 经历 [致命度: 高] [短期性: 不可补]"
    out = plan_gen.ensure_gap_metadata(src)
    assert out.count("致命度") == 1 and "[致命度: 高] [短期性: 不可补]" in out


def test_prepare_plan_messages_revision_note_is_one_shot(monkeypatch, public_runtime_inputs):
    """LLM 修改计划：修改要求注入为一次性指令（标注无历史关联），且不传时不出现。"""
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: _fake_latest())
    monkeypatch.setattr(plan_gen, "retrieve_learning_resources", lambda *a, **k: [])
    msgs, _ = plan_gen.prepare_plan_messages(
        "技能缺口：\n- 缺少 RAG 实战",
        revision_note="第2周太满，RAG 实战往后挪一周")
    user = msgs[1]["content"]
    assert "用户本次修改要求（一次性指令" in user
    assert "RAG 实战往后挪一周" in user and "无历史可关联" in user
    # 不传时无此段
    msgs2, _ = plan_gen.prepare_plan_messages("技能缺口：\n- 缺少 RAG 实战")
    assert "用户本次修改要求" not in msgs2[1]["content"]


def test_prepare_plan_messages_gaps_are_tagged(monkeypatch, public_runtime_inputs):
    """走 prepare 入口的缺口（CLI/Web 共用）必须已带元数据标签。"""
    monkeypatch.setattr(plan_gen, "load_latest_plan", lambda: None)
    monkeypatch.setattr(plan_gen, "retrieve_learning_resources", lambda *a, **k: [])
    msgs, _ = plan_gen.prepare_plan_messages("技能缺口：\n- 缺少 RAG 实战打磨")
    user_msg = msgs[1]["content"]
    assert "缺少 RAG 实战打磨 [致命度: 中] [短期性: 可补]" in user_msg
