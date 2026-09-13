# -*- coding: utf-8 -*-
"""今日计划(整体计划"当日切片")测试:解析 / 灵活选日 / 无损回写 / 双向同步 / API。

隔离纪律:
- plans/ 重定向到 tmp(monkeypatch plan_gen._plans_dir),不碰真实计划;
- 记忆层打桩(EpisodicMemory),API 测试不往真实记忆文件写事件;
- fixture 日期以"今天"为锚动态生成,测试不随日历过期。
"""
from __future__ import annotations

import datetime
import glob
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import plan_gen  # noqa: E402
from plan_daily import (  # noqa: E402
    DayNotFoundError,
    StalePlanError,
    get_today_view,
    parse_plan_days,
    replace_day_tasks,
    save_today,
)

_WD = "一二三四五六日"


def build_plan(start: datetime.date, n_days: int = 14) -> str:
    """按真实计划样式(plan_20260605 系列)构造一份含周/日两层的计划。"""
    end = start + datetime.timedelta(days=n_days - 1)
    lines = ["# 学习计划", "",
             f"计划周期：{start.isoformat()} → {end.isoformat()}", "",
             "— 周计划层 —"]
    for w in range(1, (n_days + 6) // 7 + 1):
        ws = start + datetime.timedelta(days=(w - 1) * 7)
        we = min(ws + datetime.timedelta(days=6), end)
        lines += [f"Week {w} ({ws.strftime('%m-%d')} → {we.strftime('%m-%d')}) 主题：主题{w}",
                  f"  交付物：交付{w}"]
    lines += ["", "— 日计划层（覆盖全周期，逐日） —"]
    for i in range(1, n_days + 1):
        d = start + datetime.timedelta(days=i - 1)
        lines += [f"D{i}（{d.strftime('%m-%d')} 周{_WD[d.weekday()]}）",
                  "  主线标签：[补技能]",
                  "  核心任务：",
                  f"    1. 第{i}天任务甲（预计 3h）",
                  f"    2. 第{i}天任务乙「来源：test.md」",
                  "  可选任务：",
                  f"    - 第{i}天可选项",
                  ""]
    return "\n".join(lines) + "\n"


@pytest.fixture()
def plans_tmp(tmp_path, monkeypatch):
    """plans/ 隔离到 tmp;返回 (目录, 写 fixture 计划的函数)。"""
    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plan_gen, "_plans_dir", lambda: str(pdir))

    def write(content: str, name: str = "plan_20250101_000000.md"):
        p = pdir / name
        p.write_text(content, encoding="utf-8")
        past = datetime.datetime.now().timestamp() - 3600  # 固定为过去,避免 mtime 平手
        os.utime(p, (past, past))
        return p

    return pdir, write


@pytest.fixture()
def std_plan(plans_tmp):
    """今天在周期正中的 14 天计划(today-7 → today+6),永不随日历过期。"""
    start = datetime.date.today() - datetime.timedelta(days=7)
    _pdir, write = plans_tmp
    content = build_plan(start, 14)
    write(content)
    return start, content


# ---------- 解析 ----------

def test_parse_days_structure(std_plan):
    start, content = std_plan
    parsed = parse_plan_days(content)
    assert parsed["period_start"] == start
    assert len(parsed["days"]) == 14
    dates = [d["date"] for d in parsed["days"]]
    assert dates == [start + datetime.timedelta(days=i) for i in range(14)]
    d1 = parsed["days"][0]
    assert [t for _l, _i, t in d1["task_lines"]] == ["第1天任务甲（预计 3h）", "第1天任务乙「来源：test.md」"]
    assert d1["optional_lines"] == ["第1天可选项"]
    assert parsed["days"][7]["week_n"] == 2 and parsed["weeks"][2] == "主题2"


# ---------- 灵活选日 ----------

def test_view_exact_today(std_plan):
    start, _ = std_plan
    v = get_today_view()  # 不传日期 = UI 打开当天
    assert v["has_plan"] and v["has_daily"] and v["status"] == "in_range"
    assert v["period_start"] == start.isoformat()          # 供前端预填日期框
    assert v["period_end"] == (start + datetime.timedelta(days=13)).isoformat()
    assert v["date"] == datetime.date.today().isoformat()
    assert v["day_index"] == 8 and v["total_days"] == 14
    assert v["week_n"] == 2 and v["week_theme"] == "主题2"
    assert v["tasks"] == ["第8天任务甲（预计 3h）", "第8天任务乙「来源：test.md」"]


def test_view_before_and_after_range(std_plan):
    start, _ = std_plan
    before = (start - datetime.timedelta(days=3)).isoformat()
    after = (start + datetime.timedelta(days=30)).isoformat()
    vb = get_today_view(before)
    assert vb["status"] == "before_start" and vb["date"] == start.isoformat()
    va = get_today_view(after)
    assert va["status"] == "after_end"
    assert va["date"] == (start + datetime.timedelta(days=13)).isoformat()


def test_view_no_plan_and_no_daily(plans_tmp):
    _pdir, write = plans_tmp
    assert get_today_view()["has_plan"] is False
    write("# 学习计划\n计划周期：2026-01-01 → 2026-01-28\nWeek 1 主题：仅周层\n")
    v = get_today_view()
    assert v["has_plan"] is True and v["has_daily"] is False and v["status"] == "no_daily"
    assert v["period_start"] == "2026-01-01"               # 无日层也返回周期供预填


# ---------- 无损回写 ----------

def test_replace_roundtrip_lossless(std_plan):
    start, content = std_plan
    target = start + datetime.timedelta(days=2)  # D3
    new = replace_day_tasks(content, target, ["新任务一", "新任务二", "新任务三"])
    parsed = parse_plan_days(new)
    d3 = next(d for d in parsed["days"] if d["date"] == target)
    assert [t for _l, _i, t in d3["task_lines"]] == ["新任务一", "新任务二", "新任务三"]
    assert d3["optional_lines"] == ["第3天可选项"]          # 可选任务不动
    assert "  主线标签：[补技能]" in new                      # 非编号行保留
    for i, d in enumerate(parsed["days"], 1):               # 其他天逐条不变
        if d["date"] == target:
            continue
        assert [t for _l, _i2, t in d["task_lines"]] == [
            f"第{i}天任务甲（预计 3h）", f"第{i}天任务乙「来源：test.md」"]
    assert new.count("核心任务：") == content.count("核心任务：")


def test_replace_empty_clears_day(std_plan):
    start, content = std_plan
    target = start + datetime.timedelta(days=4)
    new = replace_day_tasks(content, target, [])
    d5 = next(d for d in parse_plan_days(new)["days"] if d["date"] == target)
    assert d5["task_lines"] == [] and d5["optional_lines"] == ["第5天可选项"]


def test_replace_unknown_date_raises(std_plan):
    _start, content = std_plan
    with pytest.raises(DayNotFoundError):
        replace_day_tasks(content, datetime.date(1999, 1, 1), ["x"])


# ---------- gpt-5.6 Markdown 装饰风格(### 日头 + 加粗标签 bullet + 全角括号周头) ----------

def build_plan_gpt_style(start: datetime.date, n_days: int = 7) -> str:
    """复刻 plan_20260808_165029.md 的实测 gpt-5.6 输出结构。"""
    end = start + datetime.timedelta(days=n_days - 1)
    lines = ["========== OfferClaw 学习计划 ==========", "",
             f"计划周期：{start.isoformat()} → {end.isoformat()}（共 {n_days} 天）", "",
             f"### Week 1（{start.strftime('%m-%d')} → {end.strftime('%m-%d')}）主题：主题甲", ""]
    for i in range(1, n_days + 1):
        d = start + datetime.timedelta(days=i - 1)
        lines += [f"### D{i}（{d.strftime('%m-%d')} 周{_WD[d.weekday()]}）", "",
                  "- **今日主线标签**：[补项目]",
                  "- **核心任务**：",
                  f"  1. 第{i}天主任务甲（预计 2h）。",
                  f"  2. 第{i}天主任务乙（预计 5h）。",
                  "- **可选任务**：",
                  f"  - 建议你第{i}天做可选整理。",
                  "- **预计投入**：8h / 当日上限 9h", ""]
    return "\n".join(lines) + "\n"


@pytest.fixture()
def gpt_plan(plans_tmp):
    start = datetime.date.today() - datetime.timedelta(days=3)
    _pdir, write = plans_tmp
    content = build_plan_gpt_style(start, 7)
    write(content)
    return start, content


def test_parse_gpt_markdown_style(gpt_plan):
    start, content = gpt_plan
    parsed = parse_plan_days(content)
    assert len(parsed["days"]) == 7
    d1 = parsed["days"][0]
    assert [t for _l, _i, t in d1["task_lines"]] == ["第1天主任务甲（预计 2h）。", "第1天主任务乙（预计 5h）。"]
    assert d1["optional_lines"] == ["建议你第1天做可选整理。"]      # 标签 bullet 不混入可选
    assert not any("主线标签" in o or "预计投入" in o or "核心任务" in o
                   for d in parsed["days"] for o in d["optional_lines"])
    assert parsed["weeks"][1] == "主题甲"                          # 全角括号周头可解析
    v = get_today_view()
    assert v["status"] == "in_range" and v["day_index"] == 4 and len(v["tasks"]) == 2


def test_replace_gpt_style_preserves_labels(gpt_plan):
    start, content = gpt_plan
    target = start + datetime.timedelta(days=1)  # D2
    new = replace_day_tasks(content, target, ["改后任务"])
    assert new.count("- **核心任务**：") == content.count("- **核心任务**：")
    assert new.count("- **预计投入**：8h / 当日上限 9h") == content.count("- **预计投入**：8h / 当日上限 9h")
    parsed = parse_plan_days(new)
    d2 = next(d for d in parsed["days"] if d["date"] == target)
    assert [t for _l, _i, t in d2["task_lines"]] == ["改后任务"]
    assert d2["optional_lines"] == ["建议你第2天做可选整理。"]
    d1 = parsed["days"][0]
    assert [t for _l, _i, t in d1["task_lines"]] == ["第1天主任务甲（预计 2h）。", "第1天主任务乙（预计 5h）。"]


def test_normalize_dates_keeps_markdown_decoration(plans_tmp):
    """日期归一必须认 ### 装饰头(LLM 手排日期不可信):错排一天的 gpt 风格计划被确定性纠正。"""
    from plan_gen import normalize_plan_dates
    start = datetime.date(2026, 8, 8)
    wrong = build_plan_gpt_style(start + datetime.timedelta(days=1), 7)  # 标签整体错后一天
    wrong = wrong.replace(f"计划周期：{(start + datetime.timedelta(days=1)).isoformat()}",
                          f"计划周期：{start.isoformat()}")
    fixed = normalize_plan_dates(wrong, start.isoformat())
    parsed = parse_plan_days(fixed)
    assert [d["date"] for d in parsed["days"]] == [start + datetime.timedelta(days=i) for i in range(7)]
    assert "### D1（08-08 周六）" in fixed                          # 装饰保留 + 日期已纠正
    assert "Week 1（08-08 → 08-14）" in fixed                       # 全角周界重写生效


# ---------- 已知模型笔迹矩阵(保障措施:parse 与 normalize 两把正则钉在同一矩阵上) ----------

@pytest.mark.parametrize("header,task", [
    ("D1（08-08 周六）", "    1. 任务甲"),            # qwen 裸头
    ("### D1（08-08 周六）", "  1. 任务甲"),          # gpt-5.6 实测:Markdown 标题头
    ("**D1（08-08 周六）**", "1. 任务甲"),            # 加粗头
    ("#### D1(08-08 周六)", "1、任务甲"),             # 半角括号 + 顿号编号
    ("  D1（2026-08-08 周六）", "  1. 任务甲"),       # 带年份 + 缩进
])
def test_style_matrix_parse_and_normalize(header, task):
    """新模型笔迹进矩阵即钉死:任一条目 parse 或 normalize 失守都会红。"""
    from plan_gen import normalize_plan_dates
    content = ("计划周期：2026-08-08 → 2026-08-08\n"
               "Week 1（08-08 → 08-08）主题：T\n\n" + header + "\n" + task + "\n")
    parsed = parse_plan_days(content)
    assert len(parsed["days"]) == 1
    assert [t for _l, _i, t in parsed["days"][0]["task_lines"]] == ["任务甲"]
    fixed = normalize_plan_dates(content, "2026-08-08")
    assert "08-08 周六" in fixed                       # 日期确定性重写认同款笔迹
    assert len(parse_plan_days(fixed)["days"]) == 1    # 重写后仍可解析(一致性守卫)


# ---------- save_today:落盘 + 同步 + 并发防护 ----------

def test_save_today_persists_and_syncs(plans_tmp, std_plan):
    pdir, _write = plans_tmp
    start, _ = std_plan
    target = (start + datetime.timedelta(days=1)).isoformat()
    view = save_today(target, ["  改后的任务  ", "", "第二条"])
    assert view["tasks"] == ["改后的任务", "第二条"] and view["edited_by_user"] is True
    assert glob.glob(os.path.join(str(pdir), "*_user.md"))
    latest = plan_gen.load_latest_plan()
    assert "改后的任务" in latest["content"]                 # 今日 → 整体已同步
    assert get_today_view(target)["tasks"] == ["改后的任务", "第二条"]


def test_save_today_stale_mtime_guard(std_plan):
    start, _ = std_plan
    with pytest.raises(StalePlanError):
        save_today((start + datetime.timedelta(days=1)).isoformat(), ["x"], base_mtime=123)


# ---------- API 层(含双向同步闭环) ----------

@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import rag_api
    return TestClient(rag_api.app)


def test_api_today_view_shape(client, std_plan):
    r = client.get("/api/plan/today")
    assert r.status_code == 200
    j = r.json()
    assert j["has_plan"] and j["has_daily"] and j["status"] == "in_range"
    assert j["tasks"] and j["plan_mtime"] > 0


def test_api_bidirectional_sync(client, std_plan):
    start, _ = std_plan
    d2 = (start + datetime.timedelta(days=1)).isoformat()
    # 方向 1:改今日 → 整体计划跟着变
    v = client.get(f"/api/plan/today?date={d2}").json()
    r = client.post("/api/plan/today/save",
                    json={"date": d2, "tasks": ["今日新任务"], "base_mtime": v["plan_mtime"]})
    assert r.status_code == 200 and r.json()["tasks"] == ["今日新任务"]
    overall = client.get("/api/plan/current").json()
    assert "今日新任务" in overall["content"]
    # 方向 2:改整体计划 → 今日视图跟着变
    edited = overall["content"].replace("今日新任务", "整体改回的任务")
    r2 = client.post("/api/plan/save", json={"content": edited, "note": "test"})
    assert r2.status_code == 200
    assert client.get(f"/api/plan/today?date={d2}").json()["tasks"] == ["整体改回的任务"]


def test_api_save_guards(client, std_plan):
    start, _ = std_plan
    d2 = (start + datetime.timedelta(days=1)).isoformat()
    assert client.post("/api/plan/today/save",
                       json={"date": d2, "tasks": ["x"], "base_mtime": 1}).status_code == 409
    assert client.post("/api/plan/today/save",
                       json={"date": "1999-01-01", "tasks": ["x"]}).status_code == 404
    assert client.get("/api/plan/today?date=not-a-date").status_code == 400
