# -*- coding: utf-8 -*-
"""plan_daily.py — 学习计划的「今日切片」视图与外科手术式回写。

动机:计划 md 里已有 LLM 逐日分配的日计划层(plan_prompt 第 5.2 步,D1…Dn
连续覆盖全周期,日期由 normalize_plan_dates 确定性重写)。本模块把"今天"
这一天切出来单独展示/编辑,而**不引入第二份存储**:

- 单一事实源 = plans/ 下最新计划 md(与整体计划共用同一文件);
- 用户改整体计划 → 今日视图重读同一文件,自动一致;
- 用户改今日任务 → 只重写当日 D 块内的编号任务行(其余行原样保留),
  经 plan_gen.save_plan 另存 _user 版,整体视图自动一致。
双向同步不需要任何"同步逻辑",一致性由构造保证(不存在可漂移的副本)。

日块格式(真实计划样本):
    D5（06-09 周二）
      主线标签：[补技能]
      核心任务：
        1. 任务一（预计 3h，依据 P1-1）
        2. 任务二「来源：xxx.md」
      可选任务：
        - 可选项
「今日任务」= 编号行(核心任务);可选任务等非编号行只读展示、回写时不动。
"""
from __future__ import annotations

import datetime
import hashlib
import re
import uuid


class NoPlanError(Exception):
    """没有任何计划文件。"""


class NoDailyLayerError(Exception):
    """计划存在但没有可解析的日计划层(如被手动删除/退化产物)。"""


class DayNotFoundError(Exception):
    """指定日期不在当前计划的日计划层里。"""


class StalePlanError(Exception):
    """保存时计划文件已被更新(base_mtime 不匹配),需要刷新后重试。"""


# D 形式:D5（06-09 周二）/ ### D5（06-09 周二）/ **D5（2026-06-09 周二）**
# (不同生成模型的 Markdown 装饰不同:qwen 裸头,gpt-5.6 实测输出 ### 标题——都要认)
_D_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:\*\*)?\s*D(\d+)\s*[（(]\s*(?:(\d{4})-)?(\d{1,2})-(\d{1,2})[^）)]*[）)]")
# 裸日期形式(旧格式兜底):2026-06-09（周二） / 06-09 周二 —— 要求行短,防误伤正文
_BARE_HEADER_RE = re.compile(
    r"^\s*#{0,4}\s*(?:(\d{4})-)?(\d{1,2})-(\d{1,2})\s*[ （(]*周[一二三四五六日天]")
_TASK_RE = re.compile(r"^(\s*)(\d+)[\.、]\s*(\S.*)$")
_TASK_ID_RE = re.compile(r"\s*<!--\s*task_id:\s*([A-Za-z0-9_-]+)\s*-->\s*$")
# 小节标签行:≤12 字的短前缀紧跟冒号(可带加粗),如 **今日主线标签**：[补项目] / 预计投入：8h
_SECTION_LABEL_RE = re.compile(r"^\*{0,2}[^*：:]{1,12}\*{0,2}\s*[:：]")
_BLOCK_END_RE = re.compile(r"^\s*(?:#{1,4}\s|Week\s*\d|---\s*$|—)")
_PERIOD_RE = re.compile(
    r"计划周期[:：]\s*(\d{4}-\d{2}-\d{2})\s*[→\-~>]+\s*(\d{4}-\d{2}-\d{2})")
_WEEK_THEME_RE = re.compile(r"Week\s*(\d+)[^\n主]*主题[:：]\s*(.+)")


def _resolve_year(month: int, day: int, period_start: datetime.date | None,
                  fallback_year: int) -> datetime.date | None:
    """MM-DD → 完整日期。年份取计划起点年;跨年计划(12→1 月)按"不早于起点"顺延一年。"""
    year = period_start.year if period_start else fallback_year
    try:
        d = datetime.date(year, month, day)
    except ValueError:
        return None
    if period_start and d < period_start:
        try:
            d2 = datetime.date(year + 1, month, day)
        except ValueError:
            return None
        if (d2 - period_start).days <= 400:
            return d2
    return d


def parse_plan_days(content: str) -> dict:
    """解析计划 md 的日计划层。

    返回 {period_start, period_end, weeks: {n: theme}, days: [
      {index, date, label, header_line, block_end, task_lines: [(line_no, indent, text)],
       optional_lines: [str], week_n}
    ]}(days 按日期升序;line 均为 0-based 行号,block_end 为开区间)。
    """
    lines = content.splitlines()
    period_start = period_end = None
    mp = _PERIOD_RE.search(content)
    if mp:
        try:
            period_start = datetime.date.fromisoformat(mp.group(1))
            period_end = datetime.date.fromisoformat(mp.group(2))
        except ValueError:
            pass
    weeks = {int(n): t.strip().strip("：: ") for n, t in _WEEK_THEME_RE.findall(content)}
    today_year = datetime.date.today().year

    def _clean_label(raw: str) -> str:
        """展示用标签:剥掉 Markdown 装饰(### / **),保留「D5（08-08 周六）」本体。"""
        return raw.strip().lstrip("#").strip().strip("*").strip()

    # 先找全部日块头
    headers: list[tuple[int, int | None, datetime.date | None, str]] = []
    for i, ln in enumerate(lines):
        m = _D_HEADER_RE.match(ln)
        if m:
            idx = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else None
            d = (datetime.date(year, int(m.group(3)), int(m.group(4)))
                 if year else _resolve_year(int(m.group(3)), int(m.group(4)),
                                            period_start, today_year))
            headers.append((i, idx, d, _clean_label(ln)))
            continue
        if len(ln.strip()) <= 30:
            m2 = _BARE_HEADER_RE.match(ln)
            if m2:
                year = int(m2.group(1)) if m2.group(1) else None
                d = (datetime.date(year, int(m2.group(2)), int(m2.group(3)))
                     if year else _resolve_year(int(m2.group(2)), int(m2.group(3)),
                                                period_start, today_year))
                headers.append((i, None, d, _clean_label(ln)))

    days = []
    for k, (line_no, idx, d, label) in enumerate(headers):
        # 日期兜底:标签解析失败但有 D 序号 + 周期起点 → 起点 + (序号-1)
        if d is None and idx is not None and period_start is not None:
            d = period_start + datetime.timedelta(days=idx - 1)
        if d is None:
            continue
        next_header = headers[k + 1][0] if k + 1 < len(headers) else len(lines)
        block_end = next_header
        for j in range(line_no + 1, next_header):
            if _BLOCK_END_RE.match(lines[j]):
                block_end = j
                break
        # 「今日任务」= 编号行;可选任务 = 「可选任务」小节内的普通 bullet。
        # gpt 风格把小节标签也写成 bullet(- **核心任务**：/- **预计投入**：8h),
        # 须按状态机区分标签与内容,否则标签会被误当"可选任务"展示。
        task_lines, optional_lines, task_items, optional_task_lines = [], [], [], []
        in_optional = False
        for j in range(line_no + 1, block_end):
            mt = _TASK_RE.match(lines[j])
            if mt:
                raw_text = mt.group(3).strip()
                mid = _TASK_ID_RE.search(raw_text)
                task_id = mid.group(1) if mid else ""
                text = _TASK_ID_RE.sub("", raw_text).strip()
                task_lines.append((j, mt.group(1), text))
                task_items.append({"task_id": task_id, "text": text, "optional": False,
                                   "line_no": j, "indent": mt.group(1)})
                in_optional = False
                continue
            s = lines[j].strip()
            if "可选任务" in s:
                in_optional = True
                continue
            if not s.startswith(("-", "*")):
                continue
            text = s.lstrip("-* ").strip()
            mid = _TASK_ID_RE.search(text)
            task_id = mid.group(1) if mid else ""
            text = _TASK_ID_RE.sub("", text).strip()
            if _SECTION_LABEL_RE.match(text):   # 「**主线标签**：…」类小节标签行
                in_optional = False
                continue
            if in_optional and len(text) > 2:
                optional_lines.append(text)
                optional_task_lines.append(j)
                task_items.append({"task_id": task_id, "text": text, "optional": True,
                                   "line_no": j, "indent": re.match(r"^\s*", lines[j]).group(0)})
        index = idx if idx is not None else k + 1
        week_n = ((d - period_start).days // 7 + 1) if period_start else (index - 1) // 7 + 1
        days.append({
            "index": index, "date": d, "label": label,
            "header_line": line_no, "block_end": block_end,
            "task_lines": task_lines, "optional_lines": optional_lines,
            "task_items": task_items, "optional_task_lines": optional_task_lines,
            "week_n": week_n,
        })
    days.sort(key=lambda x: x["date"])
    return {"period_start": period_start, "period_end": period_end,
            "weeks": weeks, "days": days}


def _pick_day(days: list[dict], target: datetime.date) -> tuple[dict, str]:
    """挑选展示日:范围内精确命中;开始前显示首日;结束后显示末日;缺日取最近的过去一天。"""
    if target < days[0]["date"]:
        return days[0], "before_start"
    if target > days[-1]["date"]:
        return days[-1], "after_end"
    for d in days:
        if d["date"] == target:
            return d, "in_range"
    prev = max((d for d in days if d["date"] < target), key=lambda x: x["date"])
    return prev, "in_range_nearest"


def _build_view(latest: dict, parsed: dict, day: dict, status: str,
                requested: datetime.date) -> dict:
    hints = {
        "in_range": "仅供参考，可直接修改；改动会同步写回整体计划。",
        "in_range_nearest": "今天在计划里没有单独日块，先显示最近一天的安排。",
        "before_start": f"计划 {parsed['days'][0]['date'].isoformat()} 才开始，先显示首日安排。",
        "after_end": (f"计划已于 {parsed['days'][-1]['date'].isoformat()} 结束，"
                      "显示最后一天安排；可重新生成新周期计划。"),
    }
    return {
        "has_plan": True, "has_daily": True, "status": status,
        "period_start": parsed["period_start"].isoformat() if parsed["period_start"] else "",
        "period_end": parsed["period_end"].isoformat() if parsed["period_end"] else "",
        "date": day["date"].isoformat(), "requested_date": requested.isoformat(),
        "label": day["label"], "day_index": day["index"],
        "total_days": len(parsed["days"]), "week_n": day["week_n"],
        "week_theme": parsed["weeks"].get(day["week_n"], ""),
        "tasks": [t for _ln, _ind, t in day["task_lines"]],
        "optional_tasks": day["optional_lines"],
        "task_items": [{"task_id": t.get("task_id", ""), "text": t["text"],
                        "optional": t["optional"]} for t in day.get("task_items", [])],
        "hint": hints[status],
        "plan_file": latest["filename"], "plan_mtime": latest["mtime"],
        "edited_by_user": latest["edited_by_user"],
    }


def get_today_view(date_iso: str | None = None) -> dict:
    """读最新计划,返回"今日计划"视图 dict(供 /api/plan/today 直接 JSON 化)。

    无计划 / 无日计划层不抛异常而是返回 has_plan/has_daily=False——
    这两种是页面常态而非错误;日期非法才抛 ValueError。
    """
    from plan_gen import load_latest_plan
    target = (datetime.date.fromisoformat(date_iso) if date_iso
              else datetime.date.today())
    latest = load_latest_plan()
    if not latest:
        return {"has_plan": False, "has_daily": False, "status": "no_plan",
                "requested_date": target.isoformat(),
                "hint": "还没有计划——先在计划卡生成一份。"}
    parsed = parse_plan_days(latest["content"])
    if not parsed["days"]:
        return {"has_plan": True, "has_daily": False, "status": "no_daily",
                "requested_date": target.isoformat(),
                "period_start": parsed["period_start"].isoformat() if parsed["period_start"] else "",
                "period_end": parsed["period_end"].isoformat() if parsed["period_end"] else "",
                "plan_file": latest["filename"], "plan_mtime": latest["mtime"],
                "edited_by_user": latest["edited_by_user"],
                "hint": "当前计划没有逐日拆分（可能是旧格式或被手动删除）；重新生成计划即可恢复。"}
    day, status = _pick_day(parsed["days"], target)
    return _build_view(latest, parsed, day, status, target)


def replace_day_tasks(content: str, date: datetime.date, tasks: list[str]) -> str:
    """把指定日期日块内的编号任务行替换为 ``tasks``,其余内容逐行保留(无损)。

    策略:删除该块内全部旧编号行,在原首条编号行的位置插入新编号列表
    (沿用原缩进;块内无编号行时插到「核心任务：」行后,再退而插到块头后)。
    """
    parsed = parse_plan_days(content)
    day = next((d for d in parsed["days"] if d["date"] == date), None)
    if day is None:
        raise DayNotFoundError(f"日期 {date.isoformat()} 不在当前计划的日计划层里")
    lines = content.splitlines()
    old = day["task_lines"]
    indent = old[0][1] if old else "    "
    new_numbered = [f"{indent}{i}. {t}" for i, t in enumerate(tasks, 1)]

    if old:
        insert_at = old[0][0]
        drop = {ln for ln, _i, _t in old}
    else:
        insert_at = None
        for j in range(day["header_line"] + 1, day["block_end"]):
            if "核心任务" in lines[j]:
                insert_at = j + 1
                break
        if insert_at is None:
            insert_at = day["header_line"] + 1
        drop = set()

    out = []
    for j, ln in enumerate(lines):
        if j == insert_at:
            out.extend(new_numbered)
            if j not in drop:
                out.append(ln)
            continue
        if j in drop:
            continue
        out.append(ln)
    if insert_at == len(lines):    # 块头在文件末尾的极端情况
        out.extend(new_numbered)
    tail = "\n" if content.endswith("\n") else ""
    return "\n".join(out) + tail


def ensure_task_ids(content: str) -> str:
    """给日计划核心/可选任务补隐藏稳定 ID；已有 ID 原样保留。"""
    parsed = parse_plan_days(content)
    if not parsed["days"]:
        return content
    lines = content.splitlines()
    used: set[str] = set()
    for day in parsed["days"]:
        occurrence: dict[str, int] = {}
        for item in day.get("task_items", []):
            task_id = item.get("task_id", "")
            if task_id:
                used.add(task_id)
                continue
            text = item["text"]
            occurrence[text] = occurrence.get(text, 0) + 1
            seed = f"{day['date'].isoformat()}|{text}|{occurrence[text]}"
            task_id = "pt_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:14]
            while task_id in used:
                task_id = "pt_" + uuid.uuid4().hex[:14]
            used.add(task_id)
            lines[item["line_no"]] = lines[item["line_no"]].rstrip() + f" <!-- task_id: {task_id} -->"
    tail = "\n" if content.endswith("\n") else ""
    return "\n".join(lines) + tail


def _task_text(text: str, *, estimated_hours=None, priority: str = "",
               deliverable: str = "") -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("任务内容不能为空")
    attrs = []
    if estimated_hours not in (None, ""):
        hours = float(estimated_hours)
        if hours <= 0 or hours > 24:
            raise ValueError("预计投入必须在 0~24 小时之间")
        value = f"预计 {hours:g}h"
        if re.search(r"预计\s*\d+(?:\.\d+)?\s*h", text, re.I):
            text = re.sub(r"预计\s*\d+(?:\.\d+)?\s*h", value, text, count=1, flags=re.I)
        else:
            attrs.append(value)
    if priority:
        if priority not in {"高", "中", "低"}:
            raise ValueError("priority 必须是 高 / 中 / 低")
        value = f"优先级：{priority}"
        if re.search(r"优先级\s*[：:]\s*[高中低]", text):
            text = re.sub(r"优先级\s*[：:]\s*[高中低]", value, text, count=1)
        else:
            attrs.append(value)
    if deliverable:
        value = "验收：" + deliverable.strip()[:200]
        if re.search(r"验收(?:产物)?\s*[：:]\s*[^；）\n]+", text):
            text = re.sub(r"验收(?:产物)?\s*[：:]\s*[^；）\n]+", value, text, count=1)
        else:
            attrs.append(value)
    return text[:500] + (f"（{'；'.join(attrs)}）" if attrs else "")


def _rewrite_day(content: str, date: datetime.date, core: list[dict], optional: list[dict]) -> str:
    core_text = [f"{x['text']} <!-- task_id: {x['task_id']} -->" for x in core]
    content = replace_day_tasks(content, date, core_text)
    parsed = parse_plan_days(content)
    day = next((d for d in parsed["days"] if d["date"] == date), None)
    if not day:
        raise DayNotFoundError(date.isoformat())
    lines = content.splitlines()
    drop = set(day.get("optional_task_lines", []))
    insert_at = None
    for j in range(day["header_line"] + 1, day["block_end"]):
        if "可选任务" in lines[j]:
            insert_at = j + 1
            break
    optional_lines = [f"    - {x['text']} <!-- task_id: {x['task_id']} -->" for x in optional]
    if insert_at is None and optional_lines:
        insert_at = day["block_end"]
        optional_lines = ["    可选任务：", *optional_lines]
    out = []
    for j, line in enumerate(lines):
        if insert_at is not None and j == insert_at:
            out.extend(optional_lines)
        if j not in drop:
            out.append(line)
    if insert_at == len(lines):
        out.extend(optional_lines)
    tail = "\n" if content.endswith("\n") else ""
    return "\n".join(out) + tail


def patch_plan_tasks(operations: list[dict], base_mtime: int | None = None) -> dict:
    """按稳定 task_id 外科式调整计划，完全不调用 LLM。"""
    from plan_gen import load_latest_plan, save_plan

    latest = load_latest_plan()
    if not latest:
        raise NoPlanError("还没有任何计划")
    if base_mtime is not None and int(base_mtime) != int(latest["mtime"]):
        raise StalePlanError("计划已被更新，请刷新后再修改")
    content = ensure_task_ids(latest["content"])
    changed_ids: list[str] = []
    for operation in operations:
        op = str(operation.get("op", "")).strip()
        parsed = parse_plan_days(content)
        days_by_date = {d["date"].isoformat(): d for d in parsed["days"]}
        task_id = str(operation.get("task_id", "")).strip()
        source_day = next((d for d in parsed["days"]
                           if any(i.get("task_id") == task_id for i in d.get("task_items", []))), None)

        if op == "add":
            target = str(operation.get("date", ""))
            day = days_by_date.get(target)
            if not day:
                raise DayNotFoundError(f"日期 {target} 不在计划中")
            task_id = task_id or "pt_" + uuid.uuid4().hex[:14]
            item = {"task_id": task_id, "text": _task_text(
                operation.get("text", ""), estimated_hours=operation.get("estimated_hours"),
                priority=str(operation.get("priority", "")),
                deliverable=str(operation.get("deliverable", ""))),
                "optional": bool(operation.get("optional", False))}
            core = [dict(x) for x in day["task_items"] if not x["optional"]]
            optional = [dict(x) for x in day["task_items"] if x["optional"]]
            (optional if item["optional"] else core).append(item)
            content = _rewrite_day(content, day["date"], core, optional)
        elif op in {"edit", "delete", "move", "set_optional"}:
            if not source_day:
                raise KeyError(f"找不到 task_id：{task_id}")
            item = next(x for x in source_day["task_items"] if x.get("task_id") == task_id)
            core = [dict(x) for x in source_day["task_items"] if not x["optional"] and x.get("task_id") != task_id]
            optional = [dict(x) for x in source_day["task_items"] if x["optional"] and x.get("task_id") != task_id]
            if op != "delete":
                updated = {"task_id": task_id,
                           "text": _task_text(operation.get("text", item["text"]),
                                              estimated_hours=operation.get("estimated_hours"),
                                              priority=str(operation.get("priority", "")),
                                              deliverable=str(operation.get("deliverable", ""))),
                           "optional": bool(operation.get("optional", item["optional"]))}
                if op == "set_optional":
                    updated["optional"] = bool(operation.get("optional", True))
                if op == "move":
                    target = str(operation.get("date", ""))
                    target_day = days_by_date.get(target)
                    if not target_day:
                        raise DayNotFoundError(f"日期 {target} 不在计划中")
                    content = _rewrite_day(content, source_day["date"], core, optional)
                    reparsed = parse_plan_days(content)
                    target_day = next(d for d in reparsed["days"] if d["date"].isoformat() == target)
                    tcore = [dict(x) for x in target_day["task_items"] if not x["optional"]]
                    toptional = [dict(x) for x in target_day["task_items"] if x["optional"]]
                    (toptional if updated["optional"] else tcore).append(updated)
                    content = _rewrite_day(content, target_day["date"], tcore, toptional)
                    changed_ids.append(task_id)
                    continue
                (optional if updated["optional"] else core).append(updated)
            content = _rewrite_day(content, source_day["date"], core, optional)
        else:
            raise ValueError(f"不支持的操作：{op}")
        changed_ids.append(task_id)

    path = save_plan(content, edited_by_user=True)
    return {"status": "ok", "saved_path": path, "changed_task_ids": changed_ids,
            "plan": load_latest_plan()}


def save_today(date_iso: str, tasks: list[str], base_mtime: int | None = None,
               max_tasks: int = 20, max_len: int = 300) -> dict:
    """回写"今日任务"到整体计划:校验 → 替换当日编号行 → 另存 _user 版 → 返回新视图。

    base_mtime(可选)= 前端读到的计划 mtime;不匹配说明计划在读写之间被
    更新过(如刚重新生成),抛 StalePlanError 让前端刷新,避免盲写覆盖。
    """
    from plan_gen import load_latest_plan, save_plan
    date = datetime.date.fromisoformat(date_iso)
    latest = load_latest_plan()
    if not latest:
        raise NoPlanError("还没有任何计划,无法保存今日任务")
    if base_mtime is not None and int(base_mtime) != int(latest["mtime"]):
        raise StalePlanError("计划已被更新（可能刚重新生成/编辑过），请刷新后再改")
    clean = [t.strip() for t in tasks if t and t.strip()]
    if len(clean) > max_tasks:
        raise ValueError(f"今日任务最多 {max_tasks} 条")
    clean = [t[:max_len] for t in clean]
    if not parse_plan_days(latest["content"])["days"]:
        raise NoDailyLayerError("当前计划没有日计划层,无法按天回写")
    new_content = replace_day_tasks(latest["content"], date, clean)
    save_plan(new_content, edited_by_user=True)
    return get_today_view(date_iso)
