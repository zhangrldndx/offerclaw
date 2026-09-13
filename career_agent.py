# -*- coding: utf-8 -*-
"""
OfferClaw · 顶层 Orchestrator (career_agent.py)

V2 阶段三职责：
    把"画像 / 投递池 / 日志 / 缺口"四类状态聚合成一句"今天最该做什么"，
    供 /api/today 直接消费、UI 顶部横条直接渲染。

设计取舍：
    - 纯规则、不调 LLM（秒级返回，前端打开页面不卡）
    - 优先级：紧急投递动作 > 面试准备 > 缺口闭环 > 日志补登 > 默认建议
    - LLM 由后续阶段（resume_builder / summary_tool）按需调

读取来源：
    - applications.md（投递池状态机）
    - daily_log.md（日志最新块）
    - user_profile.md（兜底建议带方向）
"""

import datetime
import os
import re
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPLICATIONS_PATH = os.path.join(BASE_DIR, "applications.md")
DAILY_LOG_PATH = os.path.join(BASE_DIR, "daily_log.md")
PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.md")


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_applications(md: str) -> List[Dict[str, str]]:
    """从 applications.md 抓所有 markdown 表格行（投递记录）。

    只识别"日期 | 公司 | 岗位 | 来源 | ... | 当前状态 | 下一步动作 | ..."这种 9+ 列表行。
    自动跳过 ``[DEMO]`` 行（示例数据），避免 ``today_advice`` 把 demo 投递顶到 headline。
    """
    rows: List[Dict[str, str]] = []
    headers: Optional[List[str]] = None
    in_table = False
    for line in md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            in_table = False
            headers = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        # 跳过分隔行 |---|---|
        if all(re.fullmatch(r":?-+:?", c or "-") for c in cells):
            in_table = True
            continue
        if headers is None:
            # 只有当包含"当前状态"或"状态"才认作真正的投递池表头
            if any("状态" in c for c in cells) and any("公司" in c or "岗位" in c for c in cells):
                headers = cells
            continue
        if not in_table or not headers or len(cells) != len(headers):
            continue
        # 跳过 [DEMO] 示例行
        if any("[DEMO]" in c for c in cells):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


# 状态机：每个状态对应一句"该做什么"
_STATE_ACTION = {
    "已评估": ("决定投/不投", 30),
    "准备投递": ("尽快定稿简历并投出", 90),
    "已投递": ("耐心等反馈（≥7 天无回复转跟进）", 40),
    "等待反馈": ("写一封跟进邮件或转入放弃", 80),
    "面试中": ("准备面试题 & 复盘故事库", 100),
    "已 Offer": ("评估接受 / 比较其他 Offer", 95),
    "已拒绝": ("做一次失败复盘并更新故事库", 50),
    "主动放弃": ("做一次决策复盘", 20),
    "不投递": ("无需动作", 5),
}


def pick_top_application(rows: List[Dict[str, str]]) -> Optional[Tuple[Dict[str, str], str, int]]:
    """挑出当前优先级最高的一条投递。返回 (row, action_text, priority)。"""
    best: Optional[Tuple[Dict[str, str], str, int]] = None
    for r in rows:
        state = ""
        for k in r:
            if "状态" in k:
                state = r[k]
                break
        action, prio = _STATE_ACTION.get(state, ("", 0))
        if not action:
            continue
        if best is None or prio > best[2]:
            best = (r, action, prio)
    return best


def latest_log_date(log: str) -> Optional[str]:
    """daily_log.md 最近一次 ## YYYY-MM-DD 标题日期。"""
    dates = re.findall(r"^##\s+(\d{4}-\d{2}-\d{2})", log, re.MULTILINE)
    return max(dates) if dates else None


def days_since(date_str: str) -> Optional[int]:
    try:
        d = datetime.date.fromisoformat(date_str)
    except Exception:
        return None
    return (datetime.date.today() - d).days


def _load_active_adjustments() -> List[str]:
    """读取 memory 层沉淀的次日调整规则（P2）。失败静默返回 []。"""
    try:
        from memory_layers import SemanticMemory, get_active_adjustments
        return get_active_adjustments(SemanticMemory())
    except Exception:
        return []


def _load_current_plan_focus(today_iso: str) -> dict:
    """读取最新计划并定位"本周重点"，供今日建议/推送引用。失败静默返回 {has_plan:False}。"""
    try:
        from plan_gen import summarize_plan_for_automation
        return summarize_plan_for_automation(today_iso)
    except Exception:
        return {"has_plan": False}


def _assess_plan_drift(today_iso: str, plan: dict) -> dict:
    """对照 daily_log 近 3 天的未完成记录，判断计划与实际进度的偏离程度。

    设计：计划本体**只在用户主动重排（生成/LLM 修改/编辑）时改变**——复盘不会
    悄悄改计划。本函数负责把"该不该重排"明确告诉用户：
      none：无偏离，不打扰；
      info：昨日有少量未完成 → 温和提示，可顺其自然也可重排；
      warn：连续多天/累计较多未完成（或复盘已沉淀调整规则）→ 明确建议重排。
    重排时「完成度感知」会跳过已完成、顺延未完成，所以建议是安全的。
    """
    out = {"level": "none", "message": "", "evidence": []}
    try:
        if not plan.get("has_plan") or plan.get("expired"):
            return out
        from summary_tool import extract_date_block, _parse_log_block
        today = datetime.date.fromisoformat(today_iso)
        log_md = _read(DAILY_LOG_PATH)
        days_with_miss, total_miss, evidence = 0, 0, []
        for i in range(1, 4):   # 回看最近 3 天（不含今天）
            d = (today - datetime.timedelta(days=i)).isoformat()
            block = extract_date_block(log_md, d)
            if not block:
                continue
            inc = (_parse_log_block(block, d).get("incomplete") or [])
            if inc:
                days_with_miss += 1
                total_miss += len(inc)
                evidence.append(f"{d} 未完成 {len(inc)} 项（如：{str(inc[0])[:24]}…）")
        if total_miss == 0:
            return out
        strong = days_with_miss >= 2 or total_miss >= 3 or bool(_load_active_adjustments())
        out["level"] = "warn" if strong else "info"
        out["evidence"] = evidence
        if strong:
            out["message"] = (
                f"近 {days_with_miss} 天累计 {total_miss} 项任务未完成，计划排期与实际进度已明显偏离。"
                "建议点「生成计划」按实际进度重排：已完成的不会重复排，未完成的会顺延进剩余天数。")
        else:
            out["message"] = (
                f"昨日有 {total_miss} 项任务未完成。计划不会自动改动——影响不大可顺其自然；"
                "若想把它们正式排回日程，点「生成计划」即可按实际进度重排。")
        return out
    except Exception:
        return out


# ── [L1] 外循环执行追踪 checkpoint ────────────────────────────────
# 动机：agent 每次唤醒都是 cold start，不知上次建议有没有被执行 → 跨天重复给同一条建议、无递进。
# 这里持久化「上次建议签名 + 连续重复次数」，达阈值就升级建议（不再原样重复），让外循环"记得到哪了"。
_ADVICE_KEY = "last_advice"
ADVICE_ESCALATE_AFTER = int(os.environ.get("OFFERCLAW_ADVICE_ESCALATE_AFTER", "2"))


def _advice_signature(headline: str) -> str:
    """把 headline 归一成跨天可比的签名（去日期/标点/空白），检测「又建议了同一件事」。"""
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "", headline or "")
    s = re.sub(r"[\s（）()\[\]【】·:：|]+", "", s)
    return s[:48]


def compute_execution_tracking(signature: str, headline: str, today_iso: str,
                               last_record: dict, progressed: bool):
    """纯函数（可单测）：据「上次建议记录 + 是否有推进」算执行追踪 + 必要时的升级动作。

    返回 (tracking, escalation_action|None, new_record)。
    - 同签名且未推进 → repeat_count 累加，达 ADVICE_ESCALATE_AFTER → escalated + 升级动作；
    - 签名变化或已推进 → repeat_count 归 0（视为有进展，不打扰）；
    - **同日幂等**（2026-08-08 修复）：同一天内重复唤醒（刷新页面 = 每次 GET /api/today
      都会走到这里并回写记录）不再累加——"已连续 N 天"只随日期推进计数，
      否则刷 15 次页面就虚涨成"连续 15 天"。
    """
    last = last_record or {}
    same = bool(signature) and signature == last.get("signature")
    prev_count = int(last.get("repeat_count", 0) or 0)
    if same and not progressed:
        repeat_count = prev_count if last.get("date") == today_iso else prev_count + 1
    else:
        repeat_count = 0
    escalated = repeat_count >= ADVICE_ESCALATE_AFTER
    escalation = None
    if escalated:
        escalation = (
            f"已连续 {repeat_count + 1} 天给出同一建议「{(headline or '')[:24]}」但日志无推进迹象——"
            "换个更小的切入口（拆成 30 分钟能做完的第一步），或在 ⑤ 卡片记一句卡在哪。")
    tracking = {
        "repeat_count": repeat_count,
        "escalated": escalated,
        "progressed": bool(progressed),
        "last_headline": last.get("headline", ""),
        "last_advice_date": last.get("date", ""),
    }
    new_record = {"signature": signature, "headline": headline,
                  "date": today_iso, "repeat_count": repeat_count}
    return tracking, escalation, new_record


def get_today_advice(sem=None, today: str = None) -> Dict[str, object]:
    """生成"今天最该做什么"。返回结构供 /api/today 直接 JSON 化。

    [L1] sem/today 可注入便于测试与按日模拟；execution_tracking 记录「同一建议连续给了几天、
    是否需要升级」，消除跨天 cold-start 的无意义重复建议。
    """
    apps_md = _read(APPLICATIONS_PATH)
    log_md = _read(DAILY_LOG_PATH)
    rows = parse_applications(apps_md)
    today = today or datetime.date.today().isoformat()

    headline = ""
    reason = ""
    source = ""
    next_actions: List[str] = []

    # 1) 投递池状态优先
    top = pick_top_application(rows)
    if top is not None:
        row, action, _prio = top
        company = next((row[k] for k in row if "公司" in k), "未知公司")
        post = next((row[k] for k in row if "岗位" in k), "未知岗位")
        state = next((row[k] for k in row if "状态" in k), "")
        nxt = next((row[k] for k in row if "下一步" in k), "")
        headline = f"【{company} · {post}】{action}"
        reason = f"applications.md 中此岗位状态为「{state}」，状态机指示该动作优先级最高。"
        source = "applications.md"
        if nxt:
            next_actions.append(f"按 applications.md 已写下一步：{nxt}")

    # 2) 日志补登
    last_log = latest_log_date(log_md)
    gap_days = days_since(last_log) if last_log else 99
    if gap_days is not None and gap_days >= 1:
        next_actions.append(
            f"补登 daily_log.md：上次记录是 {last_log}（距今 {gap_days} 天）。"
            f"可在 ⑤ 卡片直接追加。"
        )

    # 2.5) P2：读取复盘沉淀的"次日调整规则"，并入今日建议
    adjustments = _load_active_adjustments()
    for a in adjustments:
        next_actions.append(f"【复盘调整】{a}")

    # 2.6) 学习计划：所有自动化（今日建议 / 微信推送 / 提醒 / 再规划）都以
    #      "最新计划（含用户手动调整）的本周重点"为参考。置顶为主线动作。
    plan = _load_current_plan_focus(today)
    cw = plan.get("current_week") if plan.get("has_plan") else None
    if cw:
        edited = "（你已手动调整的计划）" if plan.get("edited_by_user") else ""
        line = f"【本周计划·第{cw['n']}周】{cw['theme']}"
        if cw.get("deliverable"):
            line += f" · 本周交付：{cw['deliverable']}"
        if plan.get("expired"):
            line = "【当前计划已到期，建议重新生成】" + line
        next_actions.insert(0, line + edited)

    # 2.7) 今日对照清单（today_plan）：OfferClaw 给出的"今天该做的"，
    #      供每日执行卡对照执行 + 留痕提交后自动判定未完成。
    #      优先用计划"日计划层"里今天这一天的具体任务（逐日不同）；
    #      日层没覆盖到今天（如已进入未展开的 Week 2+）才退回周粒度。
    today_plan: List[str] = []
    day_tasks = (plan.get("today_tasks") or []) if plan.get("has_plan") else []
    if day_tasks:
        today_plan.extend(f"【今日任务】{t}" for t in day_tasks[:4])
    elif cw:
        today_plan.append(f"推进本周主线（第{cw['n']}周）：{cw['theme']}")
        if cw.get("deliverable"):
            today_plan.append(f"向本周交付推进：{cw['deliverable']}")
    if headline and source == "applications.md":
        today_plan.append(headline)

    # 3) 兜底建议
    if not headline:
        if cw and not plan.get("expired"):
            # 没有紧急投递时，以本周计划重点作为今日主线
            headline = f"本周聚焦：{cw['theme']}"
            reason = f"按你当前学习计划（第 {cw['n']} 周 · {plan.get('period','')}）推进。"
            source = plan.get("plan_file", "学习计划")
        elif rows:
            headline = "暂无紧急投递任务，建议复盘最近匹配过的岗位"
            reason = "applications.md 中没有处于活跃状态的投递。"
            source = "applications.md"
        else:
            headline = "建议先在 ② 卡片粘一段 JD 跑一次匹配，把缺口固化进 ③ 卡片"
            reason = "applications.md 暂无投递记录，建议先做匹配评估。"
            source = "默认策略"
            next_actions.append("在 ② 卡片粘 JD → ③ 看缺口 → ④ 生成 4 周计划。")

    # 2.8) [L1] 外循环执行追踪：跨天检测「又建议了同一件事但没推进」，达阈值升级（不再原样重复）
    tracking: Dict[str, object] = {}
    try:
        if sem is None:
            from memory_layers import SemanticMemory
            sem = SemanticMemory()
        last_record = sem.get(_ADVICE_KEY) or {}
        signature = _advice_signature(headline)
        # progressed：自上次建议以来有没有推进（日志推进 / 建议本身换了）。无推进且同建议才算重复。
        progressed = (not last_record) or (signature != last_record.get("signature")) \
            or (bool(last_log) and str(last_log) > str(last_record.get("date", "")))
        tracking, escalation, new_record = compute_execution_tracking(
            signature, headline, today, last_record, progressed)
        if escalation:
            # 本周计划是用户已确认的主线，执行追踪只能紧随其后，不能把主线挤下首位。
            pos = 1 if next_actions and next_actions[0].startswith("【本周计划·") else 0
            next_actions.insert(pos, f"【执行追踪】{escalation}")
        sem.set(_ADVICE_KEY, new_record)
    except Exception:
        tracking = {}

    return {
        "today": today,
        "headline": headline,
        "reason": reason,
        "source": source,
        "next_actions": next_actions,
        "adjustments": adjustments,
        "plan": plan,
        "today_plan": today_plan,
        "plan_drift": _assess_plan_drift(today, plan),
        "execution_tracking": tracking,
        "stats": {
            "applications_total": len(rows),
            "last_log_date": last_log or "",
            "log_gap_days": gap_days if gap_days is not None else -1,
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_today_advice(), ensure_ascii=False, indent=2))
