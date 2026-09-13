# -*- coding: utf-8 -*-
"""profile_evolution.py — 画像演进：周度"画像更新建议" + 成长日志归档。

用户原则：
- 画像建议来自统一的 Profile Review Workflow，模型结论必须引用代码已核验的原始证据；
- 正式画像不会自动改写，用户接受或修改后接受才会生成新版本；
- 成长日志只归档同一次结构化审计的可读视图。
"""

from __future__ import annotations

import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_LOG_PATH = os.path.join(BASE_DIR, "daily_log.md")
JOURNAL_PATH = os.path.join(BASE_DIR, "growth_journal.md")

_JOURNAL_HEADER = """# 成长日志（Growth Journal）

> 每周复盘归档证据审计结果：记录能力演进、依据与用户决策。
> **本文件不向量化、不进任何模型上下文**——只给你自己回看用。
> 正式画像只能在画像面板确认建议后更新。

---
"""


METRICS_STATE_PATH = os.path.join(BASE_DIR, "logs", "growth_metrics.json")


def _week_log_stats(log_md: str, start: datetime.date, end: datetime.date) -> dict:
    """统计 [start, end] 区间的留痕：天数 / 完成数 / 未完成数 / 平均偏离度（来自记忆层）。"""
    import re
    days = done = todo = 0
    d = start
    try:
        from summary_tool import extract_date_block, _parse_log_block
        while d <= end:
            block = extract_date_block(log_md, d.isoformat())
            if block:
                days += 1
                parsed = _parse_log_block(block, d.isoformat())
                done += len(parsed.get("completed") or [])
                todo += len(parsed.get("incomplete") or [])
            d += datetime.timedelta(days=1)
    except Exception:
        pass
    # 偏离度：episodic 里该区间的 reflection 平均分
    dev = None
    try:
        from memory_layers import EpisodicMemory
        scores = [int(e.get("deviation_score", 0) or 0)
                  for e in EpisodicMemory().all()
                  if e.get("kind") == "reflection"
                  and e.get("reflection_kind", "daily") == "daily"
                  and start.isoformat() <= str(e.get("date", "")) <= end.isoformat()]
        if scores:
            dev = round(sum(scores) / len(scores), 1)
    except Exception:
        pass
    total = done + todo
    rate = round(done * 100 / total, 1) if total else None
    return {"days_logged": days, "done": done, "todo": todo,
            "completion_rate": rate, "avg_deviation": dev}


def _count_replans(start: datetime.date, end: datetime.date) -> dict:
    """按文件名时间戳统计区间内的重排/编辑次数（plan_YYYYMMDD_*.md）。"""
    import glob
    import re
    from plan_gen import _plans_dir
    regen = edits = 0
    for p in glob.glob(os.path.join(_plans_dir(), "plan_*.md")):
        m = re.match(r"plan_(\d{8})_\d{6}(_user)?\.md$", os.path.basename(p))
        if not m:
            continue
        try:
            d = datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if start <= d <= end:
            if m.group(2):
                edits += 1
            else:
                regen += 1
    return {"replans": regen, "manual_edits": edits}


def _load_metric_snapshots() -> list:
    import json
    try:
        with open(METRICS_STATE_PATH, encoding="utf-8") as f:
            return json.load(f).get("history", [])
    except Exception:
        return []


def _append_metric_snapshot(snap: dict) -> None:
    import json
    os.makedirs(os.path.dirname(METRICS_STATE_PATH), exist_ok=True)
    hist = _load_metric_snapshots()
    hist.append(snap)
    from io_utils import atomic_write_json
    atomic_write_json(METRICS_STATE_PATH, {"history": hist[-104:]})  # B1: 原子写防半截


def _trend(cur, prev, higher_is_better: bool = True) -> str:
    """趋势箭头：与上周比。无数据返回空。"""
    if cur is None or prev is None:
        return ""
    diff = cur - prev
    if abs(diff) < 1e-9:
        return "→"
    good = (diff > 0) == higher_is_better
    return ("↑" if diff > 0 else "↓") + ("✅" if good else "⚠️")


def compute_growth_metrics(today_iso: str = "") -> dict:
    """周度养成指标（确定性，不调 LLM）：本周 vs 上周 + 资产增量。

    让"agent 是否越来越契合"可测量：留痕率、完成率、偏离度是过程指标；
    重排/编辑次数是计划稳定性指标；知识库/缺口库增量是资产指标。
    """
    today = (datetime.date.fromisoformat(today_iso)
             if today_iso else datetime.date.today())
    this_start, this_end = today - datetime.timedelta(days=6), today
    prev_start, prev_end = today - datetime.timedelta(days=13), today - datetime.timedelta(days=7)

    log_md = ""
    try:
        with open(DAILY_LOG_PATH, encoding="utf-8") as f:
            log_md = f.read()
    except OSError:
        pass
    cur = _week_log_stats(log_md, this_start, this_end)
    prev = _week_log_stats(log_md, prev_start, prev_end)
    cur.update(_count_replans(this_start, this_end))

    # 资产现状 + 与上次快照的增量
    kb_chunks = None
    # Tests must not open the user's persistent Chroma database. A native
    # storage failure on Windows can terminate Python before it can be caught.
    if not os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get(
            "OFFERCLAW_SKIP_KB_METRICS", "0") != "1":
        try:
            import chromadb
            from rag_tools import get_collection_name
            col = chromadb.PersistentClient(
                path=os.path.join(BASE_DIR, "chroma_db")).get_collection(get_collection_name())
            kb_chunks = col.count()
        except Exception:
            pass
    gap_targets = merged_gaps = None
    try:
        # 统一以投递管理中已确认、已关联 JD 且纳入计划的记录为准。
        # 旧 gap_store 仅保留为只读归档，不能继续影响当前指标口径。
        from application_jd_store import plan_gap_items, plan_targets
        target_view = plan_targets()
        gap_targets = target_view.get("total", 0)
        merged_gaps = len(plan_gap_items(target_view.get("included", [])))
    except Exception:
        pass
    adjustments = 0
    try:
        from memory_layers import SemanticMemory, get_active_adjustments
        adjustments = len(get_active_adjustments(SemanticMemory()))
    except Exception:
        pass

    snaps = _load_metric_snapshots()
    last = snaps[-1] if snaps else {}
    deltas = {}
    for key, val in (("kb_chunks", kb_chunks), ("gap_targets", gap_targets),
                     ("merged_gaps", merged_gaps)):
        if val is not None and last.get(key) is not None:
            deltas[key] = val - last[key]
    _append_metric_snapshot({
        "date": today.isoformat(), "kb_chunks": kb_chunks,
        "gap_targets": gap_targets, "merged_gaps": merged_gaps,
        "completion_rate": cur["completion_rate"], "days_logged": cur["days_logged"],
    })
    return {
        "week": f"{this_start.isoformat()} ~ {this_end.isoformat()}",
        "current": cur, "previous": prev,
        "kb_chunks": kb_chunks, "gap_targets": gap_targets,
        "merged_gaps": merged_gaps, "active_adjustments": adjustments,
        "deltas": deltas,
    }


def format_metrics_md(m: dict) -> str:
    """把指标格式化为成长日志里的小节（带与上周对比的趋势箭头）。"""
    cur, prev = m["current"], m["previous"]

    def _fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "—"

    lines = [f"### 📈 本周养成指标（{m['week']}）", ""]
    lines.append(
        f"- 留痕：{cur['days_logged']}/7 天 "
        f"{_trend(cur['days_logged'], prev['days_logged'])}（上周 {prev['days_logged']}/7）")
    lines.append(
        f"- 计划完成率：{_fmt(cur['completion_rate'], '%')} "
        f"{_trend(cur['completion_rate'], prev['completion_rate'])}"
        f"（完成 {cur['done']} / 未完成 {cur['todo']}；上周 {_fmt(prev['completion_rate'], '%')}）")
    lines.append(
        f"- 平均偏离度：{_fmt(cur['avg_deviation'])} "
        f"{_trend(cur['avg_deviation'], prev['avg_deviation'], higher_is_better=False)}"
        f"（上周 {_fmt(prev['avg_deviation'])}；越低越好）")
    lines.append(
        f"- 计划稳定性：重排 {cur['replans']} 次 · 手动编辑 {cur['manual_edits']} 次"
        "（频繁重排可能说明排期不贴合实际）")
    d = m.get("deltas", {})

    def _delta(key):
        return f"（较上次 {'+' if d[key] >= 0 else ''}{d[key]}）" if key in d else ""

    lines.append(
        f"- 资产：知识库 {_fmt(m['kb_chunks'])} 块{_delta('kb_chunks')} · "
        f"目标 JD {_fmt(m['gap_targets'])} 个{_delta('gap_targets')} · "
        f"缺口 {_fmt(m['merged_gaps'])} 条{_delta('merged_gaps')} · "
        f"生效调整规则 {m['active_adjustments']} 条")
    if cur["days_logged"] == 0:
        lines.append("- ⚠️ 本周零留痕：所有养成机制都在空转——指标的前提是留痕。")
    return "\n".join(lines)


def append_growth_journal(suggestions_md: str, date_str: str = "") -> str:
    """把本周建议归档进成长日志（append-only），返回文件路径。"""
    date_str = date_str or datetime.date.today().isoformat()
    week = datetime.date.fromisoformat(date_str).isocalendar()
    entry = (
        f"\n## {date_str}（{week.year}-W{week.week:02d}）\n\n"
        f"{suggestions_md.strip()}\n\n"
        "**采纳情况**：（待你标注：已采纳/部分采纳/未采纳 + 原因）\n\n---\n"
    )
    is_new = not os.path.exists(JOURNAL_PATH)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        if is_new:
            f.write(_JOURNAL_HEADER)
        f.write(entry)
    return JOURNAL_PATH


def suggest_profile_updates() -> dict:
    """Generate metrics and one evidence-grounded structured profile audit."""
    metrics = compute_growth_metrics()
    metrics_md = format_metrics_md(metrics)
    structured_audit = {"status": "review_unavailable", "created": [], "pending": 0}
    suggestions = ""
    try:
        from profile_review import ProfileRepository, format_audit_markdown
        repository = ProfileRepository()
        structured_audit = repository.run_audit(max_items=5, trigger_kind="weekly")
        suggestions = format_audit_markdown(structured_audit, repository)
    except Exception as e:
        structured_audit = {"status": "failed", "error": str(e), "created": [], "pending": 0}
        suggestions = "## 画像证据审计\n- 审计失败，正式画像未变化：" + str(e)[:500]

    entry = metrics_md + "\n\n---\n\n" + suggestions
    path = append_growth_journal(entry)
    has_updates = any(item.get("status") == "pending"
                      for item in structured_audit.get("created") or [])

    cur = metrics["current"]
    metric_line = (
        f"📈 本周养成：留痕 {cur['days_logged']}/7 天 · "
        f"完成率 {cur['completion_rate'] if cur['completion_rate'] is not None else '—'}"
        f"{'%' if cur['completion_rate'] is not None else ''} · "
        f"重排 {cur['replans']} 次")
    return {
        "status": "ok",
        "metrics": metrics,
        "suggestions": suggestions,
        "has_updates": has_updates,
        "structured_audit": structured_audit,
        "journal_path": os.path.relpath(path, BASE_DIR),
        "wechat_summary": metric_line + "\n" + (
            ("🌱 本周画像更新建议已进入待确认列表：\n" + suggestions[:800] +
             "\n\n请在画像面板查看原始证据并决定接受、修改或拒绝。")
            if has_updates else "🌱 本周没有通过证据门槛的画像修改建议。"
        ) + f"\n📒 已归档成长日志：{os.path.relpath(path, BASE_DIR)}",
    }
