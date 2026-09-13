# -*- coding: utf-8 -*-
"""applications_store.py — 投递管理的写入层（Web 投递功能栏的后端）。

用户流程（用户定义）：
- 用户上传自己的真实投递情况：企业名、投递岗、投递进度+时间点、经验总结；
- 经验总结（笔试/面试真题、流程、教训）可选择**加入知识库**——亲历经验是
  第一手强指导信号，入库后直接影响 RAG 问答、学习计划与每日建议。

存储：
- 投递行：写入 applications.md 的「投递清单」表（沿用既有状态机与字段，
  career_agent.parse_applications / pick_top_application 直接消费）；
- 经验总结：写成 knowledge_base/experience_posts/亲历_*.md（frontmatter
  含 company/position/stage），由 API 层决定是否增量入向量库。
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPLICATIONS_PATH = os.path.join(BASE_DIR, "applications.md")
EXPERIENCE_DIR = os.path.join(BASE_DIR, "knowledge_base", "experience_posts")

# 与 applications.md「状态枚举」表一致
STATUSES = ["已评估", "准备投递", "不投递", "已投递", "等待反馈",
            "面试中", "已 Offer", "已拒绝", "主动放弃"]

# 投递清单表的列。旧 10 列表会由 ``ensure_application_schema`` 原地、幂等升级。
_COLUMNS = ["投递ID", "日期", "公司", "岗位", "来源", "JD来源链接", "地点",
            "匹配结论", "样本定位", "JD标识", "JD版本ID", "匹配ID", "纳入计划", "长期关注",
            "计划优先级", "当前状态", "下一步动作", "备注"]
_LEGACY_COLUMNS = ["日期", "公司", "岗位", "来源", "地点", "匹配结论",
                   "样本定位", "当前状态", "下一步动作", "备注"]


def _today() -> str:
    return datetime.date.today().isoformat()


def _clean_cell(s: str) -> str:
    """单元格清洗：去掉竖线/换行，防止破坏 markdown 表格。"""
    return re.sub(r"[|\r\n]+", " ", str(s or "")).strip() or "—"


def _legacy_application_id(company: str, position: str, date: str, row_index: int) -> str:
    seed = f"{company}|{position}|{date}|{row_index}".encode("utf-8")
    return "app_legacy_" + hashlib.sha256(seed).hexdigest()[:10]


def _table_bounds(lines: list[str]) -> tuple[int, int, list[str]] | None:
    for i, line in enumerate(lines):
        if not line.strip().startswith("|") or "当前状态" not in line or "公司" not in line:
            continue
        headers = [c.strip() for c in line.strip().strip("|").split("|")]
        body_start = i + 2
        body_end = body_start
        while body_end < len(lines) and lines[body_end].strip().startswith("|"):
            body_end += 1
        return i, body_end, headers
    return None


def ensure_application_schema(path: str | None = None, *, dry_run: bool = False) -> dict:
    """Upgrade the application table to stable IDs and JD/plan relation columns.

    Existing values are copied by header name.  No JD is guessed and every migrated
    row starts with ``纳入计划=否``.
    """
    path = path or APPLICATIONS_PATH
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as exc:
        return {"status": "error", "error": str(exc)}
    bounds = _table_bounds(lines)
    if not bounds:
        return {"status": "error", "error": "applications.md 中找不到投递清单表"}
    header_idx, body_end, headers = bounds
    if headers == _COLUMNS:
        return {"status": "ok", "action": "unchanged", "rows": body_end - header_idx - 2}

    converted: list[str] = []
    for row_index, line in enumerate(lines[header_idx + 2:body_end]):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        old = dict(zip(headers, cells))
        company, position, date = old.get("公司", ""), old.get("岗位", ""), old.get("日期", "")
        values = {key: old.get(key, "—") for key in _COLUMNS}
        values["投递ID"] = old.get("投递ID") or _legacy_application_id(
            company, position, date, row_index)
        values["JD标识"] = old.get("JD标识") or "—"
        values["JD版本ID"] = old.get("JD版本ID") or "—"
        values["匹配ID"] = old.get("匹配ID") or "—"
        values["JD来源链接"] = old.get("JD来源链接") or "—"
        values["纳入计划"] = old.get("纳入计划") or "否"
        values["长期关注"] = old.get("长期关注") or "否"
        values["计划优先级"] = old.get("计划优先级") or "medium"
        converted.append("| " + " | ".join(_clean_cell(values[k]) for k in _COLUMNS) + " |")

    new_lines = list(lines)
    new_lines[header_idx] = "| " + " | ".join(_COLUMNS) + " |"
    new_lines[header_idx + 1] = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    new_lines[header_idx + 2:body_end] = converted
    if not dry_run:
        from io_utils import atomic_write_text
        atomic_write_text(path, "\n".join(new_lines) + "\n")
    return {"status": "ok", "action": "would_migrate" if dry_run else "migrated",
            "rows": len(converted)}


def list_applications(include_demo: bool = False) -> list[dict]:
    """读取投递清单（默认过滤 [DEMO] 示例行）。"""
    from career_agent import parse_applications
    if not os.path.exists(APPLICATIONS_PATH):
        return []
    with open(APPLICATIONS_PATH, encoding="utf-8") as f:
        md = f.read()
    rows = parse_applications(md)  # parse 已跳过 [DEMO]
    if include_demo:
        return rows
    return rows


def _frontmatter_value(text: str, key: str) -> str:
    """读取本模块生成的简单 YAML frontmatter 字符串字段。"""
    m = re.search(rf'^{re.escape(key)}:\s*"?(.*?)"?\s*$', text, re.MULTILINE)
    return (m.group(1).strip().strip('"') if m else "")


def list_experiences() -> list[dict]:
    """读取已保存的亲历投递经验，供 UI 与实时 RAG 状态源消费。

    兼容早期未写 company/position/stage 独立 frontmatter 字段的文件，避免用户
    已保存的经验在升级后“消失”。
    """
    if not os.path.isdir(EXPERIENCE_DIR):
        return []
    records = []
    for name in os.listdir(EXPERIENCE_DIR):
        if not name.endswith(".md") or name.startswith("_"):
            continue
        path = os.path.join(EXPERIENCE_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            continue

        company = _frontmatter_value(raw, "company")
        position = _frontmatter_value(raw, "position")
        application_id = _frontmatter_value(raw, "application_id")
        jd_version_id = _frontmatter_value(raw, "jd_version_id")
        stage = _frontmatter_value(raw, "stage")
        recorded_at = (_frontmatter_value(raw, "recorded_at")
                       or _frontmatter_value(raw, "crawl_date"))
        heading = re.search(r"^#\s+亲历投递经验：(.+?)\s*·\s*(.+?)\s*$", raw, re.MULTILINE)
        if heading:
            company = company or heading.group(1).strip()
            position = position or heading.group(2).strip()
        # experience_posts 目录也可能放通用经验资料；没有明确公司+岗位的不是
        # 用户投递记录，不能混进个人状态或投递卡片。
        if not company or not position:
            continue
        if not stage:
            stage_match = re.search(r"^>\s*阶段：(.+?)(?:\s*·|$)", raw, re.MULTILINE)
            stage = stage_match.group(1).strip() if stage_match else "投递过程"

        content = raw.split("---", 2)[-1] if raw.startswith("---") else raw
        summary_lines = [
            line.strip() for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ">"))
        ]
        records.append({
            "company": company,
            "position": position,
            "application_id": application_id,
            "jd_version_id": jd_version_id,
            "stage": stage,
            "date": (recorded_at or "")[:10],
            "recorded_at": recorded_at,
            "summary": "\n".join(summary_lines).strip(),
            "path": os.path.relpath(path, BASE_DIR),
        })
    records.sort(key=lambda r: (r.get("recorded_at", ""), r.get("path", "")), reverse=True)
    return records


def _row_value(row: dict, needle: str) -> str:
    """兼容 ``parse_applications`` 的中文列名与未来带前缀的列名。"""
    return str(next((row[k] for k in row if needle in k), "") or "").strip()


def _timeline_events(note: str) -> list[dict]:
    """从现有备注中的 ``MM-DD 状态`` 标记恢复只读生命周期事件。"""
    status_alt = "|".join(re.escape(s) for s in sorted(STATUSES, key=len, reverse=True))
    return [
        {"date": m.group(1), "status": m.group(2)}
        for m in re.finditer(rf"(?<!\d)(\d{{2}}-\d{{2}})\s+({status_alt})(?!\w)", note or "")
    ]


def application_fact_views(rows: list[dict] | None = None) -> list[dict]:
    """把 Markdown 投递行转换成 RAG 可查询的统一事实视图。

    Markdown 仍是唯一事实源；该视图只做字段归一化和派生分类，不写回状态。
    """
    rows = list_applications() if rows is None else rows
    ever_by_status = {"已投递", "等待反馈", "面试中", "已 Offer", "已拒绝"}
    terminal = {"不投递", "已 Offer", "已拒绝", "主动放弃"}
    out: list[dict] = []
    for row in rows:
        status = _row_value(row, "状态")
        try:
            from domain_status import application_status_code
            status_code = application_status_code(status).value
        except ValueError:
            status_code = "unknown"
        note = _row_value(row, "备注")
        timeline = _timeline_events(note)
        timeline_statuses = {event["status"] for event in timeline}
        source = _row_value(row, "来源")
        source_url = _row_value(row, "JD来源链接")
        url_match = re.search(r"https?://[^\s|，；]+", source_url or source)
        ever_applied = status in ever_by_status or bool(
            timeline_statuses & {"已投递", "等待反馈", "面试中", "已 Offer", "已拒绝"}
        )
        out.append({
            "application_id": _row_value(row, "投递ID"),
            "date": _row_value(row, "日期"),
            "company": _row_value(row, "公司"),
            "position": _row_value(row, "岗位"),
            "source": source,
            "official_url": url_match.group(0) if url_match else "",
            "source_url": source_url if source_url != "—" else "",
            "location": _row_value(row, "地点"),
            "match_conclusion": _row_value(row, "匹配结论"),
            "audience": _row_value(row, "样本定位"),
            "status": status,
            "status_code": status_code,
            "jd_id": "" if _row_value(row, "JD标识") == "—" else _row_value(row, "JD标识"),
            "jd_version_id": "" if _row_value(row, "JD版本ID") == "—" else _row_value(row, "JD版本ID"),
            "match_id": "" if _row_value(row, "匹配ID") == "—" else _row_value(row, "匹配ID"),
            "include_in_plan": _row_value(row, "纳入计划") in {"是", "true", "True", "1"},
            "long_term_follow": _row_value(row, "长期关注") in {"是", "true", "True", "1"},
            "plan_priority": (_row_value(row, "计划优先级")
                              if _row_value(row, "计划优先级") in {"high", "medium", "low"}
                              else "medium"),
            "next_action": _row_value(row, "下一步"),
            "note": note,
            "timeline": timeline,
            "ever_applied": ever_applied,
            "pending_submission": status == "准备投递",
            "decision_pending": status == "已评估",
            "failed": status == "已拒绝",
            "terminal": status in terminal,
        })
    return out


def _normalized_entity_text(value: str) -> str:
    return re.sub(r"[\s·•_（）()\[\]【】/\\-]+", "", str(value or "").casefold())


def _company_aliases(company: str) -> set[str]:
    """从事实源公司名派生保守品牌别名；不维护面向问题的关键词补丁。"""
    value = _normalized_entity_text(company)
    aliases = {value} if value else set()
    for suffix in ("计算产品线", "产品线", "有限责任公司", "有限公司", "股份有限公司", "集团", "公司"):
        normalized_suffix = _normalized_entity_text(suffix)
        if value.endswith(normalized_suffix):
            prefix = value[:-len(normalized_suffix)]
            if len(prefix) >= 2:
                aliases.add(prefix)
    return aliases


def _filter_application_views(views: list[dict], filters: dict | None) -> list[dict]:
    filters = filters or {}
    result = list(views)
    def values(plural: str, singular: str) -> list[str]:
        raw = filters.get(plural)
        if raw is None or raw == []:
            raw = filters.get(singular)
        if raw is None or raw == "":
            return []
        if isinstance(raw, (str, int, float)):
            raw = [raw]
        return [str(item) for item in raw if str(item)]

    application_ids = set(values("application_ids", "application_id"))
    companies = {_normalized_entity_text(x) for x in values("companies", "company")}
    positions = {_normalized_entity_text(x) for x in values("positions", "position")}
    statuses = set(values("statuses", "status"))
    if application_ids:
        result = [row for row in result if row.get("application_id") in application_ids]
    if companies:
        result = [
            row for row in result
            if companies & _company_aliases(row.get("company", ""))
        ]
    if positions:
        result = [row for row in result
                  if _normalized_entity_text(row.get("position", "")) in positions]
    if statuses:
        result = [row for row in result if row.get("status") in statuses]

    entity_query = _normalized_entity_text(filters.get("entity_query", ""))
    if entity_query:
        company_matches = [
            row for row in result
            if any(alias and alias in entity_query for alias in _company_aliases(row.get("company", "")))
        ]
        # 公司是比岗位更稳定的实体锚点；只有问题没有命中任何公司时才用岗位消歧。
        if company_matches:
            result = company_matches
        else:
            position_matches = []
            for row in result:
                position = _normalized_entity_text(row.get("position", ""))
                position_alias = re.sub(r"\d{2}届$", "", position)
                if position and (position in entity_query
                                 or len(position_alias) >= 4 and position_alias in entity_query):
                    position_matches.append(row)
            if position_matches:
                result = position_matches
            elif filters.get("strict_entity_filter"):
                result = []
    return result


def select_application_facts(operations: list[str], rows: list[dict] | None = None,
                             *, filters: dict | None = None) -> dict:
    """按白名单操作和本地实体条件返回确定性投递事实。"""
    views = _filter_application_views(application_fact_views(rows), filters)
    ops = set(operations or ["list_current"])
    return {
        "operations": sorted(ops),
        "current": views if "list_current" in ops else [],
        "ever_applied": [v for v in views if v["ever_applied"]]
                        if "list_ever_applied" in ops else [],
        "pending_submission": [v for v in views if v["pending_submission"]]
                              if "list_pending_submission" in ops else [],
        "decision_pending": [v for v in views if v["decision_pending"]]
                            if "list_pending_submission" in ops else [],
        "failed": [v for v in views if v["failed"]]
                  if "list_failed" in ops else [],
        "next_actions": [v for v in views if not v["terminal"] and v["next_action"] not in ("", "—")]
                        if "get_next_actions" in ops else [],
        "freshness": max((v["date"] for v in views if v["date"]), default=""),
        "total": len(views),
    }


def render_application_facts(facts: dict) -> str:
    """把结构化事实直接渲染成 Markdown，避免 LLM 改写公司/岗位/状态/日期。"""
    ops = set(facts.get("operations") or [])
    sections: list[str] = []

    def line(v: dict, *, include_link: bool = False) -> str:
        base = (f"- {v['company']}｜{v['position']}｜状态：{v['status']}"
                f"｜状态日期：{v['date'] or '未记录'}")
        if v.get("next_action") and v["next_action"] != "—":
            base += f"｜下一步：{v['next_action']}"
        base += f"｜JD：{v.get('jd_version_id') or '未关联'}"
        if include_link:
            base += f"｜投递来源：{v.get('source') or '未记录'}"
            base += f"｜官网链接：{v.get('official_url') or '未记录'}"
        return base

    if "list_ever_applied" in ops:
        rows = facts.get("ever_applied") or []
        sections.append("### 已投递过的企业\n" + (
            "\n".join(line(v) for v in rows) if rows else "- 当前记录中没有可确认的已投递企业。"
        ))
    if "list_pending_submission" in ops:
        rows = facts.get("pending_submission") or []
        sections.append("### 待完成投递\n" + (
            "\n".join(line(v, include_link=True) for v in rows)
            if rows else "- 当前没有状态为“准备投递”的企业。"
        ))
        undecided = facts.get("decision_pending") or []
        if undecided:
            sections.append("### 已评估、仍待决定\n" + "\n".join(line(v) for v in undecided))
    if "list_failed" in ops:
        rows = facts.get("failed") or []
        sections.append("### 已拒绝的企业\n" + (
            "\n".join(line(v) for v in rows) if rows else "- 当前没有状态为“已拒绝”的企业。"
        ))
    if "get_next_actions" in ops:
        rows = facts.get("next_actions") or []
        sections.append("### 已记录的下一步动作\n" + (
            "\n".join(line(v, include_link=v.get("pending_submission", False)) for v in rows)
            if rows else "- 当前没有已记录的非终态下一步动作。"
        ))
    if "list_current" in ops:
        rows = facts.get("current") or []
        sections.append("### 当前投递状态\n" + (
            "\n".join(line(v) for v in rows) if rows else "- 暂无真实投递记录。"
        ))
    return "\n\n".join(sections)


def get_application(application_id: str) -> dict:
    return next((x for x in application_fact_views()
                 if x.get("application_id") == application_id), {})


def _upsert_application_unlocked(company: str, position: str, status: str, *,
                       date: str = "", source: str = "", location: str = "",
                       next_action: str = "", note: str = "",
                       application_id: str = "", jd_id: str = "",
                       jd_version_id: str = "", match_id: str = "", source_url: str = "",
                       match_conclusion: str = "", audience: str = "",
                       include_in_plan: bool | None = None,
                       long_term_follow: bool | None = None,
                       plan_priority: str = "medium",
                       force_new: bool = False,
                       write: bool = True) -> dict:
    """新增或更新一条投递记录，优先按稳定 ``application_id`` 定位。

    未给 ID 的旧调用只有在同公司同岗位最多一条时才能更新；多条返回
    ``conflict``，防止错误覆盖招聘批次。更新时备注仍追加时间线。
    （`MM-DD 状态` 形式），保留投递进度的时间点轨迹。
    """
    company, position = _clean_cell(company), _clean_cell(position)
    if company == "—" or position == "—":
        return {"status": "error", "error": "企业名与投递岗不能为空"}
    if status not in STATUSES:
        return {"status": "error", "error": f"进度必须是 {STATUSES} 之一"}
    if plan_priority not in {"high", "medium", "low"}:
        return {"status": "error", "error": "计划优先级必须是 high/medium/low"}
    if include_in_plan and not jd_version_id:
        return {"status": "error", "error": "没有活动 JD 的投递不能纳入学习计划"}
    date = (date or _today()).strip()

    migrated = ensure_application_schema()
    if migrated.get("status") != "ok":
        return migrated

    with open(APPLICATIONS_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 定位「投递清单」表：找含 当前状态 的表头行
    bounds = _table_bounds(lines)
    if not bounds:
        return {"status": "error", "error": "applications.md 中找不到投递清单表"}
    header_idx, body_end, headers = bounds
    body_start = header_idx + 2

    timeline_mark = f"{date[5:]} {status}"  # MM-DD 状态
    matches: list[tuple[int, list[str]]] = []
    for i in range(body_start, body_end):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        by_id = application_id and row.get("投递ID") == application_id
        by_pair = not application_id and row.get("公司") == company and row.get("岗位") == position
        if by_id or by_pair:
            matches.append((i, cells))
    if not application_id and not force_new and len(matches) > 1:
        return {"status": "conflict", "error": "同公司同岗位存在多条投递，请提供 application_id",
                "candidates": [dict(zip(headers, cells)) for _, cells in matches]}

    updated = bool(matches) and not force_new
    if updated:
            i, cells = matches[0]
            row = dict(zip(headers, cells))
            previous_status = row.get("当前状态", "")
            status_changed = previous_status != status
            row["日期"] = date
            row["当前状态"] = status
            if source:
                row["来源"] = _clean_cell(source)
            if location:
                row["地点"] = _clean_cell(location)
            if next_action:
                row["下一步动作"] = _clean_cell(next_action)
            for key, value in (("JD标识", jd_id), ("JD版本ID", jd_version_id),
                               ("匹配ID", match_id),
                               ("JD来源链接", source_url),
                               ("匹配结论", match_conclusion), ("样本定位", audience)):
                if value:
                    row[key] = _clean_cell(value)
            if include_in_plan is not None:
                row["纳入计划"] = "是" if include_in_plan else "否"
            if long_term_follow is not None:
                row["长期关注"] = "是" if long_term_follow else "否"
            row["计划优先级"] = plan_priority
            # 只有真实状态变化才追加生命周期事件；计划等字段更新不能污染时间线。
            extra = [x for x in [timeline_mark if status_changed else "",
                                  _clean_cell(note) if note else ""] if x and x != "—"]
            old_note = row.get("备注", "") if row.get("备注") != "—" else ""
            row["备注"] = _clean_cell(" · ".join([x for x in [old_note] + extra if x]))
            lines[i] = "| " + " | ".join(_clean_cell(row.get(k, "")) for k in headers) + " |"

    if not updated:
        if not application_id:
            from application_jd_store import new_application_id
            application_id = new_application_id()
        row = {k: "—" for k in headers}
        row.update({
            "投递ID": application_id, "日期": date, "公司": company, "岗位": position,
            "来源": _clean_cell(source), "JD来源链接": _clean_cell(source_url),
            "地点": _clean_cell(location), "匹配结论": _clean_cell(match_conclusion),
            "样本定位": _clean_cell(audience), "JD标识": _clean_cell(jd_id),
            "JD版本ID": _clean_cell(jd_version_id),
            "匹配ID": _clean_cell(match_id),
            "纳入计划": "是" if include_in_plan else "否",
            "长期关注": "是" if long_term_follow else "否",
            "计划优先级": plan_priority, "当前状态": status,
            "下一步动作": _clean_cell(next_action),
            "备注": _clean_cell(note) if note else timeline_mark,
        })
        lines.insert(body_end, "| " + " | ".join(_clean_cell(row.get(k, "")) for k in headers) + " |")

    rendered_text = "\n".join(lines) + "\n"
    if write:
        from io_utils import atomic_write_text
        atomic_write_text(APPLICATIONS_PATH, rendered_text)
    return {"status": "ok", "action": "updated" if updated else "added",
            "application_id": application_id or dict(zip(headers, matches[0][1])).get("投递ID", ""),
            "company": company, "position": position, "state": status, "date": date,
            "jd_id": jd_id, "jd_version_id": jd_version_id,
            "match_id": match_id,
            "include_in_plan": bool(include_in_plan),
            "long_term_follow": bool(long_term_follow),
            "_rendered_text": rendered_text if not write else ""}


def upsert_application(company: str, position: str, status: str, **kwargs) -> dict:
    """Serialize Markdown read-modify-write across Web/CLI/automation callers."""
    import hashlib
    from io_utils import atomic_write_text, file_lock
    from memory_layers import EpisodicMemory
    from memory_store import new_id
    operation_id = kwargs.pop("operation_id", None) or new_id("op")
    epi = EpisodicMemory()
    existing_event = epi.store.get_event_by_operation(operation_id)
    if existing_event:
        return {"status": "ok", "action": "idempotent",
                "application_id": existing_event.get("application_id", ""),
                "memory_event_id": existing_event["event_id"]}
    previous_operation = epi.store.get_file_operation(operation_id)
    if previous_operation and previous_operation["status"] == "pending":
        epi.store.recover_file_operations()
        existing_event = epi.store.get_event_by_operation(operation_id)
        if existing_event:
            return {"status": "ok", "action": "idempotent",
                    "application_id": existing_event.get("application_id", ""),
                    "memory_event_id": existing_event["event_id"]}
        previous_operation = epi.store.get_file_operation(operation_id)
    if previous_operation and previous_operation["status"] == "conflict":
        return {"status": "conflict", "error": "上一次操作与当前文件版本冲突"}
    with file_lock(APPLICATIONS_PATH):
        existing_event = epi.store.get_event_by_operation(operation_id)
        if existing_event:
            return {"status": "ok", "action": "idempotent",
                    "application_id": existing_event.get("application_id", ""),
                    "memory_event_id": existing_event["event_id"]}
        previous = (get_application(str(kwargs.get("application_id") or ""))
                    if kwargs.get("application_id") else {})
        before_bytes = open(APPLICATIONS_PATH, "rb").read() if os.path.exists(APPLICATIONS_PATH) else b""
        result = _upsert_application_unlocked(company, position, status, write=False, **kwargs)
        if result.get("status") != "ok":
            return result
        rendered_text = result.pop("_rendered_text")
        after_bytes = rendered_text.encode("utf-8")
        application_id = result.get("application_id", "")
        from domain_status import application_status_code
        event_payload = {
            "application_id": application_id,
            "status_code": application_status_code(status).value,
            "status": status,
            "company": company,
            "position": position,
            "long_term_follow": (kwargs.get("long_term_follow") if kwargs.get("long_term_follow") is not None
                                 else previous.get("long_term_follow", False)),
            "include_in_plan": (kwargs.get("include_in_plan") if kwargs.get("include_in_plan") is not None
                                else previous.get("include_in_plan", False)),
            "plan_priority": kwargs.get("plan_priority") or previous.get("plan_priority", "medium"),
            "next_action": kwargs.get("next_action") or previous.get("next_action", ""),
            "note": kwargs.get("note") or previous.get("note", ""),
            "previous": {key: previous.get(key) for key in (
                "status", "long_term_follow", "include_in_plan", "plan_priority", "next_action"
            )} if previous else None,
            "jd_id": kwargs.get("jd_id") or previous.get("jd_id", ""),
            "jd_version_id": kwargs.get("jd_version_id") or previous.get("jd_version_id", ""),
        }
        event_options = {
            "actor": "user", "source": "applications_store", "operation_id": operation_id,
            "entity_type": "application", "entity_id": application_id,
            "business_date": result.get("date") or _today(),
        }
        epi.store.begin_file_operation(
            operation_id, "application_changed", APPLICATIONS_PATH,
            hashlib.sha256(before_bytes).hexdigest() if before_bytes else "",
            hashlib.sha256(after_bytes).hexdigest() if after_bytes else "",
            {"event_kind": "application_changed", "event_payload": event_payload,
             "event_options": event_options},
        )
        atomic_write_text(APPLICATIONS_PATH, rendered_text)
    if result.get("status") == "ok":
        try:
            from memory_layers import record_business_event
            event = record_business_event(
                "application_changed", event_payload, **event_options,
            )
            result["memory_event_id"] = event["event_id"]
            epi.store.finish_file_operation(operation_id, status="committed",
                                            detail={"event_id": event["event_id"]})
        except Exception as exc:
            result["memory_error"] = str(exc)
    return result


def patch_application(application_id: str, **changes) -> dict:
    current = get_application(application_id)
    if not current:
        return {"status": "error", "error": "找不到 application_id"}
    include = changes.get("include_in_plan")
    if include is None:
        include = current.get("include_in_plan", False)
    long_term_follow = changes.get("long_term_follow")
    if long_term_follow is None:
        long_term_follow = current.get("long_term_follow", False)
    requested_status = changes.get("status") or current["status"]
    status_changed = requested_status != current["status"]
    # 手动推进状态时默认记录今天；只改计划/JD 等字段时保留原状态日期。
    effective_date = (changes.get("date") or
                      (_today() if status_changed else current.get("date", "")))
    return upsert_application(
        changes.get("company") or current["company"],
        changes.get("position") or current["position"],
        requested_status,
        application_id=application_id,
        date=effective_date,
        source=changes.get("source", ""), location=changes.get("location", ""),
        next_action=changes.get("next_action", ""), note=changes.get("note", ""),
        jd_id=changes.get("jd_id") or current.get("jd_id", ""),
        jd_version_id=changes.get("jd_version_id") or current.get("jd_version_id", ""),
        match_id=changes.get("match_id") or current.get("match_id", ""),
        source_url=changes.get("source_url", ""),
        match_conclusion=changes.get("match_conclusion", ""),
        audience=changes.get("audience", ""), include_in_plan=include,
        long_term_follow=long_term_follow,
        plan_priority=changes.get("plan_priority") or current.get("plan_priority", "medium"),
        operation_id=changes.get("operation_id"),
    )


def save_experience(company: str, position: str, stage: str, text: str, *,
                    application_id: str = "", jd_version_id: str = "",
                    operation_id: str = "") -> dict:
    """把一段亲历投递经验写成知识库格式的 md（experience_posts/）。

    返回 {status, saved, saved_abs}；是否增量入向量库由调用方（API 层）决定。
    """
    text = (text or "").strip()
    if len(text) < 20:
        return {"status": "error", "error": "经验总结太短（≥20 字），写点真东西"}
    os.makedirs(EXPERIENCE_DIR, exist_ok=True)
    from memory_layers import EpisodicMemory
    from memory_store import new_id
    operation_id = operation_id or new_id("op")
    epi = EpisodicMemory()
    existing = epi.store.get_event_by_operation(operation_id)
    if existing:
        saved = str(existing.get("saved_path") or "")
        return {"status": "ok", "saved": saved,
                "saved_abs": os.path.join(BASE_DIR, saved),
                "memory_event_id": existing["event_id"], "replayed": True}
    now = datetime.datetime.now()
    today = now.date().isoformat()
    recorded_at = now.isoformat(timespec="seconds")
    slug = re.sub(r"[^\w一-鿿]+", "_", f"{company}_{position}")[:40]
    stage = _clean_cell(stage)
    stage_slug = re.sub(r"[^\w一-鿿]+", "_", stage)[:16]
    op_suffix = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:12]
    fname = f"亲历_{slug}_{stage_slug}_{today}_{op_suffix}.md"
    path = os.path.join(EXPERIENCE_DIR, fname)
    body = (
        "---\n"
        f'title: "亲历投递经验：{company} · {position}（{stage}）"\n'
        f'company: "{company}"\n'
        f'position: "{position}"\n'
        f'application_id: "{application_id}"\n'
        f'jd_version_id: "{jd_version_id}"\n'
        f'stage: "{stage}"\n'
        f'recorded_at: "{recorded_at}"\n'
        f'source_url: "(用户亲历投递)"\n'
        f'crawl_date: "{today}"\n'
        'quality: "A"\n'
        'source_type: "experience"\n'
        'owner_scope: "personal"\n'
        'review_status: "approved"\n'
        f'tags: ["亲历", "投递经验", "{stage}"]\n'
        "---\n\n"
        f"# 亲历投递经验：{company} · {position}\n\n"
        f"> 阶段：{stage} · 记录日期：{today}\n"
        f"> 本文为用户亲身经历的第一手经验，对学习计划与每日建议有强指导意义。\n\n"
        f"{text}\n"
    )
    saved = os.path.relpath(path, BASE_DIR)
    snapshot = epi.store.put_snapshot(body, media_type="text/markdown", source_path=saved)
    from memory_transactions import write_text_with_memory
    memory = write_text_with_memory(
        path, body, event_kind="application_review_recorded",
        event_payload={"application_id": application_id, "company": company,
                       "position": position, "stage": stage,
                       "snapshot_id": snapshot["snapshot_id"],
                       "content_hash": snapshot["content_hash"], "saved_path": saved},
        event_options={"actor": "user", "source": "application_review",
                       "entity_type": "application_review",
                       "entity_id": op_suffix, "business_date": today},
        operation_id=operation_id,
    )
    return {"status": "ok", "saved": saved, "saved_abs": path, **memory}
