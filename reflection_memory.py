# -*- coding: utf-8 -*-
"""长期执行/复盘记忆的本地读取与派生检索。

事实源仍是 ``daily_log.md``、``summaries/*.md`` 与 append-only episodic；本模块
只构造可重建视图。明确日期优先确定性过滤，主题查询先词面检索，弱命中时可使用
独立 Chroma collection，不与通用知识库竞争 Top-K。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from io_utils import atomic_write_json


BASE_DIR = Path(__file__).resolve().parent
DAILY_LOG_PATH = BASE_DIR / "daily_log.md"
SUMMARY_DIR = BASE_DIR / "summaries"
DERIVED_DIR = BASE_DIR / "logs" / "reflection"
DERIVED_INDEX_PATH = DERIVED_DIR / "index.json"
_REFLECTION_META_RE = re.compile(r"<!--\s*offerclaw-reflection:\s*(\{.*?\})\s*-->")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _all_log_dates(log: str) -> list[str]:
    return sorted(set(re.findall(r"^##\s+(\d{4}-\d{2}-\d{2})", log or "", re.MULTILINE)))


def daily_execution_documents() -> list[dict[str, Any]]:
    from summary_tool import extract_log_entries

    log = _read(DAILY_LOG_PATH)
    docs: list[dict[str, Any]] = []
    for date in _all_log_dates(log):
        for entry in extract_log_entries(log, date):
            text = entry.pop("block", "")
            docs.append({
                "id": entry["log_id"], "source_type": "daily_execution",
                "date_from": date, "date_to": date, "title": f"{date} 每日执行",
                "path": "daily_log.md", "text": text, "source_status": "valid",
                "matched_by": "deterministic", "metadata": entry,
                "content_hash": entry.get("content_hash") or _hash(text),
            })
    return docs


def _summary_identity(path: Path, content: str) -> tuple[str, str, str]:
    name = path.name
    m = re.match(r"summary_(daily|weekly)_(\d{4}-\d{2}-\d{2})", name)
    kind, date = (m.group(1), m.group(2)) if m else ("daily", "")
    mm = _REFLECTION_META_RE.search(content)
    meta: dict = {}
    if mm:
        try:
            meta = json.loads(mm.group(1))
        except json.JSONDecodeError:
            pass
    rid = str(meta.get("reflection_id") or f"legacy_refl_{_hash(name + content)[:16]}")
    return rid, kind, date


def reflection_documents() -> list[dict[str, Any]]:
    from summary_tool import extract_log_entries

    log = _read(DAILY_LOG_PATH)
    log_dates = _all_log_dates(log)
    log_entries_by_date = {date: extract_log_entries(log, date) for date in log_dates}
    available_log_ids = {
        entry["log_id"] for entries in log_entries_by_date.values() for entry in entries
    }
    docs: list[dict[str, Any]] = []
    for path in sorted(SUMMARY_DIR.glob("summary_*.md")) if SUMMARY_DIR.exists() else []:
        content = _read(path)
        rid, inferred_kind, inferred_date = _summary_identity(path, content)
        mm = _REFLECTION_META_RE.search(content)
        meta: dict = {}
        if mm:
            try:
                meta = json.loads(mm.group(1))
            except json.JSONDecodeError:
                pass
        kind = str(meta.get("kind") or inferred_kind)
        date_from = str(meta.get("date_from") or inferred_date)
        date_to = str(meta.get("date_to") or inferred_date)
        if kind == "weekly":
            source_dates = [d for d in log_dates if date_from <= d <= date_to]
        else:
            source_dates = [date_from] if date_from in log_dates else []
        source_log_ids = list(meta.get("source_log_ids") or [
            e["log_id"] for d in source_dates for e in log_entries_by_date.get(d, [])
        ])
        # 状态必须由当前事实源重新校验，不能信任复盘文件里曾经写入的 valid；
        # 原始日志被删除或引用失效后，该复盘立即降级为 orphaned。
        source_status = "valid" if any(x in available_log_ids for x in source_log_ids) else "orphaned"
        clean = _REFLECTION_META_RE.sub("", content).strip()
        docs.append({
            "id": rid, "source_type": "reflection_summary", "kind": kind,
            "date_from": date_from, "date_to": date_to,
            "title": f"{date_from}{' ~ ' + date_to if date_to != date_from else ''} {kind} 复盘",
            "path": _display_path(path), "text": clean,
            "source_status": source_status, "matched_by": "lexical",
            "content_hash": str(meta.get("content_hash") or _hash(clean)),
            "metadata": {**meta, "source_log_ids": source_log_ids},
        })

    # 结构化事件可能尚无对应 Markdown（历史兼容）；作为可检索但较低权威的视图补入。
    try:
        from memory_layers import EpisodicMemory
        known = {d["id"] for d in docs}
        for event in EpisodicMemory().all():
            if event.get("kind") != "reflection":
                continue
            rid = str(event.get("reflection_id") or f"legacy_event_{event.get('id', _hash(str(event))[:12])}")
            if rid in known:
                continue
            date = str(event.get("date") or event.get("date_from") or "")
            text = "\n".join([
                f"主线：{event.get('main_tag', '')}",
                "完成：" + "；".join(event.get("completed") or []),
                "未完成：" + "；".join(event.get("incomplete") or []),
                "阻碍：" + "；".join(event.get("blockers") or []),
                f"下一步：{event.get('next_day_suggestion', '')}",
            ]).strip()
            source_ids = list(event.get("source_log_ids") or [])
            source_status = ("valid" if any(x in available_log_ids for x in source_ids)
                             else ("valid" if not source_ids and date in log_dates else "orphaned"))
            docs.append({
                "id": rid, "source_type": "reflection_event", "kind": "daily",
                "date_from": date, "date_to": str(event.get("date_to") or date),
                "title": f"{date} 结构化复盘", "path": "logs/memory/episodic.jsonl",
                "text": text, "source_status": source_status,
                "matched_by": "structured", "content_hash": event.get("content_hash") or _hash(text),
                "metadata": event,
            })
    except Exception:
        pass
    return docs


def build_inventory() -> dict[str, Any]:
    daily = daily_execution_documents()
    reflections = reflection_documents()
    duplicate_dates: dict[str, int] = {}
    for d in daily:
        date = d["date_from"]
        duplicate_dates[date] = duplicate_dates.get(date, 0) + 1
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "daily": daily, "reflections": reflections,
        "stats": {
            "daily_entries": len(daily), "reflection_entries": len(reflections),
            "orphaned_reflections": sum(d.get("source_status") == "orphaned" for d in reflections),
            "duplicate_dates": {k: v for k, v in duplicate_dates.items() if v > 1},
        },
    }


def write_derived_index() -> dict[str, Any]:
    inventory = build_inventory()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(DERIVED_INDEX_PATH), inventory)
    return inventory


def _tokens(text: str) -> list[str]:
    text = (text or "").lower()
    latin = re.findall(r"[a-z][a-z0-9_+.-]{1,}", text)
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    zh = [chunk[i:i + 2] for chunk in chunks for i in range(len(chunk) - 1)]
    return latin + zh


def _lexical_rank(query: str, docs: list[dict], limit: int) -> list[dict]:
    q = set(_tokens(query))
    if not q:
        return []
    ranked = []
    for doc in docs:
        tokens = _tokens(doc.get("title", "") + "\n" + doc.get("text", ""))
        if not tokens:
            continue
        overlap = q & set(tokens)
        if not overlap:
            continue
        score = sum(1.0 + math.log1p(tokens.count(tok)) for tok in overlap) / math.sqrt(len(set(tokens)))
        if doc.get("source_status") == "orphaned":
            score *= 0.35
        ranked.append((score, doc))
    ranked.sort(key=lambda item: (item[0], item[1].get("date_to", "")), reverse=True)
    out = []
    for score, doc in ranked[:limit]:
        out.append({**doc, "score": round(score, 5), "matched_by": "lexical"})
    return out


def _date_range_from_query(query: str) -> tuple[str, str] | None:
    explicit = re.findall(r"\d{4}-\d{2}-\d{2}", query or "")
    if explicit:
        return min(explicit), max(explicit)
    if any(cue in query for cue in ("以前", "之前", "曾经", "上次", "记得")):
        return None
    today = dt.date.today()
    if "昨天" in query:
        day = today - dt.timedelta(days=1)
        return day.isoformat(), day.isoformat()
    if "今天" in query:
        return today.isoformat(), today.isoformat()
    if "上周" in query:
        start = today - dt.timedelta(days=today.weekday() + 7)
        return start.isoformat(), (start + dt.timedelta(days=6)).isoformat()
    if "本周" in query:
        start = today - dt.timedelta(days=today.weekday())
        return start.isoformat(), today.isoformat()
    return None


def _compact_items(values: list[Any], limit: int = 5) -> list[str]:
    out: list[str] = []
    for value in values or []:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" -：:")
        if not text or text in {"无", "笔记", "未完成", "已完成", "状态", "受阻原因"}:
            continue
        if text not in out:
            out.append(text[:240])
        if len(out) >= limit:
            break
    return out


def history_overview_document(inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造覆盖全部可信执行历史的确定性总览，不依赖 Top-K 或生成模型。"""
    inventory = inventory or build_inventory()
    daily = [d for d in inventory["daily"] if d.get("source_status") == "valid"]
    reflections = [d for d in inventory["reflections"] if d.get("source_status") == "valid"]
    daily.sort(key=lambda d: (d.get("date_from", ""), d.get("id", "")))
    reflections.sort(key=lambda d: (d.get("date_from", ""), d.get("id", "")))

    dates = sorted({d.get("date_from", "") for d in daily if d.get("date_from")})
    lines = [
        "# 个人学习与执行历史总览",
        (f"覆盖日期：{dates[0]} 至 {dates[-1]}；{len(dates)} 个有效日期，"
         f"{len(daily)} 条执行记录，{len(reflections)} 条有效复盘。") if dates
        else "当前没有可验证的每日执行记录。",
    ]
    log_ids: list[str] = []
    for doc in daily:
        meta = doc.get("metadata") or {}
        log_ids.append(str(doc.get("id", "")))
        tag = re.sub(r"^[：:→\s]+", "", str(meta.get("main_tag") or "")).strip()
        completed = _compact_items(list(meta.get("completed") or []), 5)
        incomplete = _compact_items(list(meta.get("incomplete") or []), 2)
        parts = []
        if tag:
            parts.append(f"主线：{tag[:100]}")
        if completed:
            parts.append("完成：" + "；".join(completed))
        if incomplete:
            parts.append("未完成/阻碍：" + "；".join(incomplete))
        if not parts:
            body = re.sub(r"\s+", " ", str(doc.get("text") or "")).strip()
            parts.append("记录：" + body[:500])
        lines.append(f"- {doc.get('date_from', '')}：" + "；".join(parts))

    reflection_ids: list[str] = []
    if reflections:
        lines.append("\n## 有效复盘结论")
    for doc in reflections:
        reflection_ids.append(str(doc.get("id", "")))
        meta = doc.get("metadata") or {}
        completed = _compact_items(list(meta.get("completed") or []), 3)
        blockers = _compact_items(list(meta.get("blockers") or []), 2)
        evidence = list(meta.get("skill_evidence") or [])
        parts = []
        if completed:
            parts.append("完成：" + "；".join(completed))
        if blockers:
            parts.append("阻碍：" + "；".join(blockers))
        if evidence:
            parts.append("能力证据：" + json.dumps(evidence[:5], ensure_ascii=False))
        if not parts:
            clean = re.sub(r"\s+", " ", str(doc.get("text") or "")).strip()
            parts.append(clean[:600])
        lines.append(f"- {doc.get('date_from', '')}：" + "；".join(parts))

    orphaned = int((inventory.get("stats") or {}).get("orphaned_reflections") or 0)
    if orphaned:
        lines.append(f"\n注：另有 {orphaned} 条孤立复盘未关联有效日志，已从事实总览排除。")
    text = "\n".join(lines)
    identity = _hash("|".join(log_ids + reflection_ids))[:16]
    return {
        "id": f"history_overview_{identity}",
        "source_type": "reflection_overview",
        "date_from": dates[0] if dates else "",
        "date_to": dates[-1] if dates else "",
        "title": "个人学习与执行历史总览",
        "path": "daily_log.md + summaries/",
        "text": text,
        "source_status": "valid" if daily else "no_evidence",
        "matched_by": "deterministic_history",
        "content_hash": _hash(text),
        "metadata": {
            "source_log_ids": log_ids,
            "reflection_ids": reflection_ids,
            "covered_dates": dates,
            "excluded_orphaned_reflections": orphaned,
        },
    }


def _dense_rank(query: str, docs: list[dict], limit: int) -> list[dict]:
    """Read the background-maintained personal index without mutating it."""
    try:
        from logging_utils import current_request_id
        from memory_search import search_personal_memory
        result = search_personal_memory(
            query, purpose="recall", limit=max(20, limit),
            exclude_operation_id=current_request_id() or "",
            exclude_latest_exact_user_query=True,
        )
        return [{
            "id": item["id"], "source_type": item["source_type"],
            "date_from": item.get("date", ""), "date_to": item.get("date", ""),
            "title": item.get("title", "个人记忆"), "path": item.get("source", "memory.sqlite3"),
            "text": item.get("text", ""), "source_status": "valid",
            "matched_by": item.get("matched_by", "dense"), "score": item.get("score", 0),
            "content_hash": _hash(item.get("text", "")),
            "metadata": {"event_id": item["event_id"], "entity_id": item.get("entity_id"),
                         "target_context_id": item.get("target_context_id")},
        } for item in result.get("items", [])]
    except Exception:
        return []


def search_memory(query: str, operation: str = "search_topic", limit: int = 5,
                  include_orphaned: bool = False) -> list[dict]:
    inventory = build_inventory()
    daily, reflections = inventory["daily"], inventory["reflections"]
    docs = daily + reflections
    if operation == "get_history_overview":
        overview = history_overview_document(inventory)
        return [overview] if overview.get("source_status") == "valid" else []
    if not include_orphaned:
        docs = [d for d in docs if d.get("source_status") == "valid"]
        reflections = [d for d in reflections if d.get("source_status") == "valid"]
    if operation in {"get_recent", "get_by_date"}:
        wanted = _date_range_from_query(query)
        if wanted:
            start, end = wanted
            docs = [d for d in docs if d.get("date_to", "") >= start and d.get("date_from", "") <= end]
        docs.sort(key=lambda d: (d.get("date_to", ""), d.get("id", "")), reverse=True)
        return [{**d, "matched_by": "deterministic_date"} for d in docs[:limit]]

    if operation == "get_profile_evidence":
        evidence_docs = []
        for doc in reflections:
            skill = (doc.get("metadata") or {}).get("skill_evidence") or []
            if skill:
                clone = dict(doc)
                clone["text"] = doc.get("text", "") + "\n技能证据：" + json.dumps(skill, ensure_ascii=False)
                evidence_docs.append(clone)
        docs = evidence_docs or reflections

    pool = max(20, limit * 4)
    lexical = _lexical_rank(query, docs, pool)
    personal = _dense_rank(query, docs, pool)
    by_id: dict[str, dict] = {}
    scores: dict[str, float] = {}
    methods: dict[str, set[str]] = {}
    for method, ranked in (("lexical", lexical), ("personal_hybrid", personal)):
        for rank, doc in enumerate(ranked, 1):
            key = str((doc.get("metadata") or {}).get("entity_id") or doc["id"])
            by_id.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            methods.setdefault(key, set()).add(method)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [{**by_id[key], "score": round(scores[key], 6),
             "matched_by": "+".join(sorted(methods[key]))}
            for key in ordered[:max(1, limit)]]


def inventory_report() -> dict[str, Any]:
    inv = build_inventory()
    return {"stats": inv["stats"], "index_path": str(DERIVED_INDEX_PATH),
            "would_write": len(inv["daily"]) + len(inv["reflections"])}
