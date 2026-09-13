# -*- coding: utf-8 -*-
"""Application-bound JD snapshots and learning-target projections.

JD analysis remains ephemeral until ``commit_from_jd`` is called.  Approved
snapshots live outside the curated knowledge base and are loaded by stable IDs;
the optional vector index is therefore always a rebuildable derivative.
"""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import os
import re
import secrets
import copy
import threading
import time
from typing import Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "application_jds")
PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.md")

PLAN_DEFAULT_ON = {"准备投递", "已投递", "等待反馈", "面试中"}
PLAN_TERMINAL = {"不投递", "已拒绝", "主动放弃", "已 Offer"}
PLAN_PRIORITIES = {"high", "medium", "low"}
_PREVIEW_TTL_SECONDS = 15 * 60
_preview_lock = threading.Lock()
_preview_cache: dict[str, dict] = {}


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return dt.date.today().isoformat()


def _token(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d')}_{secrets.token_hex(4)}"


def new_application_id() -> str:
    return _token("app")


def normalize_jd_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", str(text or "")).strip()


def jd_content_hash(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", normalize_jd_text(text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def profile_fingerprint() -> str:
    try:
        with open(PROFILE_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _cache_preview_material(*, text: str, source_url: str, parsed: dict, match: dict) -> str:
    """Keep a short-lived, server-trusted preview for the following confirm call."""
    preview_id = _token("jd_preview")
    now = time.monotonic()
    with _preview_lock:
        expired = [key for key, value in _preview_cache.items()
                   if value.get("expires_at", 0) <= now]
        for key in expired:
            _preview_cache.pop(key, None)
        _preview_cache[preview_id] = {
            "content_hash": jd_content_hash(text),
            "source_url": str(source_url or ""),
            "profile_fingerprint": profile_fingerprint(),
            "parsed": copy.deepcopy(parsed),
            "match": copy.deepcopy(match),
            "expires_at": now + _PREVIEW_TTL_SECONDS,
        }
    return preview_id


def _load_preview_material(preview_id: str, *, text: str, source_url: str) -> dict:
    if not preview_id:
        return {}
    now = time.monotonic()
    with _preview_lock:
        cached = _preview_cache.get(preview_id)
        if not cached or cached.get("expires_at", 0) <= now:
            _preview_cache.pop(preview_id, None)
            return {}
        if (cached.get("content_hash") != jd_content_hash(text)
                or cached.get("source_url") != str(source_url or "")
                or cached.get("profile_fingerprint") != profile_fingerprint()):
            return {}
        return copy.deepcopy(cached)


def default_include_in_plan(status: str) -> bool:
    return status in PLAN_DEFAULT_ON


def _source_label(source_type: str, source_url: str) -> str:
    if source_type == "official":
        u = (source_url or "").lower()
        if "mokahr.com" in u:
            return "公司招聘官网（Moka）"
        if "feishu.cn" in u:
            return "公司招聘官网（飞书招聘）"
        return "公司招聘官网"
    if source_type == "platform":
        try:
            from urllib.parse import urlparse
            return urlparse(source_url).netloc or "招聘平台"
        except Exception:
            return "招聘平台"
    if source_type == "manual_paste":
        return "用户粘贴"
    return ""


def _field_confidence(parsed: dict) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for field in ("company", "title", "location", "job_type"):
        value = str(parsed.get(field) or "").strip()
        if not value:
            out[field] = {"confidence": "missing", "source": "待用户填写"}
        elif field == "title" and value == normalize_jd_text(parsed.get("jd_text", "")).split("\n", 1)[0][:60]:
            out[field] = {"confidence": "medium", "source": "JD 首行推断"}
        else:
            out[field] = {"confidence": "high", "source": "JD 明确字段"}
    return out


def _match_payload(jd_text: str, title: str, jd_analysis: dict | None = None) -> dict:
    from match_job import format_report, run_match
    from profile_loader import load_profile

    if not jd_analysis:
        from jd_parser import analyze_jd
        jd_analysis = analyze_jd(jd_text).model_dump(mode="json")
    report = run_match(
        load_profile(), jd_text, jd_title=title or "未命名 JD",
        jd_analysis=jd_analysis,
    )
    from domain_status import match_status_code
    return {
        "status": report.conclusion,
        "status_code": match_status_code(report.conclusion).value,
        "direction": report.direction,
        "summary": format_report(report),
        "gap_list": report.gap_list or {},
        "suggestions": report.suggestions or [],
        "requirement_analysis": report.requirement_analysis or {},
    }


def preview_from_jd(jd_text: str, *, source_url: str = "",
                    application_id: str = "") -> dict:
    """Build an application draft without persisting the JD or match output."""
    text = normalize_jd_text(jd_text)
    if len(text) < 20:
        return {"status": "error", "error": "JD 原文过短（至少 20 字）"}

    from job_discovery import discover
    parsed = discover(raw=text, url=source_url)
    parsed["jd_text"] = text
    source_type = str(parsed.get("source_type") or "unknown")
    company = str(parsed.get("company") or "").strip()
    position = str(parsed.get("title") or "").strip()
    match = _match_payload(text, position, parsed.get("jd_analysis") or {})
    preview_id = _cache_preview_material(
        text=text, source_url=source_url, parsed=parsed, match=match,
    )

    from applications_store import application_fact_views
    candidates = []
    comparisons: dict[str, dict] = {}
    content_hash = jd_content_hash(text)
    for row in application_fact_views():
        snap: dict = {}
        same_content = False
        if row.get("jd_version_id"):
            snap = load_snapshot(row["jd_version_id"], jd_id=row.get("jd_id", ""))
            same_content = snap.get("content_hash") == content_hash
        same_company = company and row.get("company") == company
        same_position = position and row.get("position") == position
        if application_id and row.get("application_id") == application_id:
            score, reason = 1.0, "用户指定的现有投递"
        elif same_content:
            score, reason = 1.0, "JD 内容完全一致"
        elif same_company and same_position:
            score, reason = 1.0, "公司与岗位完全一致"
        elif same_company:
            score, reason = 0.6, "公司一致，岗位需人工确认"
        else:
            continue
        candidate = {
            "application_id": row.get("application_id", ""),
            "company": row.get("company", ""),
            "position": row.get("position", ""),
            "status": row.get("status", ""),
            "jd_version_id": row.get("jd_version_id", ""),
            "score": score,
            "reason": reason,
            "same_content": same_content,
        }
        if snap:
            old_text = snap.get("jd_text", "")
            diff = "\n".join(difflib.unified_diff(
                old_text.splitlines(), text.splitlines(),
                fromfile=row["jd_version_id"], tofile="本次待确认 JD", lineterm="",
            )) if old_text else ""
            comparisons[row.get("application_id", "")] = {
                "current_jd_version_id": row.get("jd_version_id", ""),
                "same_content": same_content,
                "diff": diff[:5000],
            }
        candidates.append(candidate)

    draft = {
        "company": company,
        "position": position,
        "location": str(parsed.get("location") or "").strip(),
        "source": _source_label(source_type, source_url),
        "source_url": source_url,
        "status": "已评估",
        "date": _today(),
        "match_conclusion": match["status"],
        "audience": match["direction"],
        "next_action": "决定投递/不投递",
        "note": "",
        "include_in_plan": False,
        "plan_priority": "medium",
    }
    return {
        "status": "ok",
        "draft": draft,
        "field_provenance": _field_confidence(parsed),
        "parsed": {k: parsed.get(k) for k in (
            "company", "title", "location", "job_type", "skills_detected",
            "duties", "requirements", "career_domain", "role_family",
            "source_type", "source_credibility", "source_url", "raw_chars",
        )},
        "jd_analysis": parsed.get("jd_analysis") or {},
        "match": match,
        "preview_id": preview_id,
        "content_hash": content_hash,
        "existing_candidates": sorted(candidates, key=lambda x: x["score"], reverse=True),
        "existing_comparisons": comparisons,
        "ephemeral": True,
    }


def _snapshot_path(jd_id: str, jd_version_id: str) -> str:
    return os.path.join(SNAPSHOT_DIR, jd_id, f"{jd_version_id}.md")


def _match_path(jd_id: str, jd_version_id: str, match_id: str) -> str:
    return os.path.join(SNAPSHOT_DIR, jd_id, f"{jd_version_id}.{match_id}.match.json")


def _quote(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _snapshot_markdown(*, application_id: str, jd_id: str, jd_version_id: str,
                       company: str, position: str, location: str,
                       source_url: str, source_type: str,
                       source_credibility: str, content_hash: str,
                       captured_at: str, jd_text: str) -> str:
    return (
        "---\n"
        f"jd_id: {_quote(jd_id)}\n"
        f"jd_version_id: {_quote(jd_version_id)}\n"
        f"application_id: {_quote(application_id)}\n"
        f"company: {_quote(company)}\n"
        f"position: {_quote(position)}\n"
        f"job_id: \"\"\n"
        f"location: {_quote(location)}\n"
        f"source_url: {_quote(source_url)}\n"
        f"source_type: \"application_jd\"\n"
        f"source_origin: {_quote(source_type)}\n"
        f"source_credibility: {_quote(source_credibility)}\n"
        f"captured_at: {_quote(captured_at)}\n"
        f"content_hash: {_quote(content_hash)}\n"
        "owner_scope: \"personal\"\n"
        "review_status: \"approved\"\n"
        "---\n\n"
        f"# {company} · {position}\n\n"
        "> 本文是该次投递绑定的 JD 原始快照；版本不可覆盖。\n\n"
        "## JD 原文\n\n"
        f"{jd_text.rstrip()}\n"
    )


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    head = text.split("---\n", 2)[1]
    out: dict[str, str] = {}
    for line in head.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip()
        try:
            out[key.strip()] = str(json.loads(raw))
        except Exception:
            out[key.strip()] = raw.strip('"')
    return out


def load_snapshot(jd_version_id: str, *, jd_id: str = "") -> dict:
    paths: list[str] = []
    if jd_id:
        paths.append(_snapshot_path(jd_id, jd_version_id))
    elif os.path.isdir(SNAPSHOT_DIR):
        for name in os.listdir(SNAPSHOT_DIR):
            paths.append(_snapshot_path(name, jd_version_id))
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        body = raw.split("---\n", 2)[-1]
        marker = "## JD 原文\n\n"
        return {
            **_frontmatter(raw),
            "jd_text": body.split(marker, 1)[1].strip() if marker in body else body.strip(),
            "path": os.path.relpath(path, BASE_DIR),
        }
    return {}


def load_bound_jd(application_id: str, jd_version_id: str = "") -> dict:
    """Load one application's active immutable JD snapshot.

    ``jd_version_id`` is an optimistic-lock value supplied by the UI.  It may be
    omitted, but when present it must still be the application's active version;
    callers never get to select an unrelated or historical snapshot by ID alone.
    """
    from applications_store import get_application

    application_id = (application_id or "").strip()
    if not application_id:
        raise ValueError("缺少 application_id")
    application = get_application(application_id)
    if not application:
        raise KeyError("找不到 application_id")
    active_version = (application.get("jd_version_id") or "").strip()
    if not active_version:
        raise ValueError("该投递尚未关联活动 JD")
    expected_version = (jd_version_id or "").strip()
    if expected_version and expected_version != active_version:
        raise RuntimeError("所选 JD 已更新，请刷新投递清单后重新选择")
    snapshot = load_snapshot(active_version, jd_id=application.get("jd_id", ""))
    if not snapshot:
        raise ValueError("该投递的活动 JD 快照不存在")
    snapshot_application_id = (snapshot.get("application_id") or "").strip()
    if snapshot_application_id and snapshot_application_id != application_id:
        raise RuntimeError("JD 快照与投递记录的绑定关系不一致")
    return {
        "application": application,
        "snapshot": snapshot,
        "application_id": application_id,
        "jd_version_id": active_version,
        "company": application.get("company", ""),
        "position": application.get("position", ""),
        "status": application.get("status", ""),
    }


def list_versions(jd_id: str) -> list[dict]:
    root = os.path.join(SNAPSHOT_DIR, jd_id)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root), reverse=True):
        if not name.endswith(".md"):
            continue
        item = load_snapshot(name[:-3], jd_id=jd_id)
        if item:
            out.append({k: item.get(k, "") for k in (
                "jd_id", "jd_version_id", "application_id", "company", "position",
                "source_url", "captured_at", "content_hash", "path",
            )})
    return out


def load_match_snapshot(jd_id: str, jd_version_id: str, match_id: str = "") -> dict:
    path = _match_path(jd_id, jd_version_id, match_id) if match_id else ""
    if not path or not os.path.exists(path):
        root = os.path.join(SNAPSHOT_DIR, jd_id)
        prefix = f"{jd_version_id}."
        candidates = [os.path.join(root, x) for x in os.listdir(root)] if os.path.isdir(root) else []
        candidates = sorted((x for x in candidates
                             if os.path.basename(x).startswith(prefix) and x.endswith(".match.json")),
                            reverse=True)
        path = candidates[0] if candidates else ""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def prepare_approved_artifacts(*, application_id: str, jd_text: str,
                               company: str, position: str, location: str = "",
                               source_url: str = "", expected_hash: str = "",
                               existing_jd_id: str = "", preview_id: str = "") -> dict:
    """Validate and materialize immutable artifacts before the application row is linked."""
    text = normalize_jd_text(jd_text)
    digest = jd_content_hash(text)
    if expected_hash and expected_hash != digest:
        return {"status": "error", "error": "JD 内容已变化，请重新预览后确认"}
    if not company.strip() or not position.strip():
        return {"status": "error", "error": "企业名与岗位必须由用户确认"}
    if len(text) < 20:
        return {"status": "error", "error": "JD 原文过短（至少 20 字）"}

    cached_preview = _load_preview_material(preview_id, text=text, source_url=source_url)
    if cached_preview:
        parsed = cached_preview["parsed"]
        match = cached_preview["match"]
    else:
        from job_discovery import discover
        parsed = discover(raw=text, url=source_url)
        match = _match_payload(text, position, parsed.get("jd_analysis") or {})
    source_type = str(parsed.get("source_type") or "unknown")
    credibility = str(parsed.get("source_credibility") or "C")
    jd_id = existing_jd_id or _token("jd")
    existing = next((x for x in list_versions(jd_id) if x.get("content_hash") == digest), None)
    jd_version_id = str(existing.get("jd_version_id")) if existing else f"jdv_{digest[:16]}"
    match_id = _token("match")
    captured_at = _now()

    os.makedirs(os.path.join(SNAPSHOT_DIR, jd_id), exist_ok=True)
    snapshot_path = _snapshot_path(jd_id, jd_version_id)
    match_path = _match_path(jd_id, jd_version_id, match_id)
    if not os.path.exists(snapshot_path):
        from io_utils import atomic_write_text
        atomic_write_text(snapshot_path, _snapshot_markdown(
            application_id=application_id, jd_id=jd_id, jd_version_id=jd_version_id,
            company=company, position=position, location=location,
            source_url=source_url, source_type=source_type,
            source_credibility=credibility, content_hash=digest,
            captured_at=captured_at, jd_text=text,
        ))
    match_payload = {
        "match_id": match_id,
        "application_id": application_id,
        "jd_id": jd_id,
        "jd_version_id": jd_version_id,
        "profile_fingerprint": profile_fingerprint(),
        "generated_at": captured_at,
        **match,
    }
    from io_utils import atomic_write_json
    atomic_write_json(match_path, match_payload)
    return {
        "status": "ok", "jd_id": jd_id, "jd_version_id": jd_version_id,
        "match_id": match_id, "match": match_payload,
        "snapshot_path": snapshot_path, "match_path": match_path,
        "content_hash": digest, "version_action": "reused" if existing else "created",
        "source_type": source_type, "source_credibility": credibility,
        "preview_reused": bool(cached_preview),
    }


def snapshot_summary(jd_id: str, jd_version_id: str, match_id: str = "") -> dict:
    snap = load_snapshot(jd_version_id, jd_id=jd_id) if jd_version_id else {}
    match = load_match_snapshot(jd_id, jd_version_id, match_id) if snap else {}
    return {
        "linked": bool(snap),
        "jd_id": jd_id,
        "jd_version_id": jd_version_id,
        "source_url": snap.get("source_url", ""),
        "captured_at": snap.get("captured_at", ""),
        "content_hash": snap.get("content_hash", ""),
        "version_count": len(list_versions(jd_id)) if jd_id else 0,
        "match_id": match.get("match_id", ""),
        "match_stale": bool(match) and match.get("profile_fingerprint") != profile_fingerprint(),
        "match": match,
    }


def find_active_duplicate_application(company: str, position: str, content_hash: str,
                                      *, exclude_application_id: str = "") -> dict:
    """Find an active application already bound to the exact same JD content.

    A repeated click must not create another active application just because the
    original save is still running. Terminal rows are deliberately excluded: a
    later recruitment cycle may legitimately reuse an unchanged JD.
    """
    wanted_identity = active_jd_identity({
        "company": company,
        "position": position,
        "content_hash": content_hash,
    })
    if not wanted_identity:
        return {}

    from applications_store import application_fact_views
    for app in application_fact_views():
        if app.get("application_id") == exclude_application_id or app.get("terminal"):
            continue
        summary = snapshot_summary(
            app.get("jd_id", ""), app.get("jd_version_id", ""), app.get("match_id", ""),
        )
        if active_jd_identity({**app, "jd": summary}) == wanted_identity:
            return {
                "application_id": app.get("application_id", ""),
                "company": app.get("company", ""),
                "position": app.get("position", ""),
                "status": app.get("status", ""),
                "jd_version_id": app.get("jd_version_id", ""),
            }
    return {}


def active_jd_identity(row: dict) -> str:
    """Return the exact-JD identity for an active application, or an empty key.

    This is deliberately narrower than a company/position duplicate check.  Two
    recruitment rounds for the same role can coexist when their immutable JD
    snapshots differ; only an active row bound to the same content is a retry
    duplicate that should be represented once in downstream planning.
    """
    if row.get("terminal") or str(row.get("status") or "") in PLAN_TERMINAL:
        return ""
    company = re.sub(r"\s+", "", str(row.get("company") or "")).casefold()
    position = re.sub(r"\s+", "", str(row.get("position") or "")).casefold()
    jd = row.get("jd") if isinstance(row.get("jd"), dict) else {}
    content_hash = str(jd.get("content_hash") or row.get("content_hash") or "").strip()
    if not company or not position or not content_hash:
        return ""
    return "\x1f".join((company, position, content_hash))


def plan_targets() -> dict:
    """Derive current learning targets from applications; no independent target DB."""
    from applications_store import application_fact_views

    included, excluded = [], []
    for app in application_fact_views():
        summary = snapshot_summary(app.get("jd_id", ""), app.get("jd_version_id", ""),
                                   app.get("match_id", ""))
        reason = ""
        if not summary["linked"]:
            reason = "JD 未关联"
        elif not app.get("include_in_plan"):
            reason = "未勾选纳入学习计划"
        target = {**app, "jd": {k: summary[k] for k in summary if k != "match"},
                  "match": summary.get("match") or {}}
        if reason:
            target["excluded_reason"] = reason
            excluded.append(target)
        else:
            included.append(target)
    priority = {"high": 0, "medium": 1, "low": 2}
    stage = {"面试中": 0, "等待反馈": 1, "已投递": 2, "准备投递": 3, "已评估": 4}
    included.sort(key=lambda x: (
        priority.get(x.get("plan_priority", "medium"), 1),
        stage.get(x.get("status", ""), 9), x.get("date", ""),
    ))
    unique_included: list[dict] = []
    representatives: dict[str, dict] = {}
    for target in included:
        identity = active_jd_identity(target)
        representative = representatives.get(identity) if identity else None
        if representative is None:
            unique_included.append(target)
            if identity:
                representatives[identity] = target
            continue
        duplicate = {
            **target,
            "duplicate_of": representative.get("application_id", ""),
            "excluded_reason": "同公司、同岗位、同一 JD 内容的重复保存；已保留一条代表记录",
        }
        excluded.append(duplicate)
    included = unique_included
    return {"included": included, "excluded": excluded, "total": len(included),
            "generated_at": _now()}


def plan_gaps_text(max_chars: int = 5000) -> str:
    targets = plan_targets()["included"]
    if not targets:
        return ""
    lines: list[str] = ["【本计划服务的投递 JD】"]
    for target in targets:
        lines.append(
            f"- {target.get('company')}｜{target.get('position')}｜"
            f"application_id={target.get('application_id')}｜"
            f"jd_version_id={target.get('jd_version_id')}｜优先级={target.get('plan_priority')}"
        )
    gap_items = plan_gap_items(targets)
    lines.extend(["", "【可进入学习计划的能力缺口】"])
    for item in [x for x in gap_items if x["learnable"]]:
        sources = "；".join(
            f"{s['company']}·{s['position']}({s['application_id']}→{s['jd_version_id']})"
            for s in item["sources"]
        )
        entry = f"{item['category']}：\n- {item['text']} [覆盖目标: {item['count']}] [来源: {sources}]"
        if len("\n".join(lines)) + len(entry) > max_chars:
            lines.append("（其余目标因上下文上限省略，完整关系仍保存在投递管理中）")
            break
        lines.append(entry)
    decisions = [x for x in gap_items if not x["learnable"]]
    if decisions:
        lines.extend(["", "【仅作投递决策/信息核验，禁止生成学习任务】"])
        for item in decisions:
            lines.append(f"- {item['text']}")
    if not gap_items:
        lines.append("- 当前匹配快照未识别到能力缺口；保留岗位要求作为面试与材料准备依据")
    return "\n\n".join(lines)


def plan_gap_items(targets: list[dict] | None = None) -> list[dict]:
    """Merge equivalent gap text while retaining every application/JD provenance edge."""
    targets = targets if targets is not None else plan_targets()["included"]
    grouped: dict[str, dict] = {}
    for rank, target in enumerate(targets):
        gaps = ((target.get("match") or {}).get("gap_list") or {})
        for category, values in gaps.items():
            for value in values or []:
                text = str(value).strip()
                key = re.sub(r"[\s\d\W_]+", "", text.lower())
                if not key:
                    continue
                item = grouped.setdefault(key, {
                    "category": category, "text": text, "sources": [], "best_rank": rank,
                    "learnable": not (
                        category == "硬门槛缺口"
                        and any(word in text for word in (
                            "学历", "专业", "地域", "地点", "城市", "年龄", "毕业时间",
                            "工作年限", "年经验", "信息不足",
                        ))
                    ),
                })
                item["best_rank"] = min(item["best_rank"], rank)
                item["sources"].append({
                    "application_id": target.get("application_id", ""),
                    "jd_version_id": target.get("jd_version_id", ""),
                    "company": target.get("company", ""),
                    "position": target.get("position", ""),
                    "priority": target.get("plan_priority", "medium"),
                })
    out = []
    category_rank = {"硬门槛缺口": 0, "技能缺口": 1, "经历缺口": 2}
    for item in grouped.values():
        item["count"] = len(item["sources"])
        out.append(item)
    out.sort(key=lambda x: (
        x["best_rank"], -x["count"], category_rank.get(x["category"], 9), x["text"],
    ))
    return out


def target_snapshot() -> dict:
    targets = plan_targets()["included"]
    rows = []
    for target in targets:
        match = target.get("match") or {}
        gap_json = json.dumps(match.get("gap_list") or {}, ensure_ascii=False, sort_keys=True)
        rows.append({
            "application_id": target.get("application_id", ""),
            "company": target.get("company", ""),
            "position": target.get("position", ""),
            "jd_id": target.get("jd_id", ""),
            "jd_version_id": target.get("jd_version_id", ""),
            "match_id": match.get("match_id", ""),
            "gap_hash": hashlib.sha256(gap_json.encode("utf-8")).hexdigest()[:16],
            "source_url": (target.get("jd") or {}).get("source_url", ""),
        })
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return {"snapshot_id": "pts_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
            "generated_at": _now(), "targets": rows}


def render_target_appendix(snapshot: dict | None = None) -> str:
    snapshot = snapshot or target_snapshot()
    rows = snapshot.get("targets") or []
    if not rows:
        return ""
    lines = ["## 🎯 本计划服务的投递 JD（确定性溯源）", "",
             f"> 目标快照：`{snapshot.get('snapshot_id', '')}`。计划不会因投递状态变化自动改写。", ""]
    for row in rows:
        lines.append(
            f"- **{row['company']}｜{row['position']}**："
            f"`{row['application_id']}` → `{row['jd_version_id']}` → `{row['match_id']}`"
            + (f" · 来源：{row['source_url']}" if row.get("source_url") else " · 来源未记录")
        )
    return "\n".join(lines)


def search_bound_jds(targets: list[dict] | None = None, query: str = "",
                     limit_per_jd: int = 3) -> list[dict]:
    """Deterministically resolve application JDs, then rank paragraphs inside each file."""
    if targets is None:
        targets = plan_targets()["included"]
    terms = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{1,}|[\u4e00-\u9fff]{2,8}", query)}
    out = []
    for target in targets:
        jd_id, version_id = target.get("jd_id", ""), target.get("jd_version_id", "")
        snap = load_snapshot(version_id, jd_id=jd_id) if version_id else {}
        if not snap:
            continue
        paragraphs = [p.strip() for p in re.split(
            r"\n\s*\n|(?=^\s*[-*\d]+[.、])", snap["jd_text"], flags=re.MULTILINE
        ) if p.strip()]
        ranked = sorted(paragraphs, key=lambda p: sum(t in p.lower() for t in terms), reverse=True)
        selected = ranked[:limit_per_jd] if terms else paragraphs[:limit_per_jd]
        out.append({
            "application_id": target.get("application_id", ""),
            "company": target.get("company", snap.get("company", "")),
            "position": target.get("position", snap.get("position", "")),
            "jd_version_id": version_id,
            "source_url": snap.get("source_url", ""),
            "captured_at": snap.get("captured_at", ""),
            "text": "\n".join(selected)[:2400],
            "path": snap.get("path", ""),
        })
    return out


def compare_versions(jd_id: str, *, from_version: str = "", to_version: str = "") -> dict:
    versions = list_versions(jd_id)
    if not versions:
        return {"status": "error", "error": "没有可比较的 JD 版本"}
    by_id = {x["jd_version_id"]: x for x in versions}
    if not to_version:
        to_version = versions[0]["jd_version_id"]
    if not from_version and len(versions) > 1:
        from_version = versions[1]["jd_version_id"]
    if not from_version or from_version == to_version:
        return {"status": "ok", "changed": False, "from_version": from_version or to_version,
                "to_version": to_version, "diff": "JD 内容未变化或只有一个版本"}
    if from_version not in by_id or to_version not in by_id:
        return {"status": "error", "error": "指定的 JD 版本不存在"}
    old = load_snapshot(from_version, jd_id=jd_id)
    new = load_snapshot(to_version, jd_id=jd_id)
    diff = "\n".join(difflib.unified_diff(
        old.get("jd_text", "").splitlines(), new.get("jd_text", "").splitlines(),
        fromfile=from_version, tofile=to_version, lineterm="",
    ))
    return {"status": "ok", "changed": bool(diff), "from_version": from_version,
            "to_version": to_version, "diff": diff[:6000]}
