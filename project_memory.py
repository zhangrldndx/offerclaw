# -*- coding: utf-8 -*-
"""个人项目记忆的确定性目录视图。

项目“有哪些”是清单问题，不应依赖向量 Top-K。这里直接读取 Markdown 事实源：

* ``knowledge_base/project_context`` 中人工审批通过的个人项目材料；
* 已形成正式简历项目段的仓库项目（目前为 OfferClaw）。

向量索引仍用于“这个项目如何实现/有哪些技术取舍”等语义问题，但不是项目目录的
事实源。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import re
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_CONTEXT_DIR = BASE_DIR / "knowledge_base" / "project_context"
RESUME_PROJECT_PATH = BASE_DIR / "docs" / "RESUME_PROJECT.md"
OFFERCLAW_OVERVIEW_PATH = BASE_DIR / "docs" / "project_one_pager.md"


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.lstrip().startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line.strip())
        if match:
            meta[match.group(1)] = match.group(2).strip().strip("'\"")
    return meta, parts[2].lstrip()


def _first_summary(body: str) -> str:
    section = re.search(
        r"^##\s+一句话(?:定位|项目简介).*?\n+(.*?)(?=\n##\s|\Z)",
        body,
        re.M | re.S,
    )
    candidate = section.group(1) if section else body
    for paragraph in re.split(r"\n\s*\n", candidate):
        cleaned = re.sub(r"^[>\-*\s]+", "", paragraph.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned and not cleaned.startswith("#") and len(cleaned) >= 12:
            return cleaned[:360]
    return "未记录项目简介"


def _display_name(title: str, fallback: str) -> str:
    value = str(title or "").strip()
    value = re.sub(r"\s*[·（(].*$", "", value).strip()
    value = re.sub(r"\s+项目现状$", "", value).strip()
    return value or fallback


def _iso_mtime(path: Path) -> str:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    except OSError:
        return ""


def _approved_project_context(root: Path) -> list[dict[str, Any]]:
    directory = root / "knowledge_base" / "project_context"
    items: list[dict[str, Any]] = []
    if not directory.exists():
        return items
    for path in sorted(directory.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _frontmatter(raw)
        if meta.get("review_status", "").lower() != "approved":
            continue
        if meta.get("source_type", "project_context") != "project_context":
            continue
        source_url = meta.get("source_url", "")
        name = _display_name(meta.get("title", ""), path.stem)
        items.append({
            "project_id": meta.get("project_id") or f"project_{path.stem.lower()}",
            "name": name,
            "title": meta.get("title") or name,
            "summary": _first_summary(body),
            "source_url": source_url,
            "source_path": str(path.relative_to(root)),
            "review_status": "approved",
            "approval_label": "已审批为个人项目知识",
            "resume_usage": meta.get("resume_usage", "not_recorded"),
            "resume_usage_label": "简历使用情况未记录",
            "open_source": bool(re.search(r"github\.com|gitlab\.com|gitee\.com", source_url, re.I)),
            "updated_at": meta.get("crawl_date") or _iso_mtime(path),
        })
    return items


def _offerclaw_resume_project(root: Path) -> dict[str, Any] | None:
    resume_path = root / "docs" / "RESUME_PROJECT.md"
    overview_path = root / "docs" / "project_one_pager.md"
    if not resume_path.exists():
        return None
    try:
        resume_text = resume_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "OfferClaw" not in resume_text or "项目经历" not in resume_text:
        return None
    overview = ""
    if overview_path.exists():
        try:
            overview = overview_path.read_text(encoding="utf-8")
        except OSError:
            pass
    url_match = re.search(r"https://github\.com/[\w.-]+/[\w.-]+", overview)
    return {
        "project_id": "project_offerclaw",
        "name": "OfferClaw",
        "title": "OfferClaw · 长期养成型求职 AI Agent",
        "summary": _first_summary(overview or resume_text),
        "source_url": url_match.group(0) if url_match else "",
        "source_path": str(resume_path.relative_to(root)),
        "review_status": "resume_confirmed",
        "approval_label": "已形成正式简历项目段",
        "resume_usage": "confirmed",
        "resume_usage_label": "已有可直接使用的简历项目段",
        "open_source": bool(url_match),
        "updated_at": _iso_mtime(resume_path),
    }


def list_project_catalog(*, root: Path | str | None = None,
                         open_source_only: bool = False,
                         approved_only: bool = False,
                         resume_confirmed_only: bool = False) -> list[dict[str, Any]]:
    """返回可审计的个人项目目录；同名项目按更强确认状态去重。"""
    base = Path(root) if root is not None else BASE_DIR
    items = _approved_project_context(base)
    resume_project = _offerclaw_resume_project(base)
    if resume_project:
        items.append(resume_project)

    rank = {"approved": 2, "resume_confirmed": 3}
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        key = str(item.get("name", "")).strip().casefold()
        previous = deduped.get(key)
        if not previous or rank.get(item.get("review_status", ""), 0) > rank.get(
                previous.get("review_status", ""), 0):
            deduped[key] = item
    result = list(deduped.values())
    if open_source_only:
        result = [item for item in result if item.get("open_source")]
    if approved_only:
        result = [item for item in result if item.get("review_status") == "approved"]
    if resume_confirmed_only:
        result = [item for item in result if item.get("resume_usage") == "confirmed"]
    return sorted(result, key=lambda item: (
        item.get("review_status") != "resume_confirmed",
        str(item.get("name", "")).casefold(),
    ))


def render_project_record(item: dict[str, Any]) -> str:
    url = item.get("source_url") or "未记录"
    return (
        f"项目：{item.get('name') or '未命名项目'}\n"
        f"确认状态：{item.get('approval_label') or '未记录'}\n"
        f"简历用途：{item.get('resume_usage_label') or '未记录'}\n"
        f"是否开源：{'是' if item.get('open_source') else '未确认'}\n"
        f"来源链接：{url}\n"
        f"项目说明：{item.get('summary') or '未记录'}"
    )


def render_project_evidence(item: dict[str, Any], *, root: Path | str | None = None,
                            max_chars: int = 3200) -> str:
    """为项目适配比较读取已确认事实源正文；路径必须仍位于项目根目录内。"""
    base = (Path(root) if root is not None else BASE_DIR).resolve()
    relative = str(item.get("source_path") or "").strip()
    detail = ""
    if relative:
        try:
            path = (base / relative).resolve()
            path.relative_to(base)
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                raw = path.read_text(encoding="utf-8")
                meta, body = _frontmatter(raw)
                if not meta or meta.get("review_status", "approved").lower() == "approved":
                    detail = body.strip()[:max_chars]
        except (OSError, ValueError):
            detail = ""
    catalog = render_project_record(item)
    return (catalog + ("\n\n已确认项目正文摘录：\n" + detail if detail else ""))[:max_chars + 900]
