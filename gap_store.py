# -*- coding: utf-8 -*-
"""gap_store.py — 目标 JD 与缺口信息库（长期累积，计划生成的"第三件套"）。

设计动机（用户流程定义）：
- 画像：用户首次上传加载；JD：用户随时上传，OfferClaw 判断简历是否符合并产出缺口；
- 学习计划基于三部分：**用户画像 + 知识库(RAG) + 缺口信息库**。
- 「基于缺口生成计划」= 把该 JD 设为目标：JD 摘要 + 对应缺口**持久入库**，
  此后每次（重新）生成计划都以累积的缺口库为背景；再上传新 JD、缺口有差异
  则继续合并维护——OfferClaw 是长期养成式 agent，要积累/储存/分析用户状态。

存储：gap_store.json（与 daily_log.md 同级的用户状态文件）。
去重：缺口文本归一化（去空白/标点取前 12 字）作 key，跨 JD 合并不重复。
"""

from __future__ import annotations

import datetime
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "gap_store.json")

# 缺口分类的展示顺序（与 match_job.gap_list / 前端缺口卡一致）
_CATEGORY_ORDER = ["硬门槛缺口", "技能缺口", "经历缺口", "其他缺口"]


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _norm_key(text: str) -> str:
    """缺口文本归一化（去空白/标点/数字），用于跨 JD 去重。不截断，配合前缀包含判重。"""
    return re.sub(r"[\s\d\.、:：（）()\[\]【】\-—,，。;；]+", "", str(text))


def _is_dup(norm: str, seen: list[str]) -> bool:
    """判重：完全相同，或一方是另一方的前缀（如「…实战」vs「…实战经验」）。
    前缀合并要求较短一方 ≥6 字，避免过短文本误合并。"""
    for s in seen:
        if norm == s:
            return True
        shorter, longer = (norm, s) if len(norm) <= len(s) else (s, norm)
        if len(shorter) >= 6 and longer.startswith(shorter):
            return True
    return False


def _load() -> dict:
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {"targets": [], "updated_at": _now_iso()}


def _save(data: dict) -> None:
    data["updated_at"] = _now_iso()
    from io_utils import atomic_write_json
    atomic_write_json(STORE_PATH, data)  # B1: 原子写防半截文件（缺口库是求职核心数据）


def _locked(fn):
    """[B2] 给 read-modify-write 函数包 STORE_PATH 文件锁，串行化 CLI/Web/cron 并发写，防 lost-update。"""
    import functools

    @functools.wraps(fn)
    def _w(*a, **k):
        from io_utils import file_lock
        with file_lock(STORE_PATH):
            return fn(*a, **k)

    return _w


def _extract_title_company(jd_text: str) -> tuple[str, str]:
    """从 JD 原文抽取岗位名/公司（规则与 job_discovery 一致，失败给空）。"""
    title = company = ""
    m = re.search(r"(?:岗位名称|职位名称|职位|岗位)[：: ]+([^\n，,；;]+)", jd_text)
    if m:
        title = m.group(1).strip()
    m = re.search(r"(?:公司名称|公司|招聘公司|Company)[：: ]+([^\n，,；;]+)", jd_text)
    if m:
        company = m.group(1).strip()
    return title, company


def _target_identity(title: str, company: str, jd_text: str) -> str:
    """目标 JD 的身份指纹：标题+公司归一；无有效标题时退回 JD 内容指纹。
    同一身份的 JD 重复添加 → 合并而非新增（累加原则：重复的不管，新的才记）。"""
    import hashlib
    t = _norm_key(title)
    c = _norm_key(company)
    if t and t != _norm_key("未命名岗位"):
        return f"tc:{t}|{c}"
    norm_jd = re.sub(r"\s+", "", jd_text or "")[:400]
    return "jd:" + hashlib.md5(norm_jd.encode("utf-8")).hexdigest()[:16]


@_locked  # B2: 缺口累积是 read-modify-write，加文件锁防并发 lost-update
def add_target(jd_text: str, gaps: dict | None, *,
               title: str = "", company: str = "", source: str = "web_match") -> dict:
    """把一个 JD 设为目标：JD 摘要 + 缺口入库。

    **累加原则**：缺口跨 JD 归一化去重（重复不记）；目标按身份指纹去重——
    同一 JD 再次添加只把它的新缺口合并进原条目（action=merged），不会多出目标。
    ``gaps``：match 产出的 dict（{分类: [条目...]}）。
    返回 {status, action(added/merged), target_id, added_gaps, duplicate_gaps,
          total_targets, merged_gap_count}。
    """
    jd_text = (jd_text or "").strip()
    gaps = gaps or {}
    if not jd_text and not any(v for v in gaps.values()):
        return {"status": "error", "error": "jd_text 与 gaps 不能都为空"}

    if not title or not company:
        t2, c2 = _extract_title_company(jd_text)
        title = title or t2 or "未命名岗位"
        company = company or c2 or ""

    data = _load()
    # 已有缺口归一化文本列表（全库），新条目据此判重（含前缀包含合并）
    seen: list[str] = []
    for tgt in data["targets"]:
        for items in (tgt.get("gaps") or {}).values():
            for it in items:
                seen.append(_norm_key(it))

    new_gaps: dict[str, list[str]] = {}
    added, dup = 0, 0
    for cat, items in gaps.items():
        for it in items or []:
            k = _norm_key(it)
            if not k:
                continue
            if _is_dup(k, seen):
                dup += 1
                continue
            seen.append(k)
            new_gaps.setdefault(cat, []).append(str(it).strip())
            added += 1

    # 目标级去重：同一身份的 JD → 合并进已有条目，不新增
    identity = _target_identity(title, company, jd_text)
    existing = next(
        (t for t in data["targets"]
         if t.get("identity") == identity
         or (not t.get("identity")
             and _target_identity(t.get("title", ""), t.get("company", ""),
                                  t.get("jd_digest", "")) == identity)),
        None,
    )
    if existing is not None:
        for cat, items in new_gaps.items():
            existing.setdefault("gaps", {}).setdefault(cat, []).extend(items)
        existing["identity"] = identity
        existing["updated_at"] = _now_iso()
        existing["raw_gap_count"] = sum(len(v or []) for v in gaps.values())
        _save(data)
        return {
            "status": "ok", "action": "merged",
            "target_id": existing["id"], "title": existing.get("title", title),
            "company": existing.get("company", company),
            "added_gaps": added, "duplicate_gaps": dup,
            "total_targets": len(data["targets"]),
            "merged_gap_count": len(seen),
        }

    tid = f"tgt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data["targets"].append({
        "id": tid,
        "identity": identity,
        "added_at": _now_iso(),
        "title": title,
        "company": company,
        "jd_digest": jd_text[:400],
        "gaps": new_gaps,           # 只存本目标贡献的"新"缺口，合并视图由 merged_gaps 计算
        "raw_gap_count": sum(len(v or []) for v in gaps.values()),
        "source": source,
    })
    _save(data)
    return {
        "status": "ok",
        "action": "added",
        "target_id": tid,
        "title": title,
        "company": company,
        "added_gaps": added,
        "duplicate_gaps": dup,
        "total_targets": len(data["targets"]),
        "merged_gap_count": len(seen),
    }


def list_targets() -> list[dict]:
    return _load()["targets"]


def merged_gaps() -> dict[str, list[str]]:
    """全库合并后的缺口（按分类聚合，保持先来先得的去重结果）。"""
    out: dict[str, list[str]] = {}
    for tgt in _load()["targets"]:
        for cat, items in (tgt.get("gaps") or {}).items():
            out.setdefault(cat, []).extend(items or [])
    return out


def merged_gaps_text(max_chars: int = 4000) -> str:
    """把缺口库导出成 plan_gen 可解析的文本（分类标题 + 条目行）。空库返回 ""。

    ``max_chars``：注入上限（用户长期累积大量目标 JD 时防上下文膨胀）——
    超限时截断并注明省略条数；完整缺口永久保存在 gap_store.json。
    """
    merged = merged_gaps()
    if not any(merged.values()):
        return ""
    lines: list[str] = []
    cats = [c for c in _CATEGORY_ORDER if merged.get(c)] + \
           [c for c in merged if c not in _CATEGORY_ORDER and merged.get(c)]
    total = omitted = 0
    for cat in cats:
        lines.append(f"{cat}：")
        for it in merged[cat]:
            entry = f"- {it}"
            if total + len(entry) > max_chars:
                omitted += 1
                continue
            lines.append(entry)
            total += len(entry)
    if omitted:
        lines.append(f"（另有 {omitted} 条缺口因注入上限省略——完整清单见 gap_store.json）")
    return "\n".join(lines)


def summary() -> dict:
    """给前端/推送的概览：目标数、缺口数、最近目标。"""
    data = _load()
    merged = merged_gaps()
    total = sum(len(v) for v in merged.values())
    last = data["targets"][-1] if data["targets"] else None
    return {
        "total_targets": len(data["targets"]),
        "merged_gap_count": total,
        "by_category": {k: len(v) for k, v in merged.items() if v},
        "latest_target": ({"title": last["title"], "company": last["company"],
                           "added_at": last["added_at"]} if last else None),
        "updated_at": data.get("updated_at", ""),
    }
