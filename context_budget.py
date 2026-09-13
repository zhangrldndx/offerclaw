# -*- coding: utf-8 -*-
"""context_budget.py — [L2] 上下文预算与压缩（loop engineering 复用件）

长程 agent 每次唤醒注入 LLM 的上下文会随时间膨胀（daily_log / profile / applications）。
本模块提供两件**确定性、可单测**的工具，避免「全量 dump → 爆窗 / 成本飙升 / 噪音」：

1. ``fit_to_budget(blocks, max_chars)``：多块按优先级装进预算——超预算时从**最低优先级**开始
   截到 ``min_keep``、仍超则丢弃，高优先级尽量整保。返回保留结果 + 是否超预算 + 警告。
2. ``select_relevant_sections(md, keywords, max_chars, always_keep)``：按 markdown 章节切，
   命中关键词（或 always_keep 章节）优先保留、填到预算为止——把「全量 profile」换成「相关切片」。
   这正是把 RAG 的「相关性优先」思路用到上下文压缩上（一鱼两吃）。

口径：用**字符数**做预算闸（中文场景 token≈字符量级，足够；要更准可换 tiktoken）。
保证：只要 ``sum(min_keep) <= max_chars``，``fit_to_budget`` 的 ``total_after`` 必 ``<= max_chars``。
"""
from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """粗估 token 数（中文保守按 1 字≈1 token 量级；宁可早降级也不爆窗）。"""
    return len(text or "")


def fit_to_budget(blocks: list[dict], max_chars: int) -> dict:
    """把多个带优先级的块装进 ``max_chars`` 预算。

    ``blocks``: ``[{name, priority(大=重要), text, min_keep(默认0)}]``。
    返回 ``{ok, total_before, total_after, kept:[{name,chars,action,text}], warnings}``。
    降级策略：超预算时按优先级升序（低的先动），逐块截到 ``min_keep``、仍超则丢弃。
    """
    items = [dict(b) for b in blocks]
    for b in items:
        b["_text"] = b.get("text", "") or ""
    total_before = sum(len(b["_text"]) for b in items)
    actions = {i: "keep" for i in range(len(items))}
    warnings: list[str] = []

    if total_before > max_chars:
        # 低优先级先动；同优先级里先动更长的
        order = sorted(range(len(items)),
                       key=lambda i: (items[i].get("priority", 0), -len(items[i]["_text"])))
        for i in order:
            total = sum(len(b["_text"]) for b in items)
            if total <= max_chars:
                break
            over = total - max_chars
            min_keep = int(items[i].get("min_keep", 0) or 0)
            cur = len(items[i]["_text"])
            target = max(min_keep, cur - over)
            if target < cur:
                items[i]["_text"] = items[i]["_text"][:target]
                actions[i] = "truncate" if target > 0 else "drop"
                warnings.append(f"{items[i]['name']} {cur}→{target}")

    total_after = sum(len(b["_text"]) for b in items)
    kept = [{"name": items[i]["name"], "chars": len(items[i]["_text"]),
             "action": actions[i], "text": items[i]["_text"]} for i in range(len(items))]
    return {"ok": total_after <= max_chars, "total_before": total_before,
            "total_after": total_after, "kept": kept, "warnings": warnings}


def _split_sections(md: str) -> list[str]:
    """按 markdown 标题（#/##/###）切块，每块含其标题行。无标题则整体一块。"""
    lines = (md or "").splitlines(keepends=True)
    sections: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if re.match(r"^#{1,3}\s", ln) and cur:
            sections.append("".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append("".join(cur))
    return sections


def keywords_from(text: str, top: int = 24) -> list[str]:
    """从一段文本（如 JD / 缺口清单）粗提关键词：英文 token + 中文 jieba 分词，去重保序。

    中文用 jieba（与 rag_bm25 一致）切出「大模型 / 应用」这类词，而非贪婪整块——
    否则相关性打分时 ``sec.count(keyword)`` 几乎匹配不上。jieba 不可用则退回 CJK 连续块。
    """
    text = text or ""
    seen, out = set(), []

    def _add(t: str):
        k = t.lower() if t and t[0].isascii() else t
        if k and k not in seen:
            seen.add(k)
            out.append(t)

    for t in re.findall(r"[A-Za-z][A-Za-z0-9+.#]{1,}", text):   # 英文/技术词
        _add(t)
    try:
        import jieba
        for t in jieba.cut_for_search(text):
            t = t.strip()
            if len(t) >= 2 and re.fullmatch(r"[一-鿿]+", t):
                _add(t)
    except Exception:
        for t in re.findall(r"[一-鿿]{2,}", text):
            _add(t)
    return out[:top]


def select_relevant_sections(markdown: str, keywords: list[str], max_chars: int,
                             always_keep: tuple = ()) -> str:
    """按相关性挑 markdown 章节，填到 ``max_chars`` 为止（已在预算内则原样返回）。

    打分：``always_keep`` 命中标题 +100（始终优先保留，如「基础信息/方向」），
    其余按关键词在章节内出现次数累加。挑中的章节**还原原文顺序**输出。
    """
    md = markdown or ""
    if len(md) <= max_chars:
        return md
    secs = _split_sections(md)
    kws = [k for k in (keywords or []) if k]

    def score(sec: str) -> int:
        head = sec.splitlines()[0] if sec else ""
        s = sum(100 for k in always_keep if k and k in head)
        s += sum(sec.count(k) for k in kws)
        return s

    ranked = sorted(range(len(secs)), key=lambda i: (-score(secs[i]), i))
    chosen: list[tuple] = []
    total = 0
    for i in ranked:
        sec = secs[i]
        if total + len(sec) <= max_chars:
            chosen.append((i, sec))
            total += len(sec)
    if not chosen:                       # 单章节就超预算 → 硬截最相关那块
        i0 = ranked[0]
        return secs[i0][:max_chars]
    chosen.sort(key=lambda t: t[0])      # 还原原文顺序
    return "".join(sec for _i, sec in chosen)
