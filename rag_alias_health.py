# -*- coding: utf-8 -*-
"""rag_alias_health.py — 别名资产的治理与体检(指导文档 §9 Alias 生产链产品化)。

Concept Alias 已从"实验脚本产物"变成**核心检索资产**,必须像索引一样被治理:
知道每条别名从哪来、是否还有效、失效了要不要重建、失败了会不会静默拖垮检索。

本模块只做**校验与体检**,不改检索行为(查询期的回退语义见 §9.7:
别名缺失/失败/过期 → 只用原文分,绝不阻塞原文检索)。

三类检查:
  ① 生成校验(§9.3):字段齐全、非空、是中文、无原文篡改、**无评测集泄漏**;
  ② 失效判定(§9.6):原文 content_hash / prompt_hash / 模型 / 版本 任一变化 → stale;
  ③ 覆盖体检:多少块有可用别名,缺口在哪(供决定是否重跑生成)。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# 状态机(指导 §9.2)。cache 里只落 succeeded;其余状态由体检推导,避免脏数据长期驻留。
STATUS = ("pending", "running", "succeeded", "partial", "failed", "stale")

_CJK = re.compile(r"[一-鿿]")


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def prompt_hash() -> str:
    """提示词指纹:提示词改了,旧别名的可比性就没了(§9.6 失效条件之一)。"""
    from rag_alias import _GEN_PROMPT
    return hashlib.sha256(_GEN_PROMPT.encode("utf-8")).hexdigest()[:12]


def validate(record: dict, source_text: str | None = None,
             leak_terms: list | None = None) -> tuple:
    """校验一条别名记录(§9.3)。返回 (status, problems)。

    ``leak_terms`` 给出时检查评测集泄漏(§10.4):别名里不得出现最终评测题的原文——
    否则等于把答案写进了检索索引,评测结果全部失效。
    """
    problems = []
    kw = record.get("zh_keywords") or []
    qs = record.get("zh_candidate_questions") or []
    if not kw:
        problems.append("zh_keywords 为空")
    if not qs:
        problems.append("zh_candidate_questions 为空")
    blob = " ".join([*kw, *qs, str(record.get("zh_summary") or "")])
    if blob and not _CJK.search(blob):
        problems.append("别名内容不含中文(生成疑似失败或语言错误)")
    if source_text is not None:
        h = record.get("source_content_hash")
        if h and h != content_hash(source_text):
            problems.append("原文已变更(content_hash 不符)→ stale")
    if record.get("prompt_hash") and record["prompt_hash"] != prompt_hash():
        problems.append("提示词已变更(prompt_hash 不符)→ stale")
    from rag_alias import ALIAS_VERSION
    if record.get("alias_version") != ALIAS_VERSION:
        problems.append(f"别名版本不符({record.get('alias_version')} != {ALIAS_VERSION})→ stale")
    for t in (leak_terms or []):
        t = (t or "").strip()
        if len(t) >= 8 and t in blob:
            problems.append(f"疑似评测集泄漏:别名内含题面片段 {t[:24]!r}")
    if not problems:
        return "succeeded", []
    if any("stale" in p for p in problems):
        return "stale", problems
    if kw or qs:
        return "partial", problems
    return "failed", problems


def health(collection: str | None = None, leak_sets: list | None = None) -> dict:
    """全量体检:覆盖率 + 各状态计数 + 问题样例。不修改任何东西。"""
    import chromadb
    from rag_alias import CACHE_PATH, alias_fields
    from rag_quota import quota_collection

    name = collection or quota_collection()
    try:
        col = chromadb.PersistentClient(
            path=os.path.join(BASE, "chroma_db")).get_collection(name)
        got = col.get(include=["documents"])
        texts = dict(zip(got["ids"], got["documents"]))
    except Exception as e:
        return {"ok": False, "detail": f"英文分区不可读:{e}", "collection": name}

    leak_terms = []
    for p in (leak_sets or []):
        try:
            with open(p, encoding="utf-8") as f:
                leak_terms += [it["q"] for it in json.load(f).get("items", [])]
        except Exception:
            continue

    rows, counts, samples = {}, {s: 0 for s in STATUS}, []
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                cid = r.get("chunk_id")
                if cid:
                    rows[cid] = r          # 同 chunk 多次生成取最后一条
    for cid, r in rows.items():
        st, probs = validate(r, texts.get(cid), leak_terms)
        counts[st] += 1
        if probs and len(samples) < 5:
            samples.append({"chunk_id": cid, "status": st, "problems": probs[:3]})

    usable = counts["succeeded"]
    missing = [i for i in texts if i not in rows]
    return {
        "ok": bool(usable) and not missing and counts["stale"] == 0,
        "collection": name, "chunks": len(texts),
        "alias_records": len(rows), "usable": usable,
        "counts": counts, "missing_chunks": len(missing),
        "coverage": round(usable / max(len(texts), 1), 4),
        "fields_in_use": list(alias_fields()),
        "prompt_hash": prompt_hash(),
        "problem_samples": samples,
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="别名资产体检(§9)")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--leak-set", action="append", default=None,
                    help="评测集路径,检查别名是否泄漏题面(可多次)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    h = health(a.collection, a.leak_set)
    print(json.dumps(h, ensure_ascii=False, indent=2))
    return 0 if h.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
