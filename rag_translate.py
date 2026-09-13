# -*- coding: utf-8 -*-
"""rag_translate.py — 查询翻译通道(方案报告 §13.2)。

**为什么是这个杠杆**(2026-08-18 实证链条,见 docs/rag_eval/quota/REPORT.md):
配额把候选池修好了(池召回 100%、零英文候选 0/60),但论文向仍只有 60%——
逐层消融把损伤定位在**精排层**:纯 dense 配额 R@3 已 84%,精排把它压到 66%。
根因是跨语精排惩罚:cross-encoder 对"中文 query ↔ 英文 passage"打分天然偏低。
同一把尺子上,英文问英文子域实测 **100%**——所以让 base 精排在"英问英"条件下工作,
就是把已知能跑满分的条件创造出来,而不是换一个更大的模型
(bge-reranker-v2-m3 质量达标但 p95 3.5s→44.8s,超预算两个数量级)。

**契约**(报告 §13.2 的结构化输出):
    {"english_query": str, "keywords": [str], "preserve_terms": [str]}
`preserve_terms` = query 里本来就是英文的专有名词(MemGPT/ReAct...),必须原样保留。

**缓存是硬要求**(报告 §13.2 明确要求 normalized_query_hash → translation_result):
翻译在查询路径上,不缓存等于每次问答多付一次 LLM 往返。本模块用**磁盘 + 进程内**
双层缓存;评测要跑几百次同一批 query,没有缓存实验不可复现也不可承受。
LLM 不可用/返回不合契约 → 返回 None,调用方**退回原中文 query**(fail-soft,绝不阻断)。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE, "logs", "query_translation_cache.json")

_MEM: dict = {}
_LOADED = False
_LOCK = threading.Lock()

_CJK = re.compile(r"[一-鿿]")
_EN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


def translate_enabled() -> bool:
    """默认关——测正才采纳(同 HyDE/doc2query/CRAG/paper_route/quota 的家法)。"""
    return os.environ.get("RAG_QUERY_TRANSLATE", "0") == "1"


def needs_translation(question: str) -> bool:
    """只翻含中文的 query。纯英文 query 本来就在"英问英"条件下,翻译纯属浪费一次调用。"""
    return bool(_CJK.search(question or ""))


def _norm(question: str) -> str:
    """归一化后再做 hash:大小写、首尾空白、内部连续空白不应产生缓存未命中。"""
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _key(question: str) -> str:
    return hashlib.sha256(_norm(question).encode("utf-8")).hexdigest()[:16]


def _load() -> dict:
    global _LOADED
    if _LOADED:
        return _MEM
    with _LOCK:
        if _LOADED:
            return _MEM
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _MEM.update(json.load(f))
        except Exception:
            pass                      # 首次运行/文件损坏 → 空缓存起步,不报错
        _LOADED = True
    return _MEM


def _persist() -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_MEM, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CACHE_PATH)   # 原子替换:并发/中断不会留下半个文件
    except Exception:
        pass


_PROMPT = (
    "You translate Chinese search queries into English for retrieving English "
    "technical papers. Output STRICT JSON only, no prose, no code fence:\n"
    '{"english_query": "...", "keywords": ["...", "..."], "preserve_terms": ["..."]}\n'
    "Rules: english_query is a faithful, retrieval-oriented English rendering "
    "(keep it a query, not an answer). keywords: 3-6 English technical terms the "
    "target passage would contain. preserve_terms: proper nouns already in Latin "
    "script in the input (model/system/paper names), verbatim; [] if none."
)


def _parse(raw: str) -> dict | None:
    """防御式解析:模型可能裹 ```json 围栏或前后加话。抓第一个平衡的 JSON 对象。"""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    for j, ch in enumerate(s[i:], start=i):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            s = s[i:j + 1]
            break
    else:
        return None
    try:
        d = json.loads(s)
    except Exception:
        return None
    eq = (d.get("english_query") or "").strip()
    if not eq:
        return None                   # 契约核心字段缺失 = 不可用,宁退回原 query
    def _strs(v):
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
    return {"english_query": eq, "keywords": _strs(d.get("keywords")),
            "preserve_terms": _strs(d.get("preserve_terms"))}


def translate_query(question: str) -> dict | None:
    """中文 query → 契约 dict。缓存命中零成本;LLM 不可用或不合契约返回 None。"""
    if not question or not needs_translation(question):
        return None
    _load()
    k = _key(question)
    if k in _MEM:
        return _MEM[k] or None        # 缓存里的 None(曾翻译失败)也算命中,不反复重试
    try:
        from rag_gate import _chat
        # max_tokens 必须给足 + 关思考(2026-08-18 实测):兜底模型 deepseek-v4-pro 是
        # 推理模型,220 tokens 会被思考过程全部吃掉,content 返回**空字符串**——
        # 现象是"翻译 100% 失败"却没有任何报错。短结构化输出任务的思考纯烧钱。
        raw = _chat([{"role": "system", "content": _PROMPT},
                     {"role": "user", "content": question}],
                    max_tokens=800, temperature=0.0,
                    extra_payload={"enable_thinking": False})
    except Exception:
        raw = None
    got = _parse(raw or "")
    if got is not None:
        # 兜底补齐 preserve_terms:query 里本来的英文 token 必须出现在英文 query 里,
        # 否则"MemGPT 的分页机制"被译成 "paging mechanism" 会丢掉最强的词法信号。
        latin = [t for t in _EN_TOKEN.findall(question) if len(t) > 2]
        for t in latin:
            if t.lower() not in got["english_query"].lower():
                got["english_query"] += f" {t}"
            if t not in got["preserve_terms"]:
                got["preserve_terms"].append(t)
    _MEM[k] = got
    _persist()
    return got


def english_query_for(question: str) -> str | None:
    """给精排/词法通道用的英文 query 串(含 keywords,提高词法覆盖)。不可用返回 None。"""
    got = translate_query(question)
    if not got:
        return None
    parts = [got["english_query"]]
    extra = [w for w in got["keywords"] if w.lower() not in got["english_query"].lower()]
    if extra:
        parts.append(" ".join(extra))
    return " ".join(parts).strip() or None


def cache_stats() -> dict:
    _load()
    return {"entries": len(_MEM),
            "failed": sum(1 for v in _MEM.values() if not v),
            "path": CACHE_PATH}
