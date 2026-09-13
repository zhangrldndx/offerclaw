# -*- coding: utf-8 -*-
"""rag_alias.py — 英文文档的中文概念别名(指导文档 §16/§17 Concept Alias)。

**要解决的缺口**:`xling_no_term` 子域(纯中文概念提问)卡在 45%,而
  · 翻译通道:45% → 45%(纹丝不动);
  · 多语精排 v2-m3:45% → 65%。
说明缺的不是语言表层转换,而是**中文概念表达 ↔ 英文学术术语/方法名**之间的映射。
例:"系统如何管理超出上下文窗口的长期记忆" 对应的英文是
`virtual context management` / `paging` / `memory hierarchy`——字面直译永远译不出这些词。

**一处与指导文档不同的实现判断(基于本项目实测)**:
指导 §16 写"这些 Alias 只用于 BM25 / Retrieval Alias"。但本项目消融实测
(docs/rag_eval/quota/REPORT.md §4)显示**候选池召回已是 100%**——目标块永远在池子里,
瓶颈在排序层而非召回层。因此纯 BM25 用法对本项目是**空操作**。
故别名的主通道设为**精排桥**:让 cross-encoder 打 (中文 query, 中文别名) 这一对,
彻底绕开跨语惩罚——与既有的语体桥/翻译桥同一机制(rerank 的 synth_map),
桥的对象从"语体"、"语言"换成"概念表达"。BM25 通道保留为可选旁路。

**证据边界(必须守住)**:别名只参与**检索排序**,永不作为回答证据、永不进引文。
最终喂给 LLM 的仍是英文原文块。别名带生成模型与版本,便于失效重建。
"""
from __future__ import annotations

import json
import os
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "logs", "concept_alias_cache.jsonl")
# 别名规格版本;改提示词/字段/生成模型时 +1。
# 2026-08-20:v1 的 194 条实际由 deepseek 兜底生成(生成期主模型全程 503),
# 而缓存按配置名误记为 gpt-5.6 —— 血缘记录已修为回填**实际服务模型**,并整体重生成。
ALIAS_VERSION = "2026-08-20-gpt"

# 指导 §17 的实验矩阵:A1 keywords / A2 summary / A3 questions / A4 keywords+questions
_ALL_FIELDS = ("zh_concepts", "zh_keywords", "zh_summary", "zh_candidate_questions")
_SHORT = {"concepts": "zh_concepts", "keywords": "zh_keywords",
          "summary": "zh_summary", "questions": "zh_candidate_questions"}


def alias_enabled() -> bool:
    """默认关——测正才采纳(同 HyDE/doc2query/CRAG/quota/translate 的家法)。"""
    return os.environ.get("RAG_CONCEPT_ALIAS", "0") == "1"


def _fnum(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def alias_discount() -> float:
    """δ:别名分参与竞争前要付的固定折扣(指导 §5.1 D1)。

    动机:无条件 `max(原分, 别名分)` 让别名分**零成本**参与,轻微虚高就能让英文候选
    抢走中文题的 Top1(held-out 回退 2~3 题)。δ 是连续可调的,粒度远细于
    "Top1 换不换"那种名次拦截——实测那种拦截是坏交易(8 道论文题换 1 道中文题)。
    """
    return max(0.0, _fnum("RAG_ALIAS_DISCOUNT", 0.0))


def alias_lift_gate() -> float:
    """τ:别名分相对原分的提升低于 τ 就不采信(指导 §5.2 D2)。

    只有当审计显示"真英文命中的 lift 明显高于 False Winner 的 lift"时才该用;
    两类分布高度重叠则本方向无效,不应继续调参(指导明确写了这条停止规则)。
    """
    return max(0.0, _fnum("RAG_ALIAS_LIFT_GATE", 0.0))


def alias_scope() -> int:
    """N:只给英文配额通道**前 N 个**候选算别名分(指导 §6);<=0 或未设 = 不限。

    动机有二:① 减少无关英文候选参与竞争(降 False Winner);② 减少交叉编码对数(降 p95)。
    """
    try:
        return int(os.environ.get("RAG_ALIAS_SCOPE", "0") or 0)
    except ValueError:
        return 0


def alias_fields() -> tuple:
    """本次启用的别名字段(env `RAG_ALIAS_FIELDS`,逗号分隔;默认全用)。"""
    raw = (os.environ.get("RAG_ALIAS_FIELDS", "") or "").strip()
    if not raw:
        return _ALL_FIELDS
    out = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        f = _SHORT.get(tok, tok if tok in _ALL_FIELDS else None)
        if f and f not in out:
            out.append(f)
    return tuple(out) or _ALL_FIELDS


_GEN_PROMPT = """你在为一个中文知识库做**检索别名**。下面是一段英文技术资料。
请用中文写出它的检索线索,使得**用中文提问的人**能检索到这段英文原文。

严格输出 JSON,不要任何解释或代码块标记:
{{"zh_concepts": ["..."], "zh_keywords": ["..."], "zh_summary": "...",
 "zh_candidate_questions": ["..."]}}

要求:
- zh_concepts:3-6 个**中文概念名**,要写读者会用的说法,而不是英文术语的死译。
  例:paging/virtual context 这类,应写成"上下文分页""虚拟上下文管理""记忆分层"。
- zh_keywords:5-10 个中文检索关键词(可含必要的英文专有名词如 MemGPT/ReAct)。
- zh_summary:1-2 句中文摘要,说清这段讲了什么。
- zh_candidate_questions:3 个中文提问,要像真人口语提问,**不要出现英文术语**。

英文资料:
{text}"""


def _parse(raw: str) -> dict:
    """从 LLM 输出里抠 JSON。失败返回 {}(该块跳过,重跑自动补)。"""
    import re
    if not isinstance(raw, str):
        return {}
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(d, dict):
        return {}
    out = {}
    for f in _ALL_FIELDS:
        v = d.get(f)
        if isinstance(v, str):
            v = [v] if f != "zh_summary" else v
        if isinstance(v, list):
            v = [str(x).strip() for x in v if str(x).strip()]
        if v:
            out[f] = v
    return out


def generate_alias(text: str, meta: dict | None = None) -> dict:
    """对一个英文 chunk 调一次 LLM 生成中文别名。走 rag_gate._chat(统一网关/计量/兜底)。

    ``max_tokens`` 给到 3000 而非 doc2query 那样的 300(实测踩坑):当前兜底是**推理模型**
    (deepseek-v4-pro),reasoning token 与 content 共用 max_tokens 预算。给 600 时
    实测 `finish_reason=length`、reasoning 1104 字而 content **空串**——
    表现为"调用成功但产出为 0",最难查的那种失败。不传 enable_thinking:那是 qwen3 系的旋钮。
    """
    from rag_gate import _chat
    try:
        budget = int(os.environ.get("RAG_ALIAS_MAX_TOKENS", "3000"))
    except ValueError:
        budget = 3000
    raw = _chat([{"role": "user", "content": _GEN_PROMPT.format(text=text[:1800])}],
                max_tokens=budget, temperature=0.3, meta=meta)
    return _parse(raw)


def _done_ids() -> set:
    done = set()
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    if r.get("alias_version") == ALIAS_VERSION:
                        done.add(r["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    continue          # 半截行跳过(中断重跑时该块重新生成)
    return done


def build_cache(collection: str | None = None, limit: int | None = None,
                workers: int | None = None) -> dict:
    """遍历英文分区,为每个未处理的 chunk 生成中文别名(断点续传,失败自动补)。"""
    import chromadb
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rag_quota import quota_collection

    name = collection or quota_collection()
    workers = workers or int(os.environ.get("ALIAS_WORKERS", "4"))
    col = chromadb.PersistentClient(
        path=os.path.join(BASE_DIR, "chroma_db")).get_collection(name)
    got = col.get(include=["documents", "metadatas"])
    done = _done_ids()
    todo = [(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            if i not in done]
    if limit:
        todo = todo[:limit]
    print(f"[alias] {name}: {len(got['ids'])} 块,已缓存 {len(done)},本次 {len(todo)}"
          f"({workers} 并发,版本 {ALIAS_VERSION})")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    lock, ok, fail = threading.Lock(), 0, 0

    with ThreadPoolExecutor(max_workers=workers) as ex, \
            open(CACHE_PATH, "a", encoding="utf-8") as f:
        metas_out: dict = {}

        def _one(cid, doc):
            mt: dict = {}
            a = generate_alias(doc, meta=mt)
            metas_out[cid] = mt
            return a

        futs = {ex.submit(_one, i, d): (i, d, m) for i, d, m in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            cid, doc, meta = futs[fut]
            try:
                alias = fut.result()
            except Exception as e:
                fail += 1
                print(f"  ✗ {type(e).__name__}", file=sys.stderr)
                continue
            if not alias:
                fail += 1
                continue
            from rag_alias_health import content_hash, prompt_hash
            from rag_doc2query import doc_key
            # 血缘字段(指导 §9.1/§9.6):原文哈希与提示词指纹让"何时该重建"可判定,
            # 而不是靠人记得改过什么。老记录没有这些字段 → 体检时跳过该项(向后兼容)。
            line = {"chunk_id": cid, "doc_key": doc_key(doc, meta),
                    "source": (meta or {}).get("source", ""),
                    "alias_version": ALIAS_VERSION,
                    "source_content_hash": content_hash(doc),
                    "prompt_hash": prompt_hash(),
                    # **实际服务的模型**(不是配置名):网关会自动切兜底,两者可能不同
                    "model": (metas_out.get(cid) or {}).get("model")
                             or os.environ.get("RAG_SYNTH_MODEL", "") or "(default)",
                    "model_configured": os.environ.get("RAG_SYNTH_MODEL", "") or "(default)",
                    "finish_reason": (metas_out.get(cid) or {}).get("finish_reason"),
                    **alias}
            with lock:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
            ok += 1
            if n % 25 == 0:
                print(f"  进度 {n}/{len(todo)}(成功 {ok} / 失败 {fail})")
    print(f"[alias] 完成:成功 {ok},失败 {fail}(重跑本命令自动补)")
    return {"ok": ok, "fail": fail}


_MAP_CACHE: dict = {}     # (mtime, fields) -> {doc_key: 别名文本}


def _alias_text(row: dict, fields: tuple) -> str:
    """把选中的字段拼成一段**中文**文本,作为精排桥的"文档侧代表"。"""
    parts = []
    for f in fields:
        v = row.get(f)
        if not v:
            continue
        parts.append(v if isinstance(v, str) else "、".join(v))
    return " ".join(p for p in parts if p).strip()


def alias_map(fields: tuple | None = None) -> dict:
    """{doc_key: 中文别名文本};按缓存文件 mtime + 字段组合做进程内缓存。"""
    fields = alias_fields() if fields is None else fields
    try:
        mtime = os.path.getmtime(CACHE_PATH)
    except OSError:
        return {}
    key = (mtime, fields)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    out = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("alias_version") != ALIAS_VERSION:
                continue
            t = _alias_text(r, fields)
            if t and r.get("doc_key"):
                out[r["doc_key"]] = t
    _MAP_CACHE.clear()          # 只留最近一份,避免扫描多组合时无限增长
    _MAP_CACHE[key] = out
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="生成英文块的中文概念别名(检索用,非证据)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--stat", action="store_true", help="只看缓存统计")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.stat:
        m = alias_map()
        print(f"别名缓存 {len(m)} 条(版本 {ALIAS_VERSION},字段 {alias_fields()})")
        for k, v in list(m.items())[:3]:
            print(f"  {k[:40]!r} → {v[:110]}")
        return 0
    build_cache(a.collection, a.limit, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
