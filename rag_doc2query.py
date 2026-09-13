# -*- coding: utf-8 -*-
"""rag_doc2query.py — 文档侧检索增强（doc2query，Round 9 / P1）。

动机（对症下药）：held-out 口语化评测显示真实口径 R@1 只有 50%——用户"说人话"、
文档是书面语，向量对不上。HyDE（查询侧现场改写）已在 Round 3 被实测证伪：每次
查询多一轮 LLM、延迟大且不治文档侧的病。doc2query 反过来：**离线**给每个文档块
生成若干"用户可能这么问"的口语化问题，把问题向量入索引——一次性成本，查询零延迟。

诚实性约束（本轮铁律）：
- 生成 prompt 是**通用的**（口语化提问风格），生成过程看不到任何评测题——
  否则就是把答案漏给系统，指标全作废；
- 是否默认开启由 held-out A/B 数据决定，不预设结论；
- 合成问题存**独立集合**（``<主集合名>_d2q``），主集合 2607 chunks 计数不受污染，
  关掉开关（``RAG_DOC2QUERY=0``）行为与改造前逐字节一致。

链路：
  生成:  build_cache()  主集合逐块 → LLM 出 3 问 → logs/doc2query_cache.jsonl（断点续传）
  入库:  ingest_cache() 缓存问题 → 本地 bge 向量化 → 独立集合（id=d2q::<父块id>::<序号>）
  查询:  rag_gate 在向量粗召回后调 query_d2q_and_merge()——合成问题命中 → 换回父块
        原文参与后续 BM25/rerank/路由/门控（下游无感知）。

CLI：
  python rag_doc2query.py --build     # 生成问题缓存（可中断重跑，走统一网关有计量）
  python rag_doc2query.py --ingest    # 缓存 → 向量库
  python rag_doc2query.py --status    # 进度一览
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CACHE_PATH = os.path.join(BASE_DIR, "logs", "doc2query_cache.jsonl")
QUESTIONS_PER_CHUNK = 3
_D2Q_SUFFIX = "_d2q"


def d2q_enabled() -> bool:
    """默认关。A/B 验证有收益后由使用方显式开启（.env.local / 部署环境）。"""
    return os.environ.get("RAG_DOC2QUERY", "0").strip().lower() in ("1", "true", "yes", "on")


def d2q_collection_name() -> str:
    from rag_tools import get_collection_name
    return get_collection_name() + _D2Q_SUFFIX


# =====================================================
# 生成
# =====================================================

_GEN_PROMPT = (
    "你在为检索系统做 doc2query 索引增强。下面是知识库中的一段文档内容。\n"
    "请生成 {n} 个「普通用户可能会随口问出来」的问题，要求：\n"
    "1) 口语化、别照抄原文里的术语标题（用大白话表达同一个意思）；\n"
    "2) 这段内容必须能直接回答该问题；\n"
    "3) 每个问题不超过 40 字；\n"
    "4) 只输出 JSON 数组，如 [\"问题1\", \"问题2\", \"问题3\"]，不要其他文字。\n\n"
    "文档内容：\n{text}"
)


def _parse_questions(raw: str) -> list[str]:
    """解析 LLM 输出：优先 JSON 数组；失败降级为逐行抽取（防模型话痨）。"""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(q).strip() for q in arr if str(q).strip()][:QUESTIONS_PER_CHUNK]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r'^[\d\.\-\*、"\s]+', "", ln).strip().strip('",')
             for ln in raw.splitlines()]
    return [ln for ln in lines if 5 <= len(ln) <= 60][:QUESTIONS_PER_CHUNK]


def generate_questions(text: str) -> list[str]:
    """对一个 chunk 调一次 LLM 生成口语化问题。走 rag_gate._chat → 统一网关（自动计量/重试）。

    契约注意：``_chat`` 返回**提取后的文本 str**（A2 防御解析已做），不是响应 dict
    ——首版在这里按 dict 解包吃了哑巴亏（TypeError 被兜底吞掉，表现为 0 产出）。
    ``enable_thinking=False``：qwen3 系思考模式对"输出 3 个短问题"这种任务纯属
    烧钱烧时间（实测思考开着单次 ~16s / 800+ tok，关掉见 build 日志）。"""
    from rag_gate import _chat
    raw = _chat([{"role": "user",
                  "content": _GEN_PROMPT.format(n=QUESTIONS_PER_CHUNK, text=text[:1500])}],
                max_tokens=300, temperature=0.7,
                extra_payload={"enable_thinking": False})
    return _parse_questions(raw) if isinstance(raw, str) else []


def _load_cache_ids() -> set:
    done = set()
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["parent_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # 半截行跳过（中断重跑时该块会重新生成）
    return done


def build_cache(limit: int | None = None, workers: int | None = None) -> dict:
    """遍历主集合，为每个未处理的 chunk 生成问题并追加进缓存（断点续传）。"""
    import chromadb
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rag_tools import get_collection_name

    workers = workers or int(os.environ.get("DOC2QUERY_WORKERS", "6"))
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    col = client.get_collection(get_collection_name())
    total = col.count()
    got = col.get(limit=total, include=["documents", "metadatas"])
    done = _load_cache_ids()
    todo = [(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            if i not in done]
    if limit:
        todo = todo[:limit]
    print(f"[build] 主集合 {total} 块，已缓存 {len(done)}，本次待生成 {len(todo)}（{workers} 并发）")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    lock = threading.Lock()
    ok = fail = 0

    def _one(item):
        pid, doc, meta = item
        qs = generate_questions(doc)
        return pid, meta, qs

    with ThreadPoolExecutor(max_workers=workers) as ex, \
         open(CACHE_PATH, "a", encoding="utf-8") as f:
        futs = [ex.submit(_one, t) for t in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                pid, meta, qs = fut.result()
            except Exception as e:  # 单块失败不拖垮整批，重跑时自动补
                fail += 1
                print(f"  ✗ 生成失败（{type(e).__name__}），跳过", file=sys.stderr)
                continue
            if qs:
                line = {"parent_id": pid, "source": meta.get("source", ""),
                        "source_type": meta.get("source_type", ""), "questions": qs}
                with lock:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
                ok += 1
            else:
                fail += 1
            if n % 100 == 0:
                print(f"  进度 {n}/{len(todo)}（成功 {ok} / 失败 {fail}）")
    print(f"[build] 完成：成功 {ok}，失败 {fail}（失败块重跑本命令自动补）")
    return {"ok": ok, "fail": fail}


# =====================================================
# 入库
# =====================================================

def ingest_cache(batch: int = 64) -> dict:
    """把缓存中的问题向量化写入独立集合（重建式：先删后建，幂等）。"""
    import chromadb
    from rag_tools import get_embeddings_batch

    rows = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    name = d2q_collection_name()
    try:
        client.delete_collection(name)  # 独立派生集合，重建最简且幂等
    except Exception:
        pass
    col = client.create_collection(name)

    ids, docs, metas = [], [], []
    for r in rows:
        for i, q in enumerate(r["questions"]):
            ids.append(f"d2q::{r['parent_id']}::{i}")
            docs.append(q)
            metas.append({"parent_id": r["parent_id"], "source": r.get("source", ""),
                          "source_type": r.get("source_type", "")})
    print(f"[ingest] {len(rows)} 块 → {len(ids)} 个合成问题，向量化中（本地 bge，批 {batch}）")
    for s in range(0, len(ids), batch):
        embs = get_embeddings_batch(docs[s:s + batch])
        col.add(ids=ids[s:s + batch], documents=docs[s:s + batch],
                metadatas=metas[s:s + batch], embeddings=embs)
        if (s // batch) % 20 == 0:
            print(f"  {min(s + batch, len(ids))}/{len(ids)}")
    print(f"[ingest] 完成：{col.count()} 条入 {name}")
    return {"questions": len(ids)}


# =====================================================
# 查询期合并（纯函数 + 装配）
# =====================================================

def merge_parent_hits(docs: list, metas: list, dists: list,
                      parent_hits: list[tuple[str, str, dict, float]]) -> tuple:
    """纯函数：把 (parent_id, 父块原文, 父块meta, 合成问题距离) 合入主召回结果。

    规则：同一父块取**更小**距离（合成问题命中说明语义更贴近用户问法）；
    合并后按距离升序。可独立单测。
    """
    by_key: dict = {}
    for d, m, dist in zip(docs, metas, dists):
        by_key[id(d) if d is None else d[:80] + str(m.get("source", ""))] = [d, m, dist]
    for pid, pdoc, pmeta, pdist in parent_hits:
        if pdoc is None:
            continue  # 父块已被重建/删除：跳过失效映射
        key = pdoc[:80] + str(pmeta.get("source", ""))
        if key in by_key:
            if pdist < by_key[key][2]:
                by_key[key][2] = pdist
        else:
            by_key[key] = [pdoc, pmeta, pdist]
    merged = sorted(by_key.values(), key=lambda x: x[2])
    return ([x[0] for x in merged], [x[1] for x in merged], [x[2] for x in merged])


def doc_key(doc: str, meta: dict) -> str:
    """候选块的对齐键(与 merge_parent_hits 同方案):文本前 80 字 + 来源文件。"""
    return (doc or "")[:80] + str((meta or {}).get("source", ""))


def build_synth_map(query_emb: list, n_results: int, client) -> dict:
    """[Round 10 语体桥] 查 d2q 集合,返回 {父块 doc_key: 与本 query 最相近的合成问题文本}。

    给精排用:reranker 对 (query, 书面父块) 打分吃语体错配的亏,而 (query, 口语合成问题)
    同语体——取两者 max 作为该父块的精排分。合成问题的选择本身用向量近邻(即本函数),
    每个父块只取最相近的一条,精排额外开销 ≤ 命中父块数。集合不存在时返回 {}(零副作用)。
    """
    from rag_tools import get_collection_name
    try:
        d2q = client.get_collection(d2q_collection_name())
    except Exception:
        return {}
    res = d2q.query(query_embeddings=query_emb, n_results=n_results,
                    include=["documents", "metadatas", "distances"])
    qs = res.get("documents", [[]])[0]
    ms = res.get("metadatas", [[]])[0]
    ds = res.get("distances", [[]])[0]
    if not qs:
        return {}
    best: dict = {}   # parent_id -> (dist, 合成问题)
    for q, m, dist in zip(qs, ms, ds):
        pid = m.get("parent_id")
        if pid and (pid not in best or dist < best[pid][0]):
            best[pid] = (dist, q)
    main = client.get_collection(get_collection_name())
    got = main.get(ids=list(best.keys()), include=["documents", "metadatas"])
    return {doc_key(doc, meta): best[pid][1]
            for pid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
            if doc is not None}


def query_d2q_and_merge(query_emb: list, docs: list, metas: list, dists: list,
                        n_results: int, client) -> tuple:
    """查合成问题集合 → 命中映射回父块 → 与主召回合并。集合不存在时原样返回（未 ingest 前无副作用）。"""
    from rag_tools import get_collection_name
    try:
        d2q = client.get_collection(d2q_collection_name())
    except Exception:
        return docs, metas, dists
    res = d2q.query(query_embeddings=query_emb, n_results=n_results,
                    include=["metadatas", "distances"])
    hit_metas = res.get("metadatas", [[]])[0]
    hit_dists = res.get("distances", [[]])[0]
    if not hit_metas:
        return docs, metas, dists
    # 每个父块只保留其最优合成问题距离
    best_by_parent: dict = {}
    for m, dist in zip(hit_metas, hit_dists):
        pid = m.get("parent_id")
        if pid and (pid not in best_by_parent or dist < best_by_parent[pid]):
            best_by_parent[pid] = dist
    main = client.get_collection(get_collection_name())
    got = main.get(ids=list(best_by_parent.keys()), include=["documents", "metadatas"])
    parent_hits = [(pid, doc, meta, best_by_parent[pid])
                   for pid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])]
    return merge_parent_hits(docs, metas, dists, parent_hits)


def main() -> int:
    ap = argparse.ArgumentParser(description="doc2query 文档侧增强")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--limit", type=int, help="--build 时只处理前 N 块（试跑用）")
    args = ap.parse_args()
    if args.build:
        build_cache(limit=args.limit)
    if args.ingest:
        ingest_cache()
    if args.status or not (args.build or args.ingest):
        import chromadb
        from rag_tools import get_collection_name
        done = _load_cache_ids()
        client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
        total = client.get_collection(get_collection_name()).count()
        try:
            n_d2q = client.get_collection(d2q_collection_name()).count()
        except Exception:
            n_d2q = 0
        print(f"缓存进度 {len(done)}/{total} 块；d2q 集合 {n_d2q} 条；开关 RAG_DOC2QUERY="
              f"{'开' if d2q_enabled() else '关'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
