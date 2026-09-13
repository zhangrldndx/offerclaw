# -*- coding: utf-8 -*-
"""eval_scale_test.py — P3：向量检索规模压测(回答"知识库涨到 10 万块会怎样")。

方法(只测事实,不碰生产集合):
1. 合成语料 = 从真实知识库随机抽句子重组成 ~300 字块(保持文本分布贴近真实,
   不用随机字符串——随机串的向量分布假,测出来的延迟没有参考意义);
2. 灌入一次性集合 scale_test_<N>(与生产集合物理隔离),分档 1万/5万/10万;
3. 每档用 30 条真实问题测**纯向量检索**延迟 p50/p95(规模敏感部分;
   reranker/BM25 的耗时与库规模基本无关,混进来会稀释结论),并记进程 RSS 内存;
4. 测完 --cleanup 删全部临时集合,不留垃圾。

用法：
    python eval_scale_test.py --scales 10000 50000 100000 \
        --out docs/rag_eval/round9/scale_test.json
    python eval_scale_test.py --cleanup          # 只清理临时集合
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PREFIX = "scale_test_"
CHUNK_CHARS = 300
QUERY_N = 30
SEED = 42  # 固定种子:语料与查询可复现


def _client():
    import chromadb
    return chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))


def _sentence_pool() -> list[str]:
    """从真实知识库抽句子池(≥8 字的中文句),作为合成块的原料。"""
    pool = []
    kb = os.path.join(BASE_DIR, "knowledge_base", "learning_resources")
    for fn in sorted(os.listdir(kb)):
        if not fn.endswith(".md"):
            continue
        text = open(os.path.join(kb, fn), encoding="utf-8").read()
        pool += [s.strip() for s in re.split(r"[。！？\n]", text) if len(s.strip()) >= 8]
    return pool


def _synth_chunks(n: int, pool: list[str], rng: random.Random) -> list[str]:
    out = []
    for _ in range(n):
        buf = []
        size = 0
        while size < CHUNK_CHARS:
            s = pool[rng.randrange(len(pool))]
            buf.append(s)
            size += len(s)
        out.append("。".join(buf)[:CHUNK_CHARS + 100])
    return out


def _real_queries() -> list[str]:
    d = json.load(open(os.path.join(BASE_DIR, "tests", "rag_bench_paraphrase_set.json"),
                       encoding="utf-8"))
    qs = [x["q"] for x in d.get("items", [])]
    rng = random.Random(SEED)
    return (qs * ((QUERY_N // max(len(qs), 1)) + 1))[:QUERY_N] if qs else []


def _rss_mb() -> float:
    import psutil
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)


def build_and_measure(n: int, pool: list[str]) -> dict:
    from rag_tools import get_embeddings_batch
    client = _client()
    name = f"{PREFIX}{n}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name)

    rng = random.Random(SEED + n)
    chunks = _synth_chunks(n, pool, rng)
    print(f"[{name}] 合成 {n} 块,向量化入库(本地 bge,批 128)…")
    t0 = time.perf_counter()
    B = 128
    for s in range(0, n, B):
        embs = get_embeddings_batch(chunks[s:s + B])
        col.add(ids=[f"s{n}_{i}" for i in range(s, min(s + B, n))],
                documents=chunks[s:s + B], embeddings=embs)
        if (s // B) % 50 == 0:
            print(f"  入库 {min(s + B, n)}/{n}")
    ingest_s = round(time.perf_counter() - t0, 1)

    # 纯向量检索延迟(embedding 查询向量的耗时也计入——这是线上每次查询的真实构成)
    queries = _real_queries()
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        emb = get_embeddings_batch([q])
        col.query(query_embeddings=emb, n_results=20)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    row = {"n_chunks": n, "ingest_s": ingest_s,
           "query_p50_ms": round(lat[len(lat) // 2], 1),
           "query_p95_ms": round(lat[int(len(lat) * 0.95)], 1),
           "rss_mb_after": _rss_mb()}
    print(f"[{name}] 检索 p50={row['query_p50_ms']}ms p95={row['query_p95_ms']}ms "
          f"RSS={row['rss_mb_after']}MB 入库{ingest_s}s")
    return row


def cleanup() -> None:
    client = _client()
    for col in client.list_collections():
        if col.name.startswith(PREFIX):
            client.delete_collection(col.name)
            print(f"[清理] 已删 {col.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[10000, 50000, 100000])
    ap.add_argument("--out", default="docs/rag_eval/round9/scale_test.json")
    ap.add_argument("--cleanup", action="store_true", help="只清理临时集合后退出")
    ap.add_argument("--keep", action="store_true", help="测完保留集合(默认删)")
    args = ap.parse_args()

    if args.cleanup:
        cleanup()
        return 0

    pool = _sentence_pool()
    print(f"[语料池] 真实句子 {len(pool)} 条(seed={SEED} 可复现)")
    baseline_rss = _rss_mb()
    results = [build_and_measure(n, pool) for n in sorted(args.scales)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "chunk_chars": CHUNK_CHARS, "query_n": QUERY_N,
                   "baseline_rss_mb": baseline_rss, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[已存] {args.out}")
    if not args.keep:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
