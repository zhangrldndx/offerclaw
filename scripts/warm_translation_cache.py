# -*- coding: utf-8 -*-
"""预热查询翻译缓存(报告 §13.2 要求 normalized_query_hash → translation_result)。

翻译在查询路径上,不缓存 = 每次问答多付一次 LLM 往返;评测要跑几百次同一批 query,
没有预热的缓存实验既不可复现也不可承受。本脚本一次性把评测集的 query 全部翻好落盘。

用法(走生产 LLM 网关,主模型不可用时自动兜底):
  .venv/bin/python scripts/warm_translation_cache.py

注意别把 OPENAI_BASE_URL 直接指到兜底端点:网关在**兜底时才**会 pop 掉
`reasoning_effort`(兜底方未必支持该扩展参数),直连会带着它发出去 → 400。

`translate_query` 会缓存失败结果(生产语义:避免坏 query 反复重试烧钱),
故预热按多轮进行,每轮把仍失败的条目从缓存清掉再试。
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SETS = ["tests/rag_bench_xling60_set.json", "tests/rag_bench_paraphrase_set.json",
        "tests/rag_bench_set.json"]


def collect() -> list:
    from rag_translate import needs_translation
    qs: list = []
    for s in SETS:
        p = os.path.join(BASE, s)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        qs += [i["q"] for i in d.get("items", [])]
        qs += d.get("gate_negatives", []) + d.get("gate_positives", [])
    p = os.path.join(BASE, "tests", "rag_gate_adversarial_negatives.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            a = json.load(f)
        qs += [x["q"] if isinstance(x, dict) else x for x in a]
    return [q for q in dict.fromkeys(qs) if q and needs_translation(q)]


def main() -> int:
    import rag_translate as T
    qs = collect()
    cache = T._load()
    workers = int(os.environ.get("WARM_WORKERS", "3"))
    rounds = int(os.environ.get("WARM_ROUNDS", "3"))
    t0 = time.time()

    for rnd in range(1, rounds + 1):
        todo = [q for q in qs if not cache.get(T._key(q))]
        if not todo:
            break
        for q in todo:                       # 清掉上一轮的失败标记,否则命中缓存不重试
            cache.pop(T._key(q), None)
        print(f"[第 {rnd} 轮] 待翻 {len(todo)} 条(已缓存 {len(qs) - len(todo)})", flush=True)
        done = {"n": 0}

        def work(q):
            r = T.translate_query(q)
            done["n"] += 1
            if done["n"] % 20 == 0:
                print(f"  {done['n']}/{len(todo)} 用时{time.time() - t0:.0f}s", flush=True)
            return r

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))

    st = T.cache_stats()
    ok = st["entries"] - st["failed"]
    print(f"[完成] 成功 {ok}/{len(qs)} · 仍失败 {st['failed']} · 用时 {time.time() - t0:.0f}s")
    print(f"        缓存 {st['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
