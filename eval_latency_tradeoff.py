# -*- coding: utf-8 -*-
"""eval_latency_tradeoff.py — P4：精排候选数(RAG_RECALL_N)的延迟-精度权衡扫描。

问题：完整检索链路 p50 ≈ 1.3s,大头在 bge-reranker 交叉编码器——它精排整个粗召回池,
池子越大越准也越慢。本脚本在同一进程内(模型只加载一次,对比公平)扫 RECALL_N 档位,
对每档跑两套冻结评测集(同分布 + held-out)的 R@1,并逐题计时取 p50/p95。

口径纪律：精度全部复用 eval_rag_bench.evaluate()(生产同路径),延迟计时打在
eval_rag_bench._ranked_sources 上(同一条检索函数)——不自造第二套评测逻辑。

用法：
    RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 python eval_latency_tradeoff.py \
        --settings 5 10 20 40 --out docs/rag_eval/round9/latency_tradeoff.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

BENCH_SET = "tests/rag_bench_set.json"
HELDOUT_SET = "tests/rag_bench_paraphrase_set.json"


def _load_questions(path: str) -> list[str]:
    d = json.load(open(path, encoding="utf-8"))
    items = d if isinstance(d, list) else d.get("items", [])
    return [x["q"] for x in items]


def time_retrieval(questions: list[str], top_k: int = 5) -> dict:
    """逐题计时完整检索路径(向量+BM25+RRF+rerank+路由+门控),返回 p50/p95/mean(ms)。"""
    from eval_rag_bench import _ranked_sources
    lat = []
    for q in questions:
        t0 = time.perf_counter()
        _ranked_sources(q, top_k)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return {"p50_ms": round(lat[len(lat) // 2], 1),
            "p95_ms": round(lat[int(len(lat) * 0.95)], 1),
            "mean_ms": round(statistics.mean(lat), 1), "n": len(lat)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--out", default="docs/rag_eval/round9/latency_tradeoff.json")
    args = ap.parse_args()

    from eval_rag_bench import evaluate

    # 预热：加载 bge / reranker / jieba,避免首题计时被模型加载污染
    print("[预热] 首次检索加载模型 …")
    time_retrieval(["预热问题：RAG 是什么"])

    results = []
    for n in args.settings:
        os.environ["RAG_RECALL_N"] = str(n)
        print(f"\n===== RAG_RECALL_N = {n} =====")
        row = {"recall_n": n}
        for tag, sp in (("bench", BENCH_SET), ("heldout", HELDOUT_SET)):
            m = evaluate(top_k=5, set_path=sp)
            row[f"{tag}_R1"] = m["overall"]["recall@1"]
            row[f"{tag}_MRR"] = m["overall"]["mrr"]
            row[f"{tag}_n"] = m["overall"]["n"]
            print(f"  [{tag}] R@1={m['overall']['recall@1']:.3f}  MRR={m['overall']['mrr']:.3f}")
        # 延迟用 held-out 全集计时(口语问法=真实负载形态)
        row["latency"] = time_retrieval(_load_questions(HELDOUT_SET))
        print(f"  [延迟] p50={row['latency']['p50_ms']}ms  p95={row['latency']['p95_ms']}ms")
        results.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"flags": {k: os.environ.get(k) for k in
                             ("RAG_RERANK", "RAG_BM25", "RAG_ROUTE", "RAG_DOC2QUERY")},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[已存] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
