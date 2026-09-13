# -*- coding: utf-8 -*-
"""
OfferClaw · RAG 检索质量评估 v2 (eval_rag.py)

⚠️ [问题2] 已降为**辅助口径**——50 题在知识库扩到 3299 块后口径漂移（Recall@5 0.88，miss 集中在 profile/status/data contract/application 类，这些 expect_source 已不一定合理）。
**主口径请用 `eval_rag_bench.py`**（58 题精确到目标文件 + Recall@1/3/5 + MRR + 拒答门槛 + 难题分桶）。

变化（v2）：
- 黄金集移到 tests/rag_eval_set.json（50 题），含 fact / explain / cross_doc 三桶
- 报指标时按"总体 + 分桶"分别输出
- 支持 --json 导出结果到 logs/rag_eval_<时间>.json，方便 CI 和回归对比
- 新增 --baseline 与上次结果比对，回归则非零退出码

使用：
    python eval_rag.py
    python eval_rag.py --k 3
    python eval_rag.py --verbose
    python eval_rag.py --json
    python eval_rag.py --baseline logs/rag_eval_xxx.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any

import chromadb

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from rag_tools import get_collection_name, get_embedding  # noqa: E402

DB_DIR = os.path.join(BASE, "chroma_db")
COLLECTION_NAME = get_collection_name()
EVAL_SET_PATH = os.path.join(BASE, "tests", "rag_eval_set.json")


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def evaluate(k: int = 5, verbose: bool = False) -> dict[str, Any]:
    if not os.path.exists(DB_DIR):
        print(f"[ERROR] {DB_DIR} 不存在，请先 python rag_ingest.py")
        sys.exit(1)
    if not os.path.exists(EVAL_SET_PATH):
        print(f"[ERROR] 评估集 {EVAL_SET_PATH} 不存在")
        sys.exit(1)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    items = load_eval_set()
    n = len(items)
    print(f"=== OfferClaw RAG 评估 v2 (K={k}, N={n}, DB={collection.count()} chunks) ===\n")

    per_item: list[dict] = []
    for idx, item in enumerate(items, 1):
        q = item["q"]
        expected = item["expected_sources"]
        cat = item.get("category", "uncategorized")

        try:
            emb = get_embedding(q)
        except Exception as e:
            print(f"[ERR] embedding fail Q{idx}: {e}")
            per_item.append({"id": item.get("id", str(idx)), "category": cat,
                             "hit": False, "rank": 0, "sources": [], "error": str(e)})
            continue

        res = collection.query(query_embeddings=[emb], n_results=k)
        sources = [m.get("source", "") for m in res["metadatas"][0]]

        rank = 0
        for j, src in enumerate(sources, 1):
            if any(exp in src for exp in expected):
                rank = j
                break
        hit = rank > 0

        per_item.append({
            "id": item.get("id", str(idx)),
            "category": cat,
            "q": q,
            "expected": expected,
            "sources": sources,
            "hit": hit,
            "rank": rank,
        })

        flag = "[HIT] " if hit else "[MISS]"
        print(f"{flag} {item.get('id','?'):>4} ({cat:<9}) {q}")
        if not hit or verbose:
            print(f"       expect: {expected}")
            print(f"       top{k}:  {sources}")
        if verbose:
            for j, (s, doc) in enumerate(zip(sources, res["documents"][0]), 1):
                print(f"        #{j} {s} :: {doc[:80].replace(chr(10),' ')}...")

    def metrics(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0, "recall_at_k": 0.0, "mrr": 0.0, "hits": 0}
        hits = sum(1 for r in rows if r["hit"])
        mrr = sum((1.0 / r["rank"]) for r in rows if r["rank"] > 0) / len(rows)
        return {"n": len(rows), "recall_at_k": hits / len(rows), "mrr": mrr, "hits": hits}

    overall = metrics(per_item)
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in per_item}):
        rows = [r for r in per_item if r["category"] == cat]
        by_cat[cat] = metrics(rows)

    print()
    print("=" * 64)
    print(f"{'bucket':<12}{'N':>4}  Recall@{k:<2}    MRR")
    print("-" * 64)
    print(f"{'overall':<12}{overall['n']:>4}    {overall['recall_at_k']:.3f}    {overall['mrr']:.3f}")
    for cat, m in by_cat.items():
        print(f"{cat:<12}{m['n']:>4}    {m['recall_at_k']:.3f}    {m['mrr']:.3f}")
    print("=" * 64)

    return {
        "k": k, "n": n, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall": overall, "by_category": by_cat,
        "per_item": per_item,
    }


def compare_baseline(current: dict, baseline_path: str) -> int:
    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    cur_r = current["overall"]["recall_at_k"]
    base_r = base["overall"]["recall_at_k"]
    delta = cur_r - base_r
    print(f"\n[Baseline] Recall@K: {base_r:.3f} -> {cur_r:.3f} (delta={delta:+.3f})")
    if delta < -0.02:
        print("[REGRESS] recall dropped >0.02")
        return 1
    print("[OK] no regression")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG eval v2")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--baseline")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    result = evaluate(k=args.k, verbose=args.verbose)

    if args.json:
        os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
        out = os.path.join(BASE, "logs", f"rag_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[exported] {out}")

    if args.baseline:
        return compare_baseline(result, args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
