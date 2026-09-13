# -*- coding: utf-8 -*-
"""eval_rag_domain.py — RAG 分域诊断：域内召回 / 跨域混淆 / in_kb 门槛准确率。

⚠️ [问题2] 已降为**辅助诊断**——曾暴露「LoRA 是什么」被门控漏判（已由 rag_gate 反向边缘救援修复，见 RAG_OPTIMIZATION_LOG）。门槛/检索主口径用 `eval_rag_bench.py`。

用于回答"要不要分库 + 路由"：
- 域内召回：每个领域的问题，top-k 是否含本域文件
- 跨域纯度：top-k 里属于"本域"的比例（越低=混淆越重=越需要路由）
- 门槛准确率：gate_negatives 是否全部 in_kb=False，gate_positives 是否全部 True

用法：python eval_rag_domain.py [--k 5]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import chromadb
from rag_tools import get_collection_name, get_embedding
from rag_gate import gated_query

EVAL = os.path.join(BASE, "tests", "rag_domain_eval_set.json")
PREFIX_TO_DOMAIN = {
    "llm_app_interview": "llm_app",
    "backend_basic": "backend",
    "llm_algorithm": "algorithm",
    "llm_app_intro": "career",
}


def src_domain(src: str) -> str:
    for pref, dom in PREFIX_TO_DOMAIN.items():
        if src.startswith(pref):
            return dom
    return "internal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    data = json.load(open(EVAL, encoding="utf-8"))
    col = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db")).get_collection(get_collection_name())
    k = args.k

    print(f"=== RAG 分域诊断 (K={k}, DB={col.count()} chunks) ===\n")

    # ---- 域内召回 + 跨域纯度 ----
    from collections import defaultdict
    dom_hit = defaultdict(list)      # 域 → [是否命中本域]
    dom_purity = defaultdict(list)   # 域 → [top-k 本域占比]
    print(f"{'ID':<5}{'域':<10}{'命中':<5}{'本域纯度':<8} 查询")
    for it in data["items"]:
        emb = get_embedding(it["q"])
        res = col.query(query_embeddings=[emb], n_results=k)
        srcs = [m.get("source", "") for m in res["metadatas"][0]]
        doms = [src_domain(s) for s in srcs]
        hit = any(s.startswith(it["expect_prefix"]) for s in srcs)
        purity = sum(1 for d in doms if d == it["domain"]) / max(len(doms), 1)
        dom_hit[it["domain"]].append(hit)
        dom_purity[it["domain"]].append(purity)
        print(f"{it['id']:<5}{it['domain']:<10}{'✓' if hit else '✗':<5}{purity:>6.0%}   {it['q'][:30]}")

    print("\n--- 分域汇总 ---")
    print(f"{'域':<12}{'N':>3}  {'Recall@'+str(k):<10}{'平均本域纯度':<10}")
    all_hit, all_pur = [], []
    for dom in ["llm_app", "backend", "algorithm", "career"]:
        hits = dom_hit[dom]; purs = dom_purity[dom]
        all_hit += hits; all_pur += purs
        r = sum(hits)/len(hits) if hits else 0
        p = sum(purs)/len(purs) if purs else 0
        print(f"{dom:<12}{len(hits):>3}  {r:<10.2%}{p:<10.0%}")
    print(f"{'总体':<12}{len(all_hit):>3}  {sum(all_hit)/len(all_hit):<10.2%}{sum(all_pur)/len(all_pur):<10.0%}")

    # ---- 门槛准确率 ----
    print("\n--- in_kb 门槛准确率 ---")
    neg = data["gate_negatives"]; pos = data["gate_positives"]
    neg_ok = 0
    for q in neg:
        d = gated_query(q)
        ok = (d["in_kb"] is False)
        neg_ok += ok
        print(f"  [负] {'✓' if ok else '✗ 误判命中'} {q}  (dist={d.get('best_distance')})")
    pos_ok = 0
    for q in pos:
        d = gated_query(q)
        ok = (d["in_kb"] is True)
        pos_ok += ok
        print(f"  [正] {'✓' if ok else '✗ 漏判'} {q}  (in_kb={d['in_kb']}, by={d.get('matched_by')})")
    print(f"\n门槛：负样本拒答 {neg_ok}/{len(neg)} · 正样本命中 {pos_ok}/{len(pos)}")


if __name__ == "__main__":
    main()
