# -*- coding: utf-8 -*-
"""eval_abstention.py — 拒答门槛独立评测（近似负样本集）。

跑 tests/rag_gate_adversarial_negatives.json：这些问题**听起来像知识库领域内**
（vLLM PagedAttention、MoE 负载均衡、YaRN 外推……），但逐条经 grep 考据确认
KB 正文 0 覆盖——门控必须全部拒答（in_kb=False）。这是 doc2query 这类"召回增强"
改动最容易打破的防线：合成口语问题可能把库外术语的向量距离拉近，A/B 必查。

用法：
    RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 python eval_abstention.py \
        [--set tests/rag_gate_adversarial_negatives.json] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_SET = os.path.join("tests", "rag_gate_adversarial_negatives.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_path", default=DEFAULT_SET)
    ap.add_argument("--json", dest="json_out", default="")
    args = ap.parse_args()

    from rag_gate import gated_query
    cases = json.load(open(args.set_path, encoding="utf-8"))
    rows, n_ok = [], 0
    for i, c in enumerate(cases, 1):
        print(f"\r  拒答评测 {i}/{len(cases)} [{c['id']}]   ", end="", file=sys.stderr, flush=True)
        res = gated_query(c["q"])
        rejected = res.get("in_kb") is False
        n_ok += rejected
        rows.append({"id": c["id"], "rejected": rejected,
                     "mode": res.get("mode"), "best_distance": res.get("best_distance")})
    print("", file=sys.stderr)

    out = {"set": args.set_path, "n": len(cases), "rejected": n_ok,
           "acc": round(n_ok / max(len(cases), 1), 4),
           "flags": {k: os.environ.get(k) for k in
                     ("RAG_RERANK", "RAG_BM25", "RAG_ROUTE", "RAG_DOC2QUERY", "RAG_QUERY_REWRITE")},
           "rows": rows}
    print(f"近似负样本拒答: {n_ok}/{len(cases)}")
    for r in rows:
        if not r["rejected"]:
            print(f"  ✗ 误收 [{r['id']}] mode={r['mode']} best={r['best_distance']}")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump(out, open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[已存] {args.json_out}")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
