# -*- coding: utf-8 -*-
"""eval_quota_gate.py — 配额对拒答边界的影响(方案报告 §27.5 / Phase 5 前置观测)。

直接走 `_retrieve_and_classify` 取 `in_kb`——与 `gated_query` 的门控判定同一真源,
但不触发答案合成,因此不依赖 LLM 可用性(评测不该因为代理 503 而失败)。

两组负样本:
  · trivial:知识库确实没有的通用问题(rag_bench_set.gate_negatives);
  · adversarial:近域诱导(rag_gate_adversarial_negatives.json),更能暴露"沾边就答"。
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def _load_negatives() -> tuple:
    with open(os.path.join(BASE, "tests", "rag_bench_set.json"), encoding="utf-8") as f:
        d = json.load(f)
    trivial, positives = d.get("gate_negatives", []), d.get("gate_positives", [])
    adv = []
    p = os.path.join(BASE, "tests", "rag_gate_adversarial_negatives.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            a = json.load(f)
        adv = a if isinstance(a, list) else (a.get("items") or a.get("negatives") or [])
        adv = [x["q"] if isinstance(x, dict) else x for x in adv]
    return trivial, adv, positives


def run() -> dict:
    from rag_gate import _retrieve_and_classify
    trivial, adv, positives = _load_negatives()
    out = {}
    for name, qs, want in (("trivial_neg", trivial, False),
                           ("adversarial_neg", adv, False),
                           ("positives", positives, True)):
        hits, bad = 0, []
        for q in qs:
            try:
                ok = _retrieve_and_classify(q, top_k=5)["in_kb"] is want
            except Exception:
                ok = False
            hits += ok
            if not ok:
                bad.append(q[:40])
        out[name] = {"ok": hits, "n": len(qs), "failed": bad}
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    label = os.environ.get("GATE_LABEL", "")
    res = run()
    print(f"=== 拒答边界 {label} ===")
    for k, v in res.items():
        print(f"  {k:<16} {v['ok']}/{v['n']}" + (f"   未过:{v['failed']}" if v["failed"] else ""))
    dst = os.environ.get("GATE_OUT")
    if dst:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        json.dump(res, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[已存] {dst}")
