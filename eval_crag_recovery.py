# -*- coding: utf-8 -*-
"""E2 — CRAG 救回:held-out 上 RAG_CRAG=0 vs =1 的救回率 + 拒答不腐蚀（docs/MULTI_AGENT_UPGRADE.md §5）。

预登记判据(先登记后跑,防挪门柱):
  go  = recovery_rate>0（held-out 尾部有真救回）**且** 三把拒答尺全不降
        (eval_abstention 简单负 12/12、近似负 ≥11/12、adv 阈值不塌) → 采纳 CRAG,默认开。
  no-go = 退回工具 + 记**第四个诚实负结果**(并列 HyDE / 查询改写 / doc2query)。

重算力提示:需 reranker + held-out(52 题)双跑,属机器吃紧项——建议异机/挂机单进程跑
(与 P3/P4 codex 包同待遇)。跑前 pgrep 查场,避免并发 torch-MPS 僵死。

用法:
  RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 python eval_crag_recovery.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

HELDOUT = os.path.join(BASE, "tests", "rag_bench_paraphrase_set.json")


def _miss_set(crag: str) -> set:
    """在 held-out 上跑一遍,返回 miss(R@1 未命中)的题目 id 集合。RAG_CRAG 由参数控制。"""
    os.environ["RAG_CRAG"] = crag
    from eval_rag_bench import evaluate
    res = evaluate(set_path=HELDOUT, label=f"CRAG={crag}")
    miss = set()
    for item in res.get("per_item", res.get("items", [])):
        if not item.get("hit@1", item.get("r@1")):
            miss.add(item.get("id") or item.get("question", "")[:40])
    return miss, res.get("overall", res)


def run() -> dict:
    print("== E2:held-out RAG_CRAG=0(基线) vs =1(恢复) ==")
    miss0, ov0 = _miss_set("0")
    miss1, ov1 = _miss_set("1")
    recovered = miss0 - miss1          # 基线 miss、开 CRAG 后命中 = 真救回
    new_miss = miss1 - miss0           # 开 CRAG 后新掉的(应为空,否则腐蚀)
    rate = (len(recovered) / len(miss0)) if miss0 else 0.0
    print(f"基线 miss={len(miss0)}  开CRAG miss={len(miss1)}  救回={len(recovered)}  "
          f"新掉={len(new_miss)}  recovery_rate={rate:.1%}")
    print("拒答尺请另跑:RAG_CRAG=1 python eval_abstention.py(须 简单负 12/12 且 近似负 ≥11/12 不塌)")
    go = rate > 0 and not new_miss
    print(f"预登记判据(救回>0 且 无新掉 且 拒答不腐蚀): {'倾向采纳(还需拒答尺确认)' if go else 'no-go → 退回工具,记第四个诚实负结果'}")
    return {"miss_base": len(miss0), "miss_crag": len(miss1),
            "recovered": len(recovered), "new_miss": len(new_miss),
            "recovery_rate": rate, "go_hint": go}


if __name__ == "__main__":
    out = run()
    json.dump(out, open(os.path.join(BASE, "docs", "rag_eval", "crag_e2_result.json"), "w"),
              ensure_ascii=False, indent=2)
