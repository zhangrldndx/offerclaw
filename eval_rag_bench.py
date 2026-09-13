# -*- coding: utf-8 -*-
"""eval_rag_bench.py — RAG 优化基准评测器（逐轮量化改进用）。

相比旧的 eval_rag_domain.py（判定宽松：命中本域任意文件即算），本评测器：
- **精确到目标文件**：expect_sources 用唯一子串（小写包含匹配），命中具体目标文件才算；
- **排名敏感指标**：Recall@1 / @3 / @5 + MRR（rerank/重排的价值主要体现在这里）；
- **难题分桶**：hard=true 的题单列（向量易漏、靠词法/rerank/混合检索救回）；
- **门槛准确率**：gate_negatives 拒答 + gate_positives 命中；
- **baseline 对比**：--save 存基线，--baseline 读基线并打印每个指标的 Δ。

接入点：通过 rag_gate._retrieve_and_classify 取检索结果（含 rerank/混合检索改造后的最终顺序），
保证评测的就是线上问答真实走的检索链路。

用法：
  python eval_rag_bench.py                          # 跑评测，打印指标
  python eval_rag_bench.py --save logs/rag_round0.json   # 存基线
  python eval_rag_bench.py --baseline logs/rag_round0.json  # 与基线对比
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

EVAL = os.path.join(BASE, "tests", "rag_bench_set.json")


def _ranked_sources(question: str, top_k: int) -> list[str]:
    """取检索后最终排序的 source 文件名列表（走线上同一条检索链路）。"""
    from rag_gate import _retrieve_and_classify
    g = _retrieve_and_classify(question, top_k=top_k)
    return [(m or {}).get("source", "") for m in g.get("metas", [])]


def _probe(question: str, top_k: int) -> dict:
    """一次检索取回「排序 + 候选池 + 延迟」三类观测(方案报告 §26 分层指标)。

    候选池指标记录在 **rerank 前**的融合池上(报告 §26.1),用于回答"召回池是否修好"——
    与最终 R@1 分开看:池子没修好时 R@1 的任何波动都是下游在补锅,不是根因。
    """
    import time
    from rag_gate import _retrieve_and_classify
    t0 = time.perf_counter()
    g = _retrieve_and_classify(question, top_k=top_k)
    return {"srcs": [(m or {}).get("source", "") for m in g.get("metas", [])],
            "pool": g.get("pool"), "in_kb": g.get("in_kb"), "diag": g.get("diag"),
            "latency_ms": (time.perf_counter() - t0) * 1000.0}


def _hit_rank(srcs: list[str], expect: list[str]) -> int:
    """目标文件在结果里的最靠前名次（1-based）；未命中返回 0。"""
    exp = [e.lower() for e in expect]
    for i, s in enumerate(srcs):
        sl = (s or "").lower()
        if any(e in sl for e in exp):
            return i + 1
    return 0


def evaluate(top_k: int = 5, verbose: bool = False, set_path: str | None = None,
             skip_gate: bool = False) -> dict:
    data = json.load(open(set_path or EVAL, encoding="utf-8"))
    items = data["items"]
    rows = []
    n_total = len(items)
    for i, it in enumerate(items, 1):
        # [问题4] 逐题进度到 stderr（rerank 耗时长，无进度易误判卡死；stderr 不污染 stdout 结果）
        if verbose:
            print(f"  [{i}/{n_total}] {it['id']} {it['q'][:30]}", file=sys.stderr, flush=True)
        else:
            print(f"\r  评测中 {i}/{n_total} [{it['id']}]      ", end="", file=sys.stderr, flush=True)
        pr = _probe(it["q"], top_k)
        rank = _hit_rank(pr["srcs"], it["expect_sources"])
        row = {"id": it["id"], "domain": it["domain"], "hard": it.get("hard", False),
               "rank": rank, "q": it["q"], "latency_ms": round(pr["latency_ms"], 1)}
        if pr.get("diag"):
            dg = dict(pr["diag"])
            # English False Winner(指导 §16.4):英文块占了 Top1 而目标没在 Top1
            # —— 中文业务题上这正是 held-out 回退的直接成因,必须逐题可数而非只看总分
            dg["english_false_winner"] = bool(dg.get("top1_is_english") and rank != 1)
            row["diag"] = dg
        if pr["pool"] is not None:          # 配额开启才有候选池观测
            pool = dict(pr["pool"])
            # 精排**前**目标是否在池中及其名次(报告 §26.1):把"召回不足"与"排序不佳"分开归因
            pool["target_rank_in_pool"] = _hit_rank(pool.pop("sources", []),
                                                    it["expect_sources"])
            row["pool"] = pool
        rows.append(row)
    if not verbose:
        print("", file=sys.stderr)  # 进度行收尾换行

    def _agg(subset):
        n = len(subset)
        if n == 0:
            return {"n": 0}
        r1 = sum(1 for r in subset if 0 < r["rank"] <= 1) / n
        r3 = sum(1 for r in subset if 0 < r["rank"] <= 3) / n
        r5 = sum(1 for r in subset if 0 < r["rank"] <= 5) / n
        mrr = sum((1 / r["rank"]) if r["rank"] else 0 for r in subset) / n
        return {"n": n, "recall@1": round(r1, 4), "recall@3": round(r3, 4),
                "recall@5": round(r5, 4), "mrr": round(mrr, 4)}

    overall = _agg(rows)
    # 分域动态化(方案报告 §26.3 要求分层报告 xling_no_term / xling_para 等子域,
    # 不能只报总平均)。固定四域保留以兼容既有 baseline JSON 的键集。
    _domains = ["llm_app", "backend", "algorithm", "career"]
    _domains += [d for d in dict.fromkeys(r["domain"] for r in rows) if d not in _domains]
    by_domain = {d: _agg([r for r in rows if r["domain"] == d]) for d in _domains}
    hard = _agg([r for r in rows if r["hard"]])
    easy = _agg([r for r in rows if not r["hard"]])
    lat = sorted(r["latency_ms"] for r in rows)
    perf = {"p50_ms": round(lat[len(lat) // 2], 1), "p95_ms": round(lat[min(int(len(lat) * 0.95), len(lat) - 1)], 1),
            "mean_ms": round(sum(lat) / len(lat), 1)} if lat else {}
    diags = [r["diag"] for r in rows if r.get("diag")]
    fw = {}
    if diags:
        _lift = [d["alias_lift_max"] for d in diags if d.get("alias_lift_max") is not None]
        fw = {"english_top1": sum(1 for d in diags if d.get("top1_is_english")),
              "english_false_winner": sum(1 for d in diags if d.get("english_false_winner")),
              "alias_pairs_mean": round(sum(d.get("alias_pairs", 0) for d in diags) / len(diags), 2),
              "alias_applied_mean": round(sum(d.get("alias_applied", 0) for d in diags) / len(diags), 2),
              "alias_lift_median": (sorted(_lift)[len(_lift) // 2] if _lift else None)}
    pools = [r["pool"] for r in rows if r.get("pool")]
    pool_agg = {}
    if pools:
        _fer = [p["first_english_rank"] for p in pools if p["first_english_rank"]]
        _tgt = [p.get("target_rank_in_pool", 0) for p in pools]
        pool_agg = {
            "n_with_pool": len(pools),
            "english_candidate_count_mean": round(
                sum(p["english_candidate_count"] for p in pools) / len(pools), 2),
            "pool_with_zero_english": sum(1 for p in pools if p["english_candidate_count"] == 0),
            "first_english_rank_median": (sorted(_fer)[len(_fer) // 2] if _fer else 0),
            # Candidate Recall:目标在**精排前**池子里的比例——池召回高而最终 R@1 低,
            # 说明瓶颈是排序层(精排/融合),继续加大配额是无效功(报告 §26.1 的用法)
            "candidate_recall": round(sum(1 for t in _tgt if t) / len(_tgt), 4),
            "channels": sorted({c for p in pools for c in p.get("channels", [])}),
        }

    # 门槛准确率（走完整 gated_query）
    from rag_gate import gated_query
    neg = [] if skip_gate else data.get("gate_negatives", [])
    pos = [] if skip_gate else data.get("gate_positives", [])
    neg_ok = sum(1 for q in neg if gated_query(q)["in_kb"] is False)
    pos_ok = sum(1 for q in pos if gated_query(q)["in_kb"] is True)
    gate = {"neg_reject": f"{neg_ok}/{len(neg)}", "pos_hit": f"{pos_ok}/{len(pos)}",
            "neg_acc": round(neg_ok / max(len(neg), 1), 4),
            "pos_acc": round(pos_ok / max(len(pos), 1), 4)}

    return {"top_k": top_k, "overall": overall, "by_domain": by_domain,
            "hard": hard, "easy": easy, "gate": gate, "perf": perf,
            "pool": pool_agg, "false_winner": fw, "rows": rows}


def _fmt(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "n=0"
    return (f"n={m['n']}  R@1={m['recall@1']:.0%}  R@3={m['recall@3']:.0%}  "
            f"R@5={m['recall@5']:.0%}  MRR={m['mrr']:.3f}")


def _diff(cur: dict, base: dict) -> str:
    if not base or base.get("n", 0) == 0:
        return ""
    out = []
    for k in ("recall@1", "recall@3", "recall@5", "mrr"):
        if k in cur and k in base:
            d = cur[k] - base[k]
            arrow = "→" if abs(d) < 1e-9 else ("↑" if d > 0 else "↓")
            out.append(f"{k.split('@')[-1] if '@' in k else k}{arrow}{d:+.3f}")
    return "  Δ[" + " ".join(out) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--save", help="把本次结果存为基线 JSON")
    ap.add_argument("--baseline", help="读取基线 JSON 并打印 Δ 对比")
    ap.add_argument("--label", default="", help="本轮标签（写入保存的 JSON）")
    ap.add_argument("--fail-under", type=float,
                    help="[B3] 总体 R@1 低于此值则退出码 1（CI / pre-push 回归门禁）")
    ap.add_argument("--verbose", action="store_true", help="[问题4] 逐题打印进度（默认显示覆盖式计数）")
    ap.add_argument("--set", dest="set_path",
                    help="评测集路径（默认 tests/rag_bench_set.json）。用 rag_bench_paraphrase_set.json "
                         "跑真实口径（口语化 paraphrase，非同分布）")
    ap.add_argument("--no-gate", action="store_true",
                    help="跳过门槛段（gated_query 会调 LLM；配额/消融扫描只看排序时用）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    res = evaluate(args.k, verbose=args.verbose, set_path=args.set_path,
                   skip_gate=args.no_gate)
    base = None
    if args.baseline and os.path.exists(args.baseline):
        base = json.load(open(args.baseline, encoding="utf-8"))

    print(f"=== RAG 基准评测 (K={args.k}, {len(res['rows'])} 题) ===\n")
    bo = (base or {}).get("overall", {})
    print("总体    " + _fmt(res["overall"]) + (_diff(res["overall"], bo) if base else ""))
    print("难题    " + _fmt(res["hard"]) + (_diff(res["hard"], (base or {}).get("hard", {})) if base else ""))
    print("易题    " + _fmt(res["easy"]))
    print("\n--- 分域 ---")
    for d, m in res["by_domain"].items():
        if m.get("n", 0) == 0 and d in ("llm_app", "backend", "algorithm", "career"):
            continue          # 跨语集里这四域为空,不刷屏
        print(f"  {d:<14} " + _fmt(m) + (_diff(m, (base or {}).get("by_domain", {}).get(d, {})) if base else ""))
    if res.get("pool"):
        p = res["pool"]
        print("\n--- 候选池(rerank 前, 报告 §26.1) ---")
        print(f"  英文候选均值 {p['english_candidate_count_mean']}  ·  "
              f"零英文候选的题 {p['pool_with_zero_english']}/{p['n_with_pool']}  ·  "
              f"首个英文块中位排名 {p['first_english_rank_median']}")
        print(f"  通道 {'+'.join(p['channels'])}")
    if res.get("false_winner"):
        f = res["false_winner"]
        print(f"\n--- 英文抢位审计(指导 §16.4) ---")
        print(f"  英文占 Top1 {f['english_top1']} 题 · 其中**抢错** {f['english_false_winner']} 题  ·  "
              f"别名对/题 {f['alias_pairs_mean']}(生效 {f['alias_applied_mean']}) · lift 中位 {f['alias_lift_median']}")
    if res.get("perf"):
        print(f"\n--- 延迟 ---\n  p50 {res['perf']['p50_ms']}ms · p95 {res['perf']['p95_ms']}ms")
    print("\n--- 门槛 ---")
    print(f"  负样本拒答 {res['gate']['neg_reject']} · 正样本命中 {res['gate']['pos_hit']}")
    print("\n--- 未命中(rank=0)的题 ---")
    miss = [r for r in res["rows"] if r["rank"] == 0]
    for r in miss:
        print(f"  ✗ [{r['id']}{'·难' if r['hard'] else ''}] {r['q'][:34]}")
    if not miss:
        print("  （全部命中 top-5）")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        res["_meta"] = {"label": args.label, "k": args.k}
        try:
            from rag_tools import index_fingerprint
            res["_meta"]["index_fingerprint"] = index_fingerprint()  # P0.3:指标绑定索引版本
        except Exception:
            pass
        json.dump(res, open(args.save, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n[已存基线] {args.save}")

    if args.fail_under is not None:  # B3: CI / pre-push 回归门禁
        r1 = res["overall"]["recall@1"]
        if r1 < args.fail_under:
            print(f"\n[CI 门禁] 总体 R@1={r1:.3f} < 阈值 {args.fail_under} → 失败", file=sys.stderr)
            sys.exit(1)
        print(f"\n[CI 门禁] 总体 R@1={r1:.3f} ≥ 阈值 {args.fail_under} → 通过")


if __name__ == "__main__":
    main()
