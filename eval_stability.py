# -*- coding: utf-8 -*-
"""eval_stability.py — 同配置重复跑 + 逐题翻转分析(指导文档 §3.3 / §4.3 Gate 0)。

**要回答的问题**:某个指标差异是稳定缺陷还是一题级排序噪声?
本项目已实测出噪声底噪 ≈1 题(n=52 时 ±1.9pp),所以单次结果不足以下结论。
判据(指导 §3.3):一题差异按噪声处理,除非**同一题在 ≥3/5 次运行中稳定翻转**。

用法:
  python eval_stability.py --set tests/rag_bench_paraphrase_set.json --runs 5 \
      --arm B0 --arm A4
臂定义见 ARMS;每臂内部跑 N 次,输出每次 R@1 与逐题稳定性。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
OUT_DIR = os.path.join(BASE, "docs", "rag_eval", "quota")

# 指导 §4.1 要冻结的两个 Profile
ARMS = {
    "B0": {},                                    # 中文生产基线:配额关、别名关
    "A4": {"RAG_EN_QUOTA": "1", "RAG_EN_QUOTA_K": "5",
           "RAG_CONCEPT_ALIAS": "1", "RAG_ALIAS_FIELDS": "keywords,questions"},
}

_KEYS = ["RAG_EN_QUOTA", "RAG_EN_QUOTA_K", "RAG_CONCEPT_ALIAS", "RAG_ALIAS_FIELDS",
         "RAG_ALIAS_DISCOUNT", "RAG_ALIAS_LIFT_GATE", "RAG_ALIAS_SCOPE",
         "RAG_QUERY_TRANSLATE", "RAG_EN_TOP1_MARGIN", "RAG_RESERVE_SLOT",
         "RAG_RANK_FUSION", "RAG_RRF_TIEBREAK"]


def _apply(env: dict) -> None:
    for k in _KEYS:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v


def profile_hash(env: dict) -> str:
    """retrieval_profile_hash(指导 §4.1):把决定检索行为的配置钉成一个短哈希。"""
    import hashlib
    from rag_alias import ALIAS_VERSION
    payload = json.dumps({"env": dict(sorted(env.items())),
                          "alias_version": ALIAS_VERSION,
                          "rerank_model": os.environ.get("RAG_RERANK_MODEL", "") or "base",
                          "rerank_max_seq": os.environ.get("RAG_RERANK_MAX_SEQ", "") or "512"},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def run_arm(name: str, env: dict, set_path: str, runs: int, top_k: int) -> dict:
    import eval_rag_bench
    _apply(env)
    ph = profile_hash(env)
    print(f"\n=== 臂 {name}  profile_hash={ph}  ×{runs} ===", file=sys.stderr, flush=True)
    out = []
    for i in range(runs):
        res = eval_rag_bench.evaluate(top_k=top_k, set_path=set_path, skip_gate=True)
        out.append(res)
        o = res["overall"]
        n1 = sum(1 for r in res["rows"] if r["rank"] == 1)
        print(f"  run{i+1}: R@1={o['recall@1']:.1%} ({n1}/{len(res['rows'])}) "
              f"R@3={o['recall@3']:.1%} p50={res['perf']['p50_ms']:.0f}ms",
              file=sys.stderr, flush=True)
    return {"profile_hash": ph, "env": env, "runs": out}


def stability(runs: list) -> dict:
    """逐题 rank==1 的翻转统计:哪些题在多次运行间不稳定。"""
    ids = [r["id"] for r in runs[0]["rows"]]
    hits = {i: [] for i in ids}
    for res in runs:
        for r in res["rows"]:
            hits[r["id"]].append(1 if r["rank"] == 1 else 0)
    flip = {i: v for i, v in hits.items() if 0 < sum(v) < len(v)}
    return {"n_questions": len(ids), "n_unstable": len(flip),
            "unstable": {i: v for i, v in sorted(flip.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_path",
                    default=os.path.join("tests", "rag_bench_paraphrase_set.json"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    arms = a.arm or ["B0", "A4"]
    result = {}
    for name in arms:
        result[name] = run_arm(name, ARMS[name], a.set_path, a.runs, a.k)

    print(f"\n=== 稳定性复跑({a.set_path}, 每臂 {a.runs} 次)===\n")
    print(f"{'臂':<6} {'profile_hash':<14} {'各次 rank1 题数':<26} {'中位':>5} {'不稳定题':>8}")
    print("-" * 66)
    for name, d in result.items():
        counts = [sum(1 for r in res["rows"] if r["rank"] == 1) for res in d["runs"]]
        st = stability(d["runs"])
        d["stability"] = st
        d["counts"] = counts
        print(f"{name:<6} {d['profile_hash']:<14} {str(counts):<26} "
              f"{sorted(counts)[len(counts)//2]:>5} {st['n_unstable']:>8}")
    for name, d in result.items():
        if d["stability"]["unstable"]:
            print(f"\n  [{name}] 跨次翻转的题(1=命中Top1):")
            for qid, v in d["stability"]["unstable"].items():
                print(f"    {qid:<10} {v}")

    # 稳定被英文抢走的题(指导 §4.3 Gate 0 的第二判据)
    for name, d in result.items():
        fw = {}
        for res in d["runs"]:
            for r in res["rows"]:
                if (r.get("diag") or {}).get("english_false_winner"):
                    fw[r["id"]] = fw.get(r["id"], 0) + 1
        if fw:
            stable = {k: v for k, v in fw.items() if v >= max(1, len(d["runs"]) * 3 // 5)}
            print(f"\n  [{name}] 英文抢错的题(出现次数/{len(d['runs'])}):"
                  f" {dict(sorted(fw.items()))}")
            print(f"    其中 ≥3/5 次稳定翻转 = {sorted(stable)} → "
                  f"{'确认为稳定缺陷' if stable else '按噪声处理'}")

    out = a.out or os.path.join(OUT_DIR, "stability.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    slim = {n: {"profile_hash": d["profile_hash"], "env": d["env"],
                "counts": d["counts"], "stability": d["stability"],
                "rows_last": d["runs"][-1]["rows"]} for n, d in result.items()}
    json.dump({"set": a.set_path, "runs": a.runs, "arms": slim},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已存] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
