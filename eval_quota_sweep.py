# -*- coding: utf-8 -*-
"""eval_quota_sweep.py — 方案报告 Phase 2/3:英文配额扫描 + 检索链逐层消融。

进程内换挡(env 惰性读 + 模型进程内缓存)跑多组配置,每组用同一评测集、同一代码路径,
输出报告 §26 要求的分层指标:子域 R@1/R@3、候选池观测、延迟 p50/p95,并做 §26.6 的
McNemar 配对检验(Top1 二元正确性,不是比平均值)。

用法:
  python eval_quota_sweep.py --mode quota  --set tests/rag_bench_xling60_set.json
  python eval_quota_sweep.py --mode ablate --set tests/rag_bench_xling60_set.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

OUT_DIR = os.path.join(BASE, "docs", "rag_eval", "quota")

# Phase 2:英文/混排保险配额扫描(报告 §10.2 待测档位 + §35 停止规则=选达标的最小 K)
QUOTA_ARMS = [
    ("K0_无配额",  {"RAG_EN_QUOTA": "0"}),
    ("K5",         {"RAG_EN_QUOTA": "1", "RAG_EN_QUOTA_K": "5"}),
    ("K8",         {"RAG_EN_QUOTA": "1", "RAG_EN_QUOTA_K": "8"}),
    ("K10",        {"RAG_EN_QUOTA": "1", "RAG_EN_QUOTA_K": "10"}),
    ("K15",        {"RAG_EN_QUOTA": "1", "RAG_EN_QUOTA_K": "15"}),
    ("K20",        {"RAG_EN_QUOTA": "1", "RAG_EN_QUOTA_K": "20"}),
]

# Phase 3:固定配额后逐层消融(报告 §15),定位哪一层开始损伤论文检索
ABLATE_ARMS = [
    ("S0_dense配额",        {"RAG_EN_QUOTA": "1", "RAG_RERANK": "0", "RAG_BM25": "0"}),
    ("S1_+BM25zh",          {"RAG_EN_QUOTA": "1", "RAG_RERANK": "0", "RAG_BM25": "1"}),
    ("S2_+BM25en",          {"RAG_EN_QUOTA": "1", "RAG_RERANK": "0", "RAG_BM25": "1",
                             "RAG_QUOTA_BM25": "1"}),
    ("S3_+Rerank",          {"RAG_EN_QUOTA": "1", "RAG_RERANK": "1", "RAG_BM25": "1",
                             "RAG_QUOTA_BM25": "1"}),
    ("S3b_Rerank无BM25en",  {"RAG_EN_QUOTA": "1", "RAG_RERANK": "1", "RAG_BM25": "1",
                             "RAG_QUOTA_BM25": "0"}),
]

E5_PATH = os.path.expanduser(
    "~/.cache/modelscope/hub/models/intfloat/multilingual-e5-base")

# Phase 4:**统一配额条件下**只换英文分区的 Dense 模型(报告 §19 明令不许拿
# "BGE+配额 vs M3 无配额"比——候选策略必须一致,只留一个变量)。
EMBED_ARMS = [
    ("A_BGE英文分区",  {"RAG_QUOTA_COLLECTION": "kb_paper_bge_v1"}),
    ("B_E5英文分区",   {"RAG_QUOTA_COLLECTION": "kb_paper_e5_v1",
                        "RAG_QUOTA_EMBED_MODEL": E5_PATH,
                        "RAG_QUOTA_EMBED_PREFIX": "query: "}),
]

# 精排序列长度扫描(2026-08-19 剖析驱动):交叉编码器耗时随序列长度**超线性**,
# 25 对 512→256 实测 2498→1182ms。这是整条链的成本大头,杠杆远大于省几个跨语桥对。
SEQ_ARMS = [
    ("seq512_现状", {"RAG_RERANK_MAX_SEQ": "512"}),
    ("seq384",     {"RAG_RERANK_MAX_SEQ": "384"}),
    ("seq320",     {"RAG_RERANK_MAX_SEQ": "320"}),
    ("seq256",     {"RAG_RERANK_MAX_SEQ": "256"}),
]


# Experiment 2(指导文档 §18):初排/精排的排名融合。初排 R@3=84% 而精排后掉到 66%,
# 目标是同时保住"精排提 Top1"与"初排的 Top3 覆盖"。纯排序改造,零额外模型成本。
FUSION_ARMS = [
    ("B0_纯精排(现状)",   {}),
    ("B1_RRF并列打破",    {"RAG_RRF_TIEBREAK": "1"}),
    ("B3_初排×精排RRF",   {"RAG_RANK_FUSION": "rrf"}),
    ("B4a_blend α=0.7",  {"RAG_RANK_FUSION": "blend", "RAG_FUSION_ALPHA": "0.7"}),
    ("B4b_blend α=0.5",  {"RAG_RANK_FUSION": "blend", "RAG_FUSION_ALPHA": "0.5"}),
    ("B4c_blend α=0.3",  {"RAG_RANK_FUSION": "blend", "RAG_FUSION_ALPHA": "0.3"}),
]

# 保底槽位(B3/B4 崩盘后的重新设计):只保护"配额通道冠军"的槽位,不让有偏的全局初排参与打分。
RESERVE_ARMS = [
    ("S0_不保底(现状)", {}),
    ("S2_保底第2位",    {"RAG_RESERVE_SLOT": "2"}),
    ("S3_保底第3位",    {"RAG_RESERVE_SLOT": "3"}),
    ("S5_保底第5位",    {"RAG_RESERVE_SLOT": "5"}),
]

# Experiment 1(§6):最小英文配额。已知 K>=5 足够,尚不知 K=1~4 是否也够。
# Phase 3(指导 §7):**固定最佳别名校准后**再扫最小配额——Concept Alias 强化后
# K=1~4 可能已足够。单变量纪律:只动 K,别名字段与 δ 全程锁死。
KMIN_ARMS = [(f"K{k}", {"RAG_EN_QUOTA_K": str(k),
                        "RAG_CONCEPT_ALIAS": "1",
                        "RAG_ALIAS_FIELDS": "keywords,questions",
                        "RAG_ALIAS_DISCOUNT": "0.03"}) for k in (1, 2, 3, 4, 5)]

# Experiment 3(§19):精排候选数。前提=Candidate Recall 不下降。
POOL_ARMS = [("N全量(25)", {}), ("N20", {"RAG_RERANK_POOL": "20"}),
             ("N16", {"RAG_RERANK_POOL": "16"}), ("N12", {"RAG_RERANK_POOL": "12"}),
             ("N8", {"RAG_RERANK_POOL": "8"})]

# Experiment 4(指导 §17):Concept Alias 字段矩阵。别名走**精排桥**而非 BM25——
# 本项目候选池召回已 100%,纯 BM25 用法是空操作;瓶颈在排序,桥才命中它。
ALIAS_ARMS = [
    ("A0_无别名",        {}),
    ("A1_keywords",     {"RAG_CONCEPT_ALIAS": "1", "RAG_ALIAS_FIELDS": "keywords"}),
    ("A2_summary",      {"RAG_CONCEPT_ALIAS": "1", "RAG_ALIAS_FIELDS": "summary"}),
    ("A3_questions",    {"RAG_CONCEPT_ALIAS": "1", "RAG_ALIAS_FIELDS": "questions"}),
    ("A4_kw+questions", {"RAG_CONCEPT_ALIAS": "1", "RAG_ALIAS_FIELDS": "keywords,questions"}),
    ("A5_全字段",        {"RAG_CONCEPT_ALIAS": "1"}),
]

# Phase 1(指导 §5.1 D1):别名桥分折扣 δ。别名分不再零成本参与竞争。
DISCOUNT_ARMS = [(f"δ={d}", {"RAG_CONCEPT_ALIAS": "1",
                             "RAG_ALIAS_FIELDS": "keywords,questions",
                             "RAG_ALIAS_DISCOUNT": d})
                 for d in ("0", "0.01", "0.02", "0.03", "0.05", "0.08", "0.10")]

# Phase 2(指导 §6):别名作用范围 N —— 只给配额通道前 N 名算别名分。
SCOPE_ARMS = [(f"N={n}", {"RAG_CONCEPT_ALIAS": "1",
                          "RAG_ALIAS_FIELDS": "keywords,questions",
                          "RAG_ALIAS_SCOPE": n})
              for n in ("1", "2", "3", "5")]

MINILM_PATH = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
M3_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3")

# Experiment 5(§20):轻量多语精排。目标=质量接近 v2-m3、延迟接近 base。
RERANKER_ARMS = [
    ("R0_base(现役)",   {}),
    ("R1_mMiniLM",      {"RAG_RERANK_MODEL": MINILM_PATH}),
    ("R2_v2m3(上限)",   {"RAG_RERANK_MODEL": M3_PATH}),
]

_ENV_KEYS = ["RAG_EN_QUOTA", "RAG_EN_QUOTA_K", "RAG_RERANK", "RAG_BM25",
             "RAG_QUOTA_BM25", "RAG_QUOTA_IN_GATE", "RAG_RERANK_POOL",
             "RAG_MAX_CHUNKS_PER_SOURCE", "RAG_QUOTA_COLLECTION",
             "RAG_QUOTA_EMBED_MODEL", "RAG_QUOTA_EMBED_PREFIX", "RAG_RERANK_MODEL",
             "RAG_QUERY_TRANSLATE", "RAG_TRANSLATE_SKIP_IF_EN", "RAG_RERANK_MAX_SEQ",
             "RAG_RANK_FUSION", "RAG_FUSION_ALPHA", "RAG_FUSION_K", "RAG_RRF_TIEBREAK",
             "RAG_RESERVE_SLOT", "RAG_CONCEPT_ALIAS", "RAG_ALIAS_FIELDS",
             "RAG_ALIAS_DISCOUNT", "RAG_ALIAS_LIFT_GATE", "RAG_ALIAS_SCOPE"]


def _apply(env: dict, defaults: dict) -> None:
    """设本臂 env,未提及的旋钮回默认值——防上一臂的设置泄漏到下一臂(污染 A/B)。"""
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in defaults.items():
        os.environ[k] = v
    for k, v in env.items():
        os.environ[k] = v


def mcnemar(a_ranks: list, b_ranks: list) -> dict:
    """Top1 二元正确性的 McNemar 配对检验(报告 §26.6)。精确二项检验,不做正态近似。

    b01 = A 对 B 错,b10 = A 错 B 对;H0: 两者同等。小样本下精确检验才可信。
    """
    from math import comb
    b01 = sum(1 for x, y in zip(a_ranks, b_ranks) if x == 1 and y != 1)
    b10 = sum(1 for x, y in zip(a_ranks, b_ranks) if x != 1 and y == 1)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p": 1.0, "significant": False}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b01": b01, "b10": b10, "p": round(p, 4), "significant": p < 0.05}


def run(mode: str, set_path: str, top_k: int, extra: dict) -> dict:
    import eval_rag_bench
    arms = {"quota": QUOTA_ARMS, "ablate": ABLATE_ARMS, "embed": EMBED_ARMS,
            "seq": SEQ_ARMS, "fusion": FUSION_ARMS, "kmin": KMIN_ARMS,
            "pool": POOL_ARMS, "reranker": RERANKER_ARMS,
            "reserve": RESERVE_ARMS, "alias": ALIAS_ARMS,
            "discount": DISCOUNT_ARMS, "scope": SCOPE_ARMS}[mode]
    defaults = {"RAG_EN_QUOTA_K": "10"}
    if mode in ("embed", "seq", "fusion", "kmin", "pool", "reranker", "reserve", "alias", "discount", "scope"):
        defaults["RAG_EN_QUOTA"] = "1"
    defaults.update(extra)
    results = {}
    for name, env in arms:
        _apply(env, defaults)
        print(f"\n=== 臂 {name}  env={ {k: v for k, v in sorted(env.items())} } ===",
              file=sys.stderr, flush=True)
        res = eval_rag_bench.evaluate(top_k=top_k, set_path=set_path, skip_gate=True)
        results[name] = res
        o = res["overall"]
        print(f"    总体 R@1={o['recall@1']:.1%} R@3={o['recall@3']:.1%} "
              f"p95={res.get('perf', {}).get('p95_ms', 0):.0f}ms", file=sys.stderr, flush=True)
    return results


def report(results: dict, baseline_name: str) -> str:
    """按报告 §26.3 分层输出:论文向合计 + 四个子域 + 中文守卫 + 候选池 + 延迟 + McNemar。"""
    PAPER = ("xling_no_term", "xling_term", "xling_para", "en_en")
    lines = []
    hdr = (f"{'臂':<20} {'论文向R@1':>9} {'论文向R@3':>9} {'无术语':>7} {'改写':>6} "
           f"{'术语':>6} {'英英':>6} {'中文守卫':>8} {'池召回':>7} {'零英文池':>8} {'p95ms':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    base_ranks = None
    for name, res in results.items():
        rows = res["rows"]
        prs = [r for r in rows if r["domain"] in PAPER]
        p1 = sum(1 for r in prs if r["rank"] == 1) / max(len(prs), 1)
        p3 = sum(1 for r in prs if 0 < r["rank"] <= 3) / max(len(prs), 1)
        bd = res["by_domain"]

        def g(d):
            return bd.get(d, {}).get("recall@1", 0.0)
        pool = res.get("pool") or {}
        zero = f"{pool.get('pool_with_zero_english', '-')}/{pool.get('n_with_pool', '-')}" \
            if pool else "-"
        cr = f"{pool['candidate_recall']:.0%}" if pool.get("candidate_recall") is not None else "-"
        lines.append(
            f"{name:<20} {p1:>9.1%} {p3:>9.1%} {g('xling_no_term'):>7.0%} "
            f"{g('xling_para'):>6.0%} {g('xling_term'):>6.0%} {g('en_en'):>6.0%} "
            f"{g('zh_guard'):>8.0%} {cr:>7} {zero:>8} {res.get('perf', {}).get('p95_ms', 0):>7.0f}")
        if name == baseline_name:
            base_ranks = [r["rank"] for r in rows]

    if base_ranks:
        lines.append("")
        lines.append(f"McNemar 配对检验(vs {baseline_name},Top1 二元正确性,精确二项):")
        for name, res in results.items():
            if name == baseline_name:
                continue
            m = mcnemar(base_ranks, [r["rank"] for r in res["rows"]])
            mark = "显著" if m["significant"] else "不显著"
            lines.append(f"  {name:<20} 仅基线对 {m['b01']:>2} / 仅本臂对 {m['b10']:>2} "
                         f" p={m['p']:.4f}  {mark}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quota", "ablate", "embed", "seq", "fusion", "kmin", "pool", "reranker", "reserve", "alias", "discount", "scope"], default="quota")
    ap.add_argument("--set", dest="set_path",
                    default=os.path.join("tests", "rag_bench_xling60_set.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None, help="结果 JSON 落盘路径")
    ap.add_argument("--env", action="append", default=[],
                    help="附加固定 env,形如 KEY=VAL(所有臂共用)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    extra = dict(kv.split("=", 1) for kv in args.env)
    results = run(args.mode, args.set_path, args.k, extra)
    baseline = list(results.keys())[0]
    text = report(results, baseline)
    _title = {"quota": "2 配额扫描", "ablate": "3 逐层消融",
              "embed": "4 统一配额下英文分区模型对比",
              "seq": "5 精排序列长度质量-延迟权衡",
              "fusion": "6 初排/精排排名融合(指导 Exp2)",
              "kmin": "7 最小英文配额 K=1-5(指导 Exp1)",
              "pool": "8 精排候选数(指导 Exp3)",
              "reranker": "9 轻量多语精排对比(指导 Exp5)",
              "reserve": "10 配额冠军保底槽位",
              "alias": "11 Concept Alias 字段矩阵(指导 Exp4)",
              "discount": "12 别名桥分折扣 δ(指导 §5.1 D1)",
              "scope": "13 别名作用范围 N(指导 §6)"}[args.mode]
    print(f"\n=== 方案报告 Phase {_title} ({args.set_path}) ===\n")
    print(text)

    out = args.out or os.path.join(OUT_DIR, f"sweep_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        from rag_tools import index_fingerprint
        fp = index_fingerprint()
    except Exception:
        fp = {}
    json.dump({"mode": args.mode, "set": args.set_path, "fingerprint": fp,
               "fixed_env": extra, "arms": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已存] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
