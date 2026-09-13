# -*- coding: utf-8 -*-
"""跨职业域泛化评测矩阵——"横向数据支撑"的可复现来源。

回答面试必问「系统只有你自己(AI 方向)能用吗?」:M 个跨域画像 × N 份 JD 全对全
跑匹配引擎,量化跨域行为。**诚实口径(不许含糊)**:
  - 画像全部为**合成**(synthetic,字段结构与真实画像一致,报告里明标);
  - JD 按 source 分层:synthetic=合成;real=公开渠道人工收集的真实原文
    (往 tests/domain_matrix_corpus.json 的 jds 里追加 source=real 条目,重跑即自动分层);
  - 纯规则引擎评测,零 LLM、零网络,确定性可复现。

指标(预登记,跑前写死):
  1. crash_rate 必须 = 0(任意画像×任意 JD 不崩);
  2. 结论合法率必须 = 100%(∈ 三档);
  3. **跨域劫持率必须 = 0**:非对角(画像域≠JD 域)判为"主方向" = 劫持,一例即报;
  4. 对角主方向命中率:同域(画像域=JD 域)应判"主方向",miss 逐条列出如实报;
  5. 确定性:全矩阵跑两遍结果逐字节一致。

用法:python eval_domain_matrix.py   → 打印报告 + 落盘 docs/rag_eval/domain_matrix.json
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CORPUS = os.path.join(BASE, "tests", "domain_matrix_corpus.json")
OUT = os.path.join(BASE, "docs", "rag_eval", "domain_matrix.json")


def run_matrix(profiles: list, jds: list) -> list:
    from match_job import run_match
    cells = []
    for p in profiles:
        for j in jds:
            cell = {"profile": p["persona_id"], "p_domain": p["domain"],
                    "jd": j["jd_id"], "j_domain": j["domain"], "source": j["source"]}
            try:
                rep = run_match(dict(p), j["jd_text"], jd_title=j["title"])
                cell["conclusion"] = rep.conclusion
                cell["direction"] = rep.direction
            except Exception as e:  # 崩溃计入指标,不中断矩阵
                cell["error"] = f"{type(e).__name__}: {e}"
            cells.append(cell)
    return cells


def summarize(cells: list, n_profiles: int, jds: list) -> dict:
    valid = ("当前适合投递", "中长期可转向", "当前暂不建议投递")
    crashes = [c for c in cells if "error" in c]
    bad_conc = [c for c in cells if "error" not in c and c["conclusion"] not in valid]
    diag = [c for c in cells if c["p_domain"] == c["j_domain"]]
    off = [c for c in cells if c["p_domain"] != c["j_domain"]]
    diag_main = [c for c in diag if c.get("direction") == "主方向"]
    hijack = [c for c in off if c.get("direction") == "主方向"]
    dist = {}
    for c in cells:
        if "error" not in c:
            dist[c["conclusion"]] = dist.get(c["conclusion"], 0) + 1
    by_source = {}
    for j in jds:
        by_source[j["source"]] = by_source.get(j["source"], 0) + 1
    return {
        "pairs": len(cells), "profiles": n_profiles, "jds": len(jds),
        "jd_by_source": by_source,                       # 诚实分层:real vs synthetic
        "crash": len(crashes), "invalid_conclusion": len(bad_conc),
        "diagonal_pairs": len(diag), "diagonal_main_hit": len(diag_main),
        "diagonal_misses": [f"{c['profile']}×{c['jd']}→{c.get('direction')}"
                            for c in diag if c.get("direction") != "主方向"],
        "offdiag_pairs": len(off), "hijack": len(hijack),
        "hijack_cases": [f"{c['profile']}×{c['jd']}" for c in hijack],
        "conclusion_dist": dist,
    }


def main() -> dict:
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    profiles, jds = corpus["profiles"], corpus["jds"]
    cells1 = run_matrix(profiles, jds)
    cells2 = run_matrix(profiles, jds)
    deterministic = cells1 == cells2                     # 预登记指标5
    s = summarize(cells1, len(profiles), jds)
    s["deterministic"] = deterministic
    s["honesty"] = ("画像全部合成;JD 分层 " + json.dumps(s["jd_by_source"], ensure_ascii=False)
                    + ";纯规则引擎,零 LLM;真实用户数据不进入公开评测产物")

    print(f"跨域泛化矩阵:{s['profiles']} 画像 × {s['jds']} JD = {s['pairs']} 对"
          f"(JD 来源 {s['jd_by_source']})")
    print(f"  崩溃 {s['crash']} · 非法结论 {s['invalid_conclusion']} · 确定性 {'一致' if deterministic else '不一致!'}")
    print(f"  跨域劫持(非对角判主方向) {s['hijack']}/{s['offdiag_pairs']}"
          + (f"  劫持案例:{s['hijack_cases']}" if s["hijack"] else ""))
    print(f"  对角主方向命中 {s['diagonal_main_hit']}/{s['diagonal_pairs']}"
          + (f"  miss:{s['diagonal_misses']}" if s["diagonal_misses"] else ""))
    print(f"  结论分布 {json.dumps(s['conclusion_dist'], ensure_ascii=False)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": s, "cells": cells1}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[已存] {OUT}")
    return s


if __name__ == "__main__":
    main()
