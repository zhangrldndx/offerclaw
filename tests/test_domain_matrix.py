# -*- coding: utf-8 -*-
"""跨域泛化矩阵回归:钉死 eval_domain_matrix 的现状快照(2026-07-12 首测)。

纪律:数字变化必须是**有意识的**(模板/规则改动后在此同步),不许静默漂移;
已知缺陷(劫持 1 例)与已知边界(对角 5 miss=命名变体 fail-safe)入档不粉饰。
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import eval_domain_matrix as edm


def _summary():
    corpus = json.load(open(edm.CORPUS, encoding="utf-8"))
    cells = edm.run_matrix(corpus["profiles"], corpus["jds"])
    return edm.summarize(cells, len(corpus["profiles"]), corpus["jds"]), corpus


def test_corpus_contract():
    corpus = json.load(open(edm.CORPUS, encoding="utf-8"))
    pids = [p["persona_id"] for p in corpus["profiles"]]
    jids = [j["jd_id"] for j in corpus["jds"]]
    assert len(set(pids)) == len(pids) and len(set(jids)) == len(jids)
    assert all(j["source"] in ("real", "synthetic") for j in corpus["jds"])
    assert all(p.get("方向优先级") for p in corpus["profiles"])


def test_matrix_hard_invariants():
    s, _ = _summary()
    assert s["crash"] == 0                      # 任意画像×任意 JD 不崩
    assert s["invalid_conclusion"] == 0         # 结论恒∈三档
    assert s["pairs"] == s["profiles"] * s["jds"]


def test_matrix_snapshot_pinned():
    """现状快照(首测 2026-07-12):劫持 1(已知缺陷:legacy AI 宽匹配)、对角 11/16
    (5 miss=命名变体 fail-safe 判不考虑)。改模板/规则后数字变了→有意识地更新这里。"""
    s, _ = _summary()
    assert s["hijack"] == 1
    assert s["hijack_cases"] == ["dm_ai_llm×jd_pd_2"]
    assert s["diagonal_main_hit"] == 11 and s["diagonal_pairs"] == 16
    assert all("不考虑" in m for m in s["diagonal_misses"])   # miss 全是保守判定,非误归类


def test_matrix_deterministic():
    corpus = json.load(open(edm.CORPUS, encoding="utf-8"))
    a = edm.run_matrix(corpus["profiles"], corpus["jds"])
    b = edm.run_matrix(corpus["profiles"], corpus["jds"])
    assert a == b
