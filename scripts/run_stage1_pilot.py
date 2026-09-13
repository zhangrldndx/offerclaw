#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 pilot: requirement coverage vs the production grade judge.

All four arms are computed from ONE retrieval per question (same pool, same
reranker scores), so the comparison isolates the ranking signal itself:

  A  grade judge + early exit          (production semantics, replicated)
  B  grade judge, full depth           (cost/diagnostic arm)
  C  requirement coverage, best chunk  (rank_key: full, count, reranker)
  D  C + minimal evidence set (<=2)

The observation arms run twice with salted judge calls (real resamples) --
GO requires the two cold runs to agree in direction.  A gateway canary aborts
without emitting a verdict; a dead gateway must never masquerade as a result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPTH = 12


_CACHE_PATH = ROOT / ".offerclaw" / "req_coverage_cache.json"
_CACHE: dict | None = None
import threading
_LOCK = threading.Lock()


def _cache() -> dict:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            try:
                _CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                _CACHE = {}
    return _CACHE


def flush_cache() -> None:
    with _LOCK:
        if _CACHE is not None:
            tmp = _CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")
            tmp.replace(_CACHE_PATH)     # atomic: a crash never leaves torn JSON


def observe(question: str, contract: dict, chunk: str, salt: str) -> dict | None:
    """Six workers share one in-process cache behind a lock; the file is written
    once, atomically -- per-call rewrites raced each other into torn JSON."""
    import hashlib

    import rag_requirement_coverage as rc
    from rag_gate import _chat

    key_req = "\n".join(f"- {r['id']}: {r['description']}" for r in contract["requirements"])
    prompt = rc._PROMPT.format(question=question, requirements=key_req,
                               chunk=(chunk or "")[:2000])
    digest = hashlib.sha256((rc.PROMPT_SHA256 + prompt + salt).encode()).hexdigest()
    store = _cache()
    with _LOCK:
        if digest in store:
            return store[digest]
    text = _chat([{"role": "user", "content": prompt}], max_tokens=800, temperature=0.0)
    parsed = rc.parse_observation(text, contract)
    if parsed is not None:
        with _LOCK:
            store[digest] = parsed
    return parsed


def main() -> None:
    import rag_requirement_coverage as rc
    from rag_answerability import grade
    from rag_colloquial_profiles import colloquial_profile
    from rag_gate import retrieve_with_trace

    scope: dict = {}
    exec(Path("/tmp/pilot_pos.py").read_text(encoding="utf-8"), scope)
    exec(Path("/tmp/pilot_neg.py").read_text(encoding="utf-8"), scope)
    positives, negatives = scope["P"], scope["N"]
    anchors = json.loads((ROOT / "docs/rag_eval/next_stage/PILOT_ANCHOR_CANDIDATES.json"
                          ).read_text(encoding="utf-8"))["anchors"]

    canary = grade("金丝雀: 这段写了部署步骤吗?", "金丝雀正文,与知识库无关。" * 20, use_cache=False)
    if canary is None:
        print("PILOT_VERDICT: GATEWAY_DOWN")
        raise SystemExit(2)

    os.environ["RAG_ANSWERABILITY"] = "0"           # retrieval without judge reordering
    profile = colloquial_profile("compact32_pool28_hydechan_bm25")
    plan = {"decision": "answer", "routes": [{"source": "reference_kb", "operation": "search"}]}

    results = {"positives": [], "negatives": []}
    for index, (aidx, style, question, contract, scoring) in enumerate(positives, 1):
        gold_id = anchors[aidx]["chunk_id"]
        trace = retrieve_with_trace(question, plan, profile, top_k=5)
        cands = trace.reranked_candidates[:28]
        ids = [c.chunk_id for c in cands]
        docs = [c.document for c in cands]
        scores = [getattr(c, "rerank_score", None) or 0.0 for c in cands]
        head = list(range(min(DEPTH, len(ids))))
        gold_pool_rank = next((i + 1 for i, c in enumerate(ids) if c == gold_id), 0)

        grades = dict(ThreadPoolExecutor(6).map(
            lambda i: (i, grade(question, docs[i])), head))
        def grade_sort(judged: set):
            def key(i):
                v = grades.get(i) if i in judged else None
                gv = v["grade"] if v else 1.5
                return (-gv, -scores[i])
            return sorted(head, key=key) + list(range(len(head), len(ids)))
        exitA = {head[0]} if (grades.get(0) or {}).get("grade") == 3 else set(head)
        orderA, orderB = grade_sort(exitA), grade_sort(set(head))

        obs = {}
        for salt in ("p1", "p2"):
            obs[salt] = dict(ThreadPoolExecutor(6).map(
                lambda i, s=salt: (i, observe(question, contract, docs[i], s)), head))
        def cov_order(o):
            return sorted(head, key=lambda i: rc.rank_key(o.get(i), contract, scores[i]),
                          reverse=True) + list(range(len(head), len(ids)))
        orderC1, orderC2 = cov_order(obs["p1"]), cov_order(obs["p2"])
        def rank_of(order):
            return next((p + 1 for p, i in enumerate(order) if ids[i] == gold_id), 0)
        setD1 = rc.minimal_evidence_set([obs["p1"].get(i) for i in orderC1[:5]], contract)
        results["positives"].append({
            "query_id": f"pilot-{index:02d}", "style": style, "anchor": aidx,
            "gold_pool_rank": gold_pool_rank,
            "rank_A": rank_of(orderA), "rank_B": rank_of(orderB),
            "rank_C1": rank_of(orderC1), "rank_C2": rank_of(orderC2),
            "D1_set_found": bool(setD1),
            "D1_gold_in_set": any(ids[orderC1[p]] == gold_id for p in setD1),
            "judgeA_calls": len(exitA), "obs_calls": 2 * len(head),
        })
        print(f"[pilot+] {index}/40 pool={gold_pool_rank} A={results['positives'][-1]['rank_A']} "
              f"C1={results['positives'][-1]['rank_C1']} C2={results['positives'][-1]['rank_C2']}",
              flush=True)

    for index, (kind, question, contract) in enumerate(negatives, 1):
        trace = retrieve_with_trace(question, plan, profile, top_k=5)
        cands = trace.reranked_candidates[:DEPTH]
        docs = [c.document for c in cands]
        scores = [getattr(c, "rerank_score", None) or 0.0 for c in cands]
        head = list(range(len(docs)))
        v = grade(question, docs[0]) if docs else None
        acceptAB = bool(v and v.get("grade") == 3 and v.get("relation") in ("entails", "contradicts"))
        accept = {}
        for salt in ("p1", "p2"):
            o = dict(ThreadPoolExecutor(6).map(
                lambda i, s=salt: (i, observe(question, contract, docs[i], s)), head))
            order = sorted(head, key=lambda i: rc.rank_key(o.get(i), contract, scores[i]),
                           reverse=True)
            top = o.get(order[0]) if order else None
            would = bool(top and rc.full_coverage(top, contract)
                         and not (contract.get("referent_required")
                                  and rc.unresolved_reference_vote([o.get(i) for i in head])))
            accept[salt] = would
        results["negatives"].append({"query_id": f"pilot-neg-{index:02d}", "kind": kind,
                                     "accept_AB": acceptAB,
                                     "accept_C1": accept["p1"], "accept_C2": accept["p2"]})
        print(f"[pilot-] {index}/20 {kind:16s} AB={acceptAB} C1={accept['p1']} C2={accept['p2']}",
              flush=True)

    flush_cache()
    out = ROOT / "docs/rag_eval/next_stage/PILOT_RESULTS.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[pilot] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
