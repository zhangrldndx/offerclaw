#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 Q1 pilot: canonical terms as a third query representation.

Candidate recall ONLY -- no judge, no gate, no generation (guide §7.5).  Two
arms share nothing but the expansion call: the production channel arm vs the
same arm whose single HyDE call also emits canonical terms used as one extra
dense + one extra lexical query.  Pool stays 28; the original-question distance
contract is untouched (the dense merge recomputes distances to the original
question for every added candidate).

Two beds: 40 authored no-term questions (the target shape) and the 40 Stage-1
pilot positives as the regression bed (standard/natural-leaning authored
questions whose candidate status is known).  GO: +2/40 on the target bed, zero
net regression on the regression bed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = {"decision": "answer", "routes": [{"source": "reference_kb", "operation": "search"}]}


def candidate_rank(question: str, gold_id: str, profile) -> int:
    from rag_gate import retrieve_with_trace

    trace = retrieve_with_trace(question, PLAN, profile, top_k=5)
    ids = [c.chunk_id for c in trace.reranked_candidates[:28]]
    return next((i + 1 for i, c in enumerate(ids) if c == gold_id), 0)


def main() -> None:
    from rag_answerability import grade
    from rag_colloquial_profiles import colloquial_profile

    canary = grade("金丝雀: 这段写了部署步骤吗?", "金丝雀正文,与知识库无关。" * 20, use_cache=False)
    if canary is None:
        print("STAGE2_VERDICT: GATEWAY_DOWN")
        raise SystemExit(2)

    os.environ["RAG_ANSWERABILITY"] = "0"       # candidate formation only
    prod = colloquial_profile("compact32_pool28_hydechan_bm25")
    terms = colloquial_profile("compact32_pool28_hydechan_terms")

    scope: dict = {}
    exec(Path("/tmp/stage2_noterm.py").read_text(encoding="utf-8"), scope)
    noterm = scope["Q"]
    anchors = json.loads((ROOT / "docs/rag_eval/next_stage/STAGE2_ANCHOR_CANDIDATES.json"
                          ).read_text(encoding="utf-8"))["anchors"]
    pilot = json.loads((ROOT / "docs/rag_eval/next_stage/pilot_dataset.json"
                        ).read_text(encoding="utf-8"))["positives"]

    results = {"noterm": [], "regression": []}
    for index, (aidx, question) in enumerate(noterm, 1):
        gold = anchors[aidx]["chunk_id"]
        row = {"query_id": f"s2-{index:02d}", "anchor": aidx,
               "rank_prod": candidate_rank(question, gold, prod),
               "rank_terms": candidate_rank(question, gold, terms)}
        results["noterm"].append(row)
        print(f"[s2-noterm] {index}/40 prod={row['rank_prod']} terms={row['rank_terms']}", flush=True)
    for index, item in enumerate(pilot, 1):
        gold = item["anchor_chunk"]
        row = {"query_id": item["query_id"],
               "rank_prod": candidate_rank(item["question"], gold, prod),
               "rank_terms": candidate_rank(item["question"], gold, terms)}
        results["regression"].append(row)
        print(f"[s2-reg] {index}/40 prod={row['rank_prod']} terms={row['rank_terms']}", flush=True)

    hit = lambda rows, key: sum(1 for r in rows if r[key])
    payload = {"schema_version": "stage2-q1-pilot-v1",
               "noterm_candidate_recall": {"prod": hit(results["noterm"], "rank_prod"),
                                            "terms": hit(results["noterm"], "rank_terms")},
               "regression_candidate_recall": {"prod": hit(results["regression"], "rank_prod"),
                                                "terms": hit(results["regression"], "rank_terms")},
               "rows": results}
    out = ROOT / "docs/rag_eval/next_stage/STAGE2_PILOT_RESULTS.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("noterm_candidate_recall",
                                              "regression_candidate_recall")}, ensure_ascii=False))
    print(f"[s2] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
