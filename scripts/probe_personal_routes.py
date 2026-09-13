#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5 first measurement: the personal vector routes' retrieval quality.

Nobody has ever measured retrieval through ``project_memory`` or
``resume_rules`` -- every blind set only exercised ``reference_kb``.  The 22
Final v2/v3 questions whose golds live in project_context/resume/jd were
scored as unreachable through the forced reference plan; here they finally run
through the route that actually serves them.  jd golds have no vector route
(recorded as a finding, not measured).

Read-only: production profile, no knob changes, no LLM beyond what the
production pipeline itself calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUTE_BY_SOURCE_TYPE = {"project_context": "project_memory",
                        "resume": "resume_rules", "resume_rule": "resume_rules"}


def main() -> None:
    import chromadb
    from rag_gate import retrieve_with_trace
    from rag_tools import get_collection_name

    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name())
    audit = json.loads((ROOT / "docs/rag_eval/final_v3/ROUTE_SCOPE_AUDIT.json"
                        ).read_text(encoding="utf-8"))
    items = {}
    for name in ("final_v2/final_v2.json", "final_v3/final_v3.json"):
        data = json.loads((ROOT / "docs/rag_eval" / name).read_text(encoding="utf-8"))
        items.update({i["query_id"]: i for i in data["items"]})

    targets = []
    for qid in audit["final_v2_unreachable"] + audit["final_v3_unreachable"]:
        item = items[qid]
        golds = [t["chunk_id"] for t in item["relevant_targets"]]
        metas = collection.get(ids=golds, include=["metadatas"]).get("metadatas") or []
        st = next(((m or {}).get("source_type") for m in metas), "?")
        scope = next(((m or {}).get("owner_scope") for m in metas), "?")
        targets.append({"query_id": qid, "question": item["question"],
                        "golds": golds, "source_type": st, "owner_scope": scope,
                        "route": ROUTE_BY_SOURCE_TYPE.get(st)})

    rows = []
    for t in targets:
        if not t["route"]:
            rows.append({**{k: t[k] for k in ("query_id", "source_type", "route")},
                         "note": "no_vector_route"})
            print(f"[route5] {t['query_id']:22s} {t['source_type']:16s} 无向量路由,跳过", flush=True)
            continue
        plan = {"decision": "answer",
                "routes": [{"source": t["route"], "operation": "search"}]}
        trace = retrieve_with_trace(t["question"], plan, None, top_k=5)
        pool = [c.chunk_id for c in trace.reranked_candidates]
        final = [c.chunk_id for c in trace.final_candidates]
        gold = set(t["golds"])
        row = {"query_id": t["query_id"], "route": t["route"],
               "source_type": t["source_type"], "owner_scope": t["owner_scope"],
               "pool_size": len(pool),
               "pool_rank": next((i + 1 for i, c in enumerate(pool) if c in gold), 0),
               "final_rank": next((i + 1 for i, c in enumerate(final) if c in gold), 0),
               "gate_decision": bool(trace.gate_decision)}
        rows.append(row)
        print(f"[route5] {t['query_id']:22s} {t['route']:14s} pool_rank={row['pool_rank']:>2} "
              f"final_rank={row['final_rank']} gate={row['gate_decision']} pool={row['pool_size']}", flush=True)

    measured = [r for r in rows if "note" not in r]
    summary = {"n_measured": len(measured),
               "candidate_recall": sum(1 for r in measured if r["pool_rank"]),
               "r1": sum(1 for r in measured if r["final_rank"] == 1),
               "r3": sum(1 for r in measured if 0 < r["final_rank"] <= 3),
               "gate_pass_on_r1": sum(1 for r in measured
                                      if r["final_rank"] == 1 and r["gate_decision"]),
               "no_vector_route": sum(1 for r in rows if r.get("note"))}
    out = ROOT / "docs/rag_eval/next_stage/STAGE5_PERSONAL_ROUTES_BASELINE.json"
    out.write_text(json.dumps({"schema_version": "stage5-personal-routes-v1",
                               "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[route5] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
