#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 0: read-only failure bucketing of Final v4 (NEXT_STAGE_GUIDE §7).

Read-only means: no prompt change, no profile change, no label rewrite, no LLM
call.  The only index access is a deterministic original-question probe at
depth 60, used to split out-of-pool misses into fusion-budget vs deep misses.

Privacy contract: per-run rows live in ``~/.offerclaw/private_eval``; this
audit publishes only query ids, ranks, counts, enum buckets, float coverages
and sha256 digests -- never question text, chunk text or judge reasons.

Bucket assignment is deterministic and its *basis* is recorded per row, so a
disputed bucket can be re-derived without re-running anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POSITIVE_BUCKETS = (
    "candidate_no_term", "candidate_single_channel_budget",
    "candidate_true_corpus_gap", "candidate_route_scope",
    "rank_same_source_incomplete", "rank_multi_requirement",
    "rank_cross_source_false_winner", "rank_equivalent_qrels",
)
NEGATIVE_BUCKETS = ("gate_numeric_missing_slot", "gate_unresolved_reference",
                    "relation_policy_dispute")

_NUMERIC_RE = re.compile(r"多少|几[个门张条步位名类种档次轮]|价格|字符|百分|分数|时长|参数量|多大|多久")
_NOREF_RE = re.compile(r"(那个东西|那事儿|这几个|这俩|上回|上次|照旧|之前那样)")
_ROLE_RE = re.compile(r"是不是(负责|用来|干)|是不是.{0,12}的[？?]")


def _grams(text: str) -> set:
    text = text.lower()
    out = set(re.findall(r"[a-z_0-9]{2,}", text))
    for run in re.findall(r"[一-鿿]+", text):
        out |= ({run[i:i + 2] for i in range(len(run) - 1)} if len(run) > 1 else {run})
    return out


def coverage(requirements: list, text: str) -> float:
    req = _grams(" ".join(requirements))
    return len(req & _grams(text)) / max(1, len(req))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="~/.offerclaw/private_eval/final_v4/runs")
    parser.add_argument("--dataset", default="docs/rag_eval/final_v4/final_v4.json")
    parser.add_argument("--arm", default="C")
    parser.add_argument("--probe-depth", type=int, default=60)
    parser.add_argument("--pool", type=int, default=28)
    parser.add_argument("--output", default="docs/rag_eval/next_stage/STAGE0_BUCKETS.json")
    args = parser.parse_args()

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    items = {i["query_id"]: i for i in dataset["items"]}

    runs_dir = Path(args.runs_dir).expanduser()
    runs = [json.loads((runs_dir / f"{args.arm}_run{i}.json").read_text(encoding="utf-8"))["runs"][0]
            for i in (1, 2, 3)]

    # Deterministic original-question probe (no LLM): dense + bm25 at depth 60.
    import chromadb
    from rag_bm25 import bm25_search
    from rag_retrieval_trace import stable_chunk_id
    from rag_tools import get_collection_name, get_embeddings_batch

    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name())
    chunk_text_cache: dict[str, str] = {}

    def chunk_text(chunk_id: str) -> str:
        if chunk_id not in chunk_text_cache:
            got = collection.get(ids=[chunk_id], include=["documents"])
            docs = got.get("documents") or [""]
            chunk_text_cache[chunk_id] = docs[0] if docs else ""
        return chunk_text_cache[chunk_id]

    def probe_ranks(question: str, gold: set) -> tuple[int, int]:
        emb = get_embeddings_batch([question])
        got = collection.query(query_embeddings=emb, n_results=args.probe_depth,
                               include=["documents", "metadatas"])
        dense_ids = [i or stable_chunk_id(d, m) for i, d, m in zip(
            got["ids"][0], got["documents"][0], got["metadatas"][0])]
        dense = next((r for r, c in enumerate(dense_ids, 1) if c in gold), 0)
        bm = next((r for r, (d, m, _s) in enumerate(bm25_search(question, args.probe_depth) or [], 1)
                   if stable_chunk_id(d, m) in gold), 0)
        return dense, bm

    rows, per_run_miss = [], []
    for run in runs:
        per_run_miss.append({r["query_id"] for r in run["positive"]["rows"] if r["final_rank"] != 1})
    always_missed = set.intersection(*per_run_miss)
    ever_missed = set.union(*per_run_miss)

    base = {r["query_id"]: r for r in runs[0]["positive"]["rows"]}
    for query_id in sorted(ever_missed):
        row = base[query_id]
        item = items[query_id]
        gold = set(row["grade3_chunk_ids"])
        requirements = item["answer_requirements"]
        gold_id = next((c for c in row["reranked_chunk_ids"] if c in gold), None) \
            or (row["grade3_chunk_ids"][0] if row["grade3_chunk_ids"] else "")
        record = {
            "query_id": query_id,
            "query_style": row["query_style"],
            "stable_across_runs": query_id in always_missed,
            "gold_candidate_present": bool(row["reranked_rank"]),
            "gold_final_rank": row["final_rank"],
            "gold_dense_rank": row["dense_rank"],
            "gold_bm25_rank": row["bm25_rank"],
            "gold_rrf_rank": row["fusion_rank"],
            "gold_chunk_sha256": hashlib.sha256(gold_id.encode()).hexdigest()[:16],
        }
        if row["reranked_rank"]:
            winner_id = row["reranked_chunk_ids"][0]
            gold_cov = coverage(requirements, chunk_text(gold_id))
            win_cov = coverage(requirements, chunk_text(winner_id))
            same = winner_id.rsplit("_", 1)[0] == gold_id.rsplit("_", 1)[0]
            gate = row["gate_features"].get("answerability_rerank") or {}
            record.update({
                "false_winner_sha256": hashlib.sha256(winner_id.encode()).hexdigest()[:16],
                "same_source_as_gold": same,
                "gold_covered_requirements": round(gold_cov, 2),
                "winner_covered_requirements": round(win_cov, 2),
                "judge_early_exit": (gate.get("calls", 0) or 0) < (gate.get("depth", 12) or 12),
                "judge_graded_depth": gate.get("graded", 0),
                "n_requirements": len(requirements),
            })
            if win_cov >= gold_cov - 0.10:
                bucket = "rank_equivalent_qrels"
            elif same:
                bucket = "rank_same_source_incomplete"
            elif len(requirements) >= 2 and 0 < win_cov < gold_cov:
                bucket = "rank_multi_requirement"
            else:
                bucket = "rank_cross_source_false_winner"
        else:
            dense60, bm60 = probe_ranks(item["question"], gold)
            term_overlap = round(len(_grams(item["question"]) & _grams(chunk_text(gold_id)))
                                 / max(1, len(_grams(item["question"]))), 2)
            record.update({"probe_dense60": dense60, "probe_bm2560": bm60,
                           "question_gold_term_overlap": term_overlap})
            reachable_pool = (0 < dense60 <= args.pool) or (0 < bm60 <= args.pool)
            reachable_60 = dense60 or bm60
            if reachable_pool:
                bucket = "candidate_single_channel_budget"
            elif reachable_60 and term_overlap < 0.25:
                bucket = "candidate_no_term"
            elif not reachable_60 and term_overlap < 0.25:
                bucket = "candidate_no_term"
            elif not reachable_60:
                bucket = "candidate_true_corpus_gap"
            else:
                bucket = "candidate_true_corpus_gap"
        record["failure_bucket"] = bucket
        rows.append(record)

    # Negatives: false accepts on must-abstain, bucketed by question shape.
    neg_rows = []
    for run_index, run in enumerate(runs, 1):
        for r in run["negative"]["rows"]:
            item = items[r["query_id"]]
            if item.get("expected_behavior") != "abstain_from_kb" or not r.get("gate_decision"):
                continue
            q = item["question"]
            if _NOREF_RE.search(q):
                bucket = "gate_unresolved_reference"
            elif _NUMERIC_RE.search(q):
                bucket = "gate_numeric_missing_slot"
            elif _ROLE_RE.search(q):
                bucket = "relation_policy_dispute"
            else:
                bucket = "gate_numeric_missing_slot"   # missing-data 默认并入数字/事实槽缺失
            neg_rows.append({"query_id": r["query_id"], "run": run_index,
                             "failure_bucket": bucket})

    from collections import Counter
    pos_counts = Counter(r["failure_bucket"] for r in rows)
    stable_counts = Counter(r["failure_bucket"] for r in rows if r["stable_across_runs"])
    payload = {
        "schema_version": "final-v4-stage0-buckets-v1",
        "arm": args.arm, "runs": 3,
        "positive_misses_ever": len(rows),
        "positive_misses_stable": sum(1 for r in rows if r["stable_across_runs"]),
        "bucket_counts_ever": dict(pos_counts),
        "bucket_counts_stable": dict(stable_counts),
        "negative_false_accepts": neg_rows,
        "rows": rows,
        "note": ("只读审计:无 LLM 调用;池外拆分依据=原问题确定性探针@60+词面重叠;"
                 "桶判定依据逐行记录可复核。Final v4 已解封,本审计只定桶大小,不选参数。"),
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[stage0] 未命中(任一跑) {len(rows)} / 稳定 {payload['positive_misses_stable']}")
    for bucket in POSITIVE_BUCKETS:
        print(f"  {bucket:36s} ever={pos_counts.get(bucket,0):2d} stable={stable_counts.get(bucket,0):2d}")
    print(f"[stage0] 负例误纳记录 {len(neg_rows)} 条(跑×题)")
    print(f"[stage0] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
