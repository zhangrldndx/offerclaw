#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report Dev80 metrics under both the frozen V1 qrels and a reviewed overlay.

Why two readings: the Dev80 ranking-failure audit found that several "wrong"
Top-1 chunks in fact satisfy the question's ``answer_requirements`` in full,
i.e. the single-gold V1 qrels understate ranking quality.  Rather than mutate
the frozen V1 gold (which would be adjudicating gold from observed failures),
this tool keeps V1 intact and recomputes the same metrics against a reviewed
overlay, so the two readings can be compared side by side.

Both readings are computed from the *same* frozen retrieval artifact and the
*same* metric primitives in :mod:`rag_eval_metrics` — no retrieval is re-run
and no second metric definition exists.

Grade handling in the reviewed reading:

- ``additions`` with grade 3 join the strict gold set (they count for R@k,
  MRR, nDCG, requirement coverage);
- ``retractions`` to grade 2 join the graded/relevant set only (they count for
  nDCG and context precision, never for strict R@k, and they do not claim
  requirement coverage — a partial chunk cannot make evidence "sufficient").
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_colloquial_v2a import resolve_excerpt  # noqa: E402
from rag_eval_metrics import (  # noqa: E402
    aggregate_ranked_rows,
    channel_exclusivity,
    context_precision,
    duplicate_rate,
    mcnemar_exact_p,
    ndcg,
    rank_of,
    reciprocal_rank,
    requirements_covered,
    wilson_interval,
)
from rag_qrels_v2 import evidence_span_hash  # noqa: E402


REPORT_SCHEMA_VERSION = "colloquial-dev80-dual-qrels-report-v1"
DEFAULT_RESULT = ROOT / "docs/rag_eval/colloquial/dev80_compact32.json"
DEFAULT_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_dev80_v1.json"
DEFAULT_OVERLAY = ROOT / "docs/rag_eval/colloquial/dev80_qrels_review_overlay_20260826.json"
DEFAULT_JSON = ROOT / "docs/rag_eval/colloquial/DEV80_DUAL_QRELS_20260826.json"
DEFAULT_MD = ROOT / "docs/rag_eval/colloquial/DEV80_DUAL_QRELS_20260826.md"


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_overlay(overlay: dict[str, Any], collection: Any) -> dict[str, Any]:
    """Verify every overlay entry is a verbatim slice of the live index."""

    entries: list[dict[str, Any]] = [
        {**row, "kind": "addition"} for row in overlay.get("additions", [])
    ] + [
        {**row, "kind": "retraction"} for row in overlay.get("retractions", [])
    ]
    ids = sorted({row["chunk_id"] for row in entries})
    snapshot = collection.get(ids=ids, include=["documents", "metadatas"])
    documents = {
        str(chunk_id): (str(document or ""), dict(metadata or {}))
        for chunk_id, document, metadata in zip(
            snapshot.get("ids") or [],
            snapshot.get("documents") or [],
            snapshot.get("metadatas") or [],
        )
    }
    missing = sorted(set(ids) - set(documents))
    if missing:
        raise SystemExit(f"overlay references chunks absent from the index: {missing}")
    resolved: list[dict[str, Any]] = []
    for row in entries:
        document, metadata = documents[row["chunk_id"]]
        excerpt = resolve_excerpt(row, document)
        if str(metadata.get("source") or "") != row["source"]:
            raise SystemExit(
                f"{row['chunk_id']}: overlay source does not match the index"
            )
        resolved.append({
            **row,
            "evidence_excerpt": excerpt,
            "evidence_span_hash": evidence_span_hash(excerpt),
        })
    return {**overlay, "resolved_entries": resolved}


def _target_maps(
    item: dict[str, Any],
    overlay_by_anchor: dict[str, list[dict[str, Any]]] | None,
) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Grades and requirement support for one question under one reading."""

    grades: dict[str, int] = {}
    requirements: dict[str, set[str]] = defaultdict(set)
    for target in item["relevant_targets"]:
        chunk_id = str(target["chunk_id"])
        grades[chunk_id] = max(grades.get(chunk_id, 0), int(target["relevance_grade"]))
        requirements[chunk_id].update(target["supported_requirements"])
    for row in (overlay_by_anchor or {}).get(item["anchor_id"], []):
        chunk_id = str(row["chunk_id"])
        grade = int(row["relevance_grade"])
        grades[chunk_id] = max(grades.get(chunk_id, 0), grade)
        # Only a full-coverage (grade 3) chunk may claim the requirement; a
        # partial chunk must not be able to turn evidence into "sufficient".
        if grade == 3:
            requirements[chunk_id].update(item["answer_requirements"])
    return grades, dict(requirements)


def _recompute_row(
    row: dict[str, Any],
    item: dict[str, Any],
    overlay_by_anchor: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    grades, requirements_by_id = _target_maps(item, overlay_by_anchor)
    grade3 = {chunk_id for chunk_id, grade in grades.items() if grade == 3}
    relevant = set(grades)
    required = set(item["answer_requirements"])
    dense_ids = list(row.get("dense_chunk_ids") or [])
    bm25_ids = list(row.get("bm25_chunk_ids") or [])
    fusion_ids = list(row.get("fusion_chunk_ids") or [])
    final_ids = list(row.get("final_chunk_ids") or [])
    reranked_ids = list(row.get("reranked_chunk_ids") or [])
    dense_rank = rank_of(dense_ids, grade3)
    bm25_rank = rank_of(bm25_ids, grade3)
    union_ranks = [rank for rank in (dense_rank, bm25_rank) if rank]
    final_rank = rank_of(final_ids, grade3)
    covered = requirements_covered(final_ids, requirements_by_id, required, 5)
    return {
        "query_id": row["query_id"],
        "anchor_id": row["anchor_id"],
        "domain": row.get("domain", ""),
        "query_style": row["query_style"],
        "grade3_chunk_ids": sorted(grade3),
        "dense_chunk_ids": dense_ids,
        "bm25_chunk_ids": bm25_ids,
        "fusion_chunk_ids": fusion_ids,
        "reranked_chunk_ids": reranked_ids,
        "final_chunk_ids": final_ids,
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "union_rank": min(union_ranks) if union_ranks else 0,
        "fusion_rank": rank_of(fusion_ids, grade3),
        "reranked_rank": rank_of(reranked_ids, grade3),
        "final_rank": final_rank,
        "reciprocal_rank_at_10": reciprocal_rank(final_ids, grade3, 10),
        "ndcg_at_5": ndcg(final_ids, grades, 5),
        "ndcg_at_10": ndcg(final_ids, grades, 10),
        "context_precision_at_5": context_precision(final_ids, relevant, 5),
        "requirement_coverage_at_5": len(covered) / len(required) if required else 0.0,
        "sufficient_evidence_at_5": bool(required and covered == required),
        "duplicate_rate_at_5": duplicate_rate(final_ids, 5),
        "gate_decision": bool(row.get("gate_decision")),
        "correct_top1_gate_pass": bool(final_rank == 1 and row.get("gate_decision")),
        "effective_evidence": bool(final_rank == 1 and row.get("gate_decision")),
        "latency_ms": float(row.get("latency_ms") or 0.0),
        "latency_by_stage": dict(row.get("latency_by_stage") or {}),
    }


def _stage_recalls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for stage in ("dense", "bm25", "union", "fusion", "reranked", "final"):
        metrics = {}
        for k in (1, 3, 5, 10, 20):
            if stage == "union":
                hits = sum(
                    bool((set((row["dense_chunk_ids"])[:k])
                          | set((row["bm25_chunk_ids"])[:k]))
                         & set(row["grade3_chunk_ids"]))
                    for row in rows
                )
            else:
                hits = sum(
                    bool(set(row[f"{stage}_chunk_ids"][:k]) & set(row["grade3_chunk_ids"]))
                    for row in rows
                )
            metrics[f"recall@{k}"] = wilson_interval(hits, len(rows))
        output[stage] = metrics
    return output


def _reading(
    runs: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    overlay_by_anchor: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in runs:
        rows = [
            _recompute_row(row, case_by_id[row["query_id"]], overlay_by_anchor)
            for row in run["positive"]["rows"]
        ]
        per_run.append({
            "metrics": {
                **aggregate_ranked_rows(rows),
                "retriever_and_fusion": _stage_recalls(rows),
                "channel_exclusivity": channel_exclusivity(rows),
            },
            "rows": rows,
        })
    top1_by_run = [
        {row["query_id"]: int(row["final_rank"] or 0) == 1 for row in run["rows"]}
        for run in per_run
    ]
    stable = all(values == top1_by_run[0] for values in top1_by_run)
    return {
        "runs": len(per_run),
        "top1_identical_across_runs": stable,
        "metrics": per_run[0]["metrics"],
        "rows": per_run[0]["rows"],
        "per_run_r1": [
            sum(1 for hit in values.values() if hit) for values in top1_by_run
        ],
    }


def build_report(
    result: dict[str, Any],
    cases: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    case_by_id = {item["query_id"]: item for item in cases["items"]}
    overlay_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlay["resolved_entries"]:
        overlay_by_anchor[row["anchor_id"]].append(row)

    original = _reading(result["runs"], case_by_id, None)
    reviewed = _reading(result["runs"], case_by_id, dict(overlay_by_anchor))

    original_hit = {row["query_id"]: int(row["final_rank"] or 0) == 1
                    for row in original["rows"]}
    reviewed_hit = {row["query_id"]: int(row["final_rank"] or 0) == 1
                    for row in reviewed["rows"]}
    original_rank = {row["query_id"]: int(row["final_rank"] or 0)
                     for row in original["rows"]}
    reviewed_rank = {row["query_id"]: int(row["final_rank"] or 0)
                     for row in reviewed["rows"]}
    changed = [
        {
            "query_id": query_id,
            "anchor_id": case_by_id[query_id]["anchor_id"],
            "query_style": case_by_id[query_id]["query_style"],
            "rank_original": original_rank[query_id],
            "rank_reviewed": reviewed_rank[query_id],
            "becomes_top1": reviewed_hit[query_id] and not original_hit[query_id],
        }
        for query_id in sorted(original_rank)
        if original_rank[query_id] != reviewed_rank[query_id]
    ]
    still_failing = [
        {
            "query_id": row["query_id"],
            "anchor_id": row["anchor_id"],
            "query_style": row["query_style"],
            "rank": int(row["final_rank"] or 0),
        }
        for row in reviewed["rows"] if int(row["final_rank"] or 0) != 1
    ]
    wins = sum(1 for query_id in original_hit
               if reviewed_hit[query_id] and not original_hit[query_id])
    losses = sum(1 for query_id in original_hit
                 if original_hit[query_id] and not reviewed_hit[query_id])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "purpose": "diagnosis_only_v1_gold_unchanged",
        "source": {
            "retrieval_arm": result["configuration"]["retrieval_arm"],
            "dataset_id": result["input"]["dataset_id"],
            "index": result["input"]["index"],
            "runs": len(result["runs"]),
        },
        "overlay": {
            "status": overlay.get("status"),
            "additions": [
                {
                    "anchor_id": row["anchor_id"],
                    "chunk_id": row["chunk_id"],
                    "relevance_grade": row["relevance_grade"],
                    "verdict": row["verdict"],
                    "evidence_span_hash": row["evidence_span_hash"],
                }
                for row in overlay["resolved_entries"] if row["kind"] == "addition"
            ],
            "retractions": [
                {
                    "anchor_id": row["anchor_id"],
                    "chunk_id": row["chunk_id"],
                    "from": row["from"],
                    "to": row["to"],
                    "relevance_grade": row["relevance_grade"],
                    "evidence_span_hash": row["evidence_span_hash"],
                }
                for row in overlay["resolved_entries"] if row["kind"] == "retraction"
            ],
        },
        "readings": {
            "original": {
                "top1_identical_across_runs": original["top1_identical_across_runs"],
                "per_run_r1": original["per_run_r1"],
                "metrics": original["metrics"],
            },
            "reviewed": {
                "top1_identical_across_runs": reviewed["top1_identical_across_runs"],
                "per_run_r1": reviewed["per_run_r1"],
                "metrics": reviewed["metrics"],
            },
        },
        "delta": {
            "wins": wins,
            "losses": losses,
            "mcnemar_exact_p": mcnemar_exact_p(wins, losses),
            "changed_rank_rows": changed,
            "still_failing_under_reviewed": still_failing,
            "still_failing_count": len(still_failing),
        },
    }


def _pick(metrics: dict[str, Any], *path: str) -> Any:
    node: Any = metrics
    for key in path:
        node = node[key]
    return node


def render_markdown(report: dict[str, Any], *, hashes: dict[str, str]) -> str:
    original = report["readings"]["original"]["metrics"]
    reviewed = report["readings"]["reviewed"]["metrics"]

    def row(label: str, *path: str, fmt: str = "count") -> str:
        left, right = _pick(original, *path), _pick(reviewed, *path)
        if fmt == "count":
            return (f"| {label} | {left['hits']}/{left['n']} | {right['hits']}/{right['n']} | "
                    f"{right['hits'] - left['hits']:+d} |")
        return f"| {label} | {left:.6f} | {right:.6f} | {right - left:+.6f} |"

    lines = [
        "# Dev80 双口径指标：原口径 vs 修正口径（compact32，2026-08-26）",
        "",
        "> **V1 金标文件未被修改**。修正口径来自 `dev80_qrels_review_overlay_20260826.json`，",
        "> 是对 17 个失败题中每一个“压在金标之上的候选”按 answer_requirements 逐条复核的结果，",
        "> 仅用于判断真实瓶颈规模；候选晋级与 Blind 判定仍以原口径为准，正式并入 V1 前需独立双审。",
        "",
        f"- 冻结产物：`{Path(hashes['result_path']).name}`（`{hashes['result_sha256']}`）",
        f"- Dev 用例：`{Path(hashes['cases_path']).name}`（`{hashes['cases_sha256']}`）",
        f"- 复核 overlay：`{Path(hashes['overlay_path']).name}`（`{hashes['overlay_sha256']}`）",
        "",
        "## 复核结论",
        "",
        "| 锚点 | 候选块 | 裁定 | 效果 |",
        "|---|---|---|---|",
    ]
    for entry in report["overlay"]["additions"]:
        lines.append(
            f"| {entry['anchor_id']} | `{entry['chunk_id']}` | "
            f"grade {entry['relevance_grade']}（{entry['verdict']}） | 计入修正口径金标 |"
        )
    for entry in report["overlay"]["retractions"]:
        lines.append(
            f"| {entry['anchor_id']} | `{entry['chunk_id']}` | "
            f"{entry['from']} → {entry['to']}（grade {entry['relevance_grade']}） | "
            "撤回负例；不计入 strict R@k |"
        )
    lines.extend([
        "",
        "## 指标对照",
        "",
        "| 指标 | 原口径 | 修正口径 | Δ |",
        "|---|---:|---:|---:|",
        row("strict R@1", "strict_ranking", "recall@1"),
        row("strict R@3", "strict_ranking", "recall@3"),
        row("strict R@5", "strict_ranking", "recall@5"),
        row("MRR@10", "strict_ranking", "mrr@10", fmt="float"),
        row("nDCG@5", "strict_ranking", "ndcg@5", fmt="float"),
        row("nDCG@10", "strict_ranking", "ndcg@10", fmt="float"),
        row("Union Candidate@20", "funnel", "union_candidate"),
        row("RRF Candidate@20", "funnel", "rrf_candidate"),
        row("Candidate→Top1", "funnel", "candidate_to_top1"),
        row("正确 Top1 的 Gate 通过", "funnel", "correct_top1_gate_pass"),
        row("Effective Evidence", "funnel", "effective_evidence"),
        row("Sufficient Evidence@5", "context", "sufficient_evidence@5"),
        "",
        f"- Reranker 促进/伤害：原口径 {original['reranker']['positive_promoted']}/"
        f"{original['reranker']['positive_harmed']}，修正口径 "
        f"{reviewed['reranker']['positive_promoted']}/{reviewed['reranker']['positive_harmed']}",
        f"- 三轮 Top1 一致：原口径 {report['readings']['original']['top1_identical_across_runs']}，"
        f"修正口径 {report['readings']['reviewed']['top1_identical_across_runs']}"
        f"（逐轮 R@1：{report['readings']['original']['per_run_r1']} / "
        f"{report['readings']['reviewed']['per_run_r1']}）",
        f"- 逐题变化：wins {report['delta']['wins']}，losses {report['delta']['losses']}，"
        f"McNemar exact p={report['delta']['mcnemar_exact_p']}",
        "",
        "## 修正口径下仍然失败的题（真正的排序攻坚目标）",
        "",
        f"共 {report['delta']['still_failing_count']} 题：",
        "",
        "| query_id | 锚点 | 问法 | 修正口径 rank |",
        "|---|---|---|---:|",
    ])
    for entry in report["delta"]["still_failing_under_reviewed"]:
        lines.append(
            f"| `{entry['query_id']}` | {entry['anchor_id']} | {entry['query_style']} | "
            f"{entry['rank'] or '未进入候选'} |"
        )
    lines.extend(["", "## 排名发生变化的题", "",
                  "| query_id | 原 rank | 修正 rank | 变为 Top1 |", "|---|---:|---:|---|"])
    for entry in report["delta"]["changed_rank_rows"]:
        lines.append(
            f"| `{entry['query_id']}` | {entry['rank_original']} | {entry['rank_reviewed']} | "
            f"{'是' if entry['becomes_top1'] else '否'} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    import chromadb
    from rag_tools import get_collection_name

    result_path = args.result.expanduser().resolve()
    cases_path = args.cases.expanduser().resolve()
    overlay_path = args.overlay.expanduser().resolve()
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    overlay = resolve_overlay(
        json.loads(overlay_path.read_text(encoding="utf-8")), collection,
    )
    report = build_report(
        json.loads(result_path.read_text(encoding="utf-8")),
        json.loads(cases_path.read_text(encoding="utf-8")),
        overlay,
    )
    hashes = {
        "result_path": str(result_path), "result_sha256": _sha256_file(result_path),
        "cases_path": str(cases_path), "cases_sha256": _sha256_file(cases_path),
        "overlay_path": str(overlay_path), "overlay_sha256": _sha256_file(overlay_path),
    }
    report["inputs"] = hashes
    json_path = args.output_json.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    args.output_md.expanduser().resolve().write_text(
        render_markdown(report, hashes=hashes), encoding="utf-8",
    )
    strict = report["readings"]
    print(json.dumps({
        "output_json": str(json_path),
        "output_md": str(args.output_md.expanduser().resolve()),
        "original_r1": strict["original"]["metrics"]["strict_ranking"]["recall@1"],
        "reviewed_r1": strict["reviewed"]["metrics"]["strict_ranking"]["recall@1"],
        "delta": {key: report["delta"][key] for key in
                  ("wins", "losses", "mcnemar_exact_p", "still_failing_count")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
