#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine real production false winners from the public colloquial Train split.

``mine`` runs the frozen production-mirror ``compact32`` chain over every
positive Train question and stores compact per-stage Top-20 traces.  ``report``
turns those traces into adjudication-gated candidate groups plus the data
difficulty report required before any F3 training build.

Contract highlights:

- only ``split=train`` rows from the public V1 file are accepted; Dev/Blind
  input is rejected (Dev failures are diagnosis-only, Blind is sealed);
- retrieval reuses :func:`rag_gate.retrieve_with_trace` with the frozen
  ``compact32`` arm — no second retrieval implementation;
- a mined candidate never becomes a hard negative automatically.  Candidates
  start as ``needs_adjudication`` (or inherit the V1 adjudication) and only an
  explicit reviewed verdict in the adjudication overlay may relabel them;
- candidates that might partially support the answer are Grade-1/2 or stay
  ``needs_adjudication`` and are counted as excluded, never as negatives.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MINING_SCHEMA_VERSION = "colloquial-train-false-winner-mining-v1"
REPORT_SCHEMA_VERSION = "colloquial-train-false-winner-report-v1"
DEFAULT_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json"
DEFAULT_TRACES = ROOT / "docs/rag_eval/colloquial/train_top20_traces_20260826.json"
DEFAULT_ADJUDICATION = (
    ROOT / "docs/rag_eval/colloquial/train_false_winner_adjudication_20260826.json"
)
DEFAULT_REPORT_JSON = (
    ROOT / "docs/rag_eval/colloquial/TRAIN_FALSE_WINNER_REPORT_20260826.json"
)
DEFAULT_REPORT_MD = (
    ROOT / "docs/rag_eval/colloquial/TRAIN_FALSE_WINNER_REPORT_20260826.md"
)

REFERENCE_PLAN = {
    "decision": "answer",
    "routes": [{"source": "reference_kb", "operation": "search"}],
}

# Statuses a mined candidate may carry.  Only ``hard_negative`` is eligible for
# future training pairs; everything else is excluded (and counted).
CANDIDATE_STATUSES = {
    "hard_negative",
    "grade1_partial",
    "grade2_partial",
    "equivalent_grade3_candidate",
    "needs_adjudication",
}
NEAR_MARGIN = 0.10  # |top-candidate score - gold score| window for extra negatives
MAX_EXTRA_CANDIDATES = 2  # per question beyond the wrong Top-1


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def train_cases_fingerprint(positives: list[dict[str, Any]]) -> str:
    """Hash only what mining consumes: the Train positives.

    Keying the staleness guard on the whole cases file made a Dev-only gold
    revision look like a Train change and forced a pointless re-mine.
    """

    payload = [
        {
            "query_id": item["query_id"],
            "anchor_id": item["anchor_id"],
            "question": item["question"],
            "answer_requirements": item["answer_requirements"],
            "relevant_targets": [
                {key: target[key] for key in
                 ("chunk_id", "source", "relevance_grade", "evidence_span_hash")}
                for target in item["relevant_targets"]
            ],
            "hard_negatives": [
                {"chunk_id": negative["chunk_id"], "reason": negative["reason"]}
                for negative in item.get("hard_negatives") or []
            ],
        }
        for item in sorted(positives, key=lambda row: row["query_id"])
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_train_positives(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lowered = str(path).lower()
    if "private_eval" in lowered or "blind" in path.name.lower():
        raise ValueError("private Blind data is forbidden")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") or []
    positives = [
        item for item in items
        if item.get("split") == "train" and item.get("case_kind") == "positive"
    ]
    if not positives:
        raise ValueError("no positive train rows found; refusing to mine Dev/Blind")
    if any(item.get("review_status") != "approved" for item in positives):
        raise ValueError("train rows must be approved before mining")
    return payload, positives


def _brief(candidate: Any) -> dict[str, Any]:
    return {
        "chunk_id": str(candidate.chunk_id),
        "source": str(candidate.source),
        "heading_path": str(candidate.heading_path or ""),
        "rank": int(candidate.rank),
        "dense_distance": candidate.dense_distance,
        "bm25_score": candidate.bm25_score,
        "rrf_score": candidate.rrf_score,
        "rerank_score": candidate.rerank_score,
    }


def _gold_rank(candidates: list[dict[str, Any]], gold_ids: set[str]) -> int:
    return next(
        (candidate["rank"] for candidate in candidates
         if candidate["chunk_id"] in gold_ids),
        0,
    )


def mine(args: argparse.Namespace) -> None:
    cases_path = Path(args.cases).expanduser().resolve()
    payload, positives = _load_train_positives(cases_path)
    output = Path(args.output).expanduser().resolve()

    done: dict[str, dict[str, Any]] = {}
    if output.exists() and args.resume:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("cases_sha256") == _sha256_file(cases_path):
            done = {row["query_id"]: row for row in existing.get("rows", [])}
            print(f"resume: {len(done)} traces already mined", file=sys.stderr)

    from rag_colloquial_profiles import colloquial_profile
    from rag_gate import retrieve_with_trace

    profile = colloquial_profile(args.arm)
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(positives, start=1):
        query_id = item["query_id"]
        if query_id in done:
            rows.append(done[query_id])
            continue
        started = time.perf_counter()
        trace = retrieve_with_trace(item["question"], REFERENCE_PLAN, profile, top_k=5)
        elapsed = (time.perf_counter() - started) * 1000
        gold_ids = {
            str(target["chunk_id"]) for target in item["relevant_targets"]
            if int(target["relevance_grade"]) == 3
        }
        stages = {
            "dense": [_brief(candidate) for candidate in trace.dense_candidates],
            "bm25": [_brief(candidate) for candidate in trace.bm25_candidates],
            "fusion": [_brief(candidate) for candidate in trace.fusion_candidates],
            "reranked": [_brief(candidate) for candidate in trace.reranked_candidates],
        }
        row = {
            "query_id": query_id,
            "anchor_id": item["anchor_id"],
            "domain": item.get("domain", ""),
            "query_style": item["query_style"],
            "question": item["question"],
            "grade3_chunk_ids": sorted(gold_ids),
            "all_relevant_chunk_ids": sorted({
                str(target["chunk_id"]) for target in item["relevant_targets"]
            }),
            "stages": stages,
            "stage_gold_ranks": {
                stage: _gold_rank(candidates, gold_ids)
                for stage, candidates in stages.items()
            },
            "gate_decision": bool(trace.gate_decision),
            "latency_ms": round(elapsed, 3),
        }
        rows.append(row)
        print(
            f"[mine] {position}/{len(positives)} {query_id} "
            f"rerank_gold_rank={row['stage_gold_ranks']['reranked']} "
            f"({elapsed:.0f} ms)",
            file=sys.stderr, flush=True,
        )
        if position % 10 == 0 or position == len(positives):
            _write_traces(output, cases_path, payload, args.arm, rows)
    _write_traces(output, cases_path, payload, args.arm, rows)
    print(json.dumps({
        "output": str(output),
        "rows": len(rows),
        "reranked_gold_top1": sum(
            row["stage_gold_ranks"]["reranked"] == 1 for row in rows
        ),
    }, ensure_ascii=False, indent=2))


def _write_traces(
    output: Path, cases_path: Path, payload: dict[str, Any], arm: str,
    rows: list[dict[str, Any]],
) -> None:
    positives = [
        item for item in payload.get("items") or []
        if item.get("split") == "train" and item.get("case_kind") == "positive"
    ]
    artifact = {
        "schema_version": MINING_SCHEMA_VERSION,
        "purpose": "train_split_false_winner_mining",
        "development_only": True,
        "private_blind_set": False,
        "arm": arm,
        "cases_path": str(cases_path),
        "cases_sha256": _sha256_file(cases_path),
        "train_cases_sha256": train_cases_fingerprint(positives),
        "index": payload.get("index"),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def _load_adjudication(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdicts: dict[str, dict[str, Any]] = {}
    for row in payload.get("verdicts", []):
        status = row["status"]
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"unknown adjudication status: {status}")
        if not str(row.get("reason") or "").strip():
            raise ValueError(f"adjudication for {row['chunk_id']} needs a reason")
        verdicts[f"{row['anchor_id']}::{row['chunk_id']}"] = row
    return verdicts


def _select_candidates(
    row: dict[str, Any], case: dict[str, Any],
) -> list[dict[str, Any]]:
    """Choose candidate negatives for one question, false winner first."""

    gold_ids = set(row["grade3_chunk_ids"])
    relevant_ids = set(row["all_relevant_chunk_ids"])
    reranked = row["stages"]["reranked"]
    gold_rank = row["stage_gold_ranks"]["reranked"]
    gold_score = next(
        (candidate["rerank_score"] for candidate in reranked
         if candidate["chunk_id"] in gold_ids),
        None,
    )
    gold_sources = {
        target["source"] for target in case["relevant_targets"]
        if int(target["relevance_grade"]) == 3
    }
    v1_negatives = {
        negative["chunk_id"]: negative
        for negative in case.get("hard_negatives") or []
    }

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _push(candidate: dict[str, Any], role: str) -> None:
        chunk_id = candidate["chunk_id"]
        if chunk_id in seen or chunk_id in relevant_ids:
            return
        seen.add(chunk_id)
        margin = None
        if gold_score is not None and candidate["rerank_score"] is not None:
            margin = round(float(candidate["rerank_score"]) - float(gold_score), 6)
        selected.append({
            "chunk_id": chunk_id,
            "source": candidate["source"],
            "heading_path": candidate["heading_path"],
            "retrieval_stage": "reranked",
            "reranked_rank": candidate["rank"],
            "rerank_score": candidate["rerank_score"],
            "margin_vs_gold": margin,
            "role": role,
            "relation_to_gold": (
                "same_document_wrong_section"
                if candidate["source"] in gold_sources
                else "cross_document_near_topic"
            ),
            "v1_adjudicated": chunk_id in v1_negatives,
            "v1_reason": (v1_negatives.get(chunk_id) or {}).get("reason", ""),
        })

    if gold_rank != 1 and reranked:
        _push(reranked[0], "current_false_winner")
    extra = 0
    for candidate in reranked:
        if extra >= MAX_EXTRA_CANDIDATES:
            break
        if candidate["chunk_id"] in gold_ids or candidate["chunk_id"] in seen:
            continue
        score = candidate["rerank_score"]
        if score is None or gold_score is None:
            continue
        if candidate["rank"] <= 3 or abs(float(score) - float(gold_score)) <= NEAR_MARGIN:
            _push(candidate, "near_margin_competitor")
            extra += 1
    return selected


def report(args: argparse.Namespace) -> None:
    traces_path = Path(args.traces).expanduser().resolve()
    cases_path = Path(args.cases).expanduser().resolve()
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    if traces.get("schema_version") != MINING_SCHEMA_VERSION:
        raise ValueError("unsupported mining artifact")
    _, positives = _load_train_positives(cases_path)
    case_by_id = {item["query_id"]: item for item in positives}
    expected = traces.get("train_cases_sha256")
    if expected is None:
        # Pre-migration artifacts only carry the whole-file hash.
        expected, actual = traces.get("cases_sha256"), _sha256_file(cases_path)
    else:
        actual = train_cases_fingerprint(positives)
    if expected != actual:
        raise ValueError(
            "mining traces were built from different Train cases; re-run `mine`"
        )
    missing = sorted(set(case_by_id) - {row["query_id"] for row in traces["rows"]})
    if missing:
        raise ValueError(f"mining traces incomplete; missing {len(missing)} rows: {missing[:5]}")
    adjudication_path = (
        Path(args.adjudication).expanduser().resolve() if args.adjudication else None
    )
    verdicts = _load_adjudication(adjudication_path)

    groups: list[dict[str, Any]] = []
    funnel = Counter()
    anomalies: list[str] = []
    per_anchor_false_winner: dict[str, set[str]] = defaultdict(set)
    for row in traces["rows"]:
        case = case_by_id[row["query_id"]]
        gold_rank = row["stage_gold_ranks"]["reranked"]
        fusion_rank = row["stage_gold_ranks"]["fusion"]
        if gold_rank == 1:
            funnel["correct_top1"] += 1
        elif 2 <= gold_rank <= 3:
            funnel["correct_rank2_3"] += 1
        elif 4 <= gold_rank <= 10:
            funnel["correct_rank4_10"] += 1
        elif gold_rank > 10:
            funnel["correct_rank_gt10"] += 1
        else:
            funnel["candidate_missing"] += 1
            anomalies.append(row["query_id"])
        candidates = _select_candidates(row, case)
        for candidate in candidates:
            key = f"{row['anchor_id']}::{candidate['chunk_id']}"
            verdict = verdicts.get(key)
            if verdict is not None:
                candidate["status"] = verdict["status"]
                candidate["adjudication_reason"] = verdict["reason"]
            elif candidate["v1_adjudicated"]:
                candidate["status"] = "hard_negative"
                candidate["adjudication_reason"] = (
                    f"V1 双审已裁决：{candidate['v1_reason']}"
                )
            else:
                candidate["status"] = "needs_adjudication"
                candidate["adjudication_reason"] = (
                    "尚无人工/确定性裁决；在裁决前不得作为训练负例。"
                )
            if candidate["role"] == "current_false_winner":
                per_anchor_false_winner[row["anchor_id"]].add(candidate["chunk_id"])
        groups.append({
            "query_id": row["query_id"],
            "anchor_id": row["anchor_id"],
            "domain": row["domain"],
            "query_style": row["query_style"],
            "question": row["question"],
            "grade3_chunk_ids": row["grade3_chunk_ids"],
            "stage_gold_ranks": row["stage_gold_ranks"],
            "reranked_gold_rank": gold_rank,
            "fusion_gold_rank": fusion_rank,
            "group_kind": (
                "stability_anchor" if gold_rank == 1 else
                "false_winner_group" if gold_rank else "candidate_missing"
            ),
            "candidates": candidates,
        })

    # --- difficulty accounting over pairs eligible for training -------------
    margin_buckets = Counter()
    pair_counter = Counter()
    relation_counter = Counter()
    status_counter = Counter()
    excluded = Counter()
    for group in groups:
        gold_score = None
        for candidate in group["candidates"]:
            status_counter[candidate["status"]] += 1
            if candidate["status"] != "hard_negative":
                excluded[candidate["status"]] += 1
                continue
            relation = (
                "relationship_error"
                if "关系" in candidate["adjudication_reason"] else
                candidate["relation_to_gold"]
            )
            relation_counter[relation] += 1
            margin = candidate["margin_vs_gold"]
            if margin is None:
                continue
            pair_counter["total"] += 1
            if margin >= 0:
                pair_counter["base_model_wrong"] += 1
            else:
                pair_counter["base_model_correct"] += 1
            separation = abs(margin)
            if separation <= 0.2:
                margin_buckets["0.0-0.2"] += 1
            elif separation <= 0.5:
                margin_buckets["0.2-0.5"] += 1
            else:
                margin_buckets[">0.5"] += 1

    total_pairs = pair_counter.get("total", 0)
    already_correct = pair_counter.get("base_model_correct", 0)
    easy_fraction = round(already_correct / total_pairs, 4) if total_pairs else None
    data_quality_gate = {
        "rule": "若 ~90% Pair 在 base 模型下已轻松分对，数据不合格，不得进入训练",
        "base_model_correct_fraction": easy_fraction,
        "verdict": (
            "insufficient_pairs" if total_pairs == 0 else
            "fail_too_easy" if easy_fraction is not None and easy_fraction >= 0.9
            else "pass"
        ),
    }
    artifact = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "development_only": True,
        "private_blind_set": False,
        "inputs": {
            "traces_path": str(traces_path),
            "traces_sha256": _sha256_file(traces_path),
            "cases_path": str(cases_path),
            "cases_sha256": _sha256_file(cases_path),
            "adjudication_path": str(adjudication_path) if adjudication_path else None,
            "index": traces.get("index"),
            "arm": traces.get("arm"),
        },
        "summary": {
            "train_questions": len(groups),
            "train_anchors": len({group["anchor_id"] for group in groups}),
            "reranked_funnel": dict(sorted(funnel.items())),
            "candidate_missing_query_ids": anomalies,
            "false_winner_groups": sum(
                group["group_kind"] == "false_winner_group" for group in groups
            ),
            "anchors_with_false_winner": len(per_anchor_false_winner),
            "unique_anchor_false_winner_pairs": sum(
                len(chunks) for chunks in per_anchor_false_winner.values()
            ),
            "candidate_status_counts": dict(sorted(status_counter.items())),
            "excluded_unadjudicated": dict(sorted(excluded.items())),
            "hard_negative_relation_distribution": dict(sorted(relation_counter.items())),
            "hard_negative_pairs": {
                "total": total_pairs,
                "base_model_correct": already_correct,
                "base_model_wrong": pair_counter.get("base_model_wrong", 0),
                "margin_buckets_abs": dict(sorted(margin_buckets.items())),
            },
            "data_quality_gate": data_quality_gate,
        },
        "groups": groups,
    }
    report_json = Path(args.output_json).expanduser().resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    report_md = Path(args.output_md).expanduser().resolve()
    report_md.write_text(_render_report_md(artifact), encoding="utf-8")
    print(json.dumps({
        "output_json": str(report_json),
        "output_md": str(report_md),
        "summary": artifact["summary"],
    }, ensure_ascii=False, indent=2))


def _render_report_md(artifact: dict[str, Any]) -> str:
    summary = artifact["summary"]
    gate = summary["data_quality_gate"]
    lines = [
        "# Train Split 真实 False-Winner 挖掘报告（compact32，2026-08-26）",
        "",
        "> 生产镜像 Top-20 trace 上的候选构造与裁决汇总。`needs_adjudication`",
        "> 候选一律排除在训练之外；只有带书面裁决理由的 `hard_negative` 才能进入",
        "> 未来的 F3 训练对。Dev80 的失败题从未进入本报告。",
        "",
        f"- Train 问法：{summary['train_questions']}（锚点 {summary['train_anchors']}）",
        f"- 精排漏斗：{json.dumps(summary['reranked_funnel'], ensure_ascii=False)}",
        f"- 候选缺失（数据/索引异常）：{summary['candidate_missing_query_ids'] or '无'}",
        f"- 真实 false-winner 组：{summary['false_winner_groups']}"
        f"（涉及锚点 {summary['anchors_with_false_winner']}，"
        f"唯一 锚点×错误冠军 对 {summary['unique_anchor_false_winner_pairs']}）",
        f"- 候选状态：{json.dumps(summary['candidate_status_counts'], ensure_ascii=False)}",
        f"- 被排除的未裁决/部分支持候选：{json.dumps(summary['excluded_unadjudicated'], ensure_ascii=False)}",
        f"- hard negative 关系分布：{json.dumps(summary['hard_negative_relation_distribution'], ensure_ascii=False)}",
        f"- hard negative 对：{json.dumps(summary['hard_negative_pairs'], ensure_ascii=False)}",
        "",
        "## 数据质量门",
        "",
        f"- 规则：{gate['rule']}",
        f"- base 模型已分对比例：{gate['base_model_correct_fraction']}",
        f"- 判定：**{gate['verdict']}**",
        "",
        "## 各锚点 false winner",
        "",
    ]
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in artifact["groups"]:
        if group["group_kind"] == "false_winner_group":
            by_anchor[group["anchor_id"]].append(group)
    for anchor_id in sorted(by_anchor):
        groups = by_anchor[anchor_id]
        lines.append(
            f"### {anchor_id}（{groups[0]['domain']}，失败问法 {len(groups)}/4）"
        )
        lines.append("")
        for group in groups:
            winner = next(
                (candidate for candidate in group["candidates"]
                 if candidate["role"] == "current_false_winner"),
                None,
            )
            if winner is None:
                continue
            lines.append(
                f"- `{group['query_id']}` gold_rank={group['reranked_gold_rank']} "
                f"冠军=`{winner['chunk_id']}`（{winner['relation_to_gold']}，"
                f"margin={winner['margin_vs_gold']}，status={winner['status']}）"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    mine_parser = sub.add_parser("mine", help="run production-mirror traces for Train")
    mine_parser.add_argument("--cases", default=str(DEFAULT_CASES))
    mine_parser.add_argument("--output", default=str(DEFAULT_TRACES))
    mine_parser.add_argument("--arm", default="compact32")
    mine_parser.add_argument("--resume", action="store_true",
                             help="skip query_ids already present in the output")
    mine_parser.set_defaults(func=mine)
    report_parser = sub.add_parser("report", help="build candidate groups + difficulty report")
    report_parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    report_parser.add_argument("--cases", default=str(DEFAULT_CASES))
    report_parser.add_argument("--adjudication", default=str(DEFAULT_ADJUDICATION))
    report_parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON))
    report_parser.add_argument("--output-md", default=str(DEFAULT_REPORT_MD))
    report_parser.set_defaults(func=report)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
