#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dev80 ranking-failure audit for the frozen colloquial ``compact32`` runs.

Diagnosis only.  This tool reads the frozen Dev80 evaluation artifact and the
approved Dev cases, extracts every positive query whose ``reranked_rank`` is
not 1, and classifies each failure.  Its output must never be copied into
training data: Dev failures define *construction rules* for Train mining, not
training rows (see NEXT_RAG_OPTIMIZATION_GUIDE_20260826.md §4.1).

Deterministic fields (ranks, margins, same/cross document, register flips,
stage attribution) are recomputed from the artifact on every run.  Content
adjudication of the wrong winners cannot be derived mechanically, so it lives
in :data:`CHUNK_ADJUDICATIONS`, a reviewed table with verbatim quotes from the
frozen index.  The table is keyed by chunk_id and versioned with this script;
re-running the audit is idempotent (no timestamps are written).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_SCHEMA_VERSION = "colloquial-ranking-failure-audit-v1"
DEFAULT_RESULT = ROOT / "docs/rag_eval/colloquial/dev80_compact32.json"
DEFAULT_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_dev80_v1.json"
DEFAULT_JSON = ROOT / "docs/rag_eval/colloquial/DEV80_RANKING_FAILURE_AUDIT_20260826.json"
DEFAULT_MD = ROOT / "docs/rag_eval/colloquial/DEV80_RANKING_FAILURE_AUDIT_20260826.md"

NEAR_MISS = "near_miss_rank_2_3"
DEEP_MISS = "deep_miss_rank_4_10"
OUT_OF_POOL = "out_of_reranked_pool"

# Reviewed content adjudication of competitor chunks (2026-08-26, in-session
# manual review against the frozen 3348-chunk index; no LLM API involved).
# ``status`` vocabulary:
#   equivalent_evidence_candidate  chunk plausibly satisfies the full answer
#                                  requirement -> qrels/multi-gold review
#   questionable_negative          previously adjudicated negative whose text
#                                  arguably covers the requirement -> review
#   partial_support                covers part of a compound requirement;
#                                  grade-1/2 territory, never a hard negative
#   adjudicated_insufficient       reviewed and confirmed not able to answer
CHUNK_ADJUDICATIONS: dict[str, dict[str, str]] = {
    "llm_algorithm_basic_05_深度学习笔记——模型压缩和优化技术_蒸馏_剪枝_量化_59540e722d0a": {
        "status": "equivalent_evidence_candidate",
        "quote": "知识蒸馏通过训练一个小模型（学生模型）来模仿一个大模型（教师模型）的行为…… 步骤：训练教师模型／准备软标签／构建学生模型／构建损失函数（软标签和硬标签损失的组合）／训练学生模型／评估模型",
        "reason": "该块含蒸馏基本思想、师生角色与完整训练步骤表，覆盖了 alg07 的全部答案要求，很可能是未标注的等价 Grade-3 证据；精排选它未必是错误。",
    },
    "backend_basic_03_06_mysql_bplus_tree_5b2625077719": {
        "status": "questionable_negative",
        "quote": "非叶子节点不存储数据，更适合磁盘存储和 I/O 优化 / 叶子节点存储所有数据并通过链表连接，更便于顺序遍历，查找效率稳定",
        "reason": "V1 已裁决为 hard negative（理由：仅目录标题），但该目录块末尾两条要点句同时给出了两项优化及其收益，与 be02 答案要求高度重合；负例裁决需复核，且它已作为负例参加过 F1/F2 训练仍稳定获胜。",
    },
    "2026-06-19_rag技术入门与架构演进_a9feef_f490c6e93b07": {
        "status": "partial_support",
        "quote": "（2）编码：将上一步生成的假设性文档输入到一个对比编码器……（3）检索：使用这个假设性文档的向量……将困难的“查询到文档”匹配问题转化为“文档到文档”匹配",
        "reason": "覆盖 HyDE 的编码与检索两步及其原理，但缺少“LLM 在无外部知识条件下生成假设回复”这一起始步骤与“假设回复可能含虚假信息”的风险说明（在相邻块），属部分支持（grade-1/2 区间），不能标负例。",
    },
    "2026-06-19_hello_agents_rag与agent核心精选_150ed4_9af3bc369699": {
        "status": "partial_support",
        "quote": "forward(self, Q, K, V, mask=None): # 1. 对 Q, K, V 进行线性变换 …… # 2. 计算缩放点积注意力 …… # 3. 合并多头输出并进行最终的线性变换",
        "reason": "多头注意力实现代码块：展示了 Q/K/V 线性投影与缩放点积调用，但 softmax 与对 V 加权求和的公式在相邻块，未完整给出 alg01 要求的计算关系；部分支持，不能标负例。",
    },
    "llm_algorithm_basic_06_模型微调之LoRA_6807535047e5": {
        "status": "equivalent_evidence_candidate",
        "quote": "Q=W_Q·input, K=W_K·input, V=W_V·input …… Attention Scores=Softmax(QK^T/√d_k) …… Attention Output=Attention Scores·V",
        "reason": "LoRA 语境下完整给出投影、缩放点积、softmax 与对 V 加权求和的全部计算关系，覆盖 alg01 答案要求；是否因 LoRA(W+ΔW) 框架而不算直接证据需 qrels 复核。",
    },
    "llm_app_interview_02_rag_basics_f0179cb29f7c": {
        "status": "adjudicated_insufficient",
        "quote": "高级 RAG 引入……预检索过程（索引优化/查询优化）和后检索过程……后检索过程引入的方法包括：",
        "reason": "高级 RAG 总览块：提到信息过载与后检索方向，但未说明上下文压缩的适用场景、处理方式或要解决的问题（列举在下一块被截断）；确认不足以回答 rag12，是真实 false winner。",
    },
    "2026-06-19_rag技术入门与架构演进_a9feef_c753fd0bb94c": {
        "status": "adjudicated_insufficient",
        "quote": "一、上下文扩展……句子窗口检索：为检索精确性而索引小块，为上下文丰富性而检索大块",
        "reason": "句子窗口检索是“上下文扩展”方案，解决块太小缺上下文的问题，方向与上下文压缩相反；不含压缩的适用场景与机制，确认不足以回答 rag12。",
    },
    "llm_app_interview_10_harness_engineering_1112749fe406": {
        "status": "adjudicated_insufficient",
        "quote": "Context Engineering 关注模型在当前步骤到底应该看到什么信息……信息如何选择、组织、压缩、保留和淘汰",
        "reason": "Context Engineering 摘要卡：出现“压缩”一词但语境是上下文工程六方面，与 RAG 检索结果压缩机制无关，确认不足以回答 rag12。",
    },
    "2026-06-19_rag技术入门与架构演进_a9feef_270715e4e09a": {
        "status": "equivalent_evidence_candidate",
        "quote": "压缩……初步检索到的文档块……包含大量无关的噪音文本……增加 API 调用的成本和延迟……降低最终生成答案的质量……1. 内容提取……2. 文档过滤",
        "reason": "另一文档的“二、压缩 (Compression)”专节完整覆盖 rag12 答案要求（过大含无关信息→成本与质量问题；提取/过滤两种处理方式），是未标注的等价 Grade-3 候选；它在精排中压在金标上方。",
    },
    "llm_app_interview_02_rag_basics_782b4979ced3": {
        "status": "adjudicated_insufficient",
        "quote": "上下文压缩和筛选、缩短窗口等词语列举",
        "reason": "V1 已裁决 hard negative：仅列举术语，未说明适用场景、处理方式与作用；维持原裁决。",
    },
}

_REVIEW_STATUSES = {"equivalent_evidence_candidate", "questionable_negative"}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_rows(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in run["positive"]["rows"]}


def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": candidate["chunk_id"],
        "source": candidate["source"],
        "heading_path": candidate.get("heading_path", ""),
        "rank": candidate["rank"],
        "rerank_score": candidate.get("rerank_score"),
    }


def _adjudication(chunk_id: str) -> dict[str, str]:
    return CHUNK_ADJUDICATIONS.get(chunk_id, {
        "status": "not_reviewed",
        "quote": "",
        "reason": "本轮未单独复核该候选内容。",
    })


def _rank_class(rank: int) -> str:
    if 2 <= rank <= 3:
        return NEAR_MISS
    if 4 <= rank <= 10:
        return DEEP_MISS
    return OUT_OF_POOL


def _root_cause(fusion_rank: int, has_review_flag: bool) -> str:
    """Deterministic priority: reranker_harm > fusion_low > ambiguous_gold
    > representation (see guide §4.1)."""

    if fusion_rank == 1:
        return "reranker_harm"
    if fusion_rank == 0 or fusion_rank > 3:
        return "fusion_low"
    if has_review_flag:
        return "ambiguous_gold"
    return "query_document_representation"


def build_audit(
    result: dict[str, Any], cases: dict[str, Any],
) -> dict[str, Any]:
    runs = result["runs"]
    if not runs:
        raise ValueError("result artifact contains no runs")
    rows_by_run = [_positive_rows(run) for run in runs]
    case_by_id = {item["query_id"]: item for item in cases["items"]}

    # Cross-run consistency: the audit is only meaningful on stable failures.
    inconsistent: list[str] = []
    for query_id, row in rows_by_run[0].items():
        ranks = {run_rows[query_id]["reranked_rank"] for run_rows in rows_by_run}
        if len(ranks) != 1:
            inconsistent.append(query_id)

    anchor_rank1_styles: dict[str, set[str]] = defaultdict(set)
    for row in rows_by_run[0].values():
        if int(row["reranked_rank"] or 0) == 1:
            anchor_rank1_styles[row["anchor_id"]].add(row["query_style"])

    records: list[dict[str, Any]] = []
    for query_id in sorted(rows_by_run[0]):
        row = rows_by_run[0][query_id]
        rank = int(row["reranked_rank"] or 0)
        if rank == 1:
            continue
        case = case_by_id.get(query_id)
        if case is None:
            raise ValueError(f"result row {query_id} is missing from the cases file")
        gold_ids = set(row["grade3_chunk_ids"])
        trace = row["retrieval_trace"]
        reranked = trace["reranked_candidates"]
        top1 = reranked[0]
        gold_candidate = next(
            (candidate for candidate in reranked if candidate["chunk_id"] in gold_ids),
            None,
        )
        gold_sources = {
            target["source"] for target in case["relevant_targets"]
            if int(target["relevance_grade"]) == 3
        }
        v1_hard_negative_ids = {
            negative["chunk_id"] for negative in case.get("hard_negatives") or []
        }
        above_gold = [
            candidate for candidate in reranked
            if gold_candidate is not None and candidate["rank"] < gold_candidate["rank"]
        ]
        competitors = []
        review_flag = False
        for candidate in above_gold:
            adjudication = _adjudication(candidate["chunk_id"])
            review_flag = review_flag or adjudication["status"] in _REVIEW_STATUSES
            competitors.append({
                **_candidate_brief(candidate),
                "is_v1_hard_negative": candidate["chunk_id"] in v1_hard_negative_ids,
                "adjudication": adjudication,
            })
        top1_adjudication = _adjudication(top1["chunk_id"])

        same_document = top1["source"] in gold_sources
        sibling_rank1 = sorted(anchor_rank1_styles.get(row["anchor_id"], set()))
        register_shift = bool(sibling_rank1)
        partial = any(
            competitor["adjudication"]["status"] == "partial_support"
            for competitor in competitors
        )

        classes = [_rank_class(rank)]
        classes.append(
            "same_document_wrong_section" if same_document
            else "cross_document_near_topic"
        )
        if register_shift:
            classes.append("query_document_register_shift")
        if partial:
            classes.append("multi_requirement_partial_match")
        if review_flag:
            classes.append("qrels_or_evidence_needs_review")

        margin = None
        if gold_candidate is not None and top1.get("rerank_score") is not None \
                and gold_candidate.get("rerank_score") is not None:
            margin = round(
                float(top1["rerank_score"]) - float(gold_candidate["rerank_score"]), 6,
            )
        records.append({
            "query_id": query_id,
            "anchor_id": row["anchor_id"],
            "domain": row.get("domain", ""),
            "query_style": row["query_style"],
            "question": case["question"],
            "answer_requirements": case["answer_requirements"],
            "grade3_chunk_ids": sorted(gold_ids),
            "gold_sources": sorted(gold_sources),
            "stage_ranks": {
                "dense": row["dense_rank"],
                "bm25": row["bm25_rank"],
                "union": row["union_rank"],
                "fusion": row["fusion_rank"],
                "reranked": rank,
                "final": row["final_rank"],
            },
            "reranked_rank_consistent_across_runs": query_id not in inconsistent,
            "wrong_top1": {
                **_candidate_brief(top1),
                "is_v1_hard_negative": top1["chunk_id"] in v1_hard_negative_ids,
                "adjudication": top1_adjudication,
            },
            "best_gold": _candidate_brief(gold_candidate) if gold_candidate else None,
            "top1_minus_gold_margin": margin,
            "competitors_above_gold": competitors,
            "gate": {
                "gate_decision": row["gate_decision"],
                "rerank_top": (row.get("gate_features") or {}).get("rerank_top"),
                "rerank_margin": (row.get("gate_features") or {}).get("rerank_margin"),
            },
            "sibling_styles_with_rank1": sibling_rank1,
            "failure_classes": classes,
            "root_cause": _root_cause(int(row["fusion_rank"] or 0), review_flag),
            "training_data_rule": (
                "forbidden_dev_leak: 该失败题与其错误冠军不得写入训练数据；"
                "只允许在 Train split 上按同类构造规则挖掘对应 false winner。"
            ),
        })

    class_counts = Counter(
        failure_class for record in records
        for failure_class in record["failure_classes"]
    )
    root_counts = Counter(record["root_cause"] for record in records)
    anchor_counts = Counter(record["anchor_id"] for record in records)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "purpose": "diagnosis_only_not_training_data",
        "source_result": {
            "dataset_id": result["input"]["dataset_id"],
            "retrieval_arm": result["configuration"]["retrieval_arm"],
            "runs": len(runs),
            "index": result["input"]["index"],
        },
        "summary": {
            "positive_rows": len(rows_by_run[0]),
            "failures": len(records),
            "failed_anchors": dict(sorted(anchor_counts.items())),
            "rank_distribution": dict(sorted(Counter(
                record["stage_ranks"]["reranked"] for record in records
            ).items())),
            "failure_class_counts": dict(sorted(class_counts.items())),
            "root_cause_counts": dict(sorted(root_counts.items())),
            "cross_run_inconsistent_query_ids": sorted(inconsistent),
            "qrels_review_chunk_ids": sorted({
                chunk_id for chunk_id, verdict in CHUNK_ADJUDICATIONS.items()
                if verdict["status"] in _REVIEW_STATUSES
            }),
        },
        "records": records,
    }


def render_markdown(audit: dict[str, Any], *, result_sha: str, cases_sha: str) -> str:
    summary = audit["summary"]
    lines = [
        "# Dev80 Ranking Failure Audit（compact32，2026-08-26）",
        "",
        "> 用途：**只诊断，不训练**。本文件列出的任何错误冠军都不得写入训练数据；",
        "> 它们只用于确认瓶颈类型并定义 Train split 上的 false-winner 构造规则。",
        "",
        f"- 冻结产物：`dev80_compact32.json`（`{result_sha}`）",
        f"- Dev 用例：`rag_colloquial_dev80_v1.json`（`{cases_sha}`）",
        f"- 正例总数 {summary['positive_rows']}，失败 {summary['failures']} 条，"
        f"三轮 reranked_rank 完全一致：{not summary['cross_run_inconsistent_query_ids']}",
        "",
        "## 失败分布",
        "",
        "| 维度 | 分布 |",
        "|---|---|",
        f"| 失败锚点 | {json.dumps(summary['failed_anchors'], ensure_ascii=False)} |",
        f"| reranked_rank | {json.dumps(summary['rank_distribution'], ensure_ascii=False)} |",
        f"| failure_class | {json.dumps(summary['failure_class_counts'], ensure_ascii=False)} |",
        f"| root_cause | {json.dumps(summary['root_cause_counts'], ensure_ascii=False)} |",
        "",
        "## 需要 qrels/证据复核的候选块",
        "",
        "以下候选在人工复核中被判定为“可能等价支持答案”或“既有负例裁决存疑”，"
        "在完成 qrels 复核前不得作为训练负例：",
        "",
    ]
    for chunk_id in summary["qrels_review_chunk_ids"]:
        verdict = CHUNK_ADJUDICATIONS[chunk_id]
        lines.append(f"- `{chunk_id}`（{verdict['status']}）：{verdict['reason']}")
    lines.extend(["", "## 逐题审计", ""])
    for record in audit["records"]:
        ranks = record["stage_ranks"]
        top1 = record["wrong_top1"]
        gold = record["best_gold"] or {}
        lines.extend([
            f"### `{record['query_id']}`（{record['query_style']}，{record['domain']}）",
            "",
            f"- 问题：{record['question']}",
            f"- 答案要求：{'；'.join(record['answer_requirements'])}",
            f"- Grade-3 金标：`{'`，`'.join(record['grade3_chunk_ids'])}`",
            f"- 阶段排名：dense={ranks['dense']} bm25={ranks['bm25']} "
            f"union={ranks['union']} fusion={ranks['fusion']} "
            f"rerank={ranks['reranked']} final={ranks['final']}",
            f"- 错误冠军：`{top1['chunk_id']}`（{top1['source']} / {top1['heading_path']}），"
            f"rerank={top1['rerank_score']}，V1 hard negative：{top1['is_v1_hard_negative']}",
            f"- 金标精排分：{gold.get('rerank_score')}，Top1−金标 margin："
            f"{record['top1_minus_gold_margin']}",
            f"- Gate：decision={record['gate']['gate_decision']} "
            f"rerank_top={record['gate']['rerank_top']} "
            f"rerank_margin={record['gate']['rerank_margin']}",
            f"- 同锚点已 Rank1 的问法：{record['sibling_styles_with_rank1'] or '无（整锚点失败）'}",
            f"- 分类：{', '.join(record['failure_classes'])}；root_cause={record['root_cause']}",
            f"- 冠军裁决（{top1['adjudication']['status']}）：{top1['adjudication']['reason']}",
        ])
        if record["competitors_above_gold"]:
            lines.append("- 压在金标上方的候选：")
            for competitor in record["competitors_above_gold"]:
                verdict = competitor["adjudication"]
                lines.append(
                    f"  - r{competitor['rank']} `{competitor['chunk_id']}` "
                    f"score={competitor['rerank_score']}（{verdict['status']}）"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT,
                        help="frozen Dev80 evaluation artifact")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES,
                        help="approved Dev cases file matching the artifact")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    result_path = args.result.expanduser().resolve()
    cases_path = args.cases.expanduser().resolve()
    audit = build_audit(_load_json(result_path), _load_json(cases_path))
    result_sha = _sha256_file(result_path)
    cases_sha = _sha256_file(cases_path)
    audit["inputs"] = {
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "cases_path": str(cases_path),
        "cases_sha256": cases_sha,
    }
    json_path = args.output_json.expanduser().resolve()
    md_path = args.output_md.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_markdown(audit, result_sha=result_sha, cases_sha=cases_sha),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_json": str(json_path),
        "output_md": str(md_path),
        "summary": audit["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
