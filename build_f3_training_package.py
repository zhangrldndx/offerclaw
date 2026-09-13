#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the F3 training package: V1 Train + V2-A Train, anchor-normalized.

F3 exists to test one claim — that *more independent knowledge anchors with
real adjudicated negatives* move end-to-end ranking, where F1/F2's 45 anchors
did not.  So the model, the objective, the trainable depth and the optimizer
settings stay exactly as they were; the corpus is what changes.

Anchor normalization is part of building that corpus, not a second lever.  A
V1 anchor contributes four phrasings × one negative = four pairs, while a V2-A
anchor contributes four phrasings × two negatives = eight.  Left alone, the
merged corpus would weight anchors by how many negatives they happen to carry
and quietly undo the point of adding anchors.  Each anchor therefore gets the
same total weight, expressed as a per-triple ``pair_weight`` that multiplies
the existing corpus-level kind weights.

The package also carries the difficulty report that decides whether training is
worth running at all: if the base reranker already separates almost every pair
by a wide margin, the data is too easy and F3 must not start.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_colloquial_reranker_training import (  # noqa: E402
    ColloquialTrainingBuildError,
    _chunk_ref,
    _load_chunks,
    _negative_kind,
    build_training_artifact,
)


SCHEMA_VERSION = "colloquial-reranker-training-f3-v1"
V1_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json"
V2A_CASES = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_train_dev_draft.json"
DEFAULT_OUTPUT = ROOT / ".offerclaw/reranker_training/f3_merged_v1_v2a.json"
DEFAULT_REPORT = ROOT / "docs/rag_eval/colloquial/F3_DATA_DIFFICULTY_20260826.json"
BASE_MODEL = "BAAI/bge-reranker-base"
# Above this share of already-separated pairs the corpus repeats the F1/F2
# failure and training must not start.
TOO_EASY_PAIR_SHARE = 0.90

# Corrective weighting (2026-08-27).  F3's telemetry showed the training signal
# was pointed away from the pairs that matter: mined production false winners
# were 24.4% of Train rows and 59% of the pairs the base model gets wrong, yet
# carried only 13.5% of the effective weight, while V1's 16.6% of rows carried
# 37.2%.  Anchor normalization causes this by construction -- the anchors that
# yielded the most mined negatives are the hard ones, and dividing by triple
# count penalises exactly them.
#
# The correction is derived from the base model's own margins rather than a
# hand-written wave/style list, so it stays true if the corpus changes.
WEIGHTING_MODES = ("anchor_normalized", "corrective")
DIFFICULTY_SHARPNESS = 4.0
# Easy pairs keep a floor of weight.  Driving them to zero invites the model to
# break what it already gets right in order to fix what it does not.
DIFFICULTY_FLOOR = 0.10


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _tagged(artifact: dict[str, Any], source: str) -> list[dict[str, Any]]:
    return [{**row, "source_wave": source, "negative_origin": "authored"}
            for row in artifact["triples"]]


def mined_triples(cases_path: Path, adjudication_path: Path, collection: Any,
                  source_wave: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn independently-adjudicated mined candidates into extra triples.

    Only ``hard_negative`` verdicts qualify.  Everything else — partial support
    or an unlabelled equivalent gold — is excluded and counted, because a false
    negative teaches the reranker to push down a chunk that answers.
    """

    payload = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    positives = [item for item in payload["items"]
                 if item["split"] == "train" and item["case_kind"] == "positive"]
    adjudication = json.loads(Path(adjudication_path).read_text(encoding="utf-8"))
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: Counter = Counter()
    for verdict in adjudication["verdicts"]:
        if verdict["status"] == "hard_negative":
            eligible[verdict["anchor_id"]].append(verdict)
        else:
            excluded[verdict["status"]] += 1

    needed = {verdict["chunk_id"]
              for rows in eligible.values() for verdict in rows}
    needed.update(
        str(target["chunk_id"]) for item in positives
        for target in item["relevant_targets"]
        if int(target["relevance_grade"]) == 3 and item["anchor_id"] in eligible
    )
    chunks = _load_chunks(collection, needed) if needed else {}

    rows: list[dict[str, Any]] = []
    for item in positives:
        verdicts = eligible.get(item["anchor_id"])
        if not verdicts:
            continue
        blocked = {negative["chunk_id"] for negative in item["hard_negatives"]}
        blocked.update(str(target["chunk_id"]) for target in item["relevant_targets"])
        golds = [str(target["chunk_id"]) for target in item["relevant_targets"]
                 if int(target["relevance_grade"]) == 3]
        for verdict in verdicts:
            if verdict["chunk_id"] in blocked:
                continue
            negative = _chunk_ref(verdict["chunk_id"], chunks)
            for gold_id in golds:
                positive = _chunk_ref(gold_id, chunks)
                rows.append({
                    "query_id": str(item["query_id"]),
                    "anchor_id": str(item["anchor_id"]),
                    "split": str(item["split"]),
                    "query_style": str(item.get("query_style") or ""),
                    "query": str(item["question"]),
                    "negative_kind": _negative_kind(positive, negative),
                    "positive": positive,
                    "negative": negative,
                    "source_wave": source_wave,
                    "negative_origin": "mined_production_false_winner",
                    "mined_rank": verdict.get("reranked_rank"),
                    "mined_margin_vs_gold": verdict.get("margin_vs_gold"),
                })
    return rows, {
        "adjudication": Path(adjudication_path).name,
        "eligible_negative_chunks": sum(len(rows_) for rows_ in eligible.values()),
        "anchors_with_mined_negative": len(eligible),
        "excluded_by_status": dict(sorted(excluded.items())),
        "unreviewed_candidates": adjudication["summary"].get(
            "skipped_unreviewed_candidates", 0
        ),
        "triples_added": len(rows),
    }


def difficulty_factor(margin: float) -> float:
    """Weight a pair by how wrong the base model currently is about it.

    Logistic in the base margin: strongly inverted pairs approach 1.0, pairs
    the base already separates decay toward :data:`DIFFICULTY_FLOOR`.  This is
    the same idea as hard-negative mining, applied to weights instead of
    membership, and it is deliberately continuous -- a threshold would put a
    cliff right where the interesting pairs sit.
    """
    return DIFFICULTY_FLOOR + (1.0 - DIFFICULTY_FLOOR) / (
        1.0 + math.exp(DIFFICULTY_SHARPNESS * margin)
    )


def assign_anchor_weights(triples: list[dict[str, Any]], *,
                          mode: str = "anchor_normalized") -> dict[str, Any]:
    """Give every anchor the same total weight within its split.

    ``mode="corrective"`` additionally scales each pair by
    :func:`difficulty_factor`, which requires ``base_scores`` to be attached
    first.  Anchor equality no longer holds in that mode -- that is the point,
    and it is reported rather than hidden.
    """

    if mode not in WEIGHTING_MODES:
        raise ColloquialTrainingBuildError(f"unknown weighting mode: {mode!r}")
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in triples:
        by_split[row["split"]].append(row)
    stats: dict[str, Any] = {}
    for split, rows in by_split.items():
        counts = Counter(row["anchor_id"] for row in rows)
        # Scale so the mean weight is 1.0: the loss is a weighted mean, so this
        # keeps the reported numbers comparable with F1/F2's unit weights.
        mean_triples = len(rows) / len(counts)
        for row in rows:
            row["pair_weight"] = mean_triples / counts[row["anchor_id"]]
        if mode == "corrective":
            for row in rows:
                scores = row.get("base_scores")
                if not scores or "margin" not in scores:
                    raise ColloquialTrainingBuildError(
                        "corrective weighting needs base_scores; score the "
                        "pairs before assigning weights"
                    )
                row["pair_weight"] *= difficulty_factor(float(scores["margin"]))
            # Balance the *query distribution* as well.  Anchor normalization
            # alone handed V1's 16.6% of rows 37.2% of the weight, because V1
            # anchors carry few triples each and the divisor is triple count.
            # That is an accounting artefact, not a judgement that V1 queries
            # deserve more say -- and V1's colloquial phrasings sit at cosine
            # 0.98 from their own standard form, i.e. they carry no colloquial
            # phenomenon at all (QUERY_REPRESENTATION_20260827.md §3).
            # Each style therefore ends up with weight proportional to its row
            # count; difficulty still orders the pairs *within* a style.
            by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_style[row.get("query_style", "unspecified")].append(row)
            total = sum(row["pair_weight"] for row in rows)
            for style_rows in by_style.values():
                current = sum(row["pair_weight"] for row in style_rows)
                target = total * len(style_rows) / len(rows)
                if current > 0:
                    for row in style_rows:
                        row["pair_weight"] *= target / current
            # Restore mean 1.0 so the reported loss stays on the F1/F2 scale.
            scale = len(rows) / sum(row["pair_weight"] for row in rows)
            for row in rows:
                row["pair_weight"] *= scale
        for row in rows:
            row["pair_weight"] = round(row["pair_weight"], 6)
        weights = [row["pair_weight"] for row in rows]
        stats[split] = {
            "anchors": len(counts),
            "triples": len(rows),
            "triples_per_anchor": {
                "min": min(counts.values()),
                "max": max(counts.values()),
                "mean": round(mean_triples, 4),
            },
            "pair_weight": {
                "min": round(min(weights), 6),
                "max": round(max(weights), 6),
                "mean": round(statistics.fmean(weights), 6),
            },
            "total_weight_per_anchor_is_equal": len({
                round(sum(row["pair_weight"] for row in rows
                          if row["anchor_id"] == anchor), 4)
                for anchor in counts
            }) == 1,
            "mode": mode,
            **_corrective_shares(rows),
        }
    return stats


def _corrective_shares(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the weight actually sits on pairs the base model gets wrong.

    This is the number F3 was missing.  It was 9.6% there, against an 11.3%
    row share -- the weighting was pointed slightly away from the signal, and
    nothing in the manifest said so.
    """
    if not all("base_scores" in row for row in rows):
        return {}
    total = sum(row["pair_weight"] for row in rows)
    inverted = [row for row in rows if float(row["base_scores"]["margin"]) <= 0]
    by_origin: dict[str, float] = defaultdict(float)
    for row in rows:
        by_origin[row.get("negative_origin", "unspecified")] += row["pair_weight"]
    return {
        "inverted_pair_row_share": round(len(inverted) / len(rows), 6),
        "inverted_pair_weight_share": round(
            sum(row["pair_weight"] for row in inverted) / total, 6),
        "weight_share_by_negative_origin": {
            key: round(value / total, 6) for key, value in sorted(by_origin.items())
        },
        "weight_share_by_query_style": _grouped_shares(rows, "query_style"),
        "weight_share_by_source_wave": _grouped_shares(rows, "source_wave"),
    }


def _grouped_shares(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = sum(row["pair_weight"] for row in rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unspecified"))].append(row["pair_weight"])
    return {
        name: {"row_share": round(len(values) / len(rows), 6),
               "weight_share": round(sum(values) / total, 6)}
        for name, values in sorted(grouped.items())
    }


def _score_pairs(triples: list[dict[str, Any]], *, base_model: str,
                 batch_size: int) -> dict[str, Any]:
    """Attach the base reranker's positive/negative scores to every triple.

    Loading goes through :func:`rag_rerank._load_reranker` rather than
    instantiating ``CrossEncoder`` directly: that resolver prefers the
    ModelScope snapshot and only falls back to HuggingFace, and a direct HF
    instantiation simply hangs on this machine.  Reusing it also guarantees the
    difficulty report is scored by the very model retrieval ranks with.
    """

    import os

    from rag_rerank import _load_reranker

    # The ONNX backend is a different score scale (raw logits vs sigmoid) and is
    # not what F3 trains, so it must not silently score the training corpus.
    previous = os.environ.pop("RAG_RERANK_ONNX_DIR", None)
    try:
        model = _load_reranker(base_model)
        if model is None:
            raise ColloquialTrainingBuildError(
                f"base reranker {base_model!r} could not be loaded; the "
                "difficulty report cannot be produced without it"
            )
        pairs: list[tuple[str, str]] = []
        for row in triples:
            pairs.append((row["query"], row["positive"]["scoring_text"]))
            pairs.append((row["query"], row["negative"]["scoring_text"]))
        scores = model.predict(pairs, batch_size=batch_size)
    finally:
        if previous is not None:
            os.environ["RAG_RERANK_ONNX_DIR"] = previous
    for index, row in enumerate(triples):
        positive = float(scores[2 * index])
        negative = float(scores[2 * index + 1])
        row["base_scores"] = {
            "positive": round(positive, 6),
            "negative": round(negative, 6),
            "margin": round(positive - negative, 6),
        }
    return {
        "model": base_model,
        "backend": type(model).__name__,
        "pairs_scored": len(pairs),
    }


def _bucket(margin: float) -> str:
    span = abs(margin)
    if span <= 0.2:
        return "0.0-0.2"
    if span <= 0.5:
        return "0.2-0.5"
    return ">0.5"


def difficulty_report(triples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in triples:
        by_split[row["split"]].append(row)
    output: dict[str, Any] = {}
    for split, rows in sorted(by_split.items()):
        margins = [row["base_scores"]["margin"] for row in rows]
        separated = [margin for margin in margins if margin > 0]
        inverted = [row for row in rows if row["base_scores"]["margin"] <= 0]
        output[split] = {
            "triples": len(rows),
            "anchors": len({row["anchor_id"] for row in rows}),
            "questions": len({row["query_id"] for row in rows}),
            "base_already_separated": len(separated),
            "base_already_separated_share": round(len(separated) / len(rows), 4),
            "base_inverted_pairs": len(inverted),
            "inverted_query_ids": sorted({row["query_id"] for row in inverted})[:40],
            "margin": {
                "mean": round(statistics.fmean(margins), 6),
                "median": round(statistics.median(margins), 6),
                "min": round(min(margins), 6),
                "max": round(max(margins), 6),
            },
            "abs_margin_buckets": dict(sorted(
                Counter(_bucket(margin) for margin in margins).items()
            )),
            "by_source_wave": dict(sorted(
                Counter(row["source_wave"] for row in rows).items()
            )),
            "by_negative_kind": dict(sorted(
                Counter(row["negative_kind"] for row in rows).items()
            )),
            "by_negative_origin": dict(sorted(
                Counter(row["negative_origin"] for row in rows).items()
            )),
            "by_query_style": dict(sorted(
                Counter(row["query_style"] for row in rows).items()
            )),
            "by_domain": dict(sorted(
                Counter(row.get("domain", "") for row in rows).items()
            )),
        }
    train = output.get("train", {})
    share = train.get("base_already_separated_share", 1.0)
    output["verdict"] = {
        "rule": (
            f"若 Train 中 base 模型已分对的 Pair 占比 ≥ {TOO_EASY_PAIR_SHARE:.0%}，"
            "监督信号过易，重复 F1/F2 的失败，不得开训。"
        ),
        "train_base_already_separated_share": share,
        "gate": "pass" if share < TOO_EASY_PAIR_SHARE else "fail_too_easy",
    }
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    import chromadb
    from rag_tools import get_collection_name

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    v1 = build_training_artifact(cases_path=args.v1_cases, collection=collection)
    v2a = build_training_artifact(cases_path=args.v2a_cases, collection=collection)

    triples = [*_tagged(v1, "v1"), *_tagged(v2a, "v2a")]
    mined_stats: dict[str, Any] = {"enabled": False}
    if args.mined_adjudication:
        mined_path = Path(args.mined_adjudication).expanduser().resolve()
        if mined_path.exists():
            extra, mined_stats = mined_triples(
                Path(args.v2a_cases), mined_path, collection, "v2a",
            )
            mined_stats["enabled"] = True
            triples.extend(extra)
        else:
            mined_stats = {"enabled": False,
                           "reason": f"{mined_path.name} not found"}
    train_anchors = {row["anchor_id"] for row in triples if row["split"] == "train"}
    dev_anchors = {row["anchor_id"] for row in triples if row["split"] == "dev"}
    overlap = sorted(train_anchors & dev_anchors)
    if overlap:
        raise ColloquialTrainingBuildError(f"anchor leakage train/dev: {overlap[:5]}")
    collisions = sorted(
        {row["anchor_id"] for row in triples if row["source_wave"] == "v1"}
        & {row["anchor_id"] for row in triples if row["source_wave"] == "v2a"}
    )
    if collisions:
        raise ColloquialTrainingBuildError(
            f"anchor id collision between waves: {collisions[:5]}"
        )

    # Scoring has to come first now: corrective weighting reads base margins,
    # and the difficulty report must describe the corpus that will be trained.
    scoring = _score_pairs(triples, base_model=args.base_model,
                           batch_size=args.batch_size)
    weight_stats = assign_anchor_weights(
        triples, mode=getattr(args, "weighting", "anchor_normalized"))
    report = difficulty_report(triples)
    report["scoring"] = scoring

    canonical = [
        {key: row[key] for key in sorted(row) if key != "base_scores"}
        for row in sorted(triples, key=lambda row: (row["query_id"],
                                                    row["negative"]["chunk_id"]))
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "contains_text": True,
        "private_blind_set": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "F3",
        "held_constant_vs_f1_f2": {
            "base_model": BASE_MODEL,
            "scoring_text_formatter": "compact32",
            "objective": "weighted_pairwise_logistic_margin",
            "trainable_encoder_layers": 2,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-5,
            "pairwise_margin": 0.2,
            "same_document_weight": 1.2,
            "candidate_pool_rrf_gate": "unchanged",
        },
        "changed_vs_f1_f2": [
            "训练语料：V1 Train 45 锚点 → V1 Train + V2-A Train 合并",
            "V2-A 侧加入经独立复核确认的生产真实 false winner 作为额外负例",
            "按锚点归一化的 pair_weight（缺省不存在时训练行为与 F1/F2 逐字节一致）",
        ],
        "sources": {
            "v1": {
                "cases": str(Path(args.v1_cases).name),
                "cases_sha256": _sha256_file(Path(args.v1_cases)),
                "dataset_id": v1["source_dataset_id"],
                "summary": v1["summary"],
            },
            "v2a": {
                "cases": str(Path(args.v2a_cases).name),
                "cases_sha256": _sha256_file(Path(args.v2a_cases)),
                "dataset_id": v2a["source_dataset_id"],
                "summary": v2a["summary"],
            },
        },
        "provenance": {
            "index": v1["provenance"]["index"],
            "scoring_text_formatter": v1["provenance"]["scoring_text_formatter"],
            "blind_set_used": False,
            "sealed_set_used": False,
        },
        "split_contract": {
            "method": "preassigned_source_group_and_anchor_split_per_wave",
            "train_anchor_ids": sorted(train_anchors),
            "dev_anchor_ids": sorted(dev_anchors),
        },
        "mined_negatives": mined_stats,
        "anchor_normalization": weight_stats,
        "difficulty": report,
        "summary": {
            "triple_count": len(triples),
            "train_triple_count": sum(row["split"] == "train" for row in triples),
            "dev_triple_count": sum(row["split"] == "dev" for row in triples),
            "train_anchor_count": len(train_anchors),
            "dev_anchor_count": len(dev_anchors),
            "by_source_wave": dict(sorted(
                Counter(row["source_wave"] for row in triples).items()
            )),
            "by_negative_kind": dict(sorted(
                Counter(row["negative_kind"] for row in triples).items()
            )),
            "by_negative_origin": dict(sorted(
                Counter(row["negative_origin"] for row in triples).items()
            )),
        },
        "canonical_triples_sha256": _sha256_payload(canonical),
        "triples": sorted(triples, key=lambda row: (row["query_id"],
                                                    row["negative"]["chunk_id"])),
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-cases", default=str(V1_CASES))
    parser.add_argument("--v2a-cases", default=str(V2A_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--mined-adjudication",
        default=str(ROOT / "docs/rag_eval/colloquial"
                    / "v2a_train_mined_negative_adjudication_20260826.json"),
        help="independently adjudicated mined negatives; empty to skip",
    )
    parser.add_argument("--weighting", choices=WEIGHTING_MODES,
                        default="anchor_normalized",
                        help="anchor_normalized reproduces the frozen F3 "
                             "package byte-for-byte; corrective additionally "
                             "weights each pair by the base model's error")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if (args.weighting != "anchor_normalized"
            and Path(args.output).expanduser().resolve() == DEFAULT_OUTPUT.resolve()):
        raise SystemExit(
            "refusing to overwrite the frozen F3 package with a different "
            "weighting; pass an explicit --output (and --report)"
        )

    artifact = build(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "schema_version": "colloquial-f3-data-difficulty-v1",
        "experiment": "F3",
        "package": output.name,
        "package_sha256": _sha256_file(output),
        "canonical_triples_sha256": artifact["canonical_triples_sha256"],
        "sources": artifact["sources"],
        "anchor_normalization": artifact["anchor_normalization"],
        "summary": artifact["summary"],
        "difficulty": artifact["difficulty"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "report": str(report_path),
        "summary": artifact["summary"],
        "verdict": artifact["difficulty"]["verdict"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
