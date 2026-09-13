# -*- coding: utf-8 -*-
"""Contracts and audits for the private answerability-student dataset.

The old answerability cache is deliberately *not* treated as a dataset.  A
cache entry becomes reusable only when the caller supplies the exact original
question, retrieval question, chunk text and judge model that reproduce its
v4 key.  This module also keeps blind rows out of the training process and
turns the F3 post-mortem into executable pre-training gates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "answerability-student-v1"
SPLITS = ("train", "validation", "blind_a", "blind_b")
TRAINING_SPLITS = ("train", "validation")
QUERY_STYLES = ("standard", "natural", "implicit_oral", "long_noisy")
PAIR_MIX = {
    "corrective": 0.50,
    "same_source_hard": 0.25,
    "cross_source_hard": 0.15,
    "stability": 0.10,
}


class StudentDataError(ValueError):
    """Raised when a dataset could leak, drift, or repeat an old failure."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return sha256_text(payload)


def _require_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise StudentDataError(f"missing non-empty {field}")
    return value


def validate_row(row: dict[str, Any], *, require_label: bool = True) -> dict[str, Any]:
    """Validate one expanded ``(question, candidate)`` row.

    Raw text is private by contract.  Public manifests are produced by
    :func:`public_manifest` and never copy these fields.
    """

    if row.get("schema_version") != SCHEMA:
        raise StudentDataError("unsupported answerability student schema")
    for field in (
        "query_id", "anchor_id", "source_id", "chunk_id", "question",
        "chunk_text", "query_style", "split", "teacher_model",
        "teacher_prompt_sha256",
    ):
        _require_text(row, field)
    prompt_hash = row["teacher_prompt_sha256"]
    if len(prompt_hash) != 64 or any(char not in "0123456789abcdef" for char in prompt_hash):
        raise StudentDataError("teacher_prompt_sha256 must be lowercase SHA256")
    if row["split"] not in SPLITS:
        raise StudentDataError(f"invalid split {row['split']!r}")
    if row["query_style"] not in QUERY_STYLES:
        raise StudentDataError(f"invalid query_style {row['query_style']!r}")
    expected_hash = sha256_text(row["chunk_text"])
    if row.get("chunk_text_sha256") != expected_hash:
        raise StudentDataError("chunk_text_sha256 does not match private text")
    rank = row.get("current_bge_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise StudentDataError("current_bge_rank must be a positive integer")
    for field in ("current_bge_score", "base_student_score"):
        value = row.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise StudentDataError(f"{field} must be numeric or null")
    if require_label:
        grade = row.get("teacher_grade")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade not in (0, 1, 2, 3):
            raise StudentDataError("teacher_grade must be one of 0,1,2,3")
        relation = row.get("teacher_relation")
        if relation not in ("entails", "contradicts", "not_established"):
            raise StudentDataError("invalid teacher_relation")
        votes = row.get("teacher_votes")
        if not isinstance(votes, list) or not votes:
            raise StudentDataError("labeled rows require teacher_votes")
        for vote in votes:
            if not isinstance(vote, dict):
                raise StudentDataError("teacher_votes entries must be objects")
            vote_prompt = vote.get("prompt_sha256")
            if vote_prompt is not None and (
                not isinstance(vote_prompt, str) or len(vote_prompt) != 64
                or any(char not in "0123456789abcdef" for char in vote_prompt)
            ):
                raise StudentDataError("vote prompt_sha256 must be lowercase SHA256")
    return row


def read_jsonl(path: str | Path, *, require_label: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudentDataError(f"invalid JSON on line {line_no}") from exc
        rows.append(validate_row(row, require_label=require_label))
    if not rows:
        raise StudentDataError("student dataset is empty")
    return rows


def training_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return train/validation rows and reject a mixed file containing blind labels."""

    material = list(rows)
    forbidden = sorted({row.get("split") for row in material if row.get("split") not in TRAINING_SPLITS})
    if forbidden:
        raise StudentDataError(
            "training input contains sealed split(s): " + ", ".join(forbidden)
        )
    return [validate_row(row) for row in material]


def reconnect_cached_verdict(
    row: dict[str, Any], cache: dict[str, Any], *, model: str,
) -> dict[str, Any] | None:
    """Reuse a hash-only v4 cache entry only after exact identity reconstruction."""

    from rag_answerability import PROMPT_SHA256, cache_key

    if row.get("teacher_prompt_sha256") != PROMPT_SHA256:
        return None
    if row.get("teacher_model") != model:
        return None

    question = _require_text(row, "question")
    chunk = _require_text(row, "chunk_text")
    retrieval_question = str(row.get("retrieval_question") or question)
    key = cache_key(
        model, question, chunk, retrieval_question=retrieval_question,
    )
    verdict = cache.get(key)
    if not isinstance(verdict, dict):
        return None
    grade = verdict.get("grade")
    relation = verdict.get("relation")
    if grade not in (0, 1, 2, 3) or relation not in (
        "entails", "contradicts", "not_established",
    ):
        return None
    return dict(verdict)


def audit_split_isolation(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Enforce query/anchor/source boundaries before any model sees labels."""

    material = list(rows)
    query_splits: dict[str, set[str]] = defaultdict(set)
    anchor_splits: dict[str, set[str]] = defaultdict(set)
    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for row in material:
        validate_row(row)
        split = row["split"]
        query_splits[row["query_id"]].add(split)
        anchor_splits[row["anchor_id"]].add(split)
        sources_by_split[split].add(row["source_id"])

    query_leaks = sorted(key for key, values in query_splits.items() if len(values) > 1)
    anchor_leaks = sorted(key for key, values in anchor_splits.items() if len(values) > 1)
    if query_leaks:
        raise StudentDataError(f"query split leakage: {query_leaks[:3]}")
    if anchor_leaks:
        raise StudentDataError(f"anchor split leakage: {anchor_leaks[:3]}")

    train_sources = sources_by_split["train"]
    validation_overlap = train_sources & sources_by_split["validation"]
    if validation_overlap:
        raise StudentDataError(
            f"train/validation source leakage: {sorted(validation_overlap)[:3]}"
        )
    blind_b_overlap = sources_by_split["blind_b"] & (
        train_sources | sources_by_split["validation"] | sources_by_split["blind_a"]
    )
    if blind_b_overlap:
        raise StudentDataError(
            f"blind_b source leakage: {sorted(blind_b_overlap)[:3]}"
        )
    return {
        "queries": len(query_splits),
        "anchors": len(anchor_splits),
        "sources_by_split": {
            split: len(sources_by_split[split]) for split in SPLITS
        },
    }


def _score(row: dict[str, Any], field: str, *, descending: bool = True) -> float:
    value = row.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if field == "current_bge_score":
        # A rank is always present, so old traces without a score remain usable.
        return -float(row["current_bge_rank"])
    return float("-inf") if descending else float("inf")


def build_strong_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build grade-gap>=2 pairs; grade 2 is never a strong side.

    Categories are mutually exclusive so their *effective* weights can be
    checked.  Corrective pairs take priority; a deterministic fifth of already
    correct pairs becomes stability coverage, with the rest split by source.
    """

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_row(row)
        if row["split"] not in TRAINING_SPLITS:
            continue
        if row.get("label_status") == "teacher_disagreement":
            continue
        if row.get("base_student_score") is None:
            # Missing is not an inversion.  F3 taught us that counting an
            # unmeasured row as difficult manufactures fake corrective mass.
            continue
        by_query[row["query_id"]].append(row)

    pairs: list[dict[str, Any]] = []
    for query_id, group in sorted(by_query.items()):
        highs = [row for row in group if row["teacher_grade"] == 3]
        lows = [row for row in group if row["teacher_grade"] in (0, 1)]
        for high in highs:
            for low in lows:
                high_score = _score(high, "base_student_score")
                low_score = _score(low, "base_student_score")
                corrective = high_score <= low_score
                pair_id = sha256_text(
                    "\0".join((query_id, high["chunk_id"], low["chunk_id"]))
                )
                if corrective:
                    category = "corrective"
                elif int(pair_id[:8], 16) % 5 == 0:
                    category = "stability"
                elif high["source_id"] == low["source_id"]:
                    category = "same_source_hard"
                else:
                    category = "cross_source_hard"
                pairs.append({
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "anchor_id": high["anchor_id"],
                    "split": high["split"],
                    "category": category,
                    "corrective": corrective,
                    "positive": high,
                    "negative": low,
                })
    # Preserve the frozen category mix while giving each source equal mass
    # *inside* a category.  This prevents the four huge documents from owning
    # the gradient without reviving F3's per-anchor normalisation bug (harder
    # anchors inside a source are not downweighted for having more negatives).
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sources_by_category: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        source = pair["positive"]["source_id"]
        buckets[(pair["category"], source)].append(pair)
        sources_by_category[pair["category"]].add(source)
    for (category, source), bucket in buckets.items():
        source_count = len(sources_by_category[category])
        source_mass = PAIR_MIX[category] / source_count if source_count else 0.0
        for pair in bucket:
            pair["sampling_weight"] = source_mass / len(bucket)
    return pairs


def pair_flip_report(
    pairs: Iterable[dict[str, Any]], post_scores: dict[tuple[str, str], float],
) -> dict[str, Any]:
    """Report actual order changes; loss and mean margin are intentionally absent."""

    corrected = regressed = stable_correct = stable_wrong = 0
    rows = []
    for pair in pairs:
        pos = pair["positive"]
        neg = pair["negative"]
        key_pos = (pair["query_id"], pos["chunk_id"])
        key_neg = (pair["query_id"], neg["chunk_id"])
        if key_pos not in post_scores or key_neg not in post_scores:
            continue
        pre_correct = _score(pos, "base_student_score") > _score(neg, "base_student_score")
        post_correct = post_scores[key_pos] > post_scores[key_neg]
        if not pre_correct and post_correct:
            corrected += 1
            change = "corrected"
        elif pre_correct and not post_correct:
            regressed += 1
            change = "regressed"
        elif post_correct:
            stable_correct += 1
            change = "stable_correct"
        else:
            stable_wrong += 1
            change = "stable_wrong"
        rows.append({"pair_id": pair["pair_id"], "change": change})
    return {
        "n": len(rows),
        "corrected": corrected,
        "regressed": regressed,
        "net_flips": corrected - regressed,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "rows": rows,
    }


def _r1_headroom(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            by_query[row["query_id"]].append(row)
    base_hits = teacher_hits = 0
    eligible = 0
    for group in by_query.values():
        if not any(row["teacher_grade"] == 3 for row in group):
            continue
        eligible += 1
        base = max(group, key=lambda row: _score(row, "current_bge_score"))
        teacher = max(
            group,
            key=lambda row: (row["teacher_grade"], _score(row, "current_bge_score")),
        )
        base_hits += int(base["teacher_grade"] == 3)
        teacher_hits += int(teacher["teacher_grade"] == 3)
    base_rate = base_hits / eligible if eligible else 0.0
    teacher_rate = teacher_hits / eligible if eligible else 0.0
    return {
        "n": eligible,
        "base_hits": base_hits,
        "teacher_hits": teacher_hits,
        "base_r1": base_rate,
        "teacher_r1": teacher_rate,
        "gain": teacher_rate - base_rate,
    }


def audit_pilot(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Executable pre-training gates from the approved plan."""

    material = list(rows)
    isolation = audit_split_isolation(material)
    pilot_rows = [row for row in material if row["split"] in TRAINING_SPLITS]
    pairs = build_strong_pairs(pilot_rows)
    corrective = [pair for pair in pairs if pair["corrective"]]
    corrective_anchors = {pair["anchor_id"] for pair in corrective}
    corrective_sources = {
        pair["positive"]["source_id"] for pair in corrective
    } | {
        pair["negative"]["source_id"] for pair in corrective
    }
    total_weight = sum(float(pair["sampling_weight"]) for pair in pairs)
    corrective_weight = sum(
        float(pair["sampling_weight"]) for pair in corrective
    )
    source_weight = Counter()
    for pair in pairs:
        source_weight[pair["positive"]["source_id"]] += float(pair["sampling_weight"])
    max_source_weight = max(source_weight.values(), default=0.0)
    audited = disagreements = 0
    for row in pilot_rows:
        votes = row.get("teacher_votes") or []
        # Only a deterministic random audit under the exact same prompt may
        # estimate teacher stability.  High-risk grade-2 rejudges and
        # cross-prompt comparisons are diagnostics, not prevalence estimates.
        random_votes = [
            vote for vote in votes if isinstance(vote, dict)
            and vote.get("audit_reason") == "random_10pct"
        ]
        for audit_vote in random_votes:
            prompt_hash = audit_vote.get("prompt_sha256")
            matching_base = next((
                vote for vote in votes if isinstance(vote, dict)
                and vote is not audit_vote
                and vote.get("audit_reason") == "base"
                and vote.get("prompt_sha256") == prompt_hash
            ), None)
            if matching_base is None:
                continue
            audited += 1
            disagreements += int(matching_base.get("grade") != audit_vote.get("grade"))
            break
    disagreement_rate = disagreements / audited if audited else 1.0
    audit_coverage = audited / len(pilot_rows) if pilot_rows else 0.0
    headroom = _r1_headroom(pilot_rows)
    checks = {
        "teacher_validation_r1_gain_gte_0_08": headroom["gain"] >= 0.08,
        "corrective_pairs_gte_100": len(corrective) >= 100,
        "corrective_anchors_gte_40": len(corrective_anchors) >= 40,
        "corrective_sources_gte_20": len(corrective_sources) >= 20,
        "corrective_weight_share_gte_0_25": (
            corrective_weight / total_weight if total_weight else 0.0
        ) >= 0.25,
        "teacher_disagreement_lte_0_15": disagreement_rate <= 0.15,
        "independent_vote_audit_gte_0_10": audit_coverage >= 0.10,
        "base_student_score_coverage_eq_1": all(
            row.get("base_student_score") is not None for row in pilot_rows
        ),
        "max_effective_source_weight_lte_0_15": max_source_weight <= 0.15,
    }
    return {
        "schema_version": "answerability-student-pilot-audit-v1",
        "status": "go_train_pilot" if all(checks.values()) else "stop_before_training",
        "checks": checks,
        "split_isolation": isolation,
        "rows": len(pilot_rows),
        "pairs": len(pairs),
        "corrective_pairs": len(corrective),
        "corrective_anchors": len(corrective_anchors),
        "corrective_sources": len(corrective_sources),
        "corrective_weight_share": (
            corrective_weight / total_weight if total_weight else 0.0
        ),
        "max_effective_source_weight": max_source_weight,
        "teacher_vote_audit_rows": audited,
        "teacher_disagreements": disagreements,
        "teacher_disagreement_rate": disagreement_rate,
        "teacher_validation_headroom": headroom,
    }


def public_manifest(rows: Iterable[dict[str, Any]], *, private_path: str | Path) -> dict[str, Any]:
    """Create a redacted, hash-addressed manifest without question/chunk text."""

    material = list(rows)
    path = Path(private_path)
    split_counts = Counter(row["split"] for row in material)
    grade_counts = Counter(str(row["teacher_grade"]) for row in material)
    prompt_hash_counts = Counter(row["teacher_prompt_sha256"] for row in material)
    vote_prompt_hash_counts = Counter(
        str(vote.get("prompt_sha256") or "missing")
        for row in material for vote in row.get("teacher_votes") or []
        if isinstance(vote, dict)
    )
    vote_reason_counts = Counter(
        str(vote.get("audit_reason") or "missing")
        for row in material for vote in row.get("teacher_votes") or []
        if isinstance(vote, dict)
    )
    label_status_counts = Counter(
        str(row.get("label_status") or "missing") for row in material
    )
    return {
        "schema_version": "answerability-student-manifest-v1",
        "dataset_schema": SCHEMA,
        "private_artifact": {
            "name": path.name,
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        },
        "rows": len(material),
        "queries": len({row["query_id"] for row in material}),
        "anchors": len({row["anchor_id"] for row in material}),
        "sources": len({row["source_id"] for row in material}),
        "split_counts": dict(sorted(split_counts.items())),
        "grade_counts": dict(sorted(grade_counts.items())),
        "teacher_prompt_sha256_counts": dict(sorted(prompt_hash_counts.items())),
        "teacher_vote_prompt_sha256_counts": dict(sorted(vote_prompt_hash_counts.items())),
        "teacher_vote_reason_counts": dict(sorted(vote_reason_counts.items())),
        "label_status_counts": dict(sorted(label_status_counts.items())),
        "teacher_votes_total": sum(vote_prompt_hash_counts.values()),
        "contains_private_text": True,
        "public_fields_exclude": ["question", "retrieval_question", "chunk_text"],
    }


__all__ = [
    "PAIR_MIX", "QUERY_STYLES", "SCHEMA", "SPLITS", "StudentDataError",
    "audit_pilot", "audit_split_isolation", "build_strong_pairs",
    "canonical_sha256", "pair_flip_report", "public_manifest", "read_jsonl",
    "reconnect_cached_verdict", "sha256_text", "training_rows", "validate_row",
]
