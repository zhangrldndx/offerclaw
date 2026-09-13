# -*- coding: utf-8 -*-
"""Pure membership logic for the isolated channel-source candidate shadow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePath
from typing import Iterable, Sequence

from rag_candidate_pool import PoolCandidate


SHADOW_CONFIG = {
    "pool_size": 20,
    "protected_global_slots": 16,
    "exact_unseen_slots": 4,
    "global_dense_depth": 20,
    "global_bm25_depth": 20,
}


def _source_key(source: str) -> str:
    value = str(source or "").strip().replace("\\", "/")
    return PurePath(value).name.casefold() if value else "<unknown>"


@dataclass(frozen=True)
class ExactSourceCandidate:
    chunk_id: str
    source: str
    cosine_score: float

    def __post_init__(self) -> None:
        if not str(self.chunk_id).strip():
            raise ValueError("exact candidate chunk_id must be non-empty")
        if not math.isfinite(float(self.cosine_score)):
            raise ValueError("exact candidate cosine_score must be finite")


@dataclass(frozen=True)
class ShadowPoolItem:
    chunk_id: str
    source: str
    origin: str
    global_rank: int | None
    exact_rank: int | None
    cosine_score: float | None


@dataclass(frozen=True)
class ShadowPoolResult:
    source_shortlist: tuple[str, ...]
    ranked_exact: tuple[ExactSourceCandidate, ...]
    pool: tuple[ShadowPoolItem, ...]
    exact_added_ids: tuple[str, ...]
    backfilled_global_ids: tuple[str, ...]


def source_shortlist_from_global(
    baseline_pool: Sequence[PoolCandidate],
    *,
    protected_global_slots: int = 16,
) -> list[str]:
    """Return all unique protected sources in first-occurrence order."""

    if protected_global_slots < 1:
        raise ValueError("protected_global_slots must be positive")
    selected = []
    seen = set()
    for candidate in list(baseline_pool)[:protected_global_slots]:
        key = _source_key(candidate.source)
        if key == "<unknown>":
            raise ValueError("protected global candidate has no source lineage")
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate.source)
    if not selected:
        raise ValueError("protected global pool produced an empty source shortlist")
    return selected


def rank_exact_candidates(
    candidates: Iterable[ExactSourceCandidate],
    *,
    allowed_sources: Sequence[str],
) -> list[ExactSourceCandidate]:
    allowed = {_source_key(source) for source in allowed_sources}
    if not allowed:
        raise ValueError("allowed_sources cannot be empty")
    by_id: dict[str, ExactSourceCandidate] = {}
    for candidate in candidates:
        source = _source_key(candidate.source)
        if source not in allowed:
            raise ValueError(
                f"exact candidate source is outside shortlist: {candidate.source!r}"
            )
        existing = by_id.get(candidate.chunk_id)
        if existing is not None and _source_key(existing.source) != source:
            raise ValueError(
                f"same chunk_id has conflicting source lineage: {candidate.chunk_id!r}"
            )
        if existing is None or candidate.cosine_score > existing.cosine_score:
            by_id[candidate.chunk_id] = candidate
    return sorted(
        by_id.values(),
        key=lambda item: (-item.cosine_score, item.chunk_id),
    )


def select_channel_source_pool(
    baseline_pool: Sequence[PoolCandidate],
    exact_candidates: Iterable[ExactSourceCandidate],
    *,
    pool_size: int = 20,
    protected_global_slots: int = 16,
    exact_unseen_slots: int = 4,
) -> ShadowPoolResult:
    """Protect global16, add top-4 exact unseen IDs, then global-backfill."""

    if protected_global_slots + exact_unseen_slots != pool_size:
        raise ValueError("protected and exact slots must exactly fill pool_size")
    baseline = list(baseline_pool)
    baseline_ids = [candidate.chunk_id for candidate in baseline]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("baseline pool contains duplicate stable chunk IDs")
    shortlist = source_shortlist_from_global(
        baseline,
        protected_global_slots=protected_global_slots,
    )
    ranked_exact = rank_exact_candidates(
        exact_candidates,
        allowed_sources=shortlist,
    )
    baseline_by_id = {candidate.chunk_id: candidate for candidate in baseline}
    baseline_source = {
        candidate.chunk_id: _source_key(candidate.source) for candidate in baseline
    }
    exact_added = []
    for candidate in ranked_exact:
        if candidate.chunk_id in baseline_by_id:
            if baseline_source[candidate.chunk_id] != _source_key(candidate.source):
                raise ValueError("exact candidate conflicts with baseline lineage")
            continue
        exact_added.append(candidate)
        if len(exact_added) >= exact_unseen_slots:
            break

    protected = baseline[:protected_global_slots]
    selected = {candidate.chunk_id for candidate in protected}
    selected.update(candidate.chunk_id for candidate in exact_added)
    backfilled = []
    for candidate in baseline[protected_global_slots:]:
        if len(selected) >= pool_size:
            break
        if candidate.chunk_id in selected:
            continue
        backfilled.append(candidate)
        selected.add(candidate.chunk_id)

    global_rank = {
        candidate.chunk_id: rank for rank, candidate in enumerate(baseline, start=1)
    }
    exact_rank = {
        candidate.chunk_id: rank
        for rank, candidate in enumerate(ranked_exact, start=1)
    }
    pool = [
        ShadowPoolItem(
            chunk_id=candidate.chunk_id,
            source=candidate.source,
            origin="global_protected",
            global_rank=global_rank[candidate.chunk_id],
            exact_rank=exact_rank.get(candidate.chunk_id),
            cosine_score=None,
        )
        for candidate in protected
    ] + [
        ShadowPoolItem(
            chunk_id=candidate.chunk_id,
            source=candidate.source,
            origin="source_exact_unseen",
            global_rank=None,
            exact_rank=exact_rank[candidate.chunk_id],
            cosine_score=candidate.cosine_score,
        )
        for candidate in exact_added
    ] + [
        ShadowPoolItem(
            chunk_id=candidate.chunk_id,
            source=candidate.source,
            origin="global_backfill",
            global_rank=global_rank[candidate.chunk_id],
            exact_rank=exact_rank.get(candidate.chunk_id),
            cosine_score=None,
        )
        for candidate in backfilled
    ]
    if len(pool) != min(pool_size, len(baseline) + len(exact_added)):
        raise AssertionError("shadow pool membership count is inconsistent")
    if len(pool) > pool_size:
        raise AssertionError("shadow pool exceeded fixed size")
    return ShadowPoolResult(
        source_shortlist=tuple(shortlist),
        ranked_exact=tuple(ranked_exact),
        pool=tuple(pool),
        exact_added_ids=tuple(candidate.chunk_id for candidate in exact_added),
        backfilled_global_ids=tuple(candidate.chunk_id for candidate in backfilled),
    )


__all__ = [
    "ExactSourceCandidate",
    "SHADOW_CONFIG",
    "ShadowPoolItem",
    "ShadowPoolResult",
    "rank_exact_candidates",
    "select_channel_source_pool",
    "source_shortlist_from_global",
]
