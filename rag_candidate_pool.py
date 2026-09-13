# -*- coding: utf-8 -*-
"""Pure candidate-pool alternatives for retrieval Stage C.

The production retriever currently fuses dense and BM25 results with RRF and
then truncates the union.  This module makes that *candidate selection* step
independently testable without loading a cross encoder.  Identity is always a
stable ``chunk_id``; document text is deliberately not accepted by the API and
therefore cannot accidentally become a deduplication key.

Nothing in this module selects a production default.  Callers must name an
explicit strategy and may compare its output with ``baseline_rrf20``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePath
from typing import Iterable, Literal


PoolStrategy = Literal[
    "baseline_rrf20",
    "source_cap4",
    "retain_channel_exclusives",
]


@dataclass(frozen=True)
class ChannelCandidate:
    """One result from one retrieval channel.

    ``rank`` is one-based.  ``score`` is diagnostic only: RRF intentionally
    uses ranks.  ``distance`` is retained for dense-channel audit output.
    """

    chunk_id: str
    source: str
    rank: int
    score: float | None = None
    distance: float | None = None

    def __post_init__(self) -> None:
        if not str(self.chunk_id).strip():
            raise ValueError("candidate chunk_id must be non-empty")
        if int(self.rank) < 1:
            raise ValueError("candidate rank must be one-based")


@dataclass(frozen=True)
class PoolCandidate:
    """A stable-ID union record shared by all candidate-pool strategies."""

    chunk_id: str
    source: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    first_seen: int = 0

    @property
    def channels(self) -> tuple[str, ...]:
        channels = []
        if self.dense_rank is not None:
            channels.append("dense")
        if self.bm25_rank is not None:
            channels.append("bm25")
        return tuple(channels)

    @property
    def channel_class(self) -> str:
        if self.dense_rank is not None and self.bm25_rank is not None:
            return "shared"
        if self.dense_rank is not None:
            return "dense_only"
        return "bm25_only"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_distance": self.dense_distance,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "channels": list(self.channels),
            "channel_class": self.channel_class,
        }


def _source_key(source: str) -> str:
    """Return a stable source bucket without interpreting arbitrary paths."""

    value = str(source or "").strip().replace("\\", "/")
    return PurePath(value).name.casefold() if value else "<unknown>"


def _dedupe_channel(candidates: Iterable[ChannelCandidate]) -> list[ChannelCandidate]:
    """Keep the earliest rank for duplicate IDs, never dedupe by text/source."""

    by_id: dict[str, ChannelCandidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.chunk_id)
        if existing is None or candidate.rank < existing.rank:
            by_id[candidate.chunk_id] = candidate
    return sorted(by_id.values(), key=lambda item: (item.rank, item.chunk_id))


def rrf_union(
    dense_candidates: Iterable[ChannelCandidate],
    bm25_candidates: Iterable[ChannelCandidate],
    *,
    rrf_k: int = 60,
) -> list[PoolCandidate]:
    """Build a deterministic RRF-ranked union keyed strictly by chunk ID.

    Ties reproduce Python's stable production ordering: dense first-seen order
    precedes BM25-only order.  ``chunk_id`` is a final deterministic tie-break.
    A stable ID that claims two different sources is rejected instead of
    silently merging corrupt lineage.
    """

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    dense = _dedupe_channel(dense_candidates)
    bm25 = _dedupe_channel(bm25_candidates)
    by_id: dict[str, PoolCandidate] = {}

    for position, candidate in enumerate(dense):
        by_id[candidate.chunk_id] = PoolCandidate(
            chunk_id=candidate.chunk_id,
            source=candidate.source,
            dense_rank=candidate.rank,
            dense_distance=candidate.distance,
            rrf_score=1.0 / (rrf_k + candidate.rank),
            first_seen=position,
        )

    offset = len(dense)
    for position, candidate in enumerate(bm25):
        existing = by_id.get(candidate.chunk_id)
        if existing is not None:
            left = _source_key(existing.source)
            right = _source_key(candidate.source)
            if left != right and left != "<unknown>" and right != "<unknown>":
                raise ValueError(
                    "same chunk_id has conflicting sources: "
                    f"{candidate.chunk_id!r}: {existing.source!r} vs {candidate.source!r}"
                )
            by_id[candidate.chunk_id] = replace(
                existing,
                source=existing.source or candidate.source,
                bm25_rank=candidate.rank,
                bm25_score=candidate.score,
                rrf_score=existing.rrf_score + 1.0 / (rrf_k + candidate.rank),
            )
        else:
            by_id[candidate.chunk_id] = PoolCandidate(
                chunk_id=candidate.chunk_id,
                source=candidate.source,
                bm25_rank=candidate.rank,
                bm25_score=candidate.score,
                rrf_score=1.0 / (rrf_k + candidate.rank),
                first_seen=offset + position,
            )

    return sorted(
        by_id.values(),
        key=lambda item: (-item.rrf_score, item.first_seen, item.chunk_id),
    )


def select_candidate_pool(
    dense_candidates: Iterable[ChannelCandidate],
    bm25_candidates: Iterable[ChannelCandidate],
    *,
    strategy: PoolStrategy,
    pool_size: int = 20,
    max_per_source: int = 4,
    exclusive_per_channel: int = 4,
    rrf_k: int = 60,
) -> list[PoolCandidate]:
    """Select one explicit Stage-C candidate pool.

    Strategies:

    ``baseline_rrf20``
        Production-equivalent stable-ID RRF truncation.
    ``source_cap4``
        Walk the same RRF union while allowing at most four chunks from one
        source; later sources backfill the pool.
    ``retain_channel_exclusives``
        Reserve the top four dense-only and top four BM25-only candidates
        before filling from the RRF union.  The final selected set is returned
        in global RRF order so the strategy changes membership, not scoring.
    """

    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    ranked = rrf_union(dense_candidates, bm25_candidates, rrf_k=rrf_k)

    if strategy == "baseline_rrf20":
        return ranked[:pool_size]

    if strategy == "source_cap4":
        if max_per_source < 1:
            raise ValueError("max_per_source must be positive")
        counts: dict[str, int] = {}
        selected: list[PoolCandidate] = []
        for candidate in ranked:
            source = _source_key(candidate.source)
            if counts.get(source, 0) >= max_per_source:
                continue
            selected.append(candidate)
            counts[source] = counts.get(source, 0) + 1
            if len(selected) >= pool_size:
                break
        return selected

    if strategy == "retain_channel_exclusives":
        if exclusive_per_channel < 0:
            raise ValueError("exclusive_per_channel cannot be negative")
        dense_only = sorted(
            (item for item in ranked if item.channel_class == "dense_only"),
            key=lambda item: (item.dense_rank or 10**9, item.chunk_id),
        )
        bm25_only = sorted(
            (item for item in ranked if item.channel_class == "bm25_only"),
            key=lambda item: (item.bm25_rank or 10**9, item.chunk_id),
        )
        reserved_ids = {
            item.chunk_id
            for item in dense_only[:exclusive_per_channel]
            + bm25_only[:exclusive_per_channel]
        }
        # If the caller asks for a very small pool, retain candidates in the
        # same global order rather than favouring whichever channel was added
        # first.
        if len(reserved_ids) > pool_size:
            stable_reserved_order = [
                item.chunk_id for item in ranked
                if item.chunk_id in reserved_ids
            ]
            reserved_ids = set(stable_reserved_order[:pool_size])
        selected_ids = set(reserved_ids)
        for item in ranked:
            if len(selected_ids) >= pool_size:
                break
            selected_ids.add(item.chunk_id)
        return [item for item in ranked if item.chunk_id in selected_ids][:pool_size]

    raise ValueError(f"unknown candidate-pool strategy: {strategy}")


def source_distribution(pool: Iterable[PoolCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in pool:
        key = _source_key(candidate.source)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def protect_multichannel_consensus(
    docs: list[str], metas: list[dict], dists: list[float],
    scores: list[float | None], *, dense_chunk_ids: list[str],
    bm25_chunk_ids: list[str], top_k: int = 5, margin: float = 0.03,
) -> tuple[list[str], list[dict], list[float], list[float | None], dict]:
    """Protect a dual-channel candidate only when reranker scores are tied.

    The helper is deliberately label-blind: stable IDs appearing in both
    channel Top-Ks define consensus.  A consensus candidate may move to first
    only when its reranker score trails the raw winner by at most ``margin``.
    """

    if top_k < 1 or margin < 0 or len(docs) < 2 or len(scores) != len(docs):
        return docs, metas, dists, scores, {"applied": False, "reason": "ineligible"}
    from rag_retrieval_trace import stable_chunk_id

    consensus = set(dense_chunk_ids[:top_k]) & set(bm25_chunk_ids[:top_k])
    if not consensus:
        return docs, metas, dists, scores, {"applied": False, "reason": "no_consensus"}
    eligible = [
        index for index, (document, metadata, score) in enumerate(zip(docs, metas, scores))
        if stable_chunk_id(document, metadata) in consensus and score is not None
    ]
    if not eligible:
        return docs, metas, dists, scores, {
            "applied": False, "reason": "consensus_not_scored",
        }
    best_index = max(eligible, key=lambda index: float(scores[index]))
    if best_index == 0:
        return docs, metas, dists, scores, {"applied": False, "reason": "already_top1"}
    winner_score = scores[0]
    consensus_score = scores[best_index]
    if winner_score is None or consensus_score is None:
        return docs, metas, dists, scores, {"applied": False, "reason": "missing_score"}
    delta = float(winner_score) - float(consensus_score)
    if delta > margin:
        return docs, metas, dists, scores, {
            "applied": False, "reason": "margin_too_large", "score_delta": delta,
        }

    def promote(values: list):
        return [values[best_index], *values[:best_index], *values[best_index + 1:]]

    promoted_id = stable_chunk_id(docs[best_index], metas[best_index])
    return (*[promote(values) for values in (docs, metas, dists, scores)], {
        "applied": True,
        "promoted_chunk_id": promoted_id,
        "from_rank": best_index + 1,
        "score_delta": delta,
        "top_k": top_k,
        "margin": margin,
    })


__all__ = [
    "ChannelCandidate",
    "PoolCandidate",
    "PoolStrategy",
    "rrf_union",
    "select_candidate_pool",
    "source_distribution",
    "protect_multichannel_consensus",
]
