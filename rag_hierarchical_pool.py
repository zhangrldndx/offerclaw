# -*- coding: utf-8 -*-
"""Qrels-blind source→section candidate selection for offline experiments.

This module is deliberately independent from the production retriever.  It
accepts collection-derived source/title metadata and stable chunk IDs, and it
never accepts relevance labels, document text, or evaluation query IDs.

The preregistered experiment keeps a 20-item final pool:

* the first 16 production RRF candidates are protected;
* at most 4 new candidates come from a source-restricted dense lookup;
* any unused hierarchy slots are backfilled from the production RRF tail.

All identity and deduplication uses ``chunk_id``.  Source and title are ranking
metadata only and can never collapse two different chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import PurePath
import re
from typing import Iterable, Sequence

from rag_candidate_pool import PoolCandidate


PREREGISTERED_HIERARCHY_CONFIG = {
    "pool_size": 20,
    "protected_global_slots": 16,
    "hierarchical_slots": 4,
    "source_limit": 4,
    "restricted_dense_depth": 20,
    "catalog_rrf_k": 60,
    "candidate_rrf_k": 60,
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _source_key(source: str) -> str:
    value = _clean(source).replace("\\", "/")
    return PurePath(value).name.casefold() if value else "<unknown>"


def _humanize_source(source: str) -> str:
    name = PurePath(_clean(source).replace("\\", "/")).name
    stem = name.rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem).strip()


@dataclass(frozen=True)
class CatalogEntry:
    catalog_id: str
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class RankedCatalogEntry:
    catalog_id: str
    source: str
    title: str
    dense_rank: int
    lexical_rank: int | None
    rrf_score: float


@dataclass(frozen=True)
class RankedSource:
    source: str
    rank: int
    best_catalog_id: str
    best_title: str
    best_rrf_score: float
    sections: tuple[RankedCatalogEntry, ...]


@dataclass(frozen=True)
class RestrictedCandidate:
    chunk_id: str
    source: str
    title: str
    dense_rank: int
    distance: float | None = None

    def __post_init__(self) -> None:
        if not _clean(self.chunk_id):
            raise ValueError("restricted candidate chunk_id must be non-empty")
        if self.dense_rank < 1:
            raise ValueError("restricted candidate dense_rank must be one-based")


@dataclass(frozen=True)
class RankedRestrictedCandidate:
    chunk_id: str
    source: str
    title: str
    dense_rank: int
    source_rank: int
    section_rank: int | None
    hierarchy_score: float
    distance: float | None = None


@dataclass(frozen=True)
class HierarchicalPoolItem:
    chunk_id: str
    source: str
    origin: str
    global_rank: int | None
    hierarchy_rank: int | None


@dataclass(frozen=True)
class HierarchicalPoolResult:
    pool: tuple[HierarchicalPoolItem, ...]
    ranked_restricted: tuple[RankedRestrictedCandidate, ...]
    hierarchy_added_ids: tuple[str, ...]
    protected_global_ids: tuple[str, ...]
    backfilled_global_ids: tuple[str, ...]


def build_source_section_catalog(
    metadatas: Iterable[dict],
) -> list[CatalogEntry]:
    """Build one stable source entry plus every unique source/title entry.

    The catalog uses metadata only.  Adding an explicit source-only entry lets
    a query match a meaningful filename even when its section titles are
    generic (for example, ``Overview`` or ``Summary``).
    """

    pairs: dict[tuple[str, str], tuple[str, str]] = {}
    sources: dict[str, str] = {}
    for raw_meta in metadatas:
        meta = raw_meta or {}
        source = _clean(meta.get("source"))
        if not source:
            continue
        source_key = _source_key(source)
        existing = sources.get(source_key)
        if existing is not None and _clean(existing) != source:
            raise ValueError(
                f"source catalog has case/path collision: {existing!r} vs {source!r}"
            )
        sources[source_key] = source
        title = _clean(meta.get("title"))
        if title:
            pairs[(source_key, title.casefold())] = (source, title)

    raw_entries = [
        (source, "") for _key, source in sorted(sources.items())
    ] + [
        value for _key, value in sorted(
            pairs.items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    entries = []
    for source, title in raw_entries:
        identity = f"{_source_key(source)}\0{title.casefold()}"
        catalog_id = "catalog_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]
        human_source = _humanize_source(source)
        text = human_source if not title else f"{human_source}\n{title}"
        entries.append(CatalogEntry(
            catalog_id=catalog_id,
            source=source,
            title=title,
            text=text,
        ))
    return sorted(entries, key=lambda item: item.catalog_id)


def rank_source_catalog(
    entries: Sequence[CatalogEntry],
    dense_scores: Sequence[float],
    lexical_scores: Sequence[float],
    *,
    source_limit: int = 4,
    rrf_k: int = 60,
) -> list[RankedSource]:
    """Fuse semantic and lexical catalog ranks, then take unique sources.

    Scores are supplied by the evaluator so this pure module has no model or
    tokeniser dependency.  Non-positive lexical scores do not create a sparse
    rank.  Every tie is broken by the immutable catalog ID.
    """

    if len(entries) != len(dense_scores) or len(entries) != len(lexical_scores):
        raise ValueError("catalog entries and score vectors must align")
    if source_limit < 1 or rrf_k < 1:
        raise ValueError("source_limit and rrf_k must be positive")
    if not entries:
        return []

    dense_order = sorted(
        range(len(entries)),
        key=lambda index: (-float(dense_scores[index]), entries[index].catalog_id),
    )
    dense_rank = {index: rank for rank, index in enumerate(dense_order, start=1)}
    lexical_order = sorted(
        (index for index, score in enumerate(lexical_scores) if float(score) > 0),
        key=lambda index: (-float(lexical_scores[index]), entries[index].catalog_id),
    )
    lexical_rank = {
        index: rank for rank, index in enumerate(lexical_order, start=1)
    }
    ranked_entries = []
    for index, entry in enumerate(entries):
        sparse_rank = lexical_rank.get(index)
        score = 1.0 / (rrf_k + dense_rank[index])
        if sparse_rank is not None:
            score += 1.0 / (rrf_k + sparse_rank)
        ranked_entries.append(RankedCatalogEntry(
            catalog_id=entry.catalog_id,
            source=entry.source,
            title=entry.title,
            dense_rank=dense_rank[index],
            lexical_rank=sparse_rank,
            rrf_score=score,
        ))
    ranked_entries.sort(key=lambda item: (
        -item.rrf_score,
        item.dense_rank,
        item.lexical_rank if item.lexical_rank is not None else math.inf,
        item.catalog_id,
    ))

    by_source: dict[str, list[RankedCatalogEntry]] = {}
    source_names: dict[str, str] = {}
    for entry in ranked_entries:
        key = _source_key(entry.source)
        source_names[key] = entry.source
        by_source.setdefault(key, []).append(entry)
    ordered_source_keys = sorted(
        by_source,
        key=lambda key: (
            -by_source[key][0].rrf_score,
            by_source[key][0].dense_rank,
            by_source[key][0].catalog_id,
            key,
        ),
    )[:source_limit]
    return [
        RankedSource(
            source=source_names[key],
            rank=rank,
            best_catalog_id=by_source[key][0].catalog_id,
            best_title=by_source[key][0].title,
            best_rrf_score=by_source[key][0].rrf_score,
            sections=tuple(by_source[key]),
        )
        for rank, key in enumerate(ordered_source_keys, start=1)
    ]


def _rank_restricted(
    candidates: Iterable[RestrictedCandidate],
    ranked_sources: Sequence[RankedSource],
    *,
    rrf_k: int,
) -> list[RankedRestrictedCandidate]:
    source_by_key = {_source_key(item.source): item for item in ranked_sources}
    section_ranks: dict[tuple[str, str], int] = {}
    for source in ranked_sources:
        source_key = _source_key(source.source)
        titled = [section for section in source.sections if section.title]
        for rank, section in enumerate(titled, start=1):
            section_ranks[(source_key, section.title.casefold())] = rank

    by_id: dict[str, RestrictedCandidate] = {}
    for candidate in candidates:
        source = source_by_key.get(_source_key(candidate.source))
        if source is None:
            raise ValueError(
                f"restricted candidate source was not selected: {candidate.source!r}"
            )
        existing = by_id.get(candidate.chunk_id)
        if existing is None or candidate.dense_rank < existing.dense_rank:
            by_id[candidate.chunk_id] = candidate
        elif _source_key(existing.source) != _source_key(candidate.source):
            raise ValueError(
                f"same restricted chunk_id has conflicting sources: {candidate.chunk_id!r}"
            )

    ranked = []
    for candidate in by_id.values():
        source = source_by_key[_source_key(candidate.source)]
        section_rank = section_ranks.get((
            _source_key(candidate.source), candidate.title.casefold(),
        )) if candidate.title else None
        score = (
            1.0 / (rrf_k + source.rank)
            + 1.0 / (rrf_k + candidate.dense_rank)
        )
        if section_rank is not None:
            score += 1.0 / (rrf_k + section_rank)
        ranked.append(RankedRestrictedCandidate(
            chunk_id=candidate.chunk_id,
            source=candidate.source,
            title=candidate.title,
            dense_rank=candidate.dense_rank,
            source_rank=source.rank,
            section_rank=section_rank,
            hierarchy_score=score,
            distance=candidate.distance,
        ))
    return sorted(ranked, key=lambda item: (
        -item.hierarchy_score,
        item.source_rank,
        item.section_rank if item.section_rank is not None else math.inf,
        item.dense_rank,
        item.chunk_id,
    ))


def select_hierarchical_pool(
    baseline_pool: Sequence[PoolCandidate],
    restricted_candidates: Iterable[RestrictedCandidate],
    ranked_sources: Sequence[RankedSource],
    *,
    pool_size: int = 20,
    protected_global_slots: int = 16,
    hierarchical_slots: int = 4,
    rrf_k: int = 60,
) -> HierarchicalPoolResult:
    """Combine protected production RRF membership with hierarchy candidates."""

    if pool_size < 1 or protected_global_slots < 0 or hierarchical_slots < 0:
        raise ValueError("pool and allocation sizes cannot be negative")
    if protected_global_slots + hierarchical_slots != pool_size:
        raise ValueError("preregistered global and hierarchy slots must fill pool_size")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    baseline = list(baseline_pool)
    baseline_ids = [item.chunk_id for item in baseline]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("baseline pool contains duplicate stable chunk IDs")
    protected = baseline[:protected_global_slots]
    baseline_source = {item.chunk_id: _source_key(item.source) for item in baseline}
    selected_ids = {item.chunk_id for item in protected}
    ranked_restricted = _rank_restricted(
        restricted_candidates, ranked_sources, rrf_k=rrf_k,
    )

    hierarchy_added = []
    for candidate in ranked_restricted:
        # A hierarchy slot must add new membership, not merely move an item
        # from the production tail.  Validate lineage and skip every baseline
        # ID before counting the fixed four hierarchy slots.
        if candidate.chunk_id in baseline_source:
            if baseline_source[candidate.chunk_id] != _source_key(candidate.source):
                raise ValueError("hierarchy candidate conflicts with baseline lineage")
            continue
        hierarchy_added.append(candidate)
        selected_ids.add(candidate.chunk_id)
        if len(hierarchy_added) >= hierarchical_slots:
            break

    backfilled = []
    for candidate in baseline[protected_global_slots:]:
        if len(selected_ids) >= pool_size:
            break
        if candidate.chunk_id in selected_ids:
            continue
        backfilled.append(candidate)
        selected_ids.add(candidate.chunk_id)

    global_rank = {item.chunk_id: rank for rank, item in enumerate(baseline, start=1)}
    hierarchy_rank = {
        item.chunk_id: rank for rank, item in enumerate(ranked_restricted, start=1)
    }
    ordered = [
        HierarchicalPoolItem(
            chunk_id=item.chunk_id,
            source=item.source,
            origin="global_protected",
            global_rank=global_rank[item.chunk_id],
            hierarchy_rank=hierarchy_rank.get(item.chunk_id),
        )
        for item in protected
    ] + [
        HierarchicalPoolItem(
            chunk_id=item.chunk_id,
            source=item.source,
            origin="hierarchical",
            global_rank=global_rank.get(item.chunk_id),
            hierarchy_rank=hierarchy_rank[item.chunk_id],
        )
        for item in hierarchy_added
    ] + [
        HierarchicalPoolItem(
            chunk_id=item.chunk_id,
            source=item.source,
            origin="global_backfill",
            global_rank=global_rank[item.chunk_id],
            hierarchy_rank=hierarchy_rank.get(item.chunk_id),
        )
        for item in backfilled
    ]
    if len(ordered) > pool_size:
        raise AssertionError("hierarchical pool exceeded preregistered size")
    return HierarchicalPoolResult(
        pool=tuple(ordered),
        ranked_restricted=tuple(ranked_restricted),
        hierarchy_added_ids=tuple(item.chunk_id for item in hierarchy_added),
        protected_global_ids=tuple(item.chunk_id for item in protected),
        backfilled_global_ids=tuple(item.chunk_id for item in backfilled),
    )


__all__ = [
    "CatalogEntry",
    "HierarchicalPoolItem",
    "HierarchicalPoolResult",
    "PREREGISTERED_HIERARCHY_CONFIG",
    "RankedCatalogEntry",
    "RankedRestrictedCandidate",
    "RankedSource",
    "RestrictedCandidate",
    "build_source_section_catalog",
    "rank_source_catalog",
    "select_hierarchical_pool",
]
