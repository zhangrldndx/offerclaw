from __future__ import annotations

from types import SimpleNamespace

from rag_candidate_pool import PoolCandidate
from rag_hierarchical_pool import (
    RestrictedCandidate,
    build_source_section_catalog,
    rank_source_catalog,
    select_hierarchical_pool,
)


def _baseline(size=20):
    return [
        PoolCandidate(
            chunk_id=f"global-{rank}",
            source=f"source-{(rank - 1) // 4}.md",
            dense_rank=rank,
            rrf_score=1 / (60 + rank),
            first_seen=rank,
        )
        for rank in range(1, size + 1)
    ]


def test_catalog_is_metadata_only_deduplicated_and_stable():
    metas = [
        {"source": "docs/RAG_Guide.md", "title": "Hybrid Retrieval"},
        {"source": "docs/RAG_Guide.md", "title": "Hybrid   Retrieval"},
        {"source": "Agent.md", "title": "Tools"},
    ]

    first = build_source_section_catalog(metas)
    second = build_source_section_catalog(reversed(metas))

    assert first == second
    assert len(first) == 4  # two source-only entries plus two unique sections
    assert len({entry.catalog_id for entry in first}) == 4
    assert all("document text" not in entry.text for entry in first)


def test_catalog_fusion_selects_unique_sources_with_stable_ties():
    catalog = build_source_section_catalog([
        {"source": "a.md", "title": "Dense"},
        {"source": "a.md", "title": "Sparse"},
        {"source": "b.md", "title": "Agent"},
        {"source": "c.md", "title": "Resume"},
    ])
    index = {entry.text: offset for offset, entry in enumerate(catalog)}
    dense = [0.0] * len(catalog)
    lexical = [0.0] * len(catalog)
    dense[index["a\nDense"]] = 1.0
    lexical[index["b\nAgent"]] = 2.0
    dense[index["c\nResume"]] = 0.5

    observed = [
        tuple(source.source for source in rank_source_catalog(
            catalog, dense, lexical, source_limit=3,
        ))
        for _ in range(20)
    ]

    assert len(set(observed)) == 1
    assert set(observed[0]) == {"a.md", "b.md", "c.md"}


def test_hierarchical_pool_protects_16_adds_4_and_uses_stable_ids():
    catalog = build_source_section_catalog([
        {"source": "target.md", "title": "Relevant"},
    ])
    ranked_sources = rank_source_catalog(
        catalog,
        [1.0] * len(catalog),
        [1.0] * len(catalog),
        source_limit=1,
    )
    restricted = [
        RestrictedCandidate(
            chunk_id=f"new-{rank}",
            source="target.md",
            title="Relevant",
            dense_rank=rank,
            distance=rank / 100,
        )
        for rank in range(1, 7)
    ]

    result = select_hierarchical_pool(
        _baseline(), restricted, ranked_sources,
    )

    assert len(result.pool) == 20
    assert [item.chunk_id for item in result.pool[:16]] == [
        f"global-{rank}" for rank in range(1, 17)
    ]
    assert result.hierarchy_added_ids == (
        "new-1", "new-2", "new-3", "new-4",
    )
    assert not result.backfilled_global_ids
    assert len({item.chunk_id for item in result.pool}) == 20


def test_hierarchy_duplicates_backfill_from_global_tail_deterministically():
    catalog = build_source_section_catalog([
        {"source": "source-4.md", "title": "Relevant"},
    ])
    ranked_sources = rank_source_catalog(
        catalog, [1.0] * len(catalog), [0.0] * len(catalog), source_limit=1,
    )
    restricted = [
        RestrictedCandidate(
            chunk_id="global-17",
            source="source-4.md",
            title="Relevant",
            dense_rank=1,
        ),
        RestrictedCandidate(
            chunk_id="new-1",
            source="source-4.md",
            title="Relevant",
            dense_rank=2,
        ),
    ]

    observed = []
    for _ in range(20):
        result = select_hierarchical_pool(
            _baseline(), restricted, ranked_sources,
        )
        observed.append(tuple(item.chunk_id for item in result.pool))

    assert len(set(observed)) == 1
    assert "new-1" in observed[0]
    assert observed[0].count("global-17") == 1
    assert observed[0][-3:] == ("global-17", "global-18", "global-19")


def test_score_report_enforces_direct_file_and_stage1_gates():
    from eval_hierarchical_candidate_pool import score_candidate_traces

    items = []
    traces = {}
    strict = {}
    targets = {}
    a0_rows = {}
    # Use the real contract sizes: 50 direct-supported + 2 file-only rows.
    for index in range(52):
        query_id = f"q{index:02d}"
        target_id = f"gold-{index}"
        source = f"target-{index}.md"
        items.append({"id": query_id, "expect_sources": [source]})
        baseline_ids = [f"base-{index}-{rank}" for rank in range(20)]
        baseline_sources = [source] + ["other.md"] * 19
        final_ids = list(baseline_ids)
        if index < 34:
            baseline_ids[0] = target_id
            final_ids[0] = target_id
        elif index < 37:
            final_ids[-1] = target_id
        traces[query_id] = {
            "baseline_pool_chunk_ids": baseline_ids,
            "baseline_pool_sources": baseline_sources,
            "selected_sources": [{"source": source}],
            "restricted_candidates": [],
            "ranked_hierarchical_candidates": [],
            "hierarchy_added_ids": [],
            "protected_global_ids": baseline_ids[:16],
            "backfilled_global_ids": [],
            "final_pool": [
                {"chunk_id": chunk_id, "source": baseline_sources[rank],
                 "origin": "test", "global_rank": rank + 1,
                 "hierarchy_rank": None}
                for rank, chunk_id in enumerate(final_ids)
            ],
        }
        a0_rows[query_id] = {"fusion_chunk_ids": baseline_ids}
        if index < 50:
            strict[query_id] = {target_id}
            targets[query_id] = {source}

    scoped = SimpleNamespace(strict_child_ids=strict)
    report = score_candidate_traces(
        items=items,
        traces=traces,
        scoped_qrels=scoped,
        direct_target_sources=targets,
        a0_rows=a0_rows,
        expected_baseline_direct_hits=34,
        expected_baseline_file_hits=52,
    )

    assert report["hierarchical"]["stage1_direct_source_recall"]["hits"] == 50
    assert report["hierarchical"]["direct_qrels_candidate_recall"]["hits"] == 37
    assert report["hierarchical"]["direct_qrels_candidate_recall"]["rescued_query_ids"] == [
        "q34", "q35", "q36",
    ]
    assert report["go_no_go"]["decision"] == "GO"


def test_candidate_generator_contract_has_no_qrels_parameter():
    import inspect
    from eval_hierarchical_candidate_pool import build_candidate_traces

    parameters = set(inspect.signature(build_candidate_traces).parameters)
    assert parameters == {"items", "collection"}
    assert not {"qrels", "targets", "expected_sources"} & parameters


def test_final_v1_rerank_comparison_reports_stable_wins_and_losses():
    from eval_hierarchical_v1_rerank import _comparison, _percentile

    baseline = {"a": True, "b": False, "c": True}
    candidate = {"a": True, "b": True, "c": False}
    assert _comparison(candidate, baseline) == {
        "baseline_hits": 2,
        "candidate_hits": 2,
        "n": 3,
        "delta_hits": 0,
        "wins": ["b"],
        "losses": ["c"],
    }
    assert _percentile([3.0, 1.0, 2.0], 0.50) == 2.0
