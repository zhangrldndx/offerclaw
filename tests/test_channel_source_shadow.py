from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from rag_candidate_pool import PoolCandidate
from rag_channel_source_shadow import (
    ExactSourceCandidate,
    rank_exact_candidates,
    select_channel_source_pool,
    source_shortlist_from_global,
)


def _baseline() -> list[PoolCandidate]:
    return [
        PoolCandidate(
            chunk_id=f"global-{rank:02d}",
            source=f"source-{(rank - 1) // 4}.md",
            dense_rank=rank,
            rrf_score=1 / (60 + rank),
            first_seen=rank,
        )
        for rank in range(1, 21)
    ]


def test_source_shortlist_uses_every_protected_source_in_first_seen_order():
    assert source_shortlist_from_global(_baseline()) == [
        "source-0.md", "source-1.md", "source-2.md", "source-3.md",
    ]


def test_exact_rank_has_stable_chunk_id_tie_break_and_dedupes_by_id():
    ranked = rank_exact_candidates(
        [
            ExactSourceCandidate("z", "source-0.md", 0.5),
            ExactSourceCandidate("a", "source-0.md", 0.5),
            ExactSourceCandidate("z", "source-0.md", 0.6),
        ],
        allowed_sources=["source-0.md"],
    )
    assert [(row.chunk_id, row.cosine_score) for row in ranked] == [
        ("z", 0.6), ("a", 0.5),
    ]


def test_shadow_skips_entire_global20_then_adds_top4_unseen():
    baseline = _baseline()
    exact = [
        ExactSourceCandidate("global-17", "source-4.md", 1.0),
        ExactSourceCandidate("global-03", "source-0.md", 0.99),
        *[
            ExactSourceCandidate(f"new-{rank}", "source-0.md", 0.9 - rank / 100)
            for rank in range(1, 6)
        ],
    ]
    # source-4 is not represented in protected global16, so the lineage guard
    # correctly rejects it before membership can be selected.
    with pytest.raises(ValueError, match="outside shortlist"):
        select_channel_source_pool(baseline, exact)

    result = select_channel_source_pool(baseline, exact[1:])
    assert len(result.pool) == 20
    assert [row.chunk_id for row in result.pool[:16]] == [
        f"global-{rank:02d}" for rank in range(1, 17)
    ]
    assert result.exact_added_ids == ("new-1", "new-2", "new-3", "new-4")
    assert not result.backfilled_global_ids
    assert "global-03" not in result.exact_added_ids


def test_shadow_backfills_global_tail_if_fewer_than_four_unseen():
    result = select_channel_source_pool(
        _baseline(),
        [ExactSourceCandidate("new", "source-0.md", 0.9)],
    )
    assert len(result.pool) == 20
    assert result.exact_added_ids == ("new",)
    assert result.backfilled_global_ids == (
        "global-17", "global-18", "global-19",
    )


def test_conflicting_stable_id_lineage_fails_closed():
    with pytest.raises(ValueError, match="conflicting source lineage"):
        rank_exact_candidates(
            [
                ExactSourceCandidate("same", "source-0.md", 0.9),
                ExactSourceCandidate("same", "source-1.md", 0.8),
            ],
            allowed_sources=["source-0.md", "source-1.md"],
        )


def _trace(query_id: str, *, baseline_source: str, candidate_source: str):
    return {
        "baseline_pool_chunk_ids": [f"base-{query_id}"],
        "baseline_pool_sources": [baseline_source],
        "source_shortlist": [baseline_source],
        "source_shortlist_chunk_count": 2,
        "exact_added_ids": [f"new-{query_id}"],
        "final_pool": [{
            "chunk_id": f"new-{query_id}",
            "source": candidate_source,
            "origin": "source_exact_unseen",
        }],
    }


def test_file_scoring_and_shadow_signal_are_independent_of_direct_qrels():
    from eval_channel_source_shadow import score_file_level_sets, shadow_signal

    labeled = {
        "zh_final90": [
            {"id": "z1", "expect_sources": ["target.md"]},
            {"id": "z2", "expect_sources": ["target.md"]},
        ],
        "rag_bench100": [
            {"id": "b1", "expect_sources": ["target.md"]},
        ],
    }
    traces = {
        "zh_final90::z1": _trace(
            "z1", baseline_source="other.md", candidate_source="target.md",
        ),
        "zh_final90::z2": _trace(
            "z2", baseline_source="target.md", candidate_source="target.md",
        ),
        "rag_bench100::b1": _trace(
            "b1", baseline_source="target.md", candidate_source="target.md",
        ),
    }
    reports = score_file_level_sets(labeled, traces)
    signal = shadow_signal(reports)
    assert reports["zh_final90"]["candidate"]["rescued_query_ids"] == ["z1"]
    assert signal["decision"] == "WARRANTS_NEW_BLIND_EVAL"
    assert signal["promotion"].startswith("BLOCKED")


def test_heldout_direct_is_dev_only_and_requires_all_a0_orders():
    from eval_channel_source_shadow import score_heldout_dev_direct

    items = []
    traces = {}
    strict = {}
    a0_rows = {}
    for index in range(52):
        query_id = f"q{index:02d}"
        items.append({"id": query_id})
        target = f"target-{index}"
        baseline = [f"base-{index}"]
        candidate = [target] if index == 34 else list(baseline)
        if index < 34:
            baseline = [target]
            candidate = [target]
        traces[f"heldout52::{query_id}"] = {
            "baseline_pool_chunk_ids": baseline,
            "final_pool": [{"chunk_id": value, "source": "x.md"} for value in candidate],
        }
        a0_rows[query_id] = {"fusion_chunk_ids": baseline}
        if index < 50:
            strict[query_id] = {target}
    report = score_heldout_dev_direct(
        heldout_items=items,
        traces=traces,
        scoped_qrels=SimpleNamespace(strict_child_ids=strict),
        a0_rows=a0_rows,
    )
    assert report["claim_scope"].startswith("DEV_ONLY")
    assert report["baseline"]["hits"] == 34
    assert report["candidate"]["hits"] == 35
    assert report["a0_stable_id_order"]["all_exact"] is True


def test_candidate_generator_contract_has_no_labels_or_qrels_parameters():
    from eval_channel_source_shadow import build_candidate_traces

    parameters = set(inspect.signature(build_candidate_traces).parameters)
    assert parameters == {"items", "collection"}
    assert not {"qrels", "labels", "expected_sources", "a0_rows"} & parameters


def test_query_embedding_batches_are_stable_per_frozen_set_namespace():
    from eval_channel_source_shadow import embed_query_inputs_by_namespace

    calls = []

    def fake_embedder(questions):
        calls.append(list(questions))
        return [[float(len(question))] for question in questions]

    vectors = embed_query_inputs_by_namespace(
        [
            {"id": "heldout52::a", "q": "aa"},
            {"id": "zh_final90::b", "q": "bbb"},
            {"id": "heldout52::c", "q": "c"},
        ],
        fake_embedder,
    )

    assert calls == [["aa", "c"], ["bbb"]]
    assert vectors == [[2.0], [3.0], [1.0]]
