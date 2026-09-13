from __future__ import annotations

import pytest

from rag_candidate_pool import (
    ChannelCandidate,
    protect_multichannel_consensus,
    rrf_union,
    select_candidate_pool,
    source_distribution,
)


def test_consensus_guard_only_promotes_close_dual_channel_candidate():
    docs = ["winner", "consensus", "other"]
    metas = [
        {"chunk_id": "w"}, {"chunk_id": "c"}, {"chunk_id": "o"},
    ]
    promoted = protect_multichannel_consensus(
        docs, metas, [0.3, 0.4, 0.5], [0.91, 0.89, 0.2],
        dense_chunk_ids=["c", "w"], bm25_chunk_ids=["c", "o"],
        top_k=2, margin=0.03,
    )
    assert promoted[0][0] == "consensus"
    assert promoted[-1]["applied"] is True

    unchanged = protect_multichannel_consensus(
        docs, metas, [0.3, 0.4, 0.5], [0.99, 0.80, 0.2],
        dense_chunk_ids=["c", "w"], bm25_chunk_ids=["c", "o"],
        top_k=2, margin=0.03,
    )
    assert unchanged[0] == docs
    assert unchanged[-1]["reason"] == "margin_too_large"


class _FakeCollection:
    def __init__(self, name, rows):
        self.name = name
        self.rows = list(rows)

    def count(self):
        return len(self.rows)

    def get(self, ids=None, include=None):
        allowed = set(ids) if ids is not None else None
        selected = [row for row in self.rows if allowed is None or row[0] in allowed]
        return {
            "ids": [row[0] for row in selected],
            "documents": [row[1] for row in selected],
            "metadatas": [row[2] for row in selected],
        }


class _FakeClient:
    def __init__(self, collections):
        self.collections = dict(collections)

    def get_collection(self, name):
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]


def _write_qrels(path, *, collection, targets):
    from rag_qrels import answer_span_hash

    payload = {
        "schema_version": "rag-qrels-overlay-v1",
        "reviewer_id": "test-reviewer",
        "base_set": "tests/rag_bench_paraphrase_set.json",
        "index": {"collection": collection.name, "count": collection.count()},
        "items": [{
            "query_id": "q1",
            "review_outcome": "accepted",
            "review_note": "reviewed",
            "relevant_targets": targets,
        }],
    }
    for target in payload["items"][0]["relevant_targets"]:
        target.setdefault("heading_path", ["Section"])
        target.setdefault("relevance", "direct")
        target.setdefault("review_note", "direct evidence")
        target.setdefault("answer_span_hash", answer_span_hash(target["evidence_excerpt"]))
    path.write_text(__import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _candidate(chunk_id, source, rank, *, score=None, distance=None):
    return ChannelCandidate(
        chunk_id=chunk_id,
        source=source,
        rank=rank,
        score=score,
        distance=distance,
    )


def test_baseline_rrf_reproduces_dense_first_stable_ties():
    dense = [
        _candidate("d1", "a.md", 1, distance=0.1),
        _candidate("shared", "b.md", 2, distance=0.2),
        _candidate("d3", "c.md", 3, distance=0.3),
    ]
    bm25 = [
        _candidate("shared", "b.md", 1, score=9.0),
        _candidate("b2", "d.md", 2, score=8.0),
        _candidate("b3", "e.md", 3, score=7.0),
    ]

    pool = select_candidate_pool(
        dense, bm25, strategy="baseline_rrf20", pool_size=5,
    )

    assert [item.chunk_id for item in pool] == ["shared", "d1", "b2", "d3", "b3"]
    assert pool[0].channels == ("dense", "bm25")
    assert pool[0].dense_rank == 2
    assert pool[0].bm25_rank == 1


def test_identity_is_chunk_id_not_duplicate_document_or_source():
    # The pure API cannot even receive document text. Two chunks from one
    # source remain distinct because their stable IDs are distinct.
    dense = [
        _candidate("chunk-a", "same.md", 1),
        _candidate("chunk-b", "same.md", 2),
    ]
    pool = select_candidate_pool(
        dense, [], strategy="baseline_rrf20", pool_size=20,
    )
    assert [item.chunk_id for item in pool] == ["chunk-a", "chunk-b"]


def test_duplicate_id_merges_channels_and_conflicting_lineage_fails():
    merged = rrf_union(
        [_candidate("same-id", "folder/a.md", 1)],
        [_candidate("same-id", "a.md", 2)],
    )
    assert len(merged) == 1
    assert merged[0].channels == ("dense", "bm25")

    with pytest.raises(ValueError, match="conflicting sources"):
        rrf_union(
            [_candidate("same-id", "a.md", 1)],
            [_candidate("same-id", "b.md", 1)],
        )


def test_source_cap_backfills_and_never_exceeds_four_per_source():
    dense = [
        _candidate(f"a-{rank}", "a.md", rank)
        for rank in range(1, 8)
    ] + [
        _candidate(f"b-{rank}", "b.md", rank + 7)
        for rank in range(1, 8)
    ]
    bm25 = [
        _candidate(f"c-{rank}", "c.md", rank)
        for rank in range(1, 8)
    ]
    pool = select_candidate_pool(
        dense,
        bm25,
        strategy="source_cap4",
        pool_size=12,
        max_per_source=4,
    )
    assert len(pool) == 12
    assert source_distribution(pool) == {"a.md": 4, "b.md": 4, "c.md": 4}


def test_exclusive_strategy_reserves_both_channels_then_uses_rrf_order():
    dense = [
        _candidate("shared-1", "shared.md", 1),
        _candidate("shared-2", "shared.md", 2),
        _candidate("dense-only-1", "dense.md", 3),
        _candidate("dense-only-2", "dense.md", 4),
    ]
    bm25 = [
        _candidate("shared-1", "shared.md", 1),
        _candidate("shared-2", "shared.md", 2),
        _candidate("bm25-only-1", "bm25.md", 3),
        _candidate("bm25-only-2", "bm25.md", 4),
    ]
    pool = select_candidate_pool(
        dense,
        bm25,
        strategy="retain_channel_exclusives",
        pool_size=4,
        exclusive_per_channel=1,
    )
    ids = [item.chunk_id for item in pool]
    assert ids == ["shared-1", "shared-2", "dense-only-1", "bm25-only-1"]
    assert {item.channel_class for item in pool} == {
        "shared", "dense_only", "bm25_only",
    }


def test_exclusive_reservation_truncation_is_deterministic():
    dense = [_candidate(f"d-{rank}", "dense.md", rank) for rank in range(1, 6)]
    bm25 = [_candidate(f"b-{rank}", "bm25.md", rank) for rank in range(1, 6)]
    observed = []
    for _ in range(20):
        pool = select_candidate_pool(
            dense,
            bm25,
            strategy="retain_channel_exclusives",
            pool_size=3,
            exclusive_per_channel=4,
        )
        observed.append(tuple(item.chunk_id for item in pool))
    assert len(set(observed)) == 1
    assert observed[0] == ("d-1", "b-1", "d-2")


def test_invalid_candidate_and_unknown_strategy_fail_closed():
    with pytest.raises(ValueError, match="chunk_id"):
        ChannelCandidate(chunk_id="", source="a.md", rank=1)
    with pytest.raises(ValueError, match="one-based"):
        ChannelCandidate(chunk_id="a", source="a.md", rank=0)
    with pytest.raises(ValueError, match="unknown"):
        select_candidate_pool(
            [_candidate("a", "a.md", 1)],
            [],
            strategy="not-real",  # type: ignore[arg-type]
        )


def test_c0_qrels_without_evidence_scope_remain_strict_child_compatible(tmp_path):
    from eval_candidate_pools import _load_qrels, _load_scoped_qrels

    child = _FakeCollection("c0", [
        ("c1", "这里有直接证据", {"source": "knowledge.md"}),
    ])
    path = _write_qrels(tmp_path / "c0.json", collection=child, targets=[{
        "source": "knowledge.md",
        "chunk_id": "c1",
        "evidence_excerpt": "直接证据",
    }])
    scoped, _payload = _load_scoped_qrels(path, ["q1"], child)
    combined, _payload = _load_qrels(path, ["q1"], child)

    assert scoped.strict_child_ids == {"q1": {"c1"}}
    assert scoped.parent_origin_ids == {"q1": set()}
    assert scoped.parent_trigger_child_ids == {"q1": set()}
    assert combined == {"q1": {"c1"}}
    assert scoped.production_collection == ""


def test_stage_d_parent_scope_loads_and_validates_parent_collection(tmp_path):
    import hashlib
    from eval_candidate_pools import _load_scoped_qrels

    parent_text = "完整证据跨越两个孩子，因此必须展开父块。"
    parent_hash = "sha256:" + hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
    child = _FakeCollection("experiment-c1", [
        ("child-a", "完整证据跨越", {
            "source": "knowledge.md",
            "origin_chunk_id": "parent-1",
            "parent_content_hash": parent_hash,
            "production_collection": "production-c0",
        }),
        ("child-b", "两个孩子，因此必须展开父块。", {
            "source": "knowledge.md",
            "origin_chunk_id": "parent-1",
            "parent_content_hash": parent_hash,
            "production_collection": "production-c0",
        }),
    ])
    parent = _FakeCollection("production-c0", [
        ("parent-1", parent_text, {"source": "knowledge.md"}),
    ])
    path = _write_qrels(tmp_path / "mapped.json", collection=child, targets=[{
        "source": "knowledge.md",
        "chunk_id": "parent-1",
        "origin_chunk_id": "parent-1",
        "parent_content_hash": parent_hash,
        "evidence_scope": "parent_expand_required",
        "evidence_excerpt": "完整证据跨越两个孩子",
    }])

    scoped, _payload = _load_scoped_qrels(
        path,
        ["q1"],
        child,
        client=_FakeClient({"production-c0": parent}),
    )

    assert scoped.strict_child_ids == {"q1": set()}
    assert scoped.parent_origin_ids == {"q1": {"parent-1"}}
    assert scoped.parent_trigger_child_ids == {"q1": {"child-a", "child-b"}}
    assert scoped.production_collection == "production-c0"


def test_stage_d_parent_scope_fails_closed_without_parent_loader(tmp_path):
    import hashlib
    from eval_candidate_pools import _load_scoped_qrels

    parent_text = "父块完整证据"
    parent_hash = "sha256:" + hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
    child = _FakeCollection("experiment-c1", [
        ("child-a", "部分", {
            "source": "knowledge.md",
            "origin_chunk_id": "parent-1",
            "parent_content_hash": parent_hash,
            "production_collection": "production-c0",
        }),
    ])
    path = _write_qrels(tmp_path / "mapped.json", collection=child, targets=[{
        "source": "knowledge.md",
        "chunk_id": "parent-1",
        "origin_chunk_id": "parent-1",
        "parent_content_hash": parent_hash,
        "evidence_scope": "parent_expand_required",
        "evidence_excerpt": parent_text,
    }])

    with pytest.raises(ValueError, match="requires a Chroma client"):
        _load_scoped_qrels(path, ["q1"], child)


def test_scoped_metrics_keep_parent_expandable_separate_from_strict_child():
    from eval_candidate_pools import ScopedDirectQrels, _strategy_report

    items = [
        {"id": "strict", "expect_sources": ["strict.md"]},
        {"id": "parent", "expect_sources": ["parent.md"]},
        {"id": "unsupported", "expect_sources": ["legacy-only.md"]},
    ]
    channels = {
        "strict": ([_candidate("strict-child", "strict.md", 1)], []),
        "parent": ([_candidate("trigger-child", "parent.md", 1)], []),
        "unsupported": ([_candidate("legacy-child", "legacy-only.md", 1)], []),
    }
    scoped = ScopedDirectQrels(
        strict_child_ids={"strict": {"strict-child"}, "parent": set()},
        parent_origin_ids={"strict": set(), "parent": {"parent-origin"}},
        parent_trigger_child_ids={"strict": set(), "parent": {"trigger-child"}},
        production_collection="production-c0",
    )
    baseline = {
        "strict_child": {"strict": False, "parent": False},
        "parent_expandable": {"strict": False, "parent": False},
        "combined": {"strict": False, "parent": False},
    }
    report = _strategy_report(
        "source_cap4", items, channels, scoped, baseline,
    )

    assert report["qrels_scopes"]["strict_child"]["hits"] == 1
    assert report["qrels_scopes"]["strict_child"]["n"] == 1
    assert report["qrels_scopes"]["parent_expandable"]["hits"] == 1
    assert report["qrels_scopes"]["parent_expandable"]["n"] == 1
    assert report["qrels_scopes"]["parent_expandable"]["parent_expansion_applied"] is False
    assert report["qrels_scopes"]["combined"]["hits"] == 2
    assert report["qrels_scopes"]["combined"]["n"] == 2
    assert report["file_level_candidate_recall"]["hits"] == 3
    assert report["file_level_candidate_recall"]["n"] == 3
    assert len(report["file_level_candidate_recall"]["rows"]) == 3
    assert report["file_level_candidate_recall"]["rescued_query_ids"] == [
        "parent", "strict", "unsupported",
    ]
    assert report["file_level_candidate_recall"]["lost_query_ids"] == []
    parent_row = next(row for row in report["rows"] if row["id"] == "parent")
    assert parent_row["direct_candidate_hit"] is False
    assert parent_row["parent_expandable_hit"] is True
    assert parent_row["parent_expansion_applied"] is False


def test_baseline_misses_scope_does_not_expand_to_unsupported_file_rows():
    from eval_candidate_pools import ScopedDirectQrels, _select_scope_items

    items = [
        {"id": "hit"},
        {"id": "miss"},
        {"id": "unsupported"},
    ]
    scoped = ScopedDirectQrels(
        strict_child_ids={"hit": {"h"}, "miss": {"m"}, "unsupported": set()},
        parent_origin_ids={"hit": set(), "miss": set(), "unsupported": set()},
        parent_trigger_child_ids={"hit": set(), "miss": set(), "unsupported": set()},
    )
    a0_rows = {
        "hit": {"direct_candidate_rank": 1},
        "miss": {"direct_candidate_rank": 0},
        "unsupported": {"direct_candidate_rank": 0},
    }

    qrels_items, file_items = _select_scope_items(
        items, scoped, a0_rows, scope="baseline-misses",
    )
    assert [item["id"] for item in qrels_items] == ["miss"]
    assert [item["id"] for item in file_items] == ["miss"]

    all_qrels, all_file = _select_scope_items(items, scoped, a0_rows, scope="all")
    assert [item["id"] for item in all_qrels] == ["hit", "miss"]
    assert [item["id"] for item in all_file] == ["hit", "miss", "unsupported"]
