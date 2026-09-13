# -*- coding: utf-8 -*-
"""HyDE as a second dense channel rather than a replacement query.

The replacement form has two costs the channel form avoids, and both were
measured on Final v2 rather than argued: it discards whatever only the original
question reaches (original 9 of 21 out-of-pool golds, HyDE 15, either 17), and it
swaps every distance in the pool for a "hypothetical answer -> document"
distance while the evidence gate still compares against thresholds calibrated on
"question -> document" distances.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_gate import _merge_dense_channels  # noqa: E402


class _Collection:
    """Returns stored vectors for ids the primary query never saw."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def get(self, ids, include):
        self.calls.append(list(ids))
        return {"ids": list(ids), "embeddings": [self.vectors[i] for i in ids]}


def test_rank_fusion_promotes_what_both_queries_found():
    primary = (["a", "b"], [{}, {}], [0.30, 0.60], ["ida", "idb"])
    secondary = (["c", "b"], [{}, {}], [0.10, 0.20], ["idc", "idb"])
    docs, _m, _d, ids = _merge_dense_channels(
        primary, secondary, collection=_Collection({"idc": [0.0, 1.0]}),
        query_embedding=[0.0, 0.0], rrf_k=60, limit=3)
    # idb is rank 2 in one list and rank 2 in the other -> highest fused score
    assert ids[0] == "idb"
    assert set(ids) == {"ida", "idb", "idc"}
    assert docs[0] == "b"


def test_a_secondary_only_chunk_keeps_the_primary_querys_distance():
    """Borrowing the HyDE distance would hand the gate a number from a different
    distribution than the one its threshold was calibrated on."""
    collection = _Collection({"idc": [3.0, 4.0]})
    docs, _m, dists, ids = _merge_dense_channels(
        (["a"], [{}], [0.30], ["ida"]),
        (["c"], [{}], [0.01], ["idc"]),          # HyDE thinks it is very close
        collection=collection, query_embedding=[0.0, 0.0], rrf_k=60, limit=2)
    position = ids.index("idc")
    assert dists[position] == 25.0                # squared L2 to [0,0], not 0.01
    assert collection.calls == [["idc"]]          # only the missing ids are fetched


def test_a_chunk_both_queries_found_keeps_its_primary_distance():
    collection = _Collection({})
    _docs, _m, dists, ids = _merge_dense_channels(
        (["a"], [{}], [0.30], ["ida"]),
        (["a"], [{}], [0.99], ["ida"]),
        collection=collection, query_embedding=[0.0, 0.0], rrf_k=60, limit=2)
    assert dists[ids.index("ida")] == 0.30
    assert collection.calls == []                 # nothing to look up


def test_an_unreadable_vector_cannot_open_the_gate():
    """A lookup failure must not become a small distance; the gate reads the
    minimum, so a wrong zero there is an accept."""
    class _Broken:
        def get(self, ids, include):
            raise RuntimeError("index unavailable")

    _docs, _m, dists, ids = _merge_dense_channels(
        (["a"], [{}], [0.30], ["ida"]),
        (["c"], [{}], [0.01], ["idc"]),
        collection=_Broken(), query_embedding=[0.0, 0.0], rrf_k=60, limit=2)
    assert dists[ids.index("idc")] == float("inf")
    assert min(dists) == 0.30


def test_the_merge_respects_the_pool_limit():
    primary = ([f"d{i}" for i in range(10)], [{}] * 10, [0.1 * i for i in range(10)],
               [f"id{i}" for i in range(10)])
    secondary = ([f"e{i}" for i in range(10)], [{}] * 10, [0.05] * 10,
                 [f"jd{i}" for i in range(10)])
    _docs, _m, _d, ids = _merge_dense_channels(
        primary, secondary, collection=_Collection({f"jd{i}": [0.0] for i in range(10)}),
        query_embedding=[0.0], rrf_k=60, limit=6)
    assert len(ids) == 6


def test_the_channel_arm_differs_from_its_baseline_in_exactly_one_field():
    from rag_colloquial_profiles import colloquial_profile

    base = colloquial_profile("compact32_pool28").to_dict()
    channel = colloquial_profile("compact32_pool28_hydechan").to_dict()
    differing = {k for k in base if base[k] != channel[k]} - {"name"}
    assert differing == {"enable_hyde_channel"}


def test_the_channel_is_off_by_default_everywhere():
    from rag_colloquial_profiles import colloquial_profile
    from rag_retrieval_trace import RetrievalProfile

    assert RetrievalProfile().enable_hyde_channel is None
    assert colloquial_profile("compact32_pool28").enable_hyde_channel is False


def test_bm25_merge_keeps_primary_tuple_and_respects_limit():
    from rag_gate import _merge_bm25_channels

    primary = [("doc-a", {"source": "s1"}, 9.0), ("doc-b", {"source": "s2"}, 8.0)]
    secondary = [("doc-b", {"source": "s2"}, 3.0), ("doc-c", {"source": "s3"}, 2.5)]
    fused = _merge_bm25_channels(primary, secondary, rrf_k=60, limit=2)
    texts = [hit[0] for hit in fused]
    assert texts[0] == "doc-b"            # found by both queries -> promoted
    assert len(fused) == 2                 # reach, not a bigger pool
    shared = next(hit for hit in fused if hit[0] == "doc-b")
    assert shared[2] == 8.0                # the primary query's score survives


def test_bm25_channel_arm_differs_in_exactly_one_field():
    from rag_colloquial_profiles import colloquial_profile

    base = colloquial_profile("compact32_pool28_hydechan").to_dict()
    lex = colloquial_profile("compact32_pool28_hydechan_bm25").to_dict()
    assert {k for k in base if base[k] != lex[k]} - {"name"} == {"enable_hyde_bm25_channel"}


def test_both_channels_share_one_hyde_expansion():
    """Two channels, one LLM call: the lexical channel reuses the dense
    channel's text rather than paying for (and possibly getting) a different
    hypothetical answer."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("rag_gate.py").read_text(encoding="utf-8")
    assert source.count("hyde_expand(question, enabled=True)") == 1
    assert "_hyde_channel_text and profile.enable_hyde_bm25_channel" in source
