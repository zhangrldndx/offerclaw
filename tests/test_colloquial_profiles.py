import pytest

from rag_colloquial_profiles import ARMS, colloquial_profile


def test_preregistered_arms_are_small_and_baseline_is_unchanged():
    assert set(ARMS) == {
        "baseline", "rrf20", "rrf40", "rrf60", "dense12", "bm2512",
        "consensus", "compact32",
        # candidate-membership arms (Union->RRF preservation, 2026-08-27)
        "compact32_pool24", "compact32_pool28",
        # channel-depth arms: ten Dev-New golds sit outside both channels at
        # depth 20, so these measure whether raw depth reaches them (2026-08-28)
        "compact32_pool44", "compact32_pool60",
        # query-side expansion arms (recall probe re-opened them, 2026-08-28)
        "compact32_pool28_hyde", "compact32_pool28_rewrite",
        "compact32_pool28_hyde_rewrite",
        "compact32_pool28_hydechan", "compact32_pool40_hydechan",
        "compact32_pool28_hydechan_bm25",
        "compact32_pool28_hydechan_terms",
        "compact32_exclusive8", "compact32_pool28_exclusive8",
    }
    baseline = colloquial_profile("baseline")
    assert baseline.pool_size == 20
    assert baseline.rrf_k == 60
    assert baseline.dense_rrf_weight == 1.0
    assert baseline.bm25_rrf_weight == 1.0
    assert baseline.reranker_prefix_mode == "none"
    assert baseline.consensus_guard_top_k == 0
    assert baseline.candidate_pool_strategy == "baseline_rrf20"
    assert not baseline.enable_query_rewrite
    assert not baseline.enable_doc2query
    assert not baseline.enable_alias


def test_candidate_arms_change_only_the_preregistered_dimension():
    assert colloquial_profile("rrf20").rrf_k == 20
    assert colloquial_profile("dense12").dense_rrf_weight == 1.2
    assert colloquial_profile("bm2512").bm25_rrf_weight == 1.2
    assert colloquial_profile("consensus").consensus_guard_top_k == 5
    assert colloquial_profile("compact32").reranker_prefix_mode == "compact32"


@pytest.mark.parametrize("arm,pool,strategy,exclusive", [
    ("compact32", 20, "baseline_rrf20", 4),
    ("compact32_pool24", 24, "baseline_rrf20", 4),
    ("compact32_pool28", 28, "baseline_rrf20", 4),
    ("compact32_exclusive8", 20, "retain_channel_exclusives", 8),
    ("compact32_pool28_exclusive8", 28, "retain_channel_exclusives", 8),
])
def test_membership_arms_differ_from_compact32_only_in_candidate_selection(
    arm, pool, strategy, exclusive,
):
    """The membership sweep must not smuggle in a second change.

    Every one of these arms is compared against ``compact32``, so anything
    other than pool depth or pool strategy differing between them would make
    the comparison uninterpretable.
    """
    profile = colloquial_profile(arm)
    reference = colloquial_profile("compact32")
    assert (profile.pool_size, profile.candidate_pool_strategy,
            profile.exclusive_per_channel) == (pool, strategy, exclusive)
    assert profile.reranker_prefix_mode == reference.reranker_prefix_mode
    assert profile.rrf_k == reference.rrf_k
    assert profile.dense_rrf_weight == reference.dense_rrf_weight
    assert profile.bm25_rrf_weight == reference.bm25_rrf_weight
    assert profile.consensus_guard_top_k == reference.consensus_guard_top_k
    assert profile.reranker_model == reference.reranker_model


def test_unknown_arm_knob_is_rejected_rather_than_silently_ignored():
    ARMS["_typo_probe"] = {"rrf_k": 60, "pool_sze": 28}
    try:
        with pytest.raises(ValueError, match="pool_sze"):
            colloquial_profile("_typo_probe")
    finally:
        ARMS.pop("_typo_probe")


def test_channel_reservations_may_not_consume_the_whole_pool():
    """2 x exclusive_per_channel >= pool_size is a silently meaningless config.

    A single-channel candidate carries about half the RRF score of a
    dual-channel one at the same rank, so once the reservations can fill the
    pool the strategy keeps only candidates that exactly one channel found and
    evicts everything both channels agreed on.  Measured on Dev-New at
    pool 20 x exclusive 12: RRF candidate recall 67/80 -> 23/80, R@1 47 -> 11.
    The pool builder truncates rather than failing, so the profile has to
    reject it.
    """
    from rag_retrieval_trace import RetrievalProfile

    with pytest.raises(ValueError, match="dual-channel"):
        RetrievalProfile(candidate_pool_strategy="retain_channel_exclusives",
                         pool_size=20, exclusive_per_channel=10)
    # ... and the same numbers are fine as soon as the pool has room
    RetrievalProfile(candidate_pool_strategy="retain_channel_exclusives",
                     pool_size=28, exclusive_per_channel=10)
    # the guard is scoped to the strategy that reads the knob
    RetrievalProfile(candidate_pool_strategy="baseline_rrf20",
                     pool_size=20, exclusive_per_channel=99)


def test_a_profile_that_enables_query_expansion_does_not_also_need_an_env_var(monkeypatch):
    """A knob the arm owns must not be vetoed by a second, hidden switch.

    ``hyde_expand``/``rewrite_query`` used to read ``RAG_HYDE``/
    ``RAG_QUERY_REWRITE`` themselves, so an arm that explicitly enabled them ran
    silently on the raw question: the funnel came out byte-identical and latency
    went *down*, which reads as "the technique does nothing" rather than "the
    technique never ran".  The same shape already cost a measurement round on
    ``RAG_RECALL_N``.
    """
    import rag_hyde

    monkeypatch.delenv("RAG_HYDE", raising=False)
    monkeypatch.delenv("RAG_QUERY_REWRITE", raising=False)
    monkeypatch.setattr(rag_hyde, "_chat_for_test", None, raising=False)

    # env off + caller says on -> the caller wins (the call is attempted)
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        return "书面检索式"

    import sys
    import types

    stub = types.ModuleType("rag_gate")
    stub._chat = fake_chat
    saved = sys.modules.get("rag_gate")
    sys.modules["rag_gate"] = stub
    try:
        assert rag_hyde.hyde_expand("口语问题", enabled=True) == "口语问题\n书面检索式"
        assert rag_hyde.rewrite_query("口语问题", enabled=True) == "口语问题\n书面检索式"
        assert len(calls) == 2
        # ... and the default still defers to env, so nothing turns on by itself
        assert rag_hyde.hyde_expand("口语问题") == "口语问题"
        assert rag_hyde.rewrite_query("口语问题") == "口语问题"
        assert len(calls) == 2
    finally:
        if saved is not None:
            sys.modules["rag_gate"] = saved
        else:
            sys.modules.pop("rag_gate", None)
