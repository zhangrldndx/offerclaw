# -*- coding: utf-8 -*-
"""Pre-registered, minimal retrieval arms for the colloquial development set."""

from __future__ import annotations

from rag_retrieval_trace import RetrievalProfile
from rag_tools import CHUNKER_VERSION


ARMS = {
    "baseline": {"rrf_k": 60},
    "rrf20": {"rrf_k": 20},
    "rrf40": {"rrf_k": 40},
    "rrf60": {"rrf_k": 60},
    "dense12": {"rrf_k": 60, "dense_rrf_weight": 1.2},
    "bm2512": {"rrf_k": 60, "bm25_rrf_weight": 1.2},
    "consensus": {"rrf_k": 60, "consensus_guard_top_k": 5,
                  "consensus_guard_margin": 0.03},
    "compact32": {"rrf_k": 60, "reranker_prefix_mode": "compact32"},
    # Union->RRF preservation arms (2026-08-27).  On Dev-New three golds sit in
    # exactly one channel at depth 12-15 and land at fused rank 23/25/27, so
    # RRF@20 loses them even though the union holds them.  Dev80 loses none,
    # which is why the older channel-exclusive strategy looked like a no-op:
    # it was measured on a set where union preservation was already 100%.
    # pool_size is the single depth knob (dense/BM25/fusion/reranker), so a
    # wider pool buys recall at a proportional reranking cost — measure, do not
    # assume.
    "compact32_pool24": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                         "pool_size": 24},
    "compact32_pool28": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                         "pool_size": 28},
    # Ten Dev-New golds sit outside both channels at depth 20, so no fusion or
    # pool policy at 20-28 can reach them.  These arms exist to measure whether
    # raw channel depth reaches them at all before anything expensive is spent
    # reranking or judging a pool that wide.
    "compact32_pool44": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                         "pool_size": 44},
    "compact32_pool60": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                         "pool_size": 60},
    # Query-side expansion arms.  HyDE and rewrite were both falsified in Round 9
    # on end-to-end R@1 -- but through a reranker that could not use whatever they
    # recalled, which is exactly why pool28 also looked like a no-op until the
    # answerability judge arrived.  The 2026-08-28 probe re-opened them on recall
    # alone: of the ten golds no channel reaches, HyDE reaches 8 and rewrite 5.
    "compact32_pool28_hyde": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                              "pool_size": 28, "enable_hyde": True},
    "compact32_pool28_rewrite": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                 "pool_size": 28, "enable_query_rewrite": True},
    # HyDE alone recovered 6 of the 10; the probe put rewrite on 5, only partly
    # overlapping, so the combination is measured rather than assumed additive.
    "compact32_pool28_hyde_rewrite": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                      "pool_size": 28, "enable_hyde": True,
                                      "enable_query_rewrite": True},
    # HyDE as a second dense channel instead of a replacement query.  Final v2
    # measured the two texts as complementary (original reaches 9 of the 21
    # out-of-pool golds, HyDE 15, either-of-them 17) and the replacement form as
    # the source of one genuine false accept, because it also replaces the
    # distances the evidence gate is calibrated on.
    "compact32_pool28_hydechan": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                  "pool_size": 28, "enable_hyde_channel": True},
    # Dense channel + lexical channel over the same HyDE text (one LLM call).
    # The gap probe puts the lexical side's exclusive reach at +3 (Final v2)
    # and +1 (dev80): golds whose wording only BM25-over-HyDE matches.
    "compact32_pool28_hydechan_bm25": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                       "pool_size": 28, "enable_hyde_channel": True,
                                       "enable_hyde_bm25_channel": True},
    # Stage 2 Q1: same-call canonical terms as a third query representation.
    "compact32_pool28_hydechan_terms": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                        "pool_size": 28, "enable_hyde_channel": True,
                                        "enable_hyde_bm25_channel": True,
                                        "enable_canonical_terms": True},
    # Two dense queries need room: at pool 28 the merged ranking has to drop
    # roughly half of what each query found.  The earlier pool44 result (a wider
    # pool bought one unreachable gold and cost R@3/R@5) was measured with a
    # *single* query, where the extra slots could only hold noise.
    "compact32_pool40_hydechan": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                  "pool_size": 40, "enable_hyde_channel": True},
    # Keeps the reranker pool at 20 but reserves slots for candidates that a
    # single channel ranks highly and fusion drops, so recall is bought with
    # membership instead of depth.
    "compact32_exclusive8": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                             "candidate_pool_strategy": "retain_channel_exclusives",
                             "exclusive_per_channel": 8},
    # Depth and membership together.  ``exclusive_per_channel`` has to stay
    # below half the pool: at 2x12 of a 20 pool the reservations evict every
    # dual-channel candidate and RRF recall collapsed 67/80 -> 23/80 on
    # Dev-New (2026-08-27).  RetrievalProfile now rejects that shape outright.
    "compact32_pool28_exclusive8": {"rrf_k": 60, "reranker_prefix_mode": "compact32",
                                    "pool_size": 28,
                                    "candidate_pool_strategy": "retain_channel_exclusives",
                                    "exclusive_per_channel": 8},
}


def colloquial_profile(
    arm: str,
    *,
    reranker_model: str | None = None,
) -> RetrievalProfile:
    """Return one frozen colloquial arm.

    ``reranker_model`` is an offline-evaluation escape hatch for a local
    development checkpoint.  The arm still owns every other retrieval knob,
    so comparing a fine-tuned model cannot silently change the candidate pool
    or the compact-prefix experiment at the same time.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown colloquial arm: {arm}")
    values = dict(ARMS[arm])
    profile = RetrievalProfile(
        name=f"colloquial-{arm}",
        pool_size=int(values.pop("pool_size", 20)),
        reranker_model=reranker_model or "BAAI/bge-reranker-base",
        reranker_use_breadcrumb=False,
        reranker_prefix_mode=values.pop("reranker_prefix_mode", "none"),
        rrf_k=int(values.pop("rrf_k", 60)),
        dense_rrf_weight=float(values.pop("dense_rrf_weight", 1.0)),
        bm25_rrf_weight=float(values.pop("bm25_rrf_weight", 1.0)),
        consensus_guard_top_k=int(values.pop("consensus_guard_top_k", 0)),
        consensus_guard_margin=float(values.pop("consensus_guard_margin", 0.03)),
        candidate_pool_strategy=str(
            values.pop("candidate_pool_strategy", "baseline_rrf20")
        ),
        exclusive_per_channel=int(values.pop("exclusive_per_channel", 4)),
        chunker_version=CHUNKER_VERSION,
        # Query-side text expansion is arm-owned rather than pinned off.  Every
        # pre-existing arm omits these keys and so is byte-identical, but the
        # 2026-08-28 probe showed HyDE reaches 8 of the 10 Dev-New golds that no
        # channel retrieves, which cannot be tested while the flag is welded shut.
        enable_hyde=bool(values.pop("enable_hyde", False)),
        enable_query_rewrite=bool(values.pop("enable_query_rewrite", False)),
        enable_doc2query=bool(values.pop("enable_doc2query", False)),
        enable_hyde_channel=bool(values.pop("enable_hyde_channel", False)),
        enable_hyde_bm25_channel=bool(values.pop("enable_hyde_bm25_channel", False)),
        enable_canonical_terms=bool(values.pop("enable_canonical_terms", False)),
        enable_quota=False,
        enable_alias=False,
        enable_rerank_bridge=False,
    )
    # Every supported knob is popped above, so anything left over is a typo in
    # ARMS.  Left unchecked it would run silently as the baseline and be
    # written up as "the arm made no difference".
    if values:
        raise ValueError(f"arm {arm!r} carries unknown knobs: {sorted(values)}")
    return profile


__all__ = ["ARMS", "colloquial_profile"]
