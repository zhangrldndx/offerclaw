# -*- coding: utf-8 -*-
"""Production-consistent retrieval trace and profile controls."""

from __future__ import annotations


def _install_retrieval_stubs(monkeypatch, *, dense_source=None):
    import chromadb
    import rag_alias
    import rag_bm25
    import rag_doc2query
    import rag_gate
    import rag_hyde
    import rag_quota
    import rag_rerank
    import rag_route
    import rag_tools

    seen = {"queries": [], "rerank_pool": None}

    class FakeCollection:
        def count(self):
            return 7

        def get(self, **kwargs):
            if kwargs.get("include") == []:
                return {"ids": [f"dense_{i}" for i in range(7)]}
            return {
                "ids": [f"dense_{i}" for i in range(7)],
                "metadatas": [{"chunker_version": "test-v1"}] * 7,
            }

        def query(self, **kwargs):
            seen["queries"].append(kwargs)
            n = kwargs["n_results"]
            return {
                "ids": [[f"dense_{i}" for i in range(n)]],
                "documents": [[f"dense document {i}" for i in range(n)]],
                "metadatas": [[{
                    "source": dense_source or f"source_{i}.md", "source_type": "doc",
                    "owner_scope": "curated", "title": f"Section {i}",
                } for i in range(n)]],
                "distances": [[0.10 + i * 0.01 for i in range(n)]],
            }

    collection = FakeCollection()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_collection(self, *args, **kwargs):
            return collection

    monkeypatch.setattr(chromadb, "PersistentClient", FakeClient)
    fake_embeddings = lambda texts: [[0.0] * 4]
    monkeypatch.setattr(rag_tools, "get_embeddings_batch", fake_embeddings)
    monkeypatch.setattr(rag_gate, "get_embeddings_batch", fake_embeddings)
    monkeypatch.setattr(rag_hyde, "hyde_expand", lambda query, **_: query)
    monkeypatch.setattr(rag_hyde, "rewrite_query", lambda query, **_: query)
    monkeypatch.setattr(rag_doc2query, "d2q_enabled", lambda: False)
    monkeypatch.setattr(rag_quota, "quota_enabled", lambda: False)
    monkeypatch.setattr(rag_alias, "alias_enabled", lambda: False)
    monkeypatch.setattr(rag_bm25, "bm25_enabled", lambda: True)
    monkeypatch.setattr(
        rag_bm25, "bm25_search",
        lambda query, top_n, collection_name=None, **kwargs: [
            ("dense document 2", {
                "source": "source_2.md", "source_type": "doc",
                "owner_scope": "curated", "title": "Section 2",
            }, 8.5),
            ("bm25 only", {
                "source": "lexical.md", "source_type": "doc",
                "owner_scope": "curated", "title": "Lexical",
            }, 7.0),
        ],
    )
    monkeypatch.setattr(rag_rerank, "rerank_enabled", lambda: True)

    def fake_rerank(query, docs, metas, dists, top_k, **kwargs):
        seen["rerank_pool"] = top_k
        scores = [0.99 - index * 0.01 for index in range(len(docs[:top_k]))]
        return docs[:top_k], metas[:top_k], dists[:top_k], scores

    monkeypatch.setattr(rag_rerank, "rerank", fake_rerank)
    monkeypatch.setattr(
        rag_route, "apply_audience_routing",
        lambda question, docs, metas, dists: (docs, metas, dists),
    )
    return seen


def test_profile_pool_controls_dense_and_rerank(monkeypatch):
    import rag_gate

    seen = _install_retrieval_stubs(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_POOL", "7")

    result = rag_gate._retrieve_and_classify(
        "RAG 如何工作", top_k=3, source_types={"doc"},
        metadata_filters={"owner_scope": {"curated"}},
        allow_paper_route=False,
    )

    assert seen["queries"][0]["n_results"] == 7
    assert seen["rerank_pool"] == 7
    assert result["retrieval_profile"] == "baseline"
    assert result["effective_hit"] is True
    trace = result["retrieval_trace"]
    assert len(trace["dense_candidates"]) == 7
    assert len(trace["bm25_candidates"]) == 2
    assert len(trace["fusion_candidates"]) == 7
    assert len(trace["reranked_candidates"]) == 7
    assert len(trace["final_candidates"]) == 3
    assert trace["dense_candidates"][0]["chunk_id"] == "dense_0"
    assert trace["dense_candidates"][0]["heading_path"] == "Section 0"
    assert trace["bm25_candidates"][0]["bm25_score"] == 8.5
    assert trace["gate_features"]["rerank_top"] == 0.99
    assert trace["reranker_requested"]
    assert trace["reranker_actual"] == trace["reranker_requested"]
    assert trace["reranker_status"] == "ok"
    assert trace["reranker_scored_count"] == 7
    assert trace["gate_candidate_chunk_id"]
    assert trace["final_top1_chunk_id"]
    assert trace["gate_alignment"] is True
    assert set(trace["latency_by_stage"]) >= {
        "embedding", "dense", "bm25", "fusion", "rerank", "finalize", "total",
    }
    assert trace["index_fingerprint"]
    assert "document" not in trace["final_candidates"][0]


def test_source_cap_profile_is_opt_in_and_baseline_membership_is_unchanged(monkeypatch):
    import rag_bm25
    import rag_gate
    from rag_retrieval_trace import RetrievalProfile

    _install_retrieval_stubs(monkeypatch, dense_source="dominant.md")
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setattr(
        rag_bm25,
        "bm25_search",
        lambda query, top_n, collection_name=None, **kwargs: [
            (
                f"lexical document {index}",
                {
                    "source": f"lexical_{index}.md",
                    "source_type": "doc",
                    "owner_scope": "curated",
                    "title": f"Lexical {index}",
                },
                10.0 - index,
            )
            for index in range(10)
        ],
    )
    baseline = RetrievalProfile(name="baseline-test", pool_size=9)
    capped = RetrievalProfile(
        name="source-cap-test", pool_size=9,
        candidate_pool_strategy="source_cap4",
    )

    reference_plan = {
        "decision": "answer",
        "routes": [{"source": "reference_kb", "operation": "search"}],
    }
    baseline_trace = rag_gate.retrieve_with_trace(
        "RAG 如何工作", reference_plan,
        baseline, top_k=3,
    )
    capped_trace = rag_gate.retrieve_with_trace(
        "RAG 如何工作", reference_plan,
        capped, top_k=3,
    )

    baseline_sources = [item.source for item in baseline_trace.fusion_candidates]
    capped_sources = [item.source for item in capped_trace.fusion_candidates]
    assert baseline_sources.count("dominant.md") > 4
    assert capped_sources.count("dominant.md") == 4
    assert len(capped_trace.fusion_candidates) == 9
    assert len({item.chunk_id for item in capped_trace.fusion_candidates}) == 9
    assert baseline_trace.retrieval_profile.candidate_pool_strategy == "baseline_rrf20"
    assert capped_trace.retrieval_profile.candidate_pool_strategy == "source_cap4"


def _install_post_rank_top1_change(monkeypatch):
    """Make deterministic audience routing select a weak reranker candidate."""

    import rag_rerank
    import rag_route

    def scored_rerank(query, docs, metas, dists, top_k, **kwargs):
        docs, metas, dists = docs[:top_k], metas[:top_k], dists[:top_k]
        return docs, metas, dists, [0.99] + [0.20] * (len(docs) - 1)

    def route_second_first(question, docs, metas, dists):
        order = [1, 0] + list(range(2, len(docs)))
        return (
            [docs[index] for index in order],
            [metas[index] for index in order],
            [dists[index] for index in order],
        )

    monkeypatch.setattr(rag_rerank, "rerank", scored_rerank)
    monkeypatch.setattr(rag_route, "apply_audience_routing", route_second_first)


def test_gate_scores_the_actual_final_top1_after_deterministic_routing(monkeypatch):
    """A post-rank candidate cannot borrow the raw reranker winner's score."""

    import rag_gate

    _install_retrieval_stubs(monkeypatch)
    _install_post_rank_top1_change(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_GATE_MIN", "0.85")
    monkeypatch.setenv("RAG_CRAG", "0")

    result = rag_gate._retrieve_and_classify(
        "RAG 如何工作", top_k=3, source_types={"doc"},
        allow_paper_route=False,
    )

    trace = result["retrieval_trace"]
    assert trace["reranked_candidates"][0]["rerank_score"] == 0.99
    assert trace["final_candidates"][0]["rerank_score"] == 0.20
    assert trace["gate_features"]["rerank_top"] == 0.20
    assert result["rerank_top"] == 0.20
    assert result["in_kb"] is False
    assert trace["gate_candidate_chunk_id"] == trace["final_top1_chunk_id"]
    assert trace["gate_alignment"] is True


def test_crag_recovery_runs_after_final_top1_gate_rejection(monkeypatch):
    import rag_gate

    _install_retrieval_stubs(monkeypatch)
    _install_post_rank_top1_change(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_GATE_MIN", "0.85")
    monkeypatch.setenv("RAG_CRAG", "1")
    seen = []
    monkeypatch.setattr(
        rag_gate, "_crag_recover",
        lambda *args, **kwargs: seen.append((args, kwargs)),
    )

    result = rag_gate._retrieve_and_classify(
        "RAG 如何工作", top_k=3, source_types={"doc"},
        allow_paper_route=False,
    )

    assert result["in_kb"] is False
    assert len(seen) == 1
    assert seen[0][1]["_shadow_internal"] is True


def test_paper_fallback_compares_against_final_main_top1_score(monkeypatch):
    import rag_gate
    import rag_paper_route

    _install_retrieval_stubs(monkeypatch)
    _install_post_rank_top1_change(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RERANK_GATE_MIN", "0.85")
    monkeypatch.setenv("RAG_CRAG", "0")
    seen = []

    def paper_probe(question, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(rag_paper_route, "maybe_paper_route", paper_probe)
    result = rag_gate._retrieve_and_classify(
        "RAG 论文如何工作", top_k=3, allow_paper_route=True,
    )

    assert result["in_kb"] is False
    assert len(seen) == 1
    assert seen[0]["main_in_kb"] is False
    assert seen[0]["main_rerank_top"] == 0.20


def test_retrieve_with_trace_uses_production_source_policy(monkeypatch):
    import rag_gate

    seen = _install_retrieval_stubs(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    trace = rag_gate.retrieve_with_trace(
        "解释 RAG", query_plan={
            "decision": "answer",
            "routes": [{"source": "reference_kb", "operation": "search"}],
        }, top_k=2,
    )

    assert seen["queries"]
    assert trace.metadata_filters["metadata"] == {"owner_scope": ["curated"]}
    assert "experience" in trace.metadata_filters["exclude_source_types"]
    assert len(trace.final_candidates) == 2
    # In-process evaluators may opt into document text; API/SSE callers cannot.
    assert trace.to_dict(include_documents=True)["final_candidates"][0]["document"]


def test_source_include_and_exclude_filters_are_both_enforced(monkeypatch):
    """An allow-list must not silently disable an independently supplied deny-list."""

    import rag_gate

    seen = _install_retrieval_stubs(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")
    rag_gate._retrieve_and_classify(
        "RAG 如何工作",
        top_k=2,
        source_types={"doc", "experience"},
        exclude_source_types={"experience"},
        allow_paper_route=False,
    )

    assert seen["queries"][0]["where"] == {
        "$and": [
            {"source_type": {"$in": ["doc", "experience"]}},
            {"source_type": {"$nin": ["application_jd", "experience", "jd"]}},
        ]
    }


def test_resolved_candidate_profile_is_opt_in(monkeypatch):
    from rag_retrieval_trace import (
        resolve_candidate_shadow_profile, resolve_retrieval_profile,
        retrieval_profile_registry,
    )

    monkeypatch.delenv("RAG_RETRIEVAL_PROFILE", raising=False)
    monkeypatch.delenv("RAG_RERANK_POOL", raising=False)
    monkeypatch.setenv("RAG_BASELINE_RERANK_MODEL", "baseline-model")
    monkeypatch.setenv("RAG_CANDIDATE_RERANK_MODEL", "candidate-model")
    monkeypatch.setenv("RAG_BASELINE_RERANK_POOL", "20")
    monkeypatch.setenv("RAG_RECALL_N", "20")
    assert resolve_retrieval_profile().name == "baseline"
    assert resolve_retrieval_profile().pool_size == 20
    assert resolve_retrieval_profile().candidate_pool_strategy == "baseline_rrf20"
    monkeypatch.setenv("RAG_CANDIDATE_RERANK_POOL", "10")
    monkeypatch.setenv("RAG_CANDIDATE_POOL_STRATEGY", "source_cap4")
    candidate = resolve_retrieval_profile("candidate")
    assert (candidate.name, candidate.pool_size, candidate.reranker_model) == (
        "candidate", 10, "candidate-model",
    )
    assert candidate.candidate_pool_strategy == "source_cap4"
    assert resolve_retrieval_profile("baseline").candidate_pool_strategy == "baseline_rrf20"
    assert resolve_retrieval_profile("baseline").reranker_model == "baseline-model"
    assert resolve_retrieval_profile("candidate_shadow") == resolve_retrieval_profile("baseline")
    assert resolve_candidate_shadow_profile() == candidate
    monkeypatch.setenv("RAG_RETRIEVAL_PROFILE", "candidate")
    assert resolve_retrieval_profile().reranker_model == "candidate-model"
    monkeypatch.setenv("RAG_RETRIEVAL_PROFILE", "baseline")
    assert resolve_retrieval_profile().reranker_model == "baseline-model"
    for profile in retrieval_profile_registry().values():
        assert profile.enable_hyde is False
        assert profile.enable_query_rewrite is False
        assert profile.enable_doc2query is False
        assert profile.enable_quota is False
        assert profile.enable_alias is False
        assert profile.enable_rerank_bridge is False
    try:
        retrieval_profile_registry()["new"] = candidate
    except TypeError:
        pass
    else:
        raise AssertionError("retrieval profile registry is mutable")


def test_candidate_shadow_keeps_synchronous_baseline_result(monkeypatch):
    import rag_gate
    import rag_retrieval_shadow

    seen = _install_retrieval_stubs(monkeypatch)
    monkeypatch.setattr(
        rag_retrieval_shadow, "enqueue_shadow_comparison", lambda *args, **kwargs: True,
    )
    monkeypatch.setenv("RAG_RERANK", "1")
    monkeypatch.setenv("RAG_RETRIEVAL_PROFILE", "candidate_shadow")
    monkeypatch.setenv("RAG_BASELINE_RERANK_POOL", "7")
    monkeypatch.setenv("RAG_CANDIDATE_RERANK_POOL", "3")
    monkeypatch.setenv("RAG_BASELINE_RERANK_MODEL", "baseline-model")
    monkeypatch.setenv("RAG_CANDIDATE_RERANK_MODEL", "candidate-model")

    result = rag_gate._retrieve_and_classify(
        "RAG 如何工作", top_k=2, source_types={"doc"},
        metadata_filters={"owner_scope": {"curated"}},
        allow_paper_route=False,
    )

    assert seen["queries"][0]["n_results"] == 7
    assert result["retrieval_profile"] == "baseline"
    assert result["retrieval_trace"]["reranker_requested"] == "baseline-model"


def test_missing_reranker_scores_are_visible_in_trace(monkeypatch):
    import rag_gate
    import rag_rerank

    _install_retrieval_stubs(monkeypatch)
    monkeypatch.setenv("RAG_RERANK", "1")

    def missing_scores(query, docs, metas, dists, top_k, **kwargs):
        return docs[:top_k], metas[:top_k], dists[:top_k], []

    monkeypatch.setattr(rag_rerank, "rerank", missing_scores)
    result = rag_gate._retrieve_and_classify(
        "RAG 如何工作", top_k=2, source_types={"doc"},
        metadata_filters={"owner_scope": {"curated"}},
        allow_paper_route=False,
    )
    trace = result["retrieval_trace"]
    assert trace["reranker_status"] == "missing_scores"
    assert trace["reranker_actual"] == ""
    assert trace["reranker_scored_count"] == 0


def test_candidate_shadow_returns_baseline_before_candidate_finishes(
    monkeypatch, tmp_path,
):
    import threading
    import rag_gate
    import rag_retrieval_shadow

    _install_retrieval_stubs(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def slow_failing_candidate(job):
        seen["top_k"] = job.top_k
        seen["source_types"] = set(job.source_types)
        seen["metadata_filters"] = {
            key: set(values) for key, values in job.metadata_filters.items()
        }
        started.set()
        release.wait(2)
        raise RuntimeError("private query must not escape through this error")

    rag_retrieval_shadow.reset_shadow_state(
        log_path=tmp_path / "shadow.jsonl", runner=slow_failing_candidate,
    )
    monkeypatch.setenv("RAG_RETRIEVAL_PROFILE", "candidate_shadow")
    monkeypatch.setenv("RAG_BASELINE_RERANK_POOL", "7")
    monkeypatch.setenv("RAG_CANDIDATE_RERANK_POOL", "3")

    result = rag_gate._retrieve_and_classify(
        "private query must not escape", top_k=2, source_types={"doc"},
        metadata_filters={"owner_scope": {"curated"}},
        allow_paper_route=False,
    )

    # The candidate is still blocked, proving the synchronous result did not
    # wait for background completion or expose the candidate profile.
    assert result["retrieval_profile"] == "baseline"
    assert started.wait(1)
    assert not release.is_set()
    assert seen == {
        "top_k": 2,
        "source_types": {"doc"},
        "metadata_filters": {"owner_scope": {"curated"}},
    }
    release.set()
    assert rag_retrieval_shadow.flush_shadow_jobs(2)
    assert rag_retrieval_shadow.shadow_metrics()["errors"] == 1
    rag_retrieval_shadow.reset_shadow_state()


def test_sse_forwards_optional_retrieval_identity(monkeypatch):
    import json
    from fastapi.testclient import TestClient
    import rag_api
    import rag_gate

    def fake_stream(query, top_k):
        yield {
            "type": "meta", "in_kb": True, "mode": "kb_grounded",
            "sources": ["source.md"], "matched_by": "vector",
            "retrieval_profile": "baseline",
            "index_fingerprint": "fp_123", "effective_hit": True,
        }
        yield {"type": "done"}

    monkeypatch.setattr(rag_gate, "gated_query_stream", fake_stream)
    response = TestClient(rag_api.app).post(
        "/api/stream", json={"query": "RAG", "top_k": 3},
    )
    events = [
        json.loads(line[6:]) for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["retrieval_profile"] == "baseline"
    assert events[0]["index_fingerprint"] == "fp_123"
    assert events[0]["effective_hit"] is True
