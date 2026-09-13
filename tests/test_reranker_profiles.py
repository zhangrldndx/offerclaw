from __future__ import annotations

from types import SimpleNamespace


def test_ab_profiles_are_explicit_and_do_not_inherit_experimental_switches():
    from eval_reranker_profiles import arm_profile

    a0 = arm_profile("A0")
    a1 = arm_profile("A1")
    a2 = arm_profile("A2")
    a3 = arm_profile("A3")
    b0 = arm_profile("B0")
    c1 = arm_profile("C1")
    c2 = arm_profile("C2")
    assert (a0.pool_size, a0.reranker_model) == (20, "BAAI/bge-reranker-base")
    assert (a1.pool_size, a1.reranker_model) == (20, "BAAI/bge-reranker-v2-m3")
    assert a2.pool_size == 10
    assert a3.reranker_use_breadcrumb is True
    assert b0.reranker_model == "BAAI/bge-reranker-base"
    assert b0.pool_size == 20 and b0.reranker_use_breadcrumb is True
    assert c1.reranker_model == "BAAI/bge-reranker-base"
    assert c1.pool_size == 20 and c1.reranker_use_breadcrumb is False
    assert c1.candidate_pool_strategy == "source_cap4"
    assert c2.reranker_model == "BAAI/bge-reranker-base"
    assert c2.pool_size == 20 and c2.reranker_use_breadcrumb is True
    assert c2.candidate_pool_strategy == "source_cap4"
    assert all(
        profile.candidate_pool_strategy == "baseline_rrf20"
        for profile in (a0, a1, a2, a3, b0)
    )
    for profile in (a0, a1, a2, a3, b0, c1, c2):
        assert profile.enable_hyde is False
        assert profile.enable_query_rewrite is False
        assert profile.enable_doc2query is False
        assert profile.enable_quota is False
        assert profile.enable_alias is False
        assert profile.enable_rerank_bridge is False


def test_hard_negative_arm_requires_an_audited_development_model(monkeypatch, tmp_path):
    from eval_reranker_profiles import arm_profile

    model = tmp_path / "h1"
    model.mkdir()
    monkeypatch.setenv("RAG_HARD_NEGATIVE_MODEL", str(model))
    try:
        arm_profile("H1")
    except ValueError as exc:
        assert "audited development model" in str(exc)
    else:
        raise AssertionError("unaudited H1 model was accepted")

    (model / "DEVELOPMENT_ONLY").write_text("Not approved.\n", encoding="utf-8")
    profile = arm_profile("H1")
    assert profile.reranker_model == str(model.resolve())
    assert profile.pool_size == 20
    assert profile.reranker_use_breadcrumb is True


def test_rerank_accepts_explicit_model_and_separate_scoring_text(monkeypatch):
    import rag_rerank

    seen = {}

    class FakeModel:
        def predict(self, pairs):
            seen["pairs"] = pairs
            return [0.1, 0.9]

    def fake_load(model_name=None):
        seen["model"] = model_name
        return FakeModel()

    monkeypatch.setattr(rag_rerank, "_load_reranker", fake_load)
    monkeypatch.setattr(rag_rerank, "rerank_enabled", lambda: True)
    docs = ["body-a", "body-b"]
    ranked, _, _, scores = rag_rerank.rerank(
        "query", docs, [{}, {}], [0.1, 0.2], 2,
        model_name="BAAI/bge-reranker-v2-m3",
        pair_docs=["title-a\nbody-a", "title-b\nbody-b"],
    )
    assert seen["model"] == "BAAI/bge-reranker-v2-m3"
    assert seen["pairs"] == [
        ["query", "title-a\nbody-a"], ["query", "title-b\nbody-b"],
    ]
    assert ranked == ["body-b", "body-a"]
    assert scores == [0.9, 0.1]


def test_profile_fingerprint_changes_with_actual_reranker():
    from eval_reranker_profiles import arm_profile
    from rag_retrieval_trace import bind_profile_fingerprint

    raw = {
        "collection": "kb", "collection_count": 3348,
        "collection_content_hash": "same", "fingerprint_id": "old",
        "generated_at": "ignored", "rerank_model": "stale-env-model",
    }
    a0 = bind_profile_fingerprint(raw, arm_profile("A0"))
    a1 = bind_profile_fingerprint(raw, arm_profile("A1"))
    assert a0["rerank_model"] == "BAAI/bge-reranker-base"
    assert a1["rerank_model"] == "BAAI/bge-reranker-v2-m3"
    assert a0["fingerprint_id"] != a1["fingerprint_id"]


def test_breadcrumb_scoring_text_uses_safe_source_and_section_labels():
    from rag_gate import _rerank_pair_documents

    pair_docs = _rerank_pair_documents(
        ["正文内容"],
        [{
            "source": "/private/absolute/path/source.md",
            "title": "当前章节",
            "heading_path": "上级 > 当前章节",
        }],
        use_breadcrumb=True,
    )
    assert pair_docs == [
        "来源文档：source.md\n章节：当前章节\n章节路径：上级 > 当前章节\n正文：\n正文内容"
    ]
    assert "/private/absolute" not in pair_docs[0]
    assert _rerank_pair_documents(["正文"], [{}], use_breadcrumb=False) is None


def test_arm_comparison_counts_paired_wins_and_losses():
    from eval_reranker_profiles import compare_arms

    def arm(name, rows):
        return {
            "arm": name,
            "sets": [{"set": "heldout52", "runs": [{
                "metrics": {"p95_ms": 100.0}, "rows": rows,
            }]}],
        }

    common = {"rerank_margin": 0.1}
    base = arm("A0", [
        {"id": "q1", "rank": 0, "top1_source": "wrong-a",
         "direct_target_chunk_ids": ["p1"], "direct_rank": 0, **common},
        {"id": "q2", "rank": 1, "top1_source": "right-b",
         "direct_target_chunk_ids": ["p2"], "direct_rank": 1, **common},
    ])
    candidate = arm("A1", [
        {"id": "q1", "rank": 1, "top1_source": "right-a",
         "direct_target_chunk_ids": ["p1"], "direct_rank": 1, **common},
        {"id": "q2", "rank": 2, "top1_source": "wrong-b",
         "direct_target_chunk_ids": ["p2"], "direct_rank": 2, **common},
    ])
    comparison = compare_arms(base, candidate, "heldout52")
    assert [row["id"] for row in comparison["wins"]] == ["q1"]
    assert [row["id"] for row in comparison["losses"]] == ["q2"]
    assert comparison["net_wins"] == 0
    assert [row["id"] for row in comparison["direct_wins"]] == ["q1"]
    assert [row["id"] for row in comparison["direct_losses"]] == ["q2"]


def test_qrels_overlay_only_uses_direct_chunks(tmp_path):
    import json
    from eval_reranker_profiles import _load_direct_qrels
    from rag_qrels import answer_span_hash

    direct_excerpt = "直接回答问题的证据。"
    supporting_excerpt = "只有相关背景。"
    overlay = {
        "schema_version": "rag-qrels-overlay-v1",
        "reviewer_id": "independent-reviewer",
        "base_set": "tests/rag_bench_paraphrase_set.json",
        "index": {"collection": "fake-kb", "count": 2},
        "items": [{
            "query_id": "q1",
            "review_outcome": "accepted",
            "review_note": "人工复核。",
            "relevant_targets": [{
                "source": "direct.md", "heading_path": ["章节"],
                "chunk_id": "direct-chunk", "relevance": "direct",
                "evidence_excerpt": direct_excerpt,
                "answer_span_hash": answer_span_hash(direct_excerpt),
                "review_note": "可直接作答。",
            }, {
                "source": "supporting.md", "heading_path": ["章节"],
                "chunk_id": "supporting-chunk", "relevance": "supporting",
                "evidence_excerpt": supporting_excerpt,
                "answer_span_hash": answer_span_hash(supporting_excerpt),
                "review_note": "仅作背景。",
            }],
        }],
    }
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(overlay, ensure_ascii=False), encoding="utf-8")

    class FakeCollection:
        name = "fake-kb"

        @staticmethod
        def count():
            return 2

        @staticmethod
        def get(ids, include):
            rows = {
                "direct-chunk": (direct_excerpt, {"source": "direct.md"}),
                "supporting-chunk": (supporting_excerpt, {"source": "supporting.md"}),
            }
            selected = [chunk_id for chunk_id in ids if chunk_id in rows]
            return {
                "ids": selected,
                "documents": [rows[chunk_id][0] for chunk_id in selected],
                "metadatas": [rows[chunk_id][1] for chunk_id in selected],
            }

    direct, outcomes, meta = _load_direct_qrels(
        path, ["q1"], collection=FakeCollection(),
    )
    assert direct == {"q1": {"direct-chunk"}}
    assert outcomes == {"q1": "accepted"}
    assert meta["reviewer_id"] == "independent-reviewer"
    assert meta["validation"] == "schema_and_live_collection_validated"


def test_release_gate_is_provisional_without_repeat_bench_and_all_gate_sets():
    from eval_reranker_profiles import _release_decision

    candidate = {
        "sets": [{
            "set": "heldout52",
            "runs": [{
                "metrics": {"r1_hits": 30, "p95_ms": 100.0},
                "rows": [],
            }],
        }],
        "qrels_overlay": {
            "validation": "schema_and_live_collection_validated",
        },
        "gates": {},
    }
    comparison = {"losses": [], "p95_ratio": 1.0}
    decision = _release_decision(candidate, comparison)
    assert decision["status"] == "provisional"
    assert decision["passed"] is False
    assert decision["checks"]["repeat_at_least_3"] is False
    assert decision["checks"]["bench100_present"] is False
    assert decision["checks"]["all_three_gate_sets_present"] is False


def test_candidate_reranker_completeness_fails_closed_on_silent_fallback():
    import pytest
    from eval_reranker_profiles import _assert_reranker_complete

    candidates = [SimpleNamespace(rerank_score=None)]
    trace = SimpleNamespace(
        fusion_candidates=candidates,
        reranked_candidates=candidates,
        reranker_status="load_failed",
        reranker_requested="expected-model",
        reranker_actual="",
        reranker_scored_count=0,
    )
    with pytest.raises(RuntimeError, match="scoring incomplete"):
        _assert_reranker_complete(trace)


def test_evaluate_set_checkpoints_after_every_question(monkeypatch):
    import copy
    import rag_gate
    from eval_reranker_profiles import _evaluate_set

    candidate = SimpleNamespace(
        source="answer.md", chunk_id="chunk-1", rerank_score=0.95,
    )
    trace = SimpleNamespace(
        index_metadata={"collection_count": 3348},
        fusion_candidates=[candidate],
        reranked_candidates=[candidate],
        final_candidates=[candidate],
        reranker_status="ok",
        reranker_requested="base",
        reranker_actual="base",
        reranker_scored_count=1,
        gate_candidate_chunk_id="chunk-1",
        final_top1_chunk_id="chunk-1",
        gate_alignment=True,
        gate_decision=True,
        gate_features={"rerank_top": 0.95, "rerank_margin": None},
        latency_by_stage={"total": 10.0},
    )
    monkeypatch.setattr(rag_gate, "retrieve_with_trace", lambda *args, **kwargs: trace)
    checkpoints = []
    result = _evaluate_set(
        "tiny",
        [
            {"id": "q1", "q": "first", "expect_sources": ["answer.md"]},
            {"id": "q2", "q": "second", "expect_sources": ["answer.md"]},
        ],
        SimpleNamespace(name="B0"),
        1,
        3348,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )
    assert result["status"] == "completed"
    assert [len(item["runs"][-1]["rows"]) for item in checkpoints] == [1, 2]
    row = result["runs"][0]["rows"][0]
    assert row["reranked_candidate_ids"] == ["chunk-1"]
    assert row["reranked_scores"] == [0.95]
    assert row["gate_alignment"] is True
    assert row["reranker_status"] == "ok"
