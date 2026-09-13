# -*- coding: utf-8 -*-
"""Contracts for the route-faithful answerability traffic producer."""

import json
from pathlib import Path
import random
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_seed_projection_hides_old_labels_and_evidence(tmp_path):
    from scripts.run_answerability_shadow_traffic_agent import load_seed_questions

    package = tmp_path / "seed"
    package.mkdir()
    (package / "private_audit.jsonl").write_text(json.dumps({
        "question": "MVCC 是不是给向量库去重？",
        "occurrence_count": 3,
        "verdicts": [{"grade": 3, "relation": "contradicts"}],
        "document": "the producer must never see this",
        "intervention_type": "correct_premise",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (package / "manifest.json").write_text(json.dumps({
        "schema_version": "answerability-trigger-seed-v1",
    }), encoding="utf-8")

    seeds, lineage = load_seed_questions(package)

    assert set(seeds[0]) == {
        "question", "question_sha256", "occurrence_count",
    }
    assert seeds[0]["occurrence_count"] == 3
    assert lineage["producer_visible_fields"] == [
        "question", "question_sha256", "occurrence_count",
    ]
    assert lineage["package"] == "seed"


def test_deterministic_producer_tags_every_row_as_agent_generated():
    from scripts.run_answerability_shadow_traffic_agent import produce_queries

    seeds = [{
        "question": "RAG 的门控如何工作？",
        "question_sha256": "a" * 64,
        "occurrence_count": 2,
    }]
    rows = produce_queries(
        seeds, 5, producer="deterministic", rng=random.Random(7),
        producer_model="unused", llm_batch_size=20,
    )

    assert len(rows) == 5
    assert {row["traffic_origin"] for row in rows} == {"agent_generated"}
    assert all(row["traffic_id"] and row["stratum"] == "main" for row in rows)
    assert all("verdict" not in row and "document" not in row for row in rows)


def test_historical_replay_uses_exact_projected_questions_and_counts():
    from scripts.run_answerability_shadow_traffic_agent import (
        historical_replay_producer,
    )

    seeds = [
        {
            "question": "  MVCC 是不是给向量库去重？  ",
            "question_sha256": "a" * 64,
            "occurrence_count": 3,
        },
        {
            "question": "RAG 的门控如何工作？",
            "question_sha256": "b" * 64,
            "occurrence_count": 1,
        },
    ]

    rows = historical_replay_producer(seeds, 2)

    assert [row["question"] for row in rows] == [
        "MVCC 是不是给向量库去重？", "RAG 的门控如何工作？",
    ]
    assert [row["occurrence_count"] for row in rows] == [3, 1]
    assert {row["traffic_origin"] for row in rows} == {"historical_replay"}
    assert {row["transform"] for row in rows} == {"exact_replay"}
    assert all(row["distribution_eligible"] is False for row in rows)


def test_reference_probe_topic_selection_is_metadata_only_and_excludes_products():
    from scripts.run_answerability_shadow_traffic_agent import (
        select_reference_probe_topics,
    )

    topics, lineage = select_reference_probe_topics([
        {
            "owner_scope": "curated", "source_type": "resource",
            "source": "/kb/rag_architecture.md", "title": "一、向量数据库的作用",
            "document": "must not be copied",
        },
        {
            "owner_scope": "personal", "source_type": "experience",
            "source": "/private/interview.md", "title": "我的面试经历",
        },
        {
            "owner_scope": "curated", "source_type": "resource",
            "source": "/kb/localflow_project_docs.md", "title": "Trace stream",
        },
        {
            "owner_scope": "curated", "source_type": "resource",
            "source": "/kb/dev80_questions.md", "title": "RRF",
        },
        {
            "owner_scope": "curated", "source_type": "resource",
            "source": "/kb/redis.md", "title": "正文内容",
        },
    ])

    assert len(topics) == 1
    assert topics[0]["source_name"] == "rag_architecture.md"
    assert topics[0]["title"] == "向量数据库的作用"
    assert set(topics[0]) == {"topic_id", "source_name", "title", "source_type"}
    assert lineage["producer_visible_fields"] == [
        "topic_id", "source_basename", "section_title", "source_type",
    ]
    assert lineage["document_text_visible"] is False
    assert lineage["judge_labels_visible"] is False


def test_reference_probe_producer_is_non_distributional_and_blind():
    from scripts.run_answerability_shadow_traffic_agent import (
        reference_probe_producer,
    )

    topics = [{
        "topic_id": "a" * 64,
        "source_name": "rag_architecture.md",
        "title": "向量数据库的作用",
        "source_type": "resource",
    }]
    captured = {}

    def caller(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        captured["kwargs"] = kwargs
        return json.dumps([{
            "question": "向量数据库在RAG系统中主要起什么作用？",
            "topic": "a" * 12,
            "transform": "mechanism_question",
        }], ensure_ascii=False)

    rows = reference_probe_producer(
        topics, 1, rng=random.Random(9), model="producer-model",
        batch_size=20, caller=caller,
    )

    assert len(rows) == 1
    assert rows[0]["traffic_origin"] == "agent_generated_reference_probe"
    assert rows[0]["stratum"] == "reference_probe_catalog"
    assert rows[0]["distribution_eligible"] is False
    assert rows[0]["probe_topic_sha256"] == "a" * 64
    assert rows[0]["seed_question_sha256"] == ""
    assert "向量数据库的作用" in captured["prompt"]
    assert "document text" in captured["prompt"]
    assert captured["kwargs"]["model"] == "producer-model"
    assert "grade" not in captured["prompt"]


def test_evidence_seed_selection_rejects_short_and_structural_chunks():
    from scripts.run_answerability_shadow_traffic_agent import (
        select_reference_probe_evidence_seeds,
    )

    good = (
        "向量数据库负责存储向量表示，并通过近似最近邻索引返回与查询最相似的内容。"
        * 12
    )
    seeds, lineage = select_reference_probe_evidence_seeds(
        [good, "太短", "# A\n## B\n### C\n#### D\n##### E\n" * 30],
        [
            {"owner_scope": "curated", "source_type": "resource",
             "source": "/kb/vector.md", "title": "向量数据库的作用"},
            {"owner_scope": "curated", "source_type": "resource",
             "source": "/kb/short.md", "title": "短文本示例"},
            {"owner_scope": "curated", "source_type": "resource",
             "source": "/kb/toc.md", "title": "结构文本示例"},
        ],
    )

    assert len(seeds) == 1
    assert seeds[0]["source_name"] == "vector.md"
    assert len(seeds[0]["excerpt"]) <= 1400
    assert lineage["document_text_visible"] is True
    assert lineage["evidence_seeded"] is True
    assert lineage["eligible_evidence_seeds"] == 1


def test_evidence_probe_positive_cannot_be_upgraded_to_wrong_relation():
    from scripts.run_answerability_shadow_traffic_agent import (
        evidence_reference_probe_producer,
    )

    seeds = [{
        "evidence_id": "c" * 64, "source_name": "vector.md",
        "title": "向量数据库的作用", "source_type": "resource",
        "excerpt": "向量数据库通过近似最近邻索引返回相似内容。",
    }]

    def caller(messages, **kwargs):
        # Violates the requested positive type and must be discarded.
        return json.dumps([{
            "question": "向量数据库是不是负责事务并发控制？",
            "evidence": "c" * 12,
            "probe_type": "evidence_wrong_relation",
            "transform": "wrong_relation",
        }], ensure_ascii=False)

    rows = evidence_reference_probe_producer(
        seeds, 1, profile="evidence_positive", rng=random.Random(5),
        model="producer-model", caller=caller,
    )

    assert rows == []


def test_adversarial_probe_keeps_negative_stratum_out_of_distribution():
    from scripts.run_answerability_shadow_traffic_agent import (
        adversarial_reference_probe_producer,
    )

    topics = [
        {
            "topic_id": "a" * 64, "source_name": "mysql.md",
            "title": "MVCC并发控制", "source_type": "resource",
        },
        {
            "topic_id": "b" * 64, "source_name": "vector.md",
            "title": "向量数据库去重", "source_type": "resource",
        },
    ]
    captured = {}

    def caller(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return json.dumps([{
            "question": "MVCC是不是专门用来给向量数据库去重的？",
            "pair": 1,
            "probe_type": "wrong_relation",
            "transform": "false_relation",
        }], ensure_ascii=False)

    rows = adversarial_reference_probe_producer(
        topics, 1, rng=random.Random(4), model="producer-model",
        batch_size=20, caller=caller,
    )

    assert len(rows) == 1
    assert rows[0]["stratum"] == "reference_probe_adversarial"
    assert rows[0]["probe_type"] == "wrong_relation"
    assert rows[0]["distribution_eligible"] is False
    assert rows[0]["traffic_origin"] == "agent_generated_reference_probe"
    assert "document text" in captured["prompt"]
    assert "judge labels" in captured["prompt"]
    assert "grade" not in captured["prompt"]


def test_execution_plan_contains_only_the_selected_reference_route():
    from rag_query_plan import QueryPlan, RouteSpec
    from scripts.run_answerability_shadow_traffic_agent import _reference_only_plan

    original = QueryPlan(planner_mode="intelligent", routes=[
        RouteSpec("application_experience", "search"),
        RouteSpec("reference_kb", "search", subquery="MVCC"),
    ])

    execution = _reference_only_plan(original)

    assert [route.source for route in execution.routes] == ["reference_kb"]
    assert [route.source for route in original.routes] == [
        "application_experience", "reference_kb",
    ]


def test_public_source_status_discards_exception_text():
    from scripts.run_answerability_shadow_traffic_agent import _public_source_status

    public = _public_source_status({
        "reference_kb": {
            "status": "error", "count": 0, "latency_ms": 2.5,
            "error": "/Users/<user>/chroma_db and raw query",
        },
    })

    assert public == {
        "reference_kb": {"status": "error", "count": 0, "latency_ms": 2.5},
    }


def test_sample_policy_keeps_accepts_stress_and_random_controls():
    from scripts.run_answerability_shadow_traffic_agent import sample_for_judging

    rows = [
        {"gate_decision": True, "stratum": "main"},
        {"gate_decision": False, "stratum": "stress"},
        {"gate_decision": False, "stratum": "main"},
    ]
    sample_for_judging(
        rows, mode="sampled", control_rate=0.0, rng=random.Random(3),
    )

    assert [row["judge_selected"] for row in rows] == [True, True, False]
    assert [row["sampling_reason"] for row in rows] == [
        "all_gate_accepts", "all_stress", "random_main_control",
    ]
    assert rows[0]["sampling_weight"] == 1.0
    assert rows[2]["sampling_weight"] is None


def test_public_export_excludes_raw_questions_evidence_and_chunk_ids():
    from scripts.run_answerability_shadow_traffic_agent import (
        package_rows, validate_public_exports,
    )

    question = "这是一个只应保存在本地审计文件中的问题"
    document = "这是一个只应保存在本地审计文件中的证据片段"
    chunk_id = "private-chunk-id-42"
    reference = {
        "traffic_id": "traffic-1",
        "traffic_origin": "agent_generated",
        "stratum": "main",
        "producer": "deterministic",
        "producer_model": "",
        "transform": "exact_replay",
        "seed_question_sha256": "1" * 64,
        "question": question,
        "retrieval_question": question,
        "original_question_sha256": "2" * 64,
        "retrieval_question_sha256": "2" * 64,
        "reference_capture": 1,
        "features": {"rerank_top": 0.95, "rerank_margin": 0.01},
        "gate_decision": True,
        "effective_hit": True,
        "routes": [{"source": "reference_kb", "operation": "search"}],
        "retrieval_profile": "baseline",
        "index_fingerprint": "fp",
        "reranker_requested": "r",
        "reranker_actual": "r",
        "reranker_status": "ok",
        "planning_ms": 1.0,
        "execution_ms": 2.0,
        "judge_selected": True,
        "sampling_probability": 1.0,
        "sampling_weight": 1.0,
        "sampling_reason": "all_gate_accepts",
        "candidates_private": [{
            "rank": 1, "chunk_id": chunk_id, "source": "/private/source.md",
            "source_type": "reference", "heading_path": "私有章节",
            "rerank_score": 0.95, "document": document,
        }],
        "verdicts": [{
            "rank": 1,
            "verdict": {"grade": 3, "relation": "entails", "reason": "ok"},
            "action": "answer", "cache_hit": False, "latency_ms": 3.0,
            "chunk_id": chunk_id, "rerank_score": 0.95,
        }],
    }
    route = {
        "schema_version": "answerability-shadow-traffic-agent-v1",
        "traffic_id": "traffic-1", "question_sha256": "2" * 64,
        "traffic_origin": "agent_generated", "stratum": "main",
        "producer": "deterministic", "transform": "exact_replay",
        "seed_question_sha256": "1" * 64, "decision": "answer",
        "resolver_mode": "llm", "planner_engine": "gpt",
        "planner_version": "v1", "route_model": "m",
        "routes": [{"source": "reference_kb", "operation": "search"}],
        "has_reference_kb": True, "planning_ms": 1.0,
    }

    public_routes, public_rows, private_rows, _ = package_rows(
        [route], [reference], judge_depth=1, judge_model="judge-model",
    )
    validate_public_exports(public_routes, public_rows, [reference])

    encoded = json.dumps(public_rows, ensure_ascii=False)
    assert question not in encoded and document not in encoded
    assert chunk_id not in encoded and "/private/source.md" not in encoded
    assert public_rows[0]["judge_schema"] == "answerability-v4"
    assert public_rows[0]["judge_question_role"] == "original_user_question"
    assert private_rows[0]["question"] == question
