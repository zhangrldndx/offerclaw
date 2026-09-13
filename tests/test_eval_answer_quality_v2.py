# -*- coding: utf-8 -*-
"""Offline contracts for the frozen answer-quality v2 evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import eval_answer_quality_v2 as aq


_GIT = {
    "head": "f" * 40,
    "branch": "test",
    "dirty": False,
    "dirty_entry_count": 0,
}
_RUNTIME = {
    "retrieval_profile": {"name": "test"},
    "index": {"fingerprint_id": "fixture"},
    "answerability": {"schema": "answerability-v4"},
    "generation": {"prompt_contract_sha256": "a" * 64},
    "judges": list(aq.JUDGE_NAMES),
    "judge_prompt_sha256": aq._sha256_text(aq.JUDGE_SYSTEM),
}


@pytest.fixture(scope="module")
def selection() -> dict:
    return aq.build_selection_manifest(git_info=_GIT, runtime=_RUNTIME)


@pytest.mark.private_artifact
def test_selection_is_deterministic_and_meets_all_quotas(selection):
    again = aq.build_selection_manifest(git_info=_GIT, runtime=_RUNTIME)
    assert selection["selection"]["seed_sha256"] == again["selection"]["seed_sha256"]
    assert selection["items"] == again["items"]

    positives = [row for row in selection["items"] if row["cohort"] == "positive"]
    negatives = [row for row in selection["items"] if row["cohort"] == "negative"]
    assert len(positives) == 40 and len(negatives) == 8
    assert len({row["anchor_id"] for row in positives}) == 40
    assert aq.Counter(row["source_key"] for row in positives) == {"v1": 20, "v2a": 20}
    assert aq.Counter(row["domain"] for row in positives) == {
        "algorithm": 10, "backend": 10, "career": 10, "llm_app": 10,
    }
    assert aq.Counter(row["style_bucket"] for row in positives) == {
        "standard": 4, "natural": 12, "oral": 12, "long": 12,
    }
    assert max(aq.Counter(
        row["primary_source_sha256"] for row in positives
    ).values()) <= 3
    assert aq.Counter(row["expected_action"] for row in negatives) == {
        "correct_premise": 4, "abstain": 4,
    }


@pytest.mark.private_artifact
def test_public_selection_contains_no_raw_question_or_evidence(selection):
    serialized = json.dumps(selection, ensure_ascii=False)
    forbidden_keys = {
        "question", "answer_requirements", "relevant_targets", "document",
        "chunks", "contexts", "answer", "reason", "local_path",
    }
    for item in selection["items"]:
        assert forbidden_keys.isdisjoint(item)
    # A representative raw question must not accidentally appear elsewhere in
    # the manifest through a note or diagnostic sample.
    assert "MVCC 是不是用来给向量库去重" not in serialized
    assert selection["release_boundary"]["not_blind"] is True
    assert selection["release_boundary"]["not_organic"] is True
    assert set(selection["code_files"]) == set(aq._code_lineage())
    assert all(
        len(entry["sha256"]) == 64 and entry["bytes"] > 0
        for entry in selection["code_files"].values()
    )


@pytest.mark.private_artifact
def test_resolve_selection_checks_hashes_without_exposing_them(selection, monkeypatch):
    # This fixture uses a synthetic HEAD for deterministic selection; allow the
    # resolver to exercise all real source hashes with that same frozen value.
    monkeypatch.setattr(aq, "_git_info", lambda: dict(_GIT))
    monkeypatch.setattr(aq, "_runtime_snapshot", lambda: dict(_RUNTIME))
    rows = aq.resolve_selection(selection)
    assert len(rows) == 48
    assert len([row for row in rows if row["cohort"] == "positive"]) == 40
    assert all(row["question"] for row in rows)
    assert all(row["answer_requirements"] for row in rows[:40])


@pytest.mark.private_artifact
def test_freeze_is_append_only_and_idempotent(tmp_path, monkeypatch):
    destination = tmp_path / "SELECTION_MANIFEST.json"
    monkeypatch.setattr(aq, "_git_info", lambda: dict(_GIT))
    monkeypatch.setattr(aq, "_runtime_snapshot", lambda: dict(_RUNTIME))
    first = aq.freeze_selection(destination)
    first_bytes = destination.read_bytes()
    second = aq.freeze_selection(destination)
    assert first == second
    assert destination.read_bytes() == first_bytes


@pytest.mark.private_artifact
def test_existing_freeze_rejects_head_or_runtime_drift(tmp_path, monkeypatch):
    destination = tmp_path / "SELECTION_MANIFEST.json"
    monkeypatch.setattr(aq, "_git_info", lambda: dict(_GIT))
    monkeypatch.setattr(aq, "_runtime_snapshot", lambda: dict(_RUNTIME))
    aq.freeze_selection(destination)
    monkeypatch.setattr(aq, "_git_info", lambda: dict(_GIT, head="e" * 40))
    with pytest.raises(aq.AnswerQualityV2Error, match="different git HEAD"):
        aq.freeze_selection(destination)
    monkeypatch.setattr(aq, "_git_info", lambda: dict(_GIT))
    monkeypatch.setattr(
        aq, "_runtime_snapshot", lambda: dict(_RUNTIME, index={"fingerprint_id": "other"})
    )
    with pytest.raises(aq.AnswerQualityV2Error, match="different runtime"):
        aq.freeze_selection(destination)


def test_raw_artifacts_are_rejected_inside_repository(tmp_path):
    with pytest.raises(aq.AnswerQualityV2Error, match="outside the repository"):
        aq.ensure_repository_external(aq.ROOT / ".offerclaw" / "eval_runs")
    assert aq.ensure_repository_external(tmp_path) == tmp_path.resolve()


def test_evaluation_runtime_does_not_read_or_write_shared_app_state(
    tmp_path, monkeypatch,
):
    import rag_answerability

    shared = tmp_path / "shared" / "answerability_cache.json"
    shared.parent.mkdir()
    shared.write_text('{"shared": true}\n', encoding="utf-8")
    private = tmp_path / "private" / "answerability_cache.json"
    private.parent.mkdir()
    original_memory = {"already_loaded": True}
    monkeypatch.setattr(rag_answerability, "CACHE_PATH", shared)
    monkeypatch.setattr(rag_answerability, "_CACHE", original_memory)
    monkeypatch.setenv("LLM_USAGE_LOG", "1")

    with aq.isolated_evaluation_runtime(private):
        assert rag_answerability.CACHE_PATH == private
        assert rag_answerability._CACHE is None
        assert os.environ["LLM_USAGE_LOG"] == "0"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        rag_answerability._CACHE = {"evaluation_only": True}

    assert shared.read_text(encoding="utf-8") == '{"shared": true}\n'
    assert json.loads(private.read_text(encoding="utf-8")) == {
        "evaluation_only": True,
    }
    assert rag_answerability.CACHE_PATH == shared
    assert rag_answerability._CACHE is original_memory
    assert os.environ["LLM_USAGE_LOG"] == "1"


def _positive_raw_row() -> dict:
    return {
        "eval_id": "aqv2-pos-001",
        "cohort": "positive",
        "expected_action": "answer",
        "observed_action": "answer",
        "question": "MVCC 是做什么的？",
        "mode": "kb_grounded",
        "contexts": ["MVCC 提供数据库事务并发控制。", "向量库检索使用向量索引。"],
        "answer_requirements": ["说明 MVCC 用于事务并发控制。"],
        "answer": "MVCC 用于事务并发控制。[资料1]",
    }


def test_structured_judgment_scores_claim_requirement_and_citation():
    row = _positive_raw_row()
    judgment = aq.validate_judgment({
        "claims": [{
            "claim": "MVCC 用于事务并发控制",
            "supported": True,
            "supported_context_ids": [1],
            "cited_context_ids": [1],
        }],
        "requirements": [{"index": 0, "coverage": "full"}],
        "observed_action": "answer",
        "correction_first_sentence_ok": None,
        "reason": "完整且有依据",
    }, row)
    scores = aq.score_judgment(row, judgment)
    assert scores["faithfulness"] == 1.0
    assert scores["completeness"] == 1.0
    assert scores["citation_precision"] == 1.0
    assert scores["citation_recall"] == 1.0
    assert scores["action_correct"] is True
    assert scores["effective_completeness"] == 1.0


def test_judgment_rejects_missing_requirements_and_bad_context_ids():
    row = _positive_raw_row()
    with pytest.raises(aq.AnswerQualityV2Error, match="coverage is incomplete"):
        aq.validate_judgment({
            "claims": [], "requirements": [], "observed_action": "answer",
            "correction_first_sentence_ok": None,
        }, row)
    with pytest.raises(aq.AnswerQualityV2Error, match="out of range"):
        aq.validate_judgment({
            "claims": [{
                "claim": "x", "supported": True,
                "supported_context_ids": [3], "cited_context_ids": [],
            }],
            "requirements": [{"index": 0, "coverage": "none"}],
            "observed_action": "answer", "correction_first_sentence_ok": None,
        }, row)


def test_correct_premise_requires_both_action_and_explicit_first_sentence():
    row = _positive_raw_row() | {
        "expected_action": "correct_premise",
        "observed_action": "correct_premise",
    }
    base = {
        "claims": [], "requirements": [{"index": 0, "coverage": "full"}],
        "observed_action": "correct_premise", "correction_first_sentence_ok": False,
    }
    assert aq.score_judgment(row, base)["action_correct"] is False
    assert aq.score_judgment(
        row, base | {"correction_first_sentence_ok": True}
    )["action_correct"] is True


def test_kb_abstention_is_not_relabelled_false_accept_by_general_fallback():
    row = _positive_raw_row() | {
        "expected_action": "abstain",
        "observed_action": "abstain",
        "mode": "general_fallback",
        "answer_requirements": [],
    }
    judgment = {
        "claims": [], "requirements": [],
        "observed_action": "answer",  # explicitly labelled general answer
        "correction_first_sentence_ok": None,
    }
    scores = aq.score_judgment(row, judgment)
    assert scores["pipeline_action_correct"] is True
    assert scores["generation_action_correct"] is False
    assert scores["action_correct"] is True


def test_answer_action_reads_panel_then_top1_relation(monkeypatch):
    assert aq._answer_action({"in_kb": False}) == "abstain"
    panel = {
        "in_kb": True,
        "retrieval_trace": {"gate_features": {"answerability_rerank": {
            "gate_panel": {"action": "correct_premise"},
        }}},
    }
    assert aq._answer_action(panel) == "correct_premise"
    relation = {
        "in_kb": True,
        "retrieval_trace": {"gate_features": {"answerability_rerank": {
            "grades": {"0": 3}, "relations": {"0": "contradicts"},
        }}},
    }
    assert aq._answer_action(relation) == "correct_premise"
    structural_anchor = {
        "in_kb": True,
        "retrieval_trace": {"gate_features": {
            "gate_anchor_rank": 2,
            "answerability_rerank": {
                "grades": {"0": 1, "1": 3},
                "relations": {"0": "not_established", "1": "contradicts"},
            },
        }},
    }
    assert aq._answer_action(structural_anchor) == "correct_premise"


def test_current_head_generation_does_not_repair_missing_action_wiring(monkeypatch):
    import rag_gate

    captured_actions = []
    retrieval = {
        "in_kb": True,
        "chunks": ["MVCC 用于数据库事务并发控制。"],
        "docs": ["MVCC 用于数据库事务并发控制。"],
        "sources": ["fixture"],
        "has_dists": True,
        "retrieval_profile": "baseline",
        "index_fingerprint": "fixture",
        "retrieval_trace": {
            "gate_features": {"answerability_rerank": {
                "gate_panel": {"action": "correct_premise"},
            }},
            "final_candidates": [],
        },
    }
    monkeypatch.setattr(rag_gate, "_retrieve_and_classify", lambda *_a, **_k: retrieval)

    def grounded(question, chunks, answer_action="answer"):
        captured_actions.append(answer_action)
        return [{"role": "user", "content": question}]

    monkeypatch.setattr(rag_gate, "_grounded_messages", grounded)
    monkeypatch.setattr(
        rag_gate, "_chat",
        lambda _messages, **_kwargs: "MVCC 用于数据库事务并发控制。",
    )
    row = aq._production_reference_once({
        "eval_id": "fixture",
        "query_id": "fixture",
        "cohort": "negative",
        "domain": "negative",
        "style_bucket": "negative",
        "negative_stratum": "wrong_relation_correctable",
        "expected_action": "correct_premise",
        "question": "MVCC 是不是给向量库去重的？",
        "answer_requirements": [],
        "relevant_targets": [],
    })
    assert captured_actions == ["answer"]
    assert row["observed_action"] == "correct_premise"
    assert row["generation_contract_action"] == "answer"


def test_frozen_answerability_environment_is_explicit_and_restored(monkeypatch):
    key = "RAG_ANSWERABILITY_GATE_VOTES"
    monkeypatch.setenv(key, "9")
    manifest = {"runtime": {"answerability": {
        "frozen_env": dict(aq.FROZEN_ANSWERABILITY_ENV),
    }}}
    with aq.frozen_answerability_environment(manifest):
        assert os.environ[key] == "3"
        assert os.environ["RAG_ANSWERABILITY_MODE"] == "teacher"
        assert os.environ["RAG_ANSWERABILITY_PROMPT"] == "v5"
    assert os.environ[key] == "9"


def test_full_context_is_sent_to_judge_without_old_700_character_truncation():
    row = _positive_raw_row()
    tail = "尾部唯一证据"
    row["contexts"] = ["甲" * 900 + tail]
    message = aq._judge_user_message(row)
    assert tail in message
    assert len(message) > 900
    assert "[expected_action]" not in message


def test_summary_keeps_fixed_denominator_and_never_imputes_unavailable():
    rows = []
    for index in range(48):
        positive = index < 40
        rows.append({
            "cohort": "positive" if positive else "negative",
            "answer": "ok",
            "mode": "kb_grounded" if positive else "general_fallback",
            "gold_in_context": positive,
            "judges": {},
        })
    summary = aq.summarize_rows(rows, selection_sha256="a" * 64,
                                raw_sha256="b" * 64)
    assert summary["status"] == "incomplete_no_score_imputation"
    assert summary["coverage"]["selected"] == 48
    for name in aq.JUDGE_NAMES:
        assert summary["judges"][name]["available"] == 0
        assert summary["judges"][name]["required"] == 48
        assert summary["judges"][name]["faithfulness"] is None
        assert summary["judges"][name]["effective_completeness_all_positive"] is None


def test_partial_judge_scores_are_not_published_on_a_smaller_denominator():
    score = {
        "faithfulness": 1.0,
        "completeness": 1.0,
        "effective_completeness": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "action_correct": True,
    }
    rows = []
    for index in range(48):
        positive = index < 40
        judges = {}
        if index < 39:
            judges[aq.JUDGE_NAMES[0]] = {"available": True, "scores": score}
        rows.append({
            "cohort": "positive" if positive else "negative",
            "answer": "ok",
            "mode": "kb_grounded" if positive else "general_fallback",
            "gold_in_context": positive,
            "judges": judges,
        })
    summary = aq.summarize_rows(rows, selection_sha256="a" * 64,
                                raw_sha256="b" * 64)
    first = summary["judges"][aq.JUDGE_NAMES[0]]
    assert first["positive_available"] == 39
    assert first["faithfulness"] is None
    assert first["completeness"] is None
    assert first["citation_precision"] is None
