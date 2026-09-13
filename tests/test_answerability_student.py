from __future__ import annotations

import json
from pathlib import Path

import pytest


def _row(*, query="q1", anchor="a1", source="s1", chunk="c1", text="doc",
         grade=3, split="train", base=0.2, rank=1, style="natural", votes=1):
    from rag_answerability_student_data import SCHEMA, sha256_text
    from rag_answerability import PROMPT_SHA256

    verdict = {"grade": grade, "relation": "entails"}
    return {
        "schema_version": SCHEMA,
        "query_id": query, "anchor_id": anchor, "source_id": source,
        "chunk_id": chunk, "question": "问题", "retrieval_question": "问题",
        "chunk_text": text, "chunk_text_sha256": sha256_text(text),
        "query_style": style, "split": split,
        "current_bge_rank": rank, "current_bge_score": 1.0 / rank,
        "base_student_score": base,
        "teacher_grade": grade, "teacher_relation": "entails",
        "teacher_model": "model", "teacher_prompt_sha256": PROMPT_SHA256,
        "teacher_votes": [dict(verdict) for _ in range(votes)],
        "label_status": "teacher_consensus" if votes > 1 else "teacher_single",
    }


def test_exact_cache_reconnect_requires_original_identity():
    from rag_answerability import cache_key
    from rag_answerability_student_data import reconnect_cached_verdict

    row = _row()
    row["teacher_model"] = "model"
    verdict = {"grade": 3, "relation": "entails"}
    key = cache_key("model", row["question"], row["chunk_text"],
                    retrieval_question=row["retrieval_question"])
    assert reconnect_cached_verdict(row, {key: verdict}, model="model") == verdict
    changed = dict(row, question="另一道问题")
    assert reconnect_cached_verdict(changed, {key: verdict}, model="model") is None
    changed_prompt = dict(row, teacher_prompt_sha256="0" * 64)
    assert reconnect_cached_verdict(changed_prompt, {key: verdict}, model="model") is None


def test_training_refuses_any_blind_row():
    from rag_answerability_student_data import StudentDataError, training_rows

    with pytest.raises(StudentDataError, match="sealed"):
        training_rows([_row(split="train"), _row(query="q2", anchor="a2",
                                                   chunk="c2", split="blind_a")])


def test_source_and_anchor_leakage_are_rejected():
    from rag_answerability_student_data import StudentDataError, audit_split_isolation

    rows = [
        _row(split="train"),
        _row(query="q2", anchor="a2", chunk="c2", split="validation", source="s1"),
    ]
    with pytest.raises(StudentDataError, match="source leakage"):
        audit_split_isolation(rows)


def test_grade2_never_becomes_a_strong_pair():
    from rag_answerability_student_data import build_strong_pairs

    rows = [
        _row(chunk="high", grade=3, base=0.1),
        _row(chunk="middle", grade=2, base=0.9, rank=2),
        _row(chunk="low", grade=1, base=0.8, rank=3),
    ]
    pairs = build_strong_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["positive"]["chunk_id"] == "high"
    assert pairs[0]["negative"]["chunk_id"] == "low"


def test_disputed_labels_and_missing_base_scores_do_not_fake_corrective_mass():
    from rag_answerability_student_data import build_strong_pairs

    high = _row(chunk="high", grade=3, base=None)
    low = _row(chunk="low", grade=0, base=0.8, rank=2)
    assert build_strong_pairs([high, low]) == []
    high["base_student_score"] = 0.1
    high["label_status"] = "teacher_disagreement"
    assert build_strong_pairs([high, low]) == []


def test_pair_weights_prioritise_corrective_examples_without_anchor_normalisation():
    from rag_answerability_student_data import build_strong_pairs

    rows = []
    for index in range(8):
        query = f"q{index}"
        rows.extend([
            _row(query=query, anchor=f"a{index}", source=f"s{index}",
                 chunk=f"h{index}", grade=3, base=0.1),
            _row(query=query, anchor=f"a{index}", source=f"s{index}",
                 chunk=f"l{index}", grade=1, base=0.9, rank=2),
        ])
    pairs = build_strong_pairs(rows)
    assert all(pair["category"] == "corrective" for pair in pairs)
    assert sum(pair["sampling_weight"] for pair in pairs) == pytest.approx(0.5)
    assert max(pair["sampling_weight"] for pair in pairs) <= 0.15


def test_pair_flip_report_measures_order_changes_not_margin():
    from rag_answerability_student_data import build_strong_pairs, pair_flip_report

    rows = [_row(chunk="h", grade=3, base=0.1),
            _row(chunk="l", grade=1, base=0.9, rank=2)]
    pair = build_strong_pairs(rows)[0]
    report = pair_flip_report(
        [pair], {("q1", "h"): 2.0, ("q1", "l"): 1.0},
    )
    assert report["corrected"] == 1 and report["net_flips"] == 1
    assert "margin" not in report


def test_public_manifest_never_copies_private_text(tmp_path):
    from rag_answerability_student_data import public_manifest

    private = tmp_path / "private.jsonl"
    private.write_text("secret question and document\n", encoding="utf-8")
    manifest = public_manifest([_row()], private_path=private)
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "secret question" not in serialized
    assert "问题" not in serialized and "doc" not in serialized


def test_runtime_mode_defaults_off_and_preserves_legacy(monkeypatch):
    import rag_answerability as answerability

    monkeypatch.delenv("RAG_ANSWERABILITY_MODE", raising=False)
    monkeypatch.setenv("RAG_ANSWERABILITY", "0")
    assert answerability.mode() == "off" and answerability.enabled() is False
    monkeypatch.setenv("RAG_ANSWERABILITY", "1")
    assert answerability.mode() == "teacher"
    monkeypatch.setenv("RAG_ANSWERABILITY_MODE", "student")
    assert answerability.mode() == "student"
    monkeypatch.setenv("RAG_ANSWERABILITY_MODE", "typo")
    assert answerability.mode() == "off"


def test_seed_summary_never_treats_mps_as_official_cuda(tmp_path):
    from scripts.summarize_answerability_student_seeds import summarize

    for seed, r1 in ((17, 0.75), (29, 0.61), (43, 0.46)):
        root = tmp_path / f"seed_{seed}"
        root.mkdir(parents=True)
        (root / "student_model_manifest.json").write_text(json.dumps({
            "seed": seed,
            "device": "mps",
            "checkpoint_sha256": str(seed) * 64,
            "dataset": {"sha256": "a" * 64},
            "frozen_hyperparameters": {"learning_rate": 2e-5},
            "final_validation": {
                "base": {"r1": 0.59},
                "post": {"r1": r1, "ndcg5": r1},
                "wins": 1, "losses": 0,
                "pair_flips": {"net_flips": 1},
            },
            "pilot_model_checks": {"check": True},
            "pilot_model_status": "seed_pass",
        }), encoding="utf-8")
    report = summarize(tmp_path)
    assert report["status"] == "diagnostic_only_requires_cuda_rerun"
    assert report["official_cuda_run"] is False


def test_student_ranks_by_grade_then_existing_bge_score():
    from rag_answerability_student import rerank_by_student

    docs = ["topic", "answer-a", "answer-b"]
    predictions = [
        {"grade": 1, "confidence": 0.9},
        {"grade": 3, "confidence": 0.8},
        {"grade": 3, "confidence": 0.7},
    ]
    stats = {}
    out, metas, dists, scores = rerank_by_student(
        "q", docs, [{"i": i} for i in range(3)], [0.1, 0.2, 0.3],
        [0.99, 0.4, 0.8], predictor=lambda _q, _docs: predictions, stats=stats,
    )
    assert out == ["answer-b", "answer-a", "topic"]
    assert metas[0] == {"i": 2} and dists[0] == 0.3 and scores[0] == 0.8
    assert stats["grades"] == {0: 3, 1: 3, 2: 1}


def test_student_unavailable_is_exact_noop():
    from rag_answerability_student import rerank_by_student

    docs = ["a", "b"]
    args = ([{}, {}], [0.1, 0.2], [0.9, 0.8])
    stats = {}
    assert rerank_by_student(
        "q", docs, *args, predictor=lambda _q, _docs: None, stats=stats,
    ) == (docs, *args)
    assert stats == {"applied": False, "reason": "student_unavailable"}


def test_balanced_requires_validation_frozen_policy(monkeypatch):
    from rag_answerability_student import rerank_by_student

    monkeypatch.delenv("RAG_ANSWERABILITY_BALANCED_POLICY", raising=False)
    stats = {}
    rerank_by_student(
        "q", ["a", "b"], [{}, {}], [0.1, 0.2], [0.9, 0.8], balanced=True,
        predictor=lambda _q, _d: ([
            {"grade": 3, "confidence": 0.2, "expected_grade": 1.5},
            {"grade": 2, "confidence": 0.3, "expected_grade": 1.4},
        ], {"model_sha256": "a" * 64}), stats=stats,
    )
    assert stats["fallback_recommended"] is False
    assert stats["balanced_policy_reason"] == "balanced_policy_unavailable"


def test_balanced_policy_is_bound_to_validation_and_checkpoint(monkeypatch, tmp_path):
    from rag_answerability_student import BALANCED_POLICY_SCHEMA, rerank_by_student

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "schema_version": BALANCED_POLICY_SCHEMA,
        "status": "balanced_candidate",
        "frozen_on_split": "validation",
        "student_model_sha256": "a" * 64,
        "min_top_confidence": 0.8,
        "min_expected_grade_gap": 0.5,
        "max_teacher_call_rate": 0.15,
    }), encoding="utf-8")
    monkeypatch.setenv("RAG_ANSWERABILITY_BALANCED_POLICY", str(policy))
    stats = {}
    rerank_by_student(
        "q", ["a", "b"], [{}, {}], [0.1, 0.2], [0.9, 0.8], balanced=True,
        predictor=lambda _q, _d: ([
            {"grade": 3, "confidence": 0.7, "expected_grade": 2.5},
            {"grade": 2, "confidence": 0.9, "expected_grade": 2.2},
        ], {"model_sha256": "a" * 64}), stats=stats,
    )
    assert stats["fallback_recommended"] is True
    assert stats["balanced_policy_reason"] == "uncertain"


def test_balanced_student_failure_cannot_trigger_teacher(monkeypatch):
    import rag_answerability as answerability
    import rag_answerability_student as student

    monkeypatch.setenv("RAG_ANSWERABILITY_MODE", "balanced")
    monkeypatch.setattr(
        student, "rerank_by_student",
        lambda q, d, m, ds, s, **kwargs: (d, m, ds, s),
    )
    monkeypatch.setattr(
        answerability, "rerank_by_answerability",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("teacher called")),
    )
    docs = ["a"]
    assert answerability.rerank_by_mode(
        "q", docs, [{}], [0.1], [0.9], stats={},
    )[0] == docs


def test_only_teacher_grades_can_drive_the_existing_answerability_gate():
    source = Path(__file__).resolve().parents[1].joinpath("rag_gate.py").read_text(
        encoding="utf-8",
    )
    block = source.split("_ans_accept = False", 1)[1].split("_rr_gate_raw", 1)[0]
    assert 'get("source") == "teacher"' in block


def test_frozen_training_configuration_matches_the_approved_plan():
    import train_answerability_student as trainer

    assert trainer.SEEDS == (17, 29, 43)
    assert trainer.MAX_LENGTH == 384
    assert trainer.GRADIENT_ACCUMULATION == 16
    assert trainer.LEARNING_RATE == 2e-5
    assert trainer.MAX_EPOCHS == 4
    assert trainer.PAIRWISE_WEIGHT == 0.5


def test_blind_evaluator_rejects_development_rows_before_loading_model(tmp_path):
    from scripts.evaluate_answerability_student_blind import evaluate
    from rag_answerability_student_data import StudentDataError

    path = tmp_path / "not_blind.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(StudentDataError, match="blind_a/blind_b"):
        evaluate(tmp_path / "missing-model", path)
