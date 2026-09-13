from __future__ import annotations

import json


def _signature(**overrides):
    from rag_gate_profiles import GateSignature

    values = {
        "embedding_model": "embed-v1",
        "reranker_model": "rerank-v2",
        "chunker_version": "chunk-v3",
        "retrieval_profile": "A1",
        "index_fingerprint": "index-123",
    }
    values.update(overrides)
    return GateSignature(**values)


def _features(*, good: bool):
    from rag_gate_profiles import GateFeatures

    if good:
        return GateFeatures(0.30, 1, 0.97, 0.40, False, True)
    return GateFeatures(0.91, None, 0.20, 0.01, False, False)


def test_trace_does_not_rebind_gate_features_to_final_top1():
    from rag_gate_profiles import gate_features_from_trace

    features = gate_features_from_trace({
        "gate_features": {
            "best_dense_distance": 0.4, "rerank_top": 0.9,
            "rerank_margin": 0.2, "vector_in_kb": True,
        },
        "final_candidates": [{"chunk_id": "b", "rank": 1}],
        "bm25_candidates": [
            {"chunk_id": "a", "rank": 1}, {"chunk_id": "b", "rank": 2},
        ],
    })
    assert features.bm25_best_rank is None
    assert features.best_dense_distance == 0.4


def test_rule_requires_reranker_and_independent_support():
    from rag_gate_profiles import GateFeatures, GateRule

    rule = GateRule(
        rerank_min=0.8,
        support_signals=("dense_distance", "rerank_margin"),
        support_at_least=2,
        dense_max=0.5,
        margin_min=0.1,
    )
    assert rule.decide(GateFeatures(0.4, None, 0.9, 0.2)) is True
    assert rule.decide(GateFeatures(0.7, None, 0.9, 0.2)) is False
    assert rule.decide(GateFeatures(0.4, None, 0.7, 0.2)) is False
    try:
        GateRule(0.5, ())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("uncorroborated score-only gate profile was accepted")


def test_unknown_or_changed_signature_fails_safe_to_baseline():
    from rag_gate_profiles import (
        GATE_PROFILE_SCHEMA, GATE_REGISTRY_SCHEMA, GateProfile, GateRule,
        decide_with_profile,
    )

    signature = _signature()
    profile = GateProfile(
        profile_id="winner", signature=signature,
        rule=GateRule(0.8, ("vector_in_kb",)),
        calibrated_at="2026-08-24", calibration_source="test",
    )
    registry = {
        "schema_version": GATE_REGISTRY_SCHEMA,
        "profiles": [profile.to_dict()],
    }
    decision, resolution = decide_with_profile(
        _features(good=True), signature, lambda _features: False, registry,
    )
    assert decision is True and resolution.mode == "calibrated"

    changed = _signature(reranker_model="different")
    decision, resolution = decide_with_profile(
        _features(good=True), changed, lambda _features: False, registry,
    )
    assert decision is False
    assert resolution.mode == "baseline"
    assert resolution.reason == "unknown_runtime_signature"
    assert profile.schema_version == GATE_PROFILE_SCHEMA


def test_malformed_registry_fails_safe_to_baseline(tmp_path):
    from rag_gate_profiles import resolve_gate_profile

    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{not-json", encoding="utf-8")
    resolution = resolve_gate_profile(_signature(), broken_json)
    assert resolution.mode == "baseline"
    assert resolution.reason == "malformed_gate_registry"

    bad_profile = {
        "schema_version": "offerclaw-gate-profile-registry-v1",
        "profiles": [{"profile_id": "incomplete"}],
    }
    resolution = resolve_gate_profile(_signature(), bad_profile)
    assert resolution.mode == "baseline"
    assert resolution.reason == "malformed_gate_profile"


def test_gate_signature_is_independent_of_audit_commit():
    from rag_gate_profiles import GateSignature
    from rag_retrieval_trace import RetrievalProfile, bind_profile_fingerprint

    profile = RetrievalProfile(
        name="candidate", reranker_model="rerank-v2", chunker_version="chunk-v3",
    )
    base = {
        "collection": "kb", "collection_count": 2,
        "collection_content_hash": "content-123",
        "embedding_model": "embed-v1",
    }
    first = bind_profile_fingerprint(
        {**base, "git_commit": "commit-a", "audit_git_commit": "commit-a"}, profile,
    )
    second = bind_profile_fingerprint(
        {**base, "git_commit": "commit-b", "audit_git_commit": "commit-b"}, profile,
    )

    assert first["fingerprint_id"] == second["fingerprint_id"]
    first_signature = GateSignature.from_trace({
        "index_metadata": first,
        "retrieval_profile": profile.to_dict(),
        "index_fingerprint": first["fingerprint_id"],
    })
    second_signature = GateSignature.from_trace({
        "index_metadata": second,
        "retrieval_profile": profile.to_dict(),
        "index_fingerprint": second["fingerprint_id"],
    })
    assert first_signature == second_signature


def test_empty_checked_in_registry_has_no_fake_thresholds():
    from rag_gate_profiles import DEFAULT_GATE_PROFILE_PATH, load_gate_registry

    raw = DEFAULT_GATE_PROFILE_PATH.read_text(encoding="utf-8")
    registry = load_gate_registry()
    assert registry["profiles"] == []
    assert "rerank_min" not in raw
    assert "dense_max" not in raw


def _calibration_examples(heldout_correct_count=27):
    from rag_gate_profiles import CalibrationExample

    def aligned(example_id, group, features, *, top1_correct=False):
        return CalibrationExample(
            example_id, group, features, top1_correct=top1_correct,
            gate_candidate_chunk_id=f"chunk-{example_id}",
            final_top1_chunk_id=f"chunk-{example_id}",
            gate_alignment=True,
        )

    rows = [
        aligned(f"heldout-{i}", "heldout", _features(good=True),
                top1_correct=True)
        for i in range(heldout_correct_count)
    ]
    rows += [
        aligned(f"heldout-wrong-{i}", "heldout", _features(good=False))
        for i in range(52 - heldout_correct_count)
    ]
    rows += [
        aligned(f"positive-{i}", "positive", _features(good=True))
        for i in range(12)
    ]
    rows += [
        aligned(f"simple-{i}", "simple_negative", _features(good=False))
        for i in range(12)
    ]
    rows += [
        aligned(f"adv-{i}", "adversarial_negative", _features(good=False))
        for i in range(11)
    ]
    # One allowed adversarial false positive.
    rows.append(aligned("adv-allowed", "adversarial_negative", _features(good=True)))
    return rows


def test_calibrator_selects_only_multi_feature_rules_when_all_gates_pass():
    from rag_gate_profiles import calibrate_gate

    report = calibrate_gate(_calibration_examples(), _signature(), source="synthetic")
    assert report["status"] == "calibration_candidate"
    assert report["release_eligible"] is True
    assert report["deployable"] is False
    assert report["selected_profile"] is not None
    assert report["misaligned_count"] == 0
    assert report["candidate_count"] > 0
    assert all(candidate["rule"]["support_signals"] for candidate in report["candidates"])
    selected = next(
        candidate for candidate in report["candidates"]
        if candidate["candidate_id"] == report["selected_candidate_id"]
    )
    assert selected["metrics"]["heldout_correct_top1_pass"] >= 27
    assert selected["metrics"]["positive_pass"] == 12
    assert selected["metrics"]["simple_negative_reject"] == 12
    assert selected["metrics"]["adversarial_negative_reject"] >= 11


def test_calibrator_refuses_profile_when_retrieval_cannot_supply_27_correct_top1():
    from rag_gate_profiles import calibrate_gate

    report = calibrate_gate(_calibration_examples(heldout_correct_count=26), _signature())
    assert report["release_eligible"] is False
    assert report["deployable"] is False
    assert report["selected_profile"] is None
    assert report["reason"] == "no_rule_meets_all_constraints"


def test_ab_rows_and_gate_rows_are_loaded_without_retrieval_rerun():
    from rag_gate_profiles import examples_from_ab_payload

    payload = {
        "arms": {
            "A1": {
                "profile": {
                    "name": "A1", "reranker_model": "rerank-v2",
                    "chunker_version": "chunk-v3",
                },
                "index": {
                    "embedding_model": "embed-v1", "fingerprint_id": "index-123",
                },
                "sets": [{"set": "heldout52", "runs": [{"rows": [{
                    "id": "q1", "rank": 1, "rerank_top": 0.9,
                    "rerank_margin": 0.2,
                    "gate_candidate_chunk_id": "chunk-q1",
                    "final_top1_chunk_id": "chunk-q1",
                    "gate_alignment": True,
                }]}]}],
                "gates": {"positive": {"rows": [{
                    "id": "p1", "expected_in_kb": True,
                    "gate_features": {"rerank_top": 0.9, "rerank_margin": 0.2},
                    "gate_candidate_chunk_id": "chunk-p1",
                    "final_top1_chunk_id": "chunk-p1",
                    "gate_alignment": True,
                }]}},
            }
        }
    }
    examples, signature = examples_from_ab_payload(payload, arm="A1")
    assert [(row.example_id, row.group) for row in examples] == [
        ("q1", "heldout"), ("p1", "positive"),
    ]
    assert examples[0].top1_correct is True
    assert all(example.alignment_valid for example in examples)
    assert signature.complete is True
    assert signature.reranker_model == "rerank-v2"


def test_misaligned_examples_are_excluded_and_block_release():
    from dataclasses import replace
    from rag_gate_profiles import calibrate_gate

    examples = _calibration_examples()
    examples[0] = replace(
        examples[0], final_top1_chunk_id="different-final", gate_alignment=False,
    )
    report = calibrate_gate(examples, _signature(), source="misaligned-test")

    assert report["status"] == "calibration_candidate"
    assert report["deployable"] is False
    assert report["release_eligible"] is False
    assert report["misaligned_count"] == 1
    assert report["misaligned_ids"] == [examples[0].example_id]
    assert report["fitting_example_counts"]["heldout"] == 51
    assert report["reason"] == "misaligned_gate_candidates"


def test_missing_alignment_fields_are_not_treated_as_legacy_success():
    from rag_gate_profiles import examples_from_ab_payload

    payload = {
        "profile": {
            "name": "A1", "reranker_model": "rerank-v2",
            "chunker_version": "chunk-v3",
        },
        "index": {"embedding_model": "embed-v1", "fingerprint_id": "index-123"},
        "sets": [{"set": "heldout52", "runs": [{"rows": [{
            "id": "legacy", "rank": 1,
            "gate_features": {"rerank_top": 0.9, "vector_in_kb": True},
        }]}]}],
        "gates": {},
    }
    examples, _signature_value = examples_from_ab_payload(payload)
    assert len(examples) == 1
    assert examples[0].gate_alignment is None
    assert examples[0].alignment_valid is False


def test_cli_disables_direct_registry_promotion(tmp_path, monkeypatch):
    import sys
    import eval_gate_profiles

    input_path = tmp_path / "ab.json"
    output_path = tmp_path / "report.json"
    registry_path = tmp_path / "registry.json"
    input_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "eval_gate_profiles.py", "--input", str(input_path),
        "--output", str(output_path), "--profile-output", str(registry_path),
    ])
    try:
        eval_gate_profiles.main()
    except SystemExit as exc:
        assert "Direct Gate promotion is disabled" in str(exc)
    else:
        raise AssertionError("calibration CLI wrote a production registry")
    assert not registry_path.exists()


def test_gate_profile_digest_tampering_is_rejected():
    from rag_gate_profiles import GateProfile, GateRule

    profile = GateProfile(
        profile_id="winner", signature=_signature(),
        rule=GateRule(0.8, ("vector_in_kb",)),
        calibrated_at="now", calibration_source="test",
    ).to_dict()
    profile["signature_digest"] = "tampered"
    try:
        GateProfile.from_dict(profile)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered signature was accepted")
