# -*- coding: utf-8 -*-
"""Stage 1 contract: the model observes, the program decides.

Pinned properties mirror the measured failures this mechanism targets: a
partial-coverage chunk must lose to a fuller one (7 stable same-source losses),
the early exit must demand full self-sufficient coverage (6 of those 7 were
early-exit amplified), and a numeric question without its value must never
count as fully covered (Final v4's one stable true false-accept, x3 runs).
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_requirement_coverage as rc  # noqa: E402


CONTRACT = {
    "requirements": [
        {"id": "r1", "description": "步骤清单", "required": True},
        {"id": "r2", "description": "触发条件", "required": True},
        {"id": "r3", "description": "背景(可选)", "required": False},
    ],
    "fact_type": "normal",
}
NUMERIC = {
    "requirements": [{"id": "r1", "description": "支持的最大字符数", "required": True}],
    "fact_type": "numeric",
    "required_slots": ["value"],
}


def _obs(ids, self_contained=True, unresolved=False, slots=None):
    return {"covered_requirement_ids": ids, "explicit_slot_values": slots or {},
            "self_contained": self_contained, "unresolved_reference": unresolved}


def test_fuller_coverage_outranks_partial_regardless_of_reranker_score():
    partial = rc.rank_key(_obs(["r1"]), CONTRACT, rerank_score=0.99)
    fuller = rc.rank_key(_obs(["r1", "r2"]), CONTRACT, rerank_score=0.10)
    assert fuller > partial


def test_reranker_still_breaks_true_ties():
    a = rc.rank_key(_obs(["r1", "r2"]), CONTRACT, 0.9)
    b = rc.rank_key(_obs(["r1", "r2"]), CONTRACT, 0.4)
    assert a > b


def test_optional_requirements_do_not_gate_full_coverage():
    assert rc.full_coverage(_obs(["r1", "r2"]), CONTRACT)      # r3 optional
    assert not rc.full_coverage(_obs(["r1", "r3"]), CONTRACT)  # r2 missing


def test_early_exit_needs_full_selfcontained_unambiguous_coverage():
    assert rc.can_early_exit(_obs(["r1", "r2"]), CONTRACT)
    # any one condition failing keeps judging the rest of the head
    assert not rc.can_early_exit(_obs(["r1"]), CONTRACT)
    assert not rc.can_early_exit(_obs(["r1", "r2"], self_contained=False), CONTRACT)
    assert not rc.can_early_exit(_obs(["r1", "r2"], unresolved=True), CONTRACT)
    assert not rc.can_early_exit(None, CONTRACT)


def test_numeric_full_coverage_requires_the_actual_value():
    """Topic-only discussion of a numeric question is not an answer."""
    assert not rc.full_coverage(_obs(["r1"]), NUMERIC)
    assert not rc.full_coverage(_obs(["r1"], slots={"value": "  "}), NUMERIC)
    assert rc.full_coverage(_obs(["r1"], slots={"value": "32,000"}), NUMERIC)


def test_minimal_evidence_set_prefers_one_chunk_then_two_never_pads():
    observations = [_obs(["r1"]), _obs(["r2"]), _obs(["r1", "r2"]), _obs([])]
    assert rc.minimal_evidence_set(observations, CONTRACT) == [2]
    observations = [_obs(["r1"]), _obs(["r2"]), _obs([])]
    assert rc.minimal_evidence_set(observations, CONTRACT) == [0, 1]
    observations = [_obs(["r1"]), _obs([]), _obs([])]
    assert rc.minimal_evidence_set(observations, CONTRACT) == []


def test_unusable_observation_is_unavailable_not_zero_coverage():
    for text in (None, "", "说不好", '{"covered_requirement_ids": "r1"}',
                 '{"covered_requirement_ids": ["r9"], "explicit_slot_values": {}, '
                 '"self_contained": true, "unresolved_reference": false}',
                 '{"covered_requirement_ids": ["r1"], "explicit_slot_values": [], '
                 '"self_contained": true, "unresolved_reference": false}'):
        assert rc.parse_observation(text, CONTRACT) is None


def test_parse_accepts_the_documented_shape():
    text = ('{"covered_requirement_ids": ["r2", "r1", "r1"], '
            '"explicit_slot_values": {"value": 32000}, '
            '"self_contained": true, "unresolved_reference": false}')
    obs = rc.parse_observation(text, CONTRACT)
    assert obs["covered_requirement_ids"] == ["r1", "r2"]      # deduped, sorted
    assert obs["explicit_slot_values"] == {"value": "32000"}   # stringified


def test_contract_validation_rejects_broken_shapes():
    with pytest.raises(ValueError):
        rc.validate_contract({"requirements": []})
    with pytest.raises(ValueError):
        rc.validate_contract({"requirements": [{"id": "r1", "description": "x"},
                                               {"id": "r1", "description": "y"}]})
    with pytest.raises(ValueError):
        rc.validate_contract({"requirements": [{"id": "r1", "description": "x"}],
                              "fact_type": "numeric"})       # numeric without slots


def test_the_prompt_never_sees_gold_material():
    """Requirements describe the question; the prompt must carry no field for
    gold text, answer_requirements or chunk ids beyond the one chunk shown."""
    assert "answer_requirements" not in rc._PROMPT
    assert "gold" not in rc._PROMPT.lower()
    assert "不做任何" in rc._PROMPT and "观察" in rc._PROMPT
