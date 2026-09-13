# -*- coding: utf-8 -*-

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_rule_matches_uses_frozen_structured_conditions():
    from scripts.evaluate_answerability_fast_validation import rule_matches

    features = {"rerank_margin": 0.01, "high_scorers": 4}
    assert rule_matches(features, [
        {"feature": "rerank_margin", "operator": "gte", "value": 0.01},
        {"feature": "high_scorers", "operator": "lte", "value": 4},
    ])
    assert not rule_matches(features, [
        {"feature": "rerank_margin", "operator": "gt", "value": 0.01},
    ])


def test_wilson_handles_empty_and_bounds():
    from scripts.evaluate_answerability_fast_validation import wilson

    assert wilson(0, 0) is None
    low, high = wilson(7, 10)
    assert 0 <= low < 0.7 < high <= 1
