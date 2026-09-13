# -*- coding: utf-8 -*-
"""Tests for the F3 package builder and the mined-negative adjudication rules."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_f3_training_package import (  # noqa: E402
    TOO_EASY_PAIR_SHARE,
    assign_anchor_weights,
    difficulty_report,
)
from scripts.adjudicate_mined_negatives import classify  # noqa: E402
from train_reranker_hard_negatives import (  # noqa: E402
    TrainingInputError,
    _pairwise_kind_weight,
)


def _triple(anchor, query_id, margin=0.3, split="train", wave="v2a",
            origin="authored", kind="same_document_wrong_section", style="standard"):
    return {
        "anchor_id": anchor,
        "query_id": query_id,
        "split": split,
        "source_wave": wave,
        "negative_origin": origin,
        "negative_kind": kind,
        "query_style": style,
        "domain": "backend",
        "base_scores": {"positive": 0.9, "negative": 0.9 - margin, "margin": margin},
    }


def test_anchor_weights_equalize_total_weight_per_anchor():
    # a1 carries twice as many triples as a2 and must not outvote it
    triples = [
        *[_triple("a1", f"q{i}") for i in range(8)],
        *[_triple("a2", f"q{i}") for i in range(8, 12)],
    ]
    stats = assign_anchor_weights(triples)
    totals = {}
    for row in triples:
        totals.setdefault(row["anchor_id"], 0.0)
        totals[row["anchor_id"]] += row["pair_weight"]
    assert round(totals["a1"], 6) == round(totals["a2"], 6)
    assert stats["train"]["total_weight_per_anchor_is_equal"] is True
    # mean weight stays 1.0 so reported losses remain comparable with F1/F2
    assert stats["train"]["pair_weight"]["mean"] == pytest.approx(1.0, abs=1e-6)


def test_anchor_weights_are_scoped_per_split():
    triples = [
        *[_triple("a1", f"q{i}") for i in range(4)],
        *[_triple("d1", f"d{i}", split="dev") for i in range(8)],
    ]
    assign_anchor_weights(triples)
    train = {row["pair_weight"] for row in triples if row["split"] == "train"}
    dev = {row["pair_weight"] for row in triples if row["split"] == "dev"}
    # a single anchor per split means every weight is 1.0 within that split
    assert train == {1.0} and dev == {1.0}


def test_pair_weight_multiplies_the_kind_weight_and_defaults_to_it():
    weights = {"same_document_wrong_section": 1.2}
    plain = {"negative_kind": "same_document_wrong_section"}
    scaled = {**plain, "pair_weight": 0.5}
    assert _pairwise_kind_weight(plain, weights) == pytest.approx(1.2)
    assert _pairwise_kind_weight(scaled, weights) == pytest.approx(0.6)
    with pytest.raises(TrainingInputError, match="pair_weight"):
        _pairwise_kind_weight({**plain, "pair_weight": 0}, weights)


def test_difficulty_gate_fails_when_the_corpus_is_too_easy():
    easy = [_triple("a1", f"q{i}", margin=0.8) for i in range(20)]
    report = difficulty_report(easy)
    assert report["verdict"]["gate"] == "fail_too_easy"
    assert report["train"]["base_already_separated_share"] == 1.0

    mixed = ([_triple("a1", f"q{i}", margin=0.8) for i in range(8)]
             + [_triple("a2", f"p{i}", margin=-0.1) for i in range(2)])
    report = difficulty_report(mixed)
    assert report["train"]["base_inverted_pairs"] == 2
    assert report["train"]["base_already_separated_share"] == 0.8
    assert report["train"]["base_already_separated_share"] < TOO_EASY_PAIR_SHARE
    assert report["verdict"]["gate"] == "pass"


def test_difficulty_report_splits_by_negative_origin():
    triples = [
        _triple("a1", "q1", margin=0.9, origin="authored"),
        _triple("a2", "q2", margin=-0.2, origin="mined_production_false_winner"),
    ]
    report = difficulty_report(triples)
    assert report["train"]["by_negative_origin"] == {
        "authored": 1, "mined_production_false_winner": 1,
    }


@pytest.mark.parametrize("verdict,answerable,supported,covered,expected", [
    ("does_not_cover", False, False, 0, "hard_negative"),
    ("fully_covers", True, True, 3, "equivalent_grade3_candidate"),
    ("partially_covers", False, False, 3, "grade2_partial"),
    ("partially_covers", False, False, 1, "grade1_partial"),
    # a reviewer that says "answerable" must never yield a negative
    ("does_not_cover", True, True, 0, "grade1_partial"),
])
def test_mined_candidate_classification(verdict, answerable, supported, covered,
                                        expected):
    review = {
        "verification": {"directly_answerable": answerable,
                         "requirement_fully_supported": supported, "reason": "r"},
        "coverage": {"verdict": verdict,
                     "covered_points": ["c"] * covered, "missing_points": ["m"]},
    }
    assert classify(review, "same_document_wrong_section")["status"] == expected


def test_frozen_f3_package_matches_its_difficulty_report():
    root = Path(__file__).resolve().parents[1]
    package = root / ".offerclaw/reranker_training/f3_merged_v1_v2a.json"
    report = root / "docs/rag_eval/colloquial/F3_DATA_DIFFICULTY_20260826.json"
    if not package.exists() or not report.exists():
        pytest.skip("F3 package unavailable")
    artifact = json.loads(package.read_text(encoding="utf-8"))
    published = json.loads(report.read_text(encoding="utf-8"))
    assert published["canonical_triples_sha256"] == artifact["canonical_triples_sha256"]
    assert published["summary"] == artifact["summary"]
    assert artifact["provenance"]["blind_set_used"] is False
    assert artifact["provenance"]["sealed_set_used"] is False
    train = {row["anchor_id"] for row in artifact["triples"]
             if row["split"] == "train"}
    dev = {row["anchor_id"] for row in artifact["triples"] if row["split"] == "dev"}
    assert not train & dev
    assert published["difficulty"]["verdict"]["gate"] == "pass"


def test_default_weighting_is_byte_identical_to_the_frozen_package():
    """The F3 corpus must stay reproducible while a new mode exists.

    A new weighting option that quietly perturbs the default would make every
    published F3 number unverifiable.
    """
    triples = [
        *[_triple("a1", f"q{i}") for i in range(8)],
        *[_triple("a2", f"p{i}") for i in range(4)],
    ]
    plain = [dict(row) for row in triples]
    assign_anchor_weights(plain)
    assert [row["pair_weight"] for row in plain] == [0.75] * 8 + [1.5] * 4


def test_corrective_weighting_moves_weight_onto_the_pairs_the_base_gets_wrong():
    from build_f3_training_package import difficulty_factor

    # a1 is a big, easy anchor; a2 is small and carries the inverted pairs
    triples = [
        *[_triple("a1", f"q{i}", margin=0.8) for i in range(8)],
        *[_triple("a2", f"p{i}", margin=-0.4) for i in range(4)],
    ]
    stats = assign_anchor_weights(triples, mode="corrective")["train"]
    easy = [row["pair_weight"] for row in triples if row["anchor_id"] == "a1"]
    hard = [row["pair_weight"] for row in triples if row["anchor_id"] == "a2"]
    assert min(hard) > max(easy)
    # the mean stays 1.0 so the reported loss remains on the F1/F2 scale
    assert stats["pair_weight"]["mean"] == pytest.approx(1.0, abs=1e-4)
    # anchor equality is deliberately given up, and reported rather than hidden
    assert stats["total_weight_per_anchor_is_equal"] is False
    assert stats["mode"] == "corrective"
    assert stats["inverted_pair_weight_share"] > stats["inverted_pair_row_share"]
    # monotone in the base margin, with a floor so easy pairs are not erased
    assert difficulty_factor(-1.0) > difficulty_factor(0.0) > difficulty_factor(1.0)
    assert difficulty_factor(10.0) > 0.0


def test_corrective_weighting_refuses_to_run_before_the_pairs_are_scored():
    from build_f3_training_package import ColloquialTrainingBuildError

    triples = [_triple("a1", "q1")]
    del triples[0]["base_scores"]
    with pytest.raises(ColloquialTrainingBuildError, match="base_scores"):
        assign_anchor_weights(triples, mode="corrective")


def test_corrective_weighting_gives_each_style_its_row_share():
    """Anchor normalization must not hand a small style a weight windfall.

    ``rare`` has a quarter of the rows of ``common`` but each of its anchors
    carries fewer triples, so plain anchor normalization inflates it.  After
    balancing, weight share tracks row share, while difficulty still orders
    pairs inside each style.
    """
    triples = [
        *[_triple("a1", f"c{i}", margin=0.5, style="common") for i in range(8)],
        *[_triple("a2", f"r{i}", margin=0.5, style="rare") for i in range(2)],
    ]
    plain = [dict(row) for row in triples]
    assign_anchor_weights(plain)
    plain_rare = sum(r["pair_weight"] for r in plain if r["query_style"] == "rare")
    plain_total = sum(r["pair_weight"] for r in plain)
    assert plain_rare / plain_total == pytest.approx(0.5)      # 20% of rows, 50% of weight

    stats = assign_anchor_weights(triples, mode="corrective")["train"]
    shares = stats["weight_share_by_query_style"]
    for style, share in shares.items():
        assert share["weight_share"] == pytest.approx(share["row_share"], abs=1e-6), style


def test_style_balancing_preserves_difficulty_order_within_a_style():
    triples = [
        _triple("a1", "easy", margin=0.9, style="natural"),
        _triple("a1", "hard", margin=-0.5, style="natural"),
        _triple("a2", "other", margin=0.2, style="standard"),
    ]
    assign_anchor_weights(triples, mode="corrective")
    weights = {row["query_id"]: row["pair_weight"] for row in triples}
    assert weights["hard"] > weights["easy"]
