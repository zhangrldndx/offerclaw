# -*- coding: utf-8 -*-
"""Tests for the query-representation audit and the query-adapter probe."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_colloquial_query_representation import (  # noqa: E402
    by_style,
    cosine_from_distance,
    paired_vs_standard,
)
from scripts.probe_query_adapter import (  # noqa: E402
    _normalize,
    apply_adapter,
    fit_ridge,
    rank_metrics,
)


def test_cosine_conversion_matches_the_collection_l2_convention():
    # Chroma's default l2 space reports squared euclidean distance, so an
    # exact match is 0 and orthogonal unit vectors are 2.  Reading it as a
    # plain euclidean distance would report 1.0 similarity for orthogonal
    # vectors and silently flatten every colloquial penalty measured here.
    assert cosine_from_distance(0.0) == pytest.approx(1.0)
    assert cosine_from_distance(2.0) == pytest.approx(0.0)
    assert cosine_from_distance(4.0) == pytest.approx(-1.0)
    a = _normalize(np.array([[1.0, 2.0, 3.0]]))[0]
    b = _normalize(np.array([[3.0, -1.0, 0.5]]))[0]
    squared_l2 = float(((a - b) ** 2).sum())
    assert cosine_from_distance(squared_l2) == pytest.approx(float(a @ b))


def _row(anchor, style, rank, gold, other, overlap=0.5, chars=20):
    return {"anchor_id": anchor, "query_id": f"{anchor}-{style}", "style": style,
            "question_chars": chars, "dense_rank": rank, "gold_similarity": gold,
            "best_other_similarity": other,
            "margin": None if gold is None or other is None else gold - other,
            "query_tokens": 6, "lexical_overlap": overlap}


def test_paired_delta_is_computed_within_anchor_not_across_styles():
    """A style mean can be dragged by which anchors carry that style.

    Here ``natural`` is uniformly one rank worse than ``standard`` on both
    anchors, but anchor a2 is much harder overall.  A cross-anchor comparison
    of raw ranks would report the anchor difficulty; the paired delta must
    report the style effect.
    """
    rows = [
        _row("a1", "standard", 1, 0.80, 0.70),
        _row("a1", "natural", 2, 0.75, 0.76),
        _row("a2", "standard", 40, 0.50, 0.60),
        _row("a2", "natural", 41, 0.45, 0.66),
    ]
    paired = paired_vs_standard(rows)["natural"]
    assert paired["pairs"] == 2
    assert paired["dense_rank_worse_than_standard"] == 2
    assert paired["dense_rank_better_than_standard"] == 0
    assert paired["delta_gold_similarity"]["mean"] == pytest.approx(-0.05)
    assert paired["delta_best_other_similarity"]["mean"] == pytest.approx(0.06)


def test_uniform_contraction_moves_gold_similarity_but_not_margin():
    """The decomposition has to separate "vaguer" from "wrong".

    Both similarities drop by the same amount, so ranking is untouched.  A
    report that only tracked cos(q, gold) would call this a regression.
    """
    rows = [
        _row("a1", "standard", 3, 0.80, 0.70),
        _row("a1", "implicit_oral", 3, 0.60, 0.50),
    ]
    paired = paired_vs_standard(rows)["implicit_oral"]
    assert paired["delta_gold_similarity"]["mean"] == pytest.approx(-0.20)
    assert paired["delta_margin"]["mean"] == pytest.approx(0.0)
    assert paired["dense_rank_tied"] == 1


def test_gold_beyond_probe_depth_is_reported_not_silently_dropped():
    rows = [_row("a1", "natural", None, None, 0.7), _row("a2", "natural", 5, 0.8, 0.7)]
    stats = by_style(rows)["natural"]
    assert stats["n"] == 2
    assert stats["gold_beyond_probe_depth"] == 1
    # similarity stats are computed over the rows that actually have a gold
    assert stats["gold_similarity"]["n"] == 1
    # ... but the distractor similarity exists for every row
    assert stats["best_other_similarity"]["n"] == 2


def test_missing_style_pair_is_skipped_rather_than_compared_to_nothing():
    rows = [_row("a1", "standard", 1, 0.8, 0.7), _row("a2", "natural", 4, 0.6, 0.7)]
    assert paired_vs_standard(rows)["natural"]["pairs"] == 0


def test_rank_metrics_counts_ties_against_the_gold():
    """Two chunks with identical scores must not both be credited R@1.

    The rank is "how many strictly outscore the gold, plus one", so an exact
    tie leaves the gold at rank 1 -- matching Chroma, which will return one of
    them first.  Using ``>=`` here would under-report R@1 on duplicate chunks.
    """
    corpus = _normalize(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    queries = _normalize(np.array([[1.0, 0.0]]))
    out = rank_metrics(queries, corpus, [{1}], ["standard"])
    assert out["ranks"] == [1]
    assert out["metrics"]["overall"]["r1"] == 1


def test_multi_gold_row_is_scored_by_its_best_placed_gold():
    corpus = _normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    queries = _normalize(np.array([[1.0, 0.05]]))
    # gold {1} alone sits behind two better-scoring chunks; adding gold 0 --
    # which is the top scorer -- must turn the row into an R@1 hit rather than
    # averaging the two gold positions
    assert rank_metrics(queries, corpus, [{1}], ["s"])["ranks"] == [3]
    assert rank_metrics(queries, corpus, [{0, 1}], ["s"])["ranks"] == [1]


def test_ridge_recovers_a_known_map_and_regularization_shrinks_it():
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(400, 6))
    truth = rng.normal(size=(6, 6))
    targets = queries @ truth
    weak = fit_ridge(queries, targets, 1e-6)
    strong = fit_ridge(queries, targets, 1e6)
    assert np.allclose(weak, truth, atol=1e-3)
    assert np.linalg.norm(strong) < np.linalg.norm(weak)


def test_alpha_interpolates_between_identity_and_full_adapter():
    rng = np.random.default_rng(11)
    queries = _normalize(rng.normal(size=(5, 4)))
    matrix = rng.normal(size=(4, 4))
    assert np.allclose(apply_adapter(queries, matrix, 0.0), queries)
    full = apply_adapter(queries, matrix, 1.0)
    assert np.allclose(full, _normalize(queries @ matrix))
    half = apply_adapter(queries, matrix, 0.5)
    # every output stays on the unit sphere, so cosine scoring stays valid
    assert np.allclose(np.linalg.norm(half, axis=1), 1.0)
    assert not np.allclose(half, queries)


def test_displacement_coherence_separates_a_shared_shift_from_content_drift():
    """The number has to distinguish "one register shift" from "per-topic".

    A single matrix can only apply a direction-consistent transform, so this
    diagnostic is what turns "the linear adapter failed" into "no linear
    adapter can succeed".
    """
    from scripts.probe_query_adapter import displacement_coherence

    dim = 8
    rng = np.random.default_rng(3)
    shift = _normalize(rng.normal(size=(1, dim)))[0] * 0.3

    # (a) every anchor needs the same correction
    bases = _normalize(rng.normal(size=(6, dim)))
    vectors, anchors, styles = [], [], []
    for index, base in enumerate(bases):
        vectors += [base, base + shift]
        anchors += [f"a{index}", f"a{index}"]
        styles += ["natural", "standard"]
    shared = displacement_coherence(np.asarray(vectors), anchors, styles)["natural"]
    assert shared["pairs"] == 6
    assert shared["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-9)
    assert shared["shared_component_fraction"] == pytest.approx(1.0, abs=1e-9)

    # (b) every anchor needs a different, unrelated correction
    vectors, anchors, styles = [], [], []
    for index in range(dim):
        base = np.zeros(dim)
        target = np.zeros(dim)
        target[index] = 1.0
        vectors += [base, target]
        anchors += [f"b{index}", f"b{index}"]
        styles += ["natural", "standard"]
    drifting = displacement_coherence(np.asarray(vectors), anchors, styles)["natural"]
    assert drifting["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert drifting["shared_component_fraction"] < 0.4


def test_displacement_coherence_skips_styles_without_a_paired_reference():
    from scripts.probe_query_adapter import displacement_coherence

    vectors = _normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    out = displacement_coherence(vectors, ["a1", "a1", "a2"],
                                 ["natural", "standard", "natural"])
    # a2 has no standard variant, so only a1 contributes -- and one pair is
    # not enough to report a pairwise cosine at all
    assert "natural" not in out


def test_anchor_level_recall_does_not_treat_styles_as_independent():
    """Four phrasings of one anchor share a gold chunk and a neighbourhood.

    Counting them as four observations inflates every effect size by roughly
    the number of styles.  Measured on 2026-08-27: an evidence-gate result
    reported as "+7 questions" was four distinct anchors, and a tie-break arm
    that looked like it broke seven Dev80 questions had broken two.
    """
    from rag_eval_metrics import anchor_level_recall

    rows = [
        {"anchor_id": "a1", "final_rank": 1, "effective_evidence": True},
        {"anchor_id": "a1", "final_rank": 1, "effective_evidence": True},
        {"anchor_id": "a1", "final_rank": 1, "effective_evidence": True},
        {"anchor_id": "a1", "final_rank": 7, "effective_evidence": False},
    ]
    stats = anchor_level_recall(rows)
    # four "wins" at query level are one anchor, and not even a clean one
    assert stats["anchors"] == 1
    assert stats["recall@1"] == {"any": 1, "all": 0, "n": 1}
    assert stats["effective_evidence"] == {"any": 1, "all": 0, "n": 1}


def test_anchor_level_all_requires_every_phrasing_to_succeed():
    from rag_eval_metrics import anchor_level_recall

    rows = [
        {"anchor_id": "solid", "final_rank": 1},
        {"anchor_id": "solid", "final_rank": 2},
        {"anchor_id": "fragile", "final_rank": 1},
        {"anchor_id": "fragile", "final_rank": 40},
    ]
    stats = anchor_level_recall(rows)
    # "all" at rank 3 separates the anchor that survives rephrasing from the
    # one that only works when asked the right way -- which is the whole point
    # of the colloquial sets
    assert stats["recall@3"] == {"any": 2, "all": 1, "n": 2}
    assert stats["recall@1"] == {"any": 2, "all": 0, "n": 2}


def test_sentence_windows_always_include_the_whole_chunk():
    """The window maximum must be bounded below by the whole-chunk score.

    Without the full chunk in the candidate set, span rescoring could *lower*
    a chunk's score, and any regression would then be ambiguous between "a
    window won that should not have" and "we simply stopped looking at the
    whole thing".
    """
    from scripts.probe_span_rerank import sentence_windows

    text = "".join(f"这是第{i}句话，内容足够长以便切分出有意义的窗口来测试。" for i in range(6))
    windows = sentence_windows(text, window=3, stride=2)
    assert text in windows
    assert len(windows) > 1


def test_sentence_windows_degrade_safely_on_short_or_empty_text():
    from scripts.probe_span_rerank import sentence_windows

    assert sentence_windows("", window=3, stride=2) == [""]
    assert sentence_windows("很短", window=3, stride=2) == ["很短"]
    # a text whose sentences are all below the minimum still yields one window
    tiny = "一句。两句。三句。"
    assert sentence_windows(tiny, window=3, stride=2) == [tiny]


@pytest.mark.private_artifact
def test_negative_guard_merges_every_source_without_duplicates():
    """Today's failure was running one guard and calling it the guard.

    Each source has a blind spot set by how it was built: the 19 colloquial
    negatives could not express "topic adjacent, absent from corpus", and that
    is exactly the shape that produced three false accepts elsewhere.
    """
    from scripts.run_negative_guard import SOURCES, collect

    rows = collect()
    sources = {row["source"] for row in rows}
    assert sources == {name for name, _path, _kind in SOURCES}
    questions = [row["q"].strip() for row in rows]
    assert len(questions) == len(set(questions)), "the same question counted twice"
    assert len(rows) > 100
