# -*- coding: utf-8 -*-
"""Tests for the colloquial second acceptance path on the evidence gate.

The gate's primary test is a dense-distance threshold calibrated on formally
phrased questions.  Colloquial phrasings of the same question sit farther from
their own gold chunk (cosine 0.56-0.62 vs 0.66), so that threshold refuses them
by construction -- measured on Dev-New, correct Top-1 answers passed the gate
82% of the time for ``standard`` and 30% for ``implicit_oral``.

This path lets a refused answer back in when the reranker is both very sure and
clearly separated from the runner-up.  Both halves are load-bearing: of the five
wrong-Top-1 rows that the confidence threshold alone would have admitted, four
were high-confidence-but-tied (the reranker picking arbitrarily among near
duplicates) and the margin condition excluded all four.
"""

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_gate import colloquial_gate_accept  # noqa: E402


@pytest.fixture
def gate_env(monkeypatch):
    def configure(tau=None, margin=None):
        monkeypatch.delenv("RAG_COLLOQUIAL_GATE_MIN", raising=False)
        monkeypatch.delenv("RAG_COLLOQUIAL_GATE_MARGIN", raising=False)
        if tau is not None:
            monkeypatch.setenv("RAG_COLLOQUIAL_GATE_MIN", tau)
        if margin is not None:
            monkeypatch.setenv("RAG_COLLOQUIAL_GATE_MARGIN", margin)
    return configure


def test_default_is_off_no_matter_how_confident_the_reranker_is(gate_env):
    gate_env()
    assert colloquial_gate_accept(1.0, 1.0) is False


def test_both_conditions_are_required(gate_env):
    """The calibration only holds as a conjunction.

    Confidence alone admitted 5 wrong-Top-1 rows on Dev-New; 4 of them were
    within 0.07 of their runner-up.  An ``or`` here would silently restore
    exactly the failure the margin condition exists to prevent.
    """
    gate_env(tau="0.87", margin="0.10")
    assert colloquial_gate_accept(0.90, 0.20) is True
    assert colloquial_gate_accept(0.9981, 0.0026) is False   # sure but tied
    assert colloquial_gate_accept(0.50, 0.90) is False       # separated but unsure


def test_thresholds_are_inclusive_at_the_calibrated_point(gate_env):
    gate_env(tau="0.87", margin="0.10")
    assert colloquial_gate_accept(0.87, 0.10) is True
    assert colloquial_gate_accept(0.8699, 0.10) is False
    assert colloquial_gate_accept(0.87, 0.0999) is False


def test_the_strongest_currently_refused_negative_stays_refused(gate_env):
    """τ=0.87 was chosen for exactly this clearance.

    The highest-scoring negative the gate currently refuses is v2aneg-011
    ("BM25 是不是负责给简历打分排序的") at 0.8630 -- a question whose vocabulary is
    entirely real, which is why it scores so high.  0.007 is the whole safety
    margin, so a test has to pin it rather than trust the round number.
    """
    gate_env(tau="0.87", margin="0.10")
    assert colloquial_gate_accept(0.8630, 0.50) is False


def test_margin_defaults_to_the_calibrated_value_when_unset(gate_env):
    gate_env(tau="0.87")
    assert colloquial_gate_accept(0.95, 0.12) is True
    assert colloquial_gate_accept(0.95, 0.08) is False


@pytest.mark.parametrize("tau,margin", [
    ("abc", "0.10"), ("0.87", "not-a-number"), ("", "0.10"), ("   ", "0.10"),
])
def test_unparseable_configuration_refuses_rather_than_guesses(gate_env, tau, margin):
    """A gate that fails open is worse than one that is switched off."""
    gate_env(tau=tau, margin=margin)
    assert colloquial_gate_accept(0.99, 0.99) is False


def test_missing_reranker_signals_refuse(gate_env):
    gate_env(tau="0.87", margin="0.10")
    assert colloquial_gate_accept(None, 0.5) is False
    assert colloquial_gate_accept(0.99, None) is False


def test_second_path_only_runs_after_the_main_gate_refused():
    """Source assertion: the path must never override an acceptance.

    It is wired inline in ``retrieve_with_trace``, so structure is pinned the
    same way the English evidence gate is.
    """
    source = (Path(__file__).resolve().parents[1] / "rag_gate.py").read_text(
        encoding="utf-8")
    assert "_rr_accept = (not in_kb) and colloquial_gate_accept(" in source
    # accepted rows take evidence by rank, not by distance cutoff: these
    # candidates are above the rescue distance by construction, so filtering by
    # distance would hand the generator an empty chunk list
    branch = source.find("if _en_accept or _rr_accept")
    assert 0 < branch < source.find("list(zip(docs, metas))[:3]")
    # and the trace has to say which path let it in
    assert '"en_evidence" if _en_accept' in source
    assert source.rstrip().count('else "rerank_confidence")') == 1
    assert '"colloquial_gate_accepted": _rr_accept,' in source


def test_calibration_knob_is_visible_in_the_trace_features():
    source = (Path(__file__).resolve().parents[1] / "rag_gate.py").read_text(
        encoding="utf-8")
    assert '"colloquial_gate_min": _rr_gate_raw or None,' in source


def test_environment_is_not_leaked_between_tests():
    # guards the fixture itself: a stray env var would make every other test
    # in this file pass for the wrong reason
    assert os.environ.get("RAG_COLLOQUIAL_GATE_MIN") in (None, "")
