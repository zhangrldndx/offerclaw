# -*- coding: utf-8 -*-
"""Tests for the colloquial-negative pipeline.

The rule these exist to protect: a candidate becomes a negative only when a
human wrote a verdict against evidence that the index still returns.  Anything
that would let an un-adjudicated question drift into the guard set is a defect,
because the guard's whole job is to stop the evidence gate from being loosened
on a set that silently agrees with it.
"""

import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manage_colloquial_negatives import (  # noqa: E402
    _load_candidates,
    build,
    render_worksheet,
)


INDEX = {
    "collection": "test_collection",
    "collection_count": 3,
    "collection_content_hash": "abc123",
    "embedding_provider": "local",
    "embedding_model": "BAAI/bge-base-zh-v1.5",
    "embedding_dimensions": 768,
    "chunker_version": "2026-08-10",
    "fingerprint_id": "deadbeef",
}


def _candidate(cid="v2aneg-001", negative_type="wrong_relation",
               style="natural", question="跳表是不是用来存向量的？"):
    return {"id": cid, "negative_type": negative_type, "query_style": style,
            "question": question, "intent": "两个词都在库里，但没有这层关系"}


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _probe_row(cid="v2aneg-001", chunk_ids=("c1", "c2"), gate=False):
    return {
        "id": cid, "negative_type": "wrong_relation", "query_style": "natural",
        "question": "跳表是不是用来存向量的？", "intent": "i",
        "gate_decision": gate, "gate_features": {"best_dense_distance": 0.9},
        "top_candidates": [
            {"rank": rank, "chunk_id": chunk_id, "source": "s.md", "heading": "",
             "distance": 0.9, "rerank_score": 0.5, "snippet": "..."}
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ],
    }


def _run_build(tmp_path, *, candidates, probe_rows, verdicts):
    args = argparse.Namespace(
        candidates=str(_write(tmp_path, "cand.json", {"candidates": candidates})),
        probe=str(_write(tmp_path, "probe.json",
                         {"arm": "compact32", "index": INDEX, "rows": probe_rows})),
        adjudication=str(_write(tmp_path, "adj.json",
                                {"review_note": "n", "verdicts": verdicts})),
        output=str(tmp_path / "out.json"),
    )
    build(args)
    return json.loads(Path(args.output).read_text(encoding="utf-8"))


def test_only_adjudicated_candidates_become_negatives(tmp_path, capsys):
    """An un-adjudicated candidate is refused, not labelled."""
    payload = _run_build(
        tmp_path,
        candidates=[_candidate("v2aneg-001"), _candidate("v2aneg-002")],
        probe_rows=[_probe_row("v2aneg-001"), _probe_row("v2aneg-002")],
        verdicts={"v2aneg-001": {"verdict": "unanswerable", "rationale": "r",
                                 "checked_chunk_ids": ["c1"]}},
    )
    assert [item["query_id"] for item in payload["items"]] == ["v2aneg-001"]
    assert "v2aneg-002" in capsys.readouterr().err


def test_a_partially_answerable_verdict_is_not_a_negative(tmp_path):
    """The one that got dropped in practice: MoE was partially covered."""
    payload = _run_build(
        tmp_path,
        candidates=[_candidate("v2aneg-001"), _candidate("v2aneg-009")],
        probe_rows=[_probe_row("v2aneg-001"), _probe_row("v2aneg-009")],
        verdicts={
            "v2aneg-001": {"verdict": "unanswerable", "rationale": "r",
                           "checked_chunk_ids": ["c1"]},
            "v2aneg-009": {"verdict": "partially_answerable",
                           "rationale": "库里点名了 MoE 并解释了只激活一部分",
                           "checked_chunk_ids": ["c1"]},
        },
    )
    assert [item["query_id"] for item in payload["items"]] == ["v2aneg-001"]


def test_verdict_without_a_rationale_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="rationale"):
        _run_build(tmp_path, candidates=[_candidate()],
                   probe_rows=[_probe_row()],
                   verdicts={"v2aneg-001": {"verdict": "unanswerable",
                                            "rationale": "   ",
                                            "checked_chunk_ids": ["c1"]}})


def test_stale_adjudication_evidence_is_rejected(tmp_path):
    """The rationale was written about specific chunks.

    If retrieval no longer returns them the reasoning behind the verdict is
    about a corpus that no longer exists, so the case has to be re-adjudicated
    rather than shipped on the strength of an old note.
    """
    with pytest.raises(SystemExit, match="no longer returns"):
        _run_build(tmp_path, candidates=[_candidate()],
                   probe_rows=[_probe_row(chunk_ids=("c9", "c8"))],
                   verdicts={"v2aneg-001": {"verdict": "unanswerable",
                                            "rationale": "r",
                                            "checked_chunk_ids": ["c1"]}})


def test_adjudicated_but_never_probed_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="never probed"):
        _run_build(tmp_path, candidates=[_candidate()], probe_rows=[],
                   verdicts={"v2aneg-001": {"verdict": "unanswerable",
                                            "rationale": "r",
                                            "checked_chunk_ids": []}})


def test_emitted_negatives_carry_no_targets_or_requirements(tmp_path):
    payload = _run_build(
        tmp_path, candidates=[_candidate()], probe_rows=[_probe_row()],
        verdicts={"v2aneg-001": {"verdict": "unanswerable", "rationale": "r",
                                 "checked_chunk_ids": ["c1"]}})
    item = payload["items"][0]
    assert item["case_kind"] == "negative"
    assert item["relevant_targets"] == [] and item["answer_requirements"] == []
    assert item["expected_behavior"] == "abstain_from_kb"
    # the live gate decision is recorded, so a later loosening can be compared
    # against what the gate did when the case was adjudicated
    assert item["adjudication"]["gate_decision_at_adjudication"] is False


def test_formal_register_candidates_are_refused(tmp_path):
    """V1 already covers formal negatives; a standard-phrased row adds nothing.

    This set exists because V1's colloquial phrasings sit at cosine 0.98 from
    their standard form, so letting formal rows in would rebuild the very blind
    spot it was created to close.
    """
    path = _write(tmp_path, "c.json",
                  {"candidates": [_candidate(style="standard")]})
    with pytest.raises(SystemExit, match="query_style must be one of"):
        _load_candidates(path)


def test_unknown_negative_type_and_duplicate_ids_are_refused(tmp_path):
    bad_type = _write(tmp_path, "a.json",
                      {"candidates": [_candidate(negative_type="vibes")]})
    with pytest.raises(SystemExit, match="unknown negative_type"):
        _load_candidates(bad_type)
    duplicate = _write(tmp_path, "b.json",
                       {"candidates": [_candidate(), _candidate()]})
    with pytest.raises(SystemExit, match="duplicate candidate id"):
        _load_candidates(duplicate)


def test_worksheet_flags_a_candidate_the_gate_already_accepts():
    """The worksheet is read by a human, so the live false accept must stand out."""
    report = {"index": {"collection": "c", "count": 3},
              "rows": [_probe_row(gate=True), _probe_row("v2aneg-002")]}
    text = render_worksheet(report)
    assert "放行 ⚠️" in text and "拒答 ✅" in text
