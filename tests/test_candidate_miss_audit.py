from __future__ import annotations

from dataclasses import dataclass

from audit_candidate_misses import _baseline_rows, _rank, _token_offsets


@dataclass(frozen=True)
class _Candidate:
    chunk_id: str


class _CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return list(text)


def test_baseline_rows_selects_only_production_strategy():
    payload = {
        "strategies": [
            {"strategy": "source_cap4", "rows": [{"id": "wrong"}]},
            {"strategy": "baseline_rrf20", "rows": [{"id": "q1"}]},
        ]
    }
    assert _baseline_rows(payload) == {"q1": {"id": "q1"}}


def test_rank_is_one_based_and_zero_when_target_is_absent():
    candidates = [_Candidate("a"), _Candidate("b")]
    assert _rank(candidates, {"b"}) == 2
    assert _rank(candidates, {"missing"}) == 0


def test_token_offsets_are_measured_in_document_coordinates():
    tokenizer = _CharTokenizer()
    assert _token_offsets(tokenizer, "prefix evidence suffix", "evidence") == (7, 15)
    assert _token_offsets(tokenizer, "prefix evidence suffix", "missing") is None
