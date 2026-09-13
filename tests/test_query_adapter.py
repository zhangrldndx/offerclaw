# -*- coding: utf-8 -*-
"""Tests for the query-side LoRA adapter.

The design rests on one property: documents never move.  The adapter maps a
question into the *existing* document space, which is what makes it deployable
without re-indexing -- and which stops being true the moment anything encodes a
document through it.  Several of these tests exist only to keep that honest.

The second theme is F3's lesson: a training run can improve every aggregate it
reports while changing no ranking decision at all.  ``flips`` is the number
that would have caught it, so it is pinned here.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from train_query_adapter import (  # noqa: E402
    LoRALinear,
    QueryAdapterError,
    attach_lora,
    flips,
    info_nce,
    load_queries,
    rank_of_gold,
    summarize,
)


def test_zero_initialised_adapter_is_an_exact_no_op():
    """Everything else depends on this.

    If the adapter did not start as the identity, an untrained run would not
    reproduce the frozen baseline and there would be no way to tell a real
    gain from an initialisation artefact.  Measured end to end: the untrained
    tower returns dev R@1 37/80, the same number the ridge probe's identity arm
    reported.
    """
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 16)
    wrapped = LoRALinear(base, rank=4, alpha=8.0)
    value = torch.randn(3, 16)
    assert torch.allclose(wrapped(value), base(value), atol=0)
    # ... and it stops being a no-op only once B has moved off zero
    with torch.no_grad():
        wrapped.lora_b.add_(0.1)
    assert not torch.allclose(wrapped(value), base(value))


def test_wrapped_base_weights_are_frozen_and_only_lora_trains():
    base = torch.nn.Linear(8, 8)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable == {"lora_a", "lora_b"}


def test_attach_lora_refuses_a_model_it_cannot_reach():
    """A rename upstream would otherwise train nothing, silently.

    That failure mode is indistinguishable from "the method does not work",
    and this project has already published one round where an env knob was
    silently swallowed and the arm ran as the baseline.
    """
    class Unrelated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(4, 4)

    with pytest.raises(QueryAdapterError, match="no projections named"):
        attach_lora(Unrelated(), rank=2, alpha=4.0)


def test_attach_lora_never_double_wraps_and_says_so_loudly():
    """Wrapping a wrapper would stack two adapters and silently change the
    geometry a checkpoint was trained with.  Already-wrapped layers are
    skipped, which leaves nothing to wrap -- and "nothing was wrapped" is
    exactly the condition that must raise rather than pass quietly."""
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(4, 4)
            self.key = torch.nn.Linear(4, 4)
            self.value = torch.nn.Linear(4, 4)

    block = Block()
    assert attach_lora(block, 2, 4.0) == 3
    with pytest.raises(QueryAdapterError, match="no projections named"):
        attach_lora(block, 2, 4.0)
    # the first wrappers are intact -- one adapter each, not two
    assert sum(1 for _ in block.query.parameters()) == 4  # base w/b + lora a/b


def test_flips_is_the_metric_that_would_have_caught_f3():
    """Aggregate movement without decision movement must read as zero.

    F3 reduced its loss, grew its mean margin 8%, and changed one pair in
    1312; the reported metrics all looked healthy.
    """
    before = [1, 5, 1, 9]
    # everything improved, but no rank-1 decision changed
    assert flips(before, [1, 3, 1, 4]) == {"gained": 0, "lost": 0, "net": 0}
    # one real gain, one real loss -- net zero, but not silent
    assert flips(before, [5, 1, 1, 9]) == {"gained": 1, "lost": 1, "net": 0}
    assert flips(before, [1, 1, 1, 1]) == {"gained": 2, "lost": 0, "net": 2}


def test_rank_of_gold_uses_the_best_placed_gold_and_counts_ties_as_winning():
    scores = torch.tensor([[0.9, 0.5, 0.9]])
    assert rank_of_gold(scores, [[2]]) == [1]      # tie with the leader
    assert rank_of_gold(scores, [[1]]) == [3]
    assert rank_of_gold(scores, [[0, 1]]) == [1]   # best of the two golds


def test_summarize_reports_per_style_and_overall():
    stats = summarize([1, 2, 1, 30], ["natural", "natural", "standard", "standard"])
    assert stats["overall"]["n"] == 4 and stats["overall"]["r1"] == 2
    assert stats["natural"]["r3"] == 2
    assert stats["standard"]["r20"] == 1


def test_info_nce_rewards_putting_the_gold_first():
    corpus_scores = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])
    good = info_nce(corpus_scores, [[0], [2]], temperature=0.05)
    bad = info_nce(corpus_scores, [[1], [1]], temperature=0.05)
    assert good < bad


def test_info_nce_accepts_multiple_golds_without_double_counting():
    scores = torch.tensor([[0.9, 0.8, 0.1]])
    single = info_nce(scores, [[0]], temperature=0.05)
    both = info_nce(scores, [[0, 1]], temperature=0.05)
    # crediting a second gold can only help, never hurt
    assert both < single


def test_load_queries_skips_negatives_and_rows_whose_gold_is_absent(tmp_path):
    import json

    payload = {"items": [
        {"query_id": "q1", "anchor_id": "a1", "split": "train",
         "case_kind": "positive", "question": "在？", "query_style": "natural",
         "relevant_targets": [{"chunk_id": "c1"}]},
        {"query_id": "q2", "anchor_id": "a2", "split": "train",
         "case_kind": "negative", "question": "无关", "relevant_targets": []},
        {"query_id": "q3", "anchor_id": "a3", "split": "train",
         "case_kind": "positive", "question": "金标不在库里",
         "relevant_targets": [{"chunk_id": "gone"}]},
        {"query_id": "q4", "anchor_id": "a4", "split": "dev",
         "case_kind": "positive", "question": "别的 split",
         "relevant_targets": [{"chunk_id": "c1"}]},
    ]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    data = load_queries([path], "train", {"c1": 0})
    assert [row["query_id"] for row in data["rows"]] == ["q1"]


def test_load_queries_refuses_an_empty_split(tmp_path):
    import json

    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(QueryAdapterError, match="no usable rows"):
        load_queries([path], "train", {})


# --------------------------------------------------------------------------
# the production hook


def test_adapter_is_off_by_default(monkeypatch):
    import rag_query_adapter as rqa

    monkeypatch.delenv("RAG_QUERY_ADAPTER_DIR", raising=False)
    assert rqa.adapter_path() == ""
    assert rqa.embed_query("随便问点什么") is None


def test_a_checkpoint_without_geometry_is_refused(tmp_path, monkeypatch):
    """Rank and alpha cannot be guessed, and guessing them wrong loads an
    adapter that is not the one that was measured."""
    import rag_query_adapter as rqa

    path = tmp_path / "bare.pt"
    torch.save({"lora_a": torch.zeros(2, 2)}, path)
    monkeypatch.setenv("RAG_QUERY_ADAPTER_DIR", str(path))
    rqa._CACHE.clear()
    with pytest.raises(RuntimeError, match="not a query-adapter checkpoint"):
        rqa.embed_query("问题")
    rqa._CACHE.clear()


def test_retrieval_calls_the_adapter_at_exactly_one_site():
    """Query-only is a structural property, not a convention.

    Ingest must have no path to the adapter; the guarantee that documents stay
    valid depends on it.
    """
    root = Path(__file__).resolve().parents[1]
    gate = (root / "rag_gate.py").read_text(encoding="utf-8")
    assert gate.count("_adapter_embed_query(") == 1
    assert "_adapted is not None else get_embeddings_batch(" in gate
    # and nothing on the ingest side reaches it
    for module in ("rag_tools.py", "rag_multi_source.py"):
        text = (root / module).read_text(encoding="utf-8")
        assert "rag_query_adapter" not in text, module
