from __future__ import annotations

import copy
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_hard_negatives import (
    HardNegativeBuildError,
    build_hard_negative_dataset,
    production_scoring_text,
)
from rag_qrels import answer_span_hash
from train_reranker_hard_negatives import (
    DEFAULT_PAIRWISE_KIND_WEIGHTS,
    TrainingInputError,
    assert_output_path_allowed,
    deterministic_group_split,
    labeled_pairs,
    load_training_artifact,
    train_binary_cross_encoder,
    train_pairwise_cross_encoder,
    triple_kind_counts,
)


class FakeCollection:
    name = "test_collection"

    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)

    def get(self, ids=None, include=None):
        ids = list(ids or self.rows)
        selected = [(chunk_id, self.rows[chunk_id]) for chunk_id in ids
                    if chunk_id in self.rows]
        return {
            "ids": [chunk_id for chunk_id, _ in selected],
            "documents": [row["document"] for _, row in selected],
            "metadatas": [row["metadata"] for _, row in selected],
        }


def _target(chunk_id, source, excerpt, relevance="direct"):
    return {
        "source": source,
        "heading_path": ["Section"],
        "chunk_id": chunk_id,
        "answer_span_hash": answer_span_hash(excerpt),
        "evidence_excerpt": excerpt,
        "relevance": relevance,
        "review_note": "independent reviewer evidence",
    }


def _fixture_inputs():
    rows = {
        "p1": {"document": "direct answer one", "metadata": {
            "source": "doc.md", "title": "Title", "heading_path": "Section A",
        }},
        "s1": {"document": "support only", "metadata": {
            "source": "doc.md", "title": "Title", "heading_path": "Section S",
        }},
        "n1": {"document": "current false winner", "metadata": {
            "source": "other.md", "title": "Other", "heading_path": "Wrong",
        }},
        "n2": {"document": "same document wrong section", "metadata": {
            "source": "doc.md", "title": "Title", "heading_path": "Section B",
        }},
        "p2": {"document": "candidate absent direct", "metadata": {
            "source": "absent.md", "title": "Absent", "heading_path": "Only",
        }},
        "p3": {"document": "already correct direct", "metadata": {
            "source": "correct.md", "title": "Correct", "heading_path": "Only",
        }},
    }
    collection = FakeCollection(rows)
    items = [
        {"id": f"q{i:02d}", "q": f"question {i}", "expect_sources": ["x"]}
        for i in range(1, 53)
    ]
    qrel_items = []
    for item in items:
        query_id = item["id"]
        if query_id == "q01":
            qrel_items.append({
                "query_id": query_id,
                "review_outcome": "accepted",
                "review_note": "direct and supporting distinction",
                "relevant_targets": [
                    _target("p1", "doc.md", "direct answer one"),
                    _target("s1", "doc.md", "support only", "supporting"),
                ],
            })
        elif query_id == "q02":
            qrel_items.append({
                "query_id": query_id,
                "review_outcome": "partial",
                "review_note": "direct target is outside candidate20",
                "relevant_targets": [
                    _target("p2", "absent.md", "candidate absent direct"),
                ],
            })
        elif query_id == "q03":
            qrel_items.append({
                "query_id": query_id,
                "review_outcome": "accepted",
                "review_note": "already ranked first",
                "relevant_targets": [
                    _target("p3", "correct.md", "already correct direct"),
                ],
            })
        else:
            qrel_items.append({
                "query_id": query_id,
                "review_outcome": "unsupported",
                "review_note": "not used by this focused fixture",
                "relevant_targets": [],
            })
    qrels = {
        "schema_version": "rag-qrels-overlay-v1",
        "reviewer_id": "adjudicated-test",
        "base_set": "tests/rag_bench_paraphrase_set.json",
        "index": {"collection": collection.name, "count": collection.count()},
        "items": qrel_items,
    }
    baseline_rows = []
    for item in items:
        if item["id"] == "q01":
            final_ids, fusion_ids = ["n1", "p1"], ["n1", "p1", "s1", "n2"]
        elif item["id"] == "q02":
            final_ids, fusion_ids = ["n1"], ["n1", "n2"]
        elif item["id"] == "q03":
            final_ids, fusion_ids = ["p3", "n1"], ["n1", "p3"]
        else:
            final_ids, fusion_ids = ["n1"], ["n1"]
        baseline_rows.append({
            "id": item["id"], "question": item["q"],
            "top1_source": rows[final_ids[0]]["metadata"]["source"],
            "final_chunk_ids": final_ids, "fusion_chunk_ids": fusion_ids,
        })
    baseline = {
        "schema_version": "reranker-profile-ab-v1",
        "arm": "A0",
        "profile": {
            "pool_size": 20,
            "reranker_model": "BAAI/bge-reranker-base",
            "reranker_use_breadcrumb": False,
            "enable_hyde": False,
            "enable_query_rewrite": False,
            "enable_doc2query": False,
            "enable_quota": False,
            "enable_alias": False,
            "enable_rerank_bridge": False,
        },
        "index": {"collection": collection.name, "collection_count": collection.count()},
        "sets": [{"set": "heldout52", "runs": [{"rows": baseline_rows}]}],
    }
    return collection, items, qrels, baseline


def test_builder_selects_only_recalled_direct_false_winners_without_text():
    collection, items, qrels, baseline = _fixture_inputs()
    artifact = build_hard_negative_dataset(
        qrels=qrels, items=items, baseline=baseline, collection=collection,
        include_stability_anchors=False,
    )

    assert artifact["development_only"] is True
    assert artifact["contains_text"] is False
    assert artifact["summary"]["query_count"] == 1
    assert artifact["summary"]["triple_count"] == 2
    assert {row["negative_kind"] for row in artifact["triples"]} == {
        "current_false_winner", "same_document_wrong_section",
    }
    assert {row["positive"]["chunk_id"] for row in artifact["triples"]} == {"p1"}
    assert all(row["positive"]["chunk_id"] != "s1" for row in artifact["triples"])
    assert all("query" not in row for row in artifact["triples"])
    assert all("scoring_text" not in row["positive"] for row in artifact["triples"])
    assert artifact["summary"]["exclusions_by_reason"]["candidate_absent_stage_c"] == 1
    assert artifact["summary"]["exclusions_by_reason"]["already_direct_top1"] == 1


def test_default_builder_adds_separately_reported_stability_anchor():
    collection, items, qrels, baseline = _fixture_inputs()
    artifact = build_hard_negative_dataset(
        qrels=qrels, items=items, baseline=baseline, collection=collection,
    )
    anchors = [
        row for row in artifact["triples"]
        if row["negative_kind"] == "stability_anchor"
    ]
    assert len(anchors) == 1
    assert anchors[0]["query_id"] == "q03"
    assert anchors[0]["positive"]["chunk_id"] == "p3"
    assert anchors[0]["negative"]["chunk_id"] == "n1"
    assert artifact["summary"]["false_winner_query_count"] == 1
    assert artifact["summary"]["stability_anchor_query_count"] == 1
    assert artifact["summary"]["stability_anchor_count"] == 1


def test_include_text_uses_the_production_breadcrumb_formatter():
    collection, items, qrels, baseline = _fixture_inputs()
    artifact = build_hard_negative_dataset(
        qrels=qrels, items=items, baseline=baseline, collection=collection,
        include_text=True,
    )
    row = artifact["triples"][0]
    expected = production_scoring_text(
        collection.rows[row["positive"]["chunk_id"]]["document"],
        collection.rows[row["positive"]["chunk_id"]]["metadata"],
    )
    assert row["query"] == "question 1"
    assert row["positive"]["scoring_text"] == expected
    assert expected == "来源文档：doc.md\n章节：Title\n章节路径：Section A\n正文：\ndirect answer one"


def test_legacy_baseline_requires_explicit_trace_reconstruction():
    collection, items, qrels, baseline = _fixture_inputs()
    legacy = copy.deepcopy(baseline)
    for row in legacy["sets"][0]["runs"][0]["rows"]:
        row.pop("final_chunk_ids")
        row.pop("fusion_chunk_ids")
    with pytest.raises(HardNegativeBuildError, match="lacks chunk IDs"):
        build_hard_negative_dataset(
            qrels=qrels, items=items, baseline=legacy, collection=collection,
        )

    source_rows = {
        row["id"]: next(
            original for original in baseline["sets"][0]["runs"][0]["rows"]
            if original["id"] == row["id"]
        )
        for row in legacy["sets"][0]["runs"][0]["rows"]
    }
    def resolver(item, old_row):
        return {**old_row, **{
            key: source_rows[item["id"]][key]
            for key in ("final_chunk_ids", "fusion_chunk_ids")
        }}

    artifact = build_hard_negative_dataset(
        qrels=qrels, items=items, baseline=legacy, collection=collection,
        reconstruct_missing=True, trace_resolver=resolver,
    )
    assert "q01" in artifact["provenance"]["reconstructed_trace_query_ids"]


def _text_training_artifact():
    triples = []
    for query_id in ("q1", "q2", "q3"):
        triples.append({
            "query_id": query_id,
            "query": f"question {query_id}",
            "positive": {"chunk_id": f"p-{query_id}", "scoring_text": "positive"},
            "negative": {"chunk_id": f"n-{query_id}", "scoring_text": "negative"},
        })
    return {
        "schema_version": "reranker-hard-negatives-v1",
        "development_only": True,
        "contains_text": True,
        "private_blind_set": False,
        "base_set": "tests/rag_bench_paraphrase_set.json",
        "provenance": {"qrels_sha256": "sha256:test", "index": {}},
        "triples": triples,
    }


def test_training_split_is_deterministic_and_grouped():
    artifact = _text_training_artifact()
    train_a, validation_a, split_a = deterministic_group_split(
        artifact["triples"], seed=7,
    )
    train_b, validation_b, split_b = deterministic_group_split(
        list(reversed(artifact["triples"])), seed=7,
    )
    assert split_a == split_b
    assert {row["query_id"] for row in train_a}.isdisjoint(
        {row["query_id"] for row in validation_a}
    )
    assert {row["query_id"] for row in train_a} == {
        row["query_id"] for row in train_b
    }
    assert len(labeled_pairs(train_a)) == 2 * len(train_a)
    assert triple_kind_counts(train_a)["stability_anchor"] == 0


def test_training_loader_requires_text_and_rejects_blind_provenance(tmp_path):
    artifact = _text_training_artifact()
    path = tmp_path / "artifact.json"
    artifact["contains_text"] = False
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="--include-text"):
        load_training_artifact(path)

    artifact["contains_text"] = True
    artifact["provenance"]["source"] = "router_blind_v1.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="blind-set"):
        load_training_artifact(path)


def test_development_model_refuses_formal_path_without_explicit_flag(tmp_path):
    formal = tmp_path / "formal-models"
    target = formal / "candidate"
    with pytest.raises(TrainingInputError, match="formal model path"):
        assert_output_path_allowed(target, extra_formal_roots=[formal])
    assert assert_output_path_allowed(
        target,
        allow_formal_model_path=True,
        extra_formal_roots=[formal],
    ) == target.resolve()


def test_manual_training_loop_never_imports_optional_datasets(monkeypatch, tmp_path):
    import torch

    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("datasets."):
            raise AssertionError("manual training must not import datasets")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("OFFERCLAW_TORCH_DEVICE", "cpu")

    class FakeTokenizer:
        def __call__(self, queries, documents, **_kwargs):
            values = [float(len(query) + len(document)) for query, document in zip(queries, documents)]
            return {"input_ids": torch.tensor(values).reshape(-1, 1)}

        def save_pretrained(self, path):
            Path(path, "tokenizer.fake").write_text("saved", encoding="utf-8")

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.01]]))

        def forward(self, input_ids):
            return SimpleNamespace(logits=input_ids * self.weight)

        def save_pretrained(self, path, safe_serialization=True):
            assert safe_serialization is True
            Path(path, "model.fake").write_text("saved", encoding="utf-8")

    class FakeEncoder:
        def __init__(self):
            self.model = FakeClassifier()
            self.tokenizer = FakeTokenizer()
            self.max_seq_length = 64

    calls = []
    def factory(model_name, num_labels, device):
        calls.append((model_name, num_labels, device))
        return FakeEncoder()

    rows = [
        {"query": "q1", "scoring_text": "p1", "label": 1},
        {"query": "q1", "scoring_text": "n1", "label": 0},
        {"query": "q2", "scoring_text": "p2", "label": 1},
        {"query": "q2", "scoring_text": "n2", "label": 0},
    ]
    output = tmp_path / "manual-model"
    result = train_binary_cross_encoder(
        train_pairs=rows,
        validation_pairs=rows[:2],
        output=output,
        base_model="fake-base",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        seed=9,
        cross_encoder_factory=factory,
    )
    assert calls == [("fake-base", 1, "cpu")]
    assert result["engine"] == "torch_transformers_binary_loop_v1"
    assert result["optimizer_steps"] == 4
    assert (output / "model.fake").exists()
    assert (output / "tokenizer.fake").exists()


def test_pairwise_training_keeps_pairs_and_reports_stability_metrics(
    monkeypatch, tmp_path,
):
    import torch

    monkeypatch.setenv("OFFERCLAW_TORCH_DEVICE", "cpu")

    class FakeTokenizer:
        def __call__(self, queries, documents, **_kwargs):
            values = [
                float(len(query) + len(document))
                for query, document in zip(queries, documents)
            ]
            return {"input_ids": torch.tensor(values).reshape(-1, 1)}

        def save_pretrained(self, path):
            Path(path, "tokenizer.fake").write_text("saved", encoding="utf-8")

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(layer=torch.nn.ModuleList([
                torch.nn.Linear(1, 1), torch.nn.Linear(1, 1),
            ]))

    class FakeClassifier(torch.nn.Module):
        base_model_prefix = "backbone"

        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.classifier = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.classifier.weight.fill_(0.01)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.classifier(input_ids))

        def save_pretrained(self, path, safe_serialization=True):
            assert safe_serialization is True
            Path(path, "model.fake").write_text("saved", encoding="utf-8")

    class FakeEncoder:
        def __init__(self):
            self.model = FakeClassifier()
            self.tokenizer = FakeTokenizer()
            self.max_seq_length = 64

    def factory(_model_name, num_labels, device):
        assert num_labels == 1
        assert device == "cpu"
        return FakeEncoder()

    def triple(query_id, positive, negative, kind):
        return {
            "query_id": query_id,
            "query": f"question {query_id}",
            "negative_kind": kind,
            "positive": {"chunk_id": f"p-{query_id}", "scoring_text": positive},
            "negative": {"chunk_id": f"n-{query_id}", "scoring_text": negative},
        }

    train = [
        triple("q1", "positive-long", "n", "current_false_winner"),
        triple("q2", "correct-long", "n", "stability_anchor"),
    ]
    validation = [
        triple("q3", "verified-long", "n", "stability_anchor"),
    ]
    output = tmp_path / "pairwise-model"
    result = train_pairwise_cross_encoder(
        train_triples=train,
        validation_triples=validation,
        output=output,
        base_model="fake-base",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        seed=9,
        margin=0.5,
        trainable_encoder_layers=0,
        cross_encoder_factory=factory,
    )

    assert result["engine"] == "torch_transformers_pairwise_loop_v1"
    assert result["objective"] == "weighted_pairwise_logistic_margin"
    assert result["kind_weights"]["stability_anchor"] == 2.0
    assert result["trainable_encoder_layers"] == 0
    assert result["pre_train_metrics"]["pair_count"] == 2
    assert result["post_validation_metrics"]["pair_count"] == 1
    assert (
        result["post_validation_metrics"]["by_kind"]["stability_anchor"]["pair_count"]
        == 1
    )
    assert DEFAULT_PAIRWISE_KIND_WEIGHTS["current_false_winner"] == 1.0
    assert (output / "model.fake").exists()
    assert (output / "tokenizer.fake").exists()


def test_pair_flip_report_separates_rescaling_from_reordering():
    """The F3 failure mode has to be visible in one number.

    Both runs below improve every aggregate the trainer already reported --
    accuracy is flat or better and the mean margin grows -- but only one of
    them changes an ordering.  ``pair_accuracy`` cannot tell them apart when
    the flips cancel, which is precisely why loss went down while end-to-end
    R@1 moved 0 wins / 0 losses.
    """
    from train_reranker_hard_negatives import pair_flip_report

    kinds = ["same_document_wrong_section", "same_document_wrong_section",
             "adjudicated_hard_negative"]
    pre = {"score_margins": [-0.5, 2.0, -0.3], "kinds": kinds}

    # (a) pure rescaling: every margin grows ~10%, nothing crosses zero
    rescaled = {"score_margins": [-0.55, 2.2, -0.33], "kinds": kinds}
    report = pair_flip_report(pre, rescaled)
    assert report["overall"]["net_flips"] == 0
    assert report["overall"]["flipped_to_correct"] == 0
    assert report["overall"]["flipped_to_wrong"] == 0

    # (b) a real correction on the hard negative, paid for elsewhere
    reordered = {"score_margins": [-0.5, -0.1, 0.4], "kinds": kinds}
    report = pair_flip_report(pre, reordered)
    assert report["overall"] == {"pairs": 3, "flipped_to_correct": 1,
                                 "flipped_to_wrong": 1, "net_flips": 0}
    assert report["adjudicated_hard_negative"]["flipped_to_correct"] == 1
    assert report["same_document_wrong_section"]["flipped_to_wrong"] == 1


def test_pair_flip_report_refuses_mismatched_measurements():
    """Diffing two different row orders would invent flips out of nothing."""
    from train_reranker_hard_negatives import TrainingInputError, pair_flip_report

    pre = {"score_margins": [0.1, 0.2], "kinds": ["a", "b"]}
    with pytest.raises(TrainingInputError, match="same rows"):
        pair_flip_report(pre, {"score_margins": [0.1], "kinds": ["a"]})
    with pytest.raises(TrainingInputError, match="same rows"):
        pair_flip_report(pre, {"score_margins": [0.2, 0.1], "kinds": ["b", "a"]})


def test_exactly_zero_margin_counts_as_not_yet_correct():
    """A tie is not a win: the gold must strictly outscore the negative."""
    from train_reranker_hard_negatives import pair_flip_report

    kinds = ["adjudicated_hard_negative"]
    assert pair_flip_report(
        {"score_margins": [-0.1], "kinds": kinds},
        {"score_margins": [0.0], "kinds": kinds},
    )["overall"]["flipped_to_correct"] == 0
    assert pair_flip_report(
        {"score_margins": [0.0], "kinds": kinds},
        {"score_margins": [0.1], "kinds": kinds},
    )["overall"]["flipped_to_correct"] == 1
