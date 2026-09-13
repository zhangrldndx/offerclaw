from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_colloquial_reranker_training import (
    ColloquialTrainingBuildError,
    build_training_artifact,
    compact32_scoring_text,
)
from rag_colloquial_profiles import colloquial_profile
from train_colloquial_reranker import (
    TrainingInputError,
    load_colloquial_training_artifact,
)


class FakeCollection:
    def __init__(self):
        self.rows = {
            "p-train": {
                "document": "训练直接答案",
                "metadata": {
                    "source": "/kb/train.md", "title": "训练标题",
                    "heading_path": "上级 > 正确章节",
                },
            },
            "n-train": {
                "document": "训练同文档错误章节",
                "metadata": {
                    "source": "/kb/train.md", "title": "训练标题",
                    "heading_path": "上级 > 错误章节",
                },
            },
            "p-dev": {
                "document": "开发直接答案",
                "metadata": {
                    "source": "/kb/dev.md", "title": "开发标题",
                    "heading_path": "正确章节",
                },
            },
            "n-dev": {
                "document": "开发近主题错误答案",
                "metadata": {
                    "source": "/kb/other.md", "title": "其他标题",
                    "heading_path": "错误章节",
                },
            },
        }

    def get(self, ids=None, include=None):
        selected = [(chunk_id, self.rows[chunk_id]) for chunk_id in ids or []]
        return {
            "ids": [chunk_id for chunk_id, _row in selected],
            "documents": [row["document"] for _chunk_id, row in selected],
            "metadatas": [row["metadata"] for _chunk_id, row in selected],
        }


def _target(chunk_id: str, source: str):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "relevance_grade": 3,
        "supported_requirements": ["回答要求"],
    }


def _item(split: str):
    return {
        "query_id": f"q-{split}",
        "anchor_id": f"anchor-{split}",
        "split": split,
        "case_kind": "positive",
        "query_style": "colloquial",
        "question": f"{split} 这个到底怎么回事？",
        "review_status": "approved",
        "relevant_targets": [_target(f"p-{split}", f"{split}.md")],
        "hard_negatives": [{
            "chunk_id": f"n-{split}",
            "source": "train.md" if split == "train" else "other.md",
            "reason": "已裁决为不足以回答",
        }],
    }


def _payload():
    return {
        "dataset_id": "public-train-dev",
        "index": {"collection": "test", "count": 4, "fingerprint": "sha256:test"},
        "items": [_item("train"), _item("dev")],
    }


def test_export_uses_frozen_splits_and_exact_compact32_text(tmp_path):
    cases = tmp_path / "train_dev.json"
    cases.write_text(json.dumps(_payload()), encoding="utf-8")
    artifact = build_training_artifact(
        cases_path=cases, collection=FakeCollection(),
    )

    assert artifact["private_blind_set"] is False
    assert artifact["summary"]["train_triple_count"] == 1
    assert artifact["summary"]["dev_triple_count"] == 1
    train = next(row for row in artifact["triples"] if row["split"] == "train")
    assert train["negative_kind"] == "same_document_wrong_section"
    assert train["positive"]["scoring_text"] == (
        "标题：训练标题\n章节：上级 > 正确章节\n正文：\n训练直接答案"
    )
    assert set(artifact["split_contract"]["train_anchor_ids"]).isdisjoint(
        artifact["split_contract"]["dev_anchor_ids"]
    )


def test_export_and_trainer_fail_closed_on_blind_paths(tmp_path):
    path = tmp_path / "rag_colloquial_blind_v1.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ColloquialTrainingBuildError, match="Blind"):
        build_training_artifact(cases_path=path, collection=FakeCollection())

    artifact = {
        "schema_version": "colloquial-reranker-training-v1",
        "development_only": True,
        "contains_text": True,
        "private_blind_set": False,
        "provenance": {"blind_set_used": False},
        "triples": [],
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="Blind"):
        load_colloquial_training_artifact(path)


def test_trainer_rejects_anchor_leakage(tmp_path):
    cases = tmp_path / "train_dev.json"
    cases.write_text(json.dumps(_payload()), encoding="utf-8")
    artifact = build_training_artifact(cases_path=cases, collection=FakeCollection())
    artifact["triples"][1]["anchor_id"] = artifact["triples"][0]["anchor_id"]
    exported = tmp_path / "training.json"
    exported.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(TrainingInputError, match="anchor leakage"):
        load_colloquial_training_artifact(exported)


def test_custom_reranker_only_changes_model_identity():
    baseline = colloquial_profile("compact32")
    custom = colloquial_profile("compact32", reranker_model="/tmp/model")
    assert custom.reranker_model == "/tmp/model"
    assert custom.reranker_prefix_mode == baseline.reranker_prefix_mode == "compact32"
    assert custom.pool_size == baseline.pool_size
    assert custom.rrf_k == baseline.rrf_k


def test_compact32_formatter_has_no_full_breadcrumb_labels():
    text = compact32_scoring_text(
        "正文", {"source": "/tmp/a.md", "heading_path": "A > B > C"},
    )
    assert text == "文档：a.md\n章节：B > C\n正文：\n正文"
    assert "来源文档：" not in text
