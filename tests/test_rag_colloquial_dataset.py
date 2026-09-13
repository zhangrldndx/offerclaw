from pathlib import Path

import pytest

from rag_colloquial_dataset import (
    DOMAIN_QUOTAS,
    NEGATIVE_QUESTIONS,
    apply_wording_edits,
    assign_anchor_splits,
    _variants,
    parse_wording_batches,
    review_batch_markdown,
    select_reviewed_split,
    select_anchor_questions,
)
from rag_qrels_v2 import GradedQrelsValidationError, evidence_span_hash


ROOT = Path(__file__).resolve().parents[1]


def test_anchor_and_negative_quotas_are_frozen():
    anchors = select_anchor_questions(ROOT)
    assert len(anchors) == 80
    assert {domain: sum(item["domain"] == domain for item in anchors)
            for domain in DOMAIN_QUOTAS} == DOMAIN_QUOTAS
    assert {category: len(items) for category, items in NEGATIVE_QUESTIONS.items()} == {
        "out_of_domain": 20,
        "near_domain_missing": 20,
        "wrong_relation": 20,
        "ambiguous_or_injection": 20,
    }


def test_grouped_split_is_exact_and_never_splits_a_source_section():
    anchors = [
        {"anchor_id": f"a{index:02d}", "source_group": f"group-{index:02d}"}
        for index in range(80)
    ]
    assignments = assign_anchor_splits(anchors)
    assert sum(value == "train" for value in assignments.values()) == 48
    assert sum(value == "dev" for value in assignments.values()) == 16
    assert sum(value == "blind" for value in assignments.values()) == 16


def test_wording_batch_exposes_question_and_fixed_gold_without_review_controls():
    markdown = review_batch_markdown([{
        "query_id": "q1", "split": "dev", "query_style": "colloquial",
        "question": "这个怎么搜？", "phenomena": ["implicit"],
        "answer_requirements": ["说明搜索机制"],
        "relevant_targets": [{
            "source": "rag.md", "chunk_id": "c1", "relevance_grade": 3,
            "evidence_excerpt": "证据原文",
        }],
        "hard_negatives": [{"chunk_id": "c2", "reason": "错误章节"}],
    }], 1)
    assert "用户问题（只改本行冒号后的文字）" in markdown
    assert "已裁决直接证据" in markdown
    assert "易混淆候选" in markdown
    assert "问题审核" not in markdown
    assert "证据审核" not in markdown


def test_wording_form_parses_only_question_and_preserves_approved_gold(tmp_path):
    payload = _public_split_payload()
    payload["items"] = payload["items"][:1]
    payload["items"][0]["question"] = "原问题"
    markdown = review_batch_markdown(payload["items"], 1).replace(
        "：原问题", "：这是用户改写后的口语问题？"
    )
    path = tmp_path / "batch.md"
    path.write_text(markdown, encoding="utf-8")
    questions = parse_wording_batches([path])
    updated, counts = apply_wording_edits(payload, questions)
    assert updated["items"][0]["question"] == "这是用户改写后的口语问题？"
    assert updated["items"][0]["review_status"] == "approved"
    assert counts == {"changed": 1, "unchanged": 0}


def test_wording_form_tolerates_removed_label_colon(tmp_path):
    path = tmp_path / "batch.md"
    path.write_text(
        "## 1. `q1` · dev · natural\n\n"
        "- 用户问题（只改本行冒号后的文字）Redis 为什么使用单线程？\n"
        "- 现象：natural_paraphrase\n",
        encoding="utf-8",
    )
    assert parse_wording_batches([path]) == {
        "q1": "Redis 为什么使用单线程？"
    }


def test_register_variants_change_style_without_adding_answer_requirements():
    base = "MVCC 的英文全称和中文名称分别是什么"
    variants = _variants(base, "unused legacy paraphrase", "a1")
    assert set(variants) == {"standard", "natural", "colloquial", "long_context"}
    assert all(base in question for question in variants.values())
    assert all("常见误区" not in question for question in variants.values())
    assert all("关键机制" not in question for question in variants.values())


def _public_split_payload():
    excerpt = "RRF 会按各检索通道名次做融合。"
    items = []
    for split, count in (("train", 240), ("dev", 80)):
        for index in range(count):
            items.append({
                "query_id": f"{split}-{index}",
                "anchor_id": f"{split}-anchor-{index}",
                "split": split,
                "case_kind": "positive",
                "query_style": "natural",
                "question": "检索结果怎么合并？",
                "phenomena": ["natural_paraphrase"],
                "answer_requirements": ["解释融合"],
                "relevant_targets": [{
                    "chunk_id": f"chunk-{split}-{index}",
                    "source": "rag.md",
                    "heading_path": ["RRF"],
                    "relevance_grade": 3,
                    "supported_requirements": ["解释融合"],
                    "evidence_excerpt": excerpt,
                    "evidence_span_hash": evidence_span_hash(excerpt),
                }],
                "hard_negatives": [],
                "review_status": "approved",
                "review_note": "人工确认",
            })
    return {
        "schema_version": "rag-graded-qrels-v2",
        "dataset_id": "combined",
        "index": {
            "collection": "test",
            "count": 320,
            "fingerprint": "sha256:" + "a" * 64,
        },
        "items": items,
    }


def test_select_reviewed_split_is_exact_and_release_gated():
    payload = _public_split_payload()
    dev = select_reviewed_split(payload, "dev")
    assert len(dev["items"]) == 80
    assert {item["split"] for item in dev["items"]} == {"dev"}
    payload["items"][-1]["review_status"] = "draft"
    with pytest.raises(GradedQrelsValidationError, match="requires approved"):
        select_reviewed_split(payload, "dev")
    diagnostic = select_reviewed_split(payload, "dev", require_approved=False)
    assert diagnostic["status"] == "diagnostic_draft"
