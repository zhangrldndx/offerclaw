import copy

import pytest

from rag_qrels_v2 import (
    GradedQrelsValidationError,
    evidence_span_hash,
    index_contract_fingerprint,
    validate_graded_qrels,
    validate_graded_qrels_against_collection,
)


def _payload():
    excerpt = "RRF 使用多个通道的名次做倒数排名融合。"
    return {
        "schema_version": "rag-graded-qrels-v2",
        "dataset_id": "test-v2",
        "index": {
            "collection": "test_collection",
            "count": 1,
            "fingerprint": "sha256:" + "a" * 64,
        },
        "items": [{
            "query_id": "q1",
            "anchor_id": "a1",
            "split": "dev",
            "case_kind": "positive",
            "query_style": "colloquial",
            "question": "好几个搜索结果按名次怎么合？",
            "phenomena": ["implicit_term"],
            "answer_requirements": ["解释 RRF 的融合机制"],
            "relevant_targets": [{
                "chunk_id": "chunk-1",
                "source": "rag.md",
                "heading_path": ["RRF"],
                "relevance_grade": 3,
                "supported_requirements": ["解释 RRF 的融合机制"],
                "evidence_excerpt": excerpt,
                "evidence_span_hash": evidence_span_hash(excerpt),
            }],
            "hard_negatives": [{"chunk_id": "chunk-2", "reason": "同主题错误章节"}],
            "review_status": "approved",
            "review_note": "人工确认",
        }],
    }


class _Collection:
    name = "test_collection"

    def get(self, ids=None, include=None):
        return {
            "ids": ["chunk-1"],
            "documents": ["前文。RRF 使用多个通道的名次做倒数排名融合。后文。"],
            "metadatas": [{"source": "rag.md"}],
        }


def test_v2_accepts_graded_multi_requirement_contract_and_live_evidence():
    payload = _payload()
    assert validate_graded_qrels(payload, require_approved=True) is payload
    assert validate_graded_qrels_against_collection(payload, _Collection()) is payload


def test_v2_release_gate_rejects_drafts():
    payload = _payload()
    payload["items"][0]["review_status"] = "draft"
    with pytest.raises(GradedQrelsValidationError, match="requires approved"):
        validate_graded_qrels(payload, require_approved=True)


def test_v2_requires_every_answer_requirement_to_have_evidence():
    payload = _payload()
    payload["items"][0]["answer_requirements"].append("说明适用边界")
    with pytest.raises(GradedQrelsValidationError, match="every answer requirement"):
        validate_graded_qrels(payload)


def test_v2_negative_cannot_invent_relevant_target():
    payload = _payload()
    item = payload["items"][0]
    item["case_kind"] = "negative"
    item["query_style"] = "negative"
    item["answer_requirements"] = []
    with pytest.raises(GradedQrelsValidationError, match="must not invent"):
        validate_graded_qrels(payload)


def test_index_contract_fingerprint_ignores_timestamp_and_git_commit():
    left = {
        "collection": "c", "collection_count": 3,
        "collection_content_hash": "abc", "embedding_model": "bge",
        "chunker_version": "v1", "generated_at": "now", "git_commit": "dirty",
    }
    right = copy.deepcopy(left)
    right.update(generated_at="later", git_commit="different")
    assert index_contract_fingerprint(left) == index_contract_fingerprint(right)

