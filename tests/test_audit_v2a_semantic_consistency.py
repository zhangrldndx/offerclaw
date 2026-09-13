# -*- coding: utf-8 -*-
"""Tests for the V2-A question / requirement / evidence consistency audit."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_v2a_semantic_consistency import (  # noqa: E402
    audit,
    coverage,
    is_content_term,
    requirement_clauses,
    terms,
)


def _item(anchor_id, style, question, *, requirement, gold_id, topic,
          negatives=(), split="train"):
    return {
        "query_id": f"{anchor_id}-{style}",
        "anchor_id": anchor_id,
        "split": split,
        "case_kind": "positive",
        "domain": "backend",
        "query_style": style,
        "question": question,
        "phenomena": ["standard_expression"],
        "answer_requirements": [requirement],
        "relevant_targets": [{
            "chunk_id": gold_id,
            "source": "doc.md",
            "heading_path": ["正文"],
            "relevance_grade": 3,
            "supported_requirements": [requirement],
            "evidence_excerpt": EXCERPTS[gold_id],
            "evidence_span_hash": "sha256:" + "0" * 64,
        }],
        "hard_negatives": [
            {"chunk_id": chunk_id, "source": "doc.md", "reason": "复核确认不足以回答。"}
            for chunk_id in negatives
        ],
        "review_status": "draft",
        "adjudication": {"topic": topic},
    }


EXCERPTS = {
    "gold_lock": "页面锁的开销、加锁速度、锁粒度、并发度都介于表级锁和行级锁之间，会出现死锁。",
    "gold_bloom": "布隆过滤器通过 K 个哈希函数把元素映射到位数组的多个位置并置一，任一位置为零则元素一定不存在。",
}
DOCUMENTS = {
    "gold_lock": EXCERPTS["gold_lock"] + " 表级锁开销小加锁快，行级锁并发度高适合并发写。",
    "gold_bloom": EXCERPTS["gold_bloom"] + " 由于位共享，多个元素可能覆盖相同的位，因此可能误报。",
    # The heading digest that *is* the answer — the be02 / ca-015 failure shape.
    "toc_lock": "页面锁的开销、加锁速度、锁粒度、并发度都介于表级锁和行级锁之间",
    "unrelated": "本节介绍如何配置开发环境与安装依赖。",
}
# An over-claim the audit can see: it reaches for terms the corpus attests in a
# *different* chunk (the real v2a-alg-004 shape).  An over-claim invented out of
# words no chunk uses is invisible to a lexical check — a documented blind spot.
CROSS_CHUNK_OVERCLAIM = "答案必须说明布隆过滤器的位数组结构，并说明它与行级锁的并发度权衡。"


def _payload(items):
    return {"dataset_id": "test-wave", "items": items}


def test_terms_and_content_term_filtering():
    values = terms("虚拟内存 HTTP 1.0")
    assert "虚拟" in values and "拟内" in values and "内存" in values
    assert "http" in values
    # sliding bigrams straddle word boundaries; glue-bearing ones are not handles
    assert is_content_term("虚拟内") is True
    assert is_content_term("构是") is False
    assert is_content_term("是什") is False
    assert is_content_term("bm25") is True


def test_coverage_restricted_to_corpus_vocabulary():
    haystack = terms("页面锁介于表级锁和行级锁之间")
    vocabulary = terms("页面锁 表级锁 行级锁")
    # Without a vocabulary my own connectives drag the score down; with one,
    # only the domain words are judged.
    assert coverage("页面锁介于两者之间", haystack) < 1.0
    assert coverage("页面锁介于两者之间", haystack, vocabulary=vocabulary) == 1.0


def test_requirement_clauses_keep_brackets_intact():
    clauses = requirement_clauses(
        "答案必须说明短期记忆是会话态上下文（最近几轮问答，高频写入），长期记忆跨会话有效。"
    )
    assert any("（最近几轮问答，高频写入）" in clause for clause in clauses)
    assert all("）" not in clause or "（" in clause for clause in clauses)


def test_requirement_grounding_separates_narrow_excerpt_from_overclaim():
    grounded = _item("v2a-be-900", "standard", "页面锁的开销和并发度介于哪两种锁之间？",
                     requirement="答案必须说明页面锁的开销与并发度介于表级锁和行级锁之间。",
                     gold_id="gold_lock", topic="锁粒度对比")
    # A clause about the *other* sentence of the chunk: grounded in the chunk,
    # absent from the excerpt -> the markers are too narrow, not an over-claim.
    narrow = _item("v2a-be-901", "standard", "表级锁和行级锁在开销上怎么取舍？",
                   requirement="答案必须说明表级锁开销小加锁快、行级锁并发度高适合并发写。",
                   gold_id="gold_lock", topic="锁开销")
    # A clause reaching for a concept that lives in another chunk -> over-claim.
    overclaim = _item("v2a-be-902", "standard", "布隆过滤器的位数组是怎么回事？",
                      requirement=CROSS_CHUNK_OVERCLAIM,
                      gold_id="gold_bloom", topic="布隆过滤器结构")
    report = audit(_payload([grounded, narrow, overclaim]), DOCUMENTS)
    verdicts = {
        finding["anchor_id"]: finding["verdict"]
        for finding in report["findings"] if finding["check"] == "requirement_grounding"
    }
    assert "v2a-be-900" not in verdicts
    assert verdicts["v2a-be-901"] == "excerpt_too_narrow"
    assert verdicts["v2a-be-902"] == "ungrounded"


def test_hard_negative_sanity_flags_a_heading_digest_that_answers():
    item = _item("v2a-be-903", "standard", "页面锁的开销和并发度介于哪两种锁之间？",
                 requirement="答案必须说明页面锁的开销、并发度介于表级锁和行级锁之间。",
                 gold_id="gold_lock", topic="锁粒度对比", negatives=["toc_lock"])
    report = audit(_payload([item]), DOCUMENTS)
    flagged = [finding for finding in report["findings"]
               if finding["check"] == "hard_negative_sanity"]
    assert [finding["chunk_id"] for finding in flagged] == ["toc_lock"]
    assert flagged[0]["ratio"] >= 0.75


def test_question_anchoring_severity_depends_on_style():
    shared = {
        "requirement": "答案必须说明布隆过滤器用 K 个哈希函数把元素映射到位数组的多个位置。",
        "gold_id": "gold_bloom",
        "topic": "布隆过滤器原理",
    }
    items = [
        _item("v2a-be-904", "standard", "那个东西到底是怎么回事？", **shared),
        _item("v2a-be-904", "implicit_oral", "那玩意咋整的来着？", **shared),
    ]
    report = audit(_payload(items), DOCUMENTS)
    severities = {
        finding["query_style"]: finding["severity"]
        for finding in report["findings"] if finding["check"] == "question_anchoring"
    }
    assert severities["standard"] == "fail"
    assert severities["implicit_oral"] == "warn"


def test_adjudication_overlay_acknowledges_and_reports_stale_entries():
    overclaim = _item("v2a-be-905", "standard", "布隆过滤器的位数组是怎么回事？",
                      requirement=CROSS_CHUNK_OVERCLAIM,
                      gold_id="gold_bloom", topic="布隆过滤器结构")
    payload = _payload([overclaim])
    before = audit(payload, DOCUMENTS)
    assert before["summary"]["blocking"] == 1

    adjudication = {"entries": [
        {"check": "requirement_grounding", "anchor_id": "v2a-be-905",
         "clause": "并说明它与行级锁的并发度权衡",
         "verdict": "paraphrase_wording_not_overclaim", "reason": "已人工复核。"},
        {"check": "requirement_grounding", "anchor_id": "v2a-be-999",
         "clause": "不存在的条目", "verdict": "x", "reason": "stale"},
    ]}
    after = audit(payload, DOCUMENTS, adjudication)
    assert after["summary"]["blocking"] == 0
    assert after["summary"]["acknowledged"] == 1
    # A stale adjudication must be visible rather than silently ignored.
    assert after["summary"]["unmatched_adjudications"] == [
        "requirement_grounding|v2a-be-999|不存在的条目"
    ]


def test_frozen_wave_has_no_blocking_findings():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "docs/rag_eval/colloquial/V2A_SEMANTIC_CONSISTENCY_20260826.json"
    if not report_path.exists():
        pytest.skip("V2-A audit report unavailable")
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["blocking"] == 0
    assert report["summary"]["unmatched_adjudications"] == []
