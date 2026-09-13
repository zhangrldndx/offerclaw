# -*- coding: utf-8 -*-
"""Contract tests for the V2-A anchor wave builder and wording pipeline."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_colloquial_v2a import (  # noqa: E402
    V2AValidationError,
    V2A_STYLES,
    apply_v2a_wording,
    assign_v2a_splits,
    build_v2a_dataset,
    parse_v2a_wording,
    resolve_excerpt,
    split_v2a_public_sealed,
    v2a_review_batches,
    validate_anchor_specs,
)


def _rows(count=12):
    rows = {}
    for index in range(count):
        chunk_id = f"doc{index:02d}_chunk"
        rows[chunk_id] = {
            "chunk_id": chunk_id,
            "document": (
                f"知识点{index}的正文。它的核心机制是第{index}号机制，"
                f"由部件甲和部件乙协作完成，并在失败时回退到部件丙。补充说明第{index}段。"
            ),
            "source": f"doc{index:02d}.md",
            "heading": "正文",
            "source_type": "feishu_wiki",
            "retrievable": True,
        }
    return rows


def _exclusions():
    return {
        "chunks": {"v1_gold_chunk"},
        "groups": {"v1_doc.md::正文"},
        "hard_negative_chunks": {"v1_negative_chunk"},
        "anchor_ids": {"alg01", "be02"},
    }


_ORAL_FORMS = [
    "就那个{index}号的，靠啥配合跑起来的来着？",
    "{index}号那玩意到底谁跟谁搭伙干活？",
    "卡在{index}号这了，里面是哪几块在配合？",
    "那{index}号机制呢，部件之间咋分工的？",
    "有点懵，{index}号是靠哪些件凑一起转的？",
    "别笑我，{index}号里头到底谁配合谁？",
]
_NOISY_FORMS = [
    "这两天在复盘项目顺便补笔记，看到知识点{index}卡住了，就想搞清楚第{index}号机制谁跟谁配合。",
    "面试前夜有点慌，翻到第{index}节没看懂，先别管别的，第{index}号机制的协作方式是什么？",
    "白天开了一天会脑子很乱，回来接着看资料，第{index}号机制里部件是怎么配合的来着？",
    "笔记抄了一半发现断片了，前文可以忽略，只想确认知识点{index}那个机制的协作关系。",
    "同事随口提了一句第{index}号机制我没接上话，回家恶补下：它内部怎么协作？",
    "背八股背串了，别的先放放，帮我掰扯清楚第{index}号机制是哪些部件在协作。",
]


def _spec(index, *, domain="backend", anchor_id=None, negative_index=None):
    negative_index = (index + 1) % 12 if negative_index is None else negative_index
    return {
        "anchor_id": anchor_id or f"v2a-be-{index:03d}",
        "domain": domain,
        "topic": f"知识点{index}",
        "gold": {"chunk_id": f"doc{index:02d}_chunk"},
        "answer_requirements": [f"答案必须说明第{index}号机制由部件甲和部件乙协作完成。"],
        "hard_negatives": [{
            "chunk_id": f"doc{negative_index:02d}_chunk",
            "reason": "邻近主题但不含本锚点机制，复核确认不足以回答。",
        }],
        "questions": {
            "standard": f"第{index}号机制的组成部件有哪些？",
            "natural": f"帮我看下知识点{index}里那个机制是靠什么协作的？",
            "implicit_oral": _ORAL_FORMS[index % len(_ORAL_FORMS)].format(index=index),
            "long_noisy": _NOISY_FORMS[index % len(_NOISY_FORMS)].format(index=index),
        },
    }


@pytest.fixture()
def small_wave():
    rows = _rows(12)
    specs = [_spec(index) for index in range(12)]
    index = {
        "collection": "test_collection",
        "collection_count": 12,
        "collection_content_hash": "abc",
        "embedding_provider": "local",
        "embedding_model": "test",
        "embedding_dimensions": 8,
        "chunker_version": "2026-08-10",
        "fingerprint_id": "deadbeef",
    }
    targets = {"train": 8, "dev": 2, "blind": 2}
    return rows, specs, index, targets


def test_resolve_excerpt_defaults_and_markers():
    document = "第一句。第二句很关键。第三句结束。"
    assert resolve_excerpt({"chunk_id": "x"}, document) == document
    excerpt = resolve_excerpt(
        {"chunk_id": "x", "excerpt_start": "第二句", "excerpt_end": "关键。"},
        document,
    )
    assert excerpt == "第二句很关键。"
    with pytest.raises(V2AValidationError, match="excerpt_start not found"):
        resolve_excerpt({"chunk_id": "x", "excerpt_start": "不存在", "excerpt_end": "。"}, document)


def test_validate_rejects_v1_gold_reuse_and_sections():
    rows = _rows(2)
    rows["v1_gold_chunk"] = {**rows["doc00_chunk"], "chunk_id": "v1_gold_chunk"}
    spec = _spec(0)
    spec["gold"]["chunk_id"] = "v1_gold_chunk"
    with pytest.raises(V2AValidationError, match="reuses V1 evidence"):
        validate_anchor_specs([spec], rows, _exclusions())
    spec = _spec(0)
    rows["doc00_chunk"]["source"] = "v1_doc.md"
    with pytest.raises(V2AValidationError, match="reuses a V1 evidence section"):
        validate_anchor_specs([spec], rows, _exclusions())


def test_validate_rejects_unreachable_source_type():
    rows = _rows(2)
    rows["doc00_chunk"]["source_type"] = "log"
    rows["doc00_chunk"]["retrievable"] = False
    with pytest.raises(V2AValidationError, match="not reachable"):
        validate_anchor_specs([_spec(0)], rows, _exclusions())


def test_validate_rejects_question_evidence_leakage():
    rows = _rows(2)
    spec = _spec(0)
    spec["questions"]["standard"] = rows["doc00_chunk"]["document"][:40]
    with pytest.raises(V2AValidationError, match="leaks a verbatim evidence span"):
        validate_anchor_specs([spec], rows, _exclusions())


def test_validate_rejects_fixed_shells():
    rows = _rows(9)
    specs = []
    for index in range(8):
        spec = _spec(index)
        spec["questions"]["implicit_oral"] = f"请问一下，第{index}号机制到底靠什么协作？"
        specs.append(spec)
    with pytest.raises(V2AValidationError, match="fixed wrappers"):
        validate_anchor_specs(specs, rows, _exclusions())


def test_split_assignment_is_group_stable(small_wave):
    rows, specs, _index, targets = small_wave
    anchors = validate_anchor_specs(specs, rows, _exclusions())
    assignments = assign_v2a_splits(anchors, targets)
    by_group = {}
    for anchor in anchors:
        by_group.setdefault(anchor["source_group"], set()).add(
            assignments[anchor["anchor_id"]]
        )
    assert all(len(splits) == 1 for splits in by_group.values())
    counts = {"train": 0, "dev": 0, "blind": 0}
    for split in assignments.values():
        counts[split] += 1
    assert counts == targets
    # determinism
    assert assignments == assign_v2a_splits(anchors, targets)


def test_build_and_split_public_sealed(small_wave):
    rows, specs, index, targets = small_wave
    payload = build_v2a_dataset(specs, rows, _exclusions(), index, split_targets=targets)
    assert len(payload["items"]) == 12 * len(V2A_STYLES)
    public, sealed = split_v2a_public_sealed(payload)
    assert {item["split"] for item in public["items"]} == {"train", "dev"}
    assert {item["split"] for item in sealed["items"]} == {"blind"}
    assert len(sealed["items"]) == targets["blind"] * len(V2A_STYLES)
    # idempotency: same inputs -> identical bytes
    payload2 = build_v2a_dataset(specs, rows, _exclusions(), index, split_targets=targets)
    assert json.dumps(payload, ensure_ascii=False, sort_keys=True) == \
        json.dumps(payload2, ensure_ascii=False, sort_keys=True)


def test_review_batches_and_wording_roundtrip(tmp_path, small_wave):
    rows, specs, index, targets = small_wave
    payload = build_v2a_dataset(specs, rows, _exclusions(), index, split_targets=targets)
    batches = v2a_review_batches(payload, batch_size=5)
    paths = []
    for number, content in enumerate(batches, start=1):
        path = tmp_path / f"batch_{number:02d}.md"
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    questions = parse_v2a_wording(paths)
    assert set(questions) == {item["query_id"] for item in payload["items"]}

    # user edits one implicit_oral line; nothing else changes
    target_qid = sorted(questions)[0]
    edited = questions[target_qid] + "，说人话版"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if questions[target_qid] in text:
            path.write_text(
                text.replace(questions[target_qid], edited, 1), encoding="utf-8",
            )
            break
    reparsed = parse_v2a_wording(paths)
    assert reparsed[target_qid] == edited

    before = {
        item["query_id"]: json.dumps(
            {key: value for key, value in item.items()
             if key not in {"question", "review_status", "human_review"}},
            ensure_ascii=False, sort_keys=True,
        )
        for item in payload["items"]
    }
    updated, counts = apply_v2a_wording(payload, reparsed)
    assert counts["changed"] == 1
    for item in updated["items"]:
        assert item["review_status"] == "approved"
        frozen = json.dumps(
            {key: value for key, value in item.items()
             if key not in {"question", "review_status", "human_review"}},
            ensure_ascii=False, sort_keys=True,
        )
        # evidence, split, anchors, negatives are untouched by wording edits
        assert frozen == before[item["query_id"]]


def test_wording_rejects_evidence_pasted_into_question(small_wave):
    rows, specs, index, targets = small_wave
    payload = build_v2a_dataset(specs, rows, _exclusions(), index, split_targets=targets)
    questions = {item["query_id"]: item["question"] for item in payload["items"]}
    victim = payload["items"][0]
    questions[victim["query_id"]] = victim["relevant_targets"][0]["evidence_excerpt"][:40]
    with pytest.raises(V2AValidationError, match="leaks evidence"):
        apply_v2a_wording(payload, questions)
