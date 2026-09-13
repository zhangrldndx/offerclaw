#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adjudicate colloquial gold evidence before handing wording to the user.

This is intentionally an offline, resumable authoring tool.  It never uses
retrieval output as a label.  Candidate chunks come from the anchor's declared
source, GPT selects a self-sufficient passage (narrowing an over-broad question
when necessary), and a second independent prompt verifies entailment.  The
sealed-blind payload and all intermediate prompts stay outside the repository.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_colloquial_dataset import (  # noqa: E402
    _terms,
    _variants,
    review_batch_markdown,
    review_summary,
    split_public_private,
)
from rag_qrels_v2 import evidence_span_hash, validate_graded_qrels  # noqa: E402


PUBLIC_PATH = ROOT / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json"
MANIFEST_PATH = ROOT / "docs/rag_eval/colloquial/rag_colloquial_blind_v1.manifest.json"

# Human code review found two legacy anchors with the same wording and evidence.
# Keep one for progressive disclosure and make the other test the definition/
# prompt distinction that is explicitly stated in a different chunk.  This is
# a documented gold correction, not a retrieval-output-driven label change.
CURATED_DISTINCTNESS_OVERRIDES = {
    "agt05": {
        "final_question": "Agent Skills 是什么，它与模型内置知识或一次性 Prompt 有什么区别？",
        "answer_requirement": (
            "答案必须说明 Agent Skills 是给 AI agents 增加能力和专业知识的标准化开放文件格式，"
            "可通过统一目录结构加载任务能力，并且不是模型内置知识或一次性 Prompt。"
        ),
        "evidence_candidate_id": "c2",
        "hard_negative_candidate_id": "c1",
        "hard_negative_reason": "该块只是页面目录，只列出了相关章节标题，没有给出 Agent Skills 的定义及其与模型内置知识或一次性 Prompt 的区别。",
    },
}


def _canonical(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        raise ValueError("empty GPT adjudication response")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("GPT adjudication response contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("GPT adjudication response root must be an object")
    return value


def _load_combined(private_root: Path) -> tuple[dict[str, Any], Path]:
    public = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    blind_path = private_root / "rag_colloquial_blind_v1.json"
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    combined = {
        **public,
        "dataset_id": "rag-colloquial-400-v1-system-adjudication",
        "items": [*public["items"], *blind["items"]],
    }
    validate_graded_qrels(combined)
    return combined, blind_path


def _collection_rows(collection: Any) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    snapshot = collection.get(include=["documents", "metadatas"])
    by_id: dict[str, dict[str, Any]] = {}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_id, document, metadata in zip(
        snapshot.get("ids") or [],
        snapshot.get("documents") or [],
        snapshot.get("metadatas") or [],
    ):
        meta = metadata or {}
        row = {
            "chunk_id": str(chunk_id),
            "document": str(document or ""),
            "source": str(meta.get("source") or ""),
            "heading": str(meta.get("heading_path") or meta.get("section_path")
                           or meta.get("title") or "未标注章节"),
        }
        by_id[row["chunk_id"]] = row
        by_source[row["source"]].append(row)
    return by_id, by_source


def _lexical_score(question: str, row: dict[str, Any]) -> tuple[int, int, str]:
    query_terms = _terms(question)
    heading_terms = _terms(row["heading"])
    body_terms = _terms(row["document"])
    return (
        len(query_terms & heading_terms) * 4 + len(query_terms & body_terms),
        -abs(len(row["document"]) - 900),
        row["chunk_id"],
    )


def _candidate_pack(payload: dict[str, Any], collection: Any) -> dict[str, Any]:
    by_id, by_source = _collection_rows(collection)
    standards = {
        item["anchor_id"]: item
        for item in payload["items"]
        if item["case_kind"] == "positive" and item["query_style"] == "standard"
    }
    packs = []
    for anchor_id, item in sorted(standards.items()):
        declared_sources = list(dict.fromkeys(
            target["source"] for target in item["relevant_targets"]
        ))
        pool = [row for source in declared_sources for row in by_source.get(source, [])]
        if not pool:
            raise ValueError(f"{anchor_id}: declared source has no indexed chunks")
        ranked = sorted(
            pool,
            key=lambda row: _lexical_score(item["question"], row),
            reverse=True,
        )
        chosen_ids = [target["chunk_id"] for target in item["relevant_targets"]]
        candidates: list[dict[str, Any]] = []
        for chunk_id in [*chosen_ids, *(row["chunk_id"] for row in ranked[:11])]:
            row = by_id.get(chunk_id)
            if row and row not in candidates:
                candidates.append(row)
            if len(candidates) >= 8:
                break
        packs.append({
            "anchor_id": anchor_id,
            "domain": item.get("domain", ""),
            "original_question": item["question"],
            "candidates": [
                {
                    "candidate_id": f"c{index}",
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "heading": row["heading"],
                    "document": row["document"],
                }
                for index, row in enumerate(candidates, start=1)
            ],
        })
    if len(packs) != 80:
        raise ValueError(f"expected 80 positive anchors, found {len(packs)}")
    return {"schema_version": "colloquial-adjudication-candidates-v1", "items": packs}


def _selection_prompt(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = []
    for item in batch:
        compact.append({
            "anchor_id": item["anchor_id"],
            "original_question": item["original_question"],
            "candidates": [{
                **{key: candidate[key] for key in (
                    "candidate_id", "source", "heading"
                )},
                "document": candidate["document"][:2200],
            } for candidate in item["candidates"]],
        })
    return [
        {"role": "system", "content": (
            "你是严格的 RAG 金标编辑，不是检索系统。只能依据给出的候选原文。"
            "对每个 anchor 选择一个能够单独、完整回答问题的候选块。若原问题比候选证据"
            "更宽，必须把问题收窄为该块能完整回答、仍有学习价值的标准技术问题；不得用"
            "外部知识补全。answer_requirement 用一句可核验的话描述答案必须覆盖什么。"
            "再选择一个不同候选作为 hard negative，前提是它与主题接近但不足以回答；"
            "没有可靠 hard negative 时填 null。只输出 JSON。"
        )},
        {"role": "user", "content": json.dumps({
            "output_schema": {
                "items": [{
                    "anchor_id": "string",
                    "final_question": "string",
                    "answer_requirement": "string",
                    "evidence_candidate_id": "cN",
                    "hard_negative_candidate_id": "cN or null",
                    "hard_negative_reason": "string",
                }]
            },
            "rules": [
                "每个输入 anchor 必须恰好输出一次且保持 anchor_id",
                "final_question 必须仅凭 evidence candidate 独立回答",
                "不要因为原问题存在就强行保留过宽问法",
                "不得输出候选列表外的 ID",
            ],
            "anchors": compact,
        }, ensure_ascii=False)},
    ]


def _verification_prompt(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = [{
        "anchor_id": row["anchor_id"],
        "question": row["final_question"],
        "answer_requirement": row["answer_requirement"],
        "evidence": row["evidence"]["document"][:2600],
        "hard_negative": (
            row["hard_negative"]["document"][:1400]
            if row.get("hard_negative") else None
        ),
        "claimed_hard_negative_reason": (
            row["hard_negative"]["reason"] if row.get("hard_negative") else ""
        ),
    } for row in rows]
    return [
        {"role": "system", "content": (
            "你是独立的 RAG 金标复核员。逐项检查：问题是否能完全由 evidence 回答；"
            "answer_requirement 是否完整且未超出原文；若存在 hard_negative，它是否确实"
            "不足以回答。不得调用外部知识，不得因为主题相关就判通过。只输出 JSON。"
        )},
        {"role": "user", "content": json.dumps({
            "output_schema": {"items": [{
                "anchor_id": "string",
                "directly_answerable": True,
                "requirement_fully_supported": True,
                "hard_negative_insufficient": True,
                "reason": "string",
            }]},
            "items": compact,
        }, ensure_ascii=False)},
    ]


def _repair_prompt(rows: list[dict[str, Any]], failures: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    compact = [{
        "anchor_id": row["anchor_id"],
        "question": row["final_question"],
        "answer_requirement": row["answer_requirement"],
        "evidence": row["evidence"]["document"][:3000],
        "hard_negative": (
            row["hard_negative"]["document"][:1600]
            if row.get("hard_negative") else None
        ),
        "verifier_reason": failures[row["anchor_id"]].get("reason", ""),
    } for row in rows]
    return [
        {"role": "system", "content": (
            "你是 RAG 金标修订员。独立复核已指出问题。只能依据 evidence 修订 question"
            "和 answer_requirement，使其不比原文更强且可由单块完整回答。若 hard_negative"
            "实际也能回答，keep_hard_negative 必须为 false。不得换证据、不得用外部知识。"
            "只输出 JSON。"
        )},
        {"role": "user", "content": json.dumps({
            "output_schema": {"items": [{
                "anchor_id": "string",
                "final_question": "string",
                "answer_requirement": "string",
                "keep_hard_negative": False,
                "repair_reason": "string",
            }]},
            "items": compact,
        }, ensure_ascii=False)},
    ]


def _call_json(messages: list[dict[str, str]], *, max_tokens: int) -> dict[str, Any]:
    from rag_gate import _chat
    raw = _chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        extra_payload={
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        },
        request_timeout=180,
        max_retries=1,
    )
    return _json_object(raw)


def _select_all(packs: dict[str, Any], cache_path: Path, *, force: bool) -> dict[str, Any]:
    cache = ({"items": {}} if force or not cache_path.exists()
             else json.loads(cache_path.read_text(encoding="utf-8")))
    by_anchor = cache.setdefault("items", {})
    pending = [item for item in packs["items"] if item["anchor_id"] not in by_anchor]
    # Four anchors keep the prompt comfortably below the proxy's long-request
    # tail.  The previous eight-anchor batch completed 72 items but one request
    # exceeded 180 seconds; smaller batches preserve the same adjudication
    # contract and resume from the durable cache.
    for offset in range(0, len(pending), 4):
        batch = pending[offset:offset + 4]
        response = _call_json(_selection_prompt(batch), max_tokens=1900)
        outputs = response.get("items") or []
        if {row.get("anchor_id") for row in outputs} != {
            row["anchor_id"] for row in batch
        }:
            raise ValueError("selection response has incomplete anchor coverage")
        packs_by_anchor = {row["anchor_id"]: row for row in batch}
        for result in outputs:
            pack = packs_by_anchor[result["anchor_id"]]
            candidates = {row["candidate_id"]: row for row in pack["candidates"]}
            evidence = candidates.get(result.get("evidence_candidate_id"))
            if not evidence:
                raise ValueError(f"{result['anchor_id']}: invalid evidence candidate")
            negative_id = result.get("hard_negative_candidate_id")
            negative = candidates.get(negative_id) if negative_id else None
            if negative and negative["chunk_id"] == evidence["chunk_id"]:
                raise ValueError(f"{result['anchor_id']}: hard negative equals evidence")
            by_anchor[result["anchor_id"]] = {
                "anchor_id": result["anchor_id"],
                "final_question": str(result["final_question"]).strip(),
                "answer_requirement": str(result["answer_requirement"]).strip(),
                "evidence": evidence,
                "hard_negative": ({
                    **negative,
                    "reason": str(result.get("hard_negative_reason") or "").strip(),
                } if negative else None),
            }
        cache_path.write_bytes(_canonical(cache))
        print(f"[selection] {len(by_anchor)}/80", flush=True)
    return cache


def _apply_curated_distinctness(
    packs: dict[str, Any], selection: dict[str, Any], verification: dict[str, Any],
    selection_path: Path, verify_path: Path,
) -> None:
    packs_by_anchor = {row["anchor_id"]: row for row in packs["items"]}
    changed = False
    for anchor, override in CURATED_DISTINCTNESS_OVERRIDES.items():
        candidates = {
            row["candidate_id"]: row for row in packs_by_anchor[anchor]["candidates"]
        }
        evidence = candidates[override["evidence_candidate_id"]]
        negative = candidates[override["hard_negative_candidate_id"]]
        replacement = {
            "anchor_id": anchor,
            "final_question": override["final_question"],
            "answer_requirement": override["answer_requirement"],
            "evidence": evidence,
            "hard_negative": {
                **negative,
                "reason": override["hard_negative_reason"],
            },
            "curation_reason": "remove duplicate anchor while preserving declared source",
        }
        old = selection["items"].get(anchor)
        comparable = ("final_question", "answer_requirement", "evidence", "hard_negative")
        if not old or any(old.get(key) != replacement.get(key) for key in comparable):
            selection["items"][anchor] = replacement
            verification["items"].pop(anchor, None)
            changed = True
    if changed:
        selection_path.write_bytes(_canonical(selection))
        verify_path.write_bytes(_canonical(verification))


def _verification_failures(cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        anchor: result for anchor, result in cache["items"].items()
        if not (
            result.get("directly_answerable") is True
            and result.get("requirement_fully_supported") is True
            and result.get("hard_negative_insufficient") is True
        )
    }


def _verify_all(selection: dict[str, Any], verify_path: Path, *, force: bool) -> dict[str, Any]:
    cache = ({"items": {}} if force or not verify_path.exists()
             else json.loads(verify_path.read_text(encoding="utf-8")))
    verified = cache.setdefault("items", {})
    pending = [row for anchor, row in sorted(selection["items"].items())
               if anchor not in verified]
    for offset in range(0, len(pending), 6):
        batch = pending[offset:offset + 6]
        response = _call_json(_verification_prompt(batch), max_tokens=1500)
        outputs = response.get("items") or []
        if {row.get("anchor_id") for row in outputs} != {
            row["anchor_id"] for row in batch
        }:
            raise ValueError("verification response has incomplete anchor coverage")
        for result in outputs:
            verified[result["anchor_id"]] = result
        verify_path.write_bytes(_canonical(cache))
        print(f"[verification] {len(verified)}/80", flush=True)
    return cache


def _repair_and_reverify(
    selection: dict[str, Any], verification: dict[str, Any],
    selection_path: Path, verify_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures = _verification_failures(verification)
    if not failures:
        return selection, verification
    failed_rows = [selection["items"][anchor] for anchor in sorted(failures)]
    response = _call_json(_repair_prompt(failed_rows, failures), max_tokens=1400)
    outputs = response.get("items") or []
    if {row.get("anchor_id") for row in outputs} != set(failures):
        raise ValueError("repair response has incomplete failed-anchor coverage")
    for result in outputs:
        anchor = result["anchor_id"]
        row = selection["items"][anchor]
        row["final_question"] = str(result["final_question"]).strip()
        row["answer_requirement"] = str(result["answer_requirement"]).strip()
        if result.get("keep_hard_negative") is not True:
            row["hard_negative"] = None
        row["repair_reason"] = str(result.get("repair_reason") or "").strip()
        verification["items"].pop(anchor, None)
    selection_path.write_bytes(_canonical(selection))
    verify_path.write_bytes(_canonical(verification))
    verification = _verify_all(selection, verify_path, force=False)
    remaining = _verification_failures(verification)
    if remaining:
        detail = "; ".join(
            f"{anchor}: {row.get('reason', '')}" for anchor, row in remaining.items()
        )
        raise ValueError(f"evidence verification still fails after repair: {detail}")
    return selection, verification


def _negative_rationale(item: dict[str, Any]) -> tuple[str, str]:
    phenomenon = (item.get("phenomena") or [""])[0]
    if phenomenon == "out_of_domain":
        return "abstain_from_kb", "问题超出当前求职/技术知识库范围，不存在内部证据。"
    if phenomenon == "near_domain_missing":
        return "state_missing", "问题要求未保存的实时或私人事实，不能由现有文件推断。"
    if phenomenon == "wrong_relation":
        return "reject_unsupported_relation", "关键词可能存在，但题设关系没有知识库证据支持。"
    question = item["question"]
    if any(word in question for word in ("忽略", "假装", "无视", "覆盖", "API Key", "删除引用")):
        return "reject_instruction", "问题试图绕过证据、隐私或只读边界，必须拒绝。"
    return "clarify", "问题包含未解析指代或缺少必要对象，应先请求澄清。"


def _apply_gold(
    payload: dict[str, Any], selection: dict[str, Any], verification: dict[str, Any],
) -> dict[str, Any]:
    for item in payload["items"]:
        if item["case_kind"] == "negative":
            expected, rationale = _negative_rationale(item)
            item["expected_behavior"] = expected
            item["negative_rationale"] = rationale
            item["review_status"] = "approved"
            item["review_note"] = "系统按预注册负例类型完成有效性裁决。"
            item["human_review"] = {"wording_edit": ""}
            continue
        gold = selection["items"][item["anchor_id"]]
        check = verification["items"][item["anchor_id"]]
        variants = _variants(gold["final_question"], None, item["anchor_id"])
        item["question"] = variants[item["query_style"]]
        item["answer_requirements"] = [gold["answer_requirement"]]
        evidence = gold["evidence"]
        item["relevant_targets"] = [{
            "chunk_id": evidence["chunk_id"],
            "source": evidence["source"],
            "heading_path": [part.strip() for part in re.split(
                r"\s*(?:>|/|›)\s*", evidence["heading"]
            ) if part.strip()],
            "relevance_grade": 3,
            "supported_requirements": [gold["answer_requirement"]],
            "evidence_excerpt": evidence["document"],
            "evidence_span_hash": evidence_span_hash(evidence["document"]),
            "proposal_method": "gpt_selection_plus_independent_entailment_verification",
        }]
        negative = gold.get("hard_negative")
        item["hard_negatives"] = ([{
            "chunk_id": negative["chunk_id"],
            "source": negative["source"],
            "reason": negative["reason"],
        }] if negative else [])
        item["review_status"] = "approved"
        item["review_note"] = "问题、答案要求和直接证据已通过独立蕴含复核。"
        item["adjudication"] = {
            "method": "two_pass_structured_gpt_plus_exact_span_validation",
            "verification_reason": check.get("reason", ""),
        }
        item["human_review"] = {"wording_edit": ""}
    payload["status"] = "system_gold_ready_for_user_wording_edits"
    validate_graded_qrels(payload, require_approved=True)
    return payload


def run(args: argparse.Namespace) -> None:
    import chromadb
    from rag_tools import get_collection_name

    private_root = Path(args.private_root or "~/.offerclaw/private_eval").expanduser().resolve()
    private_root.mkdir(parents=True, exist_ok=True)
    payload, blind_path = _load_combined(private_root)
    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name()
    )
    candidate_path = private_root / "rag_colloquial_adjudication_candidates_v1.json"
    if args.force or not candidate_path.exists():
        candidate_path.write_bytes(_canonical(_candidate_pack(payload, collection)))
    packs = json.loads(candidate_path.read_text(encoding="utf-8"))
    selection_path = private_root / "rag_colloquial_adjudication_selection_v1.json"
    verify_path = private_root / "rag_colloquial_adjudication_verification_v1.json"
    selection = _select_all(packs, selection_path, force=args.force)
    verification = ({"items": {}} if args.force or not verify_path.exists()
                    else json.loads(verify_path.read_text(encoding="utf-8")))
    _apply_curated_distinctness(
        packs, selection, verification, selection_path, verify_path
    )
    verification = _verify_all(selection, verify_path, force=False)
    selection, verification = _repair_and_reverify(
        selection, verification, selection_path, verify_path
    )
    combined = _apply_gold(payload, selection, verification)
    public, blind = split_public_private(combined)
    public["status"] = "system_gold_ready_for_user_wording_edits"
    blind["status"] = "system_gold_ready_for_user_wording_edits"
    public_bytes = _canonical(public)
    blind_bytes = _canonical(blind)
    PUBLIC_PATH.write_bytes(public_bytes)
    blind_path.write_bytes(blind_bytes)

    all_items = sorted(combined["items"], key=lambda item: hashlib.sha256(
        item["query_id"].encode("utf-8")
    ).hexdigest())
    batch_dir = private_root / "rag_colloquial_review_batches_v1"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(8):
        batch = all_items[batch_index * 50:(batch_index + 1) * 50]
        (batch_dir / f"batch_{batch_index + 1:02d}_of_08.md").write_text(
            review_batch_markdown(batch, batch_index + 1), encoding="utf-8"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["dataset_id"] = blind["dataset_id"]
    manifest["sha256"] = hashlib.sha256(blind_bytes).hexdigest()
    manifest["status"] = "approved_sealed_blind_pending_wording_edits"
    MANIFEST_PATH.write_bytes(_canonical(manifest))
    print(json.dumps({
        "summary": review_summary(combined),
        "review_batches": str(batch_dir),
        "selection_cache": str(selection_path),
        "verification_cache": str(verify_path),
    }, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--private-root")
    result.add_argument("--force", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
