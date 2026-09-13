# -*- coding: utf-8 -*-
"""Build and manage the V2-A colloquial anchor wave (120 new anchors × 4 styles).

Design contract (NEXT_RAG_OPTIMIZATION_GUIDE_20260826.md §3, handoff Phase D):

- every anchor is authored evidence-first from a real chunk of the frozen
  index; the builder verifies the excerpt is a verbatim slice and hashes it;
- V1 anchor_ids, V1 direct-evidence chunks, and V1 evidence source sections
  are excluded for V2-A *gold* evidence (hard negatives may reference any
  chunk — labels are per-question);
- splits are assigned by ``source::heading`` group so one knowledge point can
  never straddle Train / Dev-New / Sealed-New.  The sealed split reuses the
  ``blind`` split label so every existing guard (training exporter refuses
  blind, evaluation refuses writing blind results into the repository)
  protects Sealed-New without new code.  Sealed-New is an internal sealed
  set; it does not replace the original user-authored Blind80;
- the four styles are ``standard / natural / implicit_oral / long_noisy``.
  Fixed shells are rejected by a repetition check;
- users only edit the four question lines in review batches.  The wording
  parser reads nothing else, so evidence, splits, anchors, and negatives are
  immune to wording edits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from rag_colloquial_dataset import (
    _strict_target_map,
    _terms,
    select_anchor_questions,
)
from rag_qrels_v2 import (
    SCHEMA_VERSION,
    evidence_span_hash,
    index_contract_fingerprint,
    validate_graded_qrels,
)
from rag_qrels import normalize_answer_span


V2A_STYLES = ("standard", "natural", "implicit_oral", "long_noisy")
V2A_SPLIT_TARGETS = {"train": 80, "dev": 20, "blind": 20}
RETRIEVAL_EXCLUDED_SOURCE_TYPES = {
    "application", "application_jd", "experience", "jd", "log", "paper",
    "profile", "project_context", "resume", "resume_rule", "story", "system",
    "verification",
}
STYLE_PHENOMENA = {
    "standard": ["standard_expression", "technical_term_present"],
    "natural": ["query_distribution_shift", "natural_paraphrase"],
    "implicit_oral": ["query_distribution_shift", "implicit_or_colloquial"],
    "long_noisy": ["query_distribution_shift", "long_background"],
}
_MAX_SHELL_REPEATS = 3
# 20 normalized chars: long enough that a shared technical term plus a particle
# ("Decoder-Only架构的…") does not trip the check, short enough that pasting an
# evidence sentence into a question still does.
_LEAK_SPAN = 20
_DEFAULT_EXCERPT_LIMIT = 700


class V2AValidationError(ValueError):
    """Raised when an anchor spec violates the V2-A construction contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V2AValidationError(message)


def collection_rows(collection: Any) -> dict[str, dict[str, Any]]:
    """Snapshot curated rows keyed by chunk_id (personal scope excluded)."""

    snapshot = collection.get(include=["documents", "metadatas"])
    rows: dict[str, dict[str, Any]] = {}
    for chunk_id, document, metadata in zip(
        snapshot.get("ids") or [],
        snapshot.get("documents") or [],
        snapshot.get("metadatas") or [],
    ):
        meta = metadata or {}
        if meta.get("owner_scope") == "personal":
            continue
        source_type = str(meta.get("source_type") or "")
        rows[str(chunk_id)] = {
            "chunk_id": str(chunk_id),
            "document": str(document or ""),
            "source": str(meta.get("source") or ""),
            "heading": str(meta.get("heading_path") or meta.get("section_path")
                           or meta.get("title") or "未标注章节"),
            "source_type": source_type,
            "retrievable": source_type not in RETRIEVAL_EXCLUDED_SOURCE_TYPES,
        }
    _require(bool(rows), "collection contains no curated rows")
    return rows


def source_group(source: str, heading: str | list[str]) -> str:
    text = " > ".join(heading) if isinstance(heading, list) else str(heading)
    return f"{source}::{text}"


def compute_v1_exclusions(
    root: Path, rows: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """Chunks and source groups any of the 80 V1 anchors could touch.

    Public Train/Dev targets come from the repository file.  The remaining
    (sealed) V1 anchors are covered without opening any private file: their
    questions live in the public bench/final sets, so their strict-qrels
    targets and deterministic lexical proposals are recomputed from public
    inputs only, and everything is excluded uniformly.
    """

    public = json.loads(
        (root / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json")
        .read_text(encoding="utf-8")
    )
    chunks: set[str] = set()
    groups: set[str] = set()
    hard_negative_chunks: set[str] = set()
    public_anchor_ids: set[str] = set()
    for item in public["items"]:
        if item["case_kind"] == "positive":
            public_anchor_ids.add(item["anchor_id"])
        for target in item.get("relevant_targets") or []:
            chunks.add(target["chunk_id"])
            groups.add(source_group(target["source"], target.get("heading_path") or []))
        for negative in item.get("hard_negatives") or []:
            hard_negative_chunks.add(negative["chunk_id"])

    strict_targets, _ = _strict_target_map(root)
    remaining = [
        anchor for anchor in select_anchor_questions(root)
        if anchor["id"] not in public_anchor_ids
    ]
    row_list = list(rows.values())
    row_terms: list[set[str]] | None = None
    for anchor in remaining:
        mapped = strict_targets.get(anchor["id"], [])
        for target in mapped:
            chunks.add(target["chunk_id"])
            groups.add(source_group(target["source"], target.get("heading_path") or []))
        if not any(target.get("relevance") == "direct" for target in mapped):
            if row_terms is None:
                row_terms = [
                    _terms(f"{row['source']} {row['heading']} {row['document']}")
                    for row in row_list
                ]
            expected = [str(value).lower() for value in anchor.get("expect_sources") or []]
            indexed = [
                index for index, row in enumerate(row_list)
                if any(value in row["source"].lower() for value in expected)
            ] or list(range(len(row_list)))
            query_terms = _terms(anchor["q"])
            best = min(
                indexed,
                key=lambda index: (-len(query_terms & row_terms[index]),
                                   row_list[index]["chunk_id"]),
            )
            chunks.add(row_list[best]["chunk_id"])
            groups.add(source_group(row_list[best]["source"], row_list[best]["heading"]))
    return {
        "chunks": chunks,
        "groups": groups,
        "hard_negative_chunks": hard_negative_chunks,
        "anchor_ids": {anchor["id"] for anchor in select_anchor_questions(root)},
    }


def _default_excerpt(document: str, limit: int = _DEFAULT_EXCERPT_LIMIT) -> str:
    trimmed = document.strip()
    if len(trimmed) <= limit:
        return trimmed
    window = trimmed[:limit]
    for boundary in ("。", "！", "？", "；", "\n"):
        position = window.rfind(boundary)
        if position >= limit // 2:
            return window[:position + 1].strip()
    return window.strip()


def resolve_excerpt(spec_target: dict[str, Any], document: str) -> str:
    start_marker = str(spec_target.get("excerpt_start") or "")
    end_marker = str(spec_target.get("excerpt_end") or "")
    if not start_marker and not end_marker:
        excerpt = _default_excerpt(document)
    else:
        _require(bool(start_marker) and bool(end_marker),
                 f"{spec_target.get('chunk_id')}: excerpt markers must both be set")
        start = document.find(start_marker)
        _require(start >= 0,
                 f"{spec_target.get('chunk_id')}: excerpt_start not found verbatim")
        end = document.find(end_marker, start + len(start_marker))
        if end < 0 and document.startswith(end_marker, start):
            end = start
        _require(end >= 0,
                 f"{spec_target.get('chunk_id')}: excerpt_end not found after start")
        excerpt = document[start:end + len(end_marker)].strip()
    _require(bool(normalize_answer_span(excerpt)),
             f"{spec_target.get('chunk_id')}: excerpt is empty after normalization")
    _require(normalize_answer_span(excerpt) in normalize_answer_span(document),
             f"{spec_target.get('chunk_id')}: excerpt is not a verbatim slice")
    return excerpt


def _question_leaks_evidence(question: str, excerpts: Iterable[str]) -> bool:
    normalized_question = normalize_answer_span(question)
    for excerpt in excerpts:
        normalized = normalize_answer_span(excerpt)
        for start in range(0, max(1, len(normalized) - _LEAK_SPAN)):
            span = normalized[start:start + _LEAK_SPAN]
            if len(span) == _LEAK_SPAN and span in normalized_question:
                return True
    return False


def validate_anchor_specs(
    specs: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    exclusions: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Validate authored anchors and resolve evidence against the index."""

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    shell_prefixes: Counter = Counter()
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        anchor_id = str(spec.get("anchor_id") or "")
        _require(bool(re.fullmatch(r"v2a-[a-z]+-\d{3}", anchor_id)),
                 f"invalid anchor_id: {anchor_id!r}")
        _require(anchor_id not in seen_ids, f"duplicate anchor_id: {anchor_id}")
        _require(anchor_id not in exclusions["anchor_ids"],
                 f"{anchor_id}: collides with a V1 anchor id")
        seen_ids.add(anchor_id)
        domain = str(spec.get("domain") or "")
        _require(domain in {"llm_app", "backend", "algorithm", "career"},
                 f"{anchor_id}: invalid domain {domain!r}")
        requirements = spec.get("answer_requirements")
        _require(isinstance(requirements, list) and requirements
                 and all(isinstance(item, str) and item.strip() for item in requirements),
                 f"{anchor_id}: answer_requirements are required")

        gold_spec = spec.get("gold") or {}
        gold_id = str(gold_spec.get("chunk_id") or "")
        _require(gold_id in rows, f"{anchor_id}: gold chunk missing from index: {gold_id}")
        gold_row = rows[gold_id]
        _require(gold_row["retrievable"],
                 f"{anchor_id}: gold chunk source_type {gold_row['source_type']!r} "
                 "is not reachable by the reference_kb route")
        _require(gold_id not in exclusions["chunks"],
                 f"{anchor_id}: gold chunk reuses V1 evidence: {gold_id}")
        _require(gold_id not in exclusions["hard_negative_chunks"],
                 f"{anchor_id}: gold chunk was a V1 adjudicated hard negative: {gold_id}")
        group = source_group(gold_row["source"], gold_row["heading"])
        _require(group not in exclusions["groups"],
                 f"{anchor_id}: gold section reuses a V1 evidence section: {group}")
        gold_excerpt = resolve_excerpt(gold_spec, gold_row["document"])

        targets = [{
            "chunk_id": gold_id,
            "source": gold_row["source"],
            "heading_path": [part for part in gold_row["heading"].split(" > ") if part]
            or [gold_row["heading"]],
            "relevance_grade": 3,
            "supported_requirements": list(requirements),
            "evidence_excerpt": gold_excerpt,
            "evidence_span_hash": evidence_span_hash(gold_excerpt),
            "proposal_method": "v2a_authored_evidence_first",
        }]
        excerpts = [gold_excerpt]
        for grade2 in spec.get("grade2_targets") or []:
            chunk_id = str(grade2.get("chunk_id") or "")
            _require(chunk_id in rows,
                     f"{anchor_id}: grade2 chunk missing from index: {chunk_id}")
            _require(chunk_id != gold_id, f"{anchor_id}: grade2 duplicates gold")
            row = rows[chunk_id]
            excerpt = resolve_excerpt(grade2, row["document"])
            excerpts.append(excerpt)
            targets.append({
                "chunk_id": chunk_id,
                "source": row["source"],
                "heading_path": [part for part in row["heading"].split(" > ") if part]
                or [row["heading"]],
                "relevance_grade": 2,
                "supported_requirements": list(requirements),
                "evidence_excerpt": excerpt,
                "evidence_span_hash": evidence_span_hash(excerpt),
                "proposal_method": "v2a_authored_evidence_first",
            })

        negatives = []
        for negative in spec.get("hard_negatives") or []:
            chunk_id = str(negative.get("chunk_id") or "")
            reason = str(negative.get("reason") or "")
            _require(chunk_id in rows,
                     f"{anchor_id}: hard negative missing from index: {chunk_id}")
            _require(chunk_id not in {target["chunk_id"] for target in targets},
                     f"{anchor_id}: hard negative duplicates a target")
            _require(bool(reason.strip()),
                     f"{anchor_id}: hard negative needs an adjudication reason")
            negatives.append({
                "chunk_id": chunk_id,
                "source": rows[chunk_id]["source"],
                "reason": reason,
            })

        questions = spec.get("questions") or {}
        _require(set(questions) == set(V2A_STYLES),
                 f"{anchor_id}: questions must cover exactly {V2A_STYLES}")
        texts = [str(questions[style]).strip() for style in V2A_STYLES]
        _require(all(texts), f"{anchor_id}: questions must be non-empty")
        _require(len(set(texts)) == len(texts),
                 f"{anchor_id}: the four styles must be distinct")
        for style in V2A_STYLES:
            question = str(questions[style]).strip()
            _require(question not in seen_questions,
                     f"{anchor_id}: question duplicates {seen_questions.get(question)}")
            seen_questions[question] = anchor_id
            _require(not _question_leaks_evidence(question, excerpts),
                     f"{anchor_id}/{style}: question leaks a verbatim evidence span")
            if style in {"implicit_oral", "long_noisy"}:
                shell_prefixes[(style, question[:6])] += 1
        resolved.append({
            **spec,
            "anchor_id": anchor_id,
            "domain": domain,
            "source_group": group,
            "targets": targets,
            "resolved_negatives": negatives,
        })
    for (style, prefix), count in shell_prefixes.items():
        _require(count <= _MAX_SHELL_REPEATS,
                 f"shell prefix {prefix!r} repeats {count}× in {style}; "
                 "fixed wrappers are not allowed")
    return resolved


def _subset_groups(groups: list[tuple[str, int]], target: int) -> set[str]:
    states: dict[int, tuple[str, ...]] = {0: ()}
    for group, size in groups:
        updated = dict(states)
        for total, chosen in states.items():
            new_total = total + size
            if new_total <= target and new_total not in updated:
                updated[new_total] = (*chosen, group)
        states = updated
    if target not in states:
        raise V2AValidationError(
            f"source-group sizes cannot produce an exact {target}-anchor split"
        )
    return set(states[target])


def assign_v2a_splits(
    anchors: list[dict[str, Any]],
    targets: dict[str, int] | None = None,
) -> dict[str, str]:
    targets = dict(targets or V2A_SPLIT_TARGETS)
    _require(sum(targets.values()) == len(anchors),
             f"split targets {targets} do not sum to {len(anchors)} anchors")
    grouped: dict[str, list[str]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor["source_group"]].append(anchor["anchor_id"])
    ordered = sorted(
        ((group, len(ids)) for group, ids in grouped.items()),
        key=lambda pair: (hashlib.sha256(pair[0].encode("utf-8")).hexdigest(), pair[0]),
    )
    sealed = _subset_groups(ordered, targets["blind"])
    remaining = [pair for pair in ordered if pair[0] not in sealed]
    dev = _subset_groups(remaining, targets["dev"])
    assignments: dict[str, str] = {}
    for group, ids in grouped.items():
        split = "blind" if group in sealed else "dev" if group in dev else "train"
        assignments.update({anchor_id: split for anchor_id in ids})
    counts = Counter(assignments[anchor["anchor_id"]] for anchor in anchors)
    _require(dict(counts) == targets, f"split drift: {dict(counts)}")
    return assignments


def build_v2a_dataset(
    specs: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    exclusions: dict[str, set[str]],
    index: dict[str, Any],
    *,
    split_targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    anchors = validate_anchor_specs(specs, rows, exclusions)
    assignments = assign_v2a_splits(anchors, split_targets)
    items: list[dict[str, Any]] = []
    for anchor in anchors:
        split = assignments[anchor["anchor_id"]]
        for style in V2A_STYLES:
            question = str(anchor["questions"][style]).strip()
            items.append({
                "query_id": f"{anchor['anchor_id']}-{style}",
                "anchor_id": anchor["anchor_id"],
                "split": split,
                "case_kind": "positive",
                "domain": anchor["domain"],
                "query_style": style,
                "question": question,
                "phenomena": list(STYLE_PHENOMENA[style]),
                "answer_requirements": list(anchor["answer_requirements"]),
                "relevant_targets": [dict(target) for target in anchor["targets"]],
                "hard_negatives": [dict(negative) for negative in anchor["resolved_negatives"]],
                "review_status": "draft",
                "review_note": (
                    "V2-A：证据、答案要求与负例由系统在冻结索引上裁决完成；"
                    "问法等待用户口语措辞修改，导入并复验后转为 approved。"
                ),
                "human_review": {
                    "question_quality": "pending_user_wording",
                    "evidence_quality": "system_adjudicated",
                    "edited_question": "",
                    "comment": "",
                },
                "adjudication": {
                    "method": "in_session_evidence_first_manual_adjudication",
                    "topic": str(anchor.get("topic") or ""),
                },
            })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "rag-colloquial-v2a-480-v1-draft",
        "status": "draft_for_user_wording",
        "sealed_note": (
            "split=blind 的行是 V2-A 内部密封集（Sealed-New），复用 blind 守卫；"
            "它不是最初的用户独立 Blind80，也不得冒充独立作者盲集。"
        ),
        "index": {
            "collection": index["collection"],
            "count": int(index["collection_count"]),
            "fingerprint": index_contract_fingerprint(index),
            "fingerprint_id": str(index.get("fingerprint_id") or ""),
        },
        "design": {
            "anchors": len(anchors),
            "styles_per_anchor": len(V2A_STYLES),
            "rows": len(items),
            "split_anchor_targets": dict(split_targets or V2A_SPLIT_TARGETS),
            "sealed_policy": "private_file_only; repository stores sha256 manifest",
        },
        "items": sorted(items, key=lambda item: (item["split"], item["query_id"])),
    }
    validate_graded_qrels(payload)
    return payload


def split_v2a_public_sealed(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    public = {**payload, "dataset_id": "rag-colloquial-v2a-train-dev-v1-draft"}
    public["items"] = [item for item in payload["items"] if item["split"] != "blind"]
    sealed = {**payload, "dataset_id": "rag-colloquial-v2a-sealed-v1-draft"}
    sealed["items"] = [item for item in payload["items"] if item["split"] == "blind"]
    validate_graded_qrels(public, allowed_splits={"train", "dev"})
    validate_graded_qrels(sealed, allowed_splits={"blind"})
    return public, sealed


_SPLIT_DISPLAY = {"train": "train", "dev": "dev-new", "blind": "sealed-new"}
_WORDING_LABELS = {
    "standard": "用户问题 standard（只改本行冒号后的文字）",
    "natural": "用户问题 natural（只改本行冒号后的文字）",
    "implicit_oral": "用户问题 implicit_oral（建议重点修改）",
    "long_noisy": "用户问题 long_noisy（明显不自然才改）",
}


def v2a_review_batches(
    payload: dict[str, Any], batch_size: int = 30,
) -> list[str]:
    """Render review batches with only the four question lines editable."""

    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload["items"]:
        by_anchor[item["anchor_id"]].append(item)
    ordered = sorted(by_anchor, key=lambda anchor_id: hashlib.sha256(
        anchor_id.encode("utf-8")
    ).hexdigest())
    batches: list[str] = []
    total = (len(ordered) + batch_size - 1) // batch_size
    for batch_index in range(total):
        chunk = ordered[batch_index * batch_size:(batch_index + 1) * batch_size]
        lines = [
            f"# OfferClaw V2-A 口语问法用户修订 · 第 {batch_index + 1}/{total} 批",
            "",
            "> 证据、答案要点、split 与负例已由系统裁决冻结。你只需要修改每个锚点的",
            "> 四行“用户问题”，把不像真人会问的表达改自然；其中 implicit_oral 最需要",
            "> 你的口语直觉，long_noisy 只在明显不自然时调整。其他行请不要改动。",
            "",
        ]
        for position, anchor_id in enumerate(chunk, start=1):
            items = {item["query_style"]: item for item in by_anchor[anchor_id]}
            sample = items["standard"]
            target = sample["relevant_targets"][0]
            excerpt = " ".join(target["evidence_excerpt"].split())[:200]
            lines.extend([
                f"## {position}. `{anchor_id}` · {_SPLIT_DISPLAY[sample['split']]} · {sample['domain']}",
                "",
                f"- 知识点（只读）：{sample['adjudication'].get('topic', '')}",
                f"- 直接证据（只读）：`{target['source']}` / `{target['chunk_id']}`",
                f"- 证据片段（只读）：{excerpt}",
                f"- 答案要点（只读）：{'；'.join(sample['answer_requirements'])}",
            ])
            for style in V2A_STYLES:
                lines.append(
                    f"- {_WORDING_LABELS[style]}：{items[style]['question']}"
                )
            lines.append("")
        batches.append("\n".join(lines).rstrip() + "\n")
    return batches


def parse_v2a_wording(paths: Iterable[Path]) -> dict[str, str]:
    """Read only the four user question lines; everything else is ignored."""

    questions: dict[str, str] = {}
    block_pattern = re.compile(
        r"^##\s+\d+\.\s+`(?P<anchor_id>v2a-[a-z]+-\d{3})`.*?(?=^##\s+\d+\.|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for match in block_pattern.finditer(text):
            anchor_id = match.group("anchor_id")
            block = match.group(0)
            for style in V2A_STYLES:
                line = re.search(
                    rf"^- 用户问题 {style}（[^）]*）\s*[：:]\s*(.+)$",
                    block, re.MULTILINE,
                )
                _require(line is not None,
                         f"{anchor_id}/{style}: user question line is missing")
                question = line.group(1).strip()
                _require(bool(question), f"{anchor_id}/{style}: question is empty")
                query_id = f"{anchor_id}-{style}"
                _require(query_id not in questions,
                         f"duplicate wording entry: {query_id}")
                questions[query_id] = question
    return questions


def apply_v2a_wording(
    payload: dict[str, Any], questions: dict[str, str],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply wording-only edits and promote rows to approved after revalidation."""

    known = {item["query_id"] for item in payload["items"]}
    _require(set(questions) == known,
             "wording coverage mismatch; "
             f"missing={sorted(known - set(questions))[:5]}, "
             f"extra={sorted(set(questions) - known)[:5]}")
    changed = 0
    for item in payload["items"]:
        question = questions[item["query_id"]].strip()
        excerpts = [target["evidence_excerpt"] for target in item["relevant_targets"]]
        _require(not _question_leaks_evidence(question, excerpts),
                 f"{item['query_id']}: edited question leaks evidence text")
        changed += int(question != item["question"])
        item["question"] = question
        item["review_status"] = "approved"
        item["human_review"] = {"wording_edit": question}
    validate_graded_qrels(payload, require_approved=True)
    return payload, {"changed": changed, "unchanged": len(payload["items"]) - changed}


def v2a_summary(payload: dict[str, Any]) -> dict[str, Any]:
    split = Counter(item["split"] for item in payload["items"])
    domain = Counter(item["domain"] for item in payload["items"])
    style = Counter(item["query_style"] for item in payload["items"])
    review = Counter(item["review_status"] for item in payload["items"])
    sources = Counter(
        item["relevant_targets"][0]["source"] for item in payload["items"]
    )
    anchors_by_split: dict[str, set[str]] = defaultdict(set)
    for item in payload["items"]:
        anchors_by_split[item["split"]].add(item["anchor_id"])
    return {
        "rows": len(payload["items"]),
        "anchors": len({item["anchor_id"] for item in payload["items"]}),
        "by_split_rows": dict(sorted(split.items())),
        "by_split_anchors": {key: len(value) for key, value in sorted(anchors_by_split.items())},
        "by_domain_rows": dict(sorted(domain.items())),
        "by_style_rows": dict(sorted(style.items())),
        "by_review_status": dict(sorted(review.items())),
        "gold_sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
    }


__all__ = [
    "V2A_STYLES",
    "V2A_SPLIT_TARGETS",
    "V2AValidationError",
    "collection_rows",
    "compute_v1_exclusions",
    "resolve_excerpt",
    "validate_anchor_specs",
    "assign_v2a_splits",
    "build_v2a_dataset",
    "split_v2a_public_sealed",
    "v2a_review_batches",
    "parse_v2a_wording",
    "apply_v2a_wording",
    "v2a_summary",
]
