# -*- coding: utf-8 -*-
"""Independent review passes shared by every gold/negative adjudication tool.

The V1 authoring contract (``scripts/adjudicate_colloquial_gold.py``) never let
the party that proposed a label be the party that confirmed it: a structured
selection pass proposed, a separate prompt verified entailment, and exact-span
validation closed the loop.  Every later tool that promotes a chunk to gold or
demotes one to a hard negative reuses those same two passes from here, so the
independence property is a property of the codebase rather than of whoever
happened to write the calling script.

``verification_prompt``
    The V1 verifier's own predicates over (question, answer_requirement,
    evidence).  It is never told what the caller hopes the answer will be.

``coverage_prompt``
    An adversarially framed second opinion that must enumerate the covered and
    missing requirement points and commit to ``fully_covers`` /
    ``partially_covers`` / ``does_not_cover``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


VERDICTS = ("fully_covers", "partially_covers", "does_not_cover")


def json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        raise ValueError("empty adjudication response")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("adjudication response contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("adjudication response root must be an object")
    return value


def call_json(messages: list[dict[str, str]], *, max_tokens: int,
              request_timeout: int = 180) -> dict[str, Any]:
    from rag_gate import _chat

    raw = _chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        extra_payload={
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        },
        request_timeout=request_timeout,
        max_retries=1,
    )
    return json_object(raw)


def verification_prompt(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    compact = [{
        "case_id": row["case_id"],
        "question": row["question"],
        "answer_requirement": row["answer_requirement"],
        "evidence": row["document"][:2600],
    } for row in rows]
    return [
        {"role": "system", "content": (
            "你是独立的 RAG 金标复核员。逐项检查：问题是否能完全由 evidence 回答；"
            "answer_requirement 是否被 evidence 完整支持且未超出原文。不得调用外部知识，"
            "不得因为主题相关就判通过。只输出 JSON。"
        )},
        {"role": "user", "content": json.dumps({
            "output_schema": {"items": [{
                "case_id": "string",
                "directly_answerable": True,
                "requirement_fully_supported": True,
                "reason": "string",
            }]},
            "items": compact,
        }, ensure_ascii=False)},
    ]


def coverage_prompt(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    compact = [{
        "case_id": row["case_id"],
        "answer_requirement": row["answer_requirement"],
        "evidence": row["document"][:2600],
    } for row in rows]
    return [
        {"role": "system", "content": (
            "你是挑剔的证据审计员，默认立场是证据不足。把 answer_requirement 拆成若干"
            "要点，逐点判断 evidence 是否明确给出该要点；只要有一个要点缺失就不能判定为"
            "完全覆盖。covered_points 与 missing_points 都必须引用 evidence 或指出缺什么，"
            "不得用外部知识补全。verdict 取 fully_covers / partially_covers / "
            "does_not_cover。只输出 JSON。"
        )},
        {"role": "user", "content": json.dumps({
            "output_schema": {"items": [{
                "case_id": "string",
                "covered_points": ["string"],
                "missing_points": ["string"],
                "verdict": " | ".join(VERDICTS),
            }]},
            "items": compact,
        }, ensure_ascii=False)},
    ]


def review_batches(rows: list[dict[str, Any]], *, batch_size: int,
                   max_tokens: int, progress: bool = True) -> dict[str, dict[str, Any]]:
    """Run both passes over ``rows`` and return ``case_id -> merged verdict``.

    A long review is two model calls per batch with nothing to show for it in
    between, so progress is reported by default; a silent multi-minute run is
    indistinguishable from a hung one.
    """

    import sys as _sys

    merged: dict[str, dict[str, Any]] = {}
    total = -(-len(rows) // batch_size)
    for index, offset in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[offset:offset + batch_size]
        verification = {
            result["case_id"]: result
            for result in call_json(verification_prompt(batch),
                                    max_tokens=max_tokens).get("items", [])
        }
        coverage = {
            result["case_id"]: result
            for result in call_json(coverage_prompt(batch),
                                    max_tokens=max_tokens).get("items", [])
        }
        missing = [row["case_id"] for row in batch
                   if row["case_id"] not in verification
                   or row["case_id"] not in coverage]
        if missing:
            raise ValueError(f"independent review missed cases: {missing}")
        for row in batch:
            merged[row["case_id"]] = {
                "verification": verification[row["case_id"]],
                "coverage": coverage[row["case_id"]],
            }
        if progress:
            print(f"[review] batch {index}/{total} ({len(merged)}/{len(rows)} cases)",
                  file=_sys.stderr, flush=True)
    return merged


__all__ = [
    "VERDICTS",
    "call_json",
    "coverage_prompt",
    "json_object",
    "review_batches",
    "verification_prompt",
]
