# -*- coding: utf-8 -*-
"""Requirement-coverage judging: the model observes, the program decides.

Stage 1 of the next-stage guide replaces the single 0-3 grade with a structured
contract.  The measured failure it targets: on Final v4, 7 of 12 stable in-pool
misses were an adjacent chunk from the gold's own document winning while
covering the question's requirements at 0.08-0.77 against the gold's 0.79-1.0
-- and in 6 of 7 the early exit meant the gold was never judged at all.  A
single ordinal grade cannot express "contains *which parts* of the answer", so
two partially-covering chunks tie at 3 and the topical reranker keeps the
thinner one.

Contract shape (authored per question, human-confirmed; requirements describe
what the QUESTION asks, never quoting the gold -- gold-derived
``answer_requirements`` stay on the scoring side only):

    question_contract:
      requirements: [{id, description, required}]
      fact_type: normal | numeric | comparison | procedural
      required_slots: [subject, value, unit]     # numeric only
      referent_required: false

The judge emits per-chunk *observations* only:

    chunk_judgment:
      covered_requirement_ids: [r1]
      explicit_slot_values: {value: "..."}
      self_contained: true
      unresolved_reference: false

Everything decision-shaped -- coverage counts, full coverage, the action, the
minimal evidence set, the early exit -- is computed HERE, in code, auditable
and threshold-free.  The model is never asked whether to answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable


FACT_TYPES = ("normal", "numeric", "comparison", "procedural")

OBSERVATION_SCHEMA = "requirement-coverage-v1"

_PROMPT = """你是检索证据的观察员。给定【问题的原子要求清单】和一个【资料片段】，
只报告客观观察，不做任何"是否该回答"的判断。

严格规则：
1. 只看片段本身，不用你自己的知识补全。
2. 一条要求只有在片段**本身的文字**里被明确满足时才算覆盖；只沾边不算。
3. explicit_slot_values 只填片段里**逐字出现**的值；片段没写就不填。
4. self_contained：这段文字不依赖所在文档其他部分就能被读懂且支撑所覆盖的要求。
5. unresolved_reference：问题里的指代在问题文本内没有对象时为 true。
6. 只输出一个 JSON 对象，不要输出其他字段或解释。

输出格式：
{{"covered_requirement_ids": ["r1"], "explicit_slot_values": {{}}, "self_contained": true, "unresolved_reference": false}}

【问题】
{question}

【问题的原子要求清单】
{requirements}

【资料片段】
{chunk}"""
PROMPT_SHA256 = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()


def validate_contract(contract: dict) -> dict:
    requirements = contract.get("requirements") or []
    if not requirements:
        raise ValueError("contract needs at least one requirement")
    seen = set()
    for req in requirements:
        rid = req.get("id")
        if not rid or rid in seen:
            raise ValueError(f"bad requirement id: {rid!r}")
        seen.add(rid)
        if not (req.get("description") or "").strip():
            raise ValueError(f"requirement {rid} needs a description")
    fact_type = contract.get("fact_type", "normal")
    if fact_type not in FACT_TYPES:
        raise ValueError(f"unknown fact_type: {fact_type!r}")
    if fact_type == "numeric" and not contract.get("required_slots"):
        raise ValueError("numeric contract needs required_slots")
    return contract


def parse_observation(text: str | None, contract: dict) -> dict | None:
    """Parse one observation; unusable output is *unavailable*, never zeros.

    Same contract as ``parse_grade``: coercing a failure to "covers nothing"
    would make an outage look like thin evidence.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    ids = raw.get("covered_requirement_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return None
    known = {r["id"] for r in contract["requirements"]}
    if not set(ids) <= known:
        return None                      # invented requirement ids are a contract breach
    slots = raw.get("explicit_slot_values")
    if not isinstance(slots, dict):
        return None
    if not isinstance(raw.get("self_contained"), bool):
        return None
    if not isinstance(raw.get("unresolved_reference"), bool):
        return None
    return {"covered_requirement_ids": sorted(set(ids)),
            "explicit_slot_values": {str(k): str(v) for k, v in slots.items()},
            "self_contained": raw["self_contained"],
            "unresolved_reference": raw["unresolved_reference"]}


# --------------------------------------------------------------------------
# program-side decisions (no model involvement past this line)


def required_ids(contract: dict) -> set:
    return {r["id"] for r in contract["requirements"] if r.get("required", True)}


def coverage_count(observation: dict | None, contract: dict) -> int:
    if not observation:
        return 0
    return len(set(observation["covered_requirement_ids"]) & required_ids(contract))


def full_coverage(observation: dict | None, contract: dict) -> bool:
    if not observation:
        return False
    if not required_ids(contract) <= set(observation["covered_requirement_ids"]):
        return False
    if contract.get("fact_type") == "numeric":
        # A numeric question is only answered by a chunk that carries the
        # actual values: Final v4's one stable true false-accept was exactly
        # "asks a number, chunk discusses the topic without it" (x3 runs).
        slots = observation.get("explicit_slot_values") or {}
        if not all((slots.get(s) or "").strip() for s in contract.get("required_slots", [])):
            return False
    return True


def can_early_exit(observation: dict | None, contract: dict) -> bool:
    """The new early exit: only a fully self-sufficient incumbent stops judging.

    The old exit fired on grade 3 alone, which is how 6 of 7 same-source
    losses happened -- the fuller gold two ranks down was never looked at.
    """
    return bool(observation
                and full_coverage(observation, contract)
                and observation["self_contained"]
                and not observation["unresolved_reference"])


def rank_key(observation: dict | None, contract: dict, rerank_score: float):
    """Sort key: (full coverage, coverage count, reranker score) -- threshold-free."""
    return (
        1 if full_coverage(observation, contract) else 0,
        coverage_count(observation, contract),
        rerank_score if rerank_score is not None else 0.0,
    )


def minimal_evidence_set(observations: list, contract: dict,
                         max_chunks: int = 2) -> list[int]:
    """Greedy smallest set of chunk positions jointly covering all requirements.

    Returns positions into ``observations`` (post-rank order); empty when no
    combination of ``max_chunks`` covers everything -- the caller falls back to
    single-best, it never pads with uncovering chunks.
    """
    need = required_ids(contract)
    if not need:
        return []
    covers = [set(o["covered_requirement_ids"]) & need if o else set()
              for o in observations]
    best: list[int] = []
    for first in range(len(covers)):
        if covers[first] >= need:
            return [first]
    if max_chunks < 2:
        return best
    for first in range(len(covers)):
        if not covers[first]:
            continue
        for second in range(first + 1, len(covers)):
            if covers[first] | covers[second] >= need:
                return [first, second]
    return best


def unresolved_reference_vote(observations: list) -> bool:
    """A question-level property: any observation flagging it is enough."""
    return any(o and o["unresolved_reference"] for o in observations)
