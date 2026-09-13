#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the reviewed Dev80 qrels overlay into V1 under the V1 review contract.

The V1 gold was produced by ``adjudicate_colloquial_gold.py``: a structured
selection pass, an *independent* entailment-verification pass, and exact-span
validation against the frozen index.  A correction to that gold has to clear
the same bar, and in particular it must be judged by a reviewer that never saw
the proposer's reasoning — which is exactly what the proposer cannot supply.

Two independent passes run here:

``verification``
    The V1 verifier's own predicates — ``directly_answerable`` and
    ``requirement_fully_supported`` — over (question, answer_requirement,
    candidate chunk).  The prompt never says the chunk is a proposed gold.

``coverage``
    An adversarially framed second opinion that must enumerate which parts of
    the requirement the chunk covers and which it does not, and return one of
    ``fully_covers`` / ``partially_covers`` / ``does_not_cover``.

Decision (both passes must agree; disagreement means "not merged"):

- a grade-3 addition needs both verification predicates true *and*
  ``fully_covers``;
- a hard-negative retraction to grade 2 needs ``partially_covers`` — the chunk
  supports part of the requirement, so it is not a valid negative, but it does
  not answer the question either;
- ``does_not_cover`` upholds the original negative label.

Unlike V1 there is deliberately no repair loop: V1 could narrow a question to
fit its chosen evidence, but narrowing a *frozen* requirement so that a
proposed chunk qualifies would be fitting the gold to the candidate.  A failed
proposal is reported and dropped.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_colloquial_v2a import resolve_excerpt  # noqa: E402
from rag_gold_review import (  # noqa: E402
    call_json as _call_json,
    coverage_prompt as _coverage_prompt,
    verification_prompt as _verification_prompt,
)
from rag_qrels_v2 import (  # noqa: E402
    evidence_span_hash,
    validate_graded_qrels,
    validate_graded_qrels_against_collection,
)


PUBLIC_PATH = ROOT / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json"
DEV_EXPORT_PATH = ROOT / "docs/rag_eval/colloquial/rag_colloquial_dev80_v1.json"
DEFAULT_OVERLAY = ROOT / "docs/rag_eval/colloquial/dev80_qrels_review_overlay_20260826.json"
DEFAULT_TRAIL = ROOT / "docs/rag_eval/colloquial/DEV80_QRELS_MERGE_TRAIL_20260826.json"
REVISION_ID = "v1.1-dev-multigold-20260826"
PROPOSAL_METHOD = "dev80_review_overlay_plus_independent_dual_verification"


def _canonical(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def _decide(entry: dict[str, Any], verification: dict[str, Any],
            coverage: dict[str, Any]) -> dict[str, Any]:
    verdict = str(coverage.get("verdict") or "")
    supported = (verification.get("directly_answerable") is True
                 and verification.get("requirement_fully_supported") is True)
    if entry["kind"] == "addition":
        applied = supported and verdict == "fully_covers"
        return {
            "applied": applied,
            "action": "add_grade3_target" if applied else "rejected",
            "rationale": (
                "两轮独立复核一致判定该块可完整回答问题且完全覆盖答案要求。" if applied
                else "独立复核未同时确认可直接回答与完全覆盖，按契约不并入。"
            ),
        }
    # retraction: the claim is "not a valid hard negative, but not gold either"
    if verdict == "partially_covers" and not supported:
        return {
            "applied": True,
            "action": "retract_negative_add_grade2_target",
            "rationale": "独立复核确认该块覆盖答案要求的一部分但不足以回答，"
                         "既不能当负例也不能当等价金标。",
        }
    if verdict == "fully_covers" and supported:
        return {
            "applied": False,
            "action": "rejected_should_be_grade3",
            "rationale": "独立复核判定该块可完整回答，与提案的 grade 2 定级冲突，"
                         "不按 grade 2 并入，需重新提案。",
        }
    return {
        "applied": False,
        "action": "rejected_negative_upheld",
        "rationale": "独立复核判定该块不覆盖答案要求，维持原 hard negative 裁决。",
    }


def _proposals(overlay: dict[str, Any], payload: dict[str, Any],
               documents: dict[str, str]) -> list[dict[str, Any]]:
    standards = {
        item["anchor_id"]: item for item in payload["items"]
        if item["case_kind"] == "positive" and item["query_style"] == "standard"
    }
    rows: list[dict[str, Any]] = []
    for kind in ("additions", "retractions"):
        for entry in overlay.get(kind, []):
            anchor = standards.get(entry["anchor_id"])
            if anchor is None:
                raise SystemExit(f"{entry['anchor_id']}: anchor is not in the public split")
            document = documents.get(entry["chunk_id"])
            if document is None:
                raise SystemExit(f"{entry['chunk_id']}: chunk absent from the frozen index")
            rows.append({
                "case_id": f"{entry['anchor_id']}::{entry['chunk_id']}",
                "kind": "addition" if kind == "additions" else "retraction",
                "anchor_id": entry["anchor_id"],
                "chunk_id": entry["chunk_id"],
                "source": entry["source"],
                "heading_path": entry["heading_path"],
                "relevance_grade": int(entry["relevance_grade"]),
                "question": anchor["question"],
                "answer_requirement": anchor["answer_requirements"][0],
                "answer_requirements": list(anchor["answer_requirements"]),
                "document": document,
                "excerpt": resolve_excerpt(entry, document),
                "proposer_verdict": entry.get("verdict", ""),
                "proposer_reason": entry.get("reason", ""),
            })
    return rows


def _apply(payload: dict[str, Any], row: dict[str, Any],
           decision: dict[str, Any], coverage: dict[str, Any]) -> int:
    touched = 0
    for item in payload["items"]:
        if item["anchor_id"] != row["anchor_id"] or item["case_kind"] != "positive":
            continue
        if any(target["chunk_id"] == row["chunk_id"]
               for target in item["relevant_targets"]):
            continue
        item["hard_negatives"] = [
            negative for negative in item["hard_negatives"]
            if negative["chunk_id"] != row["chunk_id"]
        ]
        target = {
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "heading_path": list(row["heading_path"]),
            "relevance_grade": row["relevance_grade"],
            # The schema has no field for partial requirement coverage, so a
            # grade-2 target still lists the requirement; ``partial_support``
            # and the verifier's gaps record what is actually missing.
            "supported_requirements": list(row["answer_requirements"]),
            "evidence_excerpt": row["excerpt"],
            "evidence_span_hash": evidence_span_hash(row["excerpt"]),
            "proposal_method": PROPOSAL_METHOD,
        }
        if row["relevance_grade"] < 3:
            target["partial_support"] = True
            target["uncovered_requirement_points"] = list(
                coverage.get("missing_points") or []
            )
        item["relevant_targets"].append(target)
        item.setdefault("gold_revisions", []).append({
            "revision": REVISION_ID,
            "action": decision["action"],
            "chunk_id": row["chunk_id"],
            "rationale": decision["rationale"],
        })
        touched += 1
    return touched


def run(args: argparse.Namespace) -> int:
    import chromadb
    from rag_colloquial_dataset import select_reviewed_split
    from rag_tools import get_collection_name

    overlay_path = Path(args.overlay).expanduser().resolve()
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    payload = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    validate_graded_qrels(payload, require_approved=True,
                          allowed_splits={"train", "dev"})
    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    chunk_ids = sorted({
        entry["chunk_id"] for key in ("additions", "retractions")
        for entry in overlay.get(key, [])
    })
    snapshot = collection.get(ids=chunk_ids, include=["documents"])
    documents = {
        str(chunk_id): str(document or "")
        for chunk_id, document in zip(snapshot.get("ids") or [],
                                      snapshot.get("documents") or [])
    }
    rows = _proposals(overlay, payload, documents)
    print(f"[merge] {len(rows)} proposals -> independent verification", flush=True)

    verification = {
        result["case_id"]: result
        for result in _call_json(_verification_prompt(rows), max_tokens=2400)["items"]
    }
    coverage = {
        result["case_id"]: result
        for result in _call_json(_coverage_prompt(rows), max_tokens=2600)["items"]
    }
    missing = [row["case_id"] for row in rows
               if row["case_id"] not in verification or row["case_id"] not in coverage]
    if missing:
        raise SystemExit(f"independent review has incomplete coverage: {missing}")

    trail: list[dict[str, Any]] = []
    applied_rows = 0
    for row in rows:
        decision = _decide(row, verification[row["case_id"]], coverage[row["case_id"]])
        touched = (_apply(payload, row, decision, coverage[row["case_id"]])
                   if decision["applied"] else 0)
        applied_rows += touched
        trail.append({
            "case_id": row["case_id"],
            "kind": row["kind"],
            "anchor_id": row["anchor_id"],
            "chunk_id": row["chunk_id"],
            "proposed_grade": row["relevance_grade"],
            "proposer_verdict": row["proposer_verdict"],
            "verification": verification[row["case_id"]],
            "coverage": coverage[row["case_id"]],
            "decision": decision,
            "rows_touched": touched,
            "evidence_span_hash": evidence_span_hash(row["excerpt"]),
        })
        print(f"[merge] {row['case_id']}: {decision['action']} ({touched} rows)",
              flush=True)

    payload.setdefault("revisions", []).append({
        "revision": REVISION_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_overlay": overlay_path.name,
        "process": "v1_contract_two_independent_passes_plus_exact_span_validation",
        "note": "Dev80 多金标补录与假负例撤回；Train split 未受影响。",
        "applied_rows": applied_rows,
    })
    validate_graded_qrels(payload, require_approved=True,
                          allowed_splits={"train", "dev"})
    validate_graded_qrels_against_collection(payload, collection)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "applied_rows": applied_rows,
                          "trail": trail}, ensure_ascii=False, indent=2))
        return 0

    PUBLIC_PATH.write_bytes(_canonical(payload))
    dev = select_reviewed_split(payload, "dev", require_approved=True)
    DEV_EXPORT_PATH.write_bytes(_canonical(dev))
    trail_path = Path(args.trail).expanduser().resolve()
    trail_path.write_bytes(_canonical({
        "schema_version": "colloquial-dev80-qrels-merge-trail-v1",
        "revision": REVISION_ID,
        "process": "v1_contract_two_independent_passes_plus_exact_span_validation",
        "overlay": overlay_path.name,
        "overlay_sha256": "sha256:" + hashlib.sha256(
            overlay_path.read_bytes()).hexdigest(),
        "public_sha256": "sha256:" + hashlib.sha256(
            PUBLIC_PATH.read_bytes()).hexdigest(),
        "dev_export_sha256": "sha256:" + hashlib.sha256(
            DEV_EXPORT_PATH.read_bytes()).hexdigest(),
        "cases": trail,
    }))
    print(json.dumps({
        "public": str(PUBLIC_PATH),
        "dev_export": str(DEV_EXPORT_PATH),
        "trail": str(trail_path),
        "applied_rows": applied_rows,
        "actions": {row["case_id"]: row["decision"]["action"] for row in trail},
    }, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    result.add_argument("--trail", default=str(DEFAULT_TRAIL))
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
