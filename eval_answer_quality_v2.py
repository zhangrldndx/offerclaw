#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen, repository-safe answer-quality evaluation for OfferClaw.

This evaluator intentionally does not reuse :mod:`eval_rag_answer`.  The old
script samples the same-distribution benchmark, drops unsuccessful rows from
the denominator and judges a truncated context.  V2 instead:

* freezes 40 distinct, user-worded positive anchors from the two historical
  reranker training corpora (20 per corpus, 10 per domain) plus eight
  adjudicated negative controls;
* runs the production reference-KB retrieval and generation helpers exactly
  once per item and stores the *full* context supplied to the generator;
* sends the same immutable generation artifact to two neutral judges;
* derives claim faithfulness, requirement coverage and citation metrics from
  structured, auditable judge output;
* never writes questions, contexts, answers or judge reasons inside the
  repository.  Only a text-free selection manifest and aggregate summary may
  be published under ``docs/rag_eval/answer_quality_v2``.

The selected questions were visible during development.  This is therefore a
frozen, user-worded DEVELOPMENT REGRESSION, not organic traffic and not a new
blind release set.

Offline selection only (no model calls)::

    .venv/bin/python eval_answer_quality_v2.py freeze-selection

Live run (raw artifacts default to ~/.offerclaw/eval_runs)::

    .venv/bin/python eval_answer_quality_v2.py run

``run`` is deliberately explicit because the quality profile and the two
judges are expensive.  It checkpoints after every row so an interrupted run
still leaves an auditable partial artifact; a fresh run never silently reuses
that partial output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "docs" / "rag_eval" / "answer_quality_v2"
SELECTION_PATH = PUBLIC_DIR / "SELECTION_MANIFEST.json"
SUMMARY_PATH = PUBLIC_DIR / "SUMMARY.json"

POSITIVE_SOURCES = {
    "v1": ROOT / "docs/rag_eval/colloquial/rag_colloquial_train_dev_v1.json",
    "v2a": ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_train_dev_draft.json",
}
NEGATIVE_SOURCE = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_negatives.json"
ACTION_OVERLAY = ROOT / "docs/rag_eval/colloquial/guard/expected_action_overlay.json"

DOMAINS = ("algorithm", "backend", "career", "llm_app")
SOURCE_CAP = 3
POSITIVE_COUNT = 40
NEGATIVE_COUNT = 8
JUDGE_NAMES = ("deepseek-v3", "gpt-5.5")

# The quality package being measured is explicit even where the production
# module currently has equivalent defaults.  During a live run these exact
# values are installed temporarily, then the caller's environment is restored.
FROZEN_ANSWERABILITY_ENV = {
    "RAG_ANSWERABILITY_MODE": "teacher",
    "RAG_ANSWERABILITY_PROMPT": "v5",
    "RAG_ANSWERABILITY_DEPTH": "12",
    "RAG_ANSWERABILITY_EARLY_EXIT": "1",
    "RAG_ANSWERABILITY_TIEBREAK": "0",
    "RAG_ANSWERABILITY_GATE": "1",
    "RAG_ANSWERABILITY_GATE_VOTES": "3",
}

STYLE_BUCKET = {
    "standard": "standard",
    "natural": "natural",
    "colloquial": "oral",
    "implicit_oral": "oral",
    "long_context": "long",
    "long_noisy": "long",
}

# Each source contributes five items per domain.  Across four domains this is
# standard=2, natural=6, oral=6, long=6 per source, or 4/12/12/12 overall.
STYLE_SLOTS = {
    "algorithm": ("standard", "natural", "oral", "long", "long"),
    "backend": ("standard", "natural", "oral", "oral", "long"),
    "career": ("natural", "natural", "oral", "long", "long"),
    "llm_app": ("natural", "natural", "oral", "oral", "long"),
}

CORRECTABLE_IDS = (
    ("v1", "col-neg-053"),
    ("negative", "v2aneg-012"),
    ("negative", "v2aneg-016"),
    ("negative", "v2aneg-017"),
)
ABSTAIN_STRATA = (
    "out_of_domain",
    "near_domain_missing",
    "ambiguous_or_injection",
    "wrong_relation",
)

SCHEMA_VERSION = "answer-quality-v2"
SELECTION_SCHEMA = "answer-quality-v2-selection-v1"
RAW_SCHEMA = "answer-quality-v2-raw-v1"
SUMMARY_SCHEMA = "answer-quality-v2-summary-v1"


class AnswerQualityV2Error(RuntimeError):
    """A fail-closed data, lineage or runtime contract violation."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnswerQualityV2Error(f"{path.name}: JSON root must be an object")
    return value


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_entry_count": len(status.splitlines()) if status else 0,
    }


def _runtime_snapshot() -> dict[str, Any]:
    """Read the current local runtime contract without making model calls."""
    from day1_api_starter import load_local_env
    load_local_env()
    from rag_answerability import (PROMPT_SHA256, PROMPT_V5_SHA256, SCHEMA,
                                   active_prompt, enabled as answerability_enabled,
                                   mode as answerability_mode, resolve_model)
    from rag_gate import (_answerability_depth, _answerability_early_exit,
                          _answerability_gate, _answerability_gate_votes,
                          _answerability_tiebreak)
    from rag_retrieval_trace import resolve_retrieval_profile
    from rag_tools import get_collection_name, index_fingerprint

    fingerprint = index_fingerprint()
    profile = resolve_retrieval_profile(None)
    _active_template, active_prompt_sha256 = active_prompt()
    prompt_choice = os.environ.get("RAG_ANSWERABILITY_PROMPT", "").strip().lower()
    active_prompt_name = "v4" if prompt_choice == "v4" else "v5"
    effective_answerability = {
        "mode": answerability_mode(),
        "enabled": answerability_enabled(),
        "rerank_depth": _answerability_depth(),
        "early_exit": _answerability_early_exit(),
        "tiebreak": _answerability_tiebreak(),
        "gate_enabled": _answerability_gate(),
        "gate_votes": _answerability_gate_votes(),
        "active_prompt": active_prompt_name,
    }
    required_answerability = {
        "mode": "teacher",
        "enabled": True,
        "rerank_depth": 12,
        "early_exit": True,
        "tiebreak": False,
        "gate_enabled": True,
        "gate_votes": 3,
        "active_prompt": "v5",
    }
    if effective_answerability != required_answerability:
        raise AnswerQualityV2Error(
            "answer-quality v2 requires the frozen teacher/v5/depth12/"
            "early-exit/gate3 quality profile"
        )
    return {
        "retrieval_profile": profile.to_dict(),
        "index": {
            "collection": get_collection_name(),
            "fingerprint_id": fingerprint.get("fingerprint_id"),
            "collection_count": fingerprint.get("collection_count"),
            "collection_content_hash": fingerprint.get("collection_content_hash"),
            "embedding_model": fingerprint.get("embedding_model"),
            "chunker_version": fingerprint.get("chunker_version"),
        },
        "answerability": {
            "schema": SCHEMA,
            "requested_model": resolve_model(),
            **effective_answerability,
            "frozen_env": dict(FROZEN_ANSWERABILITY_ENV),
            "prompt_v4_sha256": PROMPT_SHA256,
            "prompt_v5_sha256": PROMPT_V5_SHA256,
            "active_prompt": active_prompt_name,
            "active_prompt_sha256": active_prompt_sha256,
        },
        "generation": {
            "model_env": os.environ.get("RAG_SYNTH_MODEL", ""),
            "prompt_contract_sha256": _sha256_file(ROOT / "rag_gate.py"),
        },
        "judges": list(JUDGE_NAMES),
        "judge_prompt_sha256": _sha256_text(JUDGE_SYSTEM),
    }


def _primary_source(item: dict[str, Any]) -> str:
    targets = list(item.get("relevant_targets") or [])
    if not targets:
        return ""
    return str(targets[0].get("source") or "")


def _approved_train_positives(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    output = []
    for item in payload.get("items") or []:
        if (item.get("split") != "train" or item.get("case_kind") != "positive"
                or item.get("review_status") != "approved"):
            continue
        question = str(item.get("question") or "")
        wording = str((item.get("human_review") or {}).get("wording_edit") or "")
        bucket = STYLE_BUCKET.get(str(item.get("query_style") or ""))
        if not question or question != wording or not bucket:
            raise AnswerQualityV2Error(
                f"{path.name}:{item.get('query_id')} is not an approved user wording"
            )
        if item.get("domain") not in DOMAINS:
            continue
        if not item.get("answer_requirements") or not item.get("relevant_targets"):
            raise AnswerQualityV2Error(
                f"{path.name}:{item.get('query_id')} lacks a positive gold contract"
            )
        copy = dict(item)
        copy["_style_bucket"] = bucket
        copy["_primary_source"] = _primary_source(item)
        output.append(copy)
    return output


def _rank_token(seed: str, *parts: str) -> str:
    return _sha256_text("\x1f".join((seed, *parts)))


def _positive_slots() -> list[tuple[str, str, str, int]]:
    slots = []
    for source_key in sorted(POSITIVE_SOURCES):
        for domain in DOMAINS:
            occurrences: Counter[str] = Counter()
            for style in STYLE_SLOTS[domain]:
                occurrences[style] += 1
                slots.append((source_key, domain, style, occurrences[style]))
    return slots


def _select_positives(seed: str,
                      corpora: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    slots = _positive_slots()
    candidates: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for slot in slots:
        source_key, domain, style, _occurrence = slot
        rows = [
            item for item in corpora[source_key]
            if item.get("domain") == domain and item["_style_bucket"] == style
        ]
        rows.sort(key=lambda item: _rank_token(
            seed, source_key, domain, style, str(item["anchor_id"]),
            str(item["query_id"])))
        candidates[slot] = rows
        if not rows:
            raise AnswerQualityV2Error(f"no candidates for slot {slot}")

    # Scarce buckets first; the hash token keeps ordering deterministic when
    # several slots have equal cardinality.
    ordered_slots = sorted(
        slots,
        key=lambda slot: (len(candidates[slot]), _rank_token(seed, *map(str, slot))),
    )
    chosen: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    used_anchors: set[str] = set()
    source_counts: Counter[str] = Counter()

    def viable_from(position: int) -> bool:
        for later in ordered_slots[position:]:
            if not any(
                str(row["anchor_id"]) not in used_anchors
                and source_counts[row["_primary_source"]] < SOURCE_CAP
                for row in candidates[later]
            ):
                return False
        return True

    def search(position: int) -> bool:
        if position == len(ordered_slots):
            return True
        slot = ordered_slots[position]
        for row in candidates[slot]:
            anchor = str(row["anchor_id"])
            source = row["_primary_source"]
            if anchor in used_anchors or not source or source_counts[source] >= SOURCE_CAP:
                continue
            chosen[slot] = row
            used_anchors.add(anchor)
            source_counts[source] += 1
            if viable_from(position + 1) and search(position + 1):
                return True
            source_counts[source] -= 1
            if not source_counts[source]:
                del source_counts[source]
            used_anchors.remove(anchor)
            del chosen[slot]
        return False

    if not search(0):
        raise AnswerQualityV2Error(
            "cannot satisfy positive quotas with a global source cap of 3"
        )
    selected = [chosen[slot] | {"_source_key": slot[0]} for slot in slots]
    if len(selected) != POSITIVE_COUNT:
        raise AnswerQualityV2Error("positive selection count drift")
    return selected


def _load_negative_maps() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    negative_payload = _read_json(NEGATIVE_SOURCE)
    negative = {
        str(item.get("query_id")): item for item in negative_payload.get("items") or []
    }
    v1 = _read_json(POSITIVE_SOURCES["v1"])
    v1_map = {str(item.get("query_id")): item for item in v1.get("items") or []}
    negative.update(v1_map)
    overlay = _read_json(ACTION_OVERLAY)
    return negative, overlay


def _select_negatives(seed: str) -> list[dict[str, Any]]:
    by_id, overlay = _load_negative_maps()
    adjudicated = dict(overlay.get("adjudicated") or {})
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for source_key, query_id in CORRECTABLE_IDS:
        row = by_id.get(query_id)
        decision = adjudicated.get(query_id)
        if not row or not decision or decision.get("expected_action") != "correct_premise":
            raise AnswerQualityV2Error(f"missing correctable adjudication for {query_id}")
        if row.get("review_status") != "approved":
            raise AnswerQualityV2Error(f"negative row is not approved: {query_id}")
        selected.append(dict(row) | {
            "_source_key": source_key,
            "_expected_action": "correct_premise",
            "_negative_stratum": "wrong_relation_correctable",
        })
        used.add(query_id)

    pending = set((overlay.get("proposals_pending_review") or {}).keys())
    for stratum in ABSTAIN_STRATA:
        candidates = [
            row for query_id, row in by_id.items()
            if query_id not in used and query_id not in pending
            and row.get("review_status") == "approved"
            and stratum in (row.get("phenomena") or [])
            and query_id not in adjudicated
            and query_id.startswith("v2aneg-")
        ]
        candidates.sort(key=lambda row: _rank_token(
            seed, "negative", stratum, str(row.get("query_id"))))
        if not candidates:
            raise AnswerQualityV2Error(f"no abstain candidate for {stratum}")
        row = candidates[0]
        selected.append(dict(row) | {
            "_source_key": "negative",
            "_expected_action": "abstain",
            "_negative_stratum": stratum,
        })
        used.add(str(row["query_id"]))

    if len(selected) != NEGATIVE_COUNT or len(used) != NEGATIVE_COUNT:
        raise AnswerQualityV2Error("negative selection count/uniqueness drift")
    return selected


def _text_hashes(item: dict[str, Any]) -> dict[str, str]:
    return {
        "question_sha256": _sha256_text(str(item.get("question") or "")),
        "requirements_sha256": _json_sha256(item.get("answer_requirements") or []),
        "targets_sha256": _json_sha256([
            {
                "chunk_id": target.get("chunk_id"),
                "evidence_span_hash": target.get("evidence_span_hash"),
                "relevance_grade": target.get("relevance_grade"),
            }
            for target in (item.get("relevant_targets") or [])
        ]),
    }


def _manifest_item(item: dict[str, Any], index: int, cohort: str) -> dict[str, Any]:
    primary_source = item.get("_primary_source") or _primary_source(item)
    return {
        "eval_id": f"aqv2-{cohort[:3]}-{index:03d}",
        "cohort": cohort,
        "source_key": item["_source_key"],
        "query_id": item["query_id"],
        "anchor_id": item["anchor_id"],
        "domain": item.get("domain", "negative"),
        "style_bucket": item.get("_style_bucket", "negative"),
        "expected_action": item.get("_expected_action", "answer"),
        "negative_stratum": item.get("_negative_stratum", ""),
        "primary_source_sha256": _sha256_text(primary_source) if primary_source else None,
        **_text_hashes(item),
    }


def _source_lineage() -> dict[str, dict[str, Any]]:
    paths = {**POSITIVE_SOURCES, "negative": NEGATIVE_SOURCE, "overlay": ACTION_OVERLAY}
    return {
        key: {
            "path": _relative(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in paths.items()
    }


def _code_lineage() -> dict[str, dict[str, Any]]:
    """Hash every local module that can change retrieval, gating or synthesis.

    ``git.head`` is not enough in this repository because development runs are
    often intentionally made from a dirty worktree.  The explicit hashes make
    the exact evaluated implementation reproducible without publishing any raw
    evaluation text.
    """
    names = (
        "rag_gate.py",
        "rag_answerability.py",
        "rag_retrieval_trace.py",
        "rag_tools.py",
        "rag_multi_source.py",
        "rag_bm25.py",
        "rag_hyde.py",
        "rag_rerank.py",
        "rag_fusion.py",
        "rag_route.py",
        "rag_candidate_pool.py",
        "day1_api_starter.py",
    )
    return {
        name: {
            "path": name,
            "sha256": _sha256_file(ROOT / name),
            "bytes": (ROOT / name).stat().st_size,
        }
        for name in names
    }


def build_selection_manifest(*, git_info: dict[str, Any] | None = None,
                             runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    git_info = dict(git_info or _git_info())
    lineage = _source_lineage()
    seed = _sha256_text("|".join([
        SCHEMA_VERSION,
        str(git_info.get("head") or ""),
        *(lineage[key]["sha256"] for key in sorted(lineage)),
    ]))
    corpora = {
        key: _approved_train_positives(path) for key, path in POSITIVE_SOURCES.items()
    }
    positives = _select_positives(seed, corpora)
    negatives = _select_negatives(seed)
    items = [
        *(_manifest_item(row, n, "positive") for n, row in enumerate(positives, 1)),
        *(_manifest_item(row, n, "negative") for n, row in enumerate(negatives, 1)),
    ]
    manifest = {
        "schema_version": SELECTION_SCHEMA,
        "dataset_id": "answer_quality_user_worded_regression_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_boundary": {
            "kind": "development_regression",
            "not_blind": True,
            "not_organic": True,
            "not_training_data_for_current_production_model": True,
            "note": (
                "Questions were visible during earlier reranker development. "
                "They are user-worded historical regression evidence, not a new blind claim."
            ),
        },
        "selection": {
            "algorithm": "sha256_stratified_backtracking_v1",
            "seed_sha256": seed,
            "positive_count": POSITIVE_COUNT,
            "negative_count": NEGATIVE_COUNT,
            "positive_corpus_quota": {"v1": 20, "v2a": 20},
            "positive_domain_quota": {domain: 10 for domain in DOMAINS},
            "positive_style_quota": {
                "standard": 4, "natural": 12, "oral": 12, "long": 12,
            },
            "positive_unique_anchor_count": 40,
            "primary_source_cap": SOURCE_CAP,
            "negative_action_quota": {"correct_premise": 4, "abstain": 4},
        },
        "git": git_info,
        "source_files": lineage,
        "code_files": _code_lineage(),
        "evaluator": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
        },
        "runtime": runtime if runtime is not None else _runtime_snapshot(),
        "items": items,
        "privacy": {
            "producer_visible_fields": [
                "query_id", "anchor_id", "domain", "style_bucket",
                "expected_action", "question_sha256", "requirements_sha256",
                "targets_sha256", "primary_source_sha256",
            ],
            "forbidden_public_fields": [
                "question", "answer_requirements", "relevant_targets", "document",
                "chunks", "contexts", "answer", "reason", "local_path",
            ],
            "raw_artifact_policy": "repository_external_only",
        },
    }
    validate_selection_manifest(manifest)
    return manifest


def validate_selection_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SELECTION_SCHEMA:
        raise AnswerQualityV2Error("selection schema mismatch")
    code_files = manifest.get("code_files")
    if not isinstance(code_files, dict) or set(code_files) != set(_code_lineage()):
        raise AnswerQualityV2Error("selection code lineage is incomplete")
    if any(
        not isinstance(entry, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256") or ""))
        or not isinstance(entry.get("bytes"), int)
        or entry["bytes"] <= 0
        for entry in code_files.values()
    ):
        raise AnswerQualityV2Error("selection code lineage is malformed")
    items = list(manifest.get("items") or [])
    positives = [row for row in items if row.get("cohort") == "positive"]
    negatives = [row for row in items if row.get("cohort") == "negative"]
    if len(positives) != POSITIVE_COUNT or len(negatives) != NEGATIVE_COUNT:
        raise AnswerQualityV2Error("selection count mismatch")
    if len({row["eval_id"] for row in items}) != len(items):
        raise AnswerQualityV2Error("duplicate eval_id")
    if len({row["anchor_id"] for row in positives}) != POSITIVE_COUNT:
        raise AnswerQualityV2Error("positive anchors are not independent")
    if Counter(row["source_key"] for row in positives) != {"v1": 20, "v2a": 20}:
        raise AnswerQualityV2Error("positive corpus quota mismatch")
    if Counter(row["domain"] for row in positives) != Counter({d: 10 for d in DOMAINS}):
        raise AnswerQualityV2Error("positive domain quota mismatch")
    if Counter(row["style_bucket"] for row in positives) != Counter(
        {"standard": 4, "natural": 12, "oral": 12, "long": 12}
    ):
        raise AnswerQualityV2Error("positive style quota mismatch")
    source_counts = Counter(row["primary_source_sha256"] for row in positives)
    if source_counts and max(source_counts.values()) > SOURCE_CAP:
        raise AnswerQualityV2Error("primary-source cap exceeded")
    if Counter(row["expected_action"] for row in negatives) != {
        "correct_premise": 4, "abstain": 4,
    }:
        raise AnswerQualityV2Error("negative action quota mismatch")
    forbidden = set(manifest.get("privacy", {}).get("forbidden_public_fields") or [])
    for row in items:
        overlap = forbidden & set(row)
        if overlap:
            raise AnswerQualityV2Error(f"public selection leaks fields: {sorted(overlap)}")


def freeze_selection(path: Path = SELECTION_PATH) -> dict[str, Any]:
    if path.exists():
        existing = _read_json(path)
        validate_selection_manifest(existing)
        if str(existing.get("git", {}).get("head") or "") != _git_info().get("head"):
            raise AnswerQualityV2Error(
                f"selection already frozen under a different git HEAD: {path}"
            )
        if _canonical_json(existing.get("runtime") or {}) != _canonical_json(
            _runtime_snapshot()
        ):
            raise AnswerQualityV2Error(
                f"selection already frozen under a different runtime: {path}"
            )
        # A freeze is append-only.  Existing bytes are never silently replaced.
        fresh = build_selection_manifest(
            git_info=existing.get("git"), runtime=existing.get("runtime")
        )
        fresh["created_at"] = existing.get("created_at")
        if _canonical_json(fresh) != _canonical_json(existing):
            raise AnswerQualityV2Error(
                f"selection already frozen and current inputs differ: {path}"
            )
        return existing
    manifest = build_selection_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return manifest


def _assert_lineage(manifest: dict[str, Any]) -> None:
    for entry in (manifest.get("source_files") or {}).values():
        path = ROOT / entry["path"]
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise AnswerQualityV2Error(f"frozen source drift: {entry['path']}")
    for entry in (manifest.get("code_files") or {}).values():
        path = ROOT / entry["path"]
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise AnswerQualityV2Error(f"frozen code drift: {entry['path']}")
    if str(manifest.get("git", {}).get("head") or "") != _git_info().get("head"):
        raise AnswerQualityV2Error("git HEAD differs from the frozen selection")
    expected_evaluator = str(manifest.get("evaluator", {}).get("sha256") or "")
    if not expected_evaluator or _sha256_file(Path(__file__)) != expected_evaluator:
        raise AnswerQualityV2Error("answer-quality evaluator differs from the frozen selection")
    current_runtime = _runtime_snapshot()
    if _canonical_json(current_runtime) != _canonical_json(manifest.get("runtime") or {}):
        raise AnswerQualityV2Error("runtime profile/index/prompt differs from the frozen selection")


def resolve_selection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_selection_manifest(manifest)
    _assert_lineage(manifest)
    payloads = {
        "v1": _read_json(POSITIVE_SOURCES["v1"]),
        "v2a": _read_json(POSITIVE_SOURCES["v2a"]),
        "negative": _read_json(NEGATIVE_SOURCE),
    }
    indexes = {
        key: {str(row.get("query_id")): row for row in data.get("items") or []}
        for key, data in payloads.items()
    }
    # One correctable control lives in the V1 dev negative split.
    indexes["v1"].update({
        str(row.get("query_id")): row
        for row in payloads["v1"].get("items") or []
    })
    resolved = []
    for ref in manifest["items"]:
        row = indexes.get(ref["source_key"], {}).get(ref["query_id"])
        if row is None:
            raise AnswerQualityV2Error(f"frozen query is missing: {ref['query_id']}")
        if _text_hashes(row) != {
            key: ref[key] for key in (
                "question_sha256", "requirements_sha256", "targets_sha256"
            )
        }:
            raise AnswerQualityV2Error(f"frozen query contract drift: {ref['query_id']}")
        resolved.append(dict(row) | {
            "eval_id": ref["eval_id"],
            "cohort": ref["cohort"],
            "expected_action": ref["expected_action"],
            "negative_stratum": ref.get("negative_stratum", ""),
            "style_bucket": ref["style_bucket"],
        })
    return resolved


def ensure_repository_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise AnswerQualityV2Error(
        f"raw evaluation artifacts must stay outside the repository: {resolved}"
    )


@contextmanager
def isolated_evaluation_runtime(cache_path: Path):
    """Keep evaluation-only caches and usage logs out of application state."""

    cache_path = ensure_repository_external(cache_path)
    if cache_path.exists():
        raise AnswerQualityV2Error(
            "evaluation answerability cache already exists; use a fresh run directory"
        )
    import rag_answerability

    previous_cache_path = rag_answerability.CACHE_PATH
    previous_cache = rag_answerability._CACHE
    env_values = {
        "LLM_USAGE_LOG": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    previous_env = {key: os.environ.get(key) for key in env_values}
    rag_answerability.CACHE_PATH = cache_path
    rag_answerability._CACHE = None
    os.environ.update(env_values)
    try:
        yield
    finally:
        try:
            rag_answerability.flush_cache()
        finally:
            rag_answerability.CACHE_PATH = previous_cache_path
            rag_answerability._CACHE = previous_cache
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@contextmanager
def frozen_answerability_environment(manifest: dict[str, Any]):
    """Install the manifest's explicit quality profile for retrieval only."""
    contract = (
        (manifest.get("runtime") or {}).get("answerability") or {}
    ).get("frozen_env")
    if contract != FROZEN_ANSWERABILITY_ENV:
        raise AnswerQualityV2Error("frozen answerability environment is missing or altered")
    previous = {key: os.environ.get(key) for key in contract}
    try:
        os.environ.update(contract)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# 动作派生的唯一实现现在住在生产侧(rag_gate.answer_action_from_retrieval),
# 本评测器 2026-08-31 前的本地副本已收编——评测与生产两把尺子迟早分家,
# 而"门说了、生成没做"这次正是靠同一把尺子才量得出来。
from rag_gate import answer_action_from_retrieval as _answer_action


def _context_ids(retrieval: dict[str, Any]) -> list[str]:
    contexts = list(retrieval.get("chunks") or [])
    all_docs = list(retrieval.get("docs") or [])
    final = list((retrieval.get("retrieval_trace") or {}).get("final_candidates") or [])
    ids_by_doc: defaultdict[str, list[str]] = defaultdict(list)
    for document, candidate in zip(all_docs, final):
        ids_by_doc[str(document)].append(str(candidate.get("chunk_id") or ""))
    seen_by_doc: Counter[str] = Counter()
    output = []
    for context in contexts:
        key = str(context)
        position = seen_by_doc[key]
        options = ids_by_doc.get(key) or []
        output.append(options[position] if position < len(options) else "")
        seen_by_doc[key] += 1
    return output


def _production_reference_once(item: dict[str, Any]) -> dict[str, Any]:
    """Run the production reference retrieval and synthesis helpers once."""
    from rag_gate import (_chat, _fallback_messages, _grounded_messages,
                          _retrieve_and_classify, FALLBACK_LABEL)
    from rag_multi_source import REFERENCE_EXCLUDES

    started = time.perf_counter()
    retrieval = _retrieve_and_classify(
        item["question"], 5,
        exclude_source_types=REFERENCE_EXCLUDES,
        metadata_filters={"owner_scope": {"curated"}},
        allow_paper_route=False,
    )
    retrieval_ms = (time.perf_counter() - started) * 1000
    pipeline_action = _answer_action(retrieval)
    contexts = list(retrieval.get("chunks") or [])
    generation_meta: dict[str, Any] = {}
    if retrieval.get("in_kb"):
        # Mirror current HEAD exactly: since 2026-08-31 the retrieval result
        # carries the gate's structured action (stamped in _finish) and
        # production synthesis consumes it, so the eval passes the same value.
        generation_contract_action = str(
            retrieval.get("answer_action") or "answer")
        messages = _grounded_messages(item["question"], contexts,
                                      generation_contract_action)
        mode = "kb_grounded"
    else:
        messages = _fallback_messages(item["question"], contexts)
        mode = "general_fallback"
        generation_contract_action = "general_fallback"
    generation_started = time.perf_counter()
    answer = _chat(messages, max_tokens=800, temperature=0.2, meta=generation_meta)
    generation_ms = (time.perf_counter() - generation_started) * 1000
    if not retrieval.get("in_kb") and answer:
        answer = FALLBACK_LABEL + answer
    context_ids = _context_ids(retrieval)
    gold_ids = {
        str(target.get("chunk_id")) for target in item.get("relevant_targets") or []
        if target.get("chunk_id")
    }
    return {
        "schema_version": RAW_SCHEMA,
        "eval_id": item["eval_id"],
        "query_id": item["query_id"],
        "cohort": item["cohort"],
        "domain": item.get("domain", "negative"),
        "style_bucket": item.get("style_bucket", "negative"),
        "negative_stratum": item.get("negative_stratum", ""),
        "expected_action": item["expected_action"],
        "question": item["question"],
        "answer_requirements": item.get("answer_requirements") or [],
        "gold_chunk_ids": sorted(gold_ids),
        "contexts": contexts,
        "context_chunk_ids": context_ids,
        "answer": answer or "",
        "observed_action": pipeline_action,
        "generation_contract_action": generation_contract_action,
        "mode": mode,
        "in_kb": bool(retrieval.get("in_kb")),
        "gold_in_context": bool(gold_ids & set(context_ids)),
        "retrieval_profile": retrieval.get("retrieval_profile", ""),
        "index_fingerprint": retrieval.get("index_fingerprint", ""),
        "retrieval_trace": retrieval.get("retrieval_trace") or {},
        "generation": {
            "requested_model": os.environ.get("RAG_SYNTH_MODEL", ""),
            "served_model": generation_meta.get("model"),
            "finish_reason": generation_meta.get("finish_reason"),
            "usage": generation_meta.get("usage"),
            "messages_sha256": _json_sha256(messages),
            "available": bool(answer),
        },
        "latency_ms": {
            "retrieval": round(retrieval_ms, 3),
            "generation": round(generation_ms, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
    }


JUDGE_SYSTEM = """你是严格的 RAG 答案审计员。只能使用给定的完整检索上下文，禁止使用外部知识。

逐项完成：
1. 把答案拆成最小可核验事实 claim；纯格式、标题和“资料不足”声明不算事实 claim。
2. 对每个 claim 列出真正支持它的上下文编号 supported_context_ids，以及答案在该 claim 附近实际引用的编号 cited_context_ids。无支持则 supported=false。
3. 对每条人工答案要求判断 full / partial / none。必须以答案实际覆盖为准；检索上下文里有但答案没说，仍是 none。
4. 判断答案实际动作 observed_action：正常回答=answer；明确指出问题前提错误并只据资料纠正=correct_premise；明确拒绝/资料不足=abstain。
5. correct_premise 时，判断首句是否明确指出前提不成立 correction_first_sentence_ok。

拒答没有编造时可以忠实，但对正例的完整度必须是 0；不要把“没答”判成完整。只输出一个 JSON 对象：
{"claims":[{"claim":"...","supported":true,"supported_context_ids":[1],"cited_context_ids":[1]}],"requirements":[{"index":0,"coverage":"full|partial|none"}],"observed_action":"answer|correct_premise|abstain","correction_first_sentence_ok":true|null,"reason":"不超过80字"}
"""


def _judge_user_message(row: dict[str, Any]) -> str:
    context = "\n\n".join(
        f"[资料{index}]\n{text}" for index, text in enumerate(row["contexts"], 1)
    )
    requirements = "\n".join(
        f"{index}. {text}" for index, text in enumerate(row["answer_requirements"])
    ) or "（本题没有正例答案要求；仍须根据问题、上下文和答案独立判断实际动作）"
    return (
        f"[问题]\n{row['question']}\n\n[人工答案要求]\n{requirements}\n\n"
        f"[完整检索上下文]\n{context or '（无）'}\n\n[系统答案]\n{row['answer'] or '（无答案）'}"
    )


def _judge_config(name: str) -> dict[str, Any]:
    from day1_api_starter import get_llm_config, load_local_env
    load_local_env()
    cfg = get_llm_config()
    if name == "deepseek-v3":
        return {
            "name": name,
            "model": "deepseek-v3",
            "base_url": os.environ.get(
                "AQV2_DEEPSEEK_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
            "api_key_env": "DASHSCOPE_API_KEY",
            "reasoning_effort": "",
        }
    if name == "gpt-5.5":
        return {
            "name": name,
            "model": "gpt-5.5",
            "base_url": os.environ.get("AQV2_GPT_BASE_URL", cfg["api_base"]).rstrip("/"),
            "api_key": os.environ.get("AQV2_GPT_API_KEY", cfg["api_key"]),
            "api_key_env": "AQV2_GPT_API_KEY|OPENAI_API_KEY",
            "reasoning_effort": os.environ.get("AQV2_GPT_REASONING_EFFORT", "medium"),
        }
    raise AnswerQualityV2Error(f"unsupported judge: {name}")


def _extract_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw or ""):
        try:
            value, _end = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AnswerQualityV2Error("judge did not return a JSON object")


def validate_judgment(value: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    claims = value.get("claims")
    requirements = value.get("requirements")
    if not isinstance(claims, list) or not isinstance(requirements, list):
        raise AnswerQualityV2Error("judge claims/requirements must be arrays")
    context_max = len(row.get("contexts") or [])
    clean_claims = []
    for claim in claims:
        if not isinstance(claim, dict) or not str(claim.get("claim") or "").strip():
            raise AnswerQualityV2Error("judge claim is malformed")
        supported_ids = sorted({int(v) for v in claim.get("supported_context_ids") or []})
        cited_ids = sorted({int(v) for v in claim.get("cited_context_ids") or []})
        if any(v < 1 or v > context_max for v in [*supported_ids, *cited_ids]):
            raise AnswerQualityV2Error("judge context id is out of range")
        supported = bool(claim.get("supported"))
        if supported != bool(supported_ids):
            raise AnswerQualityV2Error("supported flag and support ids disagree")
        clean_claims.append({
            "claim": str(claim["claim"]).strip(),
            "supported": supported,
            "supported_context_ids": supported_ids,
            "cited_context_ids": cited_ids,
        })
    expected_requirement_count = len(row.get("answer_requirements") or [])
    by_index: dict[int, str] = {}
    for result in requirements:
        try:
            index = int(result["index"])
            coverage = str(result["coverage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnswerQualityV2Error("judge requirement is malformed") from exc
        if index in by_index or coverage not in {"full", "partial", "none"}:
            raise AnswerQualityV2Error("judge requirement coverage is invalid")
        by_index[index] = coverage
    if set(by_index) != set(range(expected_requirement_count)):
        raise AnswerQualityV2Error("judge requirement coverage is incomplete")
    action = str(value.get("observed_action") or "")
    if action not in {"answer", "correct_premise", "abstain"}:
        raise AnswerQualityV2Error("judge observed_action is invalid")
    correction_ok = value.get("correction_first_sentence_ok")
    if correction_ok is not None and not isinstance(correction_ok, bool):
        raise AnswerQualityV2Error("correction_first_sentence_ok must be bool or null")
    return {
        "claims": clean_claims,
        "requirements": [
            {"index": index, "coverage": by_index[index]}
            for index in range(expected_requirement_count)
        ],
        "observed_action": action,
        "correction_first_sentence_ok": correction_ok,
        "reason": str(value.get("reason") or "")[:500],
    }


def _judge_once(name: str, row: dict[str, Any]) -> dict[str, Any]:
    from day1_api_starter import chat_completion, extract_content
    config = _judge_config(name)
    if not config["api_key"]:
        raise AnswerQualityV2Error(f"missing judge key: {config['api_key_env']}")
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _judge_user_message(row)},
        ],
        "temperature": 0.0,
        "max_tokens": 1600,
    }
    if config["reasoning_effort"]:
        payload["reasoning_effort"] = config["reasoning_effort"]
    started = time.perf_counter()
    data = chat_completion(
        config["base_url"] + "/chat/completions",
        {"Authorization": f"Bearer {config['api_key']}",
         "Content-Type": "application/json"},
        payload, timeout=120, max_retries=2,
    )
    raw = extract_content(data) or ""
    judgment = validate_judgment(_extract_json_object(raw), row)
    return {
        "available": True,
        "requested_model": config["model"],
        "served_model": (data or {}).get("model") or config["model"],
        "prompt_sha256": _sha256_text(JUDGE_SYSTEM),
        "input_sha256": _sha256_text(_judge_user_message(row)),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "usage": (data or {}).get("usage"),
        "judgment": judgment,
    }


def score_judgment(row: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    claims = list(judgment.get("claims") or [])
    supported = [claim for claim in claims if claim.get("supported")]
    cited_pairs = [
        (claim, citation)
        for claim in claims for citation in claim.get("cited_context_ids") or []
    ]
    supporting_citations = sum(
        citation in set(claim.get("supported_context_ids") or [])
        for claim, citation in cited_pairs
    )
    cited_supported_claims = sum(
        bool(set(claim.get("cited_context_ids") or [])
             & set(claim.get("supported_context_ids") or []))
        for claim in supported
    )
    coverage_value = {"full": 1.0, "partial": 0.5, "none": 0.0}
    requirement_scores = [
        coverage_value[result["coverage"]]
        for result in judgment.get("requirements") or []
    ]
    faithfulness = (
        sum(bool(claim.get("supported")) for claim in claims) / len(claims)
        if claims else (1.0 if judgment.get("observed_action") == "abstain" else 0.0)
    )
    completeness = (
        sum(requirement_scores) / len(requirement_scores)
        if requirement_scores else None
    )
    expected_action = row["expected_action"]
    judged_action = judgment.get("observed_action")
    pipeline_action_correct = row.get("observed_action") == expected_action
    generation_action_correct = judged_action == expected_action
    action_correct = bool(pipeline_action_correct and generation_action_correct)
    if expected_action == "correct_premise":
        action_correct = bool(
            action_correct and judgment.get("correction_first_sentence_ok") is True
        )
    elif expected_action == "abstain":
        # The product intentionally may offer a clearly labelled *general*
        # fallback after the KB gate abstains.  That does not turn a safe RAG
        # abstention into a false evidence accept.  Preserve the final-answer
        # action as a separate diagnostic, but score the negative RAG contract
        # from the structural pipeline action.
        action_correct = bool(pipeline_action_correct)
    grounded = row.get("mode") == "kb_grounded"
    return {
        "faithfulness": round(faithfulness, 6),
        "completeness": round(completeness, 6) if completeness is not None else None,
        "citation_precision": (
            round(supporting_citations / len(cited_pairs), 6) if cited_pairs else 0.0
        ),
        "citation_recall": (
            round(cited_supported_claims / len(supported), 6) if supported else 0.0
        ),
        "action_correct": bool(action_correct),
        "pipeline_action_correct": bool(pipeline_action_correct),
        "generation_action_correct": bool(generation_action_correct),
        "effective_completeness": (
            round(completeness, 6) if grounded and completeness is not None else 0.0
        ),
        "claim_count": len(claims),
        "supported_claim_count": len(supported),
        "cited_pair_count": len(cited_pairs),
    }


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(sum(rows) / len(rows), 6) if rows else None


def summarize_rows(rows: list[dict[str, Any]], *, selection_sha256: str,
                   raw_sha256: str) -> dict[str, Any]:
    if len(rows) != POSITIVE_COUNT + NEGATIVE_COUNT:
        raise AnswerQualityV2Error("raw result does not cover all frozen items")
    positives = [row for row in rows if row.get("cohort") == "positive"]
    negatives = [row for row in rows if row.get("cohort") == "negative"]
    coverage = {
        "selected": len(rows),
        "generated": sum(bool(row.get("answer")) for row in rows),
        "kb_grounded_positive": sum(row.get("mode") == "kb_grounded" for row in positives),
        "gold_in_context_positive": sum(bool(row.get("gold_in_context")) for row in positives),
        "generation_unavailable": sum(not bool(row.get("answer")) for row in rows),
    }
    judge_summaries: dict[str, Any] = {}
    generation_complete = coverage["generation_unavailable"] == 0
    complete = generation_complete
    for name in JUDGE_NAMES:
        available = [row for row in rows if (row.get("judges") or {}).get(name, {}).get("available")]
        if len(available) != len(rows):
            complete = False
        positive_scored = [
            row["judges"][name]["scores"] for row in positives
            if (row.get("judges") or {}).get(name, {}).get("scores")
        ]
        negative_scored = [
            row["judges"][name]["scores"] for row in negatives
            if (row.get("judges") or {}).get(name, {}).get("scores")
        ]
        positive_complete = len(positive_scored) == POSITIVE_COUNT
        negative_complete = len(negative_scored) == NEGATIVE_COUNT
        publish_positive_scores = generation_complete and positive_complete
        judge_summaries[name] = {
            "available": len(available),
            "required": len(rows),
            "positive_available": len(positive_scored),
            "positive_required": POSITIVE_COUNT,
            # Every published score retains its preregistered denominator.
            # Missing generation/judgment withholds the aggregate; it is never
            # replaced with zero, another judge, or a smaller denominator.
            "faithfulness": _mean(
                s["faithfulness"] for s in positive_scored
            ) if publish_positive_scores else None,
            "completeness": _mean(
                s["completeness"] for s in positive_scored
                if s["completeness"] is not None
            ) if publish_positive_scores else None,
            "effective_completeness_all_positive": _mean(
                s["effective_completeness"] for s in positive_scored
            ) if publish_positive_scores else None,
            "citation_precision": _mean(
                s["citation_precision"] for s in positive_scored
            ) if publish_positive_scores else None,
            "citation_recall": _mean(
                s["citation_recall"] for s in positive_scored
            ) if publish_positive_scores else None,
            "negative_action_correct": (
                sum(s["action_correct"] for s in negative_scored)
                if generation_complete and negative_complete else None
            ),
            "negative_action_required": NEGATIVE_COUNT,
        }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete" if complete else "incomplete_no_score_imputation",
        "release_boundary": "development_regression_not_blind_not_organic",
        "selection_manifest_sha256": selection_sha256,
        "raw_artifact_sha256": raw_sha256,
        "coverage": coverage,
        "judges": judge_summaries,
        "privacy": {
            "contains_questions": False,
            "contains_contexts": False,
            "contains_answers": False,
            "contains_judge_reasons": False,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AnswerQualityV2Error(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def run_evaluation(manifest_path: Path = SELECTION_PATH, *,
                   artifact_root: Path | None = None,
                   publish_summary: bool = False) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    resolved = resolve_selection(manifest)
    manifest_sha = _sha256_file(manifest_path)
    default_root = Path(os.environ.get(
        "OFFERCLAW_EVAL_ARTIFACT_ROOT",
        str(Path.home() / ".offerclaw" / "eval_runs"),
    ))
    root = ensure_repository_external(artifact_root or default_root)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + manifest_sha[:12]
    )
    run_dir = root / "answer_quality_v2" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_path = run_dir / "raw_rows.jsonl"
    private_cache_path = run_dir / "answerability_cache.json"

    rows = []
    with isolated_evaluation_runtime(private_cache_path):
        with frozen_answerability_environment(manifest):
            for position, item in enumerate(resolved, 1):
                print(f"[aqv2] generate {position}/{len(resolved)} {item['eval_id']}",
                      flush=True)
                row = _production_reference_once(item)
                row["judges"] = {}
                rows.append(row)
                _write_jsonl(raw_path, rows)

    # The immutable generation artifact is complete before either judge starts.
    generation_sha = _sha256_file(raw_path)
    for judge_name in JUDGE_NAMES:
        for position, row in enumerate(rows, 1):
            print(f"[aqv2] judge={judge_name} {position}/{len(rows)} {row['eval_id']}",
                  flush=True)
            try:
                result = _judge_once(judge_name, row)
                result["scores"] = score_judgment(row, result["judgment"])
            except Exception as exc:
                result = {
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error_sha256": _sha256_text(str(exc)),
                }
            row["judges"][judge_name] = result
            _write_jsonl(raw_path, rows)

    raw_sha = _sha256_file(raw_path)
    summary = summarize_rows(
        rows, selection_sha256=manifest_sha, raw_sha256=raw_sha
    )
    summary["run"] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_artifact_sha256_before_judging": generation_sha,
        "raw_artifact_bytes": raw_path.stat().st_size,
        "raw_artifact_location": "repository_external",
    }
    (run_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if publish_summary:
        if summary["status"] != "complete":
            raise AnswerQualityV2Error(
                "incomplete judge/generation coverage cannot be published as final"
            )
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-selection", help="freeze the text-free 48-item manifest")
    freeze.add_argument("--output", default=str(SELECTION_PATH))
    run = sub.add_parser("run", help="run production generation and two neutral judges")
    run.add_argument("--selection", default=str(SELECTION_PATH))
    run.add_argument("--artifact-root", default="")
    run.add_argument("--publish-summary", action="store_true")
    args = parser.parse_args()

    if args.command == "freeze-selection":
        manifest = freeze_selection(Path(args.output))
        print(json.dumps({
            "status": "frozen",
            "path": str(Path(args.output)),
            "sha256": _sha256_file(Path(args.output)),
            "positive": manifest["selection"]["positive_count"],
            "negative": manifest["selection"]["negative_count"],
            "boundary": manifest["release_boundary"],
        }, ensure_ascii=False, indent=2))
        return 0
    summary = run_evaluation(
        Path(args.selection),
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        publish_summary=args.publish_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
