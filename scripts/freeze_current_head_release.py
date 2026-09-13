#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze (or verify) the current production RAG release contract.

This is intentionally separate from ``freeze_final_v4_config.py``.  Final v4
is an immutable historical blind experiment bound to its original index and
must never be overwritten to make a later checkout look comparable.

The current-head release is a *regression* contract.  It reuses the approved
Final-v4 labels without changing them, first proving that every referenced
chunk/source/evidence excerpt still exists in the live index.  The companion
runner performs the index rebind only in memory; no derived qrels file is
written into the repository.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "docs" / "rag_eval" / "final_v4" / "final_v4.json"
DEFAULT_OUTPUT = (
    ROOT / "docs" / "rag_eval" / "current_head_20260831" / "FROZEN_CONFIG.json"
)
HISTORICAL_FINAL_V4 = ROOT / "docs" / "rag_eval" / "final_v4"

# Empty means deliberately empty, not "inherit from .env.local".  The project
# loader never overwrites an existing environment key, including an empty one.
# This makes a copied .env.example (which currently says answerability=off)
# unable to silently turn the measured quality path into a different product.
FROZEN_ENV: dict[str, str] = {
    "EMBEDDING_PROVIDER": "local",
    "EMBEDDING_MODEL": "BAAI/bge-base-zh-v1.5",
    "EMBEDDING_DIMENSIONS": "768",
    "RAG_COLLECTION_NAME": "offerclaw_local_bge_base_zh_768",
    "RAG_RETRIEVAL_PROFILE": "baseline",
    "RAG_RECALL_N": "28",
    "RAG_BASELINE_RERANK_POOL": "28",
    "RAG_BASELINE_RERANK_MODEL": "BAAI/bge-reranker-base",
    "RAG_BASELINE_RERANK_PREFIX_MODE": "compact32",
    "RAG_RERANK_MODEL": "BAAI/bge-reranker-base",
    "RAG_RERANK": "1",
    "RAG_RERANK_POOL": "",
    "RAG_RERANK_MAX_SEQ": "",
    "RAG_RERANK_ONNX_DIR": "",
    "RAG_RERANK_EN_ONNX_DIR": "",
    "RAG_PROFILE_DEFER_ENV": "0",
    "RAG_HYDE_CHANNELS": "1",
    "RAG_HYDE": "0",
    "RAG_QUERY_REWRITE": "0",
    "RAG_DOC2QUERY": "0",
    "RAG_EN_QUOTA": "0",
    "RAG_EN_GATE_MIN": "",
    "RAG_ANSWERABILITY_MODE": "teacher",
    "RAG_ANSWERABILITY": "",
    "RAG_ANSWERABILITY_MODEL": "gpt-5.6-terra",
    "RAG_ANSWERABILITY_PROMPT": "v5",
    "RAG_ANSWERABILITY_DEPTH": "12",
    "RAG_ANSWERABILITY_EARLY_EXIT": "1",
    "RAG_ANSWERABILITY_GATE": "1",
    "RAG_ANSWERABILITY_GATE_VOTES": "3",
    "RAG_ANSWERABILITY_TIEBREAK": "0",
    # These are explicit numeric defaults, not empty "unset" sentinels.
    # ``rag_gate._evidence_gate`` parses the rerank values with ``float(env)``;
    # freezing them as empty strings prevents .env.local inheritance but is not
    # semantically equivalent to absence and crashes the first real query.
    "RAG_RELEVANCE_MAX_DIST": "0.73",
    "RAG_LEXICAL_RESCUE_DIST": "0.73",
    "RAG_WEAK_CONTEXT_DIST": "1.10",
    "RAG_RERANK_GATE_MIN": "0.85",
    "RAG_RERANK_RESCUE_DIST": "0.80",
    "RAG_RERANK_RESCUE_MIN": "0.95",
    "RAG_STRUCTURAL_EVIDENCE_MAX": "",
    "RAG_COLLOQUIAL_GATE_MIN": "",
    "RAG_COLLOQUIAL_GATE_MARGIN": "0.10",
    "LLM_MODEL": "gpt-5.6-terra",
    "RAG_SYNTH_MODEL": "gpt-5.6-terra",
}

# Freeze every root-level RAG module rather than maintaining another fragile
# hand-written import graph.  This is cheap and prevents a newly introduced
# helper from changing production while remaining outside the manifest.
EXTRA_RUNTIME_FILES = (
    "day1_api_starter.py",
    "product_help.py",
    "semantic_query_planner.py",
    "system_diagnostics.py",
)
TOOLING_FILES = (
    "scripts/freeze_current_head_release.py",
    "scripts/run_current_head_regression.py",
)
DEPENDENCY_FILES = ("requirements.txt", "requirements-answerability-student.txt")


class ReleaseFreezeError(RuntimeError):
    """The requested freeze is unsafe or no longer matches the workspace."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(blob)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


def ensure_not_historical_output(path: Path) -> None:
    if _inside(path, HISTORICAL_FINAL_V4):
        raise ReleaseFreezeError(
            "current-head artifacts may not overwrite docs/rag_eval/final_v4"
        )


def runtime_paths() -> list[Path]:
    # Union the HEAD inventory with the working-tree inventory.  Looking only
    # at existing files would miss a locally deleted tracked module, exactly
    # the kind of dirty runtime a "current HEAD" freeze must reject.
    tracked = _git("ls-files", "--", "rag_*.py", *EXTRA_RUNTIME_FILES).splitlines()
    paths = [ROOT / name for name in tracked]
    paths.extend(ROOT.glob("rag_*.py"))
    paths.extend(ROOT / name for name in EXTRA_RUNTIME_FILES)
    return sorted({path.resolve() for path in paths})


def config_paths() -> list[Path]:
    tracked = _git("ls-files", "--", "config/*.json").splitlines()
    paths = [ROOT / name for name in tracked]
    paths.extend((ROOT / "config").glob("*.json"))
    return sorted({path.resolve() for path in paths})


def dependency_paths() -> list[Path]:
    return [ROOT / name for name in DEPENDENCY_FILES]


def tooling_paths() -> list[Path]:
    paths = [ROOT / name for name in TOOLING_FILES]
    missing = [_relative(path) for path in paths if not path.is_file()]
    if missing:
        raise ReleaseFreezeError(f"release tooling is incomplete: {missing}")
    return paths


def _file_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        _relative(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths)
    }


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    if proc.returncode != 0:
        raise ReleaseFreezeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def tracked_runtime_dirty(paths: list[Path] | None = None) -> list[str]:
    material = paths or [*runtime_paths(), *config_paths(), *dependency_paths()]
    relative = [_relative(path) for path in material]
    if not relative:
        return []
    output = _git("status", "--porcelain=v1", "--untracked-files=no", "--", *relative)
    return sorted(line for line in output.splitlines() if line.strip())


def untracked_runtime(paths: list[Path]) -> list[str]:
    relative = [_relative(path) for path in paths]
    if not relative:
        return []
    output = _git("ls-files", "--others", "--exclude-standard", "--", *relative)
    return sorted(line for line in output.splitlines() if line.strip())


@contextmanager
def frozen_environment(values: dict[str, str] | None = None) -> Iterator[None]:
    chosen = values or FROZEN_ENV
    previous = {key: os.environ.get(key) for key in chosen}
    try:
        for key, value in chosen.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def label_projection(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only fields that carry qrels labels, never the raw question."""

    fields = (
        "query_id",
        "anchor_id",
        "split",
        "case_kind",
        "expected_behavior",
        "answer_requirements",
        "relevant_targets",
        "hard_negatives",
        "review_status",
    )
    return [
        {key: item.get(key) for key in fields if key in item}
        for item in payload.get("items", [])
    ]


def label_projection_sha256(payload: dict[str, Any]) -> str:
    return _canonical_sha256(label_projection(payload))


def stable_index_payload(fingerprint: dict[str, Any]) -> dict[str, Any]:
    from rag_qrels_v2 import index_contract_fingerprint

    content_hash = str(
        fingerprint.get("collection_content_hash")
        or fingerprint.get("index_content_fingerprint")
        or ""
    )
    payload = {
        "collection": str(fingerprint.get("collection") or ""),
        "count": int(fingerprint.get("collection_count") or 0),
        # Keep the qrels-friendly display key and the canonical helper key.
        # ``index_contract_fingerprint`` deliberately consumes the latter.
        "content_hash": content_hash,
        "collection_content_hash": content_hash,
        "fingerprint_id": str(fingerprint.get("fingerprint_id") or ""),
        "embedding_provider": str(fingerprint.get("embedding_provider") or ""),
        "embedding_model": str(fingerprint.get("embedding_model") or ""),
        "embedding_dimensions": fingerprint.get("embedding_dimensions"),
        "chunker_version": str(fingerprint.get("chunker_version") or ""),
        "indexed_chunker_versions": list(
            fingerprint.get("indexed_chunker_versions") or []
        ),
    }
    # rag_qrels_v2 accepts either count/content_hash or the raw fingerprint
    # shape, so this binds the in-memory qrels to precisely the same semantics.
    payload["fingerprint"] = index_contract_fingerprint(payload)
    return payload


def _load_and_validate_dataset(
    dataset_path: Path, collection: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    from rag_qrels_v2 import load_graded_qrels, validate_graded_qrels_against_collection

    payload = load_graded_qrels(dataset_path, require_approved=True)
    validate_graded_qrels_against_collection(payload, collection)
    items = payload.get("items", [])
    targets = [target for item in items for target in item.get("relevant_targets", [])]
    stats = {
        "rows": len(items),
        "positive_rows": sum(item.get("case_kind") == "positive" for item in items),
        "negative_rows": sum(item.get("case_kind") == "negative" for item in items),
        "target_rows": len(targets),
        "unique_target_chunks": len({target.get("chunk_id") for target in targets}),
    }
    return payload, stats


def collect_release_state(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise ReleaseFreezeError(f"dataset does not exist: {dataset_path}")

    runtime = runtime_paths()
    configs = config_paths()
    dependencies = dependency_paths()
    tooling = tooling_paths()
    dirty = tracked_runtime_dirty([*runtime, *configs, *dependencies])
    untracked = untracked_runtime([*runtime, *configs, *dependencies])
    if dirty:
        raise ReleaseFreezeError(
            "tracked runtime differs from HEAD; commit or restore it before freezing: "
            + ", ".join(dirty)
        )
    if untracked:
        raise ReleaseFreezeError(
            "untracked runtime is not part of HEAD; commit or remove it before freezing: "
            + ", ".join(untracked)
        )
    missing = [_relative(path) for path in [*runtime, *configs, *dependencies]
               if not path.is_file()]
    if missing:
        raise ReleaseFreezeError(f"tracked release files are missing: {missing}")

    with frozen_environment():
        import chromadb
        from rag_answerability import (
            ANSWERABILITY_DEPTH,
            MIN_GRADE_TO_ACT,
            SCHEMA,
            active_prompt,
            mode,
            resolve_model,
        )
        from rag_gate import (
            _answerability_depth,
            _answerability_early_exit,
            _answerability_gate,
            _answerability_gate_votes,
            _synth_model,
            _thresholds,
        )
        from rag_retrieval_trace import resolve_retrieval_profile
        from rag_tools import get_collection_name, index_fingerprint

        collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
            get_collection_name()
        )
        fingerprint = index_fingerprint(collection=collection, cache_ttl=0)
        index = stable_index_payload(fingerprint)
        dataset, dataset_stats = _load_and_validate_dataset(dataset_path, collection)
        profile = resolve_retrieval_profile("baseline").to_dict()
        prompt_name = "v5" if os.environ["RAG_ANSWERABILITY_PROMPT"] == "v5" else "v4"
        prompt_sha = active_prompt()[1]

        answerability = {
            "mode": mode(),
            "model": resolve_model(),
            "schema": SCHEMA,
            "prompt": prompt_name,
            "prompt_sha256": prompt_sha,
            "module_default_depth": ANSWERABILITY_DEPTH,
            "effective_depth": _answerability_depth(),
            "early_exit": _answerability_early_exit(),
            "min_grade_to_act": MIN_GRADE_TO_ACT,
        }
        gate = {
            "answerability_gate": _answerability_gate(),
            "votes": _answerability_gate_votes(),
            "distance_thresholds": _thresholds(),
            "rerank_gate_min": float(os.environ.get("RAG_RERANK_GATE_MIN") or "0.85"),
            "rerank_rescue_dist": float(
                os.environ.get("RAG_RERANK_RESCUE_DIST") or "0.80"
            ),
            "rerank_rescue_min": float(
                os.environ.get("RAG_RERANK_RESCUE_MIN") or "0.95"
            ),
            "structural_evidence_max": (
                os.environ.get("RAG_STRUCTURAL_EVIDENCE_MAX") or None
            ),
        }
        generation = {"hyde_model": _synth_model()}

    return {
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_runtime_dirty": dirty,
            "untracked_runtime": untracked,
        },
        "protocol": {
            "profile_source": "production_registry",
            "profile_name": "baseline",
            "route_mode": "oracle_reference_kb",
            "repeat_policy": "report_all_runs_no_retuning",
            "release_status": "frozen_current_head_regression_not_blind",
            "row_output_policy": "repository_external_only",
        },
        "environment": dict(sorted(FROZEN_ENV.items())),
        "retrieval_profile": profile,
        "answerability": answerability,
        "gate": gate,
        "generation": generation,
        "index": index,
        "dataset": {
            "path": _relative(dataset_path),
            "dataset_id": dataset.get("dataset_id"),
            "sha256": sha256_file(dataset_path),
            **dataset_stats,
            "source_index": dataset.get("index"),
            "target_index": index,
            "binding_mode": "validated_in_memory_rebind",
            "labels_unchanged": True,
            "label_projection_sha256": label_projection_sha256(dataset),
        },
        "files": {
            "runtime": _file_records(runtime),
            "config": _file_records(configs),
            "dependencies": _file_records(dependencies),
            "tooling": _file_records(tooling),
        },
    }


def build_manifest(dataset_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "offerclaw-current-head-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **collect_release_state(dataset_path),
    }


def _semantic_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "offerclaw-current-head-release-v1":
        raise ReleaseFreezeError("unsupported current-head freeze schema")
    dataset_name = str((frozen.get("dataset") or {}).get("path") or "")
    dataset_path = Path(dataset_name)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    current = {
        "schema_version": "offerclaw-current-head-release-v1",
        **collect_release_state(dataset_path),
    }
    want = _semantic_manifest(frozen)
    got = _semantic_manifest(current)
    if want != got:
        sections = sorted(
            key for key in set(want) | set(got) if want.get(key) != got.get(key)
        )
        raise ReleaseFreezeError(
            "current-head freeze drift in sections: " + ", ".join(sections)
        )
    return {
        "status": "ok",
        "manifest": _relative(manifest_path),
        "git_head": current["git"]["head"],
        "index_fingerprint_id": current["index"]["fingerprint_id"],
        "dataset_sha256": current["dataset"]["sha256"],
    }


def write_manifest(output: Path, dataset: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    ensure_not_historical_output(output)
    if output.resolve() == dataset.expanduser().resolve():
        raise ReleaseFreezeError("freeze output may not replace its source dataset")
    manifest = build_manifest(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument(
        "--verify",
        nargs="?",
        const=str(DEFAULT_OUTPUT),
        metavar="MANIFEST",
        help="verify a manifest (default: the current-head default path)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.verify:
            report = verify_manifest(Path(args.verify))
        else:
            manifest = write_manifest(args.output, args.dataset)
            report = {
                "status": "written",
                "manifest": _relative(args.output),
                "git_head": manifest["git"]["head"],
                "index_fingerprint_id": manifest["index"]["fingerprint_id"],
                "dataset_sha256": manifest["dataset"]["sha256"],
            }
    except (OSError, ValueError, ReleaseFreezeError) as exc:
        print(f"[current-head-freeze] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
