#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the frozen current production profile on approved historical qrels.

The source qrels remain immutable.  Their target chunks are checked against the
live collection and the index contract is rebound only in memory.  Per-query
rows are private evaluation material and may only be written outside the
repository; the optional repository summary contains aggregate counts and
hashes, never questions, chunk text, or row arrays.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Executed as a script: its directory is already importable.
    from freeze_current_head_release import (
        HISTORICAL_FINAL_V4,
        ReleaseFreezeError,
        _inside,
        _relative,
        frozen_environment,
        label_projection_sha256,
        sha256_file,
        verify_manifest,
    )
except ImportError:  # Imported as ``scripts.run_current_head_regression``.
    from scripts.freeze_current_head_release import (  # type: ignore
        HISTORICAL_FINAL_V4,
        ReleaseFreezeError,
        _inside,
        _relative,
        frozen_environment,
        label_projection_sha256,
        sha256_file,
        verify_manifest,
    )


DEFAULT_FREEZE = (
    ROOT / "docs" / "rag_eval" / "current_head_20260831" / "FROZEN_CONFIG.json"
)
DEFAULT_OUTDIR = (
    Path.home() / ".offerclaw" / "private_eval" / "current_head_20260831" / "runs"
)
DEFAULT_SUMMARY = (
    ROOT / "docs" / "rag_eval" / "current_head_20260831" / "SUMMARY.json"
)


class RegressionRunError(RuntimeError):
    """The frozen regression cannot be run without changing its contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_private_output(path: Path) -> None:
    if _inside(path, ROOT):
        raise RegressionRunError(
            "raw current-head regression output must be outside the repository"
        )


def ensure_summary_output(path: Path) -> None:
    if _inside(path, HISTORICAL_FINAL_V4):
        raise RegressionRunError("current-head summary may not overwrite Final v4")


def install_private_answerability_cache(output: Path) -> Path:
    """Bind the judge cache to this external run instead of the app cache.

    ``rag_answerability`` intentionally keeps a process-global in-memory cache
    and a repository-local persistence path for normal application use.  A
    release evaluation must neither learn from nor write to that mutable app
    state, otherwise the measurement is not isolated and running it changes
    the developer's local product environment.
    """

    cache_path = output.parent / f"{output.stem}.answerability_cache.json"
    ensure_private_output(cache_path)
    if cache_path.exists() and not output.exists():
        raise RegressionRunError(
            "orphaned evaluation cache exists without its raw run; choose a "
            "fresh external outdir"
        )
    import rag_answerability

    rag_answerability.CACHE_PATH = cache_path
    rag_answerability._CACHE = None
    return cache_path


def _dataset_path(manifest: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    raw = str((manifest.get("dataset") or {}).get("path") or "")
    path = Path(raw)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rebind_qrels_in_memory(
    payload: dict[str, Any], *, dataset_contract: dict[str, Any]
) -> dict[str, Any]:
    """Rebind only the index object and prove the labels did not change."""

    expected = str(dataset_contract.get("label_projection_sha256") or "")
    if label_projection_sha256(payload) != expected:
        raise RegressionRunError("qrels label projection differs from the freeze")
    rebound = copy.deepcopy(payload)
    rebound["index"] = copy.deepcopy(dataset_contract["target_index"])
    if label_projection_sha256(rebound) != expected:
        raise RegressionRunError("in-memory index rebind changed qrels labels")
    return rebound


def _load_bound_dataset(
    manifest: dict[str, Any], dataset_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import chromadb
    from rag_qrels_v2 import (
        index_contract_fingerprint,
        load_graded_qrels,
        validate_graded_qrels,
        validate_graded_qrels_against_collection,
    )
    from rag_tools import get_collection_name, index_fingerprint

    contract = manifest["dataset"]
    if sha256_file(dataset_path) != contract["sha256"]:
        raise RegressionRunError("source qrels SHA256 differs from the freeze")
    payload = load_graded_qrels(dataset_path, require_approved=True)
    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name()
    )
    # This exact target/source/excerpt check is what makes re-binding legitimate.
    validate_graded_qrels_against_collection(payload, collection)
    current_index = manifest["index"]
    live = index_fingerprint(collection=collection, cache_ttl=0)
    if str(live.get("fingerprint_id") or "") != current_index["fingerprint_id"]:
        raise RegressionRunError("live index fingerprint differs from the freeze")
    if index_contract_fingerprint(current_index) != current_index["fingerprint"]:
        raise RegressionRunError("frozen target index contract is internally invalid")
    rebound = rebind_qrels_in_memory(payload, dataset_contract=contract)
    validate_graded_qrels(rebound, require_approved=True)
    if index_contract_fingerprint(rebound["index"]) != current_index["fingerprint"]:
        raise RegressionRunError("in-memory qrels did not bind to the frozen index")
    binding = {
        "mode": "validated_in_memory_rebind",
        "source_dataset_sha256": contract["sha256"],
        "source_index_fingerprint": (contract.get("source_index") or {}).get(
            "fingerprint", ""
        ),
        "target_index_fingerprint": current_index["fingerprint"],
        "label_projection_sha256": contract["label_projection_sha256"],
        "labels_unchanged": True,
        "rows": len(rebound["items"]),
    }
    return rebound, binding


def _answerability_sentinel(evaluation: dict[str, Any]) -> dict[str, int]:
    rows = evaluation["positive"]["rows"] + evaluation["negative"]["rows"]
    diagnostics = [
        ((row.get("gate_features") or {}).get("answerability_rerank") or {})
        for row in rows
    ]
    applied = sum(bool(diag.get("applied")) for diag in diagnostics)
    calls = sum(int(diag.get("calls") or 0) for diag in diagnostics)
    graded = sum(int(diag.get("graded") or 0) for diag in diagnostics)
    degraded_rows = sum(
        int(diag.get("graded") or 0) < int(diag.get("calls") or 0)
        for diag in diagnostics
    )
    if applied != len(rows):
        raise RegressionRunError(
            f"answerability sentinel applied on {applied}/{len(rows)} rows"
        )
    if graded != calls or degraded_rows:
        raise RegressionRunError(
            f"answerability judge degraded: graded={graded}, calls={calls}, "
            f"rows={degraded_rows}"
        )
    return {
        "rows": len(rows),
        "applied_rows": applied,
        "calls": calls,
        "graded": graded,
        "degraded_rows": degraded_rows,
    }


def _run_worker(
    *, manifest_path: Path, dataset_path: Path, output: Path,
    run_index: int, max_cases: int,
) -> dict[str, Any]:
    ensure_private_output(output)
    verify_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with frozen_environment(manifest["environment"]):
        cache_path = install_private_answerability_cache(output)
        payload, binding = _load_bound_dataset(manifest, dataset_path)
        if max_cases:
            payload["items"] = payload["items"][:max_cases]

        from eval_colloquial_rag import _evaluate_once
        from rag_retrieval_trace import resolve_retrieval_profile

        profile = resolve_retrieval_profile("baseline")
        if profile.to_dict() != manifest["retrieval_profile"]:
            raise RegressionRunError("resolved production profile differs from freeze")
        evaluation = _evaluate_once(
            payload["items"], profile=profile, run_index=run_index,
            route_mode="oracle",
        )
        sentinel = _answerability_sentinel(evaluation)
        try:
            from rag_answerability import flush_cache

            flush_cache()
        except Exception:
            # Cache persistence cannot turn an otherwise valid retrieval result
            # into a release; the sentinel above still proves every verdict was
            # available during this run.
            pass

    artifact = {
        "schema_version": "offerclaw-current-head-regression-raw-v1",
        "release_status": "frozen_current_head_regression_not_blind",
        "manifest_sha256": sha256_file(manifest_path),
        "run": run_index,
        "max_cases": max_cases or None,
        "qrels_binding": binding,
        "retrieval_profile": manifest["retrieval_profile"],
        "answerability": manifest["answerability"],
        "sentinel": sentinel,
        "private_cache": {
            "file": cache_path.name,
            "sha256": sha256_file(cache_path) if cache_path.is_file() else None,
        },
        "evaluation": evaluation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return artifact


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _run_metrics(
    artifact: dict[str, Any], expected_negative: dict[str, str]
) -> dict[str, Any]:
    evaluation = artifact["evaluation"]
    positive = evaluation["positive"]
    rows = positive["rows"]
    metrics = positive["metrics"]
    negative_rows = evaluation["negative"]["rows"]
    accepted = {
        row["query_id"] for row in negative_rows if row.get("gate_decision")
    }
    abstain = {
        query_id for query_id, behavior in expected_negative.items()
        if behavior == "abstain_from_kb"
    }
    correctable = {
        query_id for query_id, behavior in expected_negative.items()
        if behavior == "correct_premise"
    }
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows[1:]]
    return {
        "run": artifact["run"],
        "r1": metrics["strict_ranking"]["recall@1"]["hits"],
        "r3": metrics["strict_ranking"]["recall@3"]["hits"],
        "r5": metrics["strict_ranking"]["recall@5"]["hits"],
        "mrr": round(statistics.fmean(
            float(row["reciprocal_rank_at_10"]) for row in rows
        ), 4) if rows else 0.0,
        "ndcg5": round(statistics.fmean(
            float(row["ndcg_at_5"]) for row in rows
        ), 4) if rows else 0.0,
        "candidate": metrics["funnel"]["rrf_candidate"]["hits"],
        "gate_pass": metrics["funnel"]["correct_top1_gate_pass"]["hits"],
        "effective": metrics["funnel"]["effective_evidence"]["hits"],
        "p50_ms": round(_percentile(latencies, 0.50)) if latencies else None,
        "p95_ms": round(_percentile(latencies, 0.95)) if latencies else None,
        "false_accept_abstain": sorted(accepted & abstain),
        "correctable_reached": len(accepted & correctable),
        "n_correctable": len(correctable),
        "judge_calls": artifact["sentinel"]["calls"],
    }


def _spread(values: list[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "runs": values,
    }


def build_summary(
    *, manifest_path: Path, manifest: dict[str, Any], artifacts: list[dict[str, Any]],
    raw_paths: list[Path], dataset_payload: dict[str, Any], max_cases: int,
) -> dict[str, Any]:
    expected = {
        item["query_id"]: str(item.get("expected_behavior") or "")
        for item in dataset_payload.get("items", [])
        if item.get("case_kind") == "negative"
    }
    runs = [_run_metrics(artifact, expected) for artifact in artifacts]
    numeric = (
        "r1", "r3", "r5", "mrr", "ndcg5", "candidate", "gate_pass",
        "effective", "correctable_reached", "judge_calls",
    )
    summary = {
        key: _spread([float(run[key]) for run in runs]) for key in numeric
    }
    for key in ("p50_ms", "p95_ms"):
        values = [float(run[key]) for run in runs if run[key] is not None]
        summary[key] = _spread(values) if values else None
    summary["false_accept_abstain"] = {
        "median": statistics.median(
            len(run["false_accept_abstain"]) for run in runs
        ),
        "runs": [run["false_accept_abstain"] for run in runs],
    }
    return {
        "schema_version": "offerclaw-current-head-regression-summary-v1",
        "release_status": "frozen_current_head_regression_not_blind",
        "manifest": {
            "path": _relative(manifest_path),
            "sha256": sha256_file(manifest_path),
            "git_head": manifest["git"]["head"],
            "index_fingerprint_id": manifest["index"]["fingerprint_id"],
        },
        "dataset": {
            "path": manifest["dataset"]["path"],
            "sha256": manifest["dataset"]["sha256"],
            "binding_mode": "validated_in_memory_rebind",
            "labels_unchanged": True,
            "label_projection_sha256": manifest["dataset"][
                "label_projection_sha256"
            ],
            "max_cases": max_cases or None,
        },
        "configuration": {
            "profile_source": "production_registry",
            "retrieval_profile": manifest["retrieval_profile"],
            "answerability": manifest["answerability"],
            "gate": manifest["gate"],
        },
        "repeats": len(runs),
        "summary": summary,
        "runs": runs,
        "private_artifacts": [
            {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in raw_paths
        ],
        "privacy": {
            "contains_raw_rows": False,
            "contains_questions": False,
            "contains_chunk_text": False,
        },
    }


def _load_existing_raw(
    path: Path, manifest_sha: str, run_index: int, max_cases: int,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "offerclaw-current-head-regression-raw-v1":
        raise RegressionRunError(f"existing raw artifact has wrong schema: {path}")
    if artifact.get("manifest_sha256") != manifest_sha:
        raise RegressionRunError(f"existing raw artifact belongs to another freeze: {path}")
    if int(artifact.get("run") or 0) != run_index:
        raise RegressionRunError(f"existing raw artifact has wrong run number: {path}")
    if artifact.get("max_cases") != (max_cases or None):
        raise RegressionRunError(
            f"existing raw artifact was produced with another max-cases: {path}"
        )
    _answerability_sentinel(artifact["evaluation"])
    return artifact


def run_parent(args: argparse.Namespace) -> int:
    manifest_path = args.freeze.expanduser().resolve()
    verify_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = _dataset_path(manifest, args.dataset)
    if sha256_file(dataset_path) != manifest["dataset"]["sha256"]:
        raise RegressionRunError("dataset override differs from frozen dataset")
    outdir = args.outdir.expanduser().resolve()
    ensure_private_output(outdir)
    summary_path = args.summary.expanduser().resolve()
    ensure_summary_output(summary_path)
    if args.repeats < 1:
        raise RegressionRunError("repeats must be positive")
    if args.max_cases < 0:
        raise RegressionRunError("max-cases cannot be negative")

    manifest_sha = sha256_file(manifest_path)
    artifacts: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    for run_index in range(1, args.repeats + 1):
        output = outdir / f"run{run_index}.json"
        if output.exists():
            artifact = _load_existing_raw(
                output, manifest_sha, run_index, args.max_cases
            )
        else:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--freeze", str(manifest_path),
                "--dataset", str(dataset_path),
                "--worker-output", str(output),
                "--run-index", str(run_index),
                "--max-cases", str(args.max_cases),
            ]
            env = dict(os.environ)
            env.update(manifest["environment"])
            env.update({
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MODELSCOPE_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "LLM_USAGE_LOG": "0",
            })
            proc = subprocess.run(command, cwd=ROOT, env=env)
            if proc.returncode != 0:
                raise RegressionRunError(f"worker run {run_index} failed")
            artifact = _load_existing_raw(
                output, manifest_sha, run_index, args.max_cases
            )
        artifacts.append(artifact)
        raw_paths.append(output)

    from rag_qrels_v2 import load_graded_qrels

    dataset_payload = load_graded_qrels(dataset_path, require_approved=True)
    if args.max_cases:
        dataset_payload["items"] = dataset_payload["items"][:args.max_cases]
    summary = build_summary(
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=artifacts,
        raw_paths=raw_paths,
        dataset_payload=dataset_payload,
        max_cases=args.max_cases,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "completed",
        "summary": _relative(summary_path),
        "repeats": args.repeats,
        "private_artifacts": len(raw_paths),
    }, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--run-index", type=int, default=1, help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.worker:
            if args.worker_output is None:
                raise RegressionRunError("worker-output is required")
            manifest_path = args.freeze.expanduser().resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_path = _dataset_path(manifest, args.dataset)
            _run_worker(
                manifest_path=manifest_path,
                dataset_path=dataset_path,
                output=args.worker_output.expanduser().resolve(),
                run_index=args.run_index,
                max_cases=args.max_cases,
            )
            return 0
        return run_parent(args)
    except (OSError, ValueError, ReleaseFreezeError, RegressionRunError) as exc:
        print(f"[current-head-regression] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
