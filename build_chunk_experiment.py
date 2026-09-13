#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build/dry-run isolated C1/C2 Stage-D chunk collections.

Examples:

  # No Chroma write, no embeddings (safe default):
  .venv/bin/python build_chunk_experiment.py --profile c1 --dry-run \
      --qrels docs/rag_eval/qrels/rag_bench_paraphrase_adjudicated.json

  # Explicitly create/resume the fingerprinted experiment collection:
  .venv/bin/python build_chunk_experiment.py --profile c1 --build \
      --qrels docs/rag_eval/qrels/rag_bench_paraphrase_adjudicated.json \
      --output-dir logs/rag_eval/chunk_experiments/c1

The command never deletes a collection and refuses any name without the
``offerclaw_exp_chunk_`` prefix plus its content fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("c1", "c2"), required=True)
    parser.add_argument(
        "--baseline-collection",
        default="",
        help="explicit immutable C0 collection; defaults to the active production collection",
    )
    parser.add_argument(
        "--corpus-mode",
        choices=("snapshot_targeted", "mixed_targeted", "snapshot_children", "source_rebuild"),
        default="snapshot_targeted",
        help=(
            "snapshot_targeted replaces only qrels-blind defective immutable C0 chunks; "
            "mixed_targeted rebuilds whole selected live sources and is confounded; "
            "snapshot_children is the prior full-C1 control; source_rebuild is confounded"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: plan/map only; no Chroma write")
    mode.add_argument("--build", action="store_true", help="explicitly create/resume isolated collection")
    parser.add_argument("--files", nargs="*", type=Path,
                        help="optional explicit Markdown corpus; default mirrors production source membership")
    parser.add_argument(
        "--target-source",
        action="append",
        default=[],
        help=(
            "development diagnostic only: explicitly replace this source basename; "
            "sets calibration_leakage=true and blocks promotion"
        ),
    )
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "rag_eval" / "chunk_experiments")
    parser.add_argument("--tokenizer", choices=("active", "regex"), default="active")
    parser.add_argument("--allow-network-tokenizer", action="store_true")
    parser.add_argument(
        "--reuse-embedding-collection",
        default="",
        help=(
            "optional prior isolated experiment collection; identical child IDs/documents "
            "with the same embedding contract reuse vectors, all other children embed normally"
        ),
    )
    args = parser.parse_args()
    if not args.build:
        args.dry_run = True
    if args.build and args.tokenizer == "regex":
        parser.error("--build requires --tokenizer active")
    if args.files and args.corpus_mode != "source_rebuild":
        parser.error("--files is only valid with --corpus-mode source_rebuild")
    if args.target_source and args.corpus_mode != "mixed_targeted":
        parser.error("--target-source is only valid with --corpus-mode mixed_targeted")
    return args


def main() -> None:
    args = _args()
    import chromadb
    from rag_chunk_experiments import (
        EXPERIMENT_COLLECTION_PREFIX,
        EXPERIMENT_SPECS,
        QrelsMappingError,
        RegexTokenOffsetCodec,
        SourceDocument,
        build_experiment_plan,
        build_mixed_targeted_experiment_plan,
        build_snapshot_experiment_plan,
        build_snapshot_targeted_experiment_plan,
        load_active_token_codec,
        map_qrels_to_experiment,
        production_sources,
        select_mixed_target_sources,
        select_snapshot_target_chunks,
        write_experiment_collection,
    )
    from rag_tools import get_collection_name

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    production_name = args.baseline_collection.strip() or get_collection_name()
    if production_name.startswith(EXPERIMENT_COLLECTION_PREFIX):
        raise SystemExit(
            "baseline collection cannot be another chunk experiment; "
            "pass --baseline-collection with the explicit C0 collection"
        )
    production = client.get_collection(production_name)
    codec = (
        RegexTokenOffsetCodec() if args.tokenizer == "regex"
        else load_active_token_codec(local_files_only=not args.allow_network_tokenizer)
    )
    source_resolution = {}
    target_selection = {}
    sources = []
    if args.corpus_mode == "source_rebuild" and args.files:
        sources = [SourceDocument(path=str(path.resolve()), source=path.name) for path in args.files]
    elif args.corpus_mode == "source_rebuild":
        sources = production_sources(ROOT, production, resolution_report=source_resolution)
    elif args.corpus_mode == "mixed_targeted":
        selected_sources, target_selection = select_mixed_target_sources(
            production, codec, explicit_sources=args.target_source,
        )
        sources = production_sources(
            ROOT,
            production,
            resolution_report=source_resolution,
            include_sources=selected_sources,
        )
    selected_chunk_ids = ()
    if args.corpus_mode == "snapshot_targeted":
        selected_chunk_ids, target_selection = select_snapshot_target_chunks(
            production, codec,
        )
        plan = build_snapshot_targeted_experiment_plan(
            production,
            selected_chunk_ids,
            target_selection,
            EXPERIMENT_SPECS[args.profile],
            codec,
            production_collection=production_name,
        )
    elif args.corpus_mode == "snapshot_children":
        plan = build_snapshot_experiment_plan(
            production,
            EXPERIMENT_SPECS[args.profile],
            codec,
            production_collection=production_name,
        )
    elif args.corpus_mode == "source_rebuild":
        plan = build_experiment_plan(
            sources,
            EXPERIMENT_SPECS[args.profile],
            codec,
            production_collection=production_name,
        )
    else:
        plan = build_mixed_targeted_experiment_plan(
            production,
            sources,
            target_selection,
            EXPERIMENT_SPECS[args.profile],
            codec,
            production_collection=production_name,
        )
    output_dir = args.output_dir / f"{args.profile}_{plan.experiment_fingerprint[7:19]}"
    manifest = plan.manifest()
    drift_sources = list(source_resolution.get("content_drift_sources") or [])
    selection_leakage = bool(target_selection.get("calibration_leakage"))
    manifest["promotion_eligible_corpus"] = (
        args.corpus_mode in {"snapshot_children", "snapshot_targeted"}
        or (
            args.corpus_mode == "mixed_targeted"
            and not selection_leakage
            and not drift_sources
        )
    )
    manifest["confounded"] = (
        args.corpus_mode == "source_rebuild" or selection_leakage or bool(drift_sources)
    )
    manifest["calibration_leakage"] = selection_leakage
    if args.corpus_mode in {"snapshot_children", "snapshot_targeted"}:
        # Immutable C0 modes never read live source files, so drift is not an
        # unmeasured variable.  Record the zero explicitly for audit tooling.
        manifest["content_drift_sources"] = []
    if args.corpus_mode in {"source_rebuild", "mixed_targeted"} and not args.files:
        manifest["source_resolution"] = {
            key: value for key, value in source_resolution.items() if key != "sources"
        }
        manifest["content_drift_sources"] = drift_sources
    _write_json(output_dir / "manifest.json", manifest)
    if args.corpus_mode in {"source_rebuild", "mixed_targeted"} and not args.files:
        _write_json(output_dir / "source_resolution.json", source_resolution)
    if args.corpus_mode in {"mixed_targeted", "snapshot_targeted"}:
        _write_json(output_dir / "target_selection.json", target_selection)
    print(json.dumps({
        "mode": "build" if args.build else "dry-run",
        "corpus_mode": args.corpus_mode,
        "collection": plan.collection_name,
        "fingerprint": plan.experiment_fingerprint,
        "sources": len(plan.sources),
        "chunks": len(plan.chunks),
        "tokenizer": plan.tokenizer,
        "statistics": manifest["statistics"],
        "content_drift_source_count": len(manifest.get("content_drift_sources") or []),
        "promotion_eligible_corpus": manifest["promotion_eligible_corpus"],
        "calibration_leakage": manifest.get("calibration_leakage", False),
        "target_source_count": len(plan.target_sources),
        "target_sources": list(plan.target_sources),
        "target_chunk_count": len(
            target_selection.get("selected_chunk_ids") or []
        ),
        "selector_hash": target_selection.get("selector_hash"),
    }, ensure_ascii=False, indent=2))

    if args.qrels:
        from rag_qrels import load_qrels_overlay
        payload = load_qrels_overlay(args.qrels)
        try:
            mapped, report = map_qrels_to_experiment(payload, plan, strict=True)
        except QrelsMappingError as exc:
            _write_json(output_dir / "qrels_mapping_report.json", exc.report)
            print(f"QRELS MAPPING FAILED: {exc}", file=sys.stderr)
            raise SystemExit(2)
        _write_json(output_dir / "qrels_mapped.json", mapped)
        _write_json(output_dir / "qrels_mapping_report.json", report)
        print(f"qrels mapped: {report['mapped_target_count']} targets")

    if args.build:
        state = write_experiment_collection(
            client,
            plan,
            reuse_embedding_collection=args.reuse_embedding_collection.strip(),
        )
        _write_json(output_dir / "build_result.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print("dry-run complete: Chroma and embeddings were not touched")


if __name__ == "__main__":
    main()
