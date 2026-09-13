#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the production retrieval harness against one isolated chunk collection.

This is intentionally a thin, fail-closed wrapper around the existing
``eval_reranker_profiles.py`` child arm.  It validates the manifest/qrels
fingerprint and then starts a fresh process with ``RAG_COLLECTION_NAME`` set to
the isolated collection.  It does not compare scores across different qrels
overlays automatically; C0/C1/C2 reports must show file-level and remapped
chunk-level metrics separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sets", default="heldout52")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--reranker-breadcrumb", choices=("0", "1"), default="1",
                        help="1 runs B0 (reranker sees breadcrumb); 0 runs A0 body-only")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--candidate-miss-audit", type=Path,
                        help="optional Stage-C miss audit; writes per-query recovery details")
    parser.add_argument(
        "--allow-confounded-diagnostic",
        action="store_true",
        help="allow source_rebuild for diagnosis only; never treat it as a promotable C1/C2 arm",
    )
    return parser.parse_args()


def validate_evaluation_inputs(
    manifest: dict,
    qrels: dict,
    *,
    collection_metadata: dict,
    collection_count: int,
    allow_confounded_diagnostic: bool = False,
) -> tuple[str, str, int]:
    """Fail closed when a manifest, qrels and live collection are mixed."""

    collection = str(manifest.get("collection_name") or "")
    fingerprint = str(manifest.get("experiment_fingerprint") or "")
    corpus_mode = str(manifest.get("corpus_mode") or "")
    if not collection.startswith("offerclaw_exp_chunk_"):
        raise ValueError("manifest is not an isolated chunk experiment")
    confounded = (
        bool(manifest.get("confounded"))
        or corpus_mode not in {"snapshot_children", "snapshot_targeted", "mixed_targeted"}
        or manifest.get("promotion_eligible_corpus") is not True
    )
    if confounded and not allow_confounded_diagnostic:
        raise ValueError(
            "confounded source_rebuild cannot be evaluated as a promotion candidate; "
            "use --allow-confounded-diagnostic only for a separately labelled diagnosis"
        )
    if qrels.get("index", {}).get("collection") != collection:
        raise ValueError("qrels collection does not match manifest")
    if qrels.get("index", {}).get("fingerprint") != fingerprint:
        raise ValueError("qrels fingerprint does not match manifest")
    count = int(manifest.get("chunk_count") or 0)
    if count < 1 or qrels.get("index", {}).get("count") != count:
        raise ValueError("manifest/qrels count mismatch")
    if collection_count != count:
        raise ValueError("live collection count does not match manifest")
    expected_metadata = {
        "schema_version": manifest.get("schema_version"),
        "corpus_mode": corpus_mode,
        "experiment_fingerprint": fingerprint,
        "chunker_version": manifest.get("chunker_version"),
        "production_collection": manifest.get("production_collection"),
        "tokenizer": manifest.get("tokenizer"),
        "embed_profile": (manifest.get("embedding_contract") or {}).get("embed_profile"),
        "embedding_contract_json": json.dumps(
            manifest.get("embedding_contract") or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for key, expected in expected_metadata.items():
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"manifest lacks {key}")
        if collection_metadata.get(key) != expected:
            raise ValueError(f"live collection {key} does not match manifest")
    return collection, fingerprint, count


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parent_expandable_child_ids(qrels: dict, collection) -> dict[str, set[str]]:
    """Map reviewed C0 parent identities to all retrievable C1/C2 children."""

    requested: dict[str, set[str]] = {}
    for item in qrels.get("items") or []:
        origins = {
            str(target.get("origin_chunk_id") or "")
            for target in item.get("relevant_targets") or []
            if target.get("relevance") == "direct"
            and target.get("evidence_scope") == "parent_expand_required"
        } - {""}
        if origins:
            requested[item["query_id"]] = origins
    if not requested:
        return {}
    snapshot = collection.get(include=["metadatas"])
    origin_by_child = {
        str(chunk_id): str((metadata or {}).get("origin_chunk_id") or "")
        for chunk_id, metadata in zip(
            snapshot.get("ids") or [], snapshot.get("metadatas") or [],
        )
    }
    return {
        query_id: {
            child_id for child_id, origin in origin_by_child.items() if origin in origins
        }
        for query_id, origins in requested.items()
    }


def _append_parent_expandable_metrics(
    result: dict,
    parent_targets: dict[str, set[str]],
) -> dict:
    """Report parent-expandable retrieval separately from strict child Direct."""

    for evaluated_set in result.get("sets") or []:
        for run in evaluated_set.get("runs") or []:
            rows = run.get("rows") or []
            scoped_rows = []
            for row in rows:
                target_ids = parent_targets.get(str(row.get("id") or ""), set())
                if not target_ids:
                    continue
                final_ids = list(row.get("final_chunk_ids") or [])
                fusion_ids = list(row.get("fusion_chunk_ids") or [])
                final_rank = next(
                    (rank for rank, chunk_id in enumerate(final_ids, 1) if chunk_id in target_ids),
                    0,
                )
                candidate_rank = next(
                    (rank for rank, chunk_id in enumerate(fusion_ids, 1) if chunk_id in target_ids),
                    0,
                )
                row["parent_expand_required_target_child_ids"] = sorted(target_ids)
                row["parent_expandable_rank"] = final_rank
                row["parent_expandable_candidate_rank"] = candidate_rank
                scoped_rows.append((final_rank, candidate_rank))
            count = len(scoped_rows)
            run["parent_expandable_qrels_metrics"] = {
                "query_count": count,
                "retrieved_at_1": sum(final_rank == 1 for final_rank, _ in scoped_rows),
                "retrieved_at_5": sum(0 < final_rank <= 5 for final_rank, _ in scoped_rows),
                "candidate_recovered_at_20": sum(bool(candidate_rank) for _, candidate_rank in scoped_rows),
                "parent_expansion_applied": False,
                "interpretation": (
                    "separate diagnostic: a matching child can expand to the reviewed immutable "
                    "parent; these rows are excluded from strict child Direct metrics"
                ),
            }
    result.setdefault("qrels_overlay", {})["parent_expand_scope_reported_separately"] = True
    return result


def main() -> None:
    args = _args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    import chromadb

    collection_name = str(manifest.get("collection_name") or "")
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    try:
        live_collection = client.get_collection(collection_name)
    except Exception as exc:
        raise SystemExit(f"isolated collection is unavailable: {collection_name}") from exc
    try:
        collection, fingerprint, count = validate_evaluation_inputs(
            manifest,
            qrels,
            collection_metadata=dict(live_collection.metadata or {}),
            collection_count=live_collection.count(),
            allow_confounded_diagnostic=args.allow_confounded_diagnostic,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    from rag_qrels import validate_qrels_against_collection

    try:
        parent_collection = client.get_collection(str(manifest["production_collection"]))
        validate_qrels_against_collection(
            qrels, live_collection, parent_collection=parent_collection,
        )
    except Exception as exc:
        raise SystemExit(f"scope-aware qrels validation failed: {exc}") from exc

    arm = "B0" if args.reranker_breadcrumb == "1" else "A0"
    command = [
        sys.executable,
        str(ROOT / "eval_reranker_profiles.py"),
        "--_child-arm", arm,
        "--sets", args.sets,
        "--repeat", str(args.repeat),
        "--expected-index-count", str(count),
        "--device", args.device,
        "--qrels-overlay", str(args.qrels.resolve()),
        "--output", str(args.output.resolve()),
    ]
    if args.gate:
        command.append("--gate")
    env = dict(os.environ)
    env["RAG_COLLECTION_NAME"] = collection
    env["RAG_EVAL_CHUNKER_VERSION"] = str(manifest["chunker_version"])
    # Freeze all retrieval enhancements exactly as Stage-D specifies.
    env.update({
        "RAG_HYDE": "0",
        "RAG_QUERY_REWRITE": "0",
        "RAG_DOC2QUERY": "0",
        "RAG_EN_QUOTA": "0",
        "RAG_CONCEPT_ALIAS": "0",
        "RAG_RERANK_BRIDGE": "0",
    })
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    result = json.loads(args.output.read_text(encoding="utf-8"))
    parent_targets = _parent_expandable_child_ids(qrels, live_collection)
    result = _append_parent_expandable_metrics(result, parent_targets)
    _atomic_json_write(args.output, result)
    if args.candidate_miss_audit:
        audit = json.loads(args.candidate_miss_audit.read_text(encoding="utf-8"))
        evaluated = result.get("sets") or []
        if not evaluated or not evaluated[0].get("runs"):
            raise SystemExit("evaluation result has no rows for recovery audit")
        result_rows = {
            row["id"]: row for row in evaluated[0]["runs"][0].get("rows", [])
        }
        recovery_rows = []
        for old in audit.get("rows") or []:
            row = result_rows.get(old.get("id"))
            recovery_rows.append({
                "id": old.get("id"),
                "old_flags": list(old.get("flags") or []),
                "old_dense_rank_at_depth": old.get("dense_rank_at_depth"),
                "old_bm25_rank_at_depth": old.get("bm25_rank_at_depth"),
                "evaluated": row is not None,
                "new_direct_candidate_rank": row.get("direct_candidate_rank") if row else None,
                "new_direct_rank": row.get("direct_rank") if row else None,
                "strict_child_candidate_recovered_at_20": bool(row and row.get("direct_candidate_rank")),
                "strict_child_top1_recovered": bool(row and row.get("direct_rank") == 1),
                "parent_expandable_candidate_rank": row.get("parent_expandable_candidate_rank") if row else None,
                "parent_expandable_rank": row.get("parent_expandable_rank") if row else None,
                "parent_expandable_candidate_recovered_at_20": bool(
                    row and row.get("parent_expandable_candidate_rank")
                ),
                "parent_expandable_top1_recovered": bool(
                    row and row.get("parent_expandable_rank") == 1
                ),
                "new_target_chunk_ids": list(row.get("direct_target_chunk_ids") or []) if row else [],
                "new_fusion_chunk_ids": list(row.get("fusion_chunk_ids") or []) if row else [],
                "new_final_chunk_ids": list(row.get("final_chunk_ids") or []) if row else [],
            })
        payload = {
            "schema_version": "chunk-candidate-recovery-v1",
            "collection": collection,
            "experiment_fingerprint": fingerprint,
            "audit_path": str(args.candidate_miss_audit),
            "audited_query_count": len(recovery_rows),
            "evaluated_query_count": sum(row["evaluated"] for row in recovery_rows),
            "strict_child_candidate_recovered_at_20_count": sum(
                row["strict_child_candidate_recovered_at_20"] for row in recovery_rows
            ),
            "strict_child_top1_recovered_count": sum(
                row["strict_child_top1_recovered"] for row in recovery_rows
            ),
            "parent_expandable_candidate_recovered_at_20_count": sum(
                row["parent_expandable_candidate_recovered_at_20"] for row in recovery_rows
            ),
            "parent_expandable_top1_recovered_count": sum(
                row["parent_expandable_top1_recovered"] for row in recovery_rows
            ),
            "rows": recovery_rows,
        }
        recovery_path = args.output.with_suffix(args.output.suffix + ".candidate_recovery.json")
        _atomic_json_write(recovery_path, payload)


if __name__ == "__main__":
    main()
