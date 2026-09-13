"""Development-only hard-negative dataset builder for the held-out 52 set.

This module deliberately sits outside the production retrieval path.  It turns
an audited, production-equivalent A0 trace into reranker training triples, but
only when the reviewer-approved *direct* answer chunk was already present in
the fixed fusion pool.  Candidate-absent questions are reported for Stage C and
never mislabeled as a reranker failure.

The default artifact contains IDs and provenance only.  Question and corpus
text are included solely when ``include_text=True`` is explicitly requested.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from rag_qrels import load_qrels_overlay, validate_qrels_against_collection


ROOT = Path(__file__).resolve().parent
IMMUTABLE_SET = ROOT / "tests" / "rag_bench_paraphrase_set.json"
IMMUTABLE_SET_REF = "tests/rag_bench_paraphrase_set.json"
SCHEMA_VERSION = "reranker-hard-negatives-v1"
REFERENCE_PLAN = {
    "decision": "answer",
    "routes": [{"source": "reference_kb", "operation": "search"}],
}


class HardNegativeBuildError(ValueError):
    """Raised when an input cannot support an auditable Stage-B dataset."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_immutable_items(path: str | Path = IMMUTABLE_SET) -> list[dict[str, Any]]:
    payload = _load_json(path)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != 52:
        raise HardNegativeBuildError("the immutable development set must contain 52 items")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("id") or not item.get("q"):
            raise HardNegativeBuildError("the immutable development set contains an invalid item")
        if item["id"] in seen:
            raise HardNegativeBuildError(f"duplicate query id: {item['id']}")
        seen.add(item["id"])
    return items


def _heldout_rows(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if baseline.get("status") not in {None, "completed"}:
        raise HardNegativeBuildError("baseline evaluation is not complete")
    if baseline.get("arm") != "A0":
        raise HardNegativeBuildError("baseline artifact must be the fixed A0 arm")
    profile = baseline.get("profile") or {}
    forbidden = (
        "enable_hyde", "enable_query_rewrite", "enable_doc2query",
        "enable_quota", "enable_alias", "enable_rerank_bridge",
    )
    if int(profile.get("pool_size") or 0) != 20:
        raise HardNegativeBuildError("A0 pool_size must be exactly 20")
    if profile.get("reranker_model") != "BAAI/bge-reranker-base":
        raise HardNegativeBuildError("A0 must use BAAI/bge-reranker-base")
    if bool(profile.get("reranker_use_breadcrumb")):
        raise HardNegativeBuildError("A0 must be the body-only baseline")
    if any(bool(profile.get(name)) for name in forbidden):
        raise HardNegativeBuildError("A0 contains a forbidden retrieval enhancement")

    heldout = next(
        (entry for entry in baseline.get("sets") or []
         if entry.get("set") == "heldout52"),
        None,
    )
    if not heldout or not heldout.get("runs"):
        raise HardNegativeBuildError("baseline artifact has no heldout52 run")
    if heldout.get("status") not in {None, "completed"}:
        raise HardNegativeBuildError("baseline heldout52 run is not complete")
    rows = heldout["runs"][0].get("rows") or []
    if len(rows) != 52:
        raise HardNegativeBuildError("baseline heldout52 run must contain 52 rows")
    return {str(row.get("id")): row for row in rows}


def _fetch_chunks(collection: Any, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = sorted({str(value) for value in chunk_ids if value})
    if not ids:
        return {}
    snapshot = collection.get(ids=ids, include=["documents", "metadatas"])
    returned = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    if not (len(returned) == len(documents) == len(metadatas)):
        raise HardNegativeBuildError("collection returned misaligned chunk fields")
    result = {
        str(chunk_id): {
            "chunk_id": str(chunk_id),
            "document": str(document or ""),
            "metadata": dict(metadata or {}),
        }
        for chunk_id, document, metadata in zip(returned, documents, metadatas)
    }
    missing = sorted(set(ids) - set(result))
    if missing:
        raise HardNegativeBuildError(f"baseline references missing chunk ids: {missing}")
    return result


def production_scoring_text(document: str, metadata: dict[str, Any]) -> str:
    """Use the exact production title/breadcrumb/body formatter.

    Keeping the import inside this small adapter prevents training code from
    drifting to a subtly different separator, path-sanitisation rule, or
    heading fallback than the online A3/B0 scorer.
    """

    from rag_gate import _rerank_pair_documents

    formatted = _rerank_pair_documents(
        [document], [metadata], use_breadcrumb=True,
    )
    if not formatted or len(formatted) != 1:
        raise HardNegativeBuildError("production reranker formatter returned no text")
    return formatted[0]


def _chunk_ref(row: dict[str, Any], *, include_text: bool) -> dict[str, Any]:
    metadata = row["metadata"]
    source = Path(str(metadata.get("source") or "")).name
    output = {
        "chunk_id": row["chunk_id"],
        "source": source,
        "heading_path": str(
            metadata.get("heading_path") or metadata.get("section_path")
            or metadata.get("title") or ""
        ),
        "document_hash": _sha256_text(row["document"]),
    }
    if include_text:
        output["scoring_text"] = production_scoring_text(
            row["document"], metadata,
        )
    return output


def _row_has_ids(row: dict[str, Any]) -> bool:
    return bool(row.get("final_chunk_ids") and row.get("fusion_chunk_ids"))


def _a0_profile():
    from rag_retrieval_trace import RetrievalProfile
    from rag_tools import CHUNKER_VERSION

    return RetrievalProfile(
        name="A0",
        pool_size=20,
        reranker_model="BAAI/bge-reranker-base",
        reranker_use_breadcrumb=False,
        chunker_version=CHUNKER_VERSION,
        enable_hyde=False,
        enable_query_rewrite=False,
        enable_doc2query=False,
        enable_quota=False,
        enable_alias=False,
        enable_rerank_bridge=False,
    )


def reconstruct_trace_row(
    item: dict[str, Any],
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    """Re-run the same read-only A0 harness when an old result lacks IDs.

    The supplied baseline remains the experiment record.  Reconstruction is
    accepted only if its top source agrees, preventing a changed index or code
    path from being silently combined with an older result.
    """

    from rag_gate import retrieve_with_trace

    trace = retrieve_with_trace(
        item["q"], REFERENCE_PLAN, _a0_profile(), top_k=5,
    )
    top_source = trace.final_candidates[0].source if trace.final_candidates else ""
    recorded_source = str(baseline_row.get("top1_source") or "")
    if recorded_source and top_source != recorded_source:
        raise HardNegativeBuildError(
            f"{item['id']}: reconstructed top1 source changed "
            f"({recorded_source!r} -> {top_source!r})"
        )
    return {
        **baseline_row,
        "final_chunk_ids": [candidate.chunk_id for candidate in trace.final_candidates],
        "fusion_chunk_ids": [candidate.chunk_id for candidate in trace.fusion_candidates],
        "reconstructed_trace": True,
        "reconstructed_index_fingerprint": trace.index_fingerprint,
    }


def build_hard_negative_dataset(
    *,
    qrels: dict[str, Any],
    items: list[dict[str, Any]],
    baseline: dict[str, Any],
    collection: Any,
    include_text: bool = False,
    reconstruct_missing: bool = False,
    include_stability_anchors: bool = True,
    trace_resolver: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    qrels_sha256: str = "",
    baseline_sha256: str = "",
) -> dict[str, Any]:
    """Build audited Stage-B triples from already-recalled direct chunks."""

    if qrels.get("base_set") != IMMUTABLE_SET_REF:
        raise HardNegativeBuildError("private/blind or unknown qrels are forbidden")
    if len(items) != 52:
        raise HardNegativeBuildError("hard negatives may only use the immutable 52 set")
    item_by_id = {str(item["id"]): item for item in items}
    if len(item_by_id) != 52:
        raise HardNegativeBuildError("immutable set query ids must be unique")
    validate_qrels_against_collection(qrels, collection)
    if qrels["index"]["count"] != collection.count():
        raise HardNegativeBuildError(
            "qrels collection count differs from the live collection"
        )
    baseline_index = baseline.get("index") or {}
    if baseline_index.get("collection") != qrels["index"]["collection"]:
        raise HardNegativeBuildError("baseline and qrels collection names differ")
    if int(baseline_index.get("collection_count") or -1) != collection.count():
        raise HardNegativeBuildError("baseline index count differs from the live collection")

    baseline_rows = _heldout_rows(baseline)
    if set(baseline_rows) != set(item_by_id):
        raise HardNegativeBuildError("baseline query coverage differs from the immutable set")
    resolver = trace_resolver or reconstruct_trace_row

    triples: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    reconstructed_ids: list[str] = []
    for qrel_item in qrels["items"]:
        query_id = str(qrel_item["query_id"])
        item = item_by_id[query_id]
        outcome = qrel_item["review_outcome"]
        if outcome not in {"accepted", "partial"}:
            exclusions.append({"query_id": query_id, "reason": "unsupported_qrels"})
            continue

        direct_ids = {
            target["chunk_id"] for target in qrel_item["relevant_targets"]
            if target["relevance"] == "direct"
        }
        supporting_ids = {
            target["chunk_id"] for target in qrel_item["relevant_targets"]
            if target["relevance"] == "supporting"
        }
        row = baseline_rows[query_id]
        if str(row.get("question") or item["q"]) != item["q"]:
            raise HardNegativeBuildError(f"{query_id}: baseline question differs from immutable set")
        if not _row_has_ids(row):
            if not reconstruct_missing:
                raise HardNegativeBuildError(
                    f"{query_id}: baseline artifact lacks chunk IDs; rerun the A0 harness "
                    "or pass --reconstruct-missing-traces"
                )
            row = resolver(item, row)
            reconstructed_ids.append(query_id)

        fusion_ids = [str(value) for value in row.get("fusion_chunk_ids") or []]
        final_ids = [str(value) for value in row.get("final_chunk_ids") or []]
        if not final_ids:
            exclusions.append({"query_id": query_id, "reason": "no_final_candidate"})
            continue
        recalled_direct = [chunk_id for chunk_id in fusion_ids if chunk_id in direct_ids]
        if not recalled_direct:
            exclusions.append({"query_id": query_id, "reason": "candidate_absent_stage_c"})
            continue
        false_winner_id = final_ids[0]
        needed_ids = set(fusion_ids) | direct_ids | supporting_ids | {false_winner_id}
        chunks = _fetch_chunks(collection, needed_ids)
        known_relevant_ids = direct_ids | supporting_ids
        question_hash = _sha256_text(item["q"])

        # Stability anchors are deliberately separate from the false-winner
        # correction set.  They teach the development model not to sacrifice a
        # qrels-verified Top1 while fixing other rows, and the grouped split
        # keeps every anchor for a query on the same side of validation.
        if false_winner_id in direct_ids:
            if not include_stability_anchors:
                exclusions.append({"query_id": query_id, "reason": "already_direct_top1"})
                continue
            anchor_negative_id = next((
                candidate_id for candidate_id in fusion_ids
                if candidate_id not in direct_ids
            ), None)
            if not anchor_negative_id:
                exclusions.append({
                    "query_id": query_id,
                    "reason": "stability_anchor_no_non_direct_candidate",
                })
                continue
            positive = chunks[false_winner_id]
            negative = chunks[anchor_negative_id]
            anchor = {
                "query_id": query_id,
                "question_hash": question_hash,
                "review_outcome": outcome,
                "triple_id": _sha256_text(
                    f"{query_id}\0{false_winner_id}\0{anchor_negative_id}\0stability"
                ),
                "negative_kind": "stability_anchor",
                "positive": _chunk_ref(positive, include_text=include_text),
                "positive_fusion_rank": (
                    fusion_ids.index(false_winner_id) + 1
                    if false_winner_id in fusion_ids else None
                ),
                "negative": _chunk_ref(negative, include_text=include_text),
                "negative_fusion_rank": fusion_ids.index(anchor_negative_id) + 1,
                "negative_qrels_relevance": (
                    "supporting" if anchor_negative_id in supporting_ids else "none"
                ),
            }
            if include_text:
                anchor["query"] = item["q"]
            triples.append(anchor)
            continue

        false_winner = chunks[false_winner_id]

        for positive_id in recalled_direct:
            positive = chunks[positive_id]
            positive_source = Path(
                str(positive["metadata"].get("source") or "")
            ).name
            positive_rank = fusion_ids.index(positive_id) + 1
            common = {
                "query_id": query_id,
                "question_hash": question_hash,
                "review_outcome": outcome,
                "positive": _chunk_ref(positive, include_text=include_text),
                "positive_fusion_rank": positive_rank,
            }
            if include_text:
                common["query"] = item["q"]

            triples.append({
                **common,
                "triple_id": _sha256_text(
                    f"{query_id}\0{positive_id}\0{false_winner_id}\0false_winner"
                ),
                "negative_kind": "current_false_winner",
                "negative": _chunk_ref(false_winner, include_text=include_text),
                "negative_fusion_rank": (
                    fusion_ids.index(false_winner_id) + 1
                    if false_winner_id in fusion_ids else None
                ),
                "negative_qrels_relevance": (
                    "supporting" if false_winner_id in supporting_ids else "none"
                ),
            })

            same_document_id = next((
                candidate_id for candidate_id in fusion_ids
                if candidate_id != false_winner_id
                and candidate_id not in known_relevant_ids
                and Path(str(chunks[candidate_id]["metadata"].get("source") or "")).name
                == positive_source
            ), None)
            if same_document_id:
                triples.append({
                    **common,
                    "triple_id": _sha256_text(
                        f"{query_id}\0{positive_id}\0{same_document_id}\0same_document"
                    ),
                    "negative_kind": "same_document_wrong_section",
                    "negative": _chunk_ref(
                        chunks[same_document_id], include_text=include_text,
                    ),
                    "negative_fusion_rank": fusion_ids.index(same_document_id) + 1,
                    "negative_qrels_relevance": "none",
                })

    by_reason: dict[str, int] = {}
    for item in exclusions:
        by_reason[item["reason"]] = by_reason.get(item["reason"], 0) + 1
    query_ids = sorted({row["query_id"] for row in triples})
    return {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "contains_text": bool(include_text),
        "private_blind_set": False,
        "base_set": IMMUTABLE_SET_REF,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": {
            "review_outcomes": ["accepted", "partial"],
            "positive_relevance": "direct_only",
            "requires_direct_in_fusion_candidate20": True,
            "requires_non_direct_final_top1": True,
            "supporting_is_positive": False,
            "candidate_absent_destination": "stage_c",
            "stability_anchors": bool(include_stability_anchors),
            "scoring_text_formatter": "rag_gate._rerank_pair_documents(use_breadcrumb=True)",
        },
        "provenance": {
            "qrels_schema_version": qrels["schema_version"],
            "qrels_reviewer_id": qrels["reviewer_id"],
            "qrels_sha256": qrels_sha256,
            "baseline_schema_version": baseline.get("schema_version", ""),
            "baseline_sha256": baseline_sha256,
            "baseline_arm": baseline.get("arm"),
            "index": dict(qrels["index"]),
            "reconstructed_trace_query_ids": reconstructed_ids,
        },
        "summary": {
            "query_count": len(query_ids),
            "triple_count": len(triples),
            "false_winner_query_count": len({
                row["query_id"] for row in triples
                if row["negative_kind"] != "stability_anchor"
            }),
            "stability_anchor_query_count": len({
                row["query_id"] for row in triples
                if row["negative_kind"] == "stability_anchor"
            }),
            "current_false_winner_count": sum(
                row["negative_kind"] == "current_false_winner" for row in triples
            ),
            "same_document_wrong_section_count": sum(
                row["negative_kind"] == "same_document_wrong_section" for row in triples
            ),
            "stability_anchor_count": sum(
                row["negative_kind"] == "stability_anchor" for row in triples
            ),
            "exclusions_by_reason": by_reason,
        },
        "triples": triples,
        "exclusions": exclusions,
    }


def open_live_collection() -> Any:
    import chromadb
    from rag_tools import get_collection_name

    return chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())


def build_from_files(
    *,
    qrels_path: str | Path,
    baseline_path: str | Path,
    include_text: bool = False,
    reconstruct_missing: bool = False,
    include_stability_anchors: bool = True,
    collection: Any | None = None,
) -> dict[str, Any]:
    items = _load_immutable_items()
    expected_ids = [str(item["id"]) for item in items]
    qrels_file = Path(qrels_path)
    baseline_file = Path(baseline_path)
    qrels = load_qrels_overlay(qrels_file, expected_query_ids=expected_ids)
    baseline = _load_json(baseline_file)
    return build_hard_negative_dataset(
        qrels=qrels,
        items=items,
        baseline=baseline,
        collection=collection or open_live_collection(),
        include_text=include_text,
        reconstruct_missing=reconstruct_missing,
        include_stability_anchors=include_stability_anchors,
        qrels_sha256=_sha256_bytes(qrels_file.read_bytes()),
        baseline_sha256=_sha256_bytes(baseline_file.read_bytes()),
    )
