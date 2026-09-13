# -*- coding: utf-8 -*-
"""Isolated, token-aware chunking experiments for retrieval Stage D.

This module deliberately does **not** change :func:`rag_tools.split_markdown_document`
or the production collection.  It builds content-addressed experiment plans whose
collection names include the corpus/chunker fingerprint.  A plan may be inspected
and qrels-mapped without embedding or writing anything (the default CLI mode).

The body window is measured with the active embedding tokenizer.  Document title
and Markdown heading breadcrumb are stored separately and prepended only to the
embedding input, so ``320/64`` means 320 body tokens with a real 64-token body
overlap.  Child chunks retain a stable ``parent_id`` for post-hit parent expansion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Protocol, Sequence


EXPERIMENT_COLLECTION_PREFIX = "offerclaw_exp_chunk_"
EXPERIMENT_SCHEMA_VERSION = "chunk-experiment-v1"
# Keep model output and Chroma write payloads bounded even when the embedding
# provider itself accepts (or internally accumulates) a much larger request.
EXPERIMENT_WRITE_BATCH_SIZE = 32


class ChunkExperimentError(RuntimeError):
    """Raised when an experiment would be unsafe or cannot be audited."""


class QrelsMappingError(ChunkExperimentError):
    """Raised when a reviewer-approved span cannot map to an experiment chunk."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


class TokenOffsetCodec(Protocol):
    """Minimal tokenizer contract required for exact source-text windows."""

    @property
    def identity(self) -> str: ...

    def offsets_and_ids(self, text: str) -> tuple[list[tuple[int, int]], list[int]]: ...


class HuggingFaceTokenOffsetCodec:
    """Fast-tokenizer adapter preserving exact character offsets."""

    def __init__(self, tokenizer: Any, model_name: str):
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise ChunkExperimentError(
                "Stage-D requires a fast tokenizer with offset_mapping; "
                f"{model_name!r} resolved to a slow tokenizer"
            )
        self._tokenizer = tokenizer
        self._identity = f"hf-fast:{model_name}:{tokenizer.__class__.__name__}"

    @property
    def identity(self) -> str:
        return self._identity

    def offsets_and_ids(self, text: str) -> tuple[list[tuple[int, int]], list[int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        offsets: list[tuple[int, int]] = []
        token_ids: list[int] = []
        for raw_offset, raw_id in zip(encoded["offset_mapping"], encoded["input_ids"]):
            start, end = int(raw_offset[0]), int(raw_offset[1])
            # Some tokenizers emit synthetic (0, 0) offsets.  They cannot be
            # represented as an exact source substring and are excluded.
            if end <= start:
                continue
            offsets.append((start, end))
            token_ids.append(int(raw_id))
        return offsets, token_ids


class RegexTokenOffsetCodec:
    """Deterministic test/dry-run codec; not permitted for collection writes."""

    _PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)

    @property
    def identity(self) -> str:
        return "regex-offset-v1-approximate"

    def offsets_and_ids(self, text: str) -> tuple[list[tuple[int, int]], list[int]]:
        matches = list(self._PATTERN.finditer(text))
        offsets = [(match.start(), match.end()) for match in matches]
        # Stable IDs are enough to prove overlap in unit tests; they are not
        # embeddings and this codec is blocked from actual collection writes.
        ids = [
            int.from_bytes(hashlib.sha256(match.group(0).encode("utf-8")).digest()[:4], "big")
            for match in matches
        ]
        return offsets, ids


def load_active_token_codec(*, local_files_only: bool = True) -> HuggingFaceTokenOffsetCodec:
    """Load the configured embedding model's fast tokenizer.

    There is no silent approximate fallback: a real experiment collection must
    use the same token boundary family as its embedding model.
    """

    from transformers import AutoTokenizer
    from rag_tools import get_embedding_config

    model_name = str(get_embedding_config()["model"])
    resolved_model = model_name
    if not os.path.isdir(resolved_model):
        modelscope_cache = (
            Path(os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope/hub/models")).expanduser()
            / model_name
        )
        if modelscope_cache.is_dir():
            resolved_model = str(modelscope_cache.resolve())
        # Production embedding loading prefers ModelScope before HuggingFace.
        # Resolve the same cached snapshot first so a machine with no HF cache
        # does not fail despite being able to serve production embeddings.
        try:
            from modelscope import snapshot_download
            if resolved_model == model_name:
                resolved_model = snapshot_download(
                    model_name,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        "*.json", "*.txt", "*.model", "tokenizer*",
                        "sentencepiece*", "vocab*", "special_tokens*",
                    ],
                )
        except Exception:
            resolved_model = model_name
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        use_fast=True,
        local_files_only=local_files_only,
    )
    return HuggingFaceTokenOffsetCodec(tokenizer, model_name)


@dataclass(frozen=True)
class ChunkExperimentSpec:
    key: str
    body_max_tokens: int
    overlap_tokens: int
    include_breadcrumb: bool = True
    min_chars: int = 50
    version: str = "stage-d-v2"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,20}", self.key):
            raise ValueError(f"invalid experiment key: {self.key!r}")
        if self.body_max_tokens < 8:
            raise ValueError("body_max_tokens must be at least 8")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.body_max_tokens:
            raise ValueError("overlap_tokens must be in [0, body_max_tokens)")
        if self.min_chars < 1:
            raise ValueError("min_chars must be positive")

    @property
    def chunker_version(self) -> str:
        return (
            f"{self.version}:{self.key}:body{self.body_max_tokens}:"
            f"overlap{self.overlap_tokens}:breadcrumb{int(self.include_breadcrumb)}"
        )


C1_SPEC = ChunkExperimentSpec("c1", 320, 64)
C2_SPEC = ChunkExperimentSpec("c2", 480, 96)
EXPERIMENT_SPECS = {"c1": C1_SPEC, "c2": C2_SPEC}


@dataclass(frozen=True)
class TargetSourcePolicy:
    key: str = "generic-title-or-long-ratio-v1"
    long_chunk_tokens: int = 512
    long_chunk_ratio: float = 0.30
    generic_titles: tuple[str, ...] = ("", "正文", "正文内容", "页面结构目录")
    generic_chunk_min_tokens: int = 320
    nested_heading_min_count: int = 2


DEFAULT_TARGET_SOURCE_POLICY = TargetSourcePolicy()


@dataclass(frozen=True)
class TargetChunkPolicy:
    """Qrels-blind defects that justify replacing one immutable C0 chunk.

    This policy deliberately operates on the indexed C0 document/metadata,
    never on live Markdown files or evaluation labels.  That makes a
    ``snapshot_targeted`` plan a clean chunker experiment rather than a corpus
    refresh disguised as one.
    """

    key: str = "missing-structure-or-generic-internal-headings-v1"
    missing_structure_min_tokens: int = 512
    generic_titles: tuple[str, ...] = ("", "正文", "正文内容", "页面结构目录")
    generic_chunk_min_tokens: int = 320
    internal_heading_min_count: int = 2


DEFAULT_TARGET_CHUNK_POLICY = TargetChunkPolicy()
SNAPSHOT_TARGETED_LINEAGE_VERSION = "full-c0-expansion-parent-v2"


def select_snapshot_target_chunks(
    collection: Any,
    codec: TokenOffsetCodec,
    *,
    policy: TargetChunkPolicy = DEFAULT_TARGET_CHUNK_POLICY,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Select structurally defective C0 chunks without consulting qrels."""

    from rag_source_policy import evidence_allowed

    snapshot = collection.get(include=["documents", "metadatas"])
    ids = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    if not ids or not (len(ids) == len(documents) == len(metadatas)):
        raise ChunkExperimentError("production snapshot is empty or misaligned for chunk selection")
    selected: list[str] = []
    rows: list[dict[str, Any]] = []
    filtered_non_reference = 0
    for old_id, document, raw_metadata in sorted(
        zip(ids, documents, metadatas), key=lambda row: str(row[0])
    ):
        if not isinstance(document, str) or not isinstance(raw_metadata, dict):
            raise ChunkExperimentError(f"invalid production row during chunk selection: {old_id}")
        metadata = dict(raw_metadata)
        source = str(metadata.get("source") or "")
        eligible = bool(source) and evidence_allowed(
            "reference_kb",
            str(metadata.get("source_type") or "doc"),
            str(metadata.get("owner_scope") or ""),
            source,
        )
        if not eligible:
            filtered_non_reference += 1
            continue
        token_count = len(codec.offsets_and_ids(document)[1])
        raw_heading_path = metadata.get("heading_path")
        if isinstance(raw_heading_path, str):
            has_heading_path = bool(
                raw_heading_path.strip()
                and raw_heading_path.strip() not in {"[]", "()"}
            )
        else:
            has_heading_path = bool(raw_heading_path)
        has_structure = bool(metadata.get("breadcrumb")) or has_heading_path
        title = str(metadata.get("title") or "").strip()
        internal_heading_count = sum(
            1 for line in document.splitlines() if _HEADING_RE.match(line)
        )
        reasons: list[str] = []
        if not has_structure and token_count > policy.missing_structure_min_tokens:
            reasons.append("missing_breadcrumb_and_heading_over_limit")
        if (
            title in set(policy.generic_titles)
            and token_count > policy.generic_chunk_min_tokens
            and internal_heading_count >= policy.internal_heading_min_count
        ):
            reasons.append("generic_title_with_internal_markdown_headings")
        if reasons:
            selected.append(str(old_id))
            rows.append({
                "chunk_id": str(old_id),
                "source": source,
                "token_count": token_count,
                "title": title,
                "has_structure": has_structure,
                "internal_heading_count": internal_heading_count,
                "reasons": reasons,
            })
    if not selected:
        raise ChunkExperimentError("snapshot-targeted policy selected no C0 chunks")
    selector_payload = {
        "policy": asdict(policy),
        "mode": "qrels_blind_immutable_c0_chunk_policy",
        "qrels_consulted": False,
        "calibration_leakage": False,
        "promotion_eligible_selection": True,
        "evaluated_chunk_count": len(ids),
        "reference_kb_evaluated_chunk_count": len(ids) - filtered_non_reference,
        "non_reference_filtered_chunk_count": filtered_non_reference,
        "selected_chunk_count": len(selected),
        "selected_source_count": len({row["source"] for row in rows}),
        "selected_chunk_ids": selected,
        "selected_chunks": rows,
    }
    selector_payload["selector_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(selector_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return tuple(selected), selector_payload


def select_mixed_target_sources(
    collection: Any,
    codec: TokenOffsetCodec,
    *,
    policy: TargetSourcePolicy = DEFAULT_TARGET_SOURCE_POLICY,
    explicit_sources: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Select rebuild sources from corpus structure, never from qrels failures.

    ``explicit_sources`` exists only for developer diagnosis and is labelled as
    calibration leakage, which makes the resulting manifest ineligible for
    promotion.  The default policy is entirely qrels-blind.
    """

    from rag_source_policy import evidence_allowed

    snapshot = collection.get(include=["documents", "metadatas"])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    if len(documents) != len(metadatas):
        raise ChunkExperimentError("production snapshot is misaligned for target selection")
    source_rows: dict[str, dict[str, Any]] = {}
    for document, metadata in zip(documents, metadatas):
        if not isinstance(document, str) or not isinstance(metadata, dict):
            continue
        source = str(metadata.get("source") or "")
        if not source:
            continue
        token_count = len(codec.offsets_and_ids(document)[1])
        missing_structure = not bool(metadata.get("breadcrumb") or metadata.get("heading_path"))
        row = source_rows.setdefault(source, {
            "chunk_count": 0,
            "long_chunk_count": 0,
            "long_missing_structure_count": 0,
            "generic_title_nested_heading_count": 0,
            "max_chunk_tokens": 0,
            "reference_kb_eligible": True,
        })
        row["reference_kb_eligible"] = bool(row["reference_kb_eligible"]) and evidence_allowed(
            "reference_kb",
            str(metadata.get("source_type") or "doc"),
            str(metadata.get("owner_scope") or ""),
            source,
        )
        row["chunk_count"] += 1
        row["max_chunk_tokens"] = max(row["max_chunk_tokens"], token_count)
        if token_count > policy.long_chunk_tokens:
            row["long_chunk_count"] += 1
            if missing_structure:
                row["long_missing_structure_count"] += 1
        title = str(metadata.get("title") or "").strip()
        nested_headings = sum(
            1 for line in document.splitlines()
            if (match := _HEADING_RE.match(line)) and len(match.group(1)) >= 2
        )
        if (
            title in set(policy.generic_titles)
            and token_count > policy.generic_chunk_min_tokens
            and nested_headings >= policy.nested_heading_min_count
        ):
            row["generic_title_nested_heading_count"] += 1
    for row in source_rows.values():
        row["long_chunk_ratio"] = round(
            row["long_chunk_count"] / max(row["chunk_count"], 1), 6
        )
    if explicit_sources:
        requested = tuple(sorted(set(str(source) for source in explicit_sources)))
        missing = sorted(set(requested) - set(source_rows))
        if missing:
            raise ChunkExperimentError(f"explicit target sources absent from C0: {missing}")
        disallowed = sorted(
            source for source in requested
            if not source_rows[source]["reference_kb_eligible"]
        )
        if disallowed:
            raise ChunkExperimentError(
                f"explicit targets must remain in reference_kb: {disallowed}"
            )
        selected = requested
        mode = "explicit_development_diagnostic"
        calibration_leakage = True
        reasons = {source: "explicit_development_target" for source in selected}
    else:
        selected = tuple(sorted(
            source for source, row in source_rows.items()
            if row["reference_kb_eligible"]
            and (
                row["generic_title_nested_heading_count"] > 0
                or row["long_chunk_ratio"] >= policy.long_chunk_ratio
            )
        ))
        mode = "qrels_blind_structural_policy"
        calibration_leakage = False
        reasons = {
            source: (
                f"generic+nested={source_rows[source]['generic_title_nested_heading_count']}; "
                f">{policy.long_chunk_tokens}-token ratio="
                f"{source_rows[source]['long_chunk_ratio']:.3f}"
            )
            for source in selected
        }
    if not selected:
        raise ChunkExperimentError("mixed-targeted policy selected no sources")
    report = {
        "policy": asdict(policy),
        "mode": mode,
        "calibration_leakage": calibration_leakage,
        "promotion_eligible_selection": not calibration_leakage,
        "selected_source_count": len(selected),
        "selected_sources": list(selected),
        "selection_reasons": reasons,
        "source_structure": {source: source_rows[source] for source in selected},
        "evaluated_source_count": len(source_rows),
        "reference_kb_eligible_source_count": sum(
            bool(row["reference_kb_eligible"]) for row in source_rows.values()
        ),
        "non_reference_source_count": sum(
            not bool(row["reference_kb_eligible"]) for row in source_rows.values()
        ),
        "non_reference_sources": sorted(
            source for source, row in source_rows.items()
            if not row["reference_kb_eligible"]
        ),
        "qrels_consulted": False,
    }
    return selected, report


def active_embedding_contract() -> dict[str, Any]:
    """Return the complete, credential-free vector-space contract."""

    from rag_tools import embed_profile, get_embedding_config

    config = dict(get_embedding_config())
    return {
        "provider": str(config.get("provider") or ""),
        "model": str(config.get("model") or ""),
        "base_url": str(config.get("base_url") or ""),
        "dimensions": config.get("dimensions"),
        "batch_size": int(config.get("batch_size") or 0),
        "document_prefix": os.environ.get("OFFERCLAW_EMBED_PREFIX", ""),
        "max_sequence": os.environ.get("OFFERCLAW_EMBED_MAX_SEQ", ""),
        "embed_profile": embed_profile(),
    }


def _embedding_contract_json(contract: dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SourceDocument:
    path: str
    source: str
    source_type: str = "doc"
    owner_scope: str = "general"


@dataclass(frozen=True)
class ExperimentChunk:
    chunk_id: str
    parent_id: str
    source: str
    source_type: str
    owner_scope: str
    document_title: str
    heading_path: tuple[str, ...]
    breadcrumb: str
    text: str
    embedding_text: str
    body_token_count: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    token_ids: tuple[int, ...]
    chunker_version: str
    # Populated by snapshot_children.  This is the exact immutable C0 chunk
    # from which the child was produced, and is deliberately empty for a raw
    # source rebuild.  Qrels remapping uses it to avoid guessing between two
    # similar sections in the same source.
    origin_chunk_id: str = ""
    # Chroma accepts scalar metadata.  Snapshot children retain every scalar
    # filtering field from C0; experiment-owned keys below overwrite stale
    # parent/chunker identifiers deliberately.
    original_metadata: tuple[tuple[str, str | int | float | bool], ...] = ()
    parent_content_hash: str = ""
    production_collection: str = ""
    lineage_mode: str = "source_rebuilt"
    reuse_origin_embedding: bool = False
    # Token windows may be cut inside several Markdown sections while all
    # children still expand to one immutable C0 parent.  Keep overlap lineage
    # separate from expansion lineage so both contracts remain truthful.
    window_group_id: str = ""

    def chroma_metadata(self, experiment_fingerprint: str) -> dict[str, Any]:
        metadata = dict(self.original_metadata)
        metadata.update({
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_type": self.source_type,
            "owner_scope": self.owner_scope,
            "title": self.heading_path[-1] if self.heading_path else self.document_title,
            "document_title": self.document_title,
            "heading_path": json.dumps(self.heading_path, ensure_ascii=False),
            "breadcrumb": self.breadcrumb,
            "parent_id": self.parent_id,
            "char_len": len(self.text),
            "body_token_count": self.body_token_count,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunker_version": self.chunker_version,
            "origin_chunk_id": self.origin_chunk_id,
            "parent_content_hash": self.parent_content_hash,
            "production_collection": self.production_collection,
            "lineage_mode": self.lineage_mode,
            "reuse_origin_embedding": self.reuse_origin_embedding,
            "window_group_id": self.window_group_id or self.parent_id,
            "experiment_fingerprint": experiment_fingerprint,
        })
        return metadata


@dataclass(frozen=True)
class ExperimentPlan:
    schema_version: str
    corpus_mode: str
    spec: ChunkExperimentSpec
    tokenizer: str
    corpus_fingerprint: str
    experiment_fingerprint: str
    production_collection: str
    collection_name: str
    embedding_contract: dict[str, Any]
    sources: tuple[SourceDocument, ...]
    chunks: tuple[ExperimentChunk, ...]
    parents: dict[str, str]
    dedup_statistics: dict[str, Any]
    origin_documents: dict[str, str] = field(default_factory=dict)
    origin_metadatas: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_sources: tuple[str, ...] = ()
    target_selection: dict[str, Any] = field(default_factory=dict)

    def manifest(self, *, include_parent_text: bool = False) -> dict[str, Any]:
        parents: dict[str, Any]
        if include_parent_text:
            parents = dict(self.parents)
        else:
            parents = {
                parent_id: {
                    "sha256": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "char_len": len(text),
                }
                for parent_id, text in self.parents.items()
            }
        return {
            "schema_version": self.schema_version,
            "corpus_mode": self.corpus_mode,
            "spec": asdict(self.spec),
            "chunker_version": self.spec.chunker_version,
            "tokenizer": self.tokenizer,
            "corpus_fingerprint": self.corpus_fingerprint,
            "experiment_fingerprint": self.experiment_fingerprint,
            "production_collection": self.production_collection,
            "collection_name": self.collection_name,
            "embedding_contract": dict(self.embedding_contract),
            "source_count": len(self.sources),
            "chunk_count": len(self.chunks),
            "statistics": plan_statistics(self),
            "sources": [asdict(source) for source in self.sources],
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
            "parents": parents,
            "deduplication": self.dedup_statistics,
            "target_source_count": len(self.target_sources),
            "target_sources": list(self.target_sources),
            "target_chunk_count": len(
                self.target_selection.get("selected_chunk_ids") or []
            ),
            "target_selection": dict(self.target_selection),
            "origin_parent_count": len(self.origin_documents),
            "origin_parent_hashes": {
                origin_id: "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
                for origin_id, text in sorted(self.origin_documents.items())
            },
            "snapshot_min_chars_policy": (
                "inherit short C0 singleton parents; split parents use exact token windows "
                "whose final window overlaps by the configured amount"
                if self.corpus_mode in {"snapshot_children", "snapshot_targeted"}
                else (
                    "retain untargeted C0 chunks; targeted source rebuild filters children "
                    "shorter than spec.min_chars"
                    if self.corpus_mode == "mixed_targeted"
                    else "source_rebuild filters children shorter than spec.min_chars"
                )
            ),
        }


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[min(int(len(ordered) * fraction), len(ordered) - 1)])


def plan_statistics(plan: ExperimentPlan) -> dict[str, Any]:
    """Return auditable token-length and overlap statistics for a plan."""

    token_counts = [chunk.body_token_count for chunk in plan.chunks]
    rebuilt_token_counts = [
        chunk.body_token_count for chunk in plan.chunks
        if chunk.lineage_mode != "c0_retained"
    ]
    by_parent: dict[str, list[ExperimentChunk]] = {}
    for chunk in plan.chunks:
        by_parent.setdefault(chunk.window_group_id or chunk.parent_id, []).append(chunk)
    overlap_pairs = 0
    verified_pairs = 0
    dedup_created_gaps = 0
    for children in by_parent.values():
        ordered = sorted(children, key=lambda chunk: chunk.token_start)
        for previous, current in zip(ordered, ordered[1:]):
            # Content dedup can remove an intermediate child.  Such a gap is
            # not an overlap pair; overlap correctness was verified before
            # dedup in build_experiment_plan.
            if current.token_start != previous.token_end - plan.spec.overlap_tokens:
                dedup_created_gaps += 1
                continue
            overlap_pairs += 1
            overlap = plan.spec.overlap_tokens
            if (
                previous.token_ids[-overlap:] == current.token_ids[:overlap]
                and current.token_start == previous.token_end - overlap
            ):
                verified_pairs += 1
    by_source: dict[str, list[int]] = {}
    for chunk in plan.chunks:
        by_source.setdefault(chunk.source, []).append(chunk.body_token_count)
    return {
        "body_tokens": {
            "min": min(token_counts, default=0),
            "p50": _percentile(token_counts, 0.50),
            "p95": _percentile(token_counts, 0.95),
            "max": max(token_counts, default=0),
            "mean": round(sum(token_counts) / len(token_counts), 3) if token_counts else 0.0,
            "hard_limit": plan.spec.body_max_tokens,
            "hard_limit_violations": sum(
                count > plan.spec.body_max_tokens for count in token_counts
            ),
            "rebuilt_hard_limit_violations": sum(
                count > plan.spec.body_max_tokens for count in rebuilt_token_counts
            ),
        },
        "overlap": {
            "configured_tokens": plan.spec.overlap_tokens,
            "adjacent_pairs": overlap_pairs,
            "verified_pairs": verified_pairs,
            "violations": overlap_pairs - verified_pairs,
            "dedup_created_gaps": dedup_created_gaps,
        },
        "parent_count": len(by_parent),
        "source_count": len(by_source),
        "lineage_counts": {
            mode: sum(chunk.lineage_mode == mode for chunk in plan.chunks)
            for mode in sorted({chunk.lineage_mode for chunk in plan.chunks})
        },
        "chunks_by_source": {
            source: {
                "count": len(counts),
                "p95_body_tokens": _percentile(counts, 0.95),
                "max_body_tokens": max(counts),
            }
            for source, counts in sorted(by_source.items())
        },
    }


def _strip_frontmatter(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return text
    end = stripped.find("\n---", 3)
    if end == -1:
        return text
    header = stripped[:end]
    if "title:" not in header and "source_url" not in header:
        return text
    return stripped[end + 4 :].lstrip("\r\n")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _markdown_sections(text: str, fallback_title: str) -> tuple[str, list[tuple[tuple[str, ...], str]]]:
    """Return stable Markdown heading paths and exact body text per section."""

    text = _strip_frontmatter(text)
    first_h1 = next(
        (match.group(2).strip() for line in text.splitlines() if (match := _HEADING_RE.match(line)) and len(match.group(1)) == 1),
        "",
    )
    document_title = first_h1 or fallback_title
    # Store explicit Markdown levels.  Imported pages can jump H2 -> H4;
    # inserting a synthetic "untitled" H3 would break reviewer heading paths.
    heading_stack: list[tuple[int, str]] = [(1, document_title)]
    sections: list[tuple[tuple[str, ...], str]] = []
    current_path: tuple[str, ...] = (document_title,)
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((current_path, body))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if not match:
            body_lines.append(line)
            continue
        flush()
        body_lines = []
        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1:
            document_title = heading
            heading_stack = [(1, heading)]
        else:
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, heading))
        current_path = tuple(title for _level, title in heading_stack)
    flush()
    return document_title, sections


def _breadcrumb(document_title: str, heading_path: Sequence[str]) -> str:
    parts = [part.strip() for part in heading_path if part and part.strip()]
    if not parts or parts[0] != document_title:
        parts.insert(0, document_title)
    return " > ".join(parts)


def _safe_scalar_metadata(metadata: dict[str, Any]) -> tuple[tuple[str, str | int | float | bool], ...]:
    return tuple(sorted(
        (str(key), value)
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    ))


def _indexed_heading(metadata: dict[str, Any], source: str) -> tuple[str, tuple[str, ...]]:
    document_title = str(metadata.get("document_title") or Path(source).stem)
    raw_heading_path = metadata.get("heading_path")
    heading_path: tuple[str, ...] = ()
    if isinstance(raw_heading_path, str) and raw_heading_path.strip().startswith("["):
        try:
            parsed = json.loads(raw_heading_path)
            heading_path = tuple(str(part) for part in parsed if str(part).strip())
        except Exception:
            heading_path = ()
    indexed_title = str(metadata.get("title") or "").strip()
    if not heading_path:
        heading_parts = [document_title]
        if indexed_title and indexed_title != document_title:
            heading_parts.append(indexed_title)
        heading_path = tuple(heading_parts)
    return document_title, heading_path


def _stable_id(prefix: str, *parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def split_document_experiment(
    source: SourceDocument,
    text: str,
    spec: ChunkExperimentSpec,
    codec: TokenOffsetCodec,
    *,
    production_collection: str = "",
    preserve_all_sections: bool = False,
) -> tuple[list[ExperimentChunk], dict[str, str]]:
    """Split one Markdown document into hard-bounded overlapping child chunks."""

    document_title, sections = _markdown_sections(text, Path(source.source).stem)
    chunks: list[ExperimentChunk] = []
    parents: dict[str, str] = {}
    section_occurrences: dict[tuple[tuple[str, ...], str], int] = {}
    for heading_path, body in sections:
        # Match production ingest hygiene without calling or changing its
        # baseline splitter.  Raw image-only and imported TOC sections are not
        # answer-bearing retrieval children.
        if preserve_all_sections:
            body = body.strip()
        else:
            cleaned_lines = [
                line for line in body.splitlines()
                if not line.strip().startswith("![")
            ]
            body = "\n".join(cleaned_lines).strip()
            if heading_path and heading_path[-1] in {"页面结构目录", "图片素材"}:
                continue
        if len(body) < spec.min_chars:
            continue
        offsets, token_ids = codec.offsets_and_ids(body)
        if not offsets:
            continue
        crumb = _breadcrumb(document_title, heading_path)
        section_key = (
            tuple(heading_path),
            hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
        occurrence = section_occurrences.get(section_key, 0)
        section_occurrences[section_key] = occurrence + 1
        parent_id = _stable_id(
            "parent", source.source, json.dumps(heading_path, ensure_ascii=False),
            str(occurrence), body,
        )
        parents[parent_id] = body
        start = 0
        while start < len(offsets):
            end = min(start + spec.body_max_tokens, len(offsets))
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            body_slice = body[char_start:char_end].strip()
            ids = tuple(token_ids[start:end])
            if body_slice and len(body_slice) >= spec.min_chars:
                chunk_id = _stable_id(
                    f"{Path(source.source).stem}_{spec.key}",
                    source.source,
                    json.dumps(heading_path, ensure_ascii=False),
                    str(occurrence),
                    str(start),
                    body_slice,
                )
                embedding_text = f"{crumb}\n{body_slice}" if spec.include_breadcrumb else body_slice
                chunks.append(ExperimentChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    source=source.source,
                    source_type=source.source_type,
                    owner_scope=source.owner_scope,
                    document_title=document_title,
                    heading_path=tuple(heading_path),
                    breadcrumb=crumb,
                    text=body_slice,
                    embedding_text=embedding_text,
                    body_token_count=len(ids),
                    token_start=start,
                    token_end=end,
                    char_start=char_start,
                    char_end=char_end,
                    token_ids=ids,
                    chunker_version=spec.chunker_version,
                    parent_content_hash="sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    production_collection=production_collection,
                ))
            if end >= len(offsets):
                break
            # This is the actual overlap boundary, not merely metadata.  The
            # next child starts at the exact token that is overlap_tokens from
            # the previous child's end.
            start = end - spec.overlap_tokens
    return chunks, parents


def assert_true_overlap(chunks: Sequence[ExperimentChunk], expected_overlap: int) -> None:
    """Fail if adjacent children of a parent do not share exact token IDs."""

    by_parent: dict[str, list[ExperimentChunk]] = {}
    for chunk in chunks:
        by_parent.setdefault(chunk.window_group_id or chunk.parent_id, []).append(chunk)
    for parent_chunks in by_parent.values():
        ordered = sorted(parent_chunks, key=lambda chunk: chunk.token_start)
        for previous, current in zip(ordered, ordered[1:]):
            expected = min(expected_overlap, len(previous.token_ids), len(current.token_ids))
            if expected and previous.token_ids[-expected:] != current.token_ids[:expected]:
                raise ChunkExperimentError(
                    f"false overlap between {previous.chunk_id} and {current.chunk_id}"
                )
            if current.token_start != previous.token_end - expected_overlap:
                raise ChunkExperimentError(
                    f"non-contiguous overlap offsets between {previous.chunk_id} and {current.chunk_id}"
                )


def _source_content_fingerprint(sources: Sequence[SourceDocument]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        path = Path(source.path)
        digest.update(source.source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
        digest.update(source.source_type.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.owner_scope.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def experiment_collection_name(spec: ChunkExperimentSpec, experiment_fingerprint: str) -> str:
    digest = experiment_fingerprint.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("experiment_fingerprint must be sha256")
    return f"{EXPERIMENT_COLLECTION_PREFIX}{spec.key}_{digest[:12]}"


def assert_isolated_collection_name(
    name: str,
    *,
    production_collection: str,
    experiment_fingerprint: str,
) -> None:
    digest = experiment_fingerprint.removeprefix("sha256:")
    if name == production_collection:
        raise ChunkExperimentError("refusing to use the production collection")
    if not name.startswith(EXPERIMENT_COLLECTION_PREFIX):
        raise ChunkExperimentError("experiment collection must use the isolation prefix")
    if digest[:12] not in name:
        raise ChunkExperimentError("experiment collection name must contain its fingerprint")


def build_experiment_plan(
    sources: Sequence[SourceDocument],
    spec: ChunkExperimentSpec,
    codec: TokenOffsetCodec,
    *,
    production_collection: str,
    embedding_contract: dict[str, Any] | None = None,
) -> ExperimentPlan:
    if not sources:
        raise ChunkExperimentError("experiment corpus is empty")
    # Preserve caller order because production ingest dedup/order is auditable,
    # while rejecting duplicate source identities avoids ambiguous qrels.
    names = [source.source for source in sources]
    if len(set(names)) != len(names):
        raise ChunkExperimentError("duplicate source basenames are not allowed")
    corpus_fingerprint = _source_content_fingerprint(sources)
    embedding_contract = dict(embedding_contract or active_embedding_contract())
    signature = {
        "schema": EXPERIMENT_SCHEMA_VERSION,
        "spec": asdict(spec),
        "tokenizer": codec.identity,
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_contract": embedding_contract,
    }
    experiment_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    collection_name = experiment_collection_name(spec, experiment_fingerprint)
    assert_isolated_collection_name(
        collection_name,
        production_collection=production_collection,
        experiment_fingerprint=experiment_fingerprint,
    )
    chunks: list[ExperimentChunk] = []
    parents: dict[str, str] = {}
    seen_hashes: set[str] = set()
    pre_dedup = 0
    skipped_by_source: dict[str, int] = {}
    post_by_source: dict[str, int] = {}
    for source in sources:
        text = Path(source.path).read_text(encoding="utf-8", errors="replace")
        source_chunks, source_parents = split_document_experiment(
            source, text, spec, codec, production_collection=production_collection,
        )
        assert_true_overlap(source_chunks, spec.overlap_tokens)
        pre_dedup += len(source_chunks)
        for chunk in source_chunks:
            normalized_prefix = "".join(chunk.text.split())[:100]
            dedup_key = hashlib.md5(normalized_prefix.encode("utf-8")).hexdigest()
            if dedup_key in seen_hashes:
                skipped_by_source[source.source] = skipped_by_source.get(source.source, 0) + 1
                continue
            seen_hashes.add(dedup_key)
            chunks.append(chunk)
            post_by_source[source.source] = post_by_source.get(source.source, 0) + 1
        parents.update(source_parents)
    if not chunks:
        raise ChunkExperimentError("chunker produced no chunks")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ChunkExperimentError("stable chunk ID collision after occurrence disambiguation")
    if any(chunk.body_token_count > spec.body_max_tokens for chunk in chunks):
        raise ChunkExperimentError("body token hard limit was violated")
    return ExperimentPlan(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        corpus_mode="source_rebuild",
        spec=spec,
        tokenizer=codec.identity,
        corpus_fingerprint=corpus_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        production_collection=production_collection,
        collection_name=collection_name,
        embedding_contract=embedding_contract,
        sources=tuple(sources),
        chunks=tuple(chunks),
        parents=parents,
        dedup_statistics={
            "algorithm": "production-compatible-normalized-prefix100-md5",
            "source_order": "caller-production-ingest-order",
            "pre_dedup_chunks": pre_dedup,
            "post_dedup_chunks": len(chunks),
            "skipped_chunks": pre_dedup - len(chunks),
            "skipped_by_source": dict(sorted(skipped_by_source.items())),
            "post_chunks_by_source": dict(sorted(post_by_source.items())),
        },
    )


def build_snapshot_experiment_plan(
    collection: Any,
    spec: ChunkExperimentSpec,
    codec: TokenOffsetCodec,
    *,
    production_collection: str,
    embedding_contract: dict[str, Any] | None = None,
) -> ExperimentPlan:
    """Split the immutable live collection documents into child windows.

    This is the promotion-eligible C1/C2 corpus mode: parent membership and
    body content are byte-for-byte inherited from C0, so only child size,
    overlap, and breadcrumb change.  Raw source rebuild remains a separate,
    explicitly confounded experiment because local files can evolve after C0.
    """

    if production_collection.startswith(EXPERIMENT_COLLECTION_PREFIX):
        raise ChunkExperimentError(
            "refusing recursive snapshot from an experiment collection; "
            "provide the explicit production baseline collection"
        )
    snapshot = collection.get(include=["documents", "metadatas"])
    ids = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    if not ids or not (len(ids) == len(documents) == len(metadatas)):
        raise ChunkExperimentError("production snapshot is empty or misaligned")
    rows = sorted(zip(ids, documents, metadatas), key=lambda row: str(row[0]))
    digest = hashlib.sha256()
    chunks: list[ExperimentChunk] = []
    parents: dict[str, str] = {}
    source_meta: dict[str, tuple[str, str]] = {}
    for old_chunk_id, document, raw_metadata in rows:
        if not isinstance(document, str) or not isinstance(raw_metadata, dict):
            raise ChunkExperimentError(f"invalid production row: {old_chunk_id}")
        metadata = dict(raw_metadata)
        safe_original_metadata = _safe_scalar_metadata(metadata)
        source = str(metadata.get("source") or "")
        if not source:
            raise ChunkExperimentError(f"production row lacks source: {old_chunk_id}")
        source_type = str(metadata.get("source_type") or "doc")
        owner_scope = str(metadata.get("owner_scope") or "general")
        source_meta.setdefault(source, (source_type, owner_scope))
        digest.update(str(old_chunk_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")

        document_title, heading_path = _indexed_heading(metadata, source)
        crumb = _breadcrumb(document_title, heading_path)
        offsets, token_ids = codec.offsets_and_ids(document)
        if not offsets:
            continue
        parent_id = _stable_id(
            "snapshot_parent", production_collection, str(old_chunk_id), document,
        )
        parents[parent_id] = document
        start = 0
        while start < len(offsets):
            end = min(start + spec.body_max_tokens, len(offsets))
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            body_slice = document[char_start:char_end].strip()
            child_ids = tuple(token_ids[start:end])
            if body_slice:
                chunk_id = _stable_id(
                    f"snapshot_{spec.key}", str(old_chunk_id), str(start), body_slice,
                )
                chunks.append(ExperimentChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    source=source,
                    source_type=source_type,
                    owner_scope=owner_scope,
                    document_title=document_title,
                    heading_path=heading_path or (document_title,),
                    breadcrumb=crumb,
                    text=body_slice,
                    embedding_text=f"{crumb}\n{body_slice}" if spec.include_breadcrumb else body_slice,
                    body_token_count=len(child_ids),
                    token_start=start,
                    token_end=end,
                    char_start=char_start,
                    char_end=char_end,
                    token_ids=child_ids,
                    chunker_version=spec.chunker_version,
                    origin_chunk_id=str(old_chunk_id),
                    original_metadata=safe_original_metadata,
                    parent_content_hash="sha256:" + hashlib.sha256(document.encode("utf-8")).hexdigest(),
                    production_collection=production_collection,
                    lineage_mode="snapshot_child",
                ))
            if end >= len(offsets):
                break
            start = end - spec.overlap_tokens
    if not chunks:
        raise ChunkExperimentError("snapshot chunker produced no children")
    assert_true_overlap(chunks, spec.overlap_tokens)
    corpus_fingerprint = "sha256:" + digest.hexdigest()
    embedding_contract = dict(embedding_contract or active_embedding_contract())
    signature = {
        "schema": EXPERIMENT_SCHEMA_VERSION,
        "corpus_mode": "snapshot_children",
        "spec": asdict(spec),
        "tokenizer": codec.identity,
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_contract": embedding_contract,
    }
    experiment_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    collection_name = experiment_collection_name(spec, experiment_fingerprint)
    assert_isolated_collection_name(
        collection_name,
        production_collection=production_collection,
        experiment_fingerprint=experiment_fingerprint,
    )
    sources = tuple(
        SourceDocument("", source, values[0], values[1])
        for source, values in sorted(source_meta.items())
    )
    return ExperimentPlan(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        corpus_mode="snapshot_children",
        spec=spec,
        tokenizer=codec.identity,
        corpus_fingerprint=corpus_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        production_collection=production_collection,
        collection_name=collection_name,
        embedding_contract=embedding_contract,
        sources=sources,
        chunks=tuple(chunks),
        parents=parents,
        dedup_statistics={
            "algorithm": "not-reapplied-live-snapshot-already-deduplicated",
            "source_order": "production-chunk-id-sorted",
            "pre_dedup_chunks": len(chunks),
            "post_dedup_chunks": len(chunks),
            "skipped_chunks": 0,
            "skipped_by_source": {},
        },
        origin_documents={
            str(old_chunk_id): str(document)
            for old_chunk_id, document, _metadata in rows
        },
        origin_metadatas={
            str(old_chunk_id): dict(metadata)
            for old_chunk_id, _document, metadata in rows
        },
    )


def _split_snapshot_target_chunk(
    old_chunk_id: str,
    document: str,
    metadata: dict[str, Any],
    spec: ChunkExperimentSpec,
    codec: TokenOffsetCodec,
    *,
    production_collection: str,
) -> tuple[list[ExperimentChunk], dict[str, str]]:
    """Heading-aware split of one immutable C0 chunk with origin lineage."""

    source = str(metadata.get("source") or "")
    if not source:
        raise ChunkExperimentError(f"target C0 chunk lacks source: {old_chunk_id}")
    source_type = str(metadata.get("source_type") or "doc")
    owner_scope = str(metadata.get("owner_scope") or "general")
    # Targeting is already restricted to structurally defective, long C0
    # chunks.  Preserve every non-empty subsection rather than applying the
    # live-source hygiene/min-length policy, because C0 membership is fixed.
    split_spec = replace(spec, min_chars=1)
    raw_children, raw_parents = split_document_experiment(
        SourceDocument("", source, source_type, owner_scope),
        document,
        split_spec,
        codec,
        production_collection=production_collection,
        preserve_all_sections=True,
    )
    if not raw_children:
        raise ChunkExperimentError(f"target C0 chunk produced no children: {old_chunk_id}")
    # Verify token-window overlap while children still carry their section
    # parent.  Expansion lineage is changed below to the complete C0 document.
    assert_true_overlap(raw_children, spec.overlap_tokens)
    safe_metadata = _safe_scalar_metadata(metadata)
    expansion_parent_id = _stable_id(
        "snapshot_target_parent", production_collection, old_chunk_id, document,
    )
    expansion_parent_hash = "sha256:" + hashlib.sha256(
        document.encode("utf-8")
    ).hexdigest()
    parents = {expansion_parent_id: document}
    children: list[ExperimentChunk] = []
    for raw_child in raw_children:
        children.append(replace(
            raw_child,
            chunk_id=_stable_id(
                f"snapshot_target_{spec.key}", old_chunk_id, raw_child.chunk_id,
            ),
            parent_id=expansion_parent_id,
            window_group_id=_stable_id(
                "snapshot_target_window_group",
                production_collection,
                old_chunk_id,
                raw_child.parent_id,
            ),
            chunker_version=spec.chunker_version,
            origin_chunk_id=old_chunk_id,
            original_metadata=safe_metadata,
            parent_content_hash=expansion_parent_hash,
            production_collection=production_collection,
            lineage_mode="snapshot_target_child",
            reuse_origin_embedding=False,
        ))
    return children, parents


def build_snapshot_targeted_experiment_plan(
    collection: Any,
    selected_chunk_ids: Sequence[str],
    target_selection: dict[str, Any],
    spec: ChunkExperimentSpec,
    codec: TokenOffsetCodec,
    *,
    production_collection: str,
    embedding_contract: dict[str, Any] | None = None,
) -> ExperimentPlan:
    """Rechunk only qrels-blind defective C0 chunks; retain the rest exactly."""

    if production_collection.startswith(EXPERIMENT_COLLECTION_PREFIX):
        raise ChunkExperimentError(
            "snapshot-targeted baseline cannot be an experiment collection"
        )
    selected = tuple(sorted(set(str(value) for value in selected_chunk_ids)))
    reported = tuple(sorted(set(
        str(value) for value in target_selection.get("selected_chunk_ids") or []
    )))
    if not selected or selected != reported:
        raise ChunkExperimentError(
            "selected C0 chunks must exactly match the audited selector report"
        )
    if target_selection.get("qrels_consulted") is not False:
        raise ChunkExperimentError("snapshot-targeted selector must be qrels-blind")
    if target_selection.get("calibration_leakage"):
        raise ChunkExperimentError("snapshot-targeted selector cannot contain calibration leakage")

    snapshot = collection.get(include=["documents", "metadatas"])
    ids = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    if not ids or not (len(ids) == len(documents) == len(metadatas)):
        raise ChunkExperimentError("production snapshot is empty or misaligned")
    rows = sorted(zip(ids, documents, metadatas), key=lambda row: str(row[0]))
    live_ids = {str(row[0]) for row in rows}
    missing = sorted(set(selected) - live_ids)
    if missing:
        raise ChunkExperimentError(f"selector names absent C0 chunks: {missing}")

    digest = hashlib.sha256()
    origin_documents: dict[str, str] = {}
    origin_metadatas: dict[str, dict[str, Any]] = {}
    source_meta: dict[str, tuple[str, str]] = {}
    for old_id, document, raw_metadata in rows:
        if not isinstance(document, str) or not isinstance(raw_metadata, dict):
            raise ChunkExperimentError(f"invalid production row: {old_id}")
        metadata = dict(raw_metadata)
        source = str(metadata.get("source") or "")
        if not source:
            raise ChunkExperimentError(f"production row lacks source: {old_id}")
        source_meta.setdefault(source, (
            str(metadata.get("source_type") or "doc"),
            str(metadata.get("owner_scope") or "general"),
        ))
        origin_documents[str(old_id)] = document
        origin_metadatas[str(old_id)] = metadata
        digest.update(str(old_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")

    corpus_fingerprint = "sha256:" + digest.hexdigest()
    embedding_contract = dict(embedding_contract or active_embedding_contract())
    selector_hash = str(target_selection.get("selector_hash") or "")
    expected_selector_hash_payload = dict(target_selection)
    expected_selector_hash_payload.pop("selector_hash", None)
    expected_selector_hash = "sha256:" + hashlib.sha256(
        json.dumps(expected_selector_hash_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if selector_hash != expected_selector_hash:
        raise ChunkExperimentError("snapshot-targeted selector hash mismatch")
    signature = {
        "schema": EXPERIMENT_SCHEMA_VERSION,
        "corpus_mode": "snapshot_targeted",
        "spec": asdict(spec),
        "tokenizer": codec.identity,
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_contract": embedding_contract,
        "selector_hash": selector_hash,
        "lineage_contract": SNAPSHOT_TARGETED_LINEAGE_VERSION,
    }
    experiment_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    collection_name = experiment_collection_name(spec, experiment_fingerprint)
    assert_isolated_collection_name(
        collection_name,
        production_collection=production_collection,
        experiment_fingerprint=experiment_fingerprint,
    )

    chunks: list[ExperimentChunk] = []
    parents: dict[str, str] = {}
    retained = 0
    rebuilt = 0
    selected_sources: set[str] = set()
    for old_id_raw, document, raw_metadata in rows:
        old_id = str(old_id_raw)
        metadata = dict(raw_metadata)
        source = str(metadata["source"])
        if old_id in selected:
            target_children, target_parents = _split_snapshot_target_chunk(
                old_id,
                document,
                metadata,
                spec,
                codec,
                production_collection=production_collection,
            )
            chunks.extend(target_children)
            parents.update(target_parents)
            rebuilt += len(target_children)
            selected_sources.add(source)
            continue
        offsets, token_ids = codec.offsets_and_ids(document)
        if not offsets:
            raise ChunkExperimentError(f"retained C0 chunk has no token offsets: {old_id}")
        document_title, heading_path = _indexed_heading(metadata, source)
        parent_id = _stable_id("snapshot_target_c0_parent", production_collection, old_id, document)
        parents[parent_id] = document
        chunks.append(ExperimentChunk(
            chunk_id=old_id,
            parent_id=parent_id,
            source=source,
            source_type=str(metadata.get("source_type") or "doc"),
            owner_scope=str(metadata.get("owner_scope") or "general"),
            document_title=document_title,
            heading_path=heading_path,
            breadcrumb=_breadcrumb(document_title, heading_path),
            text=document,
            embedding_text=document,
            body_token_count=len(token_ids),
            token_start=0,
            token_end=len(token_ids),
            char_start=offsets[0][0],
            char_end=offsets[-1][1],
            token_ids=tuple(token_ids),
            chunker_version=str(metadata.get("chunker_version") or "c0-unknown"),
            origin_chunk_id=old_id,
            original_metadata=_safe_scalar_metadata(metadata),
            parent_content_hash="sha256:" + hashlib.sha256(document.encode("utf-8")).hexdigest(),
            production_collection=production_collection,
            lineage_mode="c0_retained",
            reuse_origin_embedding=True,
        ))
        retained += 1
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ChunkExperimentError("snapshot-targeted stable chunk ID collision")
    if set(selected) & {chunk.chunk_id for chunk in chunks}:
        raise ChunkExperimentError("snapshot-targeted old C0 target chunk leaked into plan")
    if any(
        chunk.body_token_count > spec.body_max_tokens
        for chunk in chunks if chunk.lineage_mode == "snapshot_target_child"
    ):
        raise ChunkExperimentError("snapshot-targeted child violates token hard limit")

    return ExperimentPlan(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        corpus_mode="snapshot_targeted",
        spec=spec,
        tokenizer=codec.identity,
        corpus_fingerprint=corpus_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        production_collection=production_collection,
        collection_name=collection_name,
        embedding_contract=embedding_contract,
        sources=tuple(
            SourceDocument("", source, values[0], values[1])
            for source, values in sorted(source_meta.items())
        ),
        chunks=tuple(chunks),
        parents=parents,
        dedup_statistics={
            "algorithm": "immutable-C0-chunk-targeted-replacement",
            "lineage_contract": SNAPSHOT_TARGETED_LINEAGE_VERSION,
            "retained_c0_chunks": retained,
            "removed_target_c0_chunks": len(selected),
            "rebuilt_target_chunks": rebuilt,
            "old_target_chunks_present": 0,
        },
        origin_documents=origin_documents,
        origin_metadatas=origin_metadatas,
        target_sources=tuple(sorted(selected_sources)),
        target_selection=dict(target_selection),
    )


def build_mixed_targeted_experiment_plan(
    collection: Any,
    targeted_sources: Sequence[SourceDocument],
    target_selection: dict[str, Any],
    spec: ChunkExperimentSpec,
    codec: TokenOffsetCodec,
    *,
    production_collection: str,
    embedding_contract: dict[str, Any] | None = None,
) -> ExperimentPlan:
    """Replace only structurally selected sources; retain every other C0 chunk.

    Targeted sources are rebuilt from their resolved Markdown files using the
    heading-aware splitter.  Their old C0 chunks are excluded completely.
    Untargeted rows retain byte-identical documents, stable IDs and an explicit
    ``reuse_origin_embedding`` lineage marker.
    """

    if production_collection.startswith(EXPERIMENT_COLLECTION_PREFIX):
        raise ChunkExperimentError("mixed-targeted baseline cannot be an experiment collection")
    selected = tuple(sorted(set(str(source) for source in target_selection.get("selected_sources") or [])))
    provided = {source.source: source for source in targeted_sources}
    if not selected or set(selected) != set(provided):
        raise ChunkExperimentError(
            "resolved targeted sources must exactly match the audited target-selection manifest"
        )
    snapshot = collection.get(include=["documents", "metadatas"])
    ids = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    if not ids or not (len(ids) == len(documents) == len(metadatas)):
        raise ChunkExperimentError("production snapshot is empty or misaligned")
    rows = sorted(zip(ids, documents, metadatas), key=lambda row: str(row[0]))
    live_sources = {
        str(metadata.get("source") or "") for _old_id, _doc, metadata in rows
        if isinstance(metadata, dict)
    }
    if not set(selected) <= live_sources:
        raise ChunkExperimentError("target selection includes a source absent from C0")

    embedding_contract = dict(embedding_contract or active_embedding_contract())
    production_digest = hashlib.sha256()
    origin_documents: dict[str, str] = {}
    origin_metadatas: dict[str, dict[str, Any]] = {}
    metadata_by_source: dict[str, list[dict[str, Any]]] = {}
    for old_chunk_id, document, raw_metadata in rows:
        if not isinstance(document, str) or not isinstance(raw_metadata, dict):
            raise ChunkExperimentError(f"invalid production row: {old_chunk_id}")
        metadata = dict(raw_metadata)
        source = str(metadata.get("source") or "")
        if not source:
            raise ChunkExperimentError(f"production row lacks source: {old_chunk_id}")
        origin_documents[str(old_chunk_id)] = document
        origin_metadatas[str(old_chunk_id)] = metadata
        metadata_by_source.setdefault(source, []).append(metadata)
        production_digest.update(str(old_chunk_id).encode("utf-8"))
        production_digest.update(b"\0")
        production_digest.update(document.encode("utf-8"))
        production_digest.update(b"\0")
        production_digest.update(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        production_digest.update(b"\0")

    source_digest = hashlib.sha256()
    for source_name in selected:
        source = provided[source_name]
        source_digest.update(source_name.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(hashlib.sha256(Path(source.path).read_bytes()).digest())
        source_digest.update(b"\0")
    corpus_signature = {
        "production_snapshot": "sha256:" + production_digest.hexdigest(),
        "target_source_files": "sha256:" + source_digest.hexdigest(),
        "target_selection": target_selection,
    }
    corpus_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(corpus_signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    signature = {
        "schema": EXPERIMENT_SCHEMA_VERSION,
        "corpus_mode": "mixed_targeted",
        "spec": asdict(spec),
        "tokenizer": codec.identity,
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_contract": embedding_contract,
        "target_policy_hash": "sha256:" + hashlib.sha256(
            json.dumps(target_selection, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    experiment_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    collection_name = experiment_collection_name(spec, experiment_fingerprint)
    assert_isolated_collection_name(
        collection_name,
        production_collection=production_collection,
        experiment_fingerprint=experiment_fingerprint,
    )

    chunks: list[ExperimentChunk] = []
    parents: dict[str, str] = {}
    retained_count_by_source: dict[str, int] = {}
    removed_count_by_source: dict[str, int] = {}
    for old_chunk_id, document, raw_metadata in rows:
        metadata = dict(raw_metadata)
        source = str(metadata["source"])
        if source in selected:
            removed_count_by_source[source] = removed_count_by_source.get(source, 0) + 1
            continue
        offsets, token_ids = codec.offsets_and_ids(document)
        if not offsets:
            raise ChunkExperimentError(f"retained C0 chunk has no token offsets: {old_chunk_id}")
        document_title, heading_path = _indexed_heading(metadata, source)
        parent_id = _stable_id(
            "mixed_c0_parent", production_collection, str(old_chunk_id), document,
        )
        parents[parent_id] = document
        chunks.append(ExperimentChunk(
            chunk_id=str(old_chunk_id),
            parent_id=parent_id,
            source=source,
            source_type=str(metadata.get("source_type") or "doc"),
            owner_scope=str(metadata.get("owner_scope") or "general"),
            document_title=document_title,
            heading_path=heading_path,
            breadcrumb=_breadcrumb(document_title, heading_path),
            text=document,
            # Retained vectors are copied from C0 during a real build.  This
            # text is kept only as a fail-visible fallback/lineage value.
            embedding_text=document,
            body_token_count=len(token_ids),
            token_start=0,
            token_end=len(token_ids),
            char_start=offsets[0][0],
            char_end=offsets[-1][1],
            token_ids=tuple(token_ids),
            chunker_version=str(metadata.get("chunker_version") or "c0-unknown"),
            origin_chunk_id=str(old_chunk_id),
            original_metadata=_safe_scalar_metadata(metadata),
            parent_content_hash="sha256:" + hashlib.sha256(document.encode("utf-8")).hexdigest(),
            production_collection=production_collection,
            lineage_mode="c0_retained",
            reuse_origin_embedding=True,
        ))
        retained_count_by_source[source] = retained_count_by_source.get(source, 0) + 1

    rebuilt_count_by_source: dict[str, int] = {}
    for source_name in selected:
        source = provided[source_name]
        source_text = Path(source.path).read_text(encoding="utf-8", errors="replace")
        rebuilt, rebuilt_parents = split_document_experiment(
            source,
            source_text,
            spec,
            codec,
            production_collection=production_collection,
        )
        assert_true_overlap(rebuilt, spec.overlap_tokens)
        old_metadata_rows = metadata_by_source[source_name]
        common_metadata = dict(_safe_scalar_metadata(old_metadata_rows[0]))
        for metadata in old_metadata_rows[1:]:
            safe = dict(_safe_scalar_metadata(metadata))
            common_metadata = {
                key: value for key, value in common_metadata.items()
                if safe.get(key) == value
            }
        rebuilt = [
            replace(
                chunk,
                original_metadata=tuple(sorted(common_metadata.items())),
                lineage_mode="source_rebuilt",
                reuse_origin_embedding=False,
            )
            for chunk in rebuilt
        ]
        if not rebuilt:
            raise ChunkExperimentError(f"targeted source produced no chunks: {source_name}")
        chunks.extend(rebuilt)
        parents.update(rebuilt_parents)
        rebuilt_count_by_source[source_name] = len(rebuilt)

    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ChunkExperimentError("mixed-targeted stable chunk ID collision")
    if any(
        chunk.body_token_count > spec.body_max_tokens
        for chunk in chunks if chunk.lineage_mode == "source_rebuilt"
    ):
        raise ChunkExperimentError("mixed-targeted rebuilt child violates token hard limit")
    if any(
        chunk.source in selected and chunk.lineage_mode == "c0_retained"
        for chunk in chunks
    ):
        raise ChunkExperimentError("targeted source leaked an old C0 chunk into mixed collection")

    return ExperimentPlan(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        corpus_mode="mixed_targeted",
        spec=spec,
        tokenizer=codec.identity,
        corpus_fingerprint=corpus_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        production_collection=production_collection,
        collection_name=collection_name,
        embedding_contract=embedding_contract,
        sources=tuple(provided[source] for source in selected),
        chunks=tuple(chunks),
        parents=parents,
        dedup_statistics={
            "algorithm": "retain-C0-membership; no cross-source re-dedup of targeted replacements",
            "retained_c0_chunks": sum(retained_count_by_source.values()),
            "removed_target_c0_chunks": sum(removed_count_by_source.values()),
            "rebuilt_target_chunks": sum(rebuilt_count_by_source.values()),
            "retained_by_source": dict(sorted(retained_count_by_source.items())),
            "removed_by_source": dict(sorted(removed_count_by_source.items())),
            "rebuilt_by_source": dict(sorted(rebuilt_count_by_source.items())),
            "old_target_chunks_present": 0,
        },
        origin_documents=origin_documents,
        origin_metadatas=origin_metadatas,
        target_sources=selected,
        target_selection=dict(target_selection),
    )


def expand_parent_context(chunk_ids: Iterable[str], plan: ExperimentPlan) -> dict[str, str]:
    """Resolve hit child IDs to their immutable section-level parent context."""

    lookup = {chunk.chunk_id: chunk.parent_id for chunk in plan.chunks}
    missing = sorted(set(chunk_ids) - set(lookup))
    if missing:
        raise ChunkExperimentError(f"unknown child chunk IDs: {missing}")
    return {chunk_id: plan.parents[lookup[chunk_id]] for chunk_id in chunk_ids}


def _heading_compatible(target: Sequence[str], actual: Sequence[str]) -> bool:
    target_tuple = tuple(part.strip() for part in target)
    actual_tuple = tuple(part.strip() for part in actual)
    return (
        target_tuple == actual_tuple
        or actual_tuple[: len(target_tuple)] == target_tuple
        or target_tuple[: len(actual_tuple)] == actual_tuple
    )


def _canonical_text_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize like qrels while retaining canonical-char -> raw-char offsets."""

    output: list[str] = []
    raw_offsets: list[int] = []
    pending_space: int | None = None
    for raw_index, raw_char in enumerate(text):
        for char in unicodedata.normalize("NFKC", raw_char):
            if char.isspace():
                if output and output[-1] != " ":
                    pending_space = raw_index
                continue
            if pending_space is not None:
                output.append(" ")
                raw_offsets.append(pending_space)
                pending_space = None
            output.append(char)
            raw_offsets.append(raw_index)
    return "".join(output), raw_offsets


def _parent_span_candidates(
    parent_text: str,
    reviewed_excerpt: str,
    children: Sequence[ExperimentChunk],
) -> tuple[bool, list[tuple[float, ExperimentChunk, str]]]:
    """Return child coverage for a reviewed span inside one immutable parent.

    Coverage is measured in the same normalized character space used by
    ``answer_span_hash``.  Measuring raw characters would make blank lines and
    punctuation-width normalization change the acceptance threshold.  The
    returned excerpt remains an actual substring of the immutable parent and
    must normalize to text contained by the selected child.
    """

    canonical_parent, raw_offsets = _canonical_text_with_offsets(parent_text)
    canonical_excerpt, _ = _canonical_text_with_offsets(reviewed_excerpt)
    if not canonical_excerpt:
        return False, []
    occurrence_starts: list[int] = []
    cursor = 0
    while True:
        found = canonical_parent.find(canonical_excerpt, cursor)
        if found < 0:
            break
        occurrence_starts.append(found)
        cursor = found + 1
    candidates: list[tuple[float, ExperimentChunk, str]] = []
    for occurrence_start in occurrence_starts:
        occurrence_end = occurrence_start + len(canonical_excerpt)
        occurrence_offsets = raw_offsets[occurrence_start:occurrence_end]
        for child in children:
            covered = [
                raw_index for raw_index in occurrence_offsets
                if child.char_start <= raw_index < child.char_end
            ]
            if not covered:
                continue
            raw_start = min(covered)
            raw_end = max(covered) + 1
            derived_excerpt = parent_text[raw_start:raw_end]
            canonical_derived, _ = _canonical_text_with_offsets(derived_excerpt)
            if not canonical_derived:
                continue
            canonical_child, _ = _canonical_text_with_offsets(child.text)
            if canonical_derived not in canonical_child:
                continue
            coverage = len(covered) / len(canonical_excerpt)
            candidates.append((coverage, child, derived_excerpt))
    candidates.sort(key=lambda row: (-row[0], row[1].chunk_id, row[2]))
    return bool(occurrence_starts), candidates


def map_qrels_to_experiment(
    payload: dict[str, Any],
    plan: ExperimentPlan,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map qrels evidence spans/headings to new stable child IDs.

    Retrieval output is never consulted.  A target maps only when its reviewed
    excerpt is contained in a child of the exact reviewed production parent.
    All overlap children containing the complete direct span are strict child
    gold.  A span split across children retains its production parent identity
    and is explicitly marked ``parent_expand_required``; it is never promoted
    to strict child gold merely because one child contains most of the text.
    """

    from rag_qrels import answer_span_hash, normalize_answer_span, validate_qrels_overlay

    validate_qrels_overlay(payload)
    by_source: dict[str, list[ExperimentChunk]] = {}
    for chunk in plan.chunks:
        by_source.setdefault(chunk.source, []).append(chunk)
    mapped = json.loads(json.dumps(payload, ensure_ascii=False))
    mapped["reviewer_id"] = f"{payload['reviewer_id']}:mapped-{plan.spec.key}"
    mapped["review_method"] = (
        (payload.get("review_method") or "")
        + f" Stage-D origin-locked child/parent-scope remap to {plan.collection_name}."
    ).strip()
    mapped["index"] = {
        "collection": plan.collection_name,
        "count": len(plan.chunks),
        "fingerprint": plan.experiment_fingerprint,
    }
    failures: list[dict[str, Any]] = []
    target_reports: list[dict[str, Any]] = []
    mapped_target_count = 0
    for item in mapped["items"]:
        original_targets = item["relevant_targets"]
        new_targets: list[dict[str, Any]] = []
        for target in original_targets:
            digest_ok = target["answer_span_hash"] == answer_span_hash(target["evidence_excerpt"])
            excerpt = normalize_answer_span(target["evidence_excerpt"])
            all_source_chunks = by_source.get(target["source"], [])
            mixed_source_rebuilt = (
                plan.corpus_mode == "mixed_targeted"
                and target["source"] in set(plan.target_sources)
            )
            exact_origin_child_lock = (
                plan.corpus_mode in {"snapshot_children", "snapshot_targeted"}
                or (plan.corpus_mode == "mixed_targeted" and not mixed_source_rebuilt)
            )
            origin_parent_locked = plan.corpus_mode in {
                "snapshot_children", "snapshot_targeted", "mixed_targeted",
            }
            reviewed_origin_document = plan.origin_documents.get(target["chunk_id"])
            reviewed_origin_metadata = plan.origin_metadatas.get(target["chunk_id"], {})
            origin_parent_verified = bool(
                reviewed_origin_document is not None
                and str(reviewed_origin_metadata.get("source") or target["source"])
                == target["source"]
                and excerpt in normalize_answer_span(reviewed_origin_document)
            )
            if exact_origin_child_lock:
                # Qrels already names the exact C0 chunk reviewed by a human.
                # Child remapping must remain inside that immutable parent; a
                # same-source semantic match elsewhere would be label leakage.
                source_chunks = [
                    chunk for chunk in all_source_chunks
                    if chunk.origin_chunk_id == target["chunk_id"]
                ]
            else:
                source_chunks = all_source_chunks
            locked_parent_ids = sorted({chunk.parent_id for chunk in source_chunks})
            text_matches = [
                chunk for chunk in source_chunks
                if excerpt in normalize_answer_span(chunk.text)
            ]
            heading_matches = [
                chunk for chunk in text_matches
                if _heading_compatible(target["heading_path"], chunk.heading_path)
            ]
            # Reviewer heading paths describe the semantic Feishu tree and can
            # drift from malformed Markdown levels.  A source-local unique span
            # is stronger identity evidence than that presentation hierarchy.
            selected_matches = heading_matches
            mapping_mode = "source_span_heading"
            heading_drift = False
            derived_excerpts: dict[str, str] = {}
            if exact_origin_child_lock and not selected_matches and text_matches:
                # The origin chunk identity disambiguates all overlapping
                # children.  A stale semantic heading must not veto exact,
                # reviewer-approved span evidence inside that parent.
                selected_matches = text_matches
                mapping_mode = "origin_parent_span_heading_drift"
                heading_drift = True
            elif mixed_source_rebuilt and selected_matches:
                mapping_mode = "mixed_source_exact_span_heading"
            elif (
                mixed_source_rebuilt
                and not selected_matches
                and text_matches
                and len({chunk.parent_id for chunk in text_matches}) == 1
            ):
                selected_matches = text_matches
                mapping_mode = "mixed_source_exact_span_heading_drift"
                heading_drift = True
            elif not selected_matches and len(text_matches) == 1:
                selected_matches = text_matches
                mapping_mode = "source_unique_span_heading_drift"
                heading_drift = True
            elif (
                not selected_matches
                and text_matches
                and len({chunk.parent_id for chunk in text_matches}) == 1
            ):
                # True overlap commonly places one reviewed span in two child
                # windows.  That is one parent occurrence, not an ambiguous
                # source match; both equivalent children are valid gold.
                selected_matches = text_matches
                mapping_mode = "source_unique_parent_span_heading_drift"
                heading_drift = True
            # Fixed windows can split a reviewed span even when its section
            # parent contains the full evidence.  In that case select exactly
            # one maximally-overlapping child, require >=80% reviewed-span
            # coverage, and record the derivation.  The mapped qrels excerpt is
            # the contained reviewed substring, so normal live-index validation
            # still works; the report retains the original reviewed hash.
            parent_span_coverage = None
            parent_span_verified = False
            if (
                not mixed_source_rebuilt
                and plan.corpus_mode != "snapshot_targeted"
                and not selected_matches and not text_matches and digest_ok
            ):
                parent_candidates: list[tuple[float, ExperimentChunk, str]] = []
                chunks_by_parent: dict[str, list[ExperimentChunk]] = {}
                for candidate_chunk in source_chunks:
                    chunks_by_parent.setdefault(candidate_chunk.parent_id, []).append(candidate_chunk)
                for parent_id, children in chunks_by_parent.items():
                    parent_text = plan.parents[parent_id]
                    verified, candidates = _parent_span_candidates(
                        parent_text, target["evidence_excerpt"], children,
                    )
                    parent_span_verified = parent_span_verified or verified
                    parent_candidates.extend(candidates)
                if parent_candidates:
                    parent_candidates.sort(key=lambda row: (-row[0], row[1].chunk_id))
                    best_coverage, best_chunk, reviewed_substring = parent_candidates[0]
                    parent_span_coverage = round(best_coverage, 6)
                    if best_coverage >= 0.80:
                        selected_matches = [best_chunk]
                        derived_excerpts[best_chunk.chunk_id] = reviewed_substring
                        mapping_mode = "parent_span_overlap"
            # Heading-aware splitting can turn a reviewer excerpt made solely
            # of adjacent Markdown headings into several section children.
            # The immutable reviewed C0 chunk still contains the complete
            # evidence.  Keep that C0 identity as parent-expanded gold rather
            # than inventing a direct child label or dropping the target.
            if (
                plan.corpus_mode == "snapshot_targeted"
                and not selected_matches
                and not text_matches
                and digest_ok
                and origin_parent_verified
                and source_chunks
            ):
                selected_matches = [sorted(source_chunks, key=lambda row: row.chunk_id)[0]]
                mapping_mode = "origin_c0_parent_expand_required"
                parent_span_verified = True
            report_row = {
                "query_id": item["query_id"],
                "source": target["source"],
                "old_chunk_id": target["chunk_id"],
                "original_answer_span_hash": target["answer_span_hash"],
                "digest_valid": digest_ok,
                "origin_parent_locked": origin_parent_locked,
                "exact_origin_child_lock": exact_origin_child_lock,
                "origin_parent_verified": origin_parent_verified,
                "mixed_source_rebuilt": mixed_source_rebuilt,
                "old_c0_parent_excluded": mixed_source_rebuilt,
                "cross_c0_boundary_mapping_allowed": mixed_source_rebuilt,
                "locked_origin_chunk_id": target["chunk_id"] if origin_parent_locked else None,
                "locked_parent_ids": locked_parent_ids,
                "all_source_chunk_count": len(all_source_chunks),
                "source_chunk_count": len(source_chunks),
                "text_match_count": len(text_matches),
                "heading_match_count": len(heading_matches),
                "new_chunk_ids": [chunk.chunk_id for chunk in selected_matches],
                "new_parent_ids": sorted({chunk.parent_id for chunk in selected_matches}),
                "mapping_mode": mapping_mode,
                "heading_drift": heading_drift,
                "parent_span_verified": parent_span_verified,
                "parent_span_coverage": parent_span_coverage,
                "derived_evidence_excerpt": (
                    derived_excerpts.get(selected_matches[0].chunk_id)
                    if len(selected_matches) == 1 else None
                ),
                "derived_answer_span_hash": (
                    answer_span_hash(derived_excerpts[selected_matches[0].chunk_id])
                    if len(selected_matches) == 1
                    and selected_matches[0].chunk_id in derived_excerpts
                    else None
                ),
            }
            target_reports.append(report_row)
            if not digest_ok or (origin_parent_locked and not origin_parent_verified) or not selected_matches:
                reason = (
                    "answer_span_hash_mismatch" if not digest_ok
                    else "reviewed_origin_parent_mismatch" if origin_parent_locked and not origin_parent_verified
                    else "origin_chunk_missing" if origin_parent_locked and not source_chunks
                    else "source_missing" if not source_chunks
                    else "span_crosses_chunk_boundary_or_missing" if not text_matches
                    else "ambiguous_span_heading_mismatch"
                )
                failures.append({**report_row, "reason": reason})
                continue
            parent_required = mapping_mode in {
                "parent_span_overlap", "origin_c0_parent_expand_required",
            }
            if parent_required:
                # Gold identity is the reviewed C0 parent, not the diagnostic
                # child with the largest partial overlap.  Any retrieved child
                # from this origin can expand to the same immutable parent.
                chunk = selected_matches[0]
                replacement = dict(target)
                replacement.update({
                    "evidence_scope": "parent_expand_required",
                    "origin_chunk_id": target["chunk_id"],
                    "parent_content_hash": (
                        "sha256:" + hashlib.sha256(
                            reviewed_origin_document.encode("utf-8")
                        ).hexdigest()
                        if reviewed_origin_document is not None
                        else chunk.parent_content_hash
                    ),
                    "review_note": (
                        replacement["review_note"]
                        + " Stage-D direct evidence requires immutable parent expansion; "
                          "it is excluded from strict child Direct metrics."
                    ),
                })
                new_targets.append(replacement)
                mapped_target_count += 1
            else:
                for chunk in selected_matches:
                    replacement = dict(target)
                    replacement.update({
                        "chunk_id": chunk.chunk_id,
                        "heading_path": list(chunk.heading_path),
                        "evidence_scope": "child",
                        "origin_chunk_id": chunk.origin_chunk_id or None,
                        "reviewed_origin_chunk_id": target["chunk_id"],
                        "parent_content_hash": chunk.parent_content_hash,
                    })
                    new_targets.append(replacement)
                    mapped_target_count += 1
        item["relevant_targets"] = new_targets
    report = {
        "schema_version": "chunk-qrels-map-report-v2",
        "experiment_fingerprint": plan.experiment_fingerprint,
        "collection": plan.collection_name,
        "original_target_count": sum(
            len(item["relevant_targets"]) for item in payload["items"]
        ),
        "mapped_target_count": mapped_target_count,
        "failure_count": len(failures),
        "heading_drift_count": sum(row.get("heading_drift", False) for row in target_reports),
        "parent_span_overlap_count": sum(
            row.get("mapping_mode") in {
                "parent_span_overlap", "origin_c0_parent_expand_required",
            }
            for row in target_reports
        ),
        "parent_expand_required_query_ids": sorted({
            row["query_id"] for row in target_reports
            if row.get("mapping_mode") in {
                "parent_span_overlap", "origin_c0_parent_expand_required",
            }
        }),
        "failures": failures,
        "targets": target_reports,
        "status": "mapped" if not failures else "mapping_failed",
    }
    if failures and strict:
        raise QrelsMappingError(
            f"{len(failures)} qrels targets could not map; see mapping report",
            report,
        )
    if failures:
        # A partial overlay would violate the reviewer contract for accepted
        # items.  It is returned only for diagnosis and must never be evaluated.
        return mapped, report
    validate_qrels_overlay(mapped)
    return mapped, report


def _planned_children_by_id(plan: ExperimentPlan) -> dict[str, ExperimentChunk]:
    planned = {chunk.chunk_id: chunk for chunk in plan.chunks}
    if len(planned) != len(plan.chunks):
        raise ChunkExperimentError("experiment plan contains duplicate child IDs")
    return planned


def _expected_child_metadata(
    chunk: ExperimentChunk,
    plan: ExperimentPlan,
    embed_profile: str,
) -> dict[str, Any]:
    metadata = chunk.chroma_metadata(plan.experiment_fingerprint)
    metadata["embed_profile"] = embed_profile
    return metadata


def _validate_child_metadata_rows(
    ids: Sequence[Any],
    metadatas: Sequence[Any],
    *,
    planned: dict[str, ExperimentChunk],
    plan: ExperimentPlan,
    embed_profile: str,
) -> None:
    if len(ids) != len(metadatas):
        raise ChunkExperimentError("existing experiment child metadata is misaligned")
    if len(set(ids)) != len(ids):
        raise ChunkExperimentError("experiment collection returned duplicate child IDs")
    for raw_child_id, child_metadata in zip(ids, metadatas):
        child_id = str(raw_child_id)
        child = planned.get(child_id)
        if child is None:
            raise ChunkExperimentError(f"unexpected experiment child ID: {child_id}")
        if not isinstance(child_metadata, dict):
            raise ChunkExperimentError(f"existing child lacks metadata: {child_id}")
        if child_metadata.get("embed_profile") != embed_profile:
            raise ChunkExperimentError(f"existing child embedding profile mismatch: {child_id}")
        if child_metadata.get("experiment_fingerprint") != plan.experiment_fingerprint:
            raise ChunkExperimentError(f"existing child experiment fingerprint mismatch: {child_id}")
        if child_metadata.get("chunk_id") != child_id:
            raise ChunkExperimentError(f"existing child stable identity mismatch: {child_id}")
        expected = _expected_child_metadata(child, plan, embed_profile)
        if child_metadata != expected:
            differing_keys = sorted({
                key for key in set(child_metadata) | set(expected)
                if child_metadata.get(key) != expected.get(key)
            })
            detail = differing_keys[0] if differing_keys else "unknown"
            raise ChunkExperimentError(
                f"existing child metadata mismatch ({detail}): {child_id}"
            )


def _assert_active_embedding_contract(plan: ExperimentPlan) -> dict[str, Any]:
    active = active_embedding_contract()
    if _embedding_contract_json(active) != _embedding_contract_json(plan.embedding_contract):
        raise ChunkExperimentError(
            "active embedding contract changed after planning; rebuild the experiment plan"
        )
    return active


def _experiment_write_batch_size(plan: ExperimentPlan) -> int:
    configured = plan.embedding_contract.get("batch_size", EXPERIMENT_WRITE_BATCH_SIZE)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
        raise ChunkExperimentError("embedding contract batch_size must be a positive integer")
    return min(configured, EXPERIMENT_WRITE_BATCH_SIZE)


def _validated_embeddings(
    embeddings: Any,
    chunks: Sequence[ExperimentChunk],
    *,
    expected_dimensions: Any,
) -> list[list[float]]:
    try:
        rows = list(embeddings)
    except TypeError as exc:
        raise ChunkExperimentError("embedding provider returned a non-iterable batch") from exc
    if len(rows) != len(chunks):
        raise ChunkExperimentError("embedding provider returned a misaligned batch")

    if expected_dimensions is not None:
        if (
            isinstance(expected_dimensions, bool)
            or not isinstance(expected_dimensions, int)
            or expected_dimensions <= 0
        ):
            raise ChunkExperimentError(
                "embedding contract dimensions must be a positive integer or null"
            )
        required_dimensions: int | None = expected_dimensions
    else:
        required_dimensions = None

    normalized: list[list[float]] = []
    observed_dimensions: int | None = None
    for chunk, row in zip(chunks, rows):
        if isinstance(row, (str, bytes)):
            raise ChunkExperimentError(
                f"embedding provider returned a non-vector for child {chunk.chunk_id}"
            )
        try:
            values = list(row)
        except TypeError as exc:
            raise ChunkExperimentError(
                f"embedding provider returned a non-vector for child {chunk.chunk_id}"
            ) from exc
        if not values:
            raise ChunkExperimentError(
                f"embedding provider returned an empty vector for child {chunk.chunk_id}"
            )
        if required_dimensions is not None and len(values) != required_dimensions:
            raise ChunkExperimentError(
                f"embedding dimension mismatch for child {chunk.chunk_id}: "
                f"expected {required_dimensions}, got {len(values)}"
            )
        if observed_dimensions is None:
            observed_dimensions = len(values)
        elif len(values) != observed_dimensions:
            raise ChunkExperimentError(
                f"embedding provider returned inconsistent dimensions for child {chunk.chunk_id}"
            )
        vector: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ChunkExperimentError(
                    f"embedding provider returned a non-numeric value for child {chunk.chunk_id}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ChunkExperimentError(
                    f"embedding provider returned a non-finite value for child {chunk.chunk_id}"
                )
            vector.append(number)
        normalized.append(vector)
    return normalized


def ensure_experiment_collection(
    client: Any,
    plan: ExperimentPlan,
    *,
    allow_approximate_tokenizer: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Create/resume an isolated collection; never delete or overwrite one."""

    assert_isolated_collection_name(
        plan.collection_name,
        production_collection=plan.production_collection,
        experiment_fingerprint=plan.experiment_fingerprint,
    )
    if plan.tokenizer.startswith("regex-") and not allow_approximate_tokenizer:
        raise ChunkExperimentError("approximate tokenizer is forbidden for collection writes")
    metadata = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "corpus_mode": plan.corpus_mode,
        "experiment_fingerprint": plan.experiment_fingerprint,
        "chunker_version": plan.spec.chunker_version,
        "production_collection": plan.production_collection,
        "tokenizer": plan.tokenizer,
        "embed_profile": str(plan.embedding_contract.get("embed_profile") or ""),
        "embedding_contract_json": _embedding_contract_json(plan.embedding_contract),
    }
    existing_names = {collection.name for collection in client.list_collections()}
    created = plan.collection_name not in existing_names
    if created:
        collection = client.create_collection(plan.collection_name, metadata=metadata)
    else:
        collection = client.get_collection(plan.collection_name)
        actual_metadata = dict(getattr(collection, "metadata", None) or {})
        if actual_metadata.get("experiment_fingerprint") != plan.experiment_fingerprint:
            raise ChunkExperimentError("existing experiment collection fingerprint mismatch")
        if actual_metadata.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise ChunkExperimentError("existing experiment collection schema mismatch")
        if actual_metadata.get("chunker_version") != plan.spec.chunker_version:
            raise ChunkExperimentError("existing experiment collection chunker mismatch")
        if actual_metadata.get("corpus_mode") != plan.corpus_mode:
            raise ChunkExperimentError("existing experiment collection corpus mode mismatch")
        if actual_metadata.get("production_collection") != plan.production_collection:
            raise ChunkExperimentError("existing experiment production collection mismatch")
        if actual_metadata.get("tokenizer") != plan.tokenizer:
            raise ChunkExperimentError("existing experiment tokenizer mismatch")
        if actual_metadata.get("embed_profile") != metadata["embed_profile"]:
            raise ChunkExperimentError("existing experiment embedding profile mismatch")
        if actual_metadata.get("embedding_contract_json") != metadata["embedding_contract_json"]:
            raise ChunkExperimentError("existing experiment embedding contract mismatch")
    planned = _planned_children_by_id(plan)
    snapshot = collection.get(include=["metadatas"])
    existing_id_rows = list(snapshot.get("ids") or [])
    existing_ids = set(existing_id_rows)
    plan_ids = set(planned)
    unexpected = sorted(existing_ids - plan_ids)
    if unexpected:
        raise ChunkExperimentError(
            f"experiment collection contains {len(unexpected)} unexpected IDs; refusing overwrite"
        )
    existing_metadatas = list(snapshot.get("metadatas") or [])
    _validate_child_metadata_rows(
        existing_id_rows,
        existing_metadatas,
        planned=planned,
        plan=plan,
        embed_profile=metadata["embed_profile"],
    )
    return collection, {
        "created": created,
        "existing": len(existing_ids),
        "missing": len(plan_ids - existing_ids),
        "complete": existing_ids == plan_ids,
    }


def write_experiment_collection(
    client: Any,
    plan: ExperimentPlan,
    *,
    reuse_embedding_collection: str = "",
) -> dict[str, Any]:
    """Embed/write missing children in bounded, restart-safe checkpoints."""

    collection, state = ensure_experiment_collection(client, plan)
    existing = set(collection.get(include=[]).get("ids") or [])
    missing = [chunk for chunk in plan.chunks if chunk.chunk_id not in existing]
    if not missing:
        return {**state, "added": 0, "count": collection.count()}
    from rag_tools import get_embeddings_batch

    active_contract = _assert_active_embedding_contract(plan)
    active_embed_profile = str(active_contract.get("embed_profile") or "")
    batch_size = _experiment_write_batch_size(plan)
    expected_dimensions = plan.embedding_contract.get("dimensions")
    planned = _planned_children_by_id(plan)
    production = (
        client.get_collection(plan.production_collection)
        if any(chunk.reuse_origin_embedding for chunk in missing)
        else None
    )
    reuse_collection = None
    if reuse_embedding_collection:
        if not reuse_embedding_collection.startswith(EXPERIMENT_COLLECTION_PREFIX):
            raise ChunkExperimentError(
                "embedding reuse source must be an isolated experiment collection"
            )
        if reuse_embedding_collection == plan.collection_name:
            raise ChunkExperimentError("embedding reuse source cannot be the destination")
        reuse_collection = client.get_collection(reuse_embedding_collection)
        reuse_metadata = dict(getattr(reuse_collection, "metadata", None) or {})
        if reuse_metadata.get("embedding_contract_json") != _embedding_contract_json(
            plan.embedding_contract
        ):
            raise ChunkExperimentError("embedding reuse collection contract mismatch")
    added = 0
    reused_experiment_embeddings = 0
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        # Fail closed if environment-backed embedding settings change during a
        # long build; already committed batches remain a valid resume point.
        _assert_active_embedding_contract(plan)
        vectors_by_id: dict[str, list[float]] = {}
        retained = [chunk for chunk in batch if chunk.reuse_origin_embedding]
        rebuilt = [chunk for chunk in batch if not chunk.reuse_origin_embedding]
        if retained:
            assert production is not None
            origin_ids = [chunk.origin_chunk_id for chunk in retained]
            if any(not origin_id for origin_id in origin_ids):
                raise ChunkExperimentError("retained C0 child lacks origin_chunk_id")
            reused = production.get(
                ids=origin_ids,
                include=["embeddings", "documents", "metadatas"],
            )
            raw_reused_embeddings = reused.get("embeddings")
            reused_embeddings = (
                [] if raw_reused_embeddings is None else list(raw_reused_embeddings)
            )
            reused_rows = {
                str(origin_id): (embedding, document, metadata)
                for origin_id, embedding, document, metadata in zip(
                    reused.get("ids") or [],
                    reused_embeddings,
                    reused.get("documents") or [],
                    reused.get("metadatas") or [],
                )
            }
            missing_origins = sorted(set(origin_ids) - set(reused_rows))
            if missing_origins:
                raise ChunkExperimentError(
                    f"production collection lacks retained origin embeddings: {missing_origins}"
                )
            ordered_vectors = []
            for chunk in retained:
                vector, document, metadata = reused_rows[chunk.origin_chunk_id]
                if document != chunk.text:
                    raise ChunkExperimentError(
                        f"retained C0 document drifted during build: {chunk.origin_chunk_id}"
                    )
                if not isinstance(metadata, dict) or metadata.get("source") != chunk.source:
                    raise ChunkExperimentError(
                        f"retained C0 metadata mismatch: {chunk.origin_chunk_id}"
                    )
                indexed_profile = str(metadata.get("embed_profile") or "")
                if indexed_profile and indexed_profile != active_embed_profile:
                    raise ChunkExperimentError(
                        f"retained C0 embedding profile mismatch: {chunk.origin_chunk_id}"
                    )
                ordered_vectors.append(vector)
            validated_reused = _validated_embeddings(
                ordered_vectors, retained, expected_dimensions=expected_dimensions,
            )
            vectors_by_id.update({
                chunk.chunk_id: vector
                for chunk, vector in zip(retained, validated_reused)
            })
        if rebuilt and reuse_collection is not None:
            reusable = reuse_collection.get(
                ids=[chunk.chunk_id for chunk in rebuilt],
                include=["embeddings", "documents", "metadatas"],
            )
            raw_vectors = reusable.get("embeddings")
            reuse_rows = {
                str(chunk_id): (vector, document, metadata)
                for chunk_id, vector, document, metadata in zip(
                    reusable.get("ids") or [],
                    [] if raw_vectors is None else list(raw_vectors),
                    reusable.get("documents") or [],
                    reusable.get("metadatas") or [],
                )
            }
            reusable_chunks = [chunk for chunk in rebuilt if chunk.chunk_id in reuse_rows]
            reuse_vectors: list[Any] = []
            for chunk in reusable_chunks:
                vector, document, metadata = reuse_rows[chunk.chunk_id]
                if document != chunk.text:
                    raise ChunkExperimentError(
                        f"embedding reuse document mismatch: {chunk.chunk_id}"
                    )
                if not isinstance(metadata, dict):
                    raise ChunkExperimentError(
                        f"embedding reuse metadata invalid: {chunk.chunk_id}"
                    )
                indexed_profile = str(metadata.get("embed_profile") or "")
                if indexed_profile and indexed_profile != active_embed_profile:
                    raise ChunkExperimentError(
                        f"embedding reuse profile mismatch: {chunk.chunk_id}"
                    )
                reuse_vectors.append(vector)
            validated_candidate_vectors = _validated_embeddings(
                reuse_vectors,
                reusable_chunks,
                expected_dimensions=expected_dimensions,
            )
            vectors_by_id.update({
                chunk.chunk_id: vector
                for chunk, vector in zip(reusable_chunks, validated_candidate_vectors)
            })
            reused_experiment_embeddings += len(reusable_chunks)
            reusable_ids = {chunk.chunk_id for chunk in reusable_chunks}
            rebuilt = [chunk for chunk in rebuilt if chunk.chunk_id not in reusable_ids]
        if rebuilt:
            validated_rebuilt = _validated_embeddings(
                get_embeddings_batch(
                    [chunk.embedding_text for chunk in rebuilt],
                    batch_size=batch_size,
                ),
                rebuilt,
                expected_dimensions=expected_dimensions,
            )
            vectors_by_id.update({
                chunk.chunk_id: vector
                for chunk, vector in zip(rebuilt, validated_rebuilt)
            })
        embeddings = [vectors_by_id[chunk.chunk_id] for chunk in batch]
        metadatas = [
            _expected_child_metadata(chunk, plan, active_embed_profile)
            for chunk in batch
        ]
        batch_ids = [chunk.chunk_id for chunk in batch]
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=[chunk.text for chunk in batch],
            metadatas=metadatas,
        )
        # Treat each successful add as a durable checkpoint only after Chroma
        # returns every requested stable ID with the exact planned metadata.
        committed = collection.get(ids=batch_ids, include=["metadatas"])
        committed_ids = list(committed.get("ids") or [])
        if set(committed_ids) != set(batch_ids) or len(committed_ids) != len(batch_ids):
            raise ChunkExperimentError("collection.add returned without committing the full batch")
        _validate_child_metadata_rows(
            committed_ids,
            list(committed.get("metadatas") or []),
            planned=planned,
            plan=plan,
            embed_profile=active_embed_profile,
        )
        added += len(batch)

    final_ids = list(collection.get(include=[]).get("ids") or [])
    if len(final_ids) != len(set(final_ids)) or set(final_ids) != set(planned):
        raise ChunkExperimentError("experiment collection is incomplete after batched write")
    final_count = collection.count()
    if final_count != len(plan.chunks):
        raise ChunkExperimentError("experiment collection count is inconsistent after batched write")
    result = {
        **state,
        "added": added,
        "count": final_count,
        "complete": True,
    }
    if reuse_embedding_collection:
        result.update({
            "reuse_embedding_collection": reuse_embedding_collection,
            "reused_experiment_embeddings": reused_experiment_embeddings,
        })
    return result


def resolve_source_file(
    root: Path,
    source: str,
    *,
    indexed_documents: Sequence[str] = (),
    indexed_titles: Sequence[str] = (),
    candidates: Sequence[Path] | None = None,
    diagnostic: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Resolve an indexed basename to one unambiguous repository Markdown file."""

    ignored = {
        ".git", ".venv", ".claude", ".pytest_cache", "chroma_db",
        "logs", "_pending", "__pycache__",
    }
    if candidates is None:
        candidates = [
            path for path in root.rglob(source)
            if path.is_file() and not any(part in ignored for part in path.parts)
        ]
    candidates = list(candidates)
    normalized_docs = ["".join(document.split()) for document in indexed_documents if document]

    def exact_coverage(path: Path) -> tuple[int, float]:
        normalized_source = "".join(path.read_text(encoding="utf-8", errors="replace").split())
        exact = sum(document in normalized_source for document in normalized_docs)
        return exact, (exact / len(normalized_docs) if normalized_docs else 1.0)

    if len(candidates) == 1:
        exact, coverage = exact_coverage(candidates[0])
        if diagnostic is not None:
            diagnostic.update({
                "candidate_count": 1,
                "selected_score": 1.0,
                "runner_up_score": None,
                "exact_indexed_chunks": exact,
                "indexed_chunk_count": len(normalized_docs),
                "content_coverage": round(coverage, 6),
                "content_drift": coverage < 1.0,
            })
        return candidates[0], "unique_basename"
    if len(candidates) > 1 and (indexed_documents or indexed_titles):
        # Basenames such as README.md are legitimately repeated.  Resolve them
        # with current collection evidence, not a guessed directory priority.
        normalized_titles = {"".join(title.split()) for title in indexed_titles if title}

        def normalized_semantic_text(value: str) -> str:
            return re.sub(r"[^\w\u3400-\u9fff]+", "", value.lower())

        def ngrams(value: str, size: int = 3) -> set[str]:
            normalized = normalized_semantic_text(value)
            if len(normalized) < size:
                return {normalized} if normalized else set()
            return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}

        indexed_grams = ngrams("\n".join(indexed_documents))
        scores: list[tuple[int, int, float, Path]] = []
        for path in candidates:
            raw_source = path.read_text(encoding="utf-8", errors="replace")
            normalized_source = "".join(raw_source.split())
            exact_score = sum(document in normalized_source for document in normalized_docs)
            candidate_titles = {
                "".join(match.group(2).strip().split())
                for line in raw_source.splitlines()
                if (match := _HEADING_RE.match(line))
            }
            heading_score = len(normalized_titles & candidate_titles)
            candidate_grams = ngrams(raw_source)
            ngram_recall = (
                len(indexed_grams & candidate_grams) / len(indexed_grams)
                if indexed_grams else 0.0
            )
            # Exact indexed body evidence is strongest; old indexes can outlive
            # edits to their source README, so stable heading evidence is the
            # explicit second key rather than a directory-name guess.
            scores.append((exact_score, heading_score, ngram_recall, path))
        exact_best = max((exact for exact, _headings, _ngram, _path in scores), default=0)
        if exact_best > 0:
            exact_winners = [path for exact, _headings, _ngram, path in scores if exact == exact_best]
            if len(exact_winners) == 1:
                exact, coverage = exact_coverage(exact_winners[0])
                if diagnostic is not None:
                    diagnostic.update({
                        "candidate_count": len(candidates),
                        "selected_score": 1.0,
                        "runner_up_score": None,
                        "exact_indexed_chunks": exact,
                        "indexed_chunk_count": len(normalized_docs),
                        "content_coverage": round(coverage, 6),
                        "content_drift": coverage < 1.0,
                    })
                return exact_winners[0], "indexed_content_evidence"
        # Old collections may outlive an edited source file.  Fuzzy matching
        # is accepted only with a meaningful absolute score and unique margin.
        heading_denominator = max(len(normalized_titles), 1)
        fuzzy_scores = sorted([
            (
                round(0.80 * ngram + 0.20 * (headings / heading_denominator), 6),
                path,
                ngram,
                headings,
            )
            for _exact, headings, ngram, path in scores
        ], key=lambda row: (-row[0], str(row[1])))
        best_score = fuzzy_scores[0][0] if fuzzy_scores else 0.0
        runner_up = fuzzy_scores[1][0] if len(fuzzy_scores) > 1 else 0.0
        if best_score >= 0.20 and best_score - runner_up >= 0.10:
            selected = fuzzy_scores[0][1]
            exact, coverage = exact_coverage(selected)
            if diagnostic is not None:
                diagnostic.update({
                    "candidate_count": len(candidates),
                    "selected_score": best_score,
                    "runner_up_score": runner_up,
                    "score_margin": round(best_score - runner_up, 6),
                    "ngram_recall": round(fuzzy_scores[0][2], 6),
                    "heading_matches": fuzzy_scores[0][3],
                    "exact_indexed_chunks": exact,
                    "indexed_chunk_count": len(normalized_docs),
                    "content_coverage": round(coverage, 6),
                    "content_drift": True,
                })
            return selected, "fuzzy_indexed_content_evidence"
    raise ChunkExperimentError(
        f"source {source!r} resolved to {len(candidates)} files without a unique "
        f"evidence match: {[str(path) for path in candidates[:5]]}"
    )


def production_sources(
    root: Path,
    collection: Any,
    *,
    resolution_report: dict[str, Any] | None = None,
    include_sources: Sequence[str] | None = None,
) -> list[SourceDocument]:
    """Resolve the exact source membership of a production collection."""

    snapshot = collection.get(include=["documents", "metadatas"])
    by_source: dict[str, dict[str, str]] = {}
    source_order: list[str] = []
    indexed_documents: dict[str, list[str]] = {}
    indexed_titles: dict[str, list[str]] = {}
    documents = snapshot.get("documents") or []
    metadatas = snapshot.get("metadatas") or []
    if len(documents) != len(metadatas):
        raise ChunkExperimentError("production collection returned misaligned source manifest")
    for document, metadata in zip(documents, metadatas):
        if not isinstance(metadata, dict) or not metadata.get("source"):
            continue
        source = str(metadata["source"])
        values = {
            "source_type": str(metadata.get("source_type") or "doc"),
            "owner_scope": str(metadata.get("owner_scope") or "general"),
        }
        if source not in by_source:
            source_order.append(source)
        previous = by_source.setdefault(source, values)
        if previous != values:
            raise ChunkExperimentError(f"inconsistent source metadata for {source}")
        if isinstance(document, str):
            indexed_documents.setdefault(source, []).append(document)
        if metadata.get("title"):
            indexed_titles.setdefault(source, []).append(str(metadata["title"]))
    if not by_source:
        raise ChunkExperimentError("production collection exposed no source metadata")
    if include_sources is not None:
        requested_sources = set(str(source) for source in include_sources)
        missing_sources = sorted(requested_sources - set(by_source))
        if missing_sources:
            raise ChunkExperimentError(
                f"requested production sources are absent from the live manifest: {missing_sources}"
            )
        by_source = {
            source: metadata for source, metadata in by_source.items()
            if source in requested_sources
        }
        source_order = [source for source in source_order if source in requested_sources]
        indexed_documents = {
            source: values for source, values in indexed_documents.items()
            if source in requested_sources
        }
        indexed_titles = {
            source: values for source, values in indexed_titles.items()
            if source in requested_sources
        }
    ignored = {
        ".git", ".venv", ".claude", ".pytest_cache", "chroma_db",
        "logs", "_pending", "__pycache__",
    }
    # Walk the repository once and prune large ignored trees.  Calling rglob
    # once per 100 sources traversed worktrees millions of times in dry-runs.
    candidate_index: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in ignored]
        directory = Path(dirpath)
        for filename in filenames:
            if filename in by_source:
                candidate_index.setdefault(filename, []).append(directory / filename)

    # Reconstruct the production ingest order while keeping collection
    # membership authoritative.  Chroma ``get`` order is not an insertion-order
    # contract and often appears ID-sorted, which changes prefix dedup winners.
    try:
        from rag_ingest import DEFAULT_FILES, _discover_knowledge_base
        configured_order = [
            Path(path).name for path, _source_type
            in [*DEFAULT_FILES, *_discover_knowledge_base()]
        ]
    except Exception:
        configured_order = []
    ordered_sources: list[str] = []
    for source in [*configured_order, *source_order]:
        if source in by_source and source not in ordered_sources:
            ordered_sources.append(source)

    resolved: list[SourceDocument] = []
    resolution_rows: list[dict[str, Any]] = []
    for source in ordered_sources:
        metadata = by_source[source]
        diagnostic: dict[str, Any] = {}
        path, method = resolve_source_file(
            root,
            source,
            indexed_documents=indexed_documents.get(source, ()),
            indexed_titles=indexed_titles.get(source, ()),
            candidates=candidate_index.get(source, ()),
            diagnostic=diagnostic,
        )
        resolved.append(SourceDocument(
            path=str(path),
            source=source,
            source_type=metadata["source_type"],
            owner_scope=metadata["owner_scope"],
        ))
        resolution_rows.append({
            "source": source,
            "path": str(path),
            "method": method,
            **diagnostic,
        })
    if resolution_report is not None:
        resolution_report.update({
            "production_chunk_count": len(documents),
            "unique_source_count": len(by_source),
            "resolved_source_count": len(resolved),
            "unresolved_source_count": 0,
            "content_drift_source_count": sum(
                bool(row.get("content_drift")) for row in resolution_rows
            ),
            "content_drift_sources": [
                row["source"] for row in resolution_rows if row.get("content_drift")
            ],
            "resolution_methods": {
                method: sum(row["method"] == method for row in resolution_rows)
                for method in sorted({row["method"] for row in resolution_rows})
            },
            "sources": resolution_rows,
            "status": "complete",
            "source_order": "production-ingest-order-reconstructed-from-live-manifest",
            "ordered_sources": ordered_sources,
            "filtered_source_manifest": include_sources is not None,
        })
    return resolved
