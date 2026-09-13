# -*- coding: utf-8 -*-
"""Production retrieval profiles and stage-level trace contracts.

This module deliberately contains no retrieval implementation.  The production
implementation remains in :mod:`rag_gate`; keeping the data contract separate
lets evaluators consume the exact same trace without maintaining a second
search pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from types import MappingProxyType
from typing import Mapping
from typing import Any, Iterable


_PROFILE_NAMES = {"baseline", "candidate_shadow", "candidate"}
_CANDIDATE_POOL_STRATEGIES = {
    "baseline_rrf20",
    "source_cap4",
    "retain_channel_exclusives",
}
_RERANK_PREFIX_MODES = {"none", "compact32", "full"}


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    from rag_mode import mode_env
    raw = mode_env(name)                       # 显式 env 优先,RAG_MODE 预设补缺省
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RetrievalProfile:
    """A fully resolved retrieval profile used by one production query.

    ``pool_size`` is the single source of truth for the main dense/BM25/RRF
    candidate depth and reranker pool.  It replaces the formerly disconnected
    ``RAG_RERANK_POOL`` knob while keeping the baseline default at 20.
    """

    name: str = "baseline"
    pool_size: int = 20
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_use_breadcrumb: bool = False
    reranker_prefix_mode: str = "none"
    rrf_k: int = 60
    dense_rrf_weight: float = 1.0
    bm25_rrf_weight: float = 1.0
    consensus_guard_top_k: int = 0
    consensus_guard_margin: float = 0.03
    # Candidate membership is independently versioned from the reranker.  The
    # baseline keeps the production RRF pool byte-for-byte unchanged; Stage-C
    # alternatives must be selected explicitly by an offline/candidate
    # profile before they can affect the fusion -> rerank boundary.
    candidate_pool_strategy: str = "baseline_rrf20"
    # Only read by ``retain_channel_exclusives``: how many top candidates each
    # channel may reserve before the fused order fills the rest of the pool.
    exclusive_per_channel: int = 4
    chunker_version: str = ""
    # ``None`` preserves the production environment switch.  Offline A/B
    # profiles set these to ``False`` so a developer shell cannot silently
    # contaminate a supposedly fixed reranker experiment.
    enable_hyde: bool | None = None
    enable_query_rewrite: bool | None = None
    enable_doc2query: bool | None = None
    enable_quota: bool | None = None
    enable_alias: bool | None = None
    enable_rerank_bridge: bool | None = None
    # HyDE as an extra dense *channel* rather than a replacement query.  The
    # existing ``enable_hyde`` rewrites the text that gets embedded, so every
    # distance in the pool becomes a "hypothetical answer -> document" distance
    # while the evidence gate still compares them against thresholds calibrated
    # on "question -> document" distances.  On Final v2 that mismatch cost a
    # genuine false accept.  The channel form keeps the original query's
    # distances for gating and uses HyDE only to widen membership.
    enable_hyde_channel: bool | None = None
    # Same idea on the lexical side: the deterministic gap probe found 3 of the
    # 21 Final-v2 out-of-pool golds (and 1 on dev80) reachable *only* by BM25
    # over the HyDE text.  No distance-distribution issue here -- BM25
    # candidates already enter the pool with a placeholder distance.
    enable_hyde_bm25_channel: bool | None = None
    # Stage 2 Q1: the same HyDE call also emits canonical terms, used as one
    # extra dense + one extra BM25 query representation.  Terms feed retrieval
    # queries only -- never evidence, citations or gate distances.
    enable_canonical_terms: bool | None = None
    def __post_init__(self) -> None:
        if self.candidate_pool_strategy not in _CANDIDATE_POOL_STRATEGIES:
            raise ValueError(
                "unknown candidate_pool_strategy: "
                f"{self.candidate_pool_strategy!r}"
            )
        if self.reranker_prefix_mode not in _RERANK_PREFIX_MODES:
            raise ValueError(f"unknown reranker_prefix_mode: {self.reranker_prefix_mode!r}")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.dense_rrf_weight <= 0 or self.bm25_rrf_weight <= 0:
            raise ValueError("RRF channel weights must be positive")
        if self.consensus_guard_top_k < 0:
            raise ValueError("consensus_guard_top_k cannot be negative")
        if self.consensus_guard_margin < 0:
            raise ValueError("consensus_guard_margin cannot be negative")
        if self.exclusive_per_channel < 0:
            raise ValueError("exclusive_per_channel cannot be negative")
        if (self.candidate_pool_strategy == "retain_channel_exclusives"
                and 2 * self.exclusive_per_channel >= self.pool_size):
            # A single-channel candidate scores about half of a dual-channel
            # one at the same rank, so once the reservations can fill the pool
            # the strategy evicts every candidate both channels agreed on.
            # Measured on Dev-New at pool 20 x exclusive 12: RRF candidate
            # recall collapsed 67/80 -> 23/80.  The pool builder's truncation
            # guard keeps that legal, so it has to be rejected here, where a
            # config error is still a config error and not 80 bad queries.
            raise ValueError(
                "retain_channel_exclusives reserves 2 x "
                f"{self.exclusive_per_channel} slots of a {self.pool_size} "
                "pool, leaving no room for dual-channel candidates"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieval_profile_registry() -> Mapping[str, RetrievalProfile]:
    """Return the immutable, fully resolved production profile registry.

    The baseline and candidate deliberately use separate environment keys.  A
    candidate may therefore be enabled, shadowed, and rolled back without
    mutating the baseline model.  The historical global reranker variables are
    accepted only as baseline compatibility inputs.

    Query/document expansion, quotas and bridge experiments are explicitly
    forbidden in both release profiles.  An offline experiment can still pass
    an explicit :class:`RetrievalProfile` to ``retrieve_with_trace``.
    """

    # Final v3 verdict (2026-08-29): B0 (pool 28 + compact32 prefix + judge) beat
    # the old default on every ranking metric of a fresh blind set -- R@1 38->43,
    # R@3 51->55, R@5 54->57, MRR .558->.613, nDCG@5 .583->.634, candidate 56->60,
    # true false-accepts 1->0 -- so pool 28 is now the production default.
    fallback_pool = _positive_int(os.environ.get("RAG_RECALL_N", "28"), 28)
    baseline_pool_raw = (
        os.environ.get("RAG_BASELINE_RERANK_POOL", "").strip()
        or os.environ.get("RAG_RERANK_POOL", "").strip()
    )
    baseline_pool = (
        _positive_int(baseline_pool_raw, fallback_pool)
        if baseline_pool_raw else fallback_pool
    )
    candidate_pool_raw = os.environ.get("RAG_CANDIDATE_RERANK_POOL", "").strip()
    candidate_pool = (
        _positive_int(candidate_pool_raw, baseline_pool)
        if candidate_pool_raw else baseline_pool
    )
    baseline_model = (
        os.environ.get("RAG_BASELINE_RERANK_MODEL", "").strip()
        or os.environ.get("RAG_RERANK_MODEL", "").strip()
        or "BAAI/bge-reranker-base"
    )
    candidate_model = (
        os.environ.get("RAG_CANDIDATE_RERANK_MODEL", "").strip()
        or baseline_model
    )
    def _positive_float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, str(default)))
        except ValueError:
            return default
        return value if value > 0 else default

    candidate_prefix_mode = os.environ.get(
        "RAG_CANDIDATE_RERANK_PREFIX_MODE", "none"
    ).strip().lower()
    if candidate_prefix_mode not in _RERANK_PREFIX_MODES:
        candidate_prefix_mode = "none"
    candidate_pool_strategy = os.environ.get(
        "RAG_CANDIDATE_POOL_STRATEGY", "baseline_rrf20"
    ).strip()
    if candidate_pool_strategy not in _CANDIDATE_POOL_STRATEGIES:
        # A malformed candidate configuration must not contaminate the
        # synchronous baseline.  Fail the candidate arm back to its audited
        # pool instead of accepting an arbitrary strategy name.
        candidate_pool_strategy = "baseline_rrf20"
    try:
        from rag_tools import CHUNKER_VERSION
        chunker_version = CHUNKER_VERSION
    except Exception:
        chunker_version = ""

    # r1 攻坚的预注册纪律:默认硬关全部增强,env 旋钮无效——保证冻结基线可复现。
    # 逃生舱(2026-08-24):RAG_PROFILE_DEFER_ENV=1 时改为 None(遵从各自 env 旋钮),
    # 供配额/别名/分工精排等实验 harness 使用;默认行为逐字节不变。
    # 动机:硬 False 曾让旧评测 harness 静默测成 B0(xling 全 0 才暴露)。
    _defer = os.environ.get("RAG_PROFILE_DEFER_ENV", "0") == "1"
    _v = None if _defer else False
    forbidden_enhancements = {
        "enable_hyde": _v,
        "enable_query_rewrite": _v,
        "enable_doc2query": _v,
        "enable_quota": _v,
        "enable_alias": _v,
        "enable_rerank_bridge": _v,
    }
    # HyDE as two extra channels is part of the accepted production default
    # since Final v4 (candidate 69->73, R@3/R@5 up, R@1 untouched).  One
    # explicit opt-out turns both off; they are not experiment knobs any more.
    _channels_on = _env_bool("RAG_HYDE_CHANNELS", True)
    forbidden_enhancements["enable_hyde_channel"] = _channels_on if not _defer else None
    forbidden_enhancements["enable_hyde_bm25_channel"] = _channels_on if not _defer else None
    baseline_prefix_mode = os.environ.get(
        "RAG_BASELINE_RERANK_PREFIX_MODE", "compact32"
    ).strip().lower()
    if baseline_prefix_mode not in _RERANK_PREFIX_MODES:
        baseline_prefix_mode = "compact32"
    baseline = RetrievalProfile(
        name="baseline",
        pool_size=baseline_pool,
        reranker_model=baseline_model,
        reranker_use_breadcrumb=_env_bool(
            "RAG_BASELINE_RERANK_BREADCRUMB", False
        ),
        reranker_prefix_mode=baseline_prefix_mode,
        rrf_k=60,
        dense_rrf_weight=1.0,
        bm25_rrf_weight=1.0,
        consensus_guard_top_k=0,
        candidate_pool_strategy="baseline_rrf20",
        chunker_version=chunker_version,
        **forbidden_enhancements,
    )
    candidate = RetrievalProfile(
        name="candidate",
        pool_size=candidate_pool,
        reranker_model=candidate_model,
        reranker_use_breadcrumb=_env_bool(
            "RAG_CANDIDATE_RERANK_BREADCRUMB", False
        ),
        reranker_prefix_mode=candidate_prefix_mode,
        rrf_k=_positive_int(os.environ.get("RAG_CANDIDATE_RRF_K", "60"), 60),
        dense_rrf_weight=_positive_float("RAG_CANDIDATE_DENSE_RRF_WEIGHT", 1.0),
        bm25_rrf_weight=_positive_float("RAG_CANDIDATE_BM25_RRF_WEIGHT", 1.0),
        consensus_guard_top_k=(
            _positive_int(os.environ.get("RAG_CANDIDATE_CONSENSUS_TOP_K", "5"), 5)
            if _env_bool("RAG_CANDIDATE_CONSENSUS_GUARD", False) else 0
        ),
        consensus_guard_margin=_positive_float(
            "RAG_CANDIDATE_CONSENSUS_MARGIN", 0.03
        ),
        candidate_pool_strategy=candidate_pool_strategy,
        chunker_version=chunker_version,
        **forbidden_enhancements,
    )
    # ``candidate_shadow`` is an execution alias: the synchronous/user-facing
    # path always resolves to baseline.  The candidate is available only via
    # ``resolve_candidate_shadow_profile`` for an isolated shadow executor.
    return MappingProxyType({
        "baseline": baseline,
        "candidate": candidate,
        "candidate_shadow": baseline,
    })


def resolve_retrieval_profile(
    profile: RetrievalProfile | str | None = None,
) -> RetrievalProfile:
    """Resolve an explicit profile or the active synchronous profile.

    ``candidate_shadow`` is deliberately an alias of baseline here.  Only the
    separate shadow accessor can obtain its candidate arm.
    """

    if isinstance(profile, RetrievalProfile):
        return profile
    name = str(profile or os.environ.get("RAG_RETRIEVAL_PROFILE", "baseline")).strip()
    if name not in _PROFILE_NAMES:
        name = "baseline"
    return retrieval_profile_registry()[name]


def resolve_candidate_shadow_profile() -> RetrievalProfile:
    """Return the isolated candidate arm for a shadow executor.

    Calling this accessor never changes the synchronous active profile.  It is
    intentionally separate so API code cannot accidentally serve the shadow
    candidate merely by setting ``RAG_RETRIEVAL_PROFILE=candidate_shadow``.
    """

    return retrieval_profile_registry()["candidate"]


def stable_chunk_id(document: str, metadata: dict | None = None) -> str:
    """Return the ingest-compatible ID for old and new Chroma chunks.

    New callers should prefer the actual Chroma ID when available.  This
    deterministic fallback mirrors ``rag_ingest`` (source stem + text MD5), so
    BM25-only and derived candidates still have a stable identity.
    """

    metadata = metadata or {}
    for key in ("chunk_id", "id"):
        if metadata.get(key):
            return str(metadata[key])
    source = os.path.basename(str(metadata.get("source") or "chunk"))
    stem = source[:-3] if source.lower().endswith(".md") else source
    digest = hashlib.md5(str(document or "").encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def bind_profile_fingerprint(
    index_metadata: dict[str, Any], profile: RetrievalProfile,
) -> dict[str, Any]:
    """Bind a collection fingerprint to the actual explicit retrieval arm.

    ``rag_tools.index_fingerprint`` reports the process environment model for
    backwards compatibility.  A/B profiles can now select a reranker without
    mutating that environment, so the trace must replace that stale field and
    derive a profile-aware fingerprint before results are compared.
    """

    out = dict(index_metadata or {})
    out["rerank_model"] = profile.reranker_model
    out["retrieval_profile"] = profile.to_dict()
    stable = {
        key: value for key, value in out.items()
        if key not in {
            "fingerprint_id", "generated_at", "git_commit", "audit_git_commit",
        }
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    out["fingerprint_id"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return out


@dataclass
class CandidateTrace:
    chunk_id: str
    source: str
    source_type: str
    heading_path: str
    rank: int
    dense_distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    channels: list[str] = field(default_factory=list)
    # The document is useful to an in-process evaluator but is intentionally
    # omitted from API/SSE serialisation to avoid leaking corpus text.
    document: str = field(default="", repr=False)

    def to_dict(self, *, include_document: bool = False) -> dict[str, Any]:
        out = asdict(self)
        if not include_document:
            out.pop("document", None)
        return out


@dataclass
class RetrievalTrace:
    retrieval_profile: RetrievalProfile
    index_fingerprint: str = ""
    index_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    dense_candidates: list[CandidateTrace] = field(default_factory=list)
    bm25_candidates: list[CandidateTrace] = field(default_factory=list)
    fusion_candidates: list[CandidateTrace] = field(default_factory=list)
    reranked_candidates: list[CandidateTrace] = field(default_factory=list)
    final_candidates: list[CandidateTrace] = field(default_factory=list)
    gate_decision: bool = False
    gate_features: dict[str, Any] = field(default_factory=dict)
    latency_by_stage: dict[str, float] = field(default_factory=dict)
    effective_hit: bool = False
    reranker_requested: str = ""
    reranker_actual: str = ""
    reranker_status: str = "not_run"
    reranker_scored_count: int = 0
    gate_candidate_chunk_id: str = ""
    final_top1_chunk_id: str = ""
    gate_alignment: bool | None = None

    def to_dict(self, *, include_documents: bool = False) -> dict[str, Any]:
        return {
            "retrieval_profile": self.retrieval_profile.to_dict(),
            "index_fingerprint": self.index_fingerprint,
            "index_metadata": dict(self.index_metadata),
            "metadata_filters": dict(self.metadata_filters),
            "dense_candidates": [
                item.to_dict(include_document=include_documents)
                for item in self.dense_candidates
            ],
            "bm25_candidates": [
                item.to_dict(include_document=include_documents)
                for item in self.bm25_candidates
            ],
            "fusion_candidates": [
                item.to_dict(include_document=include_documents)
                for item in self.fusion_candidates
            ],
            "reranked_candidates": [
                item.to_dict(include_document=include_documents)
                for item in self.reranked_candidates
            ],
            "final_candidates": [
                item.to_dict(include_document=include_documents)
                for item in self.final_candidates
            ],
            "gate_decision": self.gate_decision,
            "gate_features": dict(self.gate_features),
            "latency_by_stage": dict(self.latency_by_stage),
            "effective_hit": self.effective_hit,
            "reranker_requested": self.reranker_requested,
            "reranker_actual": self.reranker_actual,
            "reranker_status": self.reranker_status,
            "reranker_scored_count": self.reranker_scored_count,
            "gate_candidate_chunk_id": self.gate_candidate_chunk_id,
            "final_top1_chunk_id": self.final_top1_chunk_id,
            "gate_alignment": self.gate_alignment,
        }


def candidate_traces(
    docs: Iterable[str],
    metas: Iterable[dict | None],
    *,
    ids: Iterable[str] | None = None,
    dense_distances: Iterable[float | None] | None = None,
    bm25_scores: Iterable[float | None] | None = None,
    rrf_scores: dict[str, float] | None = None,
    rerank_scores: Iterable[float | None] | None = None,
    channels: dict[str, list[str]] | None = None,
) -> list[CandidateTrace]:
    """Build aligned candidate records without changing retrieval ordering."""

    docs_l = list(docs)
    metas_l = list(metas)
    ids_l = list(ids or [])
    dense_l = list(dense_distances or [])
    bm25_l = list(bm25_scores or [])
    rerank_l = list(rerank_scores or [])
    out: list[CandidateTrace] = []
    for index, document in enumerate(docs_l):
        meta = metas_l[index] if index < len(metas_l) and metas_l[index] else {}
        chunk_id = (str(ids_l[index]) if index < len(ids_l) and ids_l[index]
                    else stable_chunk_id(document, meta))
        heading = str(
            meta.get("heading_path") or meta.get("section_path")
            or meta.get("title") or ""
        )
        out.append(CandidateTrace(
            chunk_id=chunk_id,
            source=str(meta.get("source") or ""),
            source_type=str(meta.get("source_type") or ""),
            heading_path=heading,
            rank=index + 1,
            dense_distance=(float(dense_l[index])
                            if index < len(dense_l) and dense_l[index] is not None else None),
            bm25_score=(float(bm25_l[index])
                        if index < len(bm25_l) and bm25_l[index] is not None else None),
            rrf_score=(float((rrf_scores or {})[document])
                       if document in (rrf_scores or {}) else None),
            rerank_score=(float(rerank_l[index])
                          if index < len(rerank_l) and rerank_l[index] is not None else None),
            channels=list((channels or {}).get(document, [])),
            document=document,
        ))
    return out


def rrf_diagnostics(
    dense_docs: list[str], bm25_hits: list[tuple], *, k: int = 60,
    dense_weight: float = 1.0, bm25_weight: float = 1.0,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Reconstruct the production RRF scores solely for trace diagnostics."""

    scores: dict[str, float] = {}
    channels: dict[str, list[str]] = {}
    for channel, weight, docs in (
        ("dense", dense_weight, dense_docs),
        ("bm25", bm25_weight, [item[0] for item in bm25_hits]),
    ):
        for rank, document in enumerate(docs, start=1):
            scores[document] = scores.get(document, 0.0) + weight / (k + rank)
            channels.setdefault(document, []).append(channel)
    return scores, channels
