# -*- coding: utf-8 -*-
"""Version-bound Evidence Gate profiles and offline calibration.

The production gate deliberately remains in :mod:`rag_gate` until a reranker
arm wins the complete release gate.  This module provides the independent
contract needed to calibrate that winner without changing today's answers.

Two safety properties are intentionally strict:

* a calibrated profile only applies to the exact embedding/reranker/chunker/
  retrieval/index signature used to create it;
* calibration never searches a score threshold without corroboration.  Every
  candidate requires a reranker score *and* at least one independent support
  feature (dense, BM25, margin, vector membership, or lexical rescue).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


GATE_PROFILE_SCHEMA = "offerclaw-gate-profile-v1"
GATE_REGISTRY_SCHEMA = "offerclaw-gate-profile-registry-v1"
CALIBRATION_SCHEMA = "offerclaw-gate-calibration-v1"
DEFAULT_GATE_PROFILE_PATH = Path(__file__).resolve().parent / "config" / "rag_gate_profiles.json"

_SUPPORT_SIGNALS = frozenset({
    "vector_in_kb", "dense_distance", "bm25_rank", "rerank_margin",
    "lexical_rescued",
})


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


@dataclass(frozen=True)
class GateFeatures:
    """Auditable features consumed by a calibrated Evidence Gate."""

    best_dense_distance: float | None = None
    bm25_best_rank: int | None = None
    rerank_top: float | None = None
    rerank_margin: float | None = None
    lexical_rescued: bool = False
    vector_in_kb: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GateFeatures":
        payload: Mapping[str, Any] = value or {}
        nested = payload.get("gate_features")
        if isinstance(nested, Mapping):
            payload = {**payload, **nested}
        return cls(
            best_dense_distance=_finite_float(payload.get("best_dense_distance")),
            bm25_best_rank=_positive_int(payload.get("bm25_best_rank")),
            rerank_top=_finite_float(payload.get("rerank_top")),
            rerank_margin=_finite_float(payload.get("rerank_margin")),
            lexical_rescued=bool(payload.get("lexical_rescued", False)),
            vector_in_kb=bool(payload.get("vector_in_kb", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gate_features_from_trace(trace: Any) -> GateFeatures:
    """Extract gate features from a ``RetrievalTrace`` or serialized trace.

    Features are never silently reconstructed for the final Top-1 candidate.
    Gate scores were measured for ``gate_candidate_chunk_id``; substituting a
    post-fusion candidate would mix two evidence identities during calibration.
    Legacy rows missing a feature remain explicitly incomplete.
    """

    if hasattr(trace, "to_dict"):
        payload = trace.to_dict(include_documents=False)
    elif isinstance(trace, Mapping):
        payload = dict(trace)
    else:
        raise TypeError("trace must be a RetrievalTrace or mapping")

    return GateFeatures.from_mapping(payload.get("gate_features") or {})


def gate_alignment_from_trace(trace: Any) -> tuple[str, str, bool | None]:
    """Return the explicit Gate/final candidate alignment contract.

    Alignment is valid only when both IDs are present, equal, and the producer
    explicitly recorded ``gate_alignment=True``.  Missing legacy fields are not
    guessed from candidate arrays.
    """

    if hasattr(trace, "to_dict"):
        payload = trace.to_dict(include_documents=False)
    elif isinstance(trace, Mapping):
        payload = dict(trace)
    else:
        raise TypeError("trace must be a RetrievalTrace or mapping")
    gate_id = str(payload.get("gate_candidate_chunk_id") or "")
    final_id = str(payload.get("final_top1_chunk_id") or "")
    explicit = payload.get("gate_alignment")
    aligned = (
        True if explicit is True and gate_id and final_id and gate_id == final_id
        else False if explicit is False or gate_id or final_id
        else None
    )
    return gate_id, final_id, aligned


@dataclass(frozen=True)
class GateSignature:
    """Exact runtime signature to which one calibrated rule is bound."""

    embedding_model: str
    reranker_model: str
    chunker_version: str
    retrieval_profile: str
    index_fingerprint: str

    @classmethod
    def from_trace(cls, trace: Any) -> "GateSignature":
        if hasattr(trace, "to_dict"):
            payload = trace.to_dict(include_documents=False)
        elif isinstance(trace, Mapping):
            payload = dict(trace)
        else:
            raise TypeError("trace must be a RetrievalTrace or mapping")
        index = payload.get("index_metadata") or payload.get("index") or {}
        retrieval = payload.get("retrieval_profile") or payload.get("profile") or {}
        if isinstance(retrieval, str):
            retrieval = {"name": retrieval}
        return cls(
            embedding_model=str(index.get("embedding_model") or ""),
            reranker_model=str(
                retrieval.get("reranker_model") or index.get("rerank_model") or ""
            ),
            chunker_version=str(
                retrieval.get("chunker_version") or index.get("chunker_version") or ""
            ),
            retrieval_profile=str(retrieval.get("name") or ""),
            index_fingerprint=str(
                payload.get("index_fingerprint") or index.get("fingerprint_id") or ""
            ),
        )

    @property
    def complete(self) -> bool:
        return all(str(value).strip() for value in asdict(self).values())

    @property
    def digest(self) -> str:
        encoded = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GateRule:
    """One interpretable AND-of-reranker-plus-support Evidence Gate rule.

    ``support_at_least`` applies to ``support_signals``.  A signal only counts
    when its configured condition is met.  The reranker threshold is mandatory
    and is deliberately not counted as support; consequently a rule cannot
    degenerate into a bare "lower one threshold" experiment.
    """

    rerank_min: float
    support_signals: tuple[str, ...]
    support_at_least: int = 1
    dense_max: float | None = None
    bm25_rank_max: int | None = None
    margin_min: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.rerank_min)):
            raise ValueError("rerank_min must be finite")
        signals = tuple(dict.fromkeys(self.support_signals))
        if not signals or any(signal not in _SUPPORT_SIGNALS for signal in signals):
            raise ValueError("at least one known support signal is required")
        if not 1 <= int(self.support_at_least) <= len(signals):
            raise ValueError("support_at_least is outside support_signals")
        if "dense_distance" in signals and self.dense_max is None:
            raise ValueError("dense_distance support requires dense_max")
        if "bm25_rank" in signals and self.bm25_rank_max is None:
            raise ValueError("bm25_rank support requires bm25_rank_max")
        if "rerank_margin" in signals and self.margin_min is None:
            raise ValueError("rerank_margin support requires margin_min")
        object.__setattr__(self, "support_signals", signals)

    def support_evidence(self, features: GateFeatures) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        for signal in self.support_signals:
            if signal == "vector_in_kb":
                checks[signal] = bool(features.vector_in_kb)
            elif signal == "lexical_rescued":
                checks[signal] = bool(features.lexical_rescued)
            elif signal == "dense_distance":
                checks[signal] = (
                    features.best_dense_distance is not None
                    and features.best_dense_distance <= float(self.dense_max)
                )
            elif signal == "bm25_rank":
                checks[signal] = (
                    features.bm25_best_rank is not None
                    and features.bm25_best_rank <= int(self.bm25_rank_max)
                )
            elif signal == "rerank_margin":
                checks[signal] = (
                    features.rerank_margin is not None
                    and features.rerank_margin >= float(self.margin_min)
                )
        return checks

    def decide(self, features: GateFeatures) -> bool:
        if features.rerank_top is None or features.rerank_top < self.rerank_min:
            return False
        checks = self.support_evidence(features)
        return sum(checks.values()) >= self.support_at_least

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["support_signals"] = list(self.support_signals)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateRule":
        return cls(
            rerank_min=float(value["rerank_min"]),
            support_signals=tuple(value.get("support_signals") or ()),
            support_at_least=int(value.get("support_at_least", 1)),
            dense_max=_finite_float(value.get("dense_max")),
            bm25_rank_max=_positive_int(value.get("bm25_rank_max")),
            margin_min=_finite_float(value.get("margin_min")),
        )


@dataclass(frozen=True)
class GateProfile:
    profile_id: str
    signature: GateSignature
    rule: GateRule
    calibrated_at: str
    calibration_source: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GATE_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GATE_PROFILE_SCHEMA:
            raise ValueError(f"unsupported gate profile schema: {self.schema_version}")
        if not self.signature.complete:
            raise ValueError("a deployable gate profile requires a complete signature")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "signature": self.signature.to_dict(),
            "signature_digest": self.signature.digest,
            "rule": self.rule.to_dict(),
            "calibrated_at": self.calibrated_at,
            "calibration_source": self.calibration_source,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateProfile":
        signature = GateSignature(**dict(value["signature"]))
        declared_digest = str(value.get("signature_digest") or "")
        if declared_digest and declared_digest != signature.digest:
            raise ValueError("gate profile signature digest mismatch")
        return cls(
            schema_version=str(value.get("schema_version") or ""),
            profile_id=str(value["profile_id"]),
            signature=signature,
            rule=GateRule.from_dict(value["rule"]),
            calibrated_at=str(value.get("calibrated_at") or ""),
            calibration_source=str(value.get("calibration_source") or ""),
            metrics=dict(value.get("metrics") or {}),
        )


@dataclass(frozen=True)
class GateResolution:
    mode: str  # calibrated | baseline
    profile: GateProfile | None
    reason: str


def load_gate_registry(path: str | Path = DEFAULT_GATE_PROFILE_PATH) -> dict[str, Any]:
    profile_path = Path(path)
    if not profile_path.exists():
        return {"schema_version": GATE_REGISTRY_SCHEMA, "profiles": []}
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GATE_REGISTRY_SCHEMA:
        raise ValueError("unsupported gate registry schema")
    if not isinstance(payload.get("profiles"), list):
        raise ValueError("gate registry profiles must be a list")
    return payload


def resolve_gate_profile(
    signature: GateSignature,
    registry: Mapping[str, Any] | str | Path | None = None,
) -> GateResolution:
    """Resolve an exact profile; unknown or incomplete signatures use baseline."""

    if not signature.complete:
        return GateResolution("baseline", None, "incomplete_runtime_signature")
    try:
        payload = (
            load_gate_registry(registry or DEFAULT_GATE_PROFILE_PATH)
            if not isinstance(registry, Mapping) else dict(registry)
        )
        if payload.get("schema_version") != GATE_REGISTRY_SCHEMA:
            raise ValueError("unsupported gate registry schema")
        if not isinstance(payload.get("profiles"), list):
            raise ValueError("gate registry profiles must be a list")
    except Exception:
        return GateResolution("baseline", None, "malformed_gate_registry")
    matches: list[GateProfile] = []
    for raw in payload.get("profiles") or []:
        try:
            profile = GateProfile.from_dict(raw)
        except Exception:
            # A partially corrupt registry must never apply a calibrated rule,
            # even if another entry happens to match the runtime signature.
            return GateResolution("baseline", None, "malformed_gate_profile")
        if profile.signature == signature:
            matches.append(profile)
    if not matches:
        return GateResolution("baseline", None, "unknown_runtime_signature")
    if len(matches) > 1:
        return GateResolution("baseline", None, "duplicate_runtime_signature")
    return GateResolution("calibrated", matches[0], "exact_signature_match")


def decide_with_profile(
    features: GateFeatures,
    signature: GateSignature,
    baseline_decider: Callable[[GateFeatures], bool],
    registry: Mapping[str, Any] | str | Path | None = None,
) -> tuple[bool, GateResolution]:
    resolution = resolve_gate_profile(signature, registry)
    if resolution.profile is None:
        return bool(baseline_decider(features)), resolution
    return resolution.profile.rule.decide(features), resolution


@dataclass(frozen=True)
class CalibrationExample:
    example_id: str
    group: str
    features: GateFeatures
    top1_correct: bool = False
    gate_candidate_chunk_id: str = ""
    final_top1_chunk_id: str = ""
    gate_alignment: bool | None = None

    @property
    def alignment_valid(self) -> bool:
        return bool(
            self.gate_alignment is True
            and self.gate_candidate_chunk_id
            and self.final_top1_chunk_id
            and self.gate_candidate_chunk_id == self.final_top1_chunk_id
        )


@dataclass(frozen=True)
class CalibrationTargets:
    positive_pass: int = 12
    simple_negative_reject: int = 12
    adversarial_negative_reject: int = 11
    adversarial_negative_total: int = 12
    heldout_correct_top1_pass: int = 27
    heldout_size: int = 52


def _quantile_values(values: Iterable[float], limit: int = 7) -> list[float]:
    unique = sorted({round(float(value), 9) for value in values if math.isfinite(float(value))})
    if len(unique) <= limit:
        return unique
    positions = {round(index * (len(unique) - 1) / (limit - 1)) for index in range(limit)}
    return [unique[index] for index in sorted(positions)]


def _available_thresholds(examples: Sequence[CalibrationExample]) -> dict[str, list[Any]]:
    features = [example.features for example in examples]
    return {
        "rerank_top": _quantile_values(
            value.rerank_top for value in features if value.rerank_top is not None
        ),
        "dense_distance": _quantile_values(
            value.best_dense_distance for value in features
            if value.best_dense_distance is not None
        ),
        "bm25_rank": sorted({
            value.bm25_best_rank for value in features if value.bm25_best_rank is not None
        })[:7],
        "rerank_margin": _quantile_values(
            value.rerank_margin for value in features if value.rerank_margin is not None
        ),
        "vector_in_kb": [True] if any(value.vector_in_kb for value in features) else [],
        "lexical_rescued": [True] if any(value.lexical_rescued for value in features) else [],
    }


def _rule_candidates(examples: Sequence[CalibrationExample]) -> Iterable[GateRule]:
    thresholds = _available_thresholds(examples)
    rerank_values = thresholds["rerank_top"]
    if not rerank_values:
        return

    support_signals = [
        signal for signal in (
            "vector_in_kb", "dense_distance", "bm25_rank", "rerank_margin",
            "lexical_rescued",
        ) if thresholds[signal]
    ]
    seen: set[str] = set()
    # Single and pair support families keep the rule inspectable and bound the
    # scan size.  Even the smallest family requires a reranker threshold plus
    # one independently observed support condition.
    families: list[tuple[tuple[str, ...], int]] = []
    for signal in support_signals:
        families.append(((signal,), 1))
    for pair in itertools.combinations(support_signals, 2):
        # ``vector_in_kb`` is derived from dense/lexical evidence; do not count
        # either constituent twice as independent corroboration.
        if set(pair) in (
            {"vector_in_kb", "dense_distance"},
            {"vector_in_kb", "lexical_rescued"},
        ):
            continue
        families.extend(((pair, 1), (pair, 2)))

    for rerank_min in rerank_values:
        for signals, support_at_least in families:
            value_lists = [thresholds[signal] for signal in signals]
            for signal_values in itertools.product(*value_lists):
                kwargs: dict[str, Any] = {
                    "rerank_min": rerank_min,
                    "support_signals": signals,
                    "support_at_least": support_at_least,
                }
                for signal, threshold in zip(signals, signal_values):
                    if signal == "dense_distance":
                        kwargs["dense_max"] = threshold
                    elif signal == "bm25_rank":
                        kwargs["bm25_rank_max"] = threshold
                    elif signal == "rerank_margin":
                        kwargs["margin_min"] = threshold
                rule = GateRule(**kwargs)
                key = json.dumps(rule.to_dict(), sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    yield rule


def score_rule(
    rule: GateRule,
    examples: Sequence[CalibrationExample],
    targets: CalibrationTargets = CalibrationTargets(),
) -> dict[str, Any]:
    counters = {
        "positive_total": 0, "positive_pass": 0,
        "simple_negative_total": 0, "simple_negative_reject": 0,
        "adversarial_negative_total": 0, "adversarial_negative_reject": 0,
        "heldout_total": 0, "heldout_correct_top1": 0,
        "heldout_correct_top1_pass": 0,
    }
    for example in examples:
        decision = rule.decide(example.features)
        if example.group == "positive":
            counters["positive_total"] += 1
            counters["positive_pass"] += int(decision)
        elif example.group == "simple_negative":
            counters["simple_negative_total"] += 1
            counters["simple_negative_reject"] += int(not decision)
        elif example.group == "adversarial_negative":
            counters["adversarial_negative_total"] += 1
            counters["adversarial_negative_reject"] += int(not decision)
        elif example.group == "heldout":
            counters["heldout_total"] += 1
            if example.top1_correct:
                counters["heldout_correct_top1"] += 1
                counters["heldout_correct_top1_pass"] += int(decision)

    constraints = {
        "positive_12_of_12": (
            counters["positive_total"] >= targets.positive_pass
            and counters["positive_pass"] == counters["positive_total"]
        ),
        "simple_negative_12_of_12": (
            counters["simple_negative_total"] >= targets.simple_negative_reject
            and counters["simple_negative_reject"] == counters["simple_negative_total"]
        ),
        "adversarial_negative_at_least_11_of_12": (
            counters["adversarial_negative_total"] >= targets.adversarial_negative_total
            and counters["adversarial_negative_reject"]
            >= max(
                targets.adversarial_negative_reject,
                counters["adversarial_negative_total"] - 1,
            )
        ),
        "heldout_effective_at_least_27_of_52": (
            counters["heldout_total"] >= targets.heldout_size
            and counters["heldout_correct_top1_pass"]
            >= targets.heldout_correct_top1_pass
        ),
    }
    return {
        **counters,
        "constraints": constraints,
        "eligible": all(constraints.values()),
        "failed_constraints": [name for name, passed in constraints.items() if not passed],
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple:
    metrics = candidate["metrics"]
    rule = candidate["rule"]
    return (
        int(candidate["eligible"]),
        metrics["heldout_correct_top1_pass"],
        metrics["positive_pass"],
        metrics["adversarial_negative_reject"],
        metrics["simple_negative_reject"],
        -len(rule["support_signals"]),
        -int(rule["support_at_least"]),
    )


def calibrate_gate(
    examples: Sequence[CalibrationExample],
    signature: GateSignature,
    *,
    source: str = "",
    targets: CalibrationTargets = CalibrationTargets(),
) -> dict[str, Any]:
    """Fit only aligned examples and emit a non-deployable candidate report."""

    all_examples = list(examples)
    aligned_examples = [example for example in all_examples if example.alignment_valid]
    misaligned = [example for example in all_examples if not example.alignment_valid]
    candidates: list[dict[str, Any]] = []
    for index, rule in enumerate(_rule_candidates(aligned_examples), start=1):
        metrics = score_rule(rule, aligned_examples, targets)
        candidates.append({
            "candidate_id": f"gate-rule-{index:05d}",
            "rule": rule.to_dict(),
            "metrics": metrics,
            "eligible": bool(metrics["eligible"]),
        })
    candidates.sort(key=_candidate_sort_key, reverse=True)
    selected = next((candidate for candidate in candidates if candidate["eligible"]), None)
    release_eligible = bool(selected and signature.complete and not misaligned)
    reason = (
        "misaligned_gate_candidates" if misaligned else
        "calibration_candidate_meets_constraints" if release_eligible else
        "incomplete_runtime_signature" if selected else
        "no_rule_meets_all_constraints"
    )
    profile = None
    if release_eligible and selected is not None:
        profile_id = f"gate-{signature.digest}-{selected['candidate_id'].split('-')[-1]}"
        profile = GateProfile(
            profile_id=profile_id,
            signature=signature,
            rule=GateRule.from_dict(selected["rule"]),
            calibrated_at=datetime.now().isoformat(timespec="seconds"),
            calibration_source=source,
            metrics=selected["metrics"],
        ).to_dict()
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "signature": signature.to_dict(),
        "signature_digest": signature.digest,
        "signature_complete": signature.complete,
        "status": "calibration_candidate",
        "targets": asdict(targets),
        "example_counts": {
            group: sum(example.group == group for example in all_examples)
            for group in ("heldout", "positive", "simple_negative", "adversarial_negative")
        },
        "fitting_example_counts": {
            group: sum(example.group == group for example in aligned_examples)
            for group in ("heldout", "positive", "simple_negative", "adversarial_negative")
        },
        "misaligned_count": len(misaligned),
        "misaligned_ids": [example.example_id for example in misaligned],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_profile": profile,
        # Calibration alone is never a deployment authorization.  A separate
        # blind/shadow validation artifact must promote this candidate later.
        "release_eligible": release_eligible,
        "deployable": False,
        "reason": reason,
    }


def examples_from_ab_payload(
    payload: Mapping[str, Any], *, arm: str | None = None, run_index: int = 0,
    heldout_set: str = "heldout52",
) -> tuple[list[CalibrationExample], GateSignature]:
    """Read production A/B JSON or a single-arm result.

    Full ``gate_features``/trace fields are preferred.  Legacy rows containing
    only reranker score and margin remain readable, but naturally produce a
    smaller, explicitly visible rule search space.
    """

    selected: Mapping[str, Any] = payload
    if isinstance(payload.get("arms"), Mapping):
        arms = payload["arms"]
        if not arm:
            non_baseline = [name for name in arms if name != "A0"]
            if len(non_baseline) != 1:
                raise ValueError("--arm is required when an A/B file has multiple candidates")
            arm = non_baseline[0]
        if arm not in arms:
            raise ValueError(f"arm {arm!r} is not present in A/B payload")
        selected = arms[arm]

    signature = GateSignature.from_trace({
        "index": selected.get("index") or {},
        "profile": selected.get("profile") or {},
        "index_fingerprint": (selected.get("index") or {}).get("fingerprint_id", ""),
    })
    examples: list[CalibrationExample] = []
    for evaluated_set in selected.get("sets") or []:
        if str(evaluated_set.get("set") or "") != heldout_set:
            continue
        runs = evaluated_set.get("runs") or []
        if not runs or run_index >= len(runs):
            continue
        for row in runs[run_index].get("rows") or []:
            raw_features = row.get("trace") or row
            gate_id, final_id, alignment = gate_alignment_from_trace(raw_features)
            examples.append(CalibrationExample(
                example_id=str(row.get("id") or ""),
                group="heldout",
                features=(gate_features_from_trace(raw_features)
                          if isinstance(raw_features, Mapping)
                          and raw_features.get("final_candidates") is not None
                          else GateFeatures.from_mapping(raw_features)),
                top1_correct=int(row.get("rank") or 0) == 1,
                gate_candidate_chunk_id=gate_id,
                final_top1_chunk_id=final_id,
                gate_alignment=alignment,
            ))
    for group_name, group_payload in (selected.get("gates") or {}).items():
        normalized = {
            "simple_negative": "simple_negative",
            "adversarial_negative": "adversarial_negative",
            "positive": "positive",
        }.get(group_name)
        if not normalized:
            continue
        for row in group_payload.get("rows") or []:
            raw_features = row.get("trace") or row
            gate_id, final_id, alignment = gate_alignment_from_trace(raw_features)
            examples.append(CalibrationExample(
                example_id=str(row.get("id") or ""),
                group=normalized,
                features=(gate_features_from_trace(raw_features)
                          if isinstance(raw_features, Mapping)
                          and raw_features.get("final_candidates") is not None
                          else GateFeatures.from_mapping(raw_features)),
                gate_candidate_chunk_id=gate_id,
                final_top1_chunk_id=final_id,
                gate_alignment=alignment,
            ))
    return examples, signature


def collect_fixed_query_groups(
    retrieve: Callable[..., Any],
    profile: Any,
    heldout_items: Sequence[Mapping[str, Any]],
    simple_negatives: Sequence[str],
    adversarial_negatives: Sequence[Mapping[str, Any] | str],
    positives: Sequence[str],
    *,
    query_plan: Mapping[str, Any] | None = None,
) -> tuple[list[CalibrationExample], GateSignature]:
    """Run fixed groups through the caller's production retrieval entrypoint."""

    plan = query_plan or {
        "decision": "answer",
        "routes": [{"source": "reference_kb", "operation": "search"}],
    }
    examples: list[CalibrationExample] = []
    first_trace = None

    def run(question: str) -> Any:
        nonlocal first_trace
        trace = retrieve(question, plan, profile, top_k=5)
        first_trace = first_trace or trace
        return trace

    for item in heldout_items:
        trace = run(str(item["q"]))
        sources = [
            str(candidate.source if hasattr(candidate, "source") else candidate.get("source") or "")
            for candidate in trace.final_candidates
        ]
        expected = [str(value).lower() for value in item.get("expect_sources") or []]
        top1_correct = bool(sources) and any(value in sources[0].lower() for value in expected)
        gate_id, final_id, alignment = gate_alignment_from_trace(trace)
        examples.append(CalibrationExample(
            example_id=str(item.get("id") or ""), group="heldout",
            features=gate_features_from_trace(trace), top1_correct=top1_correct,
            gate_candidate_chunk_id=gate_id,
            final_top1_chunk_id=final_id,
            gate_alignment=alignment,
        ))
    groups = (
        ("simple_negative", simple_negatives),
        ("adversarial_negative", adversarial_negatives),
        ("positive", positives),
    )
    for group, rows in groups:
        for index, row in enumerate(rows, start=1):
            if isinstance(row, Mapping):
                question = str(row.get("q") or "")
                example_id = str(row.get("id") or f"{group}-{index}")
            else:
                question = str(row)
                example_id = f"{group}-{index}"
            trace = run(question)
            gate_id, final_id, alignment = gate_alignment_from_trace(trace)
            examples.append(CalibrationExample(
                example_id=example_id, group=group,
                features=gate_features_from_trace(trace),
                gate_candidate_chunk_id=gate_id,
                final_top1_chunk_id=final_id,
                gate_alignment=alignment,
            ))
    if first_trace is None:
        raise ValueError("fixed query groups are empty")
    return examples, GateSignature.from_trace(first_trace)
