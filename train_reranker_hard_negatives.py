#!/usr/bin/env python3
"""Opt-in development training for an audited hard-negative artifact.

No model is loaded unless ``--run-training`` is provided.  The script uses the
already-installed CrossEncoder tokenizer/model with a small direct torch loop,
and writes an explicitly development-only model card beside the output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_SCHEMA = "reranker-hard-negatives-v1"
EXPECTED_BASE_SET = "tests/rag_bench_paraphrase_set.json"

# Ranking losses consume the audited triples directly.  Stability anchors are
# deliberately weighted more heavily because the release contract allows at
# most one regression while asking the candidate to fix several false
# winners.  These are corpus-level kind weights, never per-query exceptions.
DEFAULT_PAIRWISE_KIND_WEIGHTS = {
    "current_false_winner": 1.0,
    "same_document_wrong_section": 1.0,
    "stability_anchor": 2.0,
}


class TrainingInputError(ValueError):
    """Raised when a training artifact violates the safety contract."""


def load_training_artifact(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != EXPECTED_SCHEMA:
        raise TrainingInputError("unsupported hard-negative schema")
    if artifact.get("development_only") is not True:
        raise TrainingInputError("training input must be development_only")
    if artifact.get("contains_text") is not True:
        raise TrainingInputError(
            "training requires an artifact built with explicit --include-text"
        )
    if artifact.get("private_blind_set") is not False:
        raise TrainingInputError("private blind-set training is forbidden")
    if artifact.get("base_set") != EXPECTED_BASE_SET:
        raise TrainingInputError("only the immutable public development set is allowed")
    serialized_provenance = json.dumps(
        artifact.get("provenance") or {}, ensure_ascii=False,
    ).lower()
    if "private_eval" in serialized_provenance or "blind_v" in serialized_provenance:
        raise TrainingInputError("private blind-set provenance is forbidden")
    triples = artifact.get("triples")
    if not isinstance(triples, list) or not triples:
        raise TrainingInputError("training artifact contains no triples")
    for row in triples:
        required = (
            row.get("query_id"), row.get("query"),
            (row.get("positive") or {}).get("chunk_id"),
            (row.get("positive") or {}).get("scoring_text"),
            (row.get("negative") or {}).get("chunk_id"),
            (row.get("negative") or {}).get("scoring_text"),
        )
        if not all(required):
            raise TrainingInputError("a triple is missing query/chunk scoring text")
    return artifact


def deterministic_group_split(
    triples: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 20260824,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split only at query boundaries, independent of input row ordering."""

    if not 0.0 < validation_fraction < 1.0:
        raise TrainingInputError("validation_fraction must be between 0 and 1")
    query_ids = sorted({str(row["query_id"]) for row in triples})
    if len(query_ids) < 2:
        raise TrainingInputError("at least two query groups are required")
    ordered = sorted(
        query_ids,
        key=lambda query_id: hashlib.sha256(
            f"{seed}:{query_id}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * validation_fraction))),
    )
    kinds_by_query: dict[str, set[str]] = {query_id: set() for query_id in query_ids}
    for row in triples:
        kinds_by_query[str(row["query_id"])].add(
            str(row.get("negative_kind") or "unspecified")
        )
    kind_frequency: dict[str, int] = {}
    for kinds in kinds_by_query.values():
        for kind in kinds:
            kind_frequency[kind] = kind_frequency.get(kind, 0) + 1

    # Reserve coverage for rare triple kinds before filling by seeded hash.
    # Query groups remain indivisible; a group containing both false-winner
    # and same-document triples can satisfy both strata without leakage.
    selected: list[str] = []
    for kind in sorted(kind_frequency, key=lambda value: (kind_frequency[value], value)):
        if len(selected) >= validation_count:
            break
        if any(kind in kinds_by_query[query_id] for query_id in selected):
            continue
        candidate = next(
            (query_id for query_id in ordered
             if query_id not in selected and kind in kinds_by_query[query_id]),
            None,
        )
        if candidate:
            selected.append(candidate)
    selected.extend(
        query_id for query_id in ordered
        if query_id not in selected
    )
    validation_ids = set(selected[:validation_count])
    train = [row for row in triples if row["query_id"] not in validation_ids]
    validation = [row for row in triples if row["query_id"] in validation_ids]
    return train, validation, {
        "method": "grouped_kind_stratified_then_sha256(seed:query_id)",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_query_ids": sorted({row["query_id"] for row in train}),
        "validation_query_ids": sorted(validation_ids),
    }


def labeled_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deduplicated positive/negative pairs for the manual trainer."""

    pairs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        query = str(row["query"])
        for side, label in (("positive", 1), ("negative", 0)):
            chunk = row[side]
            key = (query_id, str(chunk["chunk_id"]), label)
            pairs[key] = {
                "query_id": query_id,
                "query": query,
                "chunk_id": str(chunk["chunk_id"]),
                "scoring_text": str(chunk["scoring_text"]),
                "label": label,
            }
    return [pairs[key] for key in sorted(pairs)]


def triple_kind_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Keep corrective and stability examples visibly separate in reports."""

    counts = {
        "current_false_winner": 0,
        "same_document_wrong_section": 0,
        "stability_anchor": 0,
    }
    for row in rows:
        kind = str(row.get("negative_kind") or "")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def formal_model_roots(extra_roots: list[str | Path] | None = None) -> list[Path]:
    roots = [ROOT / "models", ROOT / "production_models"]
    configured = os.environ.get("RAG_FORMAL_MODEL_ROOT", "").strip()
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(Path(value).expanduser() for value in (extra_roots or []))
    return [path.resolve() for path in roots]


def assert_output_path_allowed(
    output_path: str | Path,
    *,
    allow_formal_model_path: bool = False,
    extra_formal_roots: list[str | Path] | None = None,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    in_formal_root = any(
        output == root or output.is_relative_to(root)
        for root in formal_model_roots(extra_formal_roots)
    )
    if in_formal_root and not allow_formal_model_path:
        raise TrainingInputError(
            "refusing to write a development model inside a formal model path; "
            "use --allow-formal-model-path only for an explicitly reviewed experiment"
        )
    if output.exists():
        if not output.is_dir():
            raise TrainingInputError("output path exists and is not a directory")
        if any(output.iterdir()):
            raise TrainingInputError("output directory already exists and is not empty")
    return output


def _model_card(
    *,
    artifact: dict[str, Any],
    source_artifact: Path,
    base_model: str,
    split: dict[str, Any],
    train_pair_count: int,
    validation_pair_count: int,
    train_triple_count: int,
    validation_triple_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_triples_by_kind: dict[str, int],
    validation_triples_by_kind: dict[str, int],
    training_result: dict[str, Any],
) -> str:
    provenance = artifact.get("provenance") or {}
    index = provenance.get("index") or {}
    return f"""# OfferClaw development reranker

> development_only=true — This artifact is not approved for production.

## Provenance

- Source artifact: `{source_artifact.name}`
- Source artifact SHA-256: `sha256:{hashlib.sha256(source_artifact.read_bytes()).hexdigest()}`
- Qrels SHA-256: `{provenance.get('qrels_sha256', '')}`
- Qrels reviewer: `{provenance.get('qrels_reviewer_id', '')}`
- Index collection: `{index.get('collection', '')}`
- Index count: `{index.get('count', '')}`
- Index fingerprint: `{index.get('fingerprint', '')}`
- Base set: `{artifact.get('base_set', '')}`
- Base model: `{base_model}`

## Training

- Training query groups: {len(split['train_query_ids'])}
- Validation query groups: {len(split['validation_query_ids'])}
- Training labeled pairs: {train_pair_count}
- Validation labeled pairs: {validation_pair_count}
- Training ranking triples: {train_triple_count}
- Validation ranking triples: {validation_triple_count}
- Training triples by kind: `{json.dumps(train_triples_by_kind, sort_keys=True)}`
- Validation triples by kind: `{json.dumps(validation_triples_by_kind, sort_keys=True)}`
- Epochs: {epochs}
- Batch size: {batch_size}
- Learning rate: {learning_rate}
- Group split: `{split['method']}`, seed `{split['seed']}`
- Text format: production `title + breadcrumb + 正文`
- Device: `{training_result.get('device', '')}`
- Objective: `{training_result.get('objective', training_result.get('loss', ''))}`
- Pairwise margin: `{training_result.get('margin', '')}`
- Pairwise kind weights: `{json.dumps(training_result.get('kind_weights') or {}, sort_keys=True)}`
- Optimizer steps: {training_result.get('optimizer_steps', 0)}
- Final train objective loss: {training_result.get('final_train_loss', '')}
- Validation objective loss: {training_result.get('validation_loss', '')}
- Pre-train validation ranking: `{json.dumps(training_result.get('pre_validation_metrics') or {}, sort_keys=True)}`
- Post-train validation ranking: `{json.dumps(training_result.get('post_validation_metrics') or {}, sort_keys=True)}`

The validation split is development-only.  The repository-external blind set
was neither read nor used for training.  Promotion requires the independent
retrieval, latency, Gate, and blind-set release gates.
"""


def _training_device(torch_module) -> str:
    configured = os.environ.get("OFFERCLAW_TORCH_DEVICE", "").strip()
    if configured:
        return configured
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _encode_batch(tokenizer, rows, *, max_length: int, device: str):
    encoded = tokenizer(
        [row["query"] for row in rows],
        [row["scoring_text"] for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _limit_trainable_encoder_layers(classifier, count: int | None) -> dict[str, int]:
    """Optionally train only the top encoder layers and task head.

    The audited development set is intentionally tiny.  Training all 12
    XLM-R layers both overfits that set and makes AdamW exceed the 8 GB Mac
    unified-memory budget.  This is a small-data regularizer as well as a
    resource bound; callers may still pass ``None`` for an explicit full
    fine-tune experiment.
    """

    parameters = list(classifier.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    if count is None:
        return {"trainable_encoder_layers": -1, "trainable_parameters": total,
                "total_parameters": total}
    if count < 0:
        raise TrainingInputError("trainable_encoder_layers cannot be negative")

    for parameter in parameters:
        parameter.requires_grad = False
    backbone = getattr(classifier, getattr(classifier, "base_model_prefix", ""), None)
    encoder = getattr(backbone, "encoder", None)
    layers = list(getattr(encoder, "layer", []) or [])
    if count and not layers:
        raise TrainingInputError("model does not expose encoder.layer for bounded training")
    for layer in layers[-count:] if count else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    head = getattr(classifier, "classifier", None)
    if head is None:
        raise TrainingInputError("model does not expose a classification head")
    for parameter in head.parameters():
        parameter.requires_grad = True
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return {
        "trainable_encoder_layers": min(count, len(layers)),
        "trainable_parameters": trainable,
        "total_parameters": total,
    }


def train_binary_cross_encoder(
    *,
    train_pairs: list[dict[str, Any]],
    validation_pairs: list[dict[str, Any]],
    output: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    trainable_encoder_layers: int | None = None,
    cross_encoder_factory=None,
) -> dict[str, Any]:
    """Minimal deterministic torch loop, intentionally independent of datasets.

    sentence-transformers 5.5 keeps ``CrossEncoder.fit`` for compatibility but
    delegates it to the trainer stack, which imports the optional ``datasets``
    package.  OfferClaw does not add that dependency for this tiny audited set;
    it trains the underlying transformers sequence classifier directly.
    """

    import torch
    is_default_factory = cross_encoder_factory is None
    if is_default_factory:
        from sentence_transformers import CrossEncoder
        cross_encoder_factory = CrossEncoder

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = _training_device(torch)
    factory_kwargs: dict[str, Any] = {"num_labels": 1, "device": device}
    # Torch SDPA training with dropout is not implemented on MPS.  Eager
    # attention is semantically equivalent here and keeps the safe MPS path
    # available without disabling memory limits or dropout.
    if is_default_factory and device == "mps":
        factory_kwargs["model_kwargs"] = {"attn_implementation": "eager"}
    encoder = cross_encoder_factory(base_model, **factory_kwargs)
    classifier = encoder.model
    tokenizer = encoder.tokenizer
    max_length = int(getattr(encoder, "max_seq_length", 512) or 512)
    trainable_stats = _limit_trainable_encoder_layers(
        classifier, trainable_encoder_layers,
    )
    classifier.to(device)
    classifier.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed)
    optimizer_steps = 0
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        order = torch.randperm(len(train_pairs), generator=order_generator).tolist()
        losses: list[float] = []
        for offset in range(0, len(order), batch_size):
            rows = [train_pairs[index] for index in order[offset:offset + batch_size]]
            encoded = _encode_batch(
                tokenizer, rows, max_length=max_length, device=device,
            )
            labels = torch.tensor(
                [float(row["label"]) for row in rows],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(**encoded).logits.reshape(-1)
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(losses) / len(losses))

    classifier.eval()
    validation_losses: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(validation_pairs), batch_size):
            rows = validation_pairs[offset:offset + batch_size]
            encoded = _encode_batch(
                tokenizer, rows, max_length=max_length, device=device,
            )
            labels = torch.tensor(
                [float(row["label"]) for row in rows],
                dtype=torch.float32,
                device=device,
            )
            logits = classifier(**encoded).logits.reshape(-1)
            validation_losses.append(float(
                loss_function(logits, labels).detach().cpu()
            ))

    output.mkdir(parents=True, exist_ok=True)
    classifier.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    return {
        "engine": "torch_transformers_binary_loop_v1",
        "device": device,
        "max_length": max_length,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "optimizer_steps": optimizer_steps,
        "epoch_train_losses": [round(value, 8) for value in epoch_losses],
        "final_train_loss": round(epoch_losses[-1], 8),
        "validation_loss": round(
            sum(validation_losses) / len(validation_losses), 8
        ),
        **trainable_stats,
    }


def _pairwise_kind_weight(
    row: dict[str, Any],
    kind_weights: dict[str, float],
) -> float:
    """Corpus-level kind weight, optionally scaled by a per-triple weight.

    ``pair_weight`` exists so a merged corpus can normalize by anchor: an
    anchor that happens to carry four phrasings and two negatives must not
    outvote one that carries four phrasings and one.  It multiplies the kind
    weight rather than replacing it, so an artifact without the field trains
    byte-identically to before.
    """

    kind = str(row.get("negative_kind") or "unspecified")
    value = float(kind_weights.get(kind, 1.0))
    if value <= 0:
        raise TrainingInputError(f"pairwise weight for {kind!r} must be positive")
    scale = row.get("pair_weight")
    if scale is None:
        return value
    scale = float(scale)
    if scale <= 0:
        raise TrainingInputError(
            f"pair_weight for {row.get('query_id')!r} must be positive"
        )
    return value * scale


def _pairwise_side_rows(
    rows: list[dict[str, Any]],
    side: str,
) -> list[dict[str, str]]:
    return [
        {
            "query": str(row["query"]),
            "scoring_text": str(row[side]["scoring_text"]),
        }
        for row in rows
    ]


def _pairwise_metrics(
    *,
    classifier,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    device: str,
    margin: float,
    kind_weights: dict[str, float],
) -> dict[str, Any]:
    """Measure the actual within-query ordering objective, including by kind."""

    import torch

    if not rows:
        raise TrainingInputError("pairwise metrics require at least one triple")
    differences: list[tuple[str, float, float]] = []
    classifier.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            positive = _encode_batch(
                tokenizer,
                _pairwise_side_rows(batch, "positive"),
                max_length=max_length,
                device=device,
            )
            negative = _encode_batch(
                tokenizer,
                _pairwise_side_rows(batch, "negative"),
                max_length=max_length,
                device=device,
            )
            positive_logits = classifier(**positive).logits.reshape(-1)
            negative_logits = classifier(**negative).logits.reshape(-1)
            for row, difference in zip(
                batch, (positive_logits - negative_logits).detach().cpu().tolist(),
            ):
                differences.append((
                    str(row.get("negative_kind") or "unspecified"),
                    float(difference),
                    _pairwise_kind_weight(row, kind_weights),
                ))

    def summarize(values: list[tuple[str, float, float]]) -> dict[str, Any]:
        total_weight = sum(weight for _kind, _difference, weight in values)
        losses = [
            float(torch.nn.functional.softplus(
                torch.tensor(margin - difference, dtype=torch.float32)
            ))
            for _kind, difference, _weight in values
        ]
        return {
            "pair_count": len(values),
            "pair_accuracy": round(
                sum(1 for _kind, difference, _weight in values if difference > 0)
                / len(values), 8,
            ),
            "target_margin_accuracy": round(
                sum(1 for _kind, difference, _weight in values if difference >= margin)
                / len(values), 8,
            ),
            "mean_score_margin": round(
                sum(difference for _kind, difference, _weight in values) / len(values),
                8,
            ),
            "weighted_pairwise_loss": round(
                sum(loss * values[index][2] for index, loss in enumerate(losses))
                / total_weight,
                8,
            ),
        }

    by_kind: dict[str, Any] = {}
    for kind in sorted({kind for kind, _difference, _weight in differences}):
        by_kind[kind] = summarize([
            value for value in differences if value[0] == kind
        ])
    return {
        **summarize(differences),
        "by_kind": by_kind,
        # Per-pair margins in ``rows`` order so a caller can diff pre against
        # post.  Aggregates cannot express a flip: a run that flips 40 pairs
        # each way and one that flips none report the same accuracy.
        "score_margins": [round(difference, 8)
                          for _kind, difference, _weight in differences],
        "kinds": [kind for kind, _difference, _weight in differences],
    }


def pair_flip_report(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    """Count how many pairs actually changed sign between two measurements.

    This is the only training-time number tied to end-to-end R@1.  Retrieval
    reads ``argmax``, which is invariant to any monotone rescaling of the
    scores -- and a margin objective whose target is already met by most pairs
    can reduce loss purely by rescaling.  The F3 run (2026-08-27) did exactly
    that: loss down, mean margin up 8%, and **one** pair out of 1084 flipped,
    which showed up downstream as 0 wins / 0 losses on both Dev sets.  Reading
    loss and mean margin alone, that run looks like a success.
    """
    before, after = pre["score_margins"], post["score_margins"]
    if len(before) != len(after) or pre.get("kinds") != post.get("kinds"):
        raise TrainingInputError(
            "pair flip report needs the same rows, in the same order, "
            "on both sides"
        )
    kinds = pre.get("kinds") or ["unspecified"] * len(before)
    counters: dict[str, dict[str, int]] = {}
    for kind, was, now in zip(kinds, before, after):
        for bucket in ("__all__", kind):
            slot = counters.setdefault(
                bucket, {"pairs": 0, "flipped_to_correct": 0, "flipped_to_wrong": 0},
            )
            slot["pairs"] += 1
            if was <= 0 < now:
                slot["flipped_to_correct"] += 1
            elif now <= 0 < was:
                slot["flipped_to_wrong"] += 1
    report: dict[str, Any] = {}
    for bucket, slot in counters.items():
        key = "overall" if bucket == "__all__" else bucket
        report[key] = {
            **slot,
            "net_flips": slot["flipped_to_correct"] - slot["flipped_to_wrong"],
        }
    return report


def train_pairwise_cross_encoder(
    *,
    train_triples: list[dict[str, Any]],
    validation_triples: list[dict[str, Any]],
    output: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    margin: float = 0.5,
    kind_weights: dict[str, float] | None = None,
    trainable_encoder_layers: int | None = 0,
    cross_encoder_factory=None,
) -> dict[str, Any]:
    """Train directly on ``score(query, positive) > score(query, negative)``.

    Pointwise BCE teaches absolute labels independently across queries and can
    move an already-correct document below a competitor without being
    penalized for that ordering.  This loop keeps every audited triple paired,
    optimizes a smooth margin-ranking loss, and reports pre/post ranking
    metrics so a no-op or destructive run is visible before retrieval A/B.
    """

    import torch

    if margin < 0:
        raise TrainingInputError("pairwise margin cannot be negative")
    if not train_triples or not validation_triples:
        raise TrainingInputError("pairwise training requires train and validation triples")
    weights = {
        **DEFAULT_PAIRWISE_KIND_WEIGHTS,
        **(kind_weights or {}),
    }
    for row in [*train_triples, *validation_triples]:
        _pairwise_kind_weight(row, weights)

    is_default_factory = cross_encoder_factory is None
    if is_default_factory:
        from sentence_transformers import CrossEncoder
        cross_encoder_factory = CrossEncoder

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = _training_device(torch)
    factory_kwargs: dict[str, Any] = {"num_labels": 1, "device": device}
    if is_default_factory and device == "mps":
        factory_kwargs["model_kwargs"] = {"attn_implementation": "eager"}
    encoder = cross_encoder_factory(base_model, **factory_kwargs)
    classifier = encoder.model
    tokenizer = encoder.tokenizer
    max_length = int(getattr(encoder, "max_seq_length", 512) or 512)
    trainable_stats = _limit_trainable_encoder_layers(
        classifier, trainable_encoder_layers,
    )
    classifier.to(device)

    pre_train = _pairwise_metrics(
        classifier=classifier,
        tokenizer=tokenizer,
        rows=train_triples,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        margin=margin,
        kind_weights=weights,
    )
    pre_validation = _pairwise_metrics(
        classifier=classifier,
        tokenizer=tokenizer,
        rows=validation_triples,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        margin=margin,
        kind_weights=weights,
    )

    trainable_parameters = [
        parameter for parameter in classifier.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed)
    optimizer_steps = 0
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        classifier.train()
        order = torch.randperm(
            len(train_triples), generator=order_generator,
        ).tolist()
        losses: list[float] = []
        for offset in range(0, len(order), batch_size):
            batch = [
                train_triples[index]
                for index in order[offset:offset + batch_size]
            ]
            positive = _encode_batch(
                tokenizer,
                _pairwise_side_rows(batch, "positive"),
                max_length=max_length,
                device=device,
            )
            negative = _encode_batch(
                tokenizer,
                _pairwise_side_rows(batch, "negative"),
                max_length=max_length,
                device=device,
            )
            sample_weights = torch.tensor(
                [_pairwise_kind_weight(row, weights) for row in batch],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            positive_logits = classifier(**positive).logits.reshape(-1)
            negative_logits = classifier(**negative).logits.reshape(-1)
            per_sample_loss = torch.nn.functional.softplus(
                margin - (positive_logits - negative_logits)
            )
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(losses) / len(losses))

    post_train = _pairwise_metrics(
        classifier=classifier,
        tokenizer=tokenizer,
        rows=train_triples,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        margin=margin,
        kind_weights=weights,
    )
    post_validation = _pairwise_metrics(
        classifier=classifier,
        tokenizer=tokenizer,
        rows=validation_triples,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        margin=margin,
        kind_weights=weights,
    )

    train_flips = pair_flip_report(pre_train, post_train)
    validation_flips = pair_flip_report(pre_validation, post_validation)
    print(
        "[pairwise] net pair flips  train="
        f"{train_flips['overall']['net_flips']:+d}"
        f" (+{train_flips['overall']['flipped_to_correct']}"
        f"/-{train_flips['overall']['flipped_to_wrong']}"
        f" of {train_flips['overall']['pairs']})"
        f"  validation={validation_flips['overall']['net_flips']:+d}"
        f" (+{validation_flips['overall']['flipped_to_correct']}"
        f"/-{validation_flips['overall']['flipped_to_wrong']}"
        f" of {validation_flips['overall']['pairs']})",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    classifier.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    return {
        "engine": "torch_transformers_pairwise_loop_v1",
        "device": device,
        "max_length": max_length,
        "optimizer": "AdamW",
        "objective": "weighted_pairwise_logistic_margin",
        "margin": margin,
        "kind_weights": dict(sorted(weights.items())),
        "optimizer_steps": optimizer_steps,
        "epoch_train_losses": [round(value, 8) for value in epoch_losses],
        "final_train_loss": round(epoch_losses[-1], 8),
        "validation_loss": post_validation["weighted_pairwise_loss"],
        "pre_train_metrics": pre_train,
        "post_train_metrics": post_train,
        "pre_validation_metrics": pre_validation,
        "post_validation_metrics": post_validation,
        "train_pair_flips": train_flips,
        "validation_pair_flips": validation_flips,
        **trainable_stats,
    }


def run_training(
    *,
    artifact_path: str | Path,
    output_path: str | Path,
    base_model: str = "BAAI/bge-reranker-base",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    validation_fraction: float = 0.2,
    seed: int = 20260824,
    objective: str = "pairwise_logistic",
    pairwise_margin: float = 0.5,
    stability_weight: float = 2.0,
    trainable_encoder_layers: int | None = 0,
    allow_formal_model_path: bool = False,
    extra_formal_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise TrainingInputError("epochs, batch_size, and learning_rate must be positive")
    if pairwise_margin < 0:
        raise TrainingInputError("pairwise_margin cannot be negative")
    if stability_weight <= 0:
        raise TrainingInputError("stability_weight must be positive")
    artifact_file = Path(artifact_path)
    artifact = load_training_artifact(artifact_file)
    output = assert_output_path_allowed(
        output_path,
        allow_formal_model_path=allow_formal_model_path,
        extra_formal_roots=extra_formal_roots,
    )
    train_rows, validation_rows, split = deterministic_group_split(
        artifact["triples"],
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_pairs = labeled_pairs(train_rows)
    validation_pairs = labeled_pairs(validation_rows)
    if not train_pairs or not validation_pairs:
        raise TrainingInputError("group split produced an empty pair set")

    if objective == "pairwise_logistic":
        training_result = train_pairwise_cross_encoder(
            train_triples=train_rows,
            validation_triples=validation_rows,
            output=output,
            base_model=base_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            margin=pairwise_margin,
            kind_weights={"stability_anchor": stability_weight},
            trainable_encoder_layers=trainable_encoder_layers,
        )
    elif objective == "pointwise_bce":
        training_result = train_binary_cross_encoder(
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            output=output,
            base_model=base_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            trainable_encoder_layers=trainable_encoder_layers,
        )
    else:
        raise TrainingInputError(f"unsupported training objective: {objective}")

    card = _model_card(
        artifact=artifact,
        source_artifact=artifact_file,
        base_model=base_model,
        split=split,
        train_pair_count=len(train_pairs),
        validation_pair_count=len(validation_pairs),
        train_triple_count=len(train_rows),
        validation_triple_count=len(validation_rows),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_triples_by_kind=triple_kind_counts(train_rows),
        validation_triples_by_kind=triple_kind_counts(validation_rows),
        training_result=training_result,
    )
    (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    manifest = {
        "schema_version": "reranker-development-model-v1",
        "development_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "source_artifact": artifact_file.name,
        "source_artifact_sha256": "sha256:" + hashlib.sha256(
            artifact_file.read_bytes()
        ).hexdigest(),
        "qrels_sha256": (artifact.get("provenance") or {}).get("qrels_sha256", ""),
        "index": (artifact.get("provenance") or {}).get("index", {}),
        "split": split,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "train_triple_count": len(train_rows),
        "validation_triple_count": len(validation_rows),
        "train_triples_by_kind": triple_kind_counts(train_rows),
        "validation_triples_by_kind": triple_kind_counts(validation_rows),
        "private_blind_set_used": False,
        "training": training_result,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "DEVELOPMENT_ONLY").write_text(
        "Not approved for production.\n", encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--objective",
        choices=("pairwise_logistic", "pointwise_bce"),
        default="pairwise_logistic",
        help="H2 defaults to paired ranking; pointwise BCE is retained for audit only",
    )
    parser.add_argument(
        "--pairwise-margin", type=float, default=0.5,
        help="minimum desired raw-logit positive-minus-negative margin",
    )
    parser.add_argument(
        "--stability-weight", type=float, default=2.0,
        help="corpus-level weight for already-correct stability anchors",
    )
    parser.add_argument(
        "--trainable-encoder-layers", type=int, default=0,
        help="train top N encoder layers plus classifier; H2 defaults to head-only",
    )
    parser.add_argument("--formal-model-root", action="append", default=[])
    parser.add_argument("--allow-formal-model-path", action="store_true")
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="required opt-in; without it the script only validates and shows the split",
    )
    args = parser.parse_args()

    artifact = load_training_artifact(args.artifact)
    train, validation, split = deterministic_group_split(
        artifact["triples"],
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    assert_output_path_allowed(
        args.output,
        allow_formal_model_path=args.allow_formal_model_path,
        extra_formal_roots=args.formal_model_root,
    )
    preview = {
        "development_only": True,
        "objective": args.objective,
        "pairwise_margin": args.pairwise_margin,
        "stability_weight": args.stability_weight,
        "trainable_encoder_layers": args.trainable_encoder_layers,
        "train_query_count": len(split["train_query_ids"]),
        "validation_query_count": len(split["validation_query_ids"]),
        "train_pair_count": len(labeled_pairs(train)),
        "validation_pair_count": len(labeled_pairs(validation)),
        "train_ranking_triple_count": len(train),
        "validation_ranking_triple_count": len(validation),
        "train_triples_by_kind": triple_kind_counts(train),
        "validation_triples_by_kind": triple_kind_counts(validation),
        "output": str(args.output),
    }
    if not args.run_training:
        print(json.dumps({**preview, "status": "validated_not_trained"}, indent=2))
        return 0
    manifest = run_training(
        artifact_path=args.artifact,
        output_path=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        objective=args.objective,
        pairwise_margin=args.pairwise_margin,
        stability_weight=args.stability_weight,
        trainable_encoder_layers=args.trainable_encoder_layers,
        allow_formal_model_path=args.allow_formal_model_path,
        extra_formal_roots=args.formal_model_root,
    )
    print(json.dumps({**preview, "status": "trained", "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
