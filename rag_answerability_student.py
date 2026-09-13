# -*- coding: utf-8 -*-
"""Local four-grade answerability student used only after the BGE reranker.

Loading is local-only and fail-open.  A missing, malformed, or unapproved
checkpoint returns the incumbent BGE order and never turns into an implicit
LLM call.  ``balanced`` policy thresholds live in a separately frozen
validation artifact so the blind evaluator cannot tune them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable


MODEL_MANIFEST_SCHEMA = "answerability-student-model-v1"
BALANCED_POLICY_SCHEMA = "answerability-student-balanced-policy-v1"
DEFAULT_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_DEPTH = 6
GRADES = (0, 1, 2, 3)

_LOAD_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, tuple[Any, Any, str, str] | None] = {}


class StudentModelError(RuntimeError):
    """Raised internally for invalid or unavailable local student artifacts."""


@dataclass(frozen=True)
class Prediction:
    grade: int
    confidence: float
    expected_grade: float
    probabilities: tuple[float, float, float, float]


def configured_model_path() -> str:
    return os.environ.get("RAG_ANSWERABILITY_STUDENT_MODEL", "").strip()


def _device(torch_module) -> str:
    explicit = os.environ.get("OFFERCLAW_TORCH_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "student_model_manifest.json"
    if not manifest_path.is_file():
        raise StudentModelError("student_model_manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudentModelError("student model manifest is invalid") from exc
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA:
        raise StudentModelError("unsupported student model manifest")
    status = manifest.get("release_status")
    development_allowed = os.environ.get(
        "RAG_ANSWERABILITY_STUDENT_ALLOW_DEVELOPMENT", "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if status != "approved" and not (development_allowed and status == "development"):
        raise StudentModelError("student checkpoint is not approved")
    fingerprint = manifest.get("checkpoint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise StudentModelError("student checkpoint fingerprint is invalid")
    return manifest


def _load_local_model(path_value: str | None = None):
    raw = (path_value or configured_model_path()).strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    key = str(path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    with _LOAD_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        try:
            if not path.is_dir():
                raise StudentModelError("student model path is not a directory")
            manifest = _read_manifest(path)
            if checkpoint_tree_sha256(path) != manifest["checkpoint_sha256"]:
                raise StudentModelError("student checkpoint content hash mismatch")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True,
            )
            if int(getattr(model.config, "num_labels", 0)) != 4:
                raise StudentModelError("student classifier must expose four grades")
            device = _device(torch)
            model.to(device)
            model.eval()
            loaded = (tokenizer, model, device, manifest["checkpoint_sha256"])
            _MODEL_CACHE[key] = loaded
            return loaded
        except Exception:
            # Failure is cached: repeatedly attempting to load a broken model
            # would add latency to every query and still cannot improve rank.
            _MODEL_CACHE[key] = None
            return None


def predict(
    question: str, chunks: list[str], *, model_path: str | None = None,
) -> tuple[list[Prediction], dict[str, Any]] | None:
    loaded = _load_local_model(model_path)
    if loaded is None or not chunks:
        return None
    tokenizer, model, device, fingerprint = loaded
    try:
        import torch

        batch_size_raw = os.environ.get("RAG_ANSWERABILITY_STUDENT_BATCH", "8")
        batch_size = max(1, int(batch_size_raw))
        predictions: list[Prediction] = []
        with _PREDICT_LOCK, torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                encoded = tokenizer(
                    [question] * len(batch), batch, padding=True, truncation=True,
                    max_length=384, return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 4:
                    raise StudentModelError("student returned non-four-grade logits")
                probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                for values in probs:
                    grade = max(range(4), key=lambda idx: values[idx])
                    predictions.append(Prediction(
                        grade=grade,
                        confidence=float(values[grade]),
                        expected_grade=sum(idx * float(value) for idx, value in enumerate(values)),
                        probabilities=tuple(float(value) for value in values),
                    ))
        return predictions, {
            "model_sha256": fingerprint,
            "device": device,
        }
    except Exception:
        return None


def _coerce_predictions(values: Any) -> tuple[list[Prediction], dict[str, Any]] | None:
    """Test/evaluation injection boundary that still validates all outputs."""

    if values is None:
        return None
    metadata: dict[str, Any] = {}
    if isinstance(values, tuple) and len(values) == 2:
        values, metadata = values
    if not isinstance(values, list):
        return None
    parsed: list[Prediction] = []
    for item in values:
        if isinstance(item, Prediction):
            parsed.append(item)
            continue
        if not isinstance(item, dict):
            return None
        grade = item.get("grade")
        confidence = item.get("confidence")
        expected = item.get("expected_grade", grade)
        if grade not in GRADES or not isinstance(confidence, (int, float)):
            return None
        if not 0.0 <= float(confidence) <= 1.0 or not isinstance(expected, (int, float)):
            return None
        probabilities = item.get("probabilities") or [0.0] * 4
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 4:
            return None
        parsed.append(Prediction(
            int(grade), float(confidence), float(expected),
            tuple(float(value) for value in probabilities),
        ))
    return parsed, metadata


def _load_balanced_policy(model_sha256: str) -> dict[str, Any] | None:
    raw = os.environ.get("RAG_ANSWERABILITY_BALANCED_POLICY", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != BALANCED_POLICY_SCHEMA:
        return None
    if payload.get("status") != "balanced_candidate":
        return None
    if payload.get("frozen_on_split") != "validation":
        return None
    if payload.get("student_model_sha256") != model_sha256:
        return None
    confidence = payload.get("min_top_confidence")
    gap = payload.get("min_expected_grade_gap")
    call_rate = payload.get("max_teacher_call_rate")
    if not all(isinstance(value, (int, float)) for value in (confidence, gap, call_rate)):
        return None
    if not (0 <= confidence <= 1 and 0 <= gap <= 3 and 0 <= call_rate <= 0.15):
        return None
    return payload


def rerank_by_student(
    question: str, docs: list, metas: list, dists: list, scores: list, *,
    depth: int = DEFAULT_DEPTH,
    predictor: Callable[[str, list[str]], Any] | None = None,
    stats: dict[str, Any] | None = None,
    balanced: bool = False,
):
    """Rank top candidates by predicted grade, then the existing BGE score."""

    if not docs:
        return docs, metas, dists, scores
    head = list(range(min(max(1, depth), len(docs))))
    call = predictor or (lambda q, chunks: predict(q, chunks))
    try:
        result = _coerce_predictions(call(question, [docs[index] for index in head]))
    except Exception:
        result = None
    if result is None:
        if stats is not None:
            stats.update({"applied": False, "reason": "student_unavailable"})
        return docs, metas, dists, scores
    predictions, metadata = result
    if len(predictions) != len(head):
        if stats is not None:
            stats.update({"applied": False, "reason": "prediction_count_mismatch"})
        return docs, metas, dists, scores

    def base_score(index: int) -> float:
        value = scores[index] if index < len(scores) else None
        return float(value) if isinstance(value, (int, float)) else 0.0

    by_index = dict(zip(head, predictions))
    ordered_head = sorted(
        head,
        key=lambda index: (-by_index[index].grade, -base_score(index)),
    )
    order = ordered_head + list(range(len(head), len(docs)))
    take = lambda seq: [seq[index] for index in order] if len(seq) == len(docs) else seq
    post_predictions = [by_index[index] for index in ordered_head]
    fallback = False
    policy_reason = "not_balanced"
    model_sha = str(metadata.get("model_sha256") or "")
    if balanced:
        policy = _load_balanced_policy(model_sha)
        if policy is None:
            policy_reason = "balanced_policy_unavailable"
        else:
            top = post_predictions[0]
            second_expected = post_predictions[1].expected_grade if len(post_predictions) > 1 else -1.0
            expected_gap = top.expected_grade - second_expected
            fallback = (
                top.confidence < float(policy["min_top_confidence"])
                or expected_gap < float(policy["min_expected_grade_gap"])
            )
            policy_reason = "uncertain" if fallback else "confident"
    if stats is not None:
        stats.update({
            "applied": True,
            "reason": "ok",
            "source": "student",
            "model_sha256": model_sha or None,
            "device": metadata.get("device"),
            "grades": {index: prediction.grade for index, prediction in enumerate(post_predictions)},
            "confidences": {
                index: round(prediction.confidence, 6)
                for index, prediction in enumerate(post_predictions)
            },
            "expected_grades": {
                index: round(prediction.expected_grade, 6)
                for index, prediction in enumerate(post_predictions)
            },
            "fallback_recommended": fallback,
            "balanced_policy_reason": policy_reason,
        })
    return take(docs), take(metas), take(dists), take(scores)


def checkpoint_tree_sha256(path: str | Path) -> str:
    """Stable content hash used when freezing a checkpoint before blind eval."""

    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if item.name == "student_model_manifest.json":
            continue
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


__all__ = [
    "BALANCED_POLICY_SCHEMA", "DEFAULT_DEPTH", "DEFAULT_MODEL_NAME",
    "MODEL_MANIFEST_SCHEMA", "Prediction", "StudentModelError",
    "checkpoint_tree_sha256", "configured_model_path", "predict",
    "rerank_by_student",
]
