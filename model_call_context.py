# -*- coding: utf-8 -*-
"""Request-local accounting and deadline propagation for OfferClaw model calls."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Any, Iterator


@dataclass
class ModelCallUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    models: set[str] = field(default_factory=set)
    deadline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "offerclaw_calls": self.calls,
            "openclaw_calls": 0,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "models": sorted(self.models),
        }


_CURRENT: ContextVar[ModelCallUsage | None] = ContextVar(
    "offerclaw_model_call_usage", default=None
)


@contextmanager
def model_call_scope(*, timeout_seconds: float | None = None) -> Iterator[ModelCallUsage]:
    usage = ModelCallUsage(
        deadline=(time.monotonic() + max(0.05, float(timeout_seconds)))
        if timeout_seconds is not None else None
    )
    token = _CURRENT.set(usage)
    try:
        yield usage
    finally:
        _CURRENT.reset(token)


def record_model_request(payload: Any) -> None:
    """Count an outbound provider request without retaining prompts or responses."""
    usage = _CURRENT.get()
    if usage is None:
        return
    usage.calls += 1
    if isinstance(payload, dict):
        model = payload.get("model")
        if isinstance(model, str) and model.strip():
            usage.models.add(model.strip())


def record_model_response(data: Any) -> None:
    usage = _CURRENT.get()
    if usage is None or not isinstance(data, dict):
        return
    model = data.get("model")
    if isinstance(model, str) and model.strip():
        usage.models.add(model.strip())
    tokens = data.get("usage")
    if not isinstance(tokens, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            value = max(0, int(tokens.get(key) or 0))
        except (TypeError, ValueError):
            value = 0
        setattr(usage, key, getattr(usage, key) + value)


def remaining_seconds(default: float) -> float:
    usage = _CURRENT.get()
    if usage is None or usage.deadline is None:
        return max(0.05, float(default))
    remaining = usage.deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("OfferClaw query deadline exhausted")
    return max(0.05, min(float(default), remaining))


__all__ = [
    "ModelCallUsage", "model_call_scope", "record_model_request",
    "record_model_response", "remaining_seconds",
]
