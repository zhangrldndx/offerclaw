# -*- coding: utf-8 -*-
"""Request-scoped traffic provenance for evaluation-safe production hooks."""

from __future__ import annotations

from contextvars import ContextVar, Token


ORGANIC = "organic"
AUTOMATED_ORIGINS = {
    "agent_generated", "historical_replay", "stress", "test", "wechat_direct",
}
ALLOWED_ORIGINS = {ORGANIC, *AUTOMATED_ORIGINS}
HEADER = "X-OfferClaw-Traffic-Origin"

_CURRENT: ContextVar[str] = ContextVar("offerclaw_traffic_origin", default=ORGANIC)


def normalize_traffic_origin(value: str | None) -> str:
    """No header means the human-facing app; unknown claims fail safe."""
    raw = (value or "").strip().lower()
    if not raw:
        return ORGANIC
    return raw if raw in ALLOWED_ORIGINS else "unclassified"


def current_traffic_origin() -> str:
    return _CURRENT.get()


def set_traffic_origin(value: str | None) -> Token:
    return _CURRENT.set(normalize_traffic_origin(value))


def reset_traffic_origin(token: Token) -> None:
    _CURRENT.reset(token)


class _TrafficOriginIterator:
    """Bind provenance for each ``next`` call, even across worker contexts."""

    def __init__(self, iterable, origin: str):
        self._iterator = iter(iterable)
        self._origin = normalize_traffic_origin(origin)

    def __iter__(self):
        return self

    def __next__(self):
        token = set_traffic_origin(self._origin)
        try:
            return next(self._iterator)
        finally:
            reset_traffic_origin(token)


def bind_traffic_origin_iterable(iterable, origin: str):
    """Return a sync iterator whose every advance sees the given origin."""
    return _TrafficOriginIterator(iterable, origin)


async def traffic_origin_middleware(request, call_next):
    token = set_traffic_origin(request.headers.get(HEADER))
    try:
        response = await call_next(request)
        response.headers[HEADER] = current_traffic_origin()
        return response
    finally:
        reset_traffic_origin(token)


__all__ = [
    "ALLOWED_ORIGINS", "AUTOMATED_ORIGINS", "HEADER", "ORGANIC",
    "bind_traffic_origin_iterable", "current_traffic_origin", "normalize_traffic_origin",
    "reset_traffic_origin", "set_traffic_origin", "traffic_origin_middleware",
]
