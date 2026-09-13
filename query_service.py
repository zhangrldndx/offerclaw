# -*- coding: utf-8 -*-
"""Canonical OfferClaw query execution shared by Web, CLI, and WeChat."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator

from model_call_context import model_call_scope


QUERY_SERVICE_VERSION = "offerclaw.query-service.v1"
DEFAULT_REPLY_CHARS = 1800
_FINGERPRINT_FILES = (
    "query_service.py", "rag_api.py", "rag_gate.py", "rag_multi_source.py",
    "rag_query_plan.py", "semantic_query_planner.py", "wechat_data_bridge.py",
    "wechat_dispatch.py",
)
_WARMUP_LOCK = threading.Lock()
_WARMUP_STATUS: dict[str, Any] = {
    "status": "not_started",
    "embedding": "not_started",
    "reranker": "not_started",
    "elapsed_ms": 0.0,
}


def query_runtime_status() -> dict[str, Any]:
    return dict(_WARMUP_STATUS)


def warm_query_runtime() -> dict[str, Any]:
    """Load the local embedding and reranker before the service accepts traffic."""
    if _WARMUP_STATUS.get("status") == "ready":
        return query_runtime_status()
    with _WARMUP_LOCK:
        if _WARMUP_STATUS.get("status") == "ready":
            return query_runtime_status()
        started = time.perf_counter()
        _WARMUP_STATUS.update({
            "status": "warming", "embedding": "loading", "reranker": "not_started",
        })
        try:
            from day1_api_starter import load_local_env
            from rag_tools import get_embedding_config, get_embeddings_batch

            load_local_env()
            embedding = get_embedding_config()
            if embedding.get("provider") != "local":
                raise RuntimeError("query service requires the local embedding provider")
            get_embeddings_batch(["OfferClaw query runtime warmup"], max_retries=1)
            _WARMUP_STATUS["embedding"] = "ready"

            from rag_rerank import _load_reranker, rerank_enabled
            from rag_retrieval_trace import retrieval_profile_registry

            if rerank_enabled():
                profile = retrieval_profile_registry()["baseline"]
                reranker = _load_reranker(profile.reranker_model)
                if reranker is None:
                    raise RuntimeError("local reranker could not be loaded")
                reranker.predict([["warmup", "warmup"]])
                _WARMUP_STATUS["reranker"] = "ready"
            else:
                _WARMUP_STATUS["reranker"] = "disabled"
            _WARMUP_STATUS.update({
                "status": "ready",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            })
        except Exception as exc:
            _WARMUP_STATUS.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            })
            raise
    return query_runtime_status()


def repository_fingerprint(root: str | Path | None = None) -> str:
    base = Path(root) if root else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _FINGERPRINT_FILES:
        path = base / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def split_reply_parts(text: str, max_chars: int = DEFAULT_REPLY_CHARS) -> list[str]:
    """Split complete replies at paragraph/line boundaries without dropping text."""
    value = str(text or "").strip()
    if not value:
        return []
    if max_chars < 200:
        raise ValueError("max_chars is too small")
    parts: list[str] = []
    current = ""
    blocks = value.splitlines(keepends=True)
    for block in blocks:
        while len(block) > max_chars:
            if current:
                parts.append(current.rstrip())
                current = ""
            cut = block.rfind(" ", 0, max_chars + 1)
            if cut < max_chars // 2:
                cut = max_chars
            parts.append(block[:cut].rstrip())
            block = block[cut:].lstrip()
        if len(current) + len(block) > max_chars and current:
            parts.append(current.rstrip())
            current = block
        else:
            current += block
    if current.strip():
        parts.append(current.rstrip())
    return [part for part in parts if part]


@dataclass
class QueryExecutionResult:
    payload: dict[str, Any]
    model_usage: dict[str, Any]
    data_version: str
    trace_id: str
    status: str

    @property
    def answer(self) -> str:
        return str(self.payload.get("answer") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload,
            "status": self.status,
            "data_version": self.data_version,
            "model_usage": self.model_usage,
            "trace_id": self.trace_id,
            "query_service_version": QUERY_SERVICE_VERSION,
        }

    def wechat_dict(self) -> dict[str, Any]:
        sources = [str(value) for value in self.payload.get("sources") or [] if str(value)]
        as_of = self.payload.get("freshness") or dt.datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        if isinstance(as_of, dict):
            as_of = as_of.get("as_of") or as_of.get("updated_at") or ""
        footer = (
            f"数据来源：Windows OfferClaw 真实数据｜截至 {as_of or '时间未记录'}"
            f"｜数据版本 {self.data_version[:16]}\n"
            f"OfferClaw 模型调用 {self.model_usage.get('offerclaw_calls', 0)}"
            "｜OpenClaw 模型调用 0"
        )
        answer = self.answer.strip()
        complete = f"{answer}\n\n{footer}" if answer else footer
        return {
            "schema_version": "offerclaw.wechat-query.response.v1",
            "status": self.status,
            "answer": complete,
            "reply_parts": split_reply_parts(complete),
            "service_mode": self.payload.get("service_mode")
            or (self.payload.get("intent_frame") or {}).get("service_mode") or "",
            "routes": self.payload.get("routes") or [],
            "sources": sources,
            "freshness": self.payload.get("freshness") or as_of,
            "data_version": self.data_version,
            "model_usage": self.model_usage,
            "trace_id": self.trace_id,
            "mode": self.payload.get("mode") or "",
            "answer_action": self.payload.get("answer_action") or "answer",
            "resolved_entities": self.payload.get("resolved_entities") or {},
            "intent_frame": self.payload.get("intent_frame") or {},
        }


def _data_version(result: dict[str, Any]) -> str:
    from business_read_service import authoritative_data_version

    authoritative = authoritative_data_version()
    value = {
        "authoritative": authoritative,
        "index": result.get("index_fingerprint") or "",
        "sources": sorted(str(item) for item in result.get("sources") or []),
    }
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def execute_query(question: str, top_k: int = 5, *, context: list[str] | None = None,
                  conversation_id: str = "", timeout_seconds: float | None = None,
                  trace_id: str = "") -> QueryExecutionResult:
    question = str(question or "").strip()
    if not question:
        raise ValueError("question cannot be empty")
    if len(question) > 12_000:
        raise ValueError("question is too long")
    if not 1 <= int(top_k) <= 20:
        raise ValueError("top_k must be between 1 and 20")
    from rag_gate import gated_query

    with model_call_scope(timeout_seconds=timeout_seconds) as usage:
        payload = gated_query(
            question, int(top_k), context=context, conversation_id=conversation_id,
        )
    mode = str(payload.get("mode") or "")
    status = "partial" if mode in {"retrieval_error", "general_fallback"} else "ok"
    if not trace_id:
        trace_id = "query_" + hashlib.sha256(
            f"{conversation_id}|{question}|{dt.datetime.now().timestamp()}".encode("utf-8")
        ).hexdigest()[:20]
    return QueryExecutionResult(
        payload=payload,
        model_usage=usage.to_dict(),
        data_version=_data_version(payload),
        trace_id=trace_id,
        status=status,
    )


def iter_query_events(question: str, top_k: int = 5, *, context: list[str] | None = None,
                      conversation_id: str = "") -> Iterator[dict[str, Any]]:
    """Canonical streaming event source. Consumers own channel-specific persistence."""
    from rag_gate import gated_query_stream
    import inspect

    parameters = inspect.signature(gated_query_stream).parameters
    kwargs: dict[str, Any] = {}
    if "context" in parameters:
        kwargs["context"] = context
    if "conversation_id" in parameters:
        kwargs["conversation_id"] = conversation_id
    for event in gated_query_stream(question, top_k, **kwargs):
        if event.get("type") == "meta":
            event = {
                **event,
                "data_version": _data_version(event),
                "query_service_version": QUERY_SERVICE_VERSION,
            }
        yield event


__all__ = [
    "DEFAULT_REPLY_CHARS", "QUERY_SERVICE_VERSION", "QueryExecutionResult",
    "execute_query", "iter_query_events", "query_runtime_status",
    "repository_fingerprint", "split_reply_parts", "warm_query_runtime",
]
