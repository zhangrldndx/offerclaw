# -*- coding: utf-8 -*-
"""Shared one-shot structured LLM calls with a bounded process runtime.

The primitive has no tool loop and reads no personal files.  Online and shadow
calls use isolated bounded executors, one total deadline covers initial output
and format repair, late responses are counted, and repeated provider failures
open a small circuit breaker.  No credential value is exposed in metadata.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from threading import BoundedSemaphore, Event, Lock
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)
# Existing injected callers keep this four-argument contract.
TextCaller = Callable[[list[dict[str, str]], int, float, str | None], str | None]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


@dataclass
class StructuredCallMeta:
    calls: int = 0
    repair_used: bool = False
    model: str = ""
    gateway: str = ""
    reasoning_effort: str = ""
    structured_output_mode: str = "prompt_only"
    json_capability: str = "not_requested"
    errors: list[str] = field(default_factory=list)
    raw: str = ""
    planner_queue_ms: float = 0.0
    planner_provider_ms: float = 0.0
    planner_wall_ms: float = 0.0
    planner_deadline_ms: float = 0.0
    planner_timeout_stage: str = ""
    planner_late_response: bool = False
    planner_circuit_state: str = "closed"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "repair_used": self.repair_used,
            "model": self.model,
            "gateway": self.gateway,
            "reasoning_effort": self.reasoning_effort,
            "structured_output_mode": self.structured_output_mode,
            "json_capability": self.json_capability,
            "errors": list(self.errors),
            "planner_queue_ms": round(self.planner_queue_ms, 3),
            "planner_provider_ms": round(self.planner_provider_ms, 3),
            "planner_wall_ms": round(self.planner_wall_ms, 3),
            "planner_deadline_ms": round(self.planner_deadline_ms, 3),
            "planner_timeout_stage": self.planner_timeout_stage,
            "planner_late_response": self.planner_late_response,
            "planner_circuit_state": self.planner_circuit_state,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first complete-looking JSON object from a model response."""
    raw = (text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
    candidate = fenced.group(1) if fenced else ""
    if not candidate:
        start, end = raw.find("{"), raw.rfind("}")
        candidate = raw[start:end + 1] if start >= 0 and end > start else ""
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _capability_cache_path() -> str:
    return os.environ.get(
        "STRUCTURED_LLM_CAPABILITY_CACHE",
        os.path.join(".offerclaw", "structured_provider_capabilities.json"),
    )


def _provider_fingerprint(gateway: str, model: str) -> str:
    raw = f"{gateway.rstrip('/')}|{model}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _json_capability(gateway: str, model: str) -> str:
    """Return ``verified``, ``unsupported`` or ``unknown`` for JSON mode."""
    override = os.environ.get("STRUCTURED_LLM_JSON_CAPABILITY", "").strip().lower()
    if override in {"verified", "supported", "1", "true", "yes", "on"}:
        return "verified"
    if override in {"unsupported", "0", "false", "no", "off"}:
        return "unsupported"
    try:
        with open(_capability_cache_path(), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = (payload.get("providers") or {}).get(
            _provider_fingerprint(gateway, model), {}
        )
        if row.get("json_object") is True:
            return "verified"
        if row.get("json_object") is False:
            return "unsupported"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return "unknown"


def _store_json_capability(gateway: str, model: str, supported: bool) -> None:
    """Persist only public provider capability metadata (never credentials)."""
    path = _capability_cache_path()
    payload: dict[str, Any] = {"schema_version": 1, "providers": {}}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            payload.update(loaded)
            payload["providers"] = dict(loaded.get("providers") or {})
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    payload["schema_version"] = 1
    payload["providers"][_provider_fingerprint(gateway, model)] = {
        "gateway": gateway,
        "model": model,
        "json_object": bool(supported),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def _provider_profile(model: str | None, reasoning_effort: str | None,
                      response_format: str | None,
                      lane: str) -> tuple[str, str, str, str, str, bool]:
    """Resolve public provider metadata without exposing credentials."""
    resolved_model = (model or "").strip()
    gateway = ""
    default_effort = ""
    try:
        from day1_api_starter import get_llm_config, load_local_env
        load_local_env()
        cfg = get_llm_config()
        resolved_model = resolved_model or str(cfg.get("model") or "")
        gateway = str(cfg.get("api_base") or "")
        default_effort = str(cfg.get("reasoning_effort") or "")
    except Exception:
        resolved_model = resolved_model or os.environ.get("LLM_MODEL", "")
        gateway = os.environ.get("OPENAI_BASE_URL", "")
        default_effort = os.environ.get("LLM_REASONING_EFFORT", "")

    if reasoning_effort is None:
        effort = os.environ.get("STRUCTURED_LLM_REASONING_EFFORT", "").strip()
        if not effort and resolved_model.lower().startswith(("gpt", "o")):
            effort = "low"
        if not effort:
            effort = default_effort
    else:
        effort = reasoning_effort.strip()

    if response_format is None:
        configured = os.environ.get("STRUCTURED_LLM_RESPONSE_FORMAT", "").strip().lower()
    else:
        configured = response_format.strip().lower()
    requested_json = configured == "json_object"
    capability = _json_capability(gateway, resolved_model) if requested_json else "not_requested"
    output_mode = (
        "json_object" if requested_json and capability == "verified"
        else "prompt_only"
    )
    # Unknown/unsupported JSON mode is only a prompt-only fallback in the
    # isolated shadow lane.  Online calls fail closed instead of silently
    # changing provider or structured-output semantics.
    blocked = requested_json and capability != "verified" and lane != "shadow"
    return resolved_model, gateway, effort, output_mode, capability, blocked


def _default_caller(messages: list[dict[str, str]], max_tokens: int,
                    temperature: float, model: str | None, *,
                    request_timeout: float, reasoning_effort: str = "",
                    output_mode: str = "prompt_only",
                    response_meta: dict[str, Any] | None = None) -> str | None:
    """Call the configured gateway once, optionally in provider JSON mode."""
    from day1_api_starter import (chat_completion, extract_content,
                                  get_llm_config, load_local_env,
                                  _llm_fallback_config)
    load_local_env()
    fallback = _llm_fallback_config()
    direct_fallback_endpoint = os.environ.get(
        "STRUCTURED_LLM_USE_FALLBACK_ENDPOINT", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    extra: dict[str, Any] = {}
    if reasoning_effort and (model or "").lower().startswith(("gpt", "o")):
        extra["reasoning_effort"] = reasoning_effort
    if output_mode == "json_object":
        extra["response_format"] = {"type": "json_object"}

    # Direct fallback is impossible while LLM_FALLBACK_ENABLED=0 because the
    # gateway returns no fallback configuration in that state.
    if model and fallback and (
        model == fallback.get("model") or direct_fallback_endpoint
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        data = chat_completion(
            f"{fallback['base']}/chat/completions",
            {"Authorization": f"Bearer {fallback['key']}",
             "Content-Type": "application/json"},
            payload, timeout=request_timeout, max_retries=1,
        )
        if response_meta is not None:
            response_meta["model"] = (data or {}).get("model") or model
            response_meta["usage"] = (data or {}).get("usage") or {}
        return extract_content(data)

    # Lazy import avoids rag_gate -> planner -> structured_llm -> rag_gate cycles.
    from rag_gate import _chat
    call_meta: dict[str, Any] = {}
    result = _chat(
        messages, max_tokens=max_tokens, temperature=temperature, model=model,
        request_timeout=request_timeout, max_retries=1,
        extra_payload=extra or None, meta=call_meta,
    )
    if response_meta is not None:
        response_meta.update(call_meta)
        if not response_meta.get("model"):
            response_meta["model"] = model or get_llm_config().get("model")
    return result


@dataclass
class _ExecutionResult:
    value: str | None = None
    status: str = "ok"
    queue_ms: float = 0.0
    provider_ms: float = 0.0
    started: bool = False
    late: bool = False
    circuit_state: str = "closed"


class _CircuitBreaker:
    """Small non-overlapping rolling-window breaker for provider failures."""

    def __init__(self) -> None:
        self.window_size = _env_int("STRUCTURED_LLM_CIRCUIT_WINDOW", 20, 2)
        self.failure_ratio = _env_float(
            "STRUCTURED_LLM_CIRCUIT_FAILURE_RATIO", 0.10, 0.0
        )
        self.bad_windows_required = _env_int(
            "STRUCTURED_LLM_CIRCUIT_BAD_WINDOWS", 2, 1
        )
        self.open_seconds = _env_float("STRUCTURED_LLM_CIRCUIT_OPEN_SECONDS", 30.0)
        self._lock = Lock()
        self._window: list[bool] = []
        self._bad_windows = 0
        self._open_until = 0.0
        self._half_open_inflight = False

    def allow(self) -> tuple[bool, str]:
        now = time.monotonic()
        with self._lock:
            if self._open_until > now:
                return False, "open"
            if self._open_until:
                if self._half_open_inflight:
                    return False, "half_open"
                self._half_open_inflight = True
                return True, "half_open"
            return True, "closed"

    def record(self, failed: bool, state: str) -> None:
        now = time.monotonic()
        with self._lock:
            if state == "half_open":
                self._half_open_inflight = False
                if failed:
                    self._open_until = now + self.open_seconds
                else:
                    self._open_until = 0.0
                    self._window.clear()
                    self._bad_windows = 0
                return
            if state != "closed":
                return
            self._window.append(bool(failed))
            if len(self._window) < self.window_size:
                return
            ratio = sum(self._window) / len(self._window)
            self._window.clear()
            self._bad_windows = (
                self._bad_windows + 1 if ratio >= self.failure_ratio else 0
            )
            if self._bad_windows >= self.bad_windows_required:
                self._open_until = now + self.open_seconds
                self._bad_windows = 0

    def state(self) -> str:
        now = time.monotonic()
        with self._lock:
            if self._open_until > now:
                return "open"
            if self._open_until:
                return "half_open"
            return "closed"


class _BoundedLane:
    def __init__(self, name: str, workers: int, queued: int) -> None:
        self.name = name
        self.workers = workers
        self.queued = queued
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"structured-{name}"
        )
        self.capacity = BoundedSemaphore(workers + queued)
        self._lock = Lock()
        self._late_count = 0

    def _count_late(self, _future: Future[Any]) -> None:
        with self._lock:
            self._late_count += 1

    @property
    def late_count(self) -> int:
        with self._lock:
            return self._late_count

    def run(self, fn: Callable[[], str | None], *, deadline: float,
            queue_timeout: float, breaker: _CircuitBreaker) -> _ExecutionResult:
        allowed, circuit_state = breaker.allow()
        if not allowed:
            return _ExecutionResult(status="circuit_open", circuit_state=circuit_state)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            breaker.record(True, circuit_state)
            return _ExecutionResult(status="deadline", circuit_state=circuit_state)
        admission_timeout = min(max(0.0, queue_timeout), remaining)
        queued_at = time.monotonic()
        if not self.capacity.acquire(timeout=admission_timeout):
            return _ExecutionResult(
                status="queue_timeout",
                queue_ms=(time.monotonic() - queued_at) * 1000,
                circuit_state=circuit_state,
            )

        started = Event()
        started_at: list[float] = []

        def wrapped() -> tuple[str | None, float]:
            start = time.monotonic()
            started_at.append(start)
            started.set()
            value = fn()
            return value, (time.monotonic() - start) * 1000

        try:
            context = copy_context()
            future = self.executor.submit(context.run, wrapped)
        except Exception:
            self.capacity.release()
            breaker.record(True, circuit_state)
            return _ExecutionResult(status="provider_error", circuit_state=circuit_state)
        future.add_done_callback(lambda _f: self.capacity.release())

        start_wait = min(admission_timeout, max(0.0, deadline - time.monotonic()))
        if not started.wait(timeout=start_wait):
            cancelled = future.cancel()
            if not cancelled:
                future.add_done_callback(self._count_late)
            return _ExecutionResult(
                status="queue_timeout",
                queue_ms=(time.monotonic() - queued_at) * 1000,
                late=not cancelled,
                circuit_state=circuit_state,
            )

        queue_ms = ((started_at[0] if started_at else time.monotonic()) - queued_at) * 1000
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cancelled = future.cancel()
            if not cancelled:
                future.add_done_callback(self._count_late)
            breaker.record(True, circuit_state)
            return _ExecutionResult(
                status="timeout", queue_ms=queue_ms, started=True,
                late=not cancelled, circuit_state=circuit_state,
            )
        try:
            value, provider_ms = future.result(timeout=remaining)
        except FutureTimeout:
            cancelled = future.cancel()
            if not cancelled:
                future.add_done_callback(self._count_late)
            breaker.record(True, circuit_state)
            return _ExecutionResult(
                status="timeout", queue_ms=queue_ms, started=True,
                late=not cancelled, circuit_state=circuit_state,
            )
        except Exception:
            breaker.record(True, circuit_state)
            return _ExecutionResult(
                status="provider_error", queue_ms=queue_ms, started=True,
                circuit_state=circuit_state,
            )

        status = "ok" if value is not None else "empty"
        breaker.record(status != "ok", circuit_state)
        return _ExecutionResult(
            value=value, status=status, queue_ms=queue_ms,
            provider_ms=provider_ms, started=True, circuit_state=circuit_state,
        )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class _StructuredRuntime:
    def __init__(self) -> None:
        self.online = _BoundedLane(
            "online",
            _env_int("STRUCTURED_LLM_ONLINE_WORKERS", 2),
            _env_int("STRUCTURED_LLM_ONLINE_QUEUE", 4, 0),
        )
        self.shadow = _BoundedLane(
            "shadow",
            _env_int("STRUCTURED_LLM_SHADOW_WORKERS", 1),
            _env_int("STRUCTURED_LLM_SHADOW_QUEUE", 2, 0),
        )
        self.online_breaker = _CircuitBreaker()
        self.shadow_breaker = _CircuitBreaker()

    def lane(self, name: str) -> _BoundedLane:
        return self.shadow if name == "shadow" else self.online

    def breaker(self, name: str) -> _CircuitBreaker:
        return self.shadow_breaker if name == "shadow" else self.online_breaker

    def stats(self) -> dict[str, Any]:
        return {
            "online_late_responses": self.online.late_count,
            "shadow_late_responses": self.shadow.late_count,
            # Compatibility: circuit_state continues to mean online state.
            "circuit_state": self.online_breaker.state(),
            "online_circuit_state": self.online_breaker.state(),
            "shadow_circuit_state": self.shadow_breaker.state(),
        }

    def shutdown(self) -> None:
        self.online.shutdown()
        self.shadow.shutdown()


_RUNTIME_LOCK = Lock()
_RUNTIME: _StructuredRuntime | None = None


def _runtime() -> _StructuredRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = _StructuredRuntime()
        return _RUNTIME


def structured_runtime_stats() -> dict[str, Any]:
    """Return non-sensitive runtime counters for health/diagnostic endpoints."""
    return _runtime().stats()


def _reset_structured_runtime_for_tests() -> None:
    """Re-read runtime env settings; intended for isolated unit tests only."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        previous = _RUNTIME
        _RUNTIME = _StructuredRuntime()
    if previous is not None:
        previous.shutdown()


def probe_json_object_capability(*, execute: bool = False,
                                 timeout_seconds: float = 8.0) -> dict[str, Any]:
    """Probe GPT ``response_format=json_object`` support on explicit request.

    The default is a dry inspection that performs no network request.  With
    ``execute=True`` the existing local credential is consumed internally, the
    public capability result is cached, and neither the key nor model output is
    returned.
    """
    model, gateway, effort, _mode, cached, _blocked = _provider_profile(
        None, "low", "json_object", "shadow"
    )
    public = {
        "executed": False,
        "gateway": gateway,
        "model": model,
        "cached_capability": cached,
        "status": "not_run",
    }
    if not execute:
        return public
    deadline = time.monotonic() + max(0.05, float(timeout_seconds))
    response_meta: dict[str, Any] = {}
    messages = [
        {"role": "system", "content": "只返回 JSON 对象。"},
        {"role": "user", "content": '返回且只返回 {"probe":"ok"}'},
    ]
    runtime = _runtime()
    result = runtime.shadow.run(
        lambda: _default_caller(
            messages, 80, 0.0, model or None,
            request_timeout=max(0.05, deadline - time.monotonic()),
            reasoning_effort=effort, output_mode="json_object",
            response_meta=response_meta,
        ),
        deadline=deadline,
        queue_timeout=_env_float("STRUCTURED_LLM_QUEUE_TIMEOUT_SECONDS", 0.1),
        breaker=runtime.shadow_breaker,
    )
    public["executed"] = result.started
    if result.status != "ok":
        public["status"] = "probe_failed"
        public["failure_stage"] = result.status
        return public
    parsed = extract_json_object(result.value or "")
    supported = bool(parsed and parsed.get("probe") == "ok")
    _store_json_capability(gateway, model, supported)
    public["status"] = "verified" if supported else "unsupported"
    public["cached_capability"] = public["status"]
    return public


def _validate(model_type: type[T], raw: str) -> tuple[T | None, str]:
    data = extract_json_object(raw)
    if data is None:
        return None, "invalid_json"
    try:
        return model_type.model_validate(data), ""
    except ValidationError as exc:
        compact = "; ".join(
            f"{'.'.join(str(x) for x in item.get('loc', ()))}: {item.get('msg', '')}"
            for item in exc.errors()[:8]
        )
        return None, f"schema_error: {compact}"[:1200]


def _apply_execution_meta(meta: StructuredCallMeta, result: _ExecutionResult) -> None:
    if result.started:
        meta.calls += 1
    meta.planner_queue_ms += result.queue_ms
    meta.planner_provider_ms += result.provider_ms
    meta.planner_late_response = meta.planner_late_response or result.late
    meta.planner_circuit_state = result.circuit_state
    if result.status in {"queue_timeout", "timeout", "deadline", "circuit_open"}:
        meta.planner_timeout_stage = result.status


def _execution_error(status: str, *, repair: bool = False) -> str:
    prefix = "repair_" if repair else ""
    return {
        "queue_timeout": prefix + "queue_timeout",
        "timeout": prefix + "timeout_or_model_unavailable",
        "deadline": prefix + "deadline_exhausted",
        "provider_error": prefix + "provider_error",
        "empty": prefix + "empty_model_response",
        "circuit_open": prefix + "circuit_open",
    }.get(status, prefix + "model_unavailable")


def call_structured(model_type: type[T], messages: list[dict[str, str]], *,
                    model: str | None = None, timeout_seconds: float = 5.0,
                    max_tokens: int = 1200, repair: bool = True,
                    caller: TextCaller | None = None,
                    reasoning_effort: str | None = None,
                    response_format: str | None = None,
                    schema_instructions: str | None = None,
                    repair_context: str | None = None,
                    lane: str = "online",
                    queue_timeout_seconds: float | None = None,
                    repair_min_remaining_seconds: float | None = None,
                    ) -> tuple[T | None, StructuredCallMeta]:
    """Call once and optionally repair within one total wall-clock budget.

    Existing callers remain source-compatible; all new controls are optional
    keyword arguments. ``schema_instructions`` lets a small, repeatedly used
    contract send an equivalent compact shape; validation and repair still use
    the authoritative Pydantic schema locally.
    """
    wall_start = time.monotonic()
    total_timeout = max(0.001, float(timeout_seconds))
    deadline = wall_start + total_timeout
    resolved_model, gateway, effort, output_mode, capability, json_mode_blocked = _provider_profile(
        model, reasoning_effort, response_format, lane
    )
    meta = StructuredCallMeta(
        model=resolved_model,
        gateway=gateway,
        reasoning_effort=effort,
        structured_output_mode=output_mode,
        json_capability=capability,
        planner_deadline_ms=total_timeout * 1000,
    )
    if json_mode_blocked and caller is None:
        meta.errors.append("json_object_capability_unverified")
        meta.planner_timeout_stage = "capability_gate"
        meta.planner_wall_ms = (time.monotonic() - wall_start) * 1000
        return None, meta
    schema = model_type.model_json_schema()
    schema_contract = (
        schema_instructions.strip()
        if schema_instructions is not None and schema_instructions.strip()
        else (
            "必须只返回一个 JSON 对象，并严格满足下面的 JSON Schema；"
            "additionalProperties=false 的位置不得增加字段：\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
    )
    schema_message = {
        "role": "system",
        "content": schema_contract,
    }
    call_messages = [*messages, schema_message]
    queue_timeout = (
        _env_float("STRUCTURED_LLM_QUEUE_TIMEOUT_SECONDS", 0.1)
        if queue_timeout_seconds is None else max(0.0, queue_timeout_seconds)
    )
    repair_min_remaining = (
        _env_float("STRUCTURED_LLM_REPAIR_MIN_REMAINING_SECONDS", 1.5)
        if repair_min_remaining_seconds is None
        else max(0.0, repair_min_remaining_seconds)
    )
    selected_lane = _runtime().lane(lane)
    response_meta: dict[str, Any] = {}

    def invoke(call_messages_arg: list[dict[str, str]]) -> str | None:
        remaining = max(0.05, deadline - time.monotonic())
        if caller is not None:
            return caller(call_messages_arg, max_tokens, 0.0, resolved_model or None)
        return _default_caller(
            call_messages_arg, max_tokens, 0.0, resolved_model or None,
            request_timeout=remaining, reasoning_effort=effort,
            output_mode=output_mode, response_meta=response_meta,
        )

    runtime = _runtime()
    result = selected_lane.run(
        lambda: invoke(call_messages), deadline=deadline,
        queue_timeout=queue_timeout, breaker=runtime.breaker(lane),
    )
    _apply_execution_meta(meta, result)
    meta.raw = result.value or ""
    if response_meta.get("model"):
        meta.model = str(response_meta["model"])
    usage = response_meta.get("usage") or {}
    if usage:
        meta.prompt_tokens = usage.get("prompt_tokens")
        meta.completion_tokens = usage.get("completion_tokens")
    if result.status != "ok":
        meta.errors.append(_execution_error(result.status))
        meta.planner_wall_ms = (time.monotonic() - wall_start) * 1000
        return None, meta

    raw = result.value or ""
    value, error = _validate(model_type, raw)
    if value is not None:
        meta.planner_wall_ms = (time.monotonic() - wall_start) * 1000
        return value, meta
    meta.errors.append(error)
    remaining = deadline - time.monotonic()
    if not repair or remaining < repair_min_remaining:
        if repair and remaining < repair_min_remaining:
            meta.errors.append("repair_skipped_insufficient_budget")
        meta.planner_wall_ms = (time.monotonic() - wall_start) * 1000
        return None, meta

    repair_messages = [
        {
            "role": "system",
            "content": (
                "只修复 JSON 格式和 schema，保持原回答已经表达的语义选择；"
                "若提供 original_request，可据此补齐能直接确定的必填契约字段；"
                "不得引入原请求未表达的动作、路由或事实。只输出 JSON。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "invalid_output": raw[:12000],
                "validation_error": error,
                **({"original_request": repair_context[:4000]}
                   if repair_context else {}),
                "required_schema": schema,
            }, ensure_ascii=False)[:24000],
        },
    ]
    meta.repair_used = True
    repair_response_meta: dict[str, Any] = {}

    def invoke_repair() -> str | None:
        remaining_repair = max(0.05, deadline - time.monotonic())
        if caller is not None:
            return caller(repair_messages, max_tokens, 0.0, resolved_model or None)
        return _default_caller(
            repair_messages, max_tokens, 0.0, resolved_model or None,
            request_timeout=remaining_repair, reasoning_effort=effort,
            output_mode=output_mode, response_meta=repair_response_meta,
        )

    repaired = selected_lane.run(
        invoke_repair, deadline=deadline, queue_timeout=queue_timeout,
        breaker=runtime.breaker(lane),
    )
    _apply_execution_meta(meta, repaired)
    if repair_response_meta.get("model"):
        meta.model = str(repair_response_meta["model"])
    repair_usage = repair_response_meta.get("usage") or {}
    if repair_usage:
        prompt = repair_usage.get("prompt_tokens")
        completion = repair_usage.get("completion_tokens")
        if prompt is not None:
            meta.prompt_tokens = (meta.prompt_tokens or 0) + prompt
        if completion is not None:
            meta.completion_tokens = (meta.completion_tokens or 0) + completion
    if repaired.status != "ok":
        meta.errors.append(_execution_error(repaired.status, repair=True))
        meta.planner_wall_ms = (time.monotonic() - wall_start) * 1000
        return None, meta
    value, repair_error = _validate(model_type, repaired.value or "")
    if value is None:
        meta.errors.append(f"repair_{repair_error}")
    meta.planner_wall_ms = (time.monotonic() - wall_start) * 1000
    return value, meta
