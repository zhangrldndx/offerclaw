# -*- coding: utf-8 -*-
"""observability.py — OfferClaw 结构化 Trace + 重放（V4 §4）

为什么需要：
    既有 ``career_flow`` 把节点动作写在 ``state["trace"]`` 列表里，但仅存活于
    单次调用 / 单次 HTTP 请求，**事后无法回看**——更别提"昨天那次匹配
    跑了哪些节点 / 哪里挂了"。本模块把每次 ``CareerFlow`` 执行落成一份
    JSONL 文件（``logs/traces/<trace_id>.jsonl``），并提供：

    1. ``TraceWriter`` — 上下文管理器，按事件 append
    2. ``read_trace(trace_id)`` — 重放：读回全部事件
    3. ``list_traces(limit)`` — 列出最近的 N 条 trace（按文件 mtime 倒序）
    4. ``trace_career_flow(...)`` — 包装层：跑 routed flow + 自动落 trace

设计：
    - 完全文件型，零依赖、零进程外服务，与项目"状态走文件"哲学一致
    - 每行一个 JSON 对象，便于 ``jq``、``grep``、``tail -f`` 调试
    - trace_id 用日期前缀 + 短 uuid，文件名本身就排好序了
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import uuid
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

TRACE_DIR = os.path.join(BASE_DIR, "logs", "traces")


def _new_trace_id() -> str:
    """前缀化的 trace id：``YYYYMMDD_HHMMSS_<6 字符>``。
    日期前缀让 ``ls -lt`` 直接按时间顺序排，6 字符随机段避免同秒撞名。"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def _iso_now() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _ensure_trace_dir() -> None:
    os.makedirs(TRACE_DIR, exist_ok=True)


# =====================================================
# TraceWriter
# =====================================================

class TraceWriter:
    """JSONL trace 写入器。建议用 ``with`` 语法，自动 flush + close。

    每条 event 自动注入 ``trace_id`` 与 ``ts_iso``，不需要调用方写。
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or _new_trace_id()
        _ensure_trace_dir()
        self.path = os.path.join(TRACE_DIR, f"{self.trace_id}.jsonl")
        self._f = None
        self._closed = True

    def __enter__(self) -> "TraceWriter":
        self._f = open(self.path, "a", encoding="utf-8")
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def record(self, event: dict[str, Any]) -> None:
        if self._closed or self._f is None:
            raise RuntimeError("TraceWriter 已关闭；请用 with 语法。")
        full = {
            "trace_id": self.trace_id,
            "ts_iso": _iso_now(),
            **event,
        }
        self._f.write(json.dumps(full, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._closed and self._f is not None:
            self._f.close()
            self._closed = True


# =====================================================
# 读取与列表
# =====================================================

def read_trace(trace_id: str) -> dict:
    """读回单条 trace 的全部事件。"""
    path = os.path.join(TRACE_DIR, f"{trace_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"trace not found: {trace_id}")
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "trace_id": trace_id,
        "path": path,
        "n_events": len(events),
        "events": events,
    }


def list_traces(limit: int = 20) -> list[dict]:
    """列最近的 N 条 trace（按 mtime 倒序），返回轻量元信息。"""
    if not os.path.isdir(TRACE_DIR):
        return []
    files: list[tuple[float, str]] = []
    for name in os.listdir(TRACE_DIR):
        if not name.endswith(".jsonl"):
            continue
        p = os.path.join(TRACE_DIR, name)
        try:
            files.append((os.path.getmtime(p), name))
        except OSError:
            continue
    files.sort(reverse=True)
    out: list[dict] = []
    for mtime, name in files[:limit]:
        trace_id = name[:-len(".jsonl")]
        try:
            # 偷一眼第一行抓 meta
            with open(os.path.join(TRACE_DIR, name), "r",
                      encoding="utf-8") as f:
                first = json.loads((f.readline() or "{}").strip())
        except (OSError, json.JSONDecodeError):
            first = {}
        out.append({
            "trace_id": trace_id,
            "mtime_iso": datetime.datetime.fromtimestamp(mtime).isoformat(
                timespec="seconds"),
            "jd_title": first.get("jd_title", ""),
            "kind": first.get("event", ""),
        })
    return out


# =====================================================
# CareerFlow instrumentation
# =====================================================

def _summarize_state(out: dict) -> dict:
    """把最终 state 压成轻量摘要（不放进 trace 文件全身，避免噪音）。"""
    return {
        "status": out.get("match_report", {}).get("status"),
        "status_code": out.get("match_report", {}).get("status_code"),
        "direction": out.get("match_report", {}).get("direction"),
        "gap_total": out.get("gaps", {}).get("total"),
        "route_taken": out.get("route_taken"),
        "route_history": out.get("route_history", []),
        "checkpoint_status": out.get("checkpoint_status"),
        "memory_status": out.get("memory_status"),
        "n_errors": len(out.get("errors", [])),
        "n_confirmations": len(out.get("requires_confirmation", [])),
    }


def trace_career_flow(jd_text: str, *, jd_title: str = "未命名 JD",
                      skip_llm: bool = True, routed: bool = True,
                      trace_id: str | None = None) -> dict:
    """跑 CareerFlow 并把每个节点动作落成 JSONL trace。

    Returns: 原 state dict + ``_trace_id`` 字段（方便前端拿去查 /api/trace）。
    """
    from career_flow import run_career_flow, run_career_flow_routed
    runner = run_career_flow_routed if routed else run_career_flow

    with TraceWriter(trace_id) as tw:
        tw.record({
            "event": "start",
            "jd_title": jd_title,
            "jd_chars": len(jd_text or ""),
            "skip_llm": skip_llm,
            "routed": routed,
        })
        try:
            out = runner(jd_text, jd_title=jd_title, skip_llm=skip_llm)
        except Exception as e:
            tw.record({"event": "fatal", "error": str(e)[:300]})
            raise
        for i, t in enumerate(out.get("trace", [])):
            tw.record({"event": "node", "seq": i, **t})
        for e in out.get("errors", []) or []:
            tw.record({"event": "node_error", **e})
        tw.record({"event": "end", "summary": _summarize_state(out)})
        return {**out, "_trace_id": tw.trace_id}


if __name__ == "__main__":
    sample = (
        "岗位名称：AI Agent 应用开发实习生\n工作地点：上海\n"
        "技术要求：Python / LangGraph / RAG / FastAPI\n工作性质：实习\n"
    )
    out = trace_career_flow(sample, jd_title="DEMO")
    print(json.dumps({
        "trace_id": out["_trace_id"],
        "status": out.get("match_report", {}).get("status"),
        "route": out.get("route_taken"),
    }, ensure_ascii=False, indent=2))
    print("\n--- read back ---")
    print(json.dumps(read_trace(out["_trace_id"]),
                     ensure_ascii=False, indent=2)[:500])
