# -*- coding: utf-8 -*-
"""traj_adapter.py — OfferClaw 轨迹 → 公共元数据 Adapter(实验规划 v2 · P0-c)。

设计纪律(v2 §0 修订4):**不与 LocalFlow 强行统一内部字段**。LocalFlow 侧是
环境执行轨迹(动作/验证/回滚),本侧是业务决策轨迹(节点/结论/路由)——两者只共享
一层公共元数据(origin / run_id / n_steps / steps / outcome / provenance /
schema_version),供跨项目的数据卡与统计汇总消费。

两个入口:
- ``adapt_trace_file(path)``  — 消费 observability 落盘的 ``logs/traces/<id>.jsonl``;
- ``adapt_state(final)``      — 消费 ``run_career_flow_routed`` 返回的终态 dict。

同样带脱敏(独立实现,不跨仓库 import):密钥模式 + 本机用户路径。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

COMMON_SCHEMA_VERSION = 1
_REDACTED = "[REDACTED]"
_SECRETS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
]
_USERS = re.compile(r"/Users/[A-Za-z0-9._\-]+")


def _scrub(text: str) -> str:
    for pat in _SECRETS:
        text = pat.sub(_REDACTED, text)
    return _USERS.sub("~", text)


def _scrub_obj(obj):
    if isinstance(obj, str):
        return _scrub(obj)
    if isinstance(obj, list):
        return [_scrub_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _scrub_obj(v) for k, v in obj.items()}
    return obj


def _common(run_id: str, steps: list[dict], outcome: dict) -> dict:
    return _scrub_obj({
        "schema_version": COMMON_SCHEMA_VERSION,
        "origin": "offerclaw",
        "task_kind": "career_flow",
        "run_id": run_id,
        "n_steps": len(steps),
        "steps": steps,
        "outcome": outcome,
        "provenance": "runs",
    })


def adapt_trace_file(path: str | Path) -> dict:
    """observability JSONL → 公共元数据记录。"""
    steps, outcome, run_id = [], {}, Path(path).stem
    for line in Path(path).open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        ev = e.get("event")
        if ev == "start":
            outcome["jd_title"] = e.get("jd_title", "")
            outcome["skip_llm"] = bool(e.get("skip_llm", True))
        elif ev == "node":
            steps.append({"index": e.get("seq", len(steps)),
                          "node": e.get("node", "?"),
                          "detail": str(e.get("action", "")),
                          "source": str(e.get("source", "")),
                          "ts": str(e.get("ts", ""))})
        elif ev == "end":
            outcome["status"] = e.get("status", "")
    return _common(run_id, steps, outcome)


def adapt_state(final: dict, *, run_id: str = "state") -> dict:
    """CareerFlow 终态 dict → 公共元数据记录。"""
    steps = [{"index": i, "node": t.get("node", "?"),
              "detail": str(t.get("action", "")), "source": str(t.get("source", "")),
              "ts": str(t.get("ts", ""))}
             for i, t in enumerate(final.get("trace") or [])]
    outcome = {
        "route_taken": final.get("route_taken", ""),
        "conclusion": (final.get("match_report") or {}).get("status", ""),
        "errors": len(final.get("errors") or []),
        "confirm_pending": len(final.get("requires_confirmation") or []),
    }
    return _common(run_id, steps, outcome)


def export_all(traces_dir: str | Path = "logs/traces", out: str | Path | None = None) -> list[dict]:
    """批量导出目录下全部 trace 文件;可选写出 JSONL。"""
    records = [adapt_trace_file(p) for p in sorted(Path(traces_dir).glob("*.jsonl"))]
    if out is not None:
        with Path(out).open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


if __name__ == "__main__":
    recs = export_all()
    print(f"traces adapted: {len(recs)}")
    for r in recs[:3]:
        print(f"  {r['run_id']}: steps={r['n_steps']} outcome={r['outcome']}")
