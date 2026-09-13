# -*- coding: utf-8 -*-
"""Probe the authenticated WeChat query path without printing message content."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    question = sys.stdin.readline().strip()
    if not question:
        raise SystemExit("one question is required on stdin")
    opaque = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    from wechat_data_bridge import REQUEST_SCHEMA, handle

    started = time.perf_counter()
    result = handle({
        "schema_version": REQUEST_SCHEMA,
        "operation": "query.answer",
        "payload": {
            "question": question,
            "conversation_id": "conv_" + opaque,
            "message_id": "msg_" + opaque,
            "operation_id": "op_" + opaque,
            "top_k": 5,
        },
    })
    answer = str(result.get("answer") or "")
    routes = result.get("routes") or []
    safe = {
        "status": result.get("status"),
        "service_mode": result.get("service_mode"),
        "mode": result.get("mode"),
        "answer_action": result.get("answer_action"),
        "route_operations": [
            f"{item.get('source', '')}.{item.get('operation', '')}"
            for item in routes if isinstance(item, dict)
        ],
        "source_count": len(result.get("sources") or []),
        "answer_chars": len(answer),
        "part_lengths": [len(part) for part in result.get("reply_parts") or []],
        "offerclaw_calls": (result.get("model_usage") or {}).get("offerclaw_calls", 0),
        "openclaw_calls": (result.get("model_usage") or {}).get("openclaw_calls", 0),
        "data_version": str(result.get("data_version") or "")[:16],
        "unavailable_claim": "不能访问本地文件" in answer,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    print(json.dumps(safe, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
