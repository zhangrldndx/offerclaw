# -*- coding: utf-8 -*-
"""Seed one evidence-grounded pending suggestion in the isolated WeChat lab."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
EXPECTED_APP_DIR = Path.home() / ".local" / "share" / "offerclaw-lab" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from memory_layers import EpisodicMemory
from profile_review import ProfileRepository


def _caller(payload):
    def call(messages, max_tokens, temperature, model):
        return json.dumps(payload, ensure_ascii=False)
    return call


def main() -> None:
    if APP_DIR != EXPECTED_APP_DIR:
        raise SystemExit(
            f"Refusing to seed outside the isolated lab: {APP_DIR} != {EXPECTED_APP_DIR}"
        )

    repo = ProfileRepository()
    skill = next(
        item for item in repo.current()["profile_spec"]["skills"]
        if item["name"] == "Python 工程" and item["source_section_id"] == "3"
    )
    field_path = f"/skills/{skill['assessment_id']}/level"
    existing = next(
        (item for item in repo.list_suggestions("pending")
         if item.get("field_path") == field_path),
        None,
    )
    if existing:
        print(json.dumps({
            "status": "ok", "created": False,
            "suggestion_id": existing["suggestion_id"],
        }, ensure_ascii=False, indent=2))
        return

    episodic = EpisodicMemory()
    evidence_ids = []
    rows = [
        ("2026-09-10", "完成 Python 检索接口回归测试，12 条合成用例全部通过"),
        ("2026-09-11", "完成 Python 状态刷新故障测试，旧向量块完整保留"),
    ]
    for index, (day, quote) in enumerate(rows, 1):
        event = episodic.append({
            "kind": "daily_log_recorded", "actor": "user", "source": "daily_log",
            "business_date": day, "log_id": f"wechat-lab-evidence-{index}",
            "date": day, "status": "done", "notes": quote,
            "occurred_at": dt.datetime.fromisoformat(f"{day}T20:00:00+08:00").isoformat(),
        }, export=False, index=False)
        candidate = {
            "claim": "完成了可验证的 Python 工程测试",
            "source_event_id": event["event_id"], "source_quote": quote,
            "capability_id": skill["capability_id"],
            "activity_level": "delivered", "scope": "wechat-lab-synthetic",
            "verification_candidate": "test_result", "criteria_ids": [],
            "explicit_statement": False, "relation": "direct",
            "rationale": "微信实验合成证据，不对应真实经历",
        }
        evidence = repo.ingest_candidates([candidate])["items"][0]
        evidence_ids.append(evidence["evidence_id"])

    payload = {"changes": [{
        "field_path": field_path,
        "current_value": skill["level"],
        "observed_change": "两次独立的合成测试记录支持提高一级",
        "operation": "replace", "proposed_value": min(skill["level"] + 1, 5),
        "evidence_ids": evidence_ids, "counter_evidence_ids": [],
        "capability_id": skill["capability_id"], "new_capability": None,
        "rationale": "仅用于微信审批闭环实验，不代表真实能力变化",
        "requirement_status": "satisfied",
    }]}
    audit = repo.run_audit(
        caller=_caller(payload), model="synthetic-wechat-lab", extract_sources=False,
    )
    suggestion = next(
        item for item in repo.list_suggestions("pending")
        if item.get("field_path") == field_path
    )
    print(json.dumps({
        "status": "ok", "created": True, "audit_status": audit.get("status"),
        "suggestion_id": suggestion["suggestion_id"],
        "base_revision": suggestion["base_revision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
