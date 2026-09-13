# -*- coding: utf-8 -*-
"""P0-c 单测:OfferClaw 轨迹 → 公共元数据 Adapter(契约 + 脱敏 + 两入口)。"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import traj_adapter as ta

_COMMON_KEYS = {"schema_version", "origin", "task_kind", "run_id",
                "n_steps", "steps", "outcome", "provenance"}


def _write_trace(tmp_path, name="t1.jsonl"):
    lines = [
        {"trace_id": "t1", "event": "start", "jd_title": "测试JD", "skip_llm": True},
        {"trace_id": "t1", "event": "node", "seq": 0, "node": "profile",
         "action": "loaded api_key=SECRETSECRET123 fields",
         "source": "/" + "Users/test-user/user_profile.md",
         "ts": "10:00:00"},
        {"trace_id": "t1", "event": "node", "seq": 1, "node": "match",
         "action": "conclusion=当前适合投递", "source": "match_job.run_match", "ts": "10:00:01"},
    ]
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return p


def test_adapt_trace_file_contract_and_scrub(tmp_path):
    rec = ta.adapt_trace_file(_write_trace(tmp_path))
    assert set(rec.keys()) == _COMMON_KEYS
    assert rec["origin"] == "offerclaw" and rec["schema_version"] == ta.COMMON_SCHEMA_VERSION
    assert rec["n_steps"] == 2 and rec["steps"][1]["node"] == "match"
    blob = json.dumps(rec, ensure_ascii=False)
    assert "SECRETSECRET" not in blob and ta._REDACTED in blob   # 密钥被脱敏
    assert "/Users/" not in blob                                  # 本机路径被规约


def test_adapt_state_contract():
    final = {
        "trace": [{"node": "profile", "action": "loaded", "source": "x", "ts": "1"},
                  {"node": "match", "action": "conclusion=当前适合投递", "source": "y", "ts": "2"}],
        "route_taken": "suitable:full_path",
        "match_report": {"status": "当前适合投递"},
        "errors": [], "requires_confirmation": [{"a": 1}],
    }
    rec = ta.adapt_state(final, run_id="r1")
    assert set(rec.keys()) == _COMMON_KEYS
    assert rec["run_id"] == "r1" and rec["n_steps"] == 2
    assert rec["outcome"]["conclusion"] == "当前适合投递"
    assert rec["outcome"]["confirm_pending"] == 1


def test_export_all_writes_jsonl(tmp_path):
    _write_trace(tmp_path, "a.jsonl")
    _write_trace(tmp_path, "b.jsonl")
    out = tmp_path / "out.jsonl"
    recs = ta.export_all(tmp_path, out=out)
    assert len(recs) == 2
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2
