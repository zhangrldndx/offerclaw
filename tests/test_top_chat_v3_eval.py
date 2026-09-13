# -*- coding: utf-8 -*-
from __future__ import annotations

import json


def _row(**updates):
    row = {
        "contract_ok": True,
        "action_ok": True,
        "expected_capability": True,
        "expected_service": "guide",
        "actual_service": "guide",
        "expected_decision": "answer",
        "actual_decision": "answer",
        "write_command_recall": False,
        "timeout": False,
    }
    row.update(updates)
    return row


def test_private_manifest_encodes_the_approved_release_cohorts():
    from eval_top_chat_v3 import MANIFEST

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["expected_count"] == 220
    assert manifest["quotas"] == {
        "action_query_contrast": 120,
        "multi_turn": 60,
        "ambiguity": 40,
    }
    assert manifest["sha256"] is None
    assert manifest["status"] == "not_provisioned"


def test_release_summary_enforces_all_gates():
    from eval_top_chat_v3 import summarize

    thresholds = {
        "action_capability_accuracy": 0.98,
        "guide_recall_confusion_rate_lt": 0.01,
        "write_command_recall_count": 0,
        "clarify_precision": 0.95,
        "clarify_recall": 0.95,
        "timeout_rate_lte": 0.01,
        "planning_p95_ms_lte": 5000,
        "business_or_memory_event_writes": 0,
    }
    rows = [_row() for _ in range(120)]
    rows += [_row(
        expected_capability=False, expected_service="recall",
        actual_service="recall",
    ) for _ in range(60)]
    rows += [_row(
        expected_capability=False, expected_service="recall",
        actual_service="recall", expected_decision="clarify",
        actual_decision="clarify",
    ) for _ in range(40)]
    passing = summarize(rows, [4999.0] * 220, writes=0, thresholds=thresholds)
    assert passing["passed"] is True

    rows[0] = _row(write_command_recall=True)
    failing = summarize(rows, [4999.0] * 220, writes=0, thresholds=thresholds)
    assert failing["passed"] is False
    assert failing["metrics"]["write_command_recall_count"] == 1


def test_unprovisioned_private_set_can_never_report_passed():
    import pytest
    from eval_private_sets import PrivateEvalUnavailable, load_private_eval
    from eval_top_chat_v3 import MANIFEST

    with pytest.raises(PrivateEvalUnavailable, match="not provisioned"):
        load_private_eval(MANIFEST)
