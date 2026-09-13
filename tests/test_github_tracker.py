# -*- coding: utf-8 -*-
"""test_github_tracker.py — P3：GitHub 留痕抓取的纯解析逻辑离线单测。

只测 parse_push_events（不发网络请求）。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import github_tracker as gt


def _push(repo, date, msgs):
    return {
        "type": "PushEvent",
        "created_at": f"{date}T10:00:00Z",
        "repo": {"name": f"u/{repo}"},
        "payload": {"commits": [{"message": m} for m in msgs]},
    }


def test_parse_push_extracts_commit_first_lines():
    events = [_push("offerclaw", "2026-06-03", ["feat: 加 RAG 门槛\n\n详情..."])]
    items = gt.parse_push_events(events, "2026-06-03")
    assert items == ["offerclaw: feat: 加 RAG 门槛"]


def test_parse_push_date_filter():
    events = [
        _push("a", "2026-06-03", ["today work"]),
        _push("b", "2026-06-01", ["old work"]),
    ]
    items = gt.parse_push_events(events, "2026-06-03")
    assert items == ["a: today work"]


def test_parse_push_dedup():
    events = [
        _push("a", "2026-06-03", ["fix bug", "fix bug"]),
    ]
    items = gt.parse_push_events(events, "2026-06-03")
    assert items == ["a: fix bug"]


def test_parse_create_event():
    events = [{
        "type": "CreateEvent",
        "created_at": "2026-06-03T09:00:00Z",
        "repo": {"name": "u/newproj"},
        "payload": {"ref_type": "repository"},
    }]
    items = gt.parse_push_events(events, "2026-06-03")
    assert items == ["newproj: 新建仓库"]


def test_parse_no_date_filter_returns_all():
    events = [_push("a", "2026-06-03", ["x"]), _push("b", "2026-06-01", ["y"])]
    items = gt.parse_push_events(events, None)
    assert len(items) == 2


def test_parse_empty_events():
    assert gt.parse_push_events([], "2026-06-03") == []
    assert gt.parse_push_events(None, None) == []


def test_parse_ignores_other_event_types():
    events = [{"type": "WatchEvent", "created_at": "2026-06-03T10:00:00Z",
               "repo": {"name": "u/x"}, "payload": {}}]
    assert gt.parse_push_events(events, "2026-06-03") == []
