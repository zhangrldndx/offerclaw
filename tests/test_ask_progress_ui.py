# -*- coding: utf-8 -*-
"""Top ask UI must expose real SSE stages rather than a static loading label."""

from pathlib import Path


INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


def test_top_ask_has_accessible_stage_progress_panel():
    source = INDEX.read_text(encoding="utf-8")
    assert 'id="askProgress"' in source
    assert 'role="status"' in source
    assert 'aria-live="polite"' in source
    assert 'id="askProgressStages"' in source
    assert "ask-progress-spinner" in source


def test_top_ask_consumes_real_stage_events_and_tracks_elapsed_time():
    source = INDEX.read_text(encoding="utf-8")
    assert "onEvent: ev => updateAskProgress(ev)" in source
    assert "ev.type !== 'stage'" in source
    assert "performance.now() - askProgressStartedAt" in source
    assert "理解问题与规划路径" in source
    assert "读取所需数据源" in source
    assert "校验证据与事实边界" in source
    assert "组织最终回答" in source
