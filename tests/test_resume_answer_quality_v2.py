from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import resume_answer_quality_v2 as resume


def test_generation_projection_only_clears_judges():
    rows = [{"eval_id": "q1", "answer": "a", "judges": {"j": {"available": True}}}]
    projected = resume._generation_rows(rows)
    assert projected == [{"eval_id": "q1", "answer": "a", "judges": {}}]
    assert rows[0]["judges"]


def test_validate_partial_rejects_incomplete_generation(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "raw_rows.jsonl").write_text(
        json.dumps({"eval_id": "q1", "generation": {"available": True}}) + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(resume.aq, "_read_json", lambda _path: {})
    monkeypatch.setattr(
        resume.aq,
        "resolve_selection",
        lambda _manifest: [{"eval_id": "q1"}, {"eval_id": "q2"}],
    )
    monkeypatch.setattr(resume.aq, "ensure_repository_external", lambda path: path)
    with pytest.raises(resume.ResumeError, match="generation is incomplete"):
        resume.validate_partial(run_dir, selection)
