import json
from pathlib import Path

from eval_intelligent_router import _rows


def test_intelligent_frozen_set_is_240_independent_questions():
    rows = _rows()
    assert len(rows) == 240
    assert len({row["question"] for row in rows}) == 240
    assert all(row["expected"] for row in rows)


def test_reviewed_router_set_is_explicitly_development_only():
    path = Path(__file__).parent / "fixtures" / "intelligent_router_frozen_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["version"] == "router_reviewed_regression_v1"
    assert manifest["set_kind"] == "reviewed_regression_dev"
    assert "not" in manifest["note"].lower() and "blind" in manifest["note"].lower()


def test_surface_invariance_set_is_480_wrapper_cases():
    rows = _rows("surface_invariance")
    assert len(rows) == 480
    # This suite deliberately applies ten fixed wrappers to 48 semantic seeds;
    # it measures surface invariance rather than claiming 480 independent intents.
    assert len({row["id"] for row in rows}) == 480
