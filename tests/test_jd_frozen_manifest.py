import json
from pathlib import Path

from eval_jd_analysis import load_cases


def test_jd_frozen_set_has_120_unique_ground_truth_documents():
    cases = load_cases()
    assert len(cases) == 120
    assert len({case["jd_text"] for case in cases}) == 120
    assert all(case["evidence"] in case["jd_text"] for case in cases)
    assert all(case["requirements"][0]["text"] in case["jd_text"] for case in cases)


def test_jd_set_is_explicitly_a_phenomena_development_set():
    path = Path(__file__).parent / "fixtures" / "jd_analysis_frozen_v1.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["version"] == "jd_phenomena_dev_v1"
    assert fixture["set_kind"] == "phenomena_development"
    assert "not" in fixture["note"].lower() and "blind" in fixture["note"].lower()
