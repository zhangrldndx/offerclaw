import json
from pathlib import Path

import pytest

from scripts.run_product_acceptance import (
    AcceptanceContractError,
    ROOT,
    load_manifest,
)


def test_checked_in_product_acceptance_has_six_balanced_areas():
    payload = load_manifest(ROOT / "tests" / "product_acceptance_v2.json")
    assert len(payload["cases"]) == 36
    assert {row["area"] for row in payload["cases"]} == {
        "profile_governance", "jd_match", "plan_today",
        "application_lifecycle", "reflection_memory", "resume_and_flow",
    }


def test_acceptance_manifest_rejects_duplicate_nodeid(tmp_path: Path):
    payload = json.loads(
        (ROOT / "tests" / "product_acceptance_v2.json").read_text(encoding="utf-8")
    )
    payload["cases"][1]["nodeid"] = payload["cases"][0]["nodeid"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AcceptanceContractError, match="unique"):
        load_manifest(path)


def test_acceptance_manifest_rejects_external_llm(tmp_path: Path):
    payload = json.loads(
        (ROOT / "tests" / "product_acceptance_v2.json").read_text(encoding="utf-8")
    )
    payload["isolation"]["external_llm"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AcceptanceContractError, match="offline"):
        load_manifest(path)
