from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from eval_private_sets import (PrivateEvalIntegrityError, PrivateEvalUnavailable,
                               SHA256_RE, load_private_eval, sha256_file)


BASE = Path(__file__).resolve().parents[1]
FIXTURES = BASE / "tests" / "fixtures"


def _write_manifest(path: Path, *, digest: str | None, count: int = 1) -> None:
    path.write_text(json.dumps({
        "dataset_id": "private-test-v1",
        "dataset_file": "private.json",
        "expected_count": count,
        "sha256": digest,
    }), encoding="utf-8")


def test_private_bundle_requires_root_and_pinned_sha(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, digest=None)
    monkeypatch.delenv("OFFERCLAW_PRIVATE_EVAL_ROOT", raising=False)
    with pytest.raises(PrivateEvalUnavailable, match="SHA-256 is unset"):
        load_private_eval(manifest)


def test_private_bundle_validates_sha_count_and_ids(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    dataset = root / "private.json"
    dataset.write_text(json.dumps({
        "version": "private-test-v1",
        "items": [{"id": "case-1", "secret_question": "not reported"}],
    }), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, digest=sha256_file(dataset))
    bundle = load_private_eval(manifest, private_root=root)
    assert bundle.expected_count == 1
    assert bundle.sha256 == sha256_file(dataset)

    _write_manifest(manifest, digest="0" * 64)
    with pytest.raises(PrivateEvalIntegrityError, match="SHA-256 mismatch"):
        load_private_eval(manifest, private_root=root)


def test_repository_private_manifests_do_not_contain_blind_items():
    for name, expected_count in (
        ("router_blind_v1.manifest.json", 240),
        ("jd_blind_v1.manifest.json", 120),
        ("retrieval_blind_v1.manifest.json", 100),
    ):
        manifest = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert manifest["expected_count"] == expected_count
        assert manifest["sha256"] is None or SHA256_RE.fullmatch(manifest["sha256"])
        assert "items" not in manifest and "questions" not in manifest


@pytest.mark.parametrize("script", ["eval_intelligent_router.py", "eval_jd_analysis.py"])
def test_missing_private_eval_is_explicit_nonpassing_skip(monkeypatch, script):
    monkeypatch.delenv("OFFERCLAW_PRIVATE_EVAL_ROOT", raising=False)
    completed = subprocess.run(
        [sys.executable, str(BASE / script), "--set", "blind"],
        cwd=BASE, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "skipped"
    assert report["passed"] is False
    assert "failures" not in report


def test_router_gold_accepts_alternative_plan_and_rejects_forbidden_route():
    from eval_intelligent_router import _gold_score, _row_from_item

    row = _row_from_item({
        "id": "blind-1", "question": "secret",
        "required_routes": ["application_state.list_current"],
        "allowed_optional_routes": ["application_jd.get_bound_jd"],
        "forbidden_routes": ["general_fallback.answer"],
        "acceptable_plans": [
            {"required_routes": ["project_memory.rank_for_application"],
             "allowed_optional_routes": ["application_jd.get_bound_jd"]},
        ],
        "source_roles": {
            "application_state.list_current": "answer_source",
            "project_memory.rank_for_application": "supporting_context",
        },
    })
    roles = [
        "application_state.list_current=answer_source",
        "project_memory.rank_for_application=supporting_context",
    ]
    good = _gold_score(
        row,
        {"application_state.list_current", "project_memory.rank_for_application"},
        actual_roles=roles, actual_dependencies=[],
    )
    assert good["matched"] is True
    bad = _gold_score(
        row,
        {"application_state.list_current", "project_memory.rank_for_application",
         "general_fallback.answer"},
        actual_roles=roles, actual_dependencies=[],
    )
    assert bad["matched"] is False
    assert bad["forbidden_hits"] == ["general_fallback.answer"]


def test_full_jd_gold_matching_is_one_to_one_and_tracks_exact_spans():
    from eval_jd_analysis import _match_requirements

    text = "Required: Python. Preferred: Go."
    gold = [
        {"text": "Required: Python.", "start": 0, "end": 17},
        {"text": "Preferred: Go.", "start": 18, "end": 32},
    ]
    predicted = [
        SimpleNamespace(
            text="Required: Python.",
            evidence_spans=[SimpleNamespace(start=0, end=17, text="Required: Python.")],
        ),
        SimpleNamespace(
            text="Preferred: Go.",
            evidence_spans=[SimpleNamespace(start=18, end=32, text="Preferred: Go.")],
        ),
    ]
    assert text[0:17] == gold[0]["text"] and text[18:32] == gold[1]["text"]
    matches = _match_requirements(gold, predicted)
    assert len(matches) == 2
    assert all(match["exact_span"] for match in matches)
