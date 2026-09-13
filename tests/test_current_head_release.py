from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import freeze_current_head_release as freeze
from scripts import run_current_head_regression as runner


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.private_artifact
def test_historical_final_v4_cannot_be_overwritten(tmp_path: Path) -> None:
    historical = freeze.HISTORICAL_FINAL_V4 / "FROZEN_CONFIG.json"
    before = _file_sha(historical)
    with pytest.raises(freeze.ReleaseFreezeError, match="may not overwrite"):
        freeze.write_manifest(historical, freeze.DEFAULT_DATASET)
    assert _file_sha(historical) == before


def test_frozen_environment_resolves_real_production_profile() -> None:
    assert freeze.FROZEN_ENV["RAG_ANSWERABILITY_MODE"] == "teacher"
    assert freeze.FROZEN_ENV["RAG_ANSWERABILITY_PROMPT"] == "v5"
    assert freeze.FROZEN_ENV["RAG_ANSWERABILITY_DEPTH"] == "12"
    assert freeze.FROZEN_ENV["RAG_ANSWERABILITY_EARLY_EXIT"] == "1"
    assert freeze.FROZEN_ENV["RAG_ANSWERABILITY_GATE"] == "1"
    assert freeze.FROZEN_ENV["RAG_ANSWERABILITY_GATE_VOTES"] == "3"

    with freeze.frozen_environment():
        from rag_answerability import active_prompt, mode
        from rag_gate import (
            _answerability_depth,
            _answerability_early_exit,
            _answerability_gate,
            _answerability_gate_votes,
            _evidence_gate,
        )
        from rag_retrieval_trace import resolve_retrieval_profile

        profile = resolve_retrieval_profile("baseline")
        assert profile.name == "baseline"
        assert profile.pool_size == 28
        assert profile.reranker_prefix_mode == "compact32"
        assert profile.enable_hyde_channel is True
        assert profile.enable_hyde_bm25_channel is True
        assert "enable_jd_channel" not in profile.to_dict()
        assert "RAG_JD_CHANNEL" not in freeze.FROZEN_ENV
        assert mode() == "teacher"
        assert active_prompt()[1] == (
            "ae846212637e4b34b954c9dbe80fe30dd9ad5b1578948fcf63f66641ec84a12c"
        )
        assert _answerability_depth() == 12
        assert _answerability_early_exit() is True
        assert _answerability_gate() is True
        assert _answerability_gate_votes() == 3
        # Regression: an empty frozen numeric value reaches float("") in the
        # real retrieval path even though the manifest collector used ``or``.
        assert _evidence_gate(True, 0.90, best=0.60, strong=0.73) is True
        assert _evidence_gate(False, 0.96, best=0.75, strong=0.73) is True
        assert freeze.FROZEN_ENV["RAG_RERANK_GATE_MIN"] == "0.85"
        assert freeze.FROZEN_ENV["RAG_RERANK_RESCUE_DIST"] == "0.80"
        assert freeze.FROZEN_ENV["RAG_RERANK_RESCUE_MIN"] == "0.95"


def test_qrels_rebind_is_memory_only_and_labels_are_unchanged() -> None:
    payload = {
        "index": {"fingerprint": "old"},
        "items": [{
            "query_id": "q1",
            "anchor_id": "a1",
            "split": "blind",
            "case_kind": "positive",
            "expected_behavior": "answer",
            "answer_requirements": ["fact"],
            "relevant_targets": [{"chunk_id": "c1", "relevance_grade": 3}],
            "hard_negatives": [],
            "review_status": "approved",
            "question": "private question",
        }],
    }
    original = copy.deepcopy(payload)
    label_sha = freeze.label_projection_sha256(payload)
    target = {
        "collection": "collection",
        "count": 1,
        "fingerprint": "sha256:" + "1" * 64,
    }
    rebound = runner.rebind_qrels_in_memory(
        payload,
        dataset_contract={
            "label_projection_sha256": label_sha,
            "target_index": target,
        },
    )
    assert payload == original
    assert rebound is not payload
    assert rebound["index"] == target
    assert freeze.label_projection_sha256(rebound) == label_sha
    assert rebound["items"] == payload["items"]


def test_rebind_refuses_any_label_change() -> None:
    payload = {"index": {}, "items": [{"query_id": "q1", "case_kind": "negative"}]}
    with pytest.raises(runner.RegressionRunError, match="label projection"):
        runner.rebind_qrels_in_memory(
            payload,
            dataset_contract={
                "label_projection_sha256": "0" * 64,
                "target_index": {},
            },
        )


def test_stable_index_contract_includes_content_hash() -> None:
    from rag_qrels_v2 import index_contract_fingerprint

    payload = freeze.stable_index_payload({
        "collection": "c",
        "collection_count": 3,
        "collection_content_hash": "content-v1",
        "fingerprint_id": "id",
        "embedding_provider": "local",
        "embedding_model": "model",
        "embedding_dimensions": 768,
        "chunker_version": "v1",
        "indexed_chunker_versions": ["v1"],
    })
    assert payload["content_hash"] == "content-v1"
    assert payload["collection_content_hash"] == "content-v1"
    assert payload["fingerprint"] == index_contract_fingerprint(payload)


def test_verify_compares_git_dirty_index_profile_config_dataset_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    state = {
        "git": {"head": "abc", "tree": "tree", "tracked_runtime_dirty": []},
        "protocol": {"profile_source": "production_registry"},
        "environment": {"RAG_ANSWERABILITY_MODE": "teacher"},
        "retrieval_profile": {"name": "baseline", "enable_hyde_channel": True},
        "answerability": {"prompt": "v5", "effective_depth": 12},
        "gate": {"votes": 3},
        "generation": {"hyde_model": "model"},
        "index": {"fingerprint_id": "index"},
        "dataset": {"path": str(dataset), "sha256": "dataset"},
        "files": {"runtime": {"rag_gate.py": {"sha256": "code"}}},
    }
    manifest = {
        "schema_version": "offerclaw-current-head-release-v1",
        "created_at": "ignored",
        **state,
    }
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(freeze, "collect_release_state", lambda _path: copy.deepcopy(state))
    assert freeze.verify_manifest(manifest_path)["status"] == "ok"

    drifted = copy.deepcopy(state)
    drifted["git"]["head"] = "changed"
    drifted["git"]["tracked_runtime_dirty"] = [" M rag_gate.py"]
    drifted["index"]["fingerprint_id"] = "changed-index"
    drifted["retrieval_profile"]["enable_hyde_channel"] = False
    drifted["gate"]["votes"] = 1
    drifted["dataset"]["sha256"] = "changed-data"
    drifted["files"]["runtime"]["rag_gate.py"]["sha256"] = "changed-code"
    monkeypatch.setattr(freeze, "collect_release_state", lambda _path: drifted)
    with pytest.raises(freeze.ReleaseFreezeError) as exc:
        freeze.verify_manifest(manifest_path)
    message = str(exc.value)
    for section in ("git", "index", "retrieval_profile", "gate", "dataset", "files"):
        assert section in message


def test_raw_output_must_be_outside_repo_and_summary_cannot_touch_final_v4(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.RegressionRunError, match="outside the repository"):
        runner.ensure_private_output(freeze.ROOT / "raw.json")
    runner.ensure_private_output(tmp_path / "raw.json")
    with pytest.raises(runner.RegressionRunError, match="may not overwrite"):
        runner.ensure_summary_output(freeze.HISTORICAL_FINAL_V4 / "SUMMARY.json")


def test_regression_uses_a_fresh_external_answerability_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_answerability

    original_path = rag_answerability.CACHE_PATH
    original_cache = rag_answerability._CACHE
    output = tmp_path / "run1.json"
    path = runner.install_private_answerability_cache(output)
    assert path == tmp_path / "run1.answerability_cache.json"
    assert rag_answerability.CACHE_PATH == path
    assert rag_answerability._CACHE is None
    monkeypatch.setattr(rag_answerability, "CACHE_PATH", original_path)
    monkeypatch.setattr(rag_answerability, "_CACHE", original_cache)

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.RegressionRunError, match="orphaned evaluation cache"):
        runner.install_private_answerability_cache(output)


def test_public_summary_has_aggregates_and_no_raw_rows_or_text(tmp_path: Path) -> None:
    raw = tmp_path / "run1.json"
    raw.write_text("private", encoding="utf-8")
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text("{}", encoding="utf-8")
    evaluation = {
        "positive": {
            "metrics": {
                "strict_ranking": {
                    "recall@1": {"hits": 1},
                    "recall@3": {"hits": 1},
                    "recall@5": {"hits": 1},
                },
                "funnel": {
                    "rrf_candidate": {"hits": 1},
                    "correct_top1_gate_pass": {"hits": 1},
                    "effective_evidence": {"hits": 1},
                },
            },
            "rows": [{
                "query_id": "q1",
                "reciprocal_rank_at_10": 1.0,
                "ndcg_at_5": 1.0,
                "latency_ms": 10.0,
            }],
        },
        "negative": {"metrics": {}, "rows": []},
    }
    artifact = {
        "run": 1,
        "sentinel": {"calls": 2},
        "evaluation": evaluation,
    }
    manifest = {
        "git": {"head": "abc"},
        "index": {"fingerprint_id": "idx"},
        "dataset": {
            "path": "dataset.json",
            "sha256": "data",
            "label_projection_sha256": "labels",
        },
        "retrieval_profile": {"name": "baseline"},
        "answerability": {"prompt": "v5"},
        "gate": {"votes": 3},
    }
    summary = runner.build_summary(
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=[artifact],
        raw_paths=[raw],
        dataset_payload={"items": []},
        max_cases=0,
    )
    encoded = json.dumps(summary, ensure_ascii=False)
    assert summary["privacy"] == {
        "contains_raw_rows": False,
        "contains_questions": False,
        "contains_chunk_text": False,
    }
    assert "evaluation" not in encoded
    assert "private question" not in encoded
    assert summary["summary"]["r1"]["median"] == 1.0
    assert summary["private_artifacts"][0]["sha256"] == _file_sha(raw)


def test_answerability_sentinel_rejects_silent_degradation() -> None:
    evaluation = {
        "positive": {"rows": [{
            "gate_features": {"answerability_rerank": {
                "applied": True, "calls": 2, "graded": 1,
            }},
        }]},
        "negative": {"rows": []},
    }
    with pytest.raises(runner.RegressionRunError, match="degraded"):
        runner._answerability_sentinel(evaluation)


def test_smoke_raw_cannot_be_reused_as_a_full_run(tmp_path: Path) -> None:
    raw = tmp_path / "run1.json"
    raw.write_text(json.dumps({
        "schema_version": "offerclaw-current-head-regression-raw-v1",
        "manifest_sha256": "manifest",
        "run": 1,
        "max_cases": 2,
        "evaluation": {"positive": {"rows": []}, "negative": {"rows": []}},
    }), encoding="utf-8")
    with pytest.raises(runner.RegressionRunError, match="another max-cases"):
        runner._load_existing_raw(raw, "manifest", 1, 0)
