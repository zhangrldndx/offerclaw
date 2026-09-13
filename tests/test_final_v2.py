# -*- coding: utf-8 -*-
"""Guards for the Final v2 one-shot protocol.

Final v2 can only be spent once, so the failure modes worth pinning are the ones
that would waste it silently: running a configuration that has drifted from the
freeze, scoring labels a human has not confirmed, and reading a run in which the
judge quietly did nothing.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FREEZE = ROOT / "docs" / "rag_eval" / "final_v2" / "FROZEN_CONFIG.json"


def _load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_final_v2", ROOT / "scripts" / "run_final_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.private_artifact
def test_the_freeze_pins_more_than_the_knobs():
    """Three knobs this round turned out to be inert while reading as set, so a
    freeze that records only env values would certify a configuration that never
    ran."""
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["arm"] == "compact32_pool28"
    assert frozen["judge"]["effective_depth"] == 12
    assert frozen["judge"]["prompt_sha256"]          # judge behaviour, not just its name
    assert frozen["index"]["content_hash"]           # the corpus it was measured on
    assert frozen["code"]["file_sha256"]             # the code that implements the path
    for name in ("rag_answerability.py", "rag_gate.py", "rag_colloquial_profiles.py"):
        assert name in frozen["code"]["file_sha256"]


@pytest.mark.private_artifact
def test_every_rejected_knob_is_frozen_off():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    for knob in ("RAG_ANSWERABILITY_GATE", "RAG_ANSWERABILITY_TIEBREAK",
                 "RAG_HYDE", "RAG_QUERY_REWRITE", "RAG_ANSWERABILITY_MODE"):
        assert frozen["env"][knob] == "", f"{knob} 必须冻结为关"
    assert frozen["retrieval_profile"]["enable_hyde"] is False
    assert frozen["retrieval_profile"]["pool_size"] == 28


@pytest.mark.private_artifact
def test_verify_detects_a_changed_source_file(tmp_path, monkeypatch):
    """A freeze nobody can fail is decoration.

    Deliberately does not assume the working tree still matches: the freeze
    records the code as it stood when Final v2 was scored, and development
    continues afterwards.  What must keep working is the *detection*.
    """
    saved = json.loads(FREEZE.read_text(encoding="utf-8"))
    tampered = json.loads(FREEZE.read_text(encoding="utf-8"))
    tampered["code"]["file_sha256"]["rag_gate.py"] = "0" * 64
    FREEZE.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "freeze_final_v2_config.py"), "--verify"],
            cwd=ROOT, capture_output=True, text=True)
        assert result.returncode != 0
        assert "rag_gate.py" in result.stdout
    finally:
        FREEZE.write_text(json.dumps(saved, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def test_the_runner_refuses_an_unconfirmed_dataset(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"status": "draft_pending_human_review", "items": []}),
                     encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_final_v2.py"),
         "--dataset", str(draft)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "需人工确认" in combined
    # the label check must come first: it is true whatever the code looks like
    assert "冻结件与工作树不一致" not in combined


def test_arms_cover_the_four_preregistered_configurations():
    runner = _load_runner()
    assert set(runner.ARMS) == {"A1", "A2", "B", "C"}
    assert runner.ARMS["A1"][0] == "baseline"          # production default: prefix none
    assert runner.ARMS["A2"][0] == "compact32"         # this round's own baseline
    assert runner.ARMS["A1"][2] is False and runner.ARMS["A2"][2] is False
    assert runner.ARMS["B"][2] is True and runner.ARMS["C"][2] is True
    assert runner.ARMS["C"][1]["RAG_HYDE"] == "1"
    assert "RAG_HYDE" not in runner.ARMS["B"][1]
    for knob in ("RAG_ANSWERABILITY_GATE", "RAG_ANSWERABILITY_TIEBREAK"):
        assert knob in runner.FORCE_OFF


def _run_file(tmp_path, diags, negatives=0):
    rows = [{"gate_features": {"answerability_rerank": d}} for d in diags]
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"runs": [{"positive": {"rows": rows},
                                          "negative": {"rows": []}}]}), encoding="utf-8")
    return path


def test_sentinel_rejects_a_judge_arm_that_did_nothing(tmp_path):
    runner = _load_runner()
    path = _run_file(tmp_path, [{"applied": True, "calls": 12, "graded": 12},
                                {"applied": False, "reason": "disabled"}])
    with pytest.raises(SystemExit, match="判据只在"):
        runner._check_sentinels(path, expect_judge=True)


def test_sentinel_rejects_a_degraded_run(tmp_path):
    """An outage makes the judge return nothing and the arm silently becomes its
    own baseline; the numbers still look like a result."""
    runner = _load_runner()
    path = _run_file(tmp_path, [{"applied": True, "calls": 12, "graded": 3}])
    with pytest.raises(SystemExit, match="掉线"):
        runner._check_sentinels(path, expect_judge=True)


def test_sentinel_rejects_a_baseline_arm_that_used_the_judge(tmp_path):
    runner = _load_runner()
    path = _run_file(tmp_path, [{"applied": True, "calls": 12, "graded": 12}])
    with pytest.raises(SystemExit, match="不该开判据"):
        runner._check_sentinels(path, expect_judge=False)


def test_the_report_keeps_the_spread_not_only_the_winner():
    """Cold judge calls are not reproducible, so a best-of-three would report
    noise as a gain."""
    runner = _load_runner()
    spread = runner._spread([24.0, 26.0, 25.0])
    assert spread["median"] == 25.0
    assert (spread["min"], spread["max"]) == (24.0, 26.0)
    assert spread["runs"] == [24.0, 26.0, 25.0]


def test_the_blind_run_writes_rows_outside_the_repository():
    """Per-row output carries the sealed questions and the passages retrieved for
    them; under version control that is how a blind set stops being blind."""
    runner = _load_runner()
    import argparse
    import inspect

    source = inspect.getsource(runner.main)
    assert '"~/.offerclaw/private_eval/final_v2/runs"' in source
    assert "--summary" in source


@pytest.mark.private_artifact
def test_the_set_has_the_preregistered_shape_and_no_judge_labels():
    draft = json.loads(
        (ROOT / "docs" / "rag_eval" / "final_v2" / "final_v2.json").read_text(
            encoding="utf-8"))
    design = draft["design"]
    assert (design["positives"], design["anchors"], design["negatives"]) == (80, 40, 40)
    assert design["answerability_judge_used_for_labels"] is False
    positives = [i for i in draft["items"] if i["case_kind"] == "positive"]
    styles = {s: sum(1 for i in positives if i["query_style"] == s)
              for s in design["styles"]}
    assert set(styles.values()) == {20}, styles
    assert {i["split"] for i in draft["items"]} == {"blind"}
    assert all(i["adjudication"]["judge_used"] is False for i in draft["items"])
    # every positive carries exactly one authored gold, and it is a real chunk id
    dual = [i for i in positives if len(i["relevant_targets"]) > 1]
    assert [i["query_id"] for i in dual] == ["fv2-a36-natural"]
    assert all(1 <= len(i["relevant_targets"]) <= 2 for i in positives)
    assert all(i["relevant_targets"][0]["relevance_grade"] == 3 for i in positives)


def test_a_completed_repeat_is_resumed_not_rerun():
    """A three-hour serial run must not restart from zero because one repeat hit
    a network stall; but a run that fails its sentinel is not resumable either --
    it is deleted, because reusing it would launder a degraded run into the
    median."""
    runner = _load_runner()
    import inspect

    source = inspect.getsource(runner.main)
    assert "if out.exists():" in source
    assert "out.unlink()" in source
    assert "已存在，跳过" in source


def test_model_hub_access_is_disabled_for_the_run():
    """Every model is cached locally; a hub round-trip can only stall the run
    (one repeat hung 14 minutes at 0% CPU) and add jitter to the latency it
    reports."""
    runner = _load_runner()
    assert runner.OFFLINE["HF_HUB_OFFLINE"] == "1"
    assert runner.OFFLINE["TRANSFORMERS_OFFLINE"] == "1"
    assert runner.OFFLINE["MODELSCOPE_OFFLINE"] == "1"


def test_anchor_selector_only_offers_route_eligible_chunks():
    """12/80 of Final v3's positives were authored from chunks the reference
    route excludes by construction (project_context/resume/jd/...), so they
    scored as retrieval failures no arm could ever fix.  The selector is where
    that has to be impossible."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "scripts" / "select_final_v2_anchors.py").read_text(encoding="utf-8")
    assert "_ROUTE_EXCLUDED" in source
    for st in ("project_context", "resume", "jd", "verification"):
        assert f'"{st}"' in source.split("_ROUTE_EXCLUDED")[1].split("}")[0]
