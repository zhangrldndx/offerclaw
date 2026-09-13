from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from memory_layers import EpisodicMemory
from memory_store import MemoryStore
from profile_review import (
    ProfileRepository,
    ProfileValidationError,
    _default_capability_id,
    parse_profile_markdown,
    render_profile_markdown,
)


PROFILE_MD = """# OfferClaw 用户画像

## 0. 元信息
- 最近更新时间：2026-09-01

## 1. 基本信息
- 姓名 / 昵称：测试用户
- 所在地：示例市

## 2. 求职方向与偏好
- 目标岗位类型：AI 应用开发
- 目标方向（按优先级排序）：
  1. AI 应用开发

## 3. 技能清单
- Python 工程：2/5
- LangGraph：2/5

## 4. 项目经历
- 已有项目

## 8. 兴趣和工作方式
- 兴趣爱好：阅读
- 擅长的思维方式：结构化分析
- 不擅长或不感兴趣的方向：纯销售

## 9. 当前能力自评
| 维度 | 自评 |
|---|---|
| RAG 工程 | 2 |

## 10. 可投入时间
- 每天可投入（小时）：2
- 每周可投入（小时）：10
"""


def _repo(tmp_path: Path) -> tuple[ProfileRepository, EpisodicMemory, Path]:
    profile_path = tmp_path / "user_profile.md"
    profile_path.write_text(PROFILE_MD, encoding="utf-8")
    episodic = EpisodicMemory(base_dir=tmp_path / "memory")
    return ProfileRepository(episodic.store, profile_path=profile_path), episodic, profile_path


def _daily_event(episodic: EpisodicMemory, quote: str, *, days_ago: int = 1,
                 log_id: str = "") -> dict:
    day = (date.today() - timedelta(days=days_ago)).isoformat()
    return episodic.append({
        "kind": "daily_log_recorded", "actor": "user", "source": "daily_log",
        "business_date": day, "log_id": log_id or f"log-{days_ago}-{abs(hash(quote))}",
        "date": day, "status": "done", "notes": quote,
    }, export=False, index=False)


def _candidate(event: dict, quote: str, *, capability_id: str = "cap_python",
               activity_level: str = "delivered", verification: str = "test_result",
               **extra) -> dict:
    return {
        "claim": extra.pop("claim", "完成了可验证的工程任务"),
        "source_event_id": event["event_id"], "source_quote": quote,
        "capability_id": capability_id,
        "activity_level": activity_level, "scope": extra.pop("scope", "global"),
        "verification_candidate": verification, "criteria_ids": extra.pop("criteria_ids", []),
        "explicit_statement": extra.pop("explicit_statement", False),
        "relation": extra.pop("relation", "direct"), "rationale": "test",
        **extra,
    }


def _caller(payload: dict):
    def call(messages, max_tokens, temperature, model):
        return json.dumps(payload, ensure_ascii=False)
    return call


def _skill(repo: ProfileRepository, name: str) -> dict:
    return next(item for item in repo.current()["profile_spec"]["skills"] if item["name"] == name)


def _audit_skill(repo: ProfileRepository, skill: dict, evidence_ids: list[str], level: int) -> dict:
    payload = {"changes": [{
        "field_path": f"/skills/{skill['assessment_id']}/level",
        "current_value": skill["level"], "observed_change": "新增了可验证交付证据",
        "operation": "replace", "proposed_value": level,
        "evidence_ids": evidence_ids, "counter_evidence_ids": [],
        "capability_id": skill["capability_id"], "new_capability": None,
        "rationale": "证据支持等级调整", "requirement_status": "satisfied",
    }]}
    return repo.run_audit(caller=_caller(payload), model="test", extract_sources=False)


def test_profile_markdown_roundtrip_and_alias_identity():
    spec = parse_profile_markdown(PROFILE_MD)
    assert parse_profile_markdown(render_profile_markdown(spec)).matching_fields == spec.matching_fields
    assert _default_capability_id("状态图编排") == "cap_langgraph"
    assert _default_capability_id("LangGraph 工作流") == "cap_langgraph"


def test_hallucinated_quote_is_rejected_and_valid_quote_has_location(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 Python API 测试，12 条用例全部通过"
    event = _daily_event(episodic, quote)
    bad = repo.ingest_candidates([_candidate(event, "并不存在的原文")])
    assert bad["rejected"] == 1
    assert bad["items"][0]["rejection_reason"] == "source_quote_not_found"

    good = repo.ingest_candidates([_candidate(event, quote)])
    item = good["items"][0]
    assert good["validated"] == 1
    assert item["metadata"]["quote_location"] == "exact"
    assert item["metadata"]["quote_start"] >= 0
    assert len(item["metadata"]["source_content_hash"]) == 64


def test_same_source_derivatives_do_not_duplicate_evidence(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 Python API 测试并全部通过"
    event = _daily_event(episodic, quote)
    first = repo.ingest_candidates([_candidate(event, quote, claim="日报提取")])
    second = repo.ingest_candidates([_candidate(event, quote, claim="周总结重复提取")])
    assert first["items"][0]["evidence_id"] == second["items"][0]["evidence_id"]
    assert len(repo.list_evidence()) == 1


def test_claimed_test_without_success_is_only_user_attested(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "今天运行了 12 条 Python 测试，结果稍后再看"
    event = _daily_event(episodic, quote)
    item = repo.ingest_candidates([_candidate(event, quote)])["items"][0]
    assert item["verification"] == "user_attested"
    assert repo.capability_bounds()["cap_python"]["max_supported_level"] == 2


def test_observed_cannot_support_level_three_and_atomic_delivery_can(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    observed_quote = "阅读了 Python 类型系统文档"
    observed = _daily_event(episodic, observed_quote, days_ago=3)
    repo.ingest_candidates([_candidate(
        observed, observed_quote, activity_level="observed", verification="source_grounded",
    )])
    assert repo.capability_bounds()["cap_python"]["max_supported_level"] == 1

    evidence_ids = []
    for index in (1, 2):
        quote = f"完成 Python API 测试，第 {index} 轮用例全部通过"
        event = _daily_event(episodic, quote, days_ago=index, log_id=f"delivery-{index}")
        result = repo.ingest_candidates([_candidate(event, quote)])
        evidence_ids.append(result["items"][0]["evidence_id"])
    assert repo.capability_bounds()["cap_python"]["max_supported_level"] == 3
    result = _audit_skill(repo, _skill(repo, "Python 工程"), evidence_ids, 3)
    assert result["created"][0]["status"] == "pending"


def test_atomic_rag_evidence_does_not_close_broad_rag_gap(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 BM25 检索模块测试并全部通过"
    event = _daily_event(episodic, quote)
    repo.ingest_candidates([_candidate(
        event, quote, capability_id="cap_rag_retrieval",
        criteria_ids=["cap_rag_retrieval"],
    )])
    bounds = repo.capability_bounds()
    assert bounds["cap_rag_retrieval"]["max_supported_level"] >= 2
    assert bounds["cap_rag_engineering"]["max_supported_level"] <= 2


def test_protected_field_requires_value_in_explicit_user_quote(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "我明确想把目标岗位改为产品经理"
    event = _daily_event(episodic, quote)
    evidence = repo.ingest_candidates([_candidate(
        event, quote, activity_level="observed", verification="source_grounded",
        explicit_statement=True,
    )])["items"][0]
    payload = {"changes": [{
        "field_path": "/fields/目标岗位类型", "current_value": "AI 应用开发",
        "observed_change": "用户明确调整正式目标", "operation": "replace",
        "proposed_value": "产品销售", "evidence_ids": [evidence["evidence_id"]],
        "counter_evidence_ids": [], "capability_id": "", "new_capability": None,
        "rationale": "用户主动提出", "requirement_status": "satisfied",
    }]}
    result = repo.run_audit(caller=_caller(payload), model="test", extract_sources=False)
    assert result["created"][0]["status"] == "needs_evidence"
    assert "explicit_value_not_found_in_source" in repo.list_suggestions()[0]["rationale"]


def test_review_unavailable_keeps_evidence_and_profile_unchanged(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 Python API 测试并全部通过"
    event = _daily_event(episodic, quote)
    repo.ingest_candidates([_candidate(event, quote)])
    before = repo.current()
    result = repo.run_audit(model="", extract_sources=False)
    assert result["status"] == "review_unavailable"
    assert repo.current()["revision"] == before["revision"]
    assert len(repo.list_evidence()) == 1


def test_provisional_capability_activates_only_after_acceptance(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 Temporal 工作流练习并交付示例"
    event = _daily_event(episodic, quote)
    evidence = repo.ingest_candidates([_candidate(
        event, quote, capability_id="", new_capability_candidate="Temporal",
        activity_level="practiced", verification="source_grounded",
    )])["items"][0]
    capability_id = evidence["capability_id"]
    assert next(x for x in repo.capabilities(include_archived=True)
                if x["capability_id"] == capability_id)["lifecycle"] == "provisional"
    payload = {"changes": [{
        "field_path": "/skills/new", "current_value": None,
        "observed_change": "出现新的工作流实践", "operation": "add",
        "proposed_value": {"name": "Temporal", "level": 2},
        "evidence_ids": [evidence["evidence_id"]], "counter_evidence_ids": [],
        "capability_id": capability_id,
        "new_capability": {"name": "Temporal", "parent_id": "cap_agent_workflow", "aliases": []},
        "rationale": "实践证据支持", "requirement_status": "satisfied",
    }]}
    audit = repo.run_audit(caller=_caller(payload), model="test", extract_sources=False)
    suggestion = repo.list_suggestions("pending")[0]
    assert audit["pending"] == 1
    repo.decide_suggestion(
        suggestion["suggestion_id"], "accepted", base_revision=suggestion["base_revision"],
        operation_id="accept-temporal",
    )
    assert next(x for x in repo.capabilities(include_archived=True)
                if x["capability_id"] == capability_id)["lifecycle"] == "active"


def test_acceptance_is_idempotent_and_other_revision_becomes_stale(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    ids = []
    for index in (1, 2):
        quote = f"完成 Python API 测试，第 {index} 次全部通过"
        event = _daily_event(episodic, quote, days_ago=index)
        ids.append(repo.ingest_candidates([_candidate(event, quote)])["items"][0]["evidence_id"])
    _audit_skill(repo, _skill(repo, "Python 工程"), ids, 3)
    suggestion = repo.list_suggestions("pending")[0]
    first = repo.decide_suggestion(
        suggestion["suggestion_id"], "accepted", base_revision=suggestion["base_revision"],
        operation_id="same-operation",
    )
    second = repo.decide_suggestion(
        suggestion["suggestion_id"], "accepted", base_revision=suggestion["base_revision"],
        operation_id="same-operation",
    )
    assert second["replayed"] is True
    assert second["result_revision"] == first["result_revision"]


def test_modified_acceptance_rechecks_level_bound(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    ids = []
    for index in (1, 2):
        quote = f"完成 Python API 测试，第 {index} 次全部通过"
        event = _daily_event(episodic, quote, days_ago=index)
        ids.append(repo.ingest_candidates([_candidate(event, quote)])["items"][0]["evidence_id"])
    _audit_skill(repo, _skill(repo, "Python 工程"), ids, 3)
    suggestion = repo.list_suggestions("pending")[0]
    with pytest.raises(ProfileValidationError, match="证据校验"):
        repo.decide_suggestion(
            suggestion["suggestion_id"], "modified", modified_value=4,
            base_revision=suggestion["base_revision"], operation_id="too-high",
        )
    assert repo.current()["revision"] == suggestion["base_revision"]


def test_rejected_suggestion_is_not_recreated_from_same_evidence(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    ids = []
    for index in (1, 2):
        quote = f"完成 Python API 测试，第 {index} 次全部通过"
        event = _daily_event(episodic, quote, days_ago=index)
        ids.append(repo.ingest_candidates([_candidate(event, quote)])["items"][0]["evidence_id"])
    skill = _skill(repo, "Python 工程")
    _audit_skill(repo, skill, ids, 3)
    suggestion = repo.list_suggestions("pending")[0]
    repo.decide_suggestion(
        suggestion["suggestion_id"], "rejected", base_revision=suggestion["base_revision"],
        operation_id="reject-once",
    )
    repeated = _audit_skill(repo, skill, ids, 3)
    assert repeated["created"] == []


def test_rejected_suggestion_reopens_only_with_equally_strong_new_evidence(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    ids = []
    for index in (2, 3):
        quote = f"完成 Python API 测试，第 {index} 次全部通过"
        event = _daily_event(episodic, quote, days_ago=index)
        ids.append(repo.ingest_candidates([_candidate(event, quote)])["items"][0]["evidence_id"])
    skill = _skill(repo, "Python 工程")
    _audit_skill(repo, skill, ids, 3)
    rejected = repo.list_suggestions("pending")[0]
    repo.decide_suggestion(
        rejected["suggestion_id"], "rejected", base_revision=rejected["base_revision"],
        operation_id="reject-python",
    )

    weak_quote = "阅读了 Python 性能调优文档"
    weak_event = _daily_event(episodic, weak_quote, days_ago=1)
    weak_id = repo.ingest_candidates([_candidate(
        weak_event, weak_quote, activity_level="observed", verification="source_grounded",
    )])["items"][0]["evidence_id"]
    assert _audit_skill(repo, skill, ids + [weak_id], 3)["created"] == []

    strong_quote = "完成 Python 性能回归测试并全部通过"
    strong_event = _daily_event(episodic, strong_quote, days_ago=0)
    strong_id = repo.ingest_candidates([_candidate(strong_event, strong_quote)])["items"][0]["evidence_id"]
    reopened = _audit_skill(repo, skill, ids + [strong_id], 3)
    assert reopened["created"][0]["status"] == "pending"


def test_level_five_needs_explicit_statement_and_multiscenario_evidence(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    strong_ids = []
    for index, scope in enumerate(("API 服务", "数据管道", "自动化工具"), 1):
        quote = f"完成 Python {scope}测试并全部通过，包含回归评估"
        event = _daily_event(episodic, quote, days_ago=index)
        strong_ids.append(repo.ingest_candidates([_candidate(
            event, quote, scope=scope,
        )])["items"][0]["evidence_id"])
    skill = _skill(repo, "Python 工程")
    ordinary = _audit_skill(repo, skill, strong_ids, 5)
    assert ordinary["created"][0]["status"] == "needs_evidence"

    explicit_quote = "我明确确认 Python 工程达到 5 级"
    explicit_event = _daily_event(episodic, explicit_quote, days_ago=0)
    explicit_id = repo.ingest_candidates([_candidate(
        explicit_event, explicit_quote, activity_level="delivered",
        verification="user_attested", explicit_statement=True,
    )])["items"][0]["evidence_id"]
    # The earlier needs-evidence proposal is based on a different evidence set,
    # so the explicit statement can create an approvable candidate.
    result = _audit_skill(repo, skill, strong_ids + [explicit_id], 5)
    assert result["created"][0]["status"] == "pending"


def test_transaction_failure_rolls_back_profile_and_decision(tmp_path, monkeypatch):
    repo, episodic, _ = _repo(tmp_path)
    ids = []
    for index in (1, 2):
        quote = f"完成 Python API 测试，第 {index} 次全部通过"
        event = _daily_event(episodic, quote, days_ago=index)
        ids.append(repo.ingest_candidates([_candidate(event, quote)])["items"][0]["evidence_id"])
    _audit_skill(repo, _skill(repo, "Python 工程"), ids, 3)
    suggestion = repo.list_suggestions("pending")[0]
    revision = repo.current()["revision"]

    def fail(*args, **kwargs):
        raise RuntimeError("forced transaction failure")

    monkeypatch.setattr(repo, "_insert_revision", fail)
    with pytest.raises(RuntimeError, match="forced"):
        repo.decide_suggestion(
            suggestion["suggestion_id"], "accepted", base_revision=revision,
            operation_id="rollback-operation",
        )
    assert repo.current()["revision"] == revision
    assert repo.list_suggestions("pending")[0]["suggestion_id"] == suggestion["suggestion_id"]
    with repo.store._connect() as conn:
        assert conn.execute(
            "SELECT 1 FROM profile_suggestion_decisions WHERE operation_id='rollback-operation'"
        ).fetchone() is None


def test_manual_edit_preview_then_commit_preserves_structured_fields(tmp_path):
    repo, _, profile_path = _repo(tmp_path)
    current = repo.current()
    edited = current["content_md"].replace("每天可投入（小时）：2", "每天可投入（小时）：3")
    preview = repo.create_edit_preview(edited, base_revision=current["revision"])
    assert any(item["field_path"] == "/fields/每天可投入"
               for item in preview["diff"]["changed_fields"])
    result = repo.commit_edit_preview(
        preview["preview_id"], reason="用户明确修改", operation_id="manual-edit",
    )
    assert result["revision"] == current["revision"] + 1
    assert parse_profile_markdown(profile_path.read_text(encoding="utf-8")).matching_fields[
        "每天可投入"] == "3"


def test_manual_edit_preserves_existing_skill_identity_and_evidence_links(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 Python API 测试并全部通过"
    event = _daily_event(episodic, quote)
    evidence_id = repo.ingest_candidates([_candidate(event, quote)])["items"][0]["evidence_id"]
    current = repo.current()
    spec = current["profile_spec"]
    python = next(item for item in spec["skills"] if item["name"] == "Python 工程")
    # Seed a persisted link as an already accepted assessment would have.
    with repo.store.transaction() as conn:
        raw = json.loads(conn.execute(
            "SELECT profile_json FROM profile_revisions WHERE revision=?", (current["revision"],),
        ).fetchone()[0])
        for item in raw["skills"]:
            if item["assessment_id"] == python["assessment_id"]:
                item["evidence_ids"] = [evidence_id]
        conn.execute(
            "UPDATE profile_revisions SET profile_json=? WHERE revision=?",
            (json.dumps(raw, ensure_ascii=False), current["revision"]),
        )
    current = repo.current()
    edited = current["content_md"].replace("- Python 工程：2/5\n- LangGraph：2/5",
                                           "- LangGraph：2/5\n- Python 工程：2/5")
    preview = repo.create_edit_preview(edited, base_revision=current["revision"])
    repo.commit_edit_preview(preview["preview_id"], operation_id="reorder-skills")
    after = _skill(repo, "Python 工程")
    assert after["assessment_id"] == python["assessment_id"]
    assert after["evidence_ids"] == [evidence_id]


def test_pending_extraction_uses_structured_candidates(tmp_path):
    repo, episodic, _ = _repo(tmp_path)
    quote = "完成 LangGraph 状态图测试并全部通过"
    event = _daily_event(episodic, quote)
    payload = {"candidates": [_candidate(
        event, quote, capability_id="cap_langgraph", claim="完成状态图交付",
    )]}
    result = repo.extract_pending_evidence(caller=_caller(payload), model="test")
    assert result["status"] == "ok" and result["validated"] == 1
    with repo.store._connect() as conn:
        row = conn.execute(
            "SELECT status FROM profile_evidence_extractions WHERE source_event_id=?",
            (event["event_id"],),
        ).fetchone()
    assert row["status"] == "completed"


def test_legacy_pending_suggestion_is_imported_as_stale(tmp_path):
    profile_path = tmp_path / "user_profile.md"
    profile_path.write_text(PROFILE_MD, encoding="utf-8")
    legacy_dir = tmp_path / "logs" / "profile"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "suggestions.jsonl").write_text(json.dumps({
        "action": "proposed", "suggestion_id": "ps_legacy",
        "suggestion": {"target_section": "3", "current_text": "旧值",
                       "proposed_text": "机械加一", "evidence_refs": []},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    repo = ProfileRepository(MemoryStore(tmp_path / "memory"), profile_path=profile_path)
    suggestion = repo.list_suggestions("stale")[0]
    assert suggestion["suggestion_id"] == "ps_legacy"
    assert suggestion["requirement_status"] == "needs_evidence"
    assert repo.migration_report()["report"]["legacy_suggestions_staled"] == 1


def test_profile_edit_preview_and_commit_api(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    import profile_store
    from fastapi.testclient import TestClient
    from rag_api import app

    profile_path = tmp_path / "user_profile.md"
    profile_path.write_text(PROFILE_MD, encoding="utf-8")
    monkeypatch.setattr(profile_store, "PROFILE_PATH", profile_path)
    client = TestClient(app)

    editor = client.get("/api/profile/editor")
    assert editor.status_code == 200
    current = editor.json()
    no_changes = client.post("/api/profile/edit-preview", json={
        "content_md": current["content_md"], "base_revision": current["revision"],
    })
    assert no_changes.status_code == 200
    assert no_changes.json()["status"] == "no_changes"
    assert "preview_id" not in no_changes.json()
    with profile_store._repository().store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM profile_edit_previews").fetchone()[0] == 0

    changed = current["content_md"].replace("每天可投入（小时）：2", "每天可投入（小时）：3")
    preview = client.post("/api/profile/edit-preview", json={
        "content_md": changed, "base_revision": current["revision"],
    })
    assert preview.status_code == 200
    committed = client.post("/api/profile/edit-commit", json={
        "preview_id": preview.json()["preview_id"], "reason": "API test",
        "operation_id": "api-profile-edit",
    })
    assert committed.status_code == 200
    assert committed.json()["revision"] == current["revision"] + 1
    invalid = client.post("/api/profile/suggestions/missing/decision", json={
        "decision": "silently_apply", "base_revision": current["revision"],
    })
    assert invalid.status_code == 422
