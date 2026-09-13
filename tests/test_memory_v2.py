# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

import memory_layers as ml
from domain_status import (HardGateStatusCode, MatchStatusCode,
                           SoftConditionStatusCode, match_status_code,
                           requirement_status_code)
from memory_store import MemoryStore


def _write_events(base: str, count: int) -> None:
    memory = ml.EpisodicMemory(base)
    for index in range(count):
        memory.append({"kind": "concurrency_probe", "worker_item": index,
                       "actor": "system", "source": "concurrency_test"})


def test_match_status_uses_exact_machine_enum():
    assert match_status_code("适合") is MatchStatusCode.SUITABLE
    assert match_status_code("不适合") is MatchStatusCode.NOT_RECOMMENDED
    assert match_status_code("很适合") is MatchStatusCode.UNKNOWN
    assert requirement_status_code("✗") == HardGateStatusCode.UNMET.value
    assert requirement_status_code("部分命中") == SoftConditionStatusCode.PARTIAL.value
    with pytest.raises(ValueError):
        requirement_status_code("基本命中")


def test_uuid_sequence_and_multiprocess_writes(tmp_path):
    base = str(tmp_path / "memory")
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_write_events, args=(base, 12)) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    rows = MemoryStore(base).list_events()
    assert len(rows) == 36
    assert len({row["event_id"] for row in rows}) == 36
    assert [row["seq"] for row in rows] == list(range(1, 37))


def test_operation_id_is_idempotent(tmp_path):
    epi = ml.EpisodicMemory(str(tmp_path))
    first = epi.append({"kind": "user_action", "operation_id": "same-op",
                        "actor": "user", "source": "test", "value": 1})
    second = epi.append({"kind": "user_action", "operation_id": "same-op",
                         "actor": "user", "source": "test", "value": 2})
    assert first["event_id"] == second["event_id"]
    assert len(epi.all()) == 1
    assert epi.all()[0]["value"] == 1


def test_known_event_schema_rejects_invalid_enum(tmp_path):
    epi = ml.EpisodicMemory(str(tmp_path))
    with pytest.raises(ValueError):
        epi.append({"kind": "application_changed", "application_id": "a1",
                    "status": "随便写", "status_code": "whatever"})


def test_same_day_reflections_do_not_form_three_day_streak(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    for score in (60, 70, 80):
        ml.record_reflection(epi, {"date": "2026-06-02", "deviation_score": score})
    out = ml.distill_reflections_to_semantic(epi, sem, streak=3)
    assert not any(rule["pattern"] == "high_deviation_streak" for rule in out["rules"])


def test_task_fingerprint_preserves_meaningful_suffixes():
    assert ml._norm_task_key("学习 Python 3.12 异步任务") != ml._norm_task_key("学习 Python 3.13 异步任务")
    assert ml._norm_task_key("掌握 C++") != ml._norm_task_key("掌握 C#")


def test_inferred_topic_needs_three_operations_and_two_days(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    for index, day in enumerate(("2026-06-01", "2026-06-01", "2026-06-02")):
        epi.append({"kind": "conversation_message", "role": "user",
                    "content": "我最近经常关注 LangGraph 调试", "state": "sent",
                    "message_id": f"m{index}", "operation_id": f"op{index}",
                    "business_date": day, "actor": "user", "source": "top_chat"})
    result = ml.distill_to_semantic(epi, sem)
    assert "langgraph" in result["frequent_topics"]


def test_profile_and_daily_records_distill_with_evidence(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    profile = """## 1. 基础信息\n- 学历层次：硕士\n- 可接受工作地域：上海/南京\n\n## 2. 求职方向与偏好\n- 目标方向（按优先级排序）：\n  1. AI 应用开发\n- 明确不做的方向：\n  - Java 主线\n\n## 10. 可投入时间\n- 每天可投入（小时）：4 小时\n"""
    snapshot = epi.store.put_snapshot(profile, media_type="text/markdown")
    profile_event = epi.append({"kind": "historical_snapshot_imported", "actor": "import",
                                "source": "test", "traffic_origin": "organic",
                                "source_kind": "profile", "snapshot_id": snapshot["snapshot_id"],
                                "content_hash": snapshot["content_hash"]})
    for index, day in enumerate(("2026-09-01", "2026-09-02", "2026-09-03")):
        epi.append({"kind": "daily_log_recorded", "log_id": f"l{index}", "date": day,
                    "status": "partial", "done": ["A"], "incomplete": ["B"],
                    "notes": "", "minutes": 180, "actor": "user", "source": "test",
                    "operation_id": f"daily-{index}"})
    ml.distill_to_semantic(epi, sem)
    identity = sem.store.get_semantic("profile:identity_constraints")
    capacity = sem.store.get_semantic(
        "execution:observed_capacity", target_context_id=epi.store.active_goal_id())
    assert identity["value"]["accepted_locations"] == "上海/南京"
    assert profile_event["event_id"] in identity["evidence_ids"]
    assert capacity["value"]["median_minutes"] == 180
    assert len(capacity["evidence_ids"]) == 3


def test_unfollow_archives_long_term_interest(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    base = {"kind": "application_changed", "application_id": "a1", "status": "已评估",
            "status_code": "evaluated", "company": "A", "position": "RAG 工程师",
            "actor": "user", "source": "test"}
    epi.append({**base, "long_term_follow": True, "operation_id": "follow"})
    ml.distill_to_semantic(epi, sem)
    key = f"long_term_interest:{ml._norm_task_key('RAG 工程师')}"
    target = epi.store.active_goal_id()
    assert sem.store.get_semantic(key, target_context_id=target)["lifecycle"] == "active"
    epi.append({**base, "long_term_follow": False, "operation_id": "unfollow"})
    ml.distill_to_semantic(epi, sem)
    assert sem.store.get_semantic(key, target_context_id=target)["lifecycle"] == "archived"


def test_historical_application_snapshot_distills_current_choice(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    content = '{"application_id":"a-old","company":"A","position":"RAG 工程师",' \
              '"status":"持续关注","status_code":"watching","long_term_follow":true}'
    snapshot = epi.store.put_snapshot(content, media_type="application/json")
    event = epi.append({"kind": "historical_snapshot_imported", "actor": "import",
                        "source": "memory_migration", "traffic_origin": "organic",
                        "source_kind": "application", "title": "A · RAG 工程师",
                        "snapshot_id": snapshot["snapshot_id"],
                        "content_hash": snapshot["content_hash"],
                        "source_path": "applications.md", "historical_snapshot": True})
    ml.distill_to_semantic(epi, sem)
    target = epi.store.active_goal_id()
    choices = sem.store.get_semantic("application:current_choices",
                                     target_context_id=target)
    assert choices["value"][0]["application_id"] == "a-old"
    interest = next(row for row in sem.store.list_semantic()
                    if row["memory_key"].startswith("long_term_interest:"))
    assert event["event_id"] in interest["evidence_ids"]


def test_explicit_preference_correction_keeps_opposing_evidence(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    first = epi.append({"kind": "conversation_message", "role": "user",
                        "content": "我偏好远程岗位。", "state": "sent", "message_id": "m1",
                        "actor": "user", "source": "top_chat"})
    second = epi.append({"kind": "conversation_message", "role": "user",
                         "content": "我不考虑远程岗位。", "state": "sent", "message_id": "m2",
                         "actor": "user", "source": "top_chat"})
    ml.distill_to_semantic(epi, sem)
    rows = [row for row in sem.store.list_semantic()
            if row["memory_key"].startswith("explicit_preference:")]
    assert len(rows) == 1 and rows[0]["value"]["polarity"] == "negative"
    assert second["event_id"] in rows[0]["evidence_ids"]
    assert first["event_id"] in rows[0]["opposing_evidence_ids"]


def test_same_sop_name_is_isolated_by_goal(tmp_path):
    store = MemoryStore(str(tmp_path))
    store.upsert_sop("review", "global", {"global": True})
    goal = store.switch_goal("销售")
    store.upsert_sop("review", "sales", {"task_type": "sales"},
                     target_context_id=goal["context_id"])
    rows = [row for row in store.list_sops() if row["name"] == "review"]
    assert len(rows) == 2


def test_goal_scope_isolated_but_recall_can_cross_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    epi = ml.EpisodicMemory(str(tmp_path))
    old = epi.append({"kind": "conversation_message", "role": "user",
                      "content": "我练习了 Python 协程", "state": "sent", "message_id": "old",
                      "actor": "user", "source": "top_chat"})
    new_goal = ml.switch_goal_context("产品销售")
    current = epi.append({"kind": "conversation_message", "role": "user",
                          "content": "我练习了客户需求访谈", "state": "sent", "message_id": "new",
                          "actor": "user", "source": "top_chat"})
    assert old["target_context_id"] != current["target_context_id"] == new_goal["context_id"]
    from memory_search import search_personal_memory
    monkeypatch.setenv("MEMORY_DENSE", "0")
    assert search_personal_memory("Python 协程", purpose="advice")["items"] == []
    recalled = search_personal_memory("Python 协程", purpose="recall")["items"]
    assert recalled and recalled[0]["event_id"] == old["event_id"]


def test_sop_requires_outcomes_and_suspends_after_two_failures(tmp_path):
    epi, proc = ml.EpisodicMemory(str(tmp_path)), ml.ProceduralMemory(str(tmp_path))
    proc.add("interview_review", body="面试后当天记录问题与改进", trigger={"task_type": "interview"})
    assert proc.get("interview_review")["lifecycle"] == "candidate"
    for index, day in enumerate(("2026-06-01", "2026-06-02", "2026-06-02")):
        event = epi.append({"kind": "sop_execution", "actor": "user", "source": "test", "result": "success"})
        ml.record_sop_outcome(proc, "interview_review", event["event_id"], "success", day)
    assert proc.get("interview_review")["lifecycle"] == "active"
    for index in range(2):
        event = epi.append({"kind": "sop_execution", "actor": "user", "source": "test", "result": "failure"})
        ml.record_sop_outcome(proc, "interview_review", event["event_id"], "failure", f"2026-06-0{3+index}")
    assert proc.get("interview_review")["lifecycle"] == "suspended"


def test_delete_creates_tombstone_and_removes_from_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_DENSE", "0")
    event = ml.EpisodicMemory(str(tmp_path)).append({
        "kind": "conversation_message", "role": "user", "content": "独特知识点 AlphaBeta",
        "state": "sent", "message_id": "m", "actor": "user", "source": "top_chat"})
    store = MemoryStore(str(tmp_path))
    assert store.delete_object("event", event["event_id"], "test") is True
    from memory_search import search_personal_memory
    assert search_personal_memory("AlphaBeta", purpose="recall")["items"] == []
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] == 1


def test_dense_personal_recall_uses_sqlite_index_and_cascades_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_DENSE", "1")
    epi = ml.EpisodicMemory(str(tmp_path))
    target = epi.append({
        "kind": "conversation_message", "role": "user",
        "content": "我完成了向量检索召回练习", "state": "sent", "message_id": "dense-1",
        "actor": "user", "source": "top_chat",
    })
    epi.append({
        "kind": "conversation_message", "role": "user",
        "content": "我整理了销售访谈问题", "state": "sent", "message_id": "dense-2",
        "actor": "user", "source": "top_chat",
    })
    import rag_tools
    monkeypatch.setattr(
        rag_tools, "get_embeddings_batch",
        lambda texts: [[1.0, 0.0] if "向量检索" in text else [0.0, 1.0] for text in texts],
    )
    monkeypatch.setattr(rag_tools, "get_embedding", lambda _query: [1.0, 0.0])
    import memory_search
    from memory_search import rebuild_index, search_personal_memory
    rebuilt = rebuild_index(store=epi.store)
    assert rebuilt == {"status": "ok", "indexed": 2, "chunks": 2, "failed": 0}
    result = search_personal_memory("我之前做过语义搜索吗", purpose="recall")
    assert result["dense_status"] == "ok"
    assert result["items"][0]["event_id"] == target["event_id"]
    assert "dense" in result["items"][0]["matched_by"]
    assert epi.store.delete_object("event", target["event_id"], "test") is True
    rows = epi.store.list_search_chunks(memory_search._embedding_profile())
    assert not any(row["event_id"] == target["event_id"] for row in rows)


def test_experience_recall_does_not_treat_plan_as_completed_work(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_DENSE", "0")
    epi = ml.EpisodicMemory(str(tmp_path))
    snapshot = epi.store.put_snapshot("明天计划学习 RAG 向量检索")
    epi.append({
        "kind": "historical_snapshot_imported", "source_kind": "plan",
        "title": "RAG 学习计划", "snapshot_id": snapshot["snapshot_id"],
        "content_hash": snapshot["content_hash"],
        "historical_snapshot": True, "actor": "import", "source": "memory_migration",
    })
    actual = epi.append({
        "kind": "daily_log_recorded", "log_id": "log-1", "date": "2026-09-10",
        "notes": "今天实际完成 RAG 向量检索练习",
        "actor": "user", "source": "daily_log",
    })
    from memory_search import search_personal_memory
    result = search_personal_memory("我以前学过 RAG 吗", purpose="recall")
    assert result["items"]
    assert result["items"][0]["event_id"] == actual["event_id"]
    assert all(item["evidence_role"] != "planned" for item in result["items"])


def test_file_memory_bridge_retries_without_duplicate_append(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    from memory_transactions import write_text_with_memory
    path = tmp_path / "daily.md"
    options = {"actor": "user", "source": "test", "entity_type": "note", "entity_id": "n1"}
    first = write_text_with_memory(
        path, "first\n", event_kind="note_appended", event_payload={"content": "first"},
        event_options=options, operation_id="retry-op", append=True,
    )
    second = write_text_with_memory(
        path, "first\n", event_kind="note_appended", event_payload={"content": "first"},
        event_options=options, operation_id="retry-op", append=True,
    )
    assert first["memory_event_id"] == second["memory_event_id"]
    assert path.read_text(encoding="utf-8") == "first\n"


def test_file_memory_bridge_recovers_event_after_file_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    from memory_transactions import write_text_with_memory
    original = ml.record_business_event
    monkeypatch.setattr(ml, "record_business_event",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    path = tmp_path / "profile.md"
    result = write_text_with_memory(
        path, "confirmed\n", event_kind="note_saved", event_payload={"content": "confirmed"},
        event_options={"actor": "user", "source": "test"}, operation_id="recover-op",
    )
    assert result["memory_error"] == "boom"
    assert path.read_text(encoding="utf-8") == "confirmed\n"
    monkeypatch.setattr(ml, "record_business_event", original)
    recovered = MemoryStore().recover_file_operations()
    assert recovered == [{"operation_id": "recover-op", "status": "committed",
                          "target_path": str(path.resolve())}]
    assert MemoryStore().get_event_by_operation("recover-op")


def test_delete_event_withdraws_derived_semantic(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    event = epi.append({"kind": "conversation_message", "role": "user",
                        "content": "我明确长期关注向量数据库", "state": "sent",
                        "message_id": "m1", "actor": "user", "source": "top_chat"})
    memory = sem.store.upsert_semantic(
        "topic:vector-db", {"topic": "向量数据库"}, memory_type="stage_interest",
        certainty="inferred", confidence=.8, evidence_ids=[event["event_id"]],
    )
    assert epi.store.delete_object("event", event["event_id"], "user request")
    rows = sem.store.list_semantic(include_inactive=True)
    updated = next(row for row in rows if row["memory_id"] == memory["memory_id"])
    assert updated["lifecycle"] == "archived"


def test_plan_draft_requires_approval_and_reject_keeps_current(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    import plan_drafts
    monkeypatch.setattr(plan_drafts, "_current_plan_version",
                        lambda: {"filename": "", "mtime": 0, "content_hash": ""})
    draft = plan_drafts.create_plan_draft("# Draft")
    assert draft["status"] == "pending"
    result = plan_drafts.decide_plan_draft(draft["draft_id"], "reject")
    assert result["saved_path"] == ""
    replay = plan_drafts.decide_plan_draft(draft["draft_id"], "reject")
    assert replay["replayed"] is True


def test_plan_save_operation_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    import plan_gen
    monkeypatch.setattr(plan_gen, "OUTPUT_DIR", str(tmp_path / "plans"))
    first = plan_gen.save_plan("# Plan\n", edited_by_user=True,
                               operation_id="plan-save-retry", note="manual edit")
    second = plan_gen.save_plan("# Different retry body\n", edited_by_user=True,
                                operation_id="plan-save-retry", note="manual edit")
    assert first == second
    assert Path(first).read_text(encoding="utf-8").startswith("# Plan")
    events = MemoryStore().list_events(kind="plan_saved")
    assert len(events) == 1 and events[0]["note"] == "manual edit"


def test_profile_save_retry_returns_original_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(tmp_path / "memory"))
    import profile_store
    path = tmp_path / "user_profile.md"
    original = "\n".join(("## 0. 元数据", "最近更新时间：2026-01-01", "更新人：用户",
                           "## 2. 方向", "## 3. 能力", "## 9. 约束", "## 10. 时间"))
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(profile_store, "PROFILE_PATH", path)
    updated = original + "\n- 目标：RAG"
    first = profile_store.save_profile(
        updated, profile_store.content_hash(original), operation_id="profile-save-retry")
    second = profile_store.save_profile(
        "\n".join(("## 0. x", "## 2. x", "## 3. x", "## 9. x", "## 10. x")),
        profile_store.content_hash(original), operation_id="profile-save-retry")
    assert second["replayed"] is True
    assert second["memory_event_id"] == first["memory_event_id"]
    assert second["content_md"] == first["content_md"]
    assert len(MemoryStore().list_events(kind="profile_edited")) == 1


def _free_text_events(epi):
    rows = []
    for index, (day, content) in enumerate((
        ("2026-09-08", "我使用分块练习复习 RAG，感觉更容易坚持。"),
        ("2026-09-09", "我继续把向量检索拆开做分块练习，完成得比较顺利。"),
        ("2026-09-09", "我又用分块练习准备 Agent 面试，能按计划完成。"),
    ), start=1):
        rows.append(epi.append({
            "kind": "conversation_message", "role": "user", "content": content,
            "state": "sent", "message_id": f"model-memory-{index}",
            "operation_id": f"model-memory-op-{index}", "business_date": day,
            "actor": "user", "source": "top_chat", "traffic_origin": "organic",
        }))
    return rows


def test_model_distillation_accepts_only_verified_cross_day_evidence(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    rows = _free_text_events(epi)

    def fake_llm(_messages, **_kwargs):
        return json.dumps({"candidates": [{
            "kind": "behavior", "topic": "分块练习", "polarity": "positive",
            "summary": "用户反复采用分块练习，并反馈这种方法有助于完成任务。",
            "scope": "current_goal",
            "support": [
                {"event_id": row["event_id"], "quote": "分块练习"} for row in rows
            ],
            "oppose": [],
        }]}, ensure_ascii=False)

    result = ml.distill_free_text_with_model(epi, sem, llm_call=fake_llm, force=True)
    assert result["status"] == "ok" and len(result["accepted"]) == 1
    memory = next(item for item in sem.store.list_semantic()
                  if item["memory_type"] == "behavior_pattern")
    assert set(memory["evidence_ids"]) == {row["event_id"] for row in rows}
    assert memory["value"]["extraction"] == "model_assisted"
    assert sem.store.get_runtime("model_distill_status")["accepted"] == 1


def test_model_distillation_rejects_fabricated_support_quote(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    rows = _free_text_events(epi)

    def fake_llm(_messages, **_kwargs):
        support = [{"event_id": row["event_id"], "quote": "分块练习"} for row in rows]
        support[-1]["quote"] = "原文里不存在的支持证据"
        return json.dumps({"candidates": [{
            "kind": "behavior", "topic": "分块练习", "polarity": "positive",
            "summary": "用户持续采用分块练习。", "scope": "current_goal",
            "support": support, "oppose": [],
        }]}, ensure_ascii=False)

    result = ml.distill_free_text_with_model(epi, sem, llm_call=fake_llm, force=True)
    assert result["accepted"] == []
    assert result["rejected"] == [{
        "topic": "分块练习", "reason": "insufficient_verified_evidence",
    }]
    assert not any(item["memory_type"] == "behavior_pattern"
                   for item in sem.store.list_semantic())


def test_model_inference_cannot_override_explicit_preference(tmp_path):
    epi, sem = ml.EpisodicMemory(str(tmp_path)), ml.SemanticMemory(str(tmp_path))
    rows = _free_text_events(epi)
    sem.store.upsert_semantic(
        "explicit_preference:chunking", {"topic": "分块练习", "polarity": "negative"},
        memory_type="explicit_preference", certainty="explicit", confidence=1.0,
    )

    def fake_llm(_messages, **_kwargs):
        return json.dumps({"candidates": [{
            "kind": "preference", "topic": "分块练习", "polarity": "positive",
            "summary": "用户偏好使用分块练习。", "scope": "current_goal",
            "support": [
                {"event_id": row["event_id"], "quote": "分块练习"} for row in rows
            ],
            "oppose": [],
        }]}, ensure_ascii=False)

    result = ml.distill_free_text_with_model(epi, sem, llm_call=fake_llm, force=True)
    assert result["accepted"] == []
    assert result["rejected"] == [{
        "topic": "分块练习", "reason": "explicit_memory_precedence",
    }]
    assert len([item for item in sem.store.list_semantic()
                if item["memory_type"] == "explicit_preference"]) == 1
