from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_duplicate_daily_blocks_are_aggregated():
    from summary_tool import extract_date_block, extract_date_blocks, _parse_log_block

    log = """## 2026-08-22
### 已完成
- 学习 BM25

## 2026-08-22
### 已完成
- 完成 RRF 实验
### 未完成
- 写复盘

## 2026-08-21
- 历史
"""
    assert len(extract_date_blocks(log, "2026-08-22")) == 2
    combined = extract_date_block(log, "2026-08-22")
    parsed = _parse_log_block(combined, "2026-08-22")
    assert parsed["completed"] == ["学习 BM25", "完成 RRF 实验"]
    assert parsed["incomplete"] == ["写复盘"]


def test_structured_daily_log_has_stable_metadata(tmp_path, monkeypatch):
    import summary_tool as st

    monkeypatch.setattr(st, "DAILY_LOG_PATH", str(tmp_path / "daily_log.md"))
    out = st.append_structured_daily_log(
        tag="补技能", done=["实现混合检索"], task_id="pt_1", status="partial",
        minutes=90, attachment_refs=["/daily/a.pdf"], date_str="2026-08-22",
    )
    text = (tmp_path / "daily_log.md").read_text(encoding="utf-8")
    rows = st.extract_log_entries(text, "2026-08-22")
    assert out["log_id"].startswith("log_20260822_")
    assert rows[0]["log_id"] == out["log_id"]
    assert rows[0]["task_id"] == "pt_1"
    assert rows[0]["status"] == "partial"
    assert rows[0]["attachment_refs"] == ["/daily/a.pdf"]


def test_one_invalid_evidence_candidate_does_not_discard_valid_candidates():
    from summary_tool import build_structured_reflection

    block = "## 2026-08-22\n### 已完成\n- 完成 LangGraph 测试并全部通过\n"
    response = """```json
{"completed":["完成 LangGraph 测试并全部通过"],"evidence_candidates":[
  {"claim":"完成状态图测试","source_event_id":"ep_11111111-1111-1111-1111-111111111111",
   "source_quote":"完成 LangGraph 测试并全部通过","capability_id":"cap_langgraph",
   "activity_level":"delivered","scope":"状态图","verification_candidate":"test_result",
   "criteria_ids":["cap_langgraph"],"explicit_statement":false,"relation":"direct","rationale":"测试通过"},
  {"claim":"缺少活动等级","source_event_id":"bad","source_quote":"bad"}
]}
```"""
    reflection = build_structured_reflection(block, "2026-08-22", response)
    assert len(reflection["evidence_candidates"]) == 1
    assert reflection["evidence_candidates"][0]["capability_id"] == "cap_langgraph"


def test_reflection_inventory_marks_orphan_without_daily_log(tmp_path, monkeypatch):
    import reflection_memory as rm
    import memory_layers

    daily = tmp_path / "daily_log.md"
    daily.write_text("## 2026-08-20\n### 已完成\n- 有来源的记录\n", encoding="utf-8")
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "summary_daily_2026-08-20.md").write_text("有效复盘", encoding="utf-8")
    stale_meta = {"reflection_id": "refl_stale", "kind": "daily",
                  "date_from": "2026-08-21", "date_to": "2026-08-21",
                  "source_log_ids": ["log_missing"], "source_status": "valid"}
    (summaries / "summary_daily_2026-08-21.md").write_text(
        f"<!-- offerclaw-reflection: {json.dumps(stale_meta)} -->\n孤立复盘",
        encoding="utf-8",
    )
    monkeypatch.setattr(rm, "DAILY_LOG_PATH", daily)
    monkeypatch.setattr(rm, "SUMMARY_DIR", summaries)
    monkeypatch.setattr(memory_layers.EpisodicMemory, "all", lambda self: [])

    rows = rm.reflection_documents()
    by_date = {r["date_from"]: r for r in rows}
    assert by_date["2026-08-20"]["source_status"] == "valid"
    assert by_date["2026-08-21"]["source_status"] == "orphaned"


def _profile_text() -> str:
    return """# OfferClaw · 用户画像
## 0. 元信息
- 最近更新时间：2026-08-01
- 更新人：用户
## 2. 求职方向
- AI 应用
## 3. 技能清单
- LangGraph：2/5
## 9. 当前能力自评
| 维度 | 自评 |
|---|---|
| Python | 2 |
## 10. 可投入时间
- 每天 4h
"""


def test_profile_edit_conflict_and_suggestion_decision(tmp_path, monkeypatch):
    import profile_store as ps

    profile = tmp_path / "user_profile.md"
    profile.write_text(_profile_text(), encoding="utf-8")
    monkeypatch.setattr(ps, "PROFILE_PATH", profile)

    current = ps.read_profile()
    updated = current["content_md"].replace("LangGraph：2/5", "LangGraph：3/5")
    result = ps.save_profile(updated, current["base_hash"], operation_id="manual-profile-edit")
    assert result["status"] == "ok"
    assert "LangGraph：3/5" in profile.read_text(encoding="utf-8")
    with pytest.raises(ps.ProfileConflictError):
        ps.save_profile(_profile_text(), "bad-hash")


def test_legacy_reflection_evidence_cannot_trigger_mechanical_score_change(tmp_path, monkeypatch):
    import profile_store as ps
    import reflection_memory

    profile = tmp_path / "user_profile.md"
    profile.write_text(_profile_text(), encoding="utf-8")
    monkeypatch.setattr(ps, "PROFILE_PATH", profile)
    evidence = [
        {"id": "r1", "source_status": "valid", "date_to": "2026-08-20", "path": "s1.md",
         "metadata": {"skill_evidence": [{"skill": "LangGraph", "level": "practiced", "evidence": "练习"}]}},
        {"id": "r2", "source_status": "valid", "date_to": "2026-08-21", "path": "s2.md",
         "metadata": {"skill_evidence": [{"skill": "LangGraph", "level": "verified", "evidence": "测试"}]}},
    ]
    monkeypatch.setattr(reflection_memory, "reflection_documents", lambda: evidence)
    before = ps.read_profile()
    first = ps.audit_profile_suggestions(model="", extract_sources=False)
    assert first["created"] == []
    evidence.append({"id": "r3", "source_status": "valid", "date_to": "2026-08-22", "path": "s3.md",
                     "metadata": {"skill_evidence": [{"skill": "LangGraph", "level": "practiced", "evidence": "又练习"}]}})
    assert ps.audit_profile_suggestions(model="", extract_sources=False)["created"] == []
    evidence.append({"id": "r4", "source_status": "valid", "date_to": "2026-08-23", "path": "s4.md",
                     "metadata": {"skill_evidence": [{"skill": "LangGraph", "level": "verified", "evidence": "新测试"}]}})
    assert ps.audit_profile_suggestions(model="", extract_sources=False)["created"] == []
    assert ps.read_profile()["revision"] == before["revision"]
    assert "LangGraph：2/5" in ps.read_profile()["content_md"]


def test_plan_task_patch_uses_stable_id_and_no_llm(tmp_path, monkeypatch):
    import plan_gen
    import memory_layers
    from plan_daily import ensure_task_ids, parse_plan_days, patch_plan_tasks

    monkeypatch.setattr(plan_gen, "_plans_dir", lambda: str(tmp_path))
    monkeypatch.setattr(memory_layers.EpisodicMemory, "append", lambda self, event: event)
    content = """# 计划
计划周期：2026-08-22 → 2026-08-23
### D1（08-22 周六）
核心任务：
1. 学习 BM25
可选任务：
- 看博客
### D2（08-23 周日）
核心任务：
1. 完成 RRF 实验
"""
    first = Path(plan_gen.save_plan(content))
    latest = plan_gen.load_latest_plan()
    parsed = parse_plan_days(ensure_task_ids(latest["content"]))
    task_id = parsed["days"][0]["task_items"][0]["task_id"]
    assert task_id.startswith("pt_")
    out = patch_plan_tasks([
        {"op": "edit", "task_id": task_id, "text": "实现 BM25（预计 4h，依据 P1）",
         "estimated_hours": 3, "priority": "高", "deliverable": "实验报告"},
        {"op": "move", "task_id": task_id, "date": "2026-08-23"},
        {"op": "add", "date": "2026-08-22", "text": "新增任务", "optional": True},
    ], base_mtime=latest["mtime"])
    final = Path(out["saved_path"]).read_text(encoding="utf-8")
    days = parse_plan_days(final)["days"]
    moved = next(x for x in days[1]["task_items"] if x["task_id"] == task_id)
    assert "实现 BM25" in moved["text"]
    assert "预计 3h" in moved["text"] and "预计 4h" not in moved["text"]
    assert "优先级：高" in moved["text"] and "验收：实验报告" in moved["text"]
    assert any(x["optional"] and x["text"] == "新增任务" for x in days[0]["task_items"])
    assert first.exists()


@pytest.mark.parametrize("question,expected", [
    ("我以前学习 BM25 时在哪里卡住？", ("reflection_memory", "search_topic")),
    ("我以前在哪里卡在 BM25，后来做过哪些练习？", ("reflection_memory", "search_topic")),
    ("我今天做了什么？", ("reflection_memory", "get_by_date")),
    ("我现在掌握 LangGraph 了吗？", ("reflection_memory", "get_profile_evidence")),
])
def test_reflection_routes(question, expected):
    from rag_query_plan import rule_plan_query

    routes = {(r.source, r.operation) for r in rule_plan_query(question).routes}
    assert expected in routes


def test_application_reflection_stays_application_experience():
    from rag_query_plan import rule_plan_query

    routes = {r.source for r in rule_plan_query("复盘我在华为面试失败的经验总结").routes}
    assert "application_experience" in routes
    assert "reflection_memory" not in routes
