# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import application_jd_store as jd
import applications_store as apps
import migrate_application_jd_links as migration
from rag_api import app


OLD_TABLE = """# 投递追踪

| 日期 | 公司 | 岗位 | 来源 | 地点 | 匹配结论 | 样本定位 | 当前状态 | 下一步动作 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-21 | 华为计算产品线 | AI应用工程师 | — | — | — | — | 准备投递 | 面试前准备 | 08-21 准备投递 |
"""

JD_TEXT = """公司：华为计算产品线
岗位名称：AI应用工程师
工作地点：上海
岗位职责：负责 RAG 与 Agent 应用开发、评测和部署。
任职要求：熟悉 Python、向量检索、重排和 FastAPI，有端到端项目经验。
"""


def test_application_jd_ui_uses_page_modal_not_native_dialogs():
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert "function attachJDToApplication(applicationId)" in html
    assert "previewAttachedJD(applicationId)" in html
    assert "粘贴该次投递实际对应的完整 JD" in html
    assert "function extractJDIntoApplicationForm" in html
    assert "从 URL 抽取正文" in html
    assert "投递来源 / 招聘渠道" in html
    assert "function looksLikeJDUrl" in html
    assert "raw: url ? '' : raw" in html
    assert "开始匹配”也接受只粘贴 URL" in html
    assert "function updateApplicationStatus(applicationId, selectId, buttonId)" in html
    assert "更新状态" in html
    assert "body:JSON.stringify({status})" in html
    assert "prompt(" not in html
    assert "confirm(" not in html


def test_resume_workshop_selects_bound_application_jd_instead_of_temporary_analysis():
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert "用于定制的投递 JD（单选）" in html
    assert 'name="rpTargetApplication"' in html
    assert "applicationId, jdVersionId, resumeScope" in html
    assert "项目经历使用所选 JD 定制" in html
    assert 'id="rpUseJD"' in html
    assert 'id="rpOutputScope"' in html
    assert '<option value="project_section">项目经历</option>' in html
    assert '<option value="full_resume">完整简历</option>' in html
    assert 'onclick="startResumeAgentFromWorkshop()"' in html
    assert "function buildResumeProject" not in html
    assert "按 JD 生成项目简历段" not in html
    assert html.count("Resume Agent → Critic</button>") == 1
    assert "function startResumeAgentFromWorkshop()" in html
    assert "startResumeAgent({" in html
    assert "resume_scope:resumeScope" in html
    assert "已折叠 ${collapsed} 条完全相同的活动 JD 记录" in html
    assert "const rows = allRows.filter(x=>!x.duplicate_of);" in html
    assert ">Resume→Critic 定制简历</button>" not in html
    # 断言的是"披露了不读临时分析"这个承诺,不是某一版的措辞——2026-09-01 的 UI
    # 改版把这段说明从 <p> 挪进 <details class="howto">,顺手把"不会读取"改成
    # "不读取",承诺没变但字面断言红了。改版后仍要保证这句话在页面上存在。
    assert re.search(r"不(会)?读取\s*JD 分析区的临时内容", html)


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    path = tmp_path / "applications.md"
    path.write_text(OLD_TABLE, encoding="utf-8")
    monkeypatch.setattr(apps, "APPLICATIONS_PATH", str(path))
    monkeypatch.setattr(jd, "SNAPSHOT_DIR", str(tmp_path / "application_jds"))
    monkeypatch.setattr(jd, "PROFILE_PATH", str(tmp_path / "profile.md"))
    (tmp_path / "profile.md").write_text("# profile", encoding="utf-8")
    monkeypatch.setattr(jd, "_match_payload", lambda text, title, jd_analysis=None: {
        "status": "当前适合投递", "direction": "Agent 应用工程", "summary": "ok",
        "gap_list": {"技能缺口": ["缺少 RAG 评测实战"]},
        "suggestions": ["补一个评测实验"], "requirement_analysis": {"skills": ["RAG"]},
    })
    return path


def _artifact(application_id="app_test", text=JD_TEXT, jd_id=""):
    return jd.prepare_approved_artifacts(
        application_id=application_id, jd_text=text,
        company="华为计算产品线", position="AI应用工程师", location="上海",
        source_url="https://career.huawei.com/job/1",
        expected_hash=jd.jd_content_hash(text), existing_jd_id=jd_id,
    )


def test_schema_migration_assigns_id_but_never_guesses_jd(isolated):
    out = apps.ensure_application_schema()
    assert out["status"] == "ok"
    row = apps.application_fact_views()[0]
    assert row["application_id"].startswith("app_legacy_")
    assert row["jd_version_id"] == ""
    assert row["include_in_plan"] is False


def test_legacy_gap_archive_is_idempotent_across_dates(tmp_path, monkeypatch):
    legacy = tmp_path / "gap_store.json"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    legacy.write_text('{"targets": []}', encoding="utf-8")
    old_archive = archive_dir / "gap_store_legacy_20260821.json"
    old_archive.write_text('{"targets": []}', encoding="utf-8")
    monkeypatch.setattr(migration, "LEGACY_GAPS", str(legacy))
    monkeypatch.setattr(migration, "ARCHIVE_DIR", str(archive_dir))
    monkeypatch.setattr(
        migration.applications_store, "ensure_application_schema",
        lambda dry_run=False: {"status": "unchanged"},
    )

    out = migration.migrate(apply=True)

    assert out["legacy_gap_store"]["status"] == "already_archived"
    assert out["legacy_gap_store"]["path"] == str(old_archive)
    assert list(archive_dir.glob("gap_store_legacy_*.json")) == [old_archive]


def test_preview_is_ephemeral_and_lists_existing_candidate(isolated):
    apps.ensure_application_schema()
    out = jd.preview_from_jd(JD_TEXT, source_url="https://career.huawei.com/job/1")
    assert out["ephemeral"] is True
    assert out["draft"]["company"] == "华为计算产品线"
    assert out["existing_candidates"][0]["application_id"].startswith("app_legacy_")
    assert not os.path.exists(jd.SNAPSHOT_DIR)


def test_preview_finds_same_content_even_when_jd_omits_company_and_title(isolated, monkeypatch):
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    artifact = _artifact(application_id=row["application_id"])
    apps.patch_application(
        row["application_id"], jd_id=artifact["jd_id"],
        jd_version_id=artifact["jd_version_id"], match_id=artifact["match_id"],
    )
    import job_discovery
    monkeypatch.setattr(job_discovery, "discover", lambda raw="", url="": {
        "source_type": "manual_paste", "source_credibility": "B", "jd_analysis": {},
        "company": "", "title": "", "location": "", "job_type": "",
        "skills_detected": [], "duties": "", "requirements": "", "career_domain": "",
        "role_family": "", "source_url": url, "raw_chars": len(raw),
    })

    preview = jd.preview_from_jd(JD_TEXT)

    candidate = next(x for x in preview["existing_candidates"]
                     if x["application_id"] == row["application_id"])
    assert candidate["same_content"] is True
    assert candidate["reason"] == "JD 内容完全一致"


def test_approved_jd_version_and_match_are_immutable_artifacts(isolated):
    first = _artifact()
    second = _artifact(jd_id=first["jd_id"])
    assert first["jd_version_id"] == second["jd_version_id"]
    assert first["match_id"] != second["match_id"]
    assert os.path.exists(first["snapshot_path"])
    assert os.path.exists(first["match_path"])
    assert os.path.exists(second["match_path"])
    versions = jd.list_versions(first["jd_id"])
    assert len(versions) == 1


def test_approved_artifacts_reuse_discovered_jd_analysis(isolated, monkeypatch):
    import job_discovery

    analysis = {
        "schema_version": "test", "input_hash": "hash",
        "source": "deterministic", "keywords": [], "requirements": [],
    }
    monkeypatch.setattr(job_discovery, "discover", lambda raw="", url="": {
        "source_type": "manual_paste", "source_credibility": "B",
        "jd_analysis": analysis,
    })
    seen = []
    monkeypatch.setattr(jd, "_match_payload", lambda text, title, jd_analysis=None: (
        seen.append(jd_analysis) or {
            "status": "当前适合投递", "direction": "Agent 应用工程",
            "summary": "ok", "gap_list": {}, "suggestions": [],
            "requirement_analysis": {},
        }
    ))
    out = _artifact()
    assert out["status"] == "ok"
    assert seen == [analysis]


def test_confirm_reuses_the_server_side_preview_result(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(jd, "_match_payload", lambda text, title, jd_analysis=None: (
        calls.append((text, title)) or {
            "status": "当前适合投递", "direction": "Agent 应用工程", "summary": "ok",
            "gap_list": {}, "suggestions": [], "requirement_analysis": {},
        }
    ))

    preview = jd.preview_from_jd(JD_TEXT, source_url="https://career.huawei.com/job/1")
    out = jd.prepare_approved_artifacts(
        application_id="app_preview_reuse", jd_text=JD_TEXT,
        company="华为计算产品线", position="AI应用工程师", location="上海",
        source_url="https://career.huawei.com/job/1",
        expected_hash=preview["content_hash"], preview_id=preview["preview_id"],
    )

    assert out["status"] == "ok"
    assert out["preview_reused"] is True
    assert len(calls) == 1


def test_bound_jd_loader_uses_only_active_application_version(isolated):
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    artifact = _artifact(application_id=row["application_id"])
    apps.patch_application(
        row["application_id"], jd_id=artifact["jd_id"],
        jd_version_id=artifact["jd_version_id"], match_id=artifact["match_id"],
    )

    bound = jd.load_bound_jd(row["application_id"], artifact["jd_version_id"])
    assert bound["application_id"] == row["application_id"]
    assert bound["jd_version_id"] == artifact["jd_version_id"]
    assert "向量检索" in bound["snapshot"]["jd_text"]
    with pytest.raises(RuntimeError, match="JD 已更新"):
        jd.load_bound_jd(row["application_id"], "jdv_stale")


def test_changed_text_creates_new_version(isolated):
    first = _artifact()
    changed = _artifact(text=JD_TEXT + "\n加分项：熟悉 LLMOps。", jd_id=first["jd_id"])
    assert changed["jd_version_id"] != first["jd_version_id"]
    assert len(jd.list_versions(first["jd_id"])) == 2


def test_application_plan_target_keeps_full_provenance(isolated):
    apps.ensure_application_schema()
    app_row = apps.application_fact_views()[0]
    art = _artifact(application_id=app_row["application_id"])
    out = apps.upsert_application(
        app_row["company"], app_row["position"], app_row["status"],
        application_id=app_row["application_id"], jd_id=art["jd_id"],
        jd_version_id=art["jd_version_id"], match_id=art["match_id"],
        source_url="https://career.huawei.com/job/1",
        match_conclusion=art["match"]["status"], audience=art["match"]["direction"],
        include_in_plan=True, plan_priority="high",
    )
    assert out["status"] == "ok"
    targets = jd.plan_targets()
    assert targets["total"] == 1
    target = targets["included"][0]
    assert target["application_id"] == app_row["application_id"]
    assert target["match"]["gap_list"]["技能缺口"]
    assert art["jd_version_id"] in jd.plan_gaps_text()


def test_plan_targets_collapse_an_exact_active_jd_duplicate(isolated):
    apps.ensure_application_schema()
    first = apps.application_fact_views()[0]
    original = _artifact(application_id=first["application_id"])
    apps.patch_application(
        first["application_id"], jd_id=original["jd_id"],
        jd_version_id=original["jd_version_id"], match_id=original["match_id"],
        include_in_plan=True, plan_priority="high",
    )
    duplicate = apps.upsert_application(
        first["company"], first["position"], "准备投递", force_new=True,
        include_in_plan=False, plan_priority="low",
    )
    duplicate_artifact = _artifact(
        application_id=duplicate["application_id"], jd_id=original["jd_id"],
    )
    apps.patch_application(
        duplicate["application_id"], jd_id=duplicate_artifact["jd_id"],
        jd_version_id=duplicate_artifact["jd_version_id"],
        match_id=duplicate_artifact["match_id"], include_in_plan=True,
        plan_priority="low",
    )

    targets = jd.plan_targets()

    assert [target["application_id"] for target in targets["included"]] == [
        first["application_id"]
    ]
    collapsed = next(target for target in targets["excluded"]
                     if target["application_id"] == duplicate["application_id"])
    assert collapsed["duplicate_of"] == first["application_id"]
    assert "重复保存" in collapsed["excluded_reason"]


def test_bare_application_cannot_enter_jd_driven_plan(isolated):
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    out = apps.patch_application(row["application_id"], include_in_plan=True)
    assert out["status"] == "error"
    assert "JD" in out["error"]


def test_duplicate_pair_requires_id_in_legacy_upsert(isolated):
    apps.ensure_application_schema()
    apps.upsert_application("华为计算产品线", "AI应用工程师", "已评估", force_new=True)
    out = apps.upsert_application("华为计算产品线", "AI应用工程师", "面试中")
    assert out["status"] == "conflict"
    assert len(out["candidates"]) == 2


def test_plan_snapshot_becomes_stale_without_rewriting_plan(isolated):
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    art = _artifact(application_id=row["application_id"])
    apps.patch_application(
        row["application_id"], jd_id=art["jd_id"], jd_version_id=art["jd_version_id"],
        match_id=art["match_id"], include_in_plan=True,
    )
    from plan_gen import append_target_trace, plan_target_status
    plan = append_target_trace("## Week 1 主题：RAG\n")
    assert plan_target_status(plan)["stale"] is False
    apps.patch_application(row["application_id"], include_in_plan=False)
    status = plan_target_status(plan)
    assert status["stale"] is True
    assert "Week 1" in plan


def test_preview_and_commit_api_link_existing_application(isolated):
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    client = TestClient(app)
    preview = client.post("/api/applications/from-jd/preview", json={
        "jd_text": JD_TEXT, "source_url": "https://career.huawei.com/job/1",
        "application_id": row["application_id"],
    })
    assert preview.status_code == 200
    content_hash = preview.json()["content_hash"]
    committed = client.post("/api/applications/from-jd/commit", json={
        "jd_text": JD_TEXT, "source_url": "https://career.huawei.com/job/1",
        "expected_content_hash": content_hash, "mode": "link_existing",
        "application_id": row["application_id"], "company": row["company"],
        "position": row["position"], "include_in_plan": True,
        "plan_priority": "high",
    })
    assert committed.status_code == 200, committed.text
    saved = apps.get_application(row["application_id"])
    assert saved["jd_version_id"] == committed.json()["jd_version_id"]
    assert saved["include_in_plan"] is True


def test_commit_rejects_a_second_active_row_for_the_same_jd(isolated):
    apps.ensure_application_schema()
    client = TestClient(app)
    preview = client.post("/api/applications/from-jd/preview", json={
        "jd_text": JD_TEXT, "source_url": "https://career.huawei.com/job/1",
    })
    assert preview.status_code == 200
    payload = {
        "jd_text": JD_TEXT, "source_url": "https://career.huawei.com/job/1",
        "expected_content_hash": preview.json()["content_hash"], "mode": "create",
        "company": "华为计算产品线", "position": "AI应用工程师",
        "status": "准备投递", "include_in_plan": True, "plan_priority": "high",
    }
    operation_suffix = str(abs(hash(str(isolated))))
    first = client.post("/api/applications/from-jd/commit", json={
        **payload, "operation_id": "jd-duplicate-first-" + operation_suffix,
    })
    assert first.status_code == 200, first.text

    second = client.post("/api/applications/from-jd/commit", json={
        **payload, "operation_id": "jd-duplicate-second-" + operation_suffix,
    })
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["status"] == "duplicate"
    assert detail["existing_application"]["application_id"] == first.json()["application_id"]
    assert len(apps.application_fact_views()) == 2  # legacy row + this confirmed JD row


def test_commit_replay_with_one_operation_id_keeps_one_application(isolated):
    apps.ensure_application_schema()
    client = TestClient(app)
    unique_jd = JD_TEXT.replace("华为计算产品线", "测试企业").replace("AI应用工程师", "RAG工程师")
    preview = client.post("/api/applications/from-jd/preview", json={
        "jd_text": unique_jd, "source_url": "https://career.example.com/job/1",
    })
    assert preview.status_code == 200
    payload = {
        "jd_text": unique_jd, "source_url": "https://career.example.com/job/1",
        "expected_content_hash": preview.json()["content_hash"], "mode": "create",
        "company": "测试企业", "position": "RAG工程师", "status": "准备投递",
        "include_in_plan": True, "plan_priority": "high",
        "operation_id": "jd-replay-once-" + str(abs(hash(str(isolated)))),
    }
    first = client.post("/api/applications/from-jd/commit", json=payload)
    assert first.status_code == 200, first.text
    second = client.post("/api/applications/from-jd/commit", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["application_id"] == first.json()["application_id"]
    assert len([row for row in apps.application_fact_views()
                if row["company"] == "测试企业" and row["position"] == "RAG工程师"]) == 1


def test_experience_records_application_and_jd_ids(isolated, tmp_path, monkeypatch):
    monkeypatch.setattr(apps, "EXPERIENCE_DIR", str(tmp_path / "experience"))
    out = apps.save_experience(
        "华为计算产品线", "AI应用工程师", "一面",
        "本轮重点追问了检索评测、重排指标与线上问题定位，需要继续强化实验设计。",
        application_id="app_x", jd_version_id="jdv_x",
    )
    body = open(out["saved_abs"], encoding="utf-8").read()
    assert 'application_id: "app_x"' in body
    assert 'jd_version_id: "jdv_x"' in body


def test_rag_route_reads_only_the_bound_jd(isolated, monkeypatch):
    # This test verifies bound-JD execution, not the online route model. Keep it
    # deterministic even when a developer's .env enables intelligent routing.
    monkeypatch.setenv("RAG_ROUTER_MODE", "legacy")
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    art = _artifact(application_id=row["application_id"])
    apps.patch_application(
        row["application_id"], jd_id=art["jd_id"], jd_version_id=art["jd_version_id"],
        match_id=art["match_id"], include_in_plan=True,
    )
    from rag_multi_source import execute_plan
    from rag_query_plan import rule_plan_query as plan_query
    plan = plan_query("我投递的华为对应 JD 要求哪些核心能力？")
    assert any(r.source == "application_jd" for r in plan.routes)
    out = execute_plan(plan, "我投递的华为对应 JD 要求哪些核心能力？", 5,
                       retrieve=lambda *a, **k: {"in_kb": False})
    bound = [x for x in out.evidence if x.route == "application_jd"]
    assert len(bound) == 1, out.source_status
    assert art["jd_version_id"] in bound[0].text
    assert "向量检索" in bound[0].text or "RAG" in bound[0].text


def test_explicit_jd_version_hard_route_never_falls_back_to_plan_targets(
        isolated, monkeypatch):
    apps.ensure_application_schema()
    row = apps.application_fact_views()[0]
    art = _artifact(application_id=row["application_id"])
    from rag_multi_source import execute_plan
    from rag_query_plan import plan_structured_read
    plan = plan_structured_read(
        "application_jd", "get_bound_jd",
        filters={"stable_id": art["jd_version_id"]},
        subquery="查看指定 JD 版本的完整要求",
    )
    assert plan.planner_engine == "structured_read"
    assert plan.routes[0].filters["stable_id"] == art["jd_version_id"]

    out = execute_plan(plan, "查看指定 JD", 5,
                       retrieve=lambda *a, **k: {"in_kb": False})
    bound = [item for item in out.evidence if item.route == "application_jd"]
    assert len(bound) == 1
    assert bound[0].metadata["jd_version_id"] == art["jd_version_id"]
    assert "向量检索" in bound[0].text


def test_shared_gap_is_merged_but_retains_both_sources(isolated):
    apps.ensure_application_schema()
    first = apps.application_fact_views()[0]
    art1 = _artifact(application_id=first["application_id"])
    apps.patch_application(first["application_id"], jd_id=art1["jd_id"],
                           jd_version_id=art1["jd_version_id"], match_id=art1["match_id"],
                           include_in_plan=True)
    second_id = "app_second"
    art2 = _artifact(application_id=second_id)
    apps.upsert_application(
        "测试企业", "Agent工程师", "准备投递", application_id=second_id, force_new=True,
        jd_id=art2["jd_id"], jd_version_id=art2["jd_version_id"], match_id=art2["match_id"],
        include_in_plan=True,
    )
    items = jd.plan_gap_items()
    shared = next(x for x in items if "RAG 评测" in x["text"])
    assert shared["count"] == 2
    assert {x["application_id"] for x in shared["sources"]} == {
        first["application_id"], second_id,
    }
