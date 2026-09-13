# -*- coding: utf-8 -*-


def test_personal_project_memory_requires_pending_review_before_promotion(tmp_path, monkeypatch):
    import knowledge_crawler as kc

    kb = tmp_path / "knowledge_base"
    pending = kb / "_pending" / "web"
    pending.mkdir(parents=True)
    monkeypatch.setattr(kc, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(kc, "KB_DIR", str(kb))
    monkeypatch.setattr(kc, "PENDING_WEB_DIR", str(pending))

    text = "这是我的 RAG 项目材料。" + "包含混合检索、重排、评测与可观测性实现细节。" * 8
    staged = kc.stage_personal_memory(text, "个人 RAG 项目", "project_context")
    assert staged["status"] == "ok"
    assert staged["suggested_subdir"] == "project_context"
    raw = (tmp_path / staged["saved"]).read_text(encoding="utf-8")
    assert 'review_status: "pending"' in raw
    assert 'owner_scope: "personal"' in raw
    assert not (kb / "project_context").exists()

    promoted = kc.cmd_promote(staged["saved"], "project_context", ingest=False)
    assert promoted["status"] == "ok" and promoted["source_type"] == "project_context"
    final = (tmp_path / promoted["promoted_to"]).read_text(encoding="utf-8")
    assert 'review_status: "approved"' in final
    assert 'owner_scope: "personal"' in final


def test_personal_memory_subdirs_have_distinct_source_types():
    import knowledge_crawler as kc
    from rag_ingest import _infer_source_type

    assert kc.SUBDIR_SOURCE_TYPE["project_context"] == "project_context"
    assert kc.SUBDIR_SOURCE_TYPE["resume_rules"] == "resume_rule"
    assert _infer_source_type("knowledge_base/project_context/a.md") == "project_context"
    assert _infer_source_type("knowledge_base/resume_rules/a.md") == "resume_rule"
