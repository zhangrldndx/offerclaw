# -*- coding: utf-8 -*-
from pathlib import Path

from memory_layers import EpisodicMemory
from resume_project import load_approved_project_evidence


def test_only_approved_project_sources_and_artifacts_are_loaded(tmp_path, monkeypatch):
    project_context = tmp_path / "knowledge_base" / "project_context"
    project_context.mkdir(parents=True)
    approved = project_context / "approved.md"
    approved.write_text(
        "---\ntitle: \"已确认项目素材\"\nsource_type: \"project_context\"\n"
        "review_status: \"approved\"\n---\n\n# 已确认项目素材\n\n实现可验证检索评测。",
        encoding="utf-8",
    )
    (project_context / "pending.md").write_text(
        "---\ntitle: \"未确认素材\"\nsource_type: \"project_context\"\n"
        "review_status: \"pending\"\n---\n\n# 未确认素材\n\n不应进入简历。",
        encoding="utf-8",
    )

    draft_root = tmp_path / "resume_drafts"
    project_draft = draft_root / "projects" / "demo" / "project_section_ok.md"
    project_draft.parent.mkdir(parents=True)
    project_draft.write_text("# 项目经历\n\n## 项目经历\n\n- 已批准版本。", encoding="utf-8")
    (project_draft.parent / "unreviewed.md").write_text(
        "# 不应读取的普通草稿", encoding="utf-8",
    )

    memory_dir = tmp_path / "memory"
    monkeypatch.setenv("OFFERCLAW_MEMORY_DIR", str(memory_dir))
    episodic = EpisodicMemory()
    content = project_draft.read_text(encoding="utf-8")
    snapshot = episodic.store.put_snapshot(
        content, media_type="text/markdown",
        source_path="resume_drafts/projects/demo/project_section_ok.md",
    )
    episodic.append({
        "kind": "resume_artifact_changed",
        "action": "approved",
        "snapshot_id": snapshot["snapshot_id"],
        "content_hash": snapshot["content_hash"],
        "saved_path": "resume_drafts/projects/demo/project_section_ok.md",
        "actor": "user",
        "source": "agent_approval",
        "entity_type": "resume_artifact",
        "entity_id": "resume_project_approved",
    }, export=False, index=False)

    items = load_approved_project_evidence(
        base_dir=str(tmp_path), resume_draft_dir=str(draft_root),
    )
    refs = {item["evidence_ref"] for item in items}
    combined = "\n".join(item["content"] for item in items)

    assert "project_section:resume_project_approved" in refs
    assert any(ref.startswith("project_context:") for ref in refs)
    assert "已确认项目素材" in combined
    assert "已批准版本" in combined
    assert "未确认素材" not in combined
    assert "不应读取的普通草稿" not in combined
