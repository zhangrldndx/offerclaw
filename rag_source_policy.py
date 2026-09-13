# -*- coding: utf-8 -*-
"""RAG 数据归属与路由隔离策略。

事实文件、个人向量记忆和策展知识必须是三个互斥的检索域。Chroma 是派生索引；
即使旧索引缺少 ``owner_scope``，也先按 ``source_type`` 做保守推断，并由回填脚本
补齐元数据。
"""

from __future__ import annotations

import os


PERSONAL_VECTOR_SOURCE_TYPES = frozenset({
    "experience", "project_context", "resume_rule", "resume", "story",
})
CURATED_SOURCE_TYPES = frozenset({
    "resource", "career_knowledge", "career_path", "learning_resource",
    "feishu_wiki", "doc",
})
INTERNAL_SOURCE_TYPES = frozenset({
    "application", "profile", "log", "system", "jd", "verification",
    "application_jd", "paper",
})
NON_RAG_SOURCE_TYPES = frozenset({"jd", "application_jd"})
NON_RAG_SOURCE_BASENAMES = frozenset({"jd_candidates.md"})

PERSONAL_SOURCE_BASENAMES = {
    "project_one_pager.md": "project_context",
    "verification_report.md": "project_context",
    "interview_story_bank.md": "project_context",
}


def normalized_source_type(source_type: str, source: str = "") -> str:
    basename = os.path.basename(str(source or ""))
    return PERSONAL_SOURCE_BASENAMES.get(basename, str(source_type or "doc").strip())


def rag_source_excluded(source_type: str, source: str = "") -> bool:
    """Return whether a source belongs to a business record, not the RAG corpus."""
    basename = os.path.basename(str(source or ""))
    normalized = normalized_source_type(source_type, source)
    return (normalized in NON_RAG_SOURCE_TYPES
            or basename in NON_RAG_SOURCE_BASENAMES)


def infer_owner_scope(source_type: str, source: str = "", declared: str = "") -> str:
    declared = str(declared or "").strip().lower()
    if declared in {"personal", "curated", "internal"}:
        return declared
    source_type = normalized_source_type(source_type, source)
    if source_type in PERSONAL_VECTOR_SOURCE_TYPES:
        return "personal"
    if source_type in CURATED_SOURCE_TYPES:
        return "curated"
    return "internal"


def evidence_allowed(route: str, source_type: str, owner_scope: str = "",
                     source: str = "") -> bool:
    source_type = normalized_source_type(source_type, source)
    if rag_source_excluded(source_type, source):
        return False
    owner_scope = infer_owner_scope(source_type, source, owner_scope)
    if route == "application_experience":
        return owner_scope == "personal" and source_type == "experience"
    if route == "project_memory":
        return owner_scope == "personal" and source_type == "project_context"
    if route == "resume_rules":
        return owner_scope == "personal" and source_type in {"resume_rule", "resume"}
    if route == "reference_kb":
        return owner_scope == "curated" and source_type in CURATED_SOURCE_TYPES
    if route == "paper_kb":
        return source_type == "paper"
    return True
