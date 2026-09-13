# -*- coding: utf-8 -*-
"""P0.3 Index Fingerprint:指标与索引版本强绑定(方案文档),字段齐全且随 env 变化。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fingerprint_fields_and_env_sensitivity(monkeypatch):
    from rag_tools import index_fingerprint
    fp = index_fingerprint()
    for k in ("git_commit", "collection", "embedding_provider", "embedding_model",
              "rerank_model", "generated_at"):
        assert k in fp
    monkeypatch.setenv("RAG_COLLECTION_NAME", "fp_probe_collection")
    monkeypatch.setenv("OFFERCLAW_EMBED_MAX_SEQ", "512")
    fp2 = index_fingerprint()
    assert fp2["collection"] == "fp_probe_collection"   # 指纹随激活配置走
    assert fp2["embed_max_seq"] == "512"
    assert fp2["collection_count"] is None              # 不存在的集合 → 诚实 None,不编数


def test_content_fingerprint_detects_same_count_in_place_change():
    import rag_tools

    class MutableCollection:
        def __init__(self):
            self.documents = ["first body", "second body"]
            self.fingerprint_revision = 1

        def count(self):
            return 2

        def get(self, **_kwargs):
            return {
                "ids": ["stable-a", "stable-b"],
                "documents": list(self.documents),
                "metadatas": [
                    {"source": "a.md", "chunker_version": "v1"},
                    {"source": "b.md", "chunker_version": "v1"},
                ],
            }

    collection = MutableCollection()
    rag_tools._INDEX_FP_CACHE.clear()
    before = rag_tools.index_fingerprint(collection=collection)
    collection.documents[0] = "changed body with the same id and count"
    collection.fingerprint_revision += 1
    after = rag_tools.index_fingerprint(collection=collection)

    assert before["collection_count"] == after["collection_count"] == 2
    assert before["collection_content_hash"] != after["collection_content_hash"]
    assert before["fingerprint_id"] != after["fingerprint_id"]
    assert before["audit_git_commit"] == after["audit_git_commit"]
