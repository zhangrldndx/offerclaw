# -*- coding: utf-8 -*-
"""P0.1 Versioned Source Upsert:同源原子替换——先算后换,失败保旧,过期块清理。

用 chromadb 内存客户端 + 无 key 伪向量路径,零网络零真实库。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_ingest  # noqa: E402


PAD = "分块器要求最小长度,此处填充稳定语料:检索增强生成、向量数据库、混合检索、精排融合、拒答门控、评测基线、增量索引、内容寻址、原子替换、版本治理。" * 2


@pytest.fixture()
def col():
    import uuid

    import chromadb
    return chromadb.Client().create_collection("t_upsert_" + uuid.uuid4().hex[:8])


def _write(tmp_path, body):
    p = tmp_path / "doc_a.md"
    p.write_text("# 标题\n\n" + body, encoding="utf-8")
    return str(p)


def test_replace_removes_stale_keeps_unchanged(tmp_path, col, monkeypatch):
    monkeypatch.setattr(rag_ingest, "has_embedding_api_key", lambda: False)  # 伪向量路径
    f1 = _write(tmp_path, "## 甲\n第一段内容保持不变。" + PAD + "\n\n## 乙\n第二段旧版本。" + PAD)
    rag_ingest.ingest_file(f1, col, source_type="doc")
    old_ids = set(col.get(where={"source": "doc_a.md"})["ids"])
    assert old_ids
    f2 = _write(tmp_path, "## 甲\n第一段内容保持不变。" + PAD + "\n\n## 乙\n第二段新版本已改写。" + PAD)
    out = rag_ingest.replace_source(f2, col, source_type="doc")
    assert out["status"] == "ok" and out["stale_removed"] >= 1
    new_ids = set(col.get(where={"source": "doc_a.md"})["ids"])
    assert old_ids & new_ids                    # 未变块 ID 保留(内容寻址)
    assert not (old_ids - new_ids) & new_ids    # 过期块已清
    docs = col.get(where={"source": "doc_a.md"})["documents"]
    assert not any("旧版本" in d for d in docs) and any("新版本" in d for d in docs)


def test_replace_failure_keeps_old_intact(tmp_path, col, monkeypatch):
    monkeypatch.setattr(rag_ingest, "has_embedding_api_key", lambda: False)
    f1 = _write(tmp_path, "## 甲\n初版内容甲。" + PAD + "\n\n## 乙\n初版内容乙。" + PAD)
    rag_ingest.ingest_file(f1, col, source_type="doc")
    before = set(col.get(where={"source": "doc_a.md"})["ids"])

    def _boom(*a, **k):
        raise RuntimeError("模拟 embedding 失败")
    monkeypatch.setattr(rag_ingest, "fake_embedding", _boom)
    f2 = _write(tmp_path, "## 甲\n全新内容触发重嵌入改写数据。" + PAD + "\n\n## 乙\n另一段全新内容同样改写。" + PAD)
    with pytest.raises(RuntimeError):
        rag_ingest.replace_source(f2, col, source_type="doc")
    after = set(col.get(where={"source": "doc_a.md"})["ids"])
    assert after == before                      # 失败:旧数据分毫未动(先算后换)


def test_replace_identical_noop(tmp_path, col, monkeypatch):
    monkeypatch.setattr(rag_ingest, "has_embedding_api_key", lambda: False)
    f1 = _write(tmp_path, "## 甲\n内容一致无变化。" + PAD)
    rag_ingest.ingest_file(f1, col, source_type="doc")
    n = col.count()
    out = rag_ingest.replace_source(f1, col, source_type="doc")
    assert out["stale_removed"] == 0 and col.count() == n


def test_short_profile_sections_are_ingested_and_safely_replaced(tmp_path, col, monkeypatch):
    monkeypatch.setattr(rag_ingest, "has_embedding_api_key", lambda: False)
    path = tmp_path / "user_profile.md"
    path.write_text(
        "# 合成画像\n\n## 1. 基本信息\n- 所在地：上海\n- 学历：本科\n\n"
        "## 3. 技能清单\n- Python 工程：2/5\n- RAG 工程：2/5\n",
        encoding="utf-8",
    )
    first = rag_ingest.ingest_file(str(path), col, source_type="profile")
    old_ids = set(col.get(where={"source": "user_profile.md"})["ids"])
    assert first["status"] == "ok" and old_ids

    path.write_text(
        "# 合成画像\n\n## 1. 基本信息\n- 所在地：上海\n- 学历：本科\n\n"
        "## 3. 技能清单\n- Python 工程：3/5\n- RAG 工程：2/5\n",
        encoding="utf-8",
    )
    out = rag_ingest.replace_source(str(path), col, source_type="profile")
    new_ids = set(col.get(where={"source": "user_profile.md"})["ids"])

    assert out["status"] == "ok" and out["stale_removed"] >= 1
    assert new_ids and new_ids != old_ids
    assert any("Python 工程：3/5" in doc for doc in col.get(
        where={"source": "user_profile.md"}, include=["documents"],
    )["documents"])


def test_personal_entity_metadata_backfills_without_reembedding(tmp_path, col, monkeypatch):
    monkeypatch.setattr(rag_ingest, "has_embedding_api_key", lambda: False)
    path = tmp_path / "experience.md"
    body = "# 亲历投递经验：甲公司 · AI工程师\n\n" + PAD
    path.write_text(
        '---\ncompany: "甲公司"\nposition: "AI工程师"\nstage: "技术面"\n---\n\n' + body,
        encoding="utf-8",
    )
    parser = rag_ingest._frontmatter_entity_fields
    monkeypatch.setattr(rag_ingest, "_frontmatter_entity_fields", lambda _text: {})
    rag_ingest.ingest_file(str(path), col, source_type="experience")
    before = col.count()
    monkeypatch.setattr(rag_ingest, "_frontmatter_entity_fields", parser)
    out = rag_ingest.ingest_file(str(path), col, source_type="experience")
    assert col.count() == before
    assert out["metadata_updated"] >= 1
    metas = col.get(where={"source": "experience.md"}, include=["metadatas"])["metadatas"]
    assert all(meta["company"] == "甲公司" and meta["position"] == "AI工程师"
               for meta in metas)
