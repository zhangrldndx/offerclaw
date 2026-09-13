import math

import rag_tools


def test_fake_embedding_never_nan_or_inf():
    """占位伪向量必须恒为有限值——否则 ChromaDB 会拒绝写入、无 key 离线 ingest 崩溃。

    回归：旧实现用 struct.unpack('f') 把哈希字节当 IEEE-754 float32，约 95% 概率
    产出 NaN/Inf。这里抽样多条不同文本，断言全部有限且落在 [0,1]。
    """
    for i in range(500):
        vec = rag_tools.fake_embedding(f"chunk {i} RAG ReAct 大模型 LoRA")
        assert all(math.isfinite(x) for x in vec), "伪向量含 NaN/Inf"
        assert all(0.0 <= x <= 1.0 for x in vec), "伪向量越界 [0,1]"


def test_fake_embedding_is_deterministic():
    assert rag_tools.fake_embedding("同一文本") == rag_tools.fake_embedding("同一文本")


def test_bailian_embedding_collection_name(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bailian")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.delenv("RAG_COLLECTION_NAME", raising=False)

    assert rag_tools.get_collection_name() == "offerclaw_bailian_text_embedding_v4_1024"


def test_legacy_zhipu_collection_name(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "zhipu")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-3")
    monkeypatch.delenv("RAG_COLLECTION_NAME", raising=False)

    assert rag_tools.get_collection_name() == "offerclaw_docs"


def test_fake_embedding_uses_configured_dimension(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bailian")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")

    vec = rag_tools.fake_embedding("hello")

    assert len(vec) == 1024
