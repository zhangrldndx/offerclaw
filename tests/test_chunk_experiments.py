from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import rag_chunk_experiments
from rag_chunk_experiments import (
    C1_SPEC,
    C2_SPEC,
    ChunkExperimentError,
    QrelsMappingError,
    RegexTokenOffsetCodec,
    SourceDocument,
    assert_isolated_collection_name,
    build_experiment_plan,
    build_mixed_targeted_experiment_plan,
    build_snapshot_experiment_plan,
    build_snapshot_targeted_experiment_plan,
    ensure_experiment_collection,
    expand_parent_context,
    map_qrels_to_experiment,
    plan_statistics,
    production_sources,
    select_mixed_target_sources,
    select_snapshot_target_chunks,
    split_document_experiment,
    write_experiment_collection,
)
from rag_qrels import answer_span_hash
from eval_chunk_experiment import (
    _append_parent_expandable_metrics,
    _parent_expandable_child_ids,
    validate_evaluation_inputs,
)
from rag_qrels import validate_qrels_against_collection


def _source(tmp_path: Path, body: str, name: str = "source.md") -> SourceDocument:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return SourceDocument(str(path), name, "resource", "general")


def _long_markdown(tokens: int = 900) -> str:
    return "# 文档标题\n\n## 第一章\n\n### 细节\n\n" + " ".join(f"word{i}" for i in range(tokens))


@pytest.mark.parametrize("spec", [C1_SPEC, C2_SPEC])
def test_token_windows_have_real_overlap_and_hard_body_limit(tmp_path: Path, spec) -> None:
    source = _source(tmp_path, _long_markdown(1200))
    chunks, _parents = split_document_experiment(
        source, Path(source.path).read_text(encoding="utf-8"), spec, RegexTokenOffsetCodec(),
    )
    assert len(chunks) >= 3
    assert max(chunk.body_token_count for chunk in chunks) <= spec.body_max_tokens
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.token_ids[-spec.overlap_tokens :] == current.token_ids[: spec.overlap_tokens]
        assert current.token_start == previous.token_end - spec.overlap_tokens


def test_single_unbroken_paragraph_is_hard_split(tmp_path: Path) -> None:
    # No blank lines or sentence boundaries: token windows must still enforce 320.
    source = _source(tmp_path, "# T\n## H\n" + "测" * 901)
    chunks, _ = split_document_experiment(
        source, Path(source.path).read_text(encoding="utf-8"), C1_SPEC, RegexTokenOffsetCodec(),
    )
    assert len(chunks) >= 3
    assert all(chunk.body_token_count <= 320 for chunk in chunks)


def test_repeated_identical_section_occurrences_have_unique_stable_ids(tmp_path: Path) -> None:
    repeated = "相同正文内容" * 30
    source = _source(
        tmp_path,
        f"# 根\n## 重复\n{repeated}\n## 重复\n{repeated}",
    )
    text = Path(source.path).read_text(encoding="utf-8")
    first_chunks, _ = split_document_experiment(source, text, C1_SPEC, RegexTokenOffsetCodec())
    second_chunks, _ = split_document_experiment(source, text, C1_SPEC, RegexTokenOffsetCodec())
    assert len(first_chunks) == 2
    assert len({chunk.parent_id for chunk in first_chunks}) == 2
    assert len({chunk.chunk_id for chunk in first_chunks}) == 2
    assert [chunk.parent_id for chunk in first_chunks] == [chunk.parent_id for chunk in second_chunks]
    assert [chunk.chunk_id for chunk in first_chunks] == [chunk.chunk_id for chunk in second_chunks]


def test_breadcrumb_and_parent_context_are_stable(tmp_path: Path) -> None:
    source = _source(tmp_path, "# 根标题\n## 二级\n### 三级\n" + "正文内容" * 30)
    plan = build_experiment_plan(
        [source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod",
    )
    chunk = plan.chunks[0]
    assert chunk.document_title == "根标题"
    assert chunk.heading_path == ("根标题", "二级", "三级")
    assert chunk.breadcrumb == "根标题 > 二级 > 三级"
    assert chunk.embedding_text.startswith(chunk.breadcrumb + "\n")
    assert expand_parent_context([chunk.chunk_id], plan)[chunk.chunk_id].startswith("正文内容")
    stats = plan_statistics(plan)
    assert stats["body_tokens"]["hard_limit_violations"] == 0
    assert stats["overlap"]["violations"] == 0


def test_experiment_does_not_change_production_splitter_defaults(tmp_path: Path) -> None:
    from rag_tools import CHUNK_OVERLAP, CHUNK_SIZE, split_markdown_document

    before = (CHUNK_SIZE, CHUNK_OVERLAP, split_markdown_document.__defaults__)
    source = _source(tmp_path, _long_markdown())
    build_experiment_plan([source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod")
    after = (CHUNK_SIZE, CHUNK_OVERLAP, split_markdown_document.__defaults__)
    assert before == after
    assert before[:2] == (800, 80)


class _FakeCollection:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata = metadata or {}
        self.ids = []
        self.metadatas = []
        self.documents = []
        self.embeddings = []
        self.add_batch_sizes = []
        self.add_calls = 0
        self.fail_after_commit_on_call = None

    def get(self, ids=None, include=None):
        selected = list(self.ids) if ids is None else [child_id for child_id in ids if child_id in self.ids]
        positions = [self.ids.index(child_id) for child_id in selected]
        payload = {"ids": selected}
        if include and "metadatas" in include:
            payload["metadatas"] = [
                self.metadatas[index] for index in positions if index < len(self.metadatas)
            ]
        if include and "documents" in include:
            payload["documents"] = [
                self.documents[index] for index in positions if index < len(self.documents)
            ]
        if include and "embeddings" in include:
            payload["embeddings"] = [
                self.embeddings[index] for index in positions if index < len(self.embeddings)
            ]
        return payload

    def add(self, *, ids, embeddings, documents, metadatas):
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        assert not (set(ids) & set(self.ids))
        self.add_calls += 1
        self.add_batch_sizes.append(len(ids))
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        if self.fail_after_commit_on_call == self.add_calls:
            raise RuntimeError("simulated post-commit interruption")

    def count(self):
        return len(self.ids)


class _FakeClient:
    def __init__(self):
        self.collections = {"prod": _FakeCollection("prod")}

    def list_collections(self):
        return list(self.collections.values())

    def create_collection(self, name, metadata):
        assert name != "prod"
        collection = _FakeCollection(name, metadata)
        self.collections[name] = collection
        return collection

    def get_collection(self, name):
        return self.collections[name]


def test_collection_isolation_repeat_safety_and_no_overwrite(tmp_path: Path) -> None:
    source = _source(tmp_path, _long_markdown())
    plan = build_experiment_plan([source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod")
    client = _FakeClient()
    collection, first = ensure_experiment_collection(
        client, plan, allow_approximate_tokenizer=True,
    )
    assert first["created"] is True
    assert collection.name.startswith("offerclaw_exp_chunk_c1_")
    same, second = ensure_experiment_collection(
        client, plan, allow_approximate_tokenizer=True,
    )
    assert same is collection
    assert second["created"] is False
    assert set(client.collections) == {"prod", plan.collection_name}

    collection.ids.append("unexpected")
    with pytest.raises(ChunkExperimentError, match="unexpected IDs"):
        ensure_experiment_collection(client, plan, allow_approximate_tokenizer=True)


def test_existing_collection_corpus_mode_mismatch_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path, _long_markdown())
    plan = build_experiment_plan(
        [source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod",
    )
    client = _FakeClient()
    collection, _ = ensure_experiment_collection(
        client, plan, allow_approximate_tokenizer=True,
    )
    collection.metadata["corpus_mode"] = "snapshot_children"
    with pytest.raises(ChunkExperimentError, match="corpus mode"):
        ensure_experiment_collection(client, plan, allow_approximate_tokenizer=True)


def test_resume_rejects_child_embedding_profile_mismatch(tmp_path: Path) -> None:
    source = _source(tmp_path, _long_markdown())
    plan = build_experiment_plan(
        [source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod",
    )
    client = _FakeClient()
    collection, _ = ensure_experiment_collection(
        client, plan, allow_approximate_tokenizer=True,
    )
    child = plan.chunks[0]
    collection.ids = [child.chunk_id]
    collection.metadatas = [child.chroma_metadata(plan.experiment_fingerprint)]
    collection.metadatas[0]["embed_profile"] = "wrong-vector-space"
    with pytest.raises(ChunkExperimentError, match="embedding profile mismatch"):
        ensure_experiment_collection(client, plan, allow_approximate_tokenizer=True)


def test_resume_rejects_child_metadata_contract_mismatch(tmp_path: Path) -> None:
    source = _source(tmp_path, _long_markdown())
    plan = build_experiment_plan(
        [source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod",
    )
    client = _FakeClient()
    collection, _ = ensure_experiment_collection(
        client, plan, allow_approximate_tokenizer=True,
    )
    child = plan.chunks[0]
    child_metadata = child.chroma_metadata(plan.experiment_fingerprint)
    child_metadata["embed_profile"] = plan.embedding_contract["embed_profile"]
    child_metadata["parent_id"] = "tampered-parent"
    collection.ids = [child.chunk_id]
    collection.metadatas = [child_metadata]
    with pytest.raises(ChunkExperimentError, match=r"metadata mismatch \(parent_id\)"):
        ensure_experiment_collection(client, plan, allow_approximate_tokenizer=True)


class _WriteTokenOffsetCodec(RegexTokenOffsetCodec):
    @property
    def identity(self) -> str:
        return "hf-fast:test-write-codec"


def _write_plan(tmp_path: Path, *, tokens: int = 1200):
    source = _source(tmp_path, _long_markdown(tokens))
    embedding_contract = rag_chunk_experiments.active_embedding_contract()
    embedding_contract["batch_size"] = 64
    return build_experiment_plan(
        [source],
        C1_SPEC,
        _WriteTokenOffsetCodec(),
        production_collection="prod",
        embedding_contract=embedding_contract,
    )


def _fake_vectors(plan, count: int) -> list[list[float]]:
    dimensions = plan.embedding_contract.get("dimensions")
    return [[0.0] * (dimensions if isinstance(dimensions, int) else 3) for _ in range(count)]


def test_collection_write_uses_bounded_embedding_and_add_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_tools

    plan = _write_plan(tmp_path)
    assert len(plan.chunks) > 2
    client = _FakeClient()
    embed_batch_sizes = []

    def fake_embeddings(texts, *, batch_size):
        embed_batch_sizes.append(len(texts))
        assert batch_size == 2
        return _fake_vectors(plan, len(texts))

    monkeypatch.setattr(rag_chunk_experiments, "EXPERIMENT_WRITE_BATCH_SIZE", 2)
    monkeypatch.setattr(
        rag_chunk_experiments,
        "active_embedding_contract",
        lambda: dict(plan.embedding_contract),
    )
    monkeypatch.setattr(rag_tools, "get_embeddings_batch", fake_embeddings)
    state = write_experiment_collection(client, plan)
    collection = client.collections[plan.collection_name]

    assert len(embed_batch_sizes) > 1
    assert max(embed_batch_sizes) <= 2
    assert sum(embed_batch_sizes) == len(plan.chunks)
    assert collection.add_batch_sizes == embed_batch_sizes
    assert collection.ids == [chunk.chunk_id for chunk in plan.chunks]
    assert state == {
        "created": True,
        "existing": 0,
        "missing": len(plan.chunks),
        "complete": True,
        "added": len(plan.chunks),
        "count": len(plan.chunks),
    }


def test_collection_write_resumes_after_post_commit_interruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_tools

    plan = _write_plan(tmp_path)
    assert len(plan.chunks) >= 5
    client = _FakeClient()
    embedded_texts = []

    def fake_embeddings(texts, *, batch_size):
        assert batch_size == 2
        embedded_texts.extend(texts)
        return _fake_vectors(plan, len(texts))

    monkeypatch.setattr(rag_chunk_experiments, "EXPERIMENT_WRITE_BATCH_SIZE", 2)
    monkeypatch.setattr(
        rag_chunk_experiments,
        "active_embedding_contract",
        lambda: dict(plan.embedding_contract),
    )
    monkeypatch.setattr(rag_tools, "get_embeddings_batch", fake_embeddings)
    collection, _ = ensure_experiment_collection(client, plan)
    collection.fail_after_commit_on_call = 2

    with pytest.raises(RuntimeError, match="post-commit interruption"):
        write_experiment_collection(client, plan)
    assert collection.ids == [chunk.chunk_id for chunk in plan.chunks[:4]]

    embedded_texts.clear()
    resumed = write_experiment_collection(client, plan)
    assert embedded_texts == [chunk.embedding_text for chunk in plan.chunks[4:]]
    assert collection.ids == [chunk.chunk_id for chunk in plan.chunks]
    assert resumed["created"] is False
    assert resumed["existing"] == 4
    assert resumed["missing"] == len(plan.chunks) - 4
    assert resumed["added"] == len(plan.chunks) - 4
    assert resumed["complete"] is True


def test_collection_write_rejects_invalid_embedding_before_add(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_tools

    plan = _write_plan(tmp_path, tokens=400)
    client = _FakeClient()
    dimensions = plan.embedding_contract.get("dimensions")

    def invalid_embeddings(texts, *, batch_size):
        if isinstance(dimensions, int):
            return [[0.0] * max(dimensions - 1, 0) for _ in texts]
        return [[0.0], [0.0, 0.0]][:len(texts)]

    monkeypatch.setattr(rag_chunk_experiments, "EXPERIMENT_WRITE_BATCH_SIZE", 2)
    monkeypatch.setattr(
        rag_chunk_experiments,
        "active_embedding_contract",
        lambda: dict(plan.embedding_contract),
    )
    monkeypatch.setattr(rag_tools, "get_embeddings_batch", invalid_embeddings)
    with pytest.raises(ChunkExperimentError, match="embedding"):
        write_experiment_collection(client, plan)
    collection = client.collections[plan.collection_name]
    assert collection.add_calls == 0
    assert collection.count() == 0


def test_collection_write_rejects_full_embedding_contract_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_tools

    plan = _write_plan(tmp_path, tokens=400)
    drifted = {**plan.embedding_contract, "base_url": "https://drift.invalid"}
    monkeypatch.setattr(rag_chunk_experiments, "active_embedding_contract", lambda: drifted)
    monkeypatch.setattr(
        rag_tools,
        "get_embeddings_batch",
        lambda *args, **kwargs: pytest.fail("embedding must not run after contract drift"),
    )
    client = _FakeClient()
    with pytest.raises(ChunkExperimentError, match="embedding contract changed"):
        write_experiment_collection(client, plan)
    assert client.collections[plan.collection_name].count() == 0


def test_collection_write_reuses_identical_prior_experiment_vectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_tools

    plan = _write_plan(tmp_path, tokens=400)
    client = _FakeClient()
    reuse_name = "offerclaw_exp_chunk_c1_priorreuse"
    reuse = _FakeCollection(reuse_name, {
        "embedding_contract_json": rag_chunk_experiments._embedding_contract_json(
            plan.embedding_contract
        ),
    })
    reuse.ids = [chunk.chunk_id for chunk in plan.chunks]
    reuse.documents = [chunk.text for chunk in plan.chunks]
    reuse.embeddings = _fake_vectors(plan, len(plan.chunks))
    reuse.metadatas = [
        {"embed_profile": plan.embedding_contract["embed_profile"]}
        for _chunk in plan.chunks
    ]
    client.collections[reuse_name] = reuse
    monkeypatch.setattr(
        rag_chunk_experiments, "active_embedding_contract",
        lambda: dict(plan.embedding_contract),
    )
    monkeypatch.setattr(
        rag_tools, "get_embeddings_batch",
        lambda *args, **kwargs: pytest.fail("identical vectors must be reused"),
    )

    result = write_experiment_collection(
        client, plan, reuse_embedding_collection=reuse_name,
    )
    assert result["reused_experiment_embeddings"] == len(plan.chunks)
    assert result["reuse_embedding_collection"] == reuse_name
    assert client.collections[plan.collection_name].documents == [
        chunk.text for chunk in plan.chunks
    ]


def test_collection_write_reuse_fails_closed_on_document_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path, tokens=400)
    client = _FakeClient()
    reuse_name = "offerclaw_exp_chunk_c1_badreuse"
    reuse = _FakeCollection(reuse_name, {
        "embedding_contract_json": rag_chunk_experiments._embedding_contract_json(
            plan.embedding_contract
        ),
    })
    reuse.ids = [plan.chunks[0].chunk_id]
    reuse.documents = ["tampered document"]
    reuse.embeddings = _fake_vectors(plan, 1)
    reuse.metadatas = [{"embed_profile": plan.embedding_contract["embed_profile"]}]
    client.collections[reuse_name] = reuse
    monkeypatch.setattr(
        rag_chunk_experiments, "active_embedding_contract",
        lambda: dict(plan.embedding_contract),
    )
    with pytest.raises(ChunkExperimentError, match="reuse document mismatch"):
        write_experiment_collection(
            client, plan, reuse_embedding_collection=reuse_name,
        )
    assert client.collections[plan.collection_name].count() == 0


def test_collection_name_guard_blocks_production_and_unfingerprinted_names() -> None:
    fingerprint = "sha256:" + "a" * 64
    with pytest.raises(ChunkExperimentError, match="production"):
        assert_isolated_collection_name(
            "prod", production_collection="prod", experiment_fingerprint=fingerprint,
        )
    with pytest.raises(ChunkExperimentError, match="isolation prefix"):
        assert_isolated_collection_name(
            "candidate", production_collection="prod", experiment_fingerprint=fingerprint,
        )


class _ManifestCollection:
    def get(self, include):
        return {
            "documents": ["唯一索引证据段落", "另一个段落"],
            "metadatas": [
                {"source": "README.md", "source_type": "doc", "owner_scope": "curated"},
                {"source": "README.md", "source_type": "doc", "owner_scope": "curated"},
            ],
        }


def test_ambiguous_basename_uses_indexed_content_evidence(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "README.md").write_text("无关内容", encoding="utf-8")
    expected = tmp_path / "two" / "README.md"
    expected.write_text("# KB\n唯一索引证据段落\n另一个段落", encoding="utf-8")
    report = {}
    sources = production_sources(tmp_path, _ManifestCollection(), resolution_report=report)
    assert Path(sources[0].path) == expected
    assert report["unique_source_count"] == 1
    assert report["resolved_source_count"] == 1
    assert report["resolution_methods"] == {"indexed_content_evidence": 1}


def _overlay(
    source: str,
    excerpt: str,
    heading_path: list[str],
    *,
    old_chunk_id: str = "old_1",
) -> dict:
    return {
        "schema_version": "rag-qrels-overlay-v1",
        "reviewer_id": "reviewer",
        "base_set": "tests/rag_bench_paraphrase_set.json",
        "index": {"collection": "prod", "count": 1},
        "items": [{
            "query_id": "q1",
            "review_outcome": "accepted",
            "review_note": "reviewed",
            "relevant_targets": [{
                "source": source,
                "heading_path": heading_path,
                "chunk_id": old_chunk_id,
                "answer_span_hash": answer_span_hash(excerpt),
                "evidence_excerpt": excerpt,
                "relevance": "direct",
                "review_note": "direct",
            }],
        }],
    }


def test_qrels_span_and_heading_map_to_new_chunk_ids(tmp_path: Path) -> None:
    excerpt = "这是经过人工审核且能够直接回答问题的证据片段。"
    source = _source(tmp_path, "# 根标题\n## 二级\n### 三级\n" + ("前文。" * 30) + excerpt + ("后文。" * 30))
    plan = build_experiment_plan([source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod")
    mapped, report = map_qrels_to_experiment(
        _overlay(source.source, excerpt, ["根标题", "二级", "三级"]), plan,
    )
    targets = mapped["items"][0]["relevant_targets"]
    assert report["status"] == "mapped"
    assert targets
    assert all(target["chunk_id"].startswith("source_c1_") for target in targets)
    assert all(target["answer_span_hash"] == answer_span_hash(excerpt) for target in targets)
    assert mapped["index"]["collection"] == plan.collection_name


def test_qrels_mapping_failure_is_visible_and_never_silent(tmp_path: Path) -> None:
    excerpt = "不存在于新语料的审核证据"
    source = _source(tmp_path, "# 根标题\n## 二级\n" + "真实正文" * 30)
    plan = build_experiment_plan([source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod")
    with pytest.raises(QrelsMappingError) as captured:
        map_qrels_to_experiment(_overlay(source.source, excerpt, ["根标题", "二级"]), plan)
    assert captured.value.report["status"] == "mapping_failed"
    assert captured.value.report["failures"][0]["reason"] == "span_crosses_chunk_boundary_or_missing"


class _SnapshotCollection:
    def get(self, include):
        return {
            "ids": ["old_a"],
            "documents": [" ".join(f"token{i}" for i in range(700))],
            "metadatas": [{
                "source": "knowledge.md",
                "source_type": "resource",
                "owner_scope": "curated",
                "title": "Section",
                "language": "zh",
                "metadata_allowed": True,
                "stage": 7,
            }],
        }


def test_snapshot_children_preserve_c0_parent_and_are_promotion_eligible_shape() -> None:
    plan = build_snapshot_experiment_plan(
        _SnapshotCollection(), C1_SPEC, RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    assert plan.corpus_mode == "snapshot_children"
    assert len(plan.parents) == 1
    assert len(plan.chunks) >= 3
    assert all(chunk.source == "knowledge.md" for chunk in plan.chunks)
    assert all(chunk.origin_chunk_id == "old_a" for chunk in plan.chunks)
    for chunk in plan.chunks:
        metadata = chunk.chroma_metadata(plan.experiment_fingerprint)
        assert metadata["origin_chunk_id"] == "old_a"
        assert metadata["chunk_id"] == chunk.chunk_id
        assert metadata["parent_content_hash"].startswith("sha256:")
        assert metadata["production_collection"] == "prod"
        assert metadata["language"] == "zh"
        assert metadata["metadata_allowed"] is True
        assert metadata["stage"] == 7
        from rag_retrieval_trace import stable_chunk_id
        assert stable_chunk_id(chunk.text, metadata) == chunk.chunk_id
    assert all(chunk.breadcrumb == "knowledge > Section" for chunk in plan.chunks)
    assert plan.dedup_statistics["skipped_chunks"] == 0
    assert plan_statistics(plan)["overlap"]["violations"] == 0
    assert "inherit short C0 singleton parents" in plan.manifest()["snapshot_min_chars_policy"]


def test_snapshot_keeps_short_c0_singleton_without_creating_tiny_tail() -> None:
    class ShortAndLongCollection:
        def get(self, include):
            return {
                "ids": ["short", "long"],
                "documents": ["短证据", " ".join(f"long{i}" for i in range(650))],
                "metadatas": [
                    {"source": "short.md", "title": "S"},
                    {"source": "long.md", "title": "L"},
                ],
            }

    plan = build_snapshot_experiment_plan(
        ShortAndLongCollection(), C1_SPEC, RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    short_children = [chunk for chunk in plan.chunks if chunk.origin_chunk_id == "short"]
    long_children = [chunk for chunk in plan.chunks if chunk.origin_chunk_id == "long"]
    assert len(short_children) == 1 and short_children[0].text == "短证据"
    assert min(chunk.body_token_count for chunk in long_children) >= C1_SPEC.overlap_tokens


def test_overlap_verification_never_crosses_parent_boundary(tmp_path: Path) -> None:
    first = "甲父块内容" * 40
    second = "乙父块内容" * 40
    source = _source(tmp_path, f"# 根\n## A\n{first}\n## B\n{second}")
    plan = build_experiment_plan(
        [source], C1_SPEC, RegexTokenOffsetCodec(), production_collection="prod",
    )
    assert len({chunk.parent_id for chunk in plan.chunks}) == 2
    stats = plan_statistics(plan)
    assert stats["overlap"]["adjacent_pairs"] == 0
    assert stats["overlap"]["violations"] == 0


def test_snapshot_qrels_lock_exact_origin_and_accept_overlap_children() -> None:
    plan = build_snapshot_experiment_plan(
        _SnapshotCollection(), C1_SPEC, RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    excerpt = " ".join(f"token{i}" for i in range(270, 281))
    mapped, report = map_qrels_to_experiment(
        _overlay(
            "knowledge.md", excerpt, ["semantic", "heading", "drift"],
            old_chunk_id="old_a",
        ),
        plan,
    )
    targets = mapped["items"][0]["relevant_targets"]
    assert len(targets) == 2  # both real-overlap children are equivalent gold
    assert report["failure_count"] == 0
    row = report["targets"][0]
    assert row["origin_parent_locked"] is True
    assert row["locked_origin_chunk_id"] == "old_a"
    assert row["mapping_mode"] == "origin_parent_span_heading_drift"
    assert {chunk.origin_chunk_id for chunk in plan.chunks if chunk.chunk_id in row["new_chunk_ids"]} == {"old_a"}


def test_snapshot_cross_child_span_uses_normalized_80_percent_mapping() -> None:
    plan = build_snapshot_experiment_plan(
        _SnapshotCollection(), C1_SPEC, RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    # This crosses child 0's end; child 1 contains >80%, but no child contains
    # the entire reviewer span.  It must map to one child, not every sibling.
    excerpt = " ".join(f"token{i}" for i in range(250, 371))
    original_hash = answer_span_hash(excerpt)
    mapped, report = map_qrels_to_experiment(
        _overlay(
            "knowledge.md", excerpt, ["knowledge", "Section"],
            old_chunk_id="old_a",
        ),
        plan,
    )
    targets = mapped["items"][0]["relevant_targets"]
    assert len(targets) == 1
    row = report["targets"][0]
    assert row["mapping_mode"] == "parent_span_overlap"
    assert row["parent_span_verified"] is True
    assert row["parent_span_coverage"] >= 0.80
    assert row["original_answer_span_hash"] == original_hash
    assert row["derived_answer_span_hash"] != original_hash
    assert targets[0]["answer_span_hash"] == original_hash
    assert targets[0]["evidence_excerpt"] == excerpt
    assert targets[0]["evidence_scope"] == "parent_expand_required"
    assert targets[0]["chunk_id"] == "old_a"
    assert targets[0]["origin_chunk_id"] == "old_a"
    assert targets[0]["parent_content_hash"].startswith("sha256:")
    assert "excluded from strict child Direct" in targets[0]["review_note"]
    assert report["parent_expand_required_query_ids"] == ["q1"]


def test_snapshot_qrels_never_guesses_across_origin_parent() -> None:
    class SameSourceCollection:
        def get(self, include):
            phrase = "审核证据片段"
            return {
                "ids": ["old_a", "old_b"],
                "documents": [phrase + " A" * 100, phrase + " B" * 100],
                "metadatas": [
                    {"source": "same.md", "title": "A"},
                    {"source": "same.md", "title": "B"},
                ],
            }

    plan = build_snapshot_experiment_plan(
        SameSourceCollection(), C1_SPEC, RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    mapped, report = map_qrels_to_experiment(
        _overlay("same.md", "审核证据片段", ["wrong"], old_chunk_id="old_b"),
        plan,
    )
    target_ids = {target["chunk_id"] for target in mapped["items"][0]["relevant_targets"]}
    origins = {chunk.origin_chunk_id for chunk in plan.chunks if chunk.chunk_id in target_ids}
    assert origins == {"old_b"}
    assert report["targets"][0]["all_source_chunk_count"] == 2
    assert report["targets"][0]["source_chunk_count"] == 1


def test_recursive_snapshot_from_experiment_collection_is_rejected() -> None:
    with pytest.raises(ChunkExperimentError, match="recursive snapshot"):
        build_snapshot_experiment_plan(
            _SnapshotCollection(), C1_SPEC, RegexTokenOffsetCodec(),
            production_collection="offerclaw_exp_chunk_c1_deadbeef0000",
        )


def test_evaluator_rejects_confounded_or_mixed_collection_metadata() -> None:
    fingerprint = "sha256:" + "a" * 64
    manifest = {
        "schema_version": "chunk-experiment-v1",
        "corpus_mode": "snapshot_children",
        "collection_name": "offerclaw_exp_chunk_c1_aaaaaaaaaaaa",
        "experiment_fingerprint": fingerprint,
        "chunk_count": 3,
        "promotion_eligible_corpus": True,
        "confounded": False,
        "chunker_version": C1_SPEC.chunker_version,
        "production_collection": "prod",
        "tokenizer": "hf-fast:test",
        "embedding_contract": {"embed_profile": "test"},
        "spec": {},
    }
    qrels = {
        "index": {
            "collection": manifest["collection_name"],
            "fingerprint": fingerprint,
            "count": 3,
        },
    }
    metadata = {
        "schema_version": "chunk-experiment-v1",
        "corpus_mode": "snapshot_children",
        "experiment_fingerprint": fingerprint,
        "chunker_version": C1_SPEC.chunker_version,
        "production_collection": "prod",
        "tokenizer": "hf-fast:test",
        "embed_profile": "test",
        "embedding_contract_json": '{"embed_profile":"test"}',
    }
    assert validate_evaluation_inputs(
        manifest, qrels, collection_metadata=metadata, collection_count=3,
    ) == (manifest["collection_name"], fingerprint, 3)

    confounded = copy.deepcopy(manifest)
    confounded.update({
        "corpus_mode": "source_rebuild",
        "promotion_eligible_corpus": False,
        "confounded": True,
    })
    with pytest.raises(ValueError, match="confounded"):
        validate_evaluation_inputs(
            confounded, qrels, collection_metadata=metadata, collection_count=3,
        )
    wrong_metadata = {**metadata, "corpus_mode": "source_rebuild"}
    with pytest.raises(ValueError, match="corpus_mode"):
        validate_evaluation_inputs(
            manifest, qrels, collection_metadata=wrong_metadata, collection_count=3,
        )


def test_embedding_contract_changes_experiment_identity(tmp_path: Path) -> None:
    source = _source(tmp_path, _long_markdown())
    common = dict(
        sources=[source], spec=C1_SPEC, codec=RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    first = build_experiment_plan(
        **common,
        embedding_contract={"embed_profile": "model-a", "dimensions": 768},
    )
    second = build_experiment_plan(
        **common,
        embedding_contract={"embed_profile": "model-b", "dimensions": 768},
    )
    assert first.experiment_fingerprint != second.experiment_fingerprint
    assert first.collection_name != second.collection_name


def test_scope_aware_qrels_validation_checks_immutable_parent() -> None:
    plan = build_snapshot_experiment_plan(
        _SnapshotCollection(), C1_SPEC, RegexTokenOffsetCodec(),
        production_collection="prod",
    )
    excerpt = " ".join(f"token{i}" for i in range(250, 371))
    mapped, _ = map_qrels_to_experiment(
        _overlay(
            "knowledge.md", excerpt, ["knowledge", "Section"],
            old_chunk_id="old_a",
        ),
        plan,
    )

    class ChildCollection:
        name = plan.collection_name

        def get(self, ids=None, include=None):
            selected = [
                chunk for chunk in plan.chunks if ids is None or chunk.chunk_id in set(ids)
            ]
            return {
                "ids": [chunk.chunk_id for chunk in selected],
                "documents": [chunk.text for chunk in selected],
                "metadatas": [
                    chunk.chroma_metadata(plan.experiment_fingerprint) for chunk in selected
                ],
            }

    class ParentCollection:
        def get(self, ids=None, include=None):
            assert ids == ["old_a"]
            return {
                "ids": ["old_a"],
                "documents": [_SnapshotCollection().get([])["documents"][0]],
                "metadatas": [{"source": "knowledge.md"}],
            }

    validate_qrels_against_collection(
        mapped, ChildCollection(), parent_collection=ParentCollection(),
    )
    parent_targets = _parent_expandable_child_ids(mapped, ChildCollection())
    assert parent_targets["q1"] == {chunk.chunk_id for chunk in plan.chunks}

    result = {
        "sets": [{"runs": [{"rows": [{
            "id": "q1",
            "final_chunk_ids": [plan.chunks[0].chunk_id],
            "fusion_chunk_ids": ["miss", plan.chunks[-1].chunk_id],
        }]}]}],
    }
    enriched = _append_parent_expandable_metrics(result, parent_targets)
    row = enriched["sets"][0]["runs"][0]["rows"][0]
    metrics = enriched["sets"][0]["runs"][0]["parent_expandable_qrels_metrics"]
    assert row["parent_expandable_rank"] == 1
    assert row["parent_expandable_candidate_rank"] == 2
    assert metrics["parent_expansion_applied"] is False


def test_evaluator_profile_uses_manifest_chunker_override(monkeypatch) -> None:
    from eval_reranker_profiles import arm_profile

    monkeypatch.setenv("RAG_EVAL_CHUNKER_VERSION", C1_SPEC.chunker_version)
    assert arm_profile("A0").chunker_version == C1_SPEC.chunker_version
    assert arm_profile("B0").chunker_version == C1_SPEC.chunker_version


class _MixedBaselineCollection:
    def __init__(self):
        self.target_excerpt = "审核确认的目标证据片段能够回答这个问题"
        self.target_prefix = "目标前文" * 35
        self.target_suffix = "目标后文" * 35

    def get(self, include):
        return {
            "ids": ["keep_c0", "target_old_a", "target_old_b"],
            "documents": [
                "保持不变的 C0 文档",
                self.target_prefix + self.target_excerpt,
                self.target_suffix,
            ],
            "metadatas": [
                {
                    "source": "keep.md", "source_type": "resource",
                    "owner_scope": "curated", "title": "Keep",
                    "chunker_version": "c0",
                },
                {
                    "source": "target.md", "source_type": "resource",
                    "owner_scope": "curated", "title": "正文",
                    "chunker_version": "c0",
                },
                {
                    "source": "target.md", "source_type": "resource",
                    "owner_scope": "curated", "title": "正文",
                    "chunker_version": "c0",
                },
            ],
        }


def _mixed_plan(tmp_path: Path):
    baseline = _MixedBaselineCollection()
    target_path = tmp_path / "target.md"
    target_path.write_text(
        "# Target\n\n## Section\n\n"
        + baseline.target_prefix + baseline.target_excerpt + baseline.target_suffix,
        encoding="utf-8",
    )
    selection = {
        "policy": {"key": "test-structural-policy"},
        "mode": "qrels_blind_structural_policy",
        "calibration_leakage": False,
        "promotion_eligible_selection": True,
        "selected_source_count": 1,
        "selected_sources": ["target.md"],
        "qrels_consulted": False,
    }
    plan = build_mixed_targeted_experiment_plan(
        baseline,
        [SourceDocument(str(target_path), "target.md", "resource", "curated")],
        selection,
        C1_SPEC,
        RegexTokenOffsetCodec(),
        production_collection="prod",
        embedding_contract={"embed_profile": "test", "dimensions": 3},
    )
    return baseline, plan


def test_mixed_targeted_replaces_only_selected_source_without_old_duplicates(tmp_path: Path) -> None:
    _baseline, plan = _mixed_plan(tmp_path)
    retained = [chunk for chunk in plan.chunks if chunk.lineage_mode == "c0_retained"]
    rebuilt = [chunk for chunk in plan.chunks if chunk.lineage_mode == "source_rebuilt"]
    assert [chunk.chunk_id for chunk in retained] == ["keep_c0"]
    assert retained[0].text == "保持不变的 C0 文档"
    assert retained[0].reuse_origin_embedding is True
    assert rebuilt and all(chunk.source == "target.md" for chunk in rebuilt)
    assert {"target_old_a", "target_old_b"}.isdisjoint(
        {chunk.chunk_id for chunk in plan.chunks}
    )
    assert plan.dedup_statistics["removed_target_c0_chunks"] == 2
    assert plan.dedup_statistics["old_target_chunks_present"] == 0
    assert plan.target_selection["qrels_consulted"] is False
    assert plan.manifest()["target_sources"] == ["target.md"]
    assert plan_statistics(plan)["body_tokens"]["rebuilt_hard_limit_violations"] == 0


def test_mixed_qrels_verifies_old_parent_then_maps_exact_source_child(tmp_path: Path) -> None:
    baseline, plan = _mixed_plan(tmp_path)
    mapped, report = map_qrels_to_experiment(
        _overlay(
            "target.md",
            baseline.target_excerpt,
            ["Target", "Section"],
            old_chunk_id="target_old_a",
        ),
        plan,
    )
    row = report["targets"][0]
    targets = mapped["items"][0]["relevant_targets"]
    assert report["failure_count"] == 0
    assert row["origin_parent_verified"] is True
    assert row["mixed_source_rebuilt"] is True
    assert row["old_c0_parent_excluded"] is True
    assert row["cross_c0_boundary_mapping_allowed"] is True
    assert row["mapping_mode"].startswith("mixed_source_exact_span")
    assert targets and all(target["chunk_id"] != "target_old_a" for target in targets)
    assert all(target["reviewed_origin_chunk_id"] == "target_old_a" for target in targets)


def test_structural_target_policy_is_qrels_blind_and_reference_only() -> None:
    class PolicyCollection:
        def get(self, include):
            generic = "## A\n正文\n## B\n" + "词 " * 340
            long_plain = "词 " * 540
            return {
                "documents": [generic, long_plain, generic, "短文"],
                "metadatas": [
                    {"source": "generic.md", "source_type": "resource", "owner_scope": "curated", "title": "正文"},
                    {"source": "ratio.md", "source_type": "doc", "owner_scope": "curated", "title": "Specific"},
                    {"source": "internal.md", "source_type": "application", "owner_scope": "internal", "title": "正文"},
                    {"source": "short.md", "source_type": "resource", "owner_scope": "curated", "title": "正文"},
                ],
            }

    selected, report = select_mixed_target_sources(
        PolicyCollection(), RegexTokenOffsetCodec(),
    )
    assert selected == ("generic.md", "ratio.md")
    assert report["qrels_consulted"] is False
    assert report["calibration_leakage"] is False
    assert report["non_reference_sources"] == ["internal.md"]

    explicit, explicit_report = select_mixed_target_sources(
        PolicyCollection(), RegexTokenOffsetCodec(), explicit_sources=["short.md"],
    )
    assert explicit == ("short.md",)
    assert explicit_report["calibration_leakage"] is True
    with pytest.raises(ChunkExperimentError, match="reference_kb"):
        select_mixed_target_sources(
            PolicyCollection(), RegexTokenOffsetCodec(), explicit_sources=["internal.md"],
        )


class _SnapshotTargetedCollection:
    """C0 snapshot with two defective chunks and same-source healthy siblings."""

    def __init__(self) -> None:
        self.generic_excerpt = "审核确认的通用标题块证据"
        self.missing_excerpt = "审核确认的无结构长块证据"
        generic_body = " ".join(f"generic{i}" for i in range(345))
        missing_prefix = " ".join(f"missing{i}" for i in range(270))
        missing_suffix = " ".join(f"tail{i}" for i in range(285))
        self.documents = {
            "generic_target": (
                "## 内部章节一\n\n"
                + self.generic_excerpt
                + "\n\n### 内部章节二\n\n"
                + generic_body
            ),
            "missing_target": (
                missing_prefix + " " + self.missing_excerpt + " " + missing_suffix
            ),
            # Same source as both selected chunks: source-level replacement
            # would incorrectly remove this immutable, healthy C0 sibling.
            "same_source_keep": "应当保持字节不变的同源健康 C0 块",
            "other_keep": "另一来源的健康 C0 块",
            "internal_filtered": (
                "## 内部一\n\n### 内部二\n\n" + " ".join(f"secret{i}" for i in range(350))
            ),
        }
        self.metadatas = {
            "generic_target": {
                "source": "shared.md",
                "source_type": "resource",
                "owner_scope": "curated",
                "title": "正文",
                "chunker_version": "c0",
            },
            "missing_target": {
                "source": "shared.md",
                "source_type": "resource",
                "owner_scope": "curated",
                "title": "Specific section",
                "chunker_version": "c0",
            },
            "same_source_keep": {
                "source": "shared.md",
                "source_type": "resource",
                "owner_scope": "curated",
                "title": "Healthy section",
                "breadcrumb": "Shared > Healthy section",
                "heading_path": '["Shared", "Healthy section"]',
                "chunker_version": "c0",
            },
            "other_keep": {
                "source": "other.md",
                "source_type": "doc",
                "owner_scope": "curated",
                "title": "Other",
                "breadcrumb": "Other",
                "heading_path": '["Other"]',
                "chunker_version": "c0",
            },
            "internal_filtered": {
                "source": "applications.md",
                "source_type": "application",
                "owner_scope": "internal",
                "title": "正文",
                "chunker_version": "c0",
            },
        }

    def get(self, include):
        ids = list(self.documents)
        return {
            "ids": ids,
            "documents": [self.documents[chunk_id] for chunk_id in ids],
            "metadatas": [self.metadatas[chunk_id] for chunk_id in ids],
        }


def _snapshot_targeted_plan():
    baseline = _SnapshotTargetedCollection()
    codec = RegexTokenOffsetCodec()
    selected, selection = select_snapshot_target_chunks(baseline, codec)
    plan = build_snapshot_targeted_experiment_plan(
        baseline,
        selected,
        selection,
        C1_SPEC,
        codec,
        production_collection="prod",
        embedding_contract={"embed_profile": "test", "dimensions": 3},
    )
    return baseline, selected, selection, plan


def test_snapshot_target_selector_is_chunk_level_qrels_blind_and_reference_only() -> None:
    baseline = _SnapshotTargetedCollection()
    selected, report = select_snapshot_target_chunks(
        baseline, RegexTokenOffsetCodec(),
    )

    assert selected == ("generic_target", "missing_target")
    assert report["selected_chunk_ids"] == list(selected)
    assert report["qrels_consulted"] is False
    assert report["calibration_leakage"] is False
    assert report["promotion_eligible_selection"] is True
    assert report["selector_hash"].startswith("sha256:")
    assert report["non_reference_filtered_chunk_count"] == 1

    rows = {row["chunk_id"]: row for row in report["selected_chunks"]}
    assert rows["generic_target"]["reasons"] == [
        "generic_title_with_internal_markdown_headings"
    ]
    assert rows["generic_target"]["internal_heading_count"] >= 2
    assert rows["missing_target"]["reasons"] == [
        "missing_breadcrumb_and_heading_over_limit"
    ]
    assert rows["missing_target"]["token_count"] > 512
    assert "internal_filtered" not in selected


def test_snapshot_targeted_replaces_only_defective_chunks_and_keeps_same_source_sibling() -> None:
    baseline, selected, selection, plan = _snapshot_targeted_plan()
    retained = [chunk for chunk in plan.chunks if chunk.lineage_mode == "c0_retained"]
    replacements = [
        chunk for chunk in plan.chunks if chunk.lineage_mode == "snapshot_target_child"
    ]

    assert plan.corpus_mode == "snapshot_targeted"
    assert {chunk.chunk_id for chunk in retained} == {
        "same_source_keep", "other_keep", "internal_filtered",
    }
    assert {chunk.source for chunk in replacements} == {"shared.md"}
    assert {chunk.origin_chunk_id for chunk in replacements} == set(selected)
    assert set(selected).isdisjoint({chunk.chunk_id for chunk in plan.chunks})
    assert all(chunk.reuse_origin_embedding is True for chunk in retained)
    assert all(chunk.reuse_origin_embedding is False for chunk in replacements)

    same_source = next(chunk for chunk in retained if chunk.chunk_id == "same_source_keep")
    assert same_source.text == baseline.documents["same_source_keep"]
    assert same_source.original_metadata == tuple(
        sorted(baseline.metadatas["same_source_keep"].items())
    )
    assert plan.origin_documents["generic_target"] == baseline.documents["generic_target"]
    assert plan.origin_documents["missing_target"] == baseline.documents["missing_target"]
    assert plan.target_selection == selection

    assert plan.dedup_statistics["old_target_chunks_present"] == 0
    assert plan.dedup_statistics["retained_c0_chunks"] == 3
    assert plan.dedup_statistics["removed_target_c0_chunks"] == 2
    assert plan.dedup_statistics["rebuilt_target_chunks"] == len(replacements)
    assert plan_statistics(plan)["body_tokens"]["rebuilt_hard_limit_violations"] == 0


def test_snapshot_targeted_manifest_has_no_content_drift_or_calibration_leakage() -> None:
    _baseline, selected, selection, plan = _snapshot_targeted_plan()
    manifest = plan.manifest()

    assert manifest["corpus_mode"] == "snapshot_targeted"
    assert manifest["target_selection"]["selected_chunk_ids"] == list(selected)
    assert manifest["target_selection"]["selector_hash"] == selection["selector_hash"]
    assert manifest["target_selection"]["calibration_leakage"] is False
    assert manifest["target_selection"]["promotion_eligible_selection"] is True
    # Every replacement comes from the immutable C0 body captured in the same
    # plan; no mutable Markdown source file is part of this experiment.
    assert manifest["source_count"] == 3
    assert manifest["origin_parent_count"] == len(_SnapshotTargetedCollection().documents)
    assert all(source["path"] == "" for source in manifest["sources"])


def test_snapshot_targeted_cli_manifest_is_clean_and_promotion_eligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import argparse
    import build_chunk_experiment
    import chromadb

    baseline = _SnapshotTargetedCollection()

    class Client:
        def get_collection(self, name):
            assert name == "prod"
            return baseline

    output_root = tmp_path / "out"
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: Client())
    monkeypatch.setattr(
        build_chunk_experiment,
        "_args",
        lambda: argparse.Namespace(
            profile="c1",
            baseline_collection="prod",
            corpus_mode="snapshot_targeted",
            dry_run=True,
            build=False,
            files=None,
            target_source=[],
            qrels=None,
            output_dir=output_root,
            tokenizer="regex",
            allow_network_tokenizer=False,
        ),
    )
    monkeypatch.setattr(
        rag_chunk_experiments,
        "active_embedding_contract",
        lambda: {"embed_profile": "test", "dimensions": 3},
    )

    build_chunk_experiment.main()
    manifest_paths = list(output_root.glob("*/manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["corpus_mode"] == "snapshot_targeted"
    assert manifest.get("content_drift_sources", []) == []
    assert manifest["calibration_leakage"] is False
    assert manifest["promotion_eligible_corpus"] is True
    assert manifest["confounded"] is False


def test_snapshot_targeted_qrels_lock_origin_and_never_keep_old_target_id() -> None:
    baseline, _selected, _selection, plan = _snapshot_targeted_plan()
    mapped, report = map_qrels_to_experiment(
        _overlay(
            "shared.md",
            baseline.generic_excerpt,
            ["stale", "reviewer", "heading"],
            old_chunk_id="generic_target",
        ),
        plan,
    )

    targets = mapped["items"][0]["relevant_targets"]
    row = report["targets"][0]
    assert report["failure_count"] == 0
    assert row["origin_parent_locked"] is True
    assert row["exact_origin_child_lock"] is True
    assert row["origin_parent_verified"] is True
    assert row["locked_origin_chunk_id"] == "generic_target"
    assert targets
    assert all(target["chunk_id"] != "generic_target" for target in targets)
    assert all(target["origin_chunk_id"] == "generic_target" for target in targets)
    assert {
        chunk.origin_chunk_id
        for chunk in plan.chunks
        if chunk.chunk_id in {target["chunk_id"] for target in targets}
    } == {"generic_target"}


def test_snapshot_targeted_qrels_for_retained_chunk_keeps_exact_c0_identity() -> None:
    baseline, _selected, _selection, plan = _snapshot_targeted_plan()
    excerpt = baseline.documents["same_source_keep"]
    mapped, report = map_qrels_to_experiment(
        _overlay(
            "shared.md", excerpt, ["Shared", "Healthy section"],
            old_chunk_id="same_source_keep",
        ),
        plan,
    )

    target = mapped["items"][0]["relevant_targets"][0]
    row = report["targets"][0]
    assert target["chunk_id"] == "same_source_keep"
    assert target["origin_chunk_id"] == "same_source_keep"
    assert row["exact_origin_child_lock"] is True
    assert row["origin_parent_verified"] is True


def test_snapshot_targeted_multisection_children_share_full_c0_expansion_parent() -> None:
    baseline, _selected, _selection, plan = _snapshot_targeted_plan()
    children = [
        chunk for chunk in plan.chunks if chunk.origin_chunk_id == "generic_target"
    ]
    assert len(children) >= 2
    assert len({chunk.parent_id for chunk in children}) == 1
    expected_hash = "sha256:" + hashlib.sha256(
        baseline.documents["generic_target"].encode("utf-8")
    ).hexdigest()
    assert {chunk.parent_content_hash for chunk in children} == {expected_hash}
    assert len({chunk.window_group_id for chunk in children}) >= 2
    assert {
        value for value in expand_parent_context(
            [chunk.chunk_id for chunk in children], plan,
        ).values()
    } == {baseline.documents["generic_target"]}


def test_snapshot_targeted_parent_scope_validates_against_original_c0_collection() -> None:
    baseline, _selected, _selection, plan = _snapshot_targeted_plan()
    excerpt = (
        "## 内部章节一\n\n" + baseline.generic_excerpt
        + "\n\n### 内部章节二"
    )
    mapped, report = map_qrels_to_experiment(
        _overlay(
            "shared.md", excerpt, ["shared", "内部章节"],
            old_chunk_id="generic_target",
        ),
        plan,
    )
    target = mapped["items"][0]["relevant_targets"][0]
    assert report["failure_count"] == 0
    assert target["evidence_scope"] == "parent_expand_required"
    assert target["origin_chunk_id"] == "generic_target"

    class ChildCollection:
        name = plan.collection_name

        def get(self, ids=None, include=None):
            selected = [
                chunk for chunk in plan.chunks
                if ids is None or chunk.chunk_id in set(ids)
            ]
            return {
                "ids": [chunk.chunk_id for chunk in selected],
                "documents": [chunk.text for chunk in selected],
                "metadatas": [
                    chunk.chroma_metadata(plan.experiment_fingerprint)
                    for chunk in selected
                ],
            }

    class ParentCollection:
        def get(self, ids=None, include=None):
            assert ids == ["generic_target"]
            return {
                "ids": ["generic_target"],
                "documents": [baseline.documents["generic_target"]],
                "metadatas": [baseline.metadatas["generic_target"]],
            }

    validate_qrels_against_collection(
        mapped, ChildCollection(), parent_collection=ParentCollection(),
    )
