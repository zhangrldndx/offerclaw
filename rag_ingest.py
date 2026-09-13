# -*- coding: utf-8 -*-
"""
OfferClaw · RAG Ingest 脚本

功能：
1. 读取指定 .md 文件
2. 按标题智能分块
3. 批量调用当前配置的 Embedding API 生成向量
4. 写入 ChromaDB

用法：
  python rag_ingest.py                          # ingest 默认文件列表
  python rag_ingest.py --files user_profile.md daily_log.md  # 指定文件
  python rag_ingest.py --rebuild                # 清空旧库重建
"""

import os
import sys
import argparse
import time

import chromadb

from rag_tools import (
    CHUNKER_VERSION,
    describe_embedding_config,
    fake_embedding,
    get_collection_name,
    get_embeddings_batch,
    has_embedding_api_key,
    split_markdown_document,
)
from rag_source_policy import (
    NON_RAG_SOURCE_BASENAMES,
    NON_RAG_SOURCE_TYPES,
    infer_owner_scope,
    normalized_source_type,
    rag_source_excluded,
)

# =====================================================
# 配置
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = get_collection_name()

# 默认 ingest 的文件列表（按优先级排序，每条带 source_type 标签）
DEFAULT_FILES = [
    ("user_profile.md", "profile"),
    ("daily_log.md", "log"),
    ("SOUL.md", "system"),
    ("target_rules.md", "system"),
    ("source_policy.md", "system"),
    ("onboarding_prompt.md", "system"),
    ("job_match_prompt.md", "system"),
    ("plan_prompt.md", "system"),
    ("summary_prompt.md", "system"),
    ("DATA_CONTRACT.md", "doc"),
    ("applications.md", "application"),
    ("interview_story_bank.md", "story"),
    ("docs/archive/resume_pitch.md", "resume"),
    ("docs/project_one_pager.md", "doc"),
    ("docs/verification_report.md", "verification"),
]


_KB_SUBDIR_TYPE = {
    "career_paths": "career_knowledge",
    "experience_posts": "experience",
    "learning_resources": "resource",
    "project_context": "project_context",   # 已有项目先验（localflow 等），只读上下文
    "resume_rules": "resume_rule",
    # papers 子目录不在此表(2026-08-09 集合分离):论文只入独立 e5 论文域集合
    # (kb_paper_e5_v1),不进中文主库——论文路由靠"主库证据弱→论文域回退"触发,
    # 主库里若混有论文块会用平庸证据骗过门控使回退失效。papers 入库走
    # --add --source-type paper + RAG_COLLECTION_NAME=kb_paper_e5_v1(e5+前缀)。
}
PAPER_SUBDIR_TYPE = {"papers": "paper"}   # 供显式 --add 推断,仍不进主库自动发现


def _infer_source_type(path: str) -> str:
    """按文件所在 knowledge_base 子目录推断 source_type；推断不出按 doc。"""
    norm = path.replace("\\", "/")
    for subdir, st in {**_KB_SUBDIR_TYPE, **PAPER_SUBDIR_TYPE}.items():
        if f"knowledge_base/{subdir}/" in norm:
            return st
    return "doc"


def _frontmatter_entity_fields(text: str) -> dict[str, str]:
    """提取个人记忆用于逻辑过滤的轻量实体字段，不引入 YAML 运行时依赖。"""
    if not (text or "").startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    allowed = {"company", "position", "stage", "owner_scope", "source_type"}
    out: dict[str, str] = {}
    for line in parts[1].splitlines():
        key, sep, value = line.partition(":")
        key = key.strip()
        if sep and key in allowed:
            cleaned = value.strip().strip('"').strip("'")[:300]
            if cleaned:
                out[key] = cleaned
    return out


def _discover_knowledge_base() -> list[tuple[str, str]]:
    """Scan knowledge_base/ for .md files (excluding templates and README)."""
    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    if not os.path.isdir(kb_dir):
        return []
    type_map = dict(_KB_SUBDIR_TYPE)
    found = []
    for subdir, source_type in type_map.items():
        dirpath = os.path.join(kb_dir, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if fname.endswith(".md") and not fname.startswith("_"):
                relpath = os.path.join("knowledge_base", subdir, fname)
                found.append((relpath, source_type))
    return found


def build_collection_name(file_path: str) -> str:
    """从文件路径生成 collection 中的 source 标记"""
    return os.path.basename(file_path)


_SHORT_STATE_SOURCE_TYPES = frozenset({"profile", "log", "application", "story"})


def _split_source_document(text: str, source_type: str) -> list[dict]:
    """Keep compact state records while retaining normal KB hygiene defaults."""
    if source_type in _SHORT_STATE_SOURCE_TYPES:
        return split_markdown_document(
            text, min_chars=20, strict_hygiene=False, preserve_short_sections=True,
        )
    return split_markdown_document(text, strict_hygiene=(source_type == "paper"))


def ingest_file(
    file_path: str,
    collection,
    source_type: str = "doc",
    seen_hashes: set | None = None,
    *,
    source_id: str = "",
) -> dict:
    """
    读取单个 .md 文件，分块 → 向量化 → 入库。
    返回统计信息。

    ``seen_hashes``：跨文件共享的内容哈希集合。若某 chunk 的归一化文本
    已在集合中出现，则视为重复并跳过（避免飞书多章节复用同一节导致
    向量库存重复块、检索结果占满 top-k）。
    """
    import hashlib

    filename = str(source_id or os.path.basename(file_path)).replace("\\", "/")
    full_path = os.path.join(BASE_DIR, file_path) if not os.path.isabs(file_path) else file_path

    print(f"\n{'='*60}")
    print(f"[INGEST] {filename}")
    print(f"{'='*60}")

    # Step 1: 读取
    if not os.path.exists(full_path):
        print(f"  [SKIP] 文件不存在: {full_path}")
        return {"file": filename, "status": "not_found", "chunks": 0, "tokens": 0}

    with open(full_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    entity_metadata = _frontmatter_entity_fields(raw_text)
    source_type = normalized_source_type(
        entity_metadata.get("source_type") or source_type, filename,
    )
    if rag_source_excluded(source_type, filename):
        print(f"  [EXCLUDED] {filename} 是业务 JD/离线样例，不进入 RAG")
        return {
            "file": filename, "status": "excluded_from_rag",
            "chunks": 0, "tokens": 0,
        }
    owner_scope = infer_owner_scope(
        source_type, filename, entity_metadata.get("owner_scope", ""),
    )
    # 即使 chunk 已存在，也通过下面的无 embedding 更新路径补齐归属字段。
    entity_metadata.update({"source_type": source_type, "owner_scope": owner_scope})

    print(f"  [READ] {len(raw_text)} 字符, {raw_text.count(chr(10))} 行")

    # 文档级标题：取第一个一级标题（# xxx），fallback 文件名。
    # 用于给每个 chunk 的向量注入"文档身份"，提升同质文档（仅主语不同、正文雷同）的
    # 检索区分度——对应知识库 02章 索引优化·添加元数据（adding metadata）。
    import re as _re
    _m = _re.search(r"^#\s+(.+)$", raw_text, _re.M)
    doc_title = _m.group(1).strip() if _m else filename[:-3]

    # Step 1.5: 图转文（IMAGE_CAPTION=1 时）—— ![](img) → [图: 描述+OCR]，让图片可检索
    try:
        from image_caption import caption_enabled, caption_markdown
        if caption_enabled() and "![" in raw_text:
            st = {}
            raw_text = caption_markdown(raw_text, md_path=full_path, stats=st)
            if st.get("captioned") or st.get("cached"):
                print(f"  [图转文] 新描述 {st.get('captioned',0)} · 缓存命中 "
                      f"{st.get('cached',0)} · 丢弃 {st.get('dropped',0)}")
    except Exception as _e:
        print(f"  [图转文] 跳过（{_e}）")

    # Step 2: 分块
    # 论文域启用严格卫生档(硬上限/引文过滤/英文信息量门);中文主库保持历史行为
    chunks = _split_source_document(raw_text, source_type)
    print(f"  [SPLIT] {len(chunks)} 块")

    # Step 2.5: 跨文件内容去重
    # 飞书多章节常复用同一节（含轻微改写），故用「归一化前缀哈希」判重：
    # 取去空白后前 100 字符做 key，能同时抓住完全重复与章节变体近重复。
    if seen_hashes is not None:
        deduped = []
        dup_count = 0
        for c in chunks:
            norm = "".join(c["text"].split())  # 去掉所有空白
            key = hashlib.md5(norm[:100].encode("utf-8")).hexdigest()
            if key in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(key)
            deduped.append(c)
        if dup_count:
            print(f"  [DEDUP] 跳过 {dup_count} 个与已入库内容重复/近重复的块")
        chunks = deduped

    if not chunks:
        print(f"  [WARN] 无有效块，跳过")
        return {"file": filename, "status": "empty", "chunks": 0, "tokens": 0}

    # 打印分块摘要
    for i, chunk in enumerate(chunks):
        title = chunk["metadata"].get("title", "N/A")
        print(f"    Block {i+1:02d}: {chunk['metadata']['char_len']:4d} 字 | "
              f"[{title}] | {chunk['text'][:50].replace(chr(10), ' ')}...")

    # Step 3: 计算稳定 ID（内容哈希）→ 跳过已入库的块（可断点续传）
    stem = (filename[:-3] if not source_id else
            "src_" + hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16])
    all_ids = [
        f"{stem}_{hashlib.md5(c['text'].encode('utf-8')).hexdigest()[:12]}"
        for c in chunks
    ]
    existing = collection.get(ids=all_ids)
    existing_ids = set(existing["ids"]) if existing and existing["ids"] else set()

    todo = [
        (cid, c) for cid, c in zip(all_ids, chunks)
        if cid not in existing_ids
    ]
    if existing_ids:
        print(f"  [SKIP] {len(existing_ids)} 个块已入库，跳过 embedding")
    if not todo:
        metadata_updated = 0
        if entity_metadata and existing_ids:
            current = collection.get(ids=sorted(existing_ids), include=["metadatas"])
            current_ids = current.get("ids") or []
            current_metas = current.get("metadatas") or []
            refreshed = []
            for meta in current_metas:
                merged = dict(meta or {})
                before = dict(merged)
                merged.update(entity_metadata)
                metadata_updated += int(merged != before)
                refreshed.append(merged)
            if current_ids and metadata_updated:
                collection.update(ids=current_ids, metadatas=refreshed)
                print(f"  [META] 回填实体元数据 {metadata_updated} 个块")
        print(f"  [OK] 全部 {len(chunks)} 块已是最新，无需重新 embedding")
        return {
            "file": filename, "status": "ok",
            "chunks": len(chunks), "tokens": sum(len(c["text"]) for c in chunks),
            "embed_time": 0.0, "metadata_updated": metadata_updated,
        }

    todo_ids = [t[0] for t in todo]
    texts = [t[1]["text"] for t in todo]
    # 索引优化（可选，默认关）：向量化文本前缀文档标题。
    # 实测（方案A）：十几字标题被几百字正文稀释、对 bge 向量影响≈0，对同质文档消歧无效，
    # 故默认关闭，保留 env 供未来强区分场景试验。career 同质问题改由混合检索(BM25)解决。
    if os.environ.get("CHUNK_DOC_TITLE", "0").strip().lower() in ("1", "true", "yes", "on"):
        embed_texts = [f"《{doc_title}》\n{t}" for t in texts]
    else:
        embed_texts = texts
    # 契约守卫:与集合既有块的 embedding 血缘不符则拒绝写入(防维度相同的静默污染)
    from rag_tools import assert_collection_contract, embed_profile as _ep
    _EMBED_PROFILE = _ep()
    assert_collection_contract(collection)

    print(f"\n  [EMBEDDING] 调用 Embedding API（{len(texts)} 条待入库，批量）...")
    t0 = time.time()

    if not has_embedding_api_key():
        print(f"  [WARN] API Key 未配置，使用 SHA256 伪向量作为占位")
        # 用伪向量占位，验证入库流程
        embeddings = [fake_embedding(t) for t in embed_texts]
    else:
        embeddings = get_embeddings_batch(embed_texts)

    embed_time = time.time() - t0
    print(f"  [EMBEDDING] 完成，耗时 {embed_time:.1f}s")

    # Step 4: 入库（仅新块）
    ids = todo_ids
    metadatas = []
    for _cid, chunk in todo:
        metadata = {
            "source": filename,
            "source_type": source_type,
            "owner_scope": owner_scope,
            "char_len": chunk["metadata"]["char_len"],
            "title": chunk["metadata"].get("title", ""),
            # 血缘(2026-08-10):哪个模型/前缀/截断/分块版本产出的,事后可查、写入前可校验
            "embed_profile": _EMBED_PROFILE,
            "chunker_version": CHUNKER_VERSION,
        }
        metadata.update(entity_metadata)
        metadatas.append(metadata)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    new_tokens = sum(len(t) for t in texts)
    print(f"  [DB] 新入库 {len(ids)} 条（本文件共 {len(chunks)} 块），新增字符 {new_tokens}")

    return {
        "file": filename,
        "status": "ok",
        "chunks": len(chunks),
        "tokens": sum(len(c["text"]) for c in chunks),
        "embed_time": round(embed_time, 2),
    }


def replace_source(file_path: str, collection, source_type: str = "doc", *,
                   source_id: str = "") -> dict:
    """Versioned Source Upsert(2026-08-09,方案文档 P0.1 的右尺寸实现):同源文件重解析
    后原子替换,消灭重复块。

    顺序保证安全(先算后换,不做朴素"先删再建"):
      1. 新版本先走 ingest_file(内容寻址 ID:未变块自然跳过,变更块 embedding+写入)
         ——**任何失败发生在此步时,旧数据分毫未动**;
      2. 仅当第 1 步成功,才删除"旧有而新无"的过期块(stale = 旧 ID 集 − 新 ID 集)。
    返回统计含 stale_removed,便于审计。
    """
    import hashlib as _h
    filename = str(source_id or os.path.basename(file_path)).replace("\\", "/")
    old = collection.get(where={"source": filename})
    old_ids = set(old["ids"] or []) if old else set()

    out = ingest_file(file_path, collection, source_type=source_type, source_id=source_id)
    if out.get("status") != "ok":
        out["stale_removed"] = 0
        return out          # 新版本失败:旧块全保留(安全兜底)

    full_path = os.path.join(BASE_DIR, file_path) if not os.path.isabs(file_path) else file_path
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = _split_source_document(content, source_type)
    stem = (filename[:-3] if not source_id else
            "src_" + _h.sha256(filename.encode("utf-8")).hexdigest()[:16])
    new_ids = {f"{stem}_{_h.md5(c['text'].encode('utf-8')).hexdigest()[:12]}" for c in chunks}
    stale = sorted(old_ids - new_ids)
    if stale:
        collection.delete(ids=stale)
        print(f"  [UPSERT] 原子替换完成:清理过期块 {len(stale)} 个(旧 {len(old_ids)} → 新 {len(new_ids)})")
    out["stale_removed"] = len(stale)
    return out


def replace_source_content(source_id: str, content: str, collection,
                           source_type: str = "doc") -> dict:
    """Safely replace one derived source received over a trusted local pipe.

    The authoritative file is never written by this process. A private transient
    file only feeds the existing versioned upsert and is removed before return.
    ``source_id`` remains the stable metadata identity, including relative paths.
    """
    import tempfile

    source_id = str(source_id or "").strip().replace("\\", "/")
    if (not source_id or source_id.startswith("/") or ".." in source_id.split("/")
            or not isinstance(content, str)):
        raise ValueError("invalid source_id/content for derived index")
    suffix = os.path.splitext(source_id)[1] or ".md"
    with tempfile.TemporaryDirectory(prefix="offerclaw-index-") as temp_dir:
        try:
            os.chmod(temp_dir, 0o700)
        except OSError:
            pass
        transient = os.path.join(temp_dir, "source" + suffix)
        with open(transient, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(transient, 0o600)
        except OSError:
            pass
        return replace_source(
            transient, collection, source_type=source_type, source_id=source_id,
        )


def purge_non_rag_sources(collection) -> int:
    """Remove legacy candidate/business JD chunks from a derived vector index."""
    stale_ids: set[str] = set()
    for source_type in sorted(NON_RAG_SOURCE_TYPES):
        result = collection.get(where={"source_type": source_type})
        stale_ids.update(str(value) for value in (result or {}).get("ids", []) if value)
    for source in sorted(NON_RAG_SOURCE_BASENAMES):
        result = collection.get(where={"source": source})
        stale_ids.update(str(value) for value in (result or {}).get("ids", []) if value)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    return len(stale_ids)


def main():
    parser = argparse.ArgumentParser(description="OfferClaw RAG Ingest")
    parser.add_argument("--files", nargs="+", help="指定要 ingest 的文件名")
    parser.add_argument("--rebuild", action="store_true", help="清空旧库重建")
    parser.add_argument("--add", help="增量添加单个文件到现有库（不重建、不影响原有内容）")
    parser.add_argument("--replace", action="store_true", help="配合 --add:同源原子替换(先算后换,清理过期块)")
    parser.add_argument("--source-type", help="配合 --add 指定 source_type（默认按子目录推断）")
    parser.add_argument(
        "--purge-non-rag-only", action="store_true",
        help="只清理旧候选/业务 JD 向量块，不执行入库",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Chroma collection 名称（默认按 embedding provider/model 自动选择）",
    )
    args = parser.parse_args()
    collection_name = args.collection

    if args.add:
        # 增量模式：只把这一个文件加入现有 collection，不删除/不重建任何已有内容
        st = args.source_type or _infer_source_type(args.add)
        files = [(args.add, st)]
        print(f"[增量] 仅添加 1 个文件：{args.add}（source_type={st}）")
    elif args.files:
        files = [(f, "doc") for f in args.files]
    else:
        kb_files = _discover_knowledge_base()
        files = DEFAULT_FILES + kb_files
        if kb_files:
            print(f"[知识库] 自动发现 {len(kb_files)} 个 knowledge_base 文件")

    print("=" * 60)
    print("OfferClaw RAG Ingest")
    print(f"数据库目录: {DB_DIR}")
    print(f"Collection: {collection_name}")
    print(f"Embedding: {describe_embedding_config()}")
    print(f"API Key: {'已配置' if has_embedding_api_key() else '⚠️ 未配置（使用伪向量占位）'}")
    print("=" * 60)

    # 初始化 ChromaDB
    client = chromadb.PersistentClient(path=DB_DIR)

    # 处理 collection（--add 增量模式下绝不重建，保护已有内容）
    if args.rebuild and not args.add:
        print("\n[REBUILD] 清空旧 collection...")
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    try:
        collection = client.get_collection(collection_name)
        print(f"[DB] 使用已有 collection，当前记录数: {collection.count()}")
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"[DB] 新建 collection")

    purged = purge_non_rag_sources(collection)
    if purged:
        print(f"[DB] 已清理不应进入 RAG 的旧 JD 块: {purged}")
    if args.purge_non_rag_only:
        print(f"[DB] 清理完成，当前记录数: {collection.count()}")
        return

    # 逐个文件 ingest（跨文件共享内容哈希集合，去重）
    stats = []
    seen_hashes: set = set()
    for f, st in files:
        if args.add and getattr(args, "replace", False):
            result = replace_source(f, collection, source_type=st)   # P0.1 原子替换
        else:
            result = ingest_file(f, collection, source_type=st, seen_hashes=seen_hashes)
        stats.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print("Ingest 完成汇总")
    print(f"{'='*60}")

    total_chunks = 0
    total_tokens = 0
    ok_files = 0

    for s in stats:
        status_icon = "[OK]" if s["status"] == "ok" else ("[MISS]" if s["status"] == "not_found" else "[ERR]")
        print(f"  {status_icon} {s['file']}: {s['chunks']} 块, {s.get('tokens', 0)} 字符")
        if s["status"] == "ok":
            total_chunks += s["chunks"]
            total_tokens += s["tokens"]
            ok_files += 1

    print(f"\n  总计: {ok_files}/{len(files)} 文件成功, {total_chunks} 块, {total_tokens} 字符")
    print(f"  Collection 总记录数: {collection.count()}")
    print(f"\n下一步: python rag_query.py  <你的问题>")


if __name__ == "__main__":
    main()
