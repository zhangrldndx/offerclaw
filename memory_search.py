# -*- coding: utf-8 -*-
"""Hybrid retrieval over authoritative personal memory events."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from memory_store import MemoryStore


_INDEX_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-index")
_INDEX_LOCK = threading.Lock()


def _tokens(text: str) -> list[str]:
    value = (text or "").casefold()
    latin = re.findall(r"[a-z][a-z0-9_+.#-]{1,}", value)
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    chinese = [chunk[i:i + 2] for chunk in chunks for i in range(len(chunk) - 1)]
    return latin + chinese


def _event_text(event: dict[str, Any], store: MemoryStore) -> str:
    fields = [
        "content", "notes", "tag", "main_tag", "next_day_suggestion", "company",
        "position", "status", "direction", "jd_title", "next_action", "note",
    ]
    parts = [str(event.get(key) or "") for key in fields if event.get(key)]
    for key in ("done", "completed", "incomplete", "blockers", "skill_evidence"):
        if event.get(key): parts.append(json.dumps(event[key], ensure_ascii=False))
    snapshot_id = event.get("snapshot_id") or event.get("after_snapshot_id")
    if snapshot_id:
        snapshot = store.get_snapshot(str(snapshot_id))
        if snapshot: parts.append(str(snapshot.get("content") or ""))
    return "\n".join(part for part in parts if part).strip()


def _event_document(event: dict[str, Any], store: MemoryStore) -> dict[str, Any] | None:
    text = _event_text(event, store)
    if not text:
        return None
    source_type = str(event.get("source_kind") or event["kind"])
    historical_without_date = bool(event.get("historical_snapshot") and not event.get("business_date"))
    return {
        "id": event["event_id"], "event_id": event["event_id"],
        "source_type": source_type,
        "date": "" if historical_without_date else str(
            event.get("business_date") or event.get("occurred_at", "")[:10]),
        "title": _title(event), "text": text,
        "target_context_id": event.get("target_context_id"),
        "archived": bool(event.get("archived")), "source": event.get("source", ""),
        "source_path": event.get("source_path", ""),
        "evidence_role": _evidence_role(source_type, text),
        "traffic_origin": event.get("traffic_origin", "organic"),
        "operation_id": event.get("operation_id"), "entity_type": event.get("entity_type"),
        "entity_id": event.get("entity_id"), "seq": int(event.get("seq") or 0),
    }


def event_documents(store: MemoryStore | None = None, *, include_archived: bool = True,
                    target_context_id: str | None = None) -> list[dict[str, Any]]:
    store = store or MemoryStore()
    rows = store.list_events(include_archived=include_archived,
                             target_context_id=target_context_id)
    return [doc for event in rows
            if event.get("traffic_origin", "organic") == "organic"
            and (doc := _event_document(event, store)) is not None]


def _title(event: dict[str, Any]) -> str:
    if event.get("title"):
        return str(event["title"])
    labels = {
        "conversation_message": "顶部问答",
        "daily_log_recorded": "每日执行",
        "reflection": "个人复盘",
        "application_changed": "投递管理",
        "profile_edited": "用户画像",
        "plan_saved": "学习计划",
        "match_completed": "岗位匹配",
        "career_flow_run": "职业流程",
    }
    detail = event.get("position") or event.get("main_tag") or event.get("jd_title") or ""
    return labels.get(event.get("kind"), str(event.get("kind") or "记忆")) + (f" · {detail}" if detail else "")


def _evidence_role(source_type: str, text: str) -> str:
    if source_type == "plan":
        return "planned"
    if source_type in {"reflection", "application_review", "sop_execution"}:
        return "actual_record"
    if source_type in {"conversation_message", "profile"}:
        return "user_statement"
    if source_type == "daily_log_recorded":
        return "actual_record"
    if source_type == "daily_execution":
        section = re.search(r"已完成：(.*?)(?:\n- 未完成：|\n- 实际投入时间：)", text, re.S)
        completed = (section.group(1) if section else "").replace("【待补充】", "")
        has_actual = "✅" in text or bool(completed.strip(" \t\r\n-"))
        return "actual_record" if has_actual else "planned"
    return "supporting_record"


def _chunks(text: str, size: int = 1600, overlap: int = 160) -> list[str]:
    if len(text) <= size: return [text]
    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step) if text[start:start + size].strip()]


def _embedding_profile() -> str:
    from rag_tools import get_embedding_config
    config = get_embedding_config()
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: bytes, dimensions: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dimensions}f", blob)


def _chunk_rows(doc: dict[str, Any], chunks: list[str], vectors: list[list[float]],
                profile: str) -> list[dict[str, Any]]:
    return [{
        "chunk_id": f"{doc['event_id']}:chunk:{index}",
        "event_id": doc["event_id"], "chunk_index": index,
        "content": chunk,
        "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
        "embedding": _pack_vector(vector), "embedding_dim": len(vector),
        "embedding_profile": profile,
        "metadata": {
            "source_type": doc["source_type"], "date": doc["date"],
            "target_context_id": doc.get("target_context_id") or "global",
        },
    } for index, (chunk, vector) in enumerate(zip(chunks, vectors))]


def index_event(event_id: str, *, store: MemoryStore | None = None) -> dict[str, Any]:
    store = store or MemoryStore()
    event = store.get_event(event_id)
    doc = _event_document(event, store) if event else None
    if not doc:
        store.delete_search_chunks(event_id)
        return {"status": "missing", "event_id": event_id}
    from rag_tools import get_embeddings_batch
    chunks = _chunks(doc["text"])
    vectors = get_embeddings_batch(chunks)
    rows = _chunk_rows(doc, chunks, vectors, _embedding_profile())
    with _INDEX_LOCK:
        store.replace_search_chunks(event_id, rows)
    return {"status": "ok", "event_id": event_id, "chunks": len(chunks)}


def schedule_index_event(event_id: str, *, store: MemoryStore | None = None) -> None:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("MEMORY_DENSE", "1") != "1":
        return
    _INDEX_POOL.submit(_safe_index, event_id, store)


def _safe_index(event_id: str, store: MemoryStore | None) -> None:
    try:
        index_event(event_id, store=store)
    except Exception:
        return


def remove_from_index(event_id: str) -> None:
    try:
        MemoryStore().delete_search_chunks(event_id)
    except Exception:
        pass


def rebuild_index(*, store: MemoryStore | None = None) -> dict[str, Any]:
    store = store or MemoryStore()
    docs = event_documents(store, include_archived=False)
    texts: list[str] = []
    owners: list[tuple[dict[str, Any], list[str]]] = []
    for doc in docs:
        chunks = _chunks(doc["text"])
        owners.append((doc, chunks))
        texts.extend(chunks)
    if not texts:
        store.replace_all_search_chunks([])
        return {"status": "ok", "indexed": 0, "chunks": 0, "failed": 0}
    try:
        from rag_tools import get_embeddings_batch
        vectors = get_embeddings_batch(texts)
        profile = _embedding_profile()
        rows: list[dict[str, Any]] = []
        offset = 0
        for doc, chunks in owners:
            rows.extend(_chunk_rows(doc, chunks, vectors[offset:offset + len(chunks)], profile))
            offset += len(chunks)
        with _INDEX_LOCK:
            store.replace_all_search_chunks(rows)
        return {"status": "ok", "indexed": len(docs), "chunks": len(rows), "failed": 0}
    except Exception as exc:
        return {"status": "failed", "indexed": 0, "chunks": 0,
                "failed": len(docs), "error": f"{type(exc).__name__}: {exc}"[:500]}


def _date_range(query: str) -> tuple[str, str] | None:
    explicit = re.findall(r"\d{4}-\d{2}-\d{2}", query or "")
    if explicit: return min(explicit), max(explicit)
    # "today I learned X, did I learn it before" is a historical topic query.
    if any(cue in query for cue in ("以前", "之前", "曾经", "上次", "记得")):
        return None
    today = dt.date.today()
    if "今天" in query: return today.isoformat(), today.isoformat()
    if "昨天" in query:
        day = today - dt.timedelta(days=1); return day.isoformat(), day.isoformat()
    return None


def _lexical(query: str, docs: list[dict[str, Any]], pool: int) -> list[tuple[str, float]]:
    wanted = set(_tokens(query)); ranked = []
    if not wanted: return []
    for doc in docs:
        tokens = _tokens(doc["title"] + "\n" + doc["text"])
        overlap = wanted & set(tokens)
        if overlap:
            score = sum(1 + math.log1p(tokens.count(token)) for token in overlap) / math.sqrt(max(1, len(set(tokens))))
            ranked.append((doc["id"], score))
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:pool]


def _dense(query: str, valid_ids: set[str], pool: int,
           store: MemoryStore) -> tuple[list[tuple[str, float]], str, dict[str, str]]:
    if os.environ.get("MEMORY_DENSE", "1") != "1": return [], "disabled", {}
    try:
        from rag_tools import get_embedding
        query_vector = get_embedding(query)
        best: dict[str, float] = {}
        snippets: dict[str, str] = {}
        for row in store.list_search_chunks(_embedding_profile()):
            event_id = str(row["event_id"])
            if event_id not in valid_ids or int(row["embedding_dim"]) != len(query_vector):
                continue
            vector = _unpack_vector(row["embedding"], int(row["embedding_dim"]))
            score = math.fsum(a * b for a, b in zip(query_vector, vector))
            if score > best.get(event_id, -1.0):
                best[event_id] = score
                snippets[event_id] = str(row["content"])
        return sorted(best.items(), key=lambda item: item[1], reverse=True)[:pool], "ok", snippets
    except Exception as exc:
        return [], f"unavailable:{type(exc).__name__}", {}


def _experience_recall(query: str) -> bool:
    return any(cue in query for cue in (
        "学过", "学习过", "做过", "实践过", "完成过", "是否接触过", "有没有学",
    ))


def _lexical_snippet(text: str, query: str, size: int = 700) -> str:
    if len(text) <= size:
        return text
    positions = [text.casefold().find(token) for token in _tokens(query)]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - size // 3)
    end = min(len(text), start + size)
    start = max(0, end - size)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def search_personal_memory(query: str, *, purpose: str = "recall", limit: int = 5,
                           target_context_id: str | None = None, include_archived: bool | None = None,
                           exclude_event_ids: set[str] | None = None,
                           exclude_operation_id: str = "",
                           exclude_latest_exact_user_query: bool = False) -> dict[str, Any]:
    store = MemoryStore(); target = target_context_id or store.active_goal_id()
    include_archived = purpose == "recall" if include_archived is None else include_archived
    docs = event_documents(store, include_archived=include_archived,
                           target_context_id=None if include_archived else target)
    excluded = exclude_event_ids or set()
    docs = [doc for doc in docs if doc["id"] not in excluded and
            (not exclude_operation_id or doc.get("operation_id") != exclude_operation_id)]
    if _experience_recall(query):
        actual = [doc for doc in docs if doc.get("evidence_role") != "planned"]
        if actual:
            docs = actual
    if exclude_latest_exact_user_query:
        exact = [doc for doc in docs if doc.get("source_type") == "conversation_message"
                 and doc.get("text", "").strip() == query.strip()]
        if exact:
            newest = max(exact, key=lambda doc: doc.get("seq", 0))["id"]
            docs = [doc for doc in docs if doc["id"] != newest]
    wanted = _date_range(query)
    if wanted:
        docs = [doc for doc in docs if wanted[0] <= doc.get("date", "") <= wanted[1]]
    pool = max(20, limit * 4)
    lexical = _lexical(query, docs, pool)
    dense, dense_status, dense_snippets = _dense(
        query, {doc["id"] for doc in docs}, pool, store)
    scores: dict[str, float] = {}; methods: dict[str, set[str]] = {}
    for method, ranked in (("lexical", lexical), ("dense", dense)):
        for rank, (doc_id, _score) in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            methods.setdefault(doc_id, set()).add(method)
    by_id = {doc["id"]: doc for doc in docs}
    ordered = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:max(1, limit)]
    items = []
    for doc_id in ordered:
        doc = dict(by_id[doc_id])
        full_text = doc.pop("text")
        excerpt = dense_snippets.get(doc_id) or _lexical_snippet(full_text, query)
        doc.update({"text": excerpt[:1200], "content_truncated": len(full_text) > len(excerpt),
                    "score": round(scores[doc_id], 6),
                    "matched_by": "+".join(sorted(methods[doc_id]))})
        items.append(doc)
    return {"status": "ok", "purpose": purpose, "items": items,
            "dense_status": dense_status, "degraded": dense_status != "ok",
            "target_context_id": target, "date_filter": wanted}
