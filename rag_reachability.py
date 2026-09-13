# -*- coding: utf-8 -*-
"""rag_reachability.py — 英文/论文域可达性自检(指导文档 §2.2 / §29 失败分类)。

**为什么要有这个**:2026-08-18 实测发现,两个**各自正确**的决策叠加出了没人预期的结果——
① 论文块从主库撤出(纯化);② 论文路由因劫持回归默认关。
于是"论文集合有 194 块、代码路径俱全",但生产上**一个英文候选入口都没启用**,
50 道论文题全 0%,而文档里还挂着 56% 的旧数字。没有任何测试会失败,因为每一处都"对"。

本模块把这条**跨模块的联合不变量**变成可断言的东西:
  英文语料存在(count>0) ⟹ 至少有一条英文召回通道处于启用状态。
违反时报 `PAPER_RETRIEVAL_UNREACHABLE`,并附可执行的修法。

顺带校验集合 schema(指导文档 §11.2):元数据过滤依赖 `document_language`,
老集合没有该字段时过滤会**静默返回空**而不报错,表现为"通道存在但永远为空"。
"""
from __future__ import annotations

import os

BASE = os.path.dirname(os.path.abspath(__file__))

PAPER_RETRIEVAL_UNREACHABLE = "PAPER_RETRIEVAL_UNREACHABLE"
COLLECTION_METADATA_SCHEMA_MISMATCH = "COLLECTION_METADATA_SCHEMA_MISMATCH"


def _collection_count(name: str) -> int:
    """集合不存在/库不可用 → 0(视为"没有英文语料",不是错误)。"""
    try:
        import chromadb
        return chromadb.PersistentClient(
            path=os.path.join(BASE, "chroma_db")).get_collection(name).count()
    except Exception:
        return 0


def english_entrypoints() -> dict:
    """列出所有英文召回入口及其启用状态。新增通道时在此登记,自检自动覆盖。

    2026-08-24 盲区修复:入口"开着"不等于"可用"——kb_paper_bge_v1 曾被集合清理删除,
    配额开关仍为 1 但通道 fail-soft 成永远空,本自检当时误报 ok。
    故入口判定同时要求**其背后的集合存在且非空**。
    """
    from rag_paper_route import paper_route_enabled
    from rag_quota import quota_collection, quota_enabled, quota_k
    return {
        "quota": bool(quota_enabled() and quota_k() > 0
                      and _collection_count(quota_collection()) > 0),
        "paper_route": bool(paper_route_enabled() and _collection_count(
            os.environ.get("RAG_PAPER_COLLECTION", "kb_paper_e5_v1")) > 0),
    }


def check_reachability(strict: bool | None = None) -> dict:
    """返回 {ok, code, detail, corpora, entrypoints}。

    ``strict`` 未指定时读 env `RAG_REACHABILITY_STRICT`(默认关)。默认不抛异常——
    英文配额当前是实验开关、默认关闭是**有意的决策**,自检的职责是让这个状态
    **显式可见**,而不是替决策者翻案。开 strict 后用于 CI/发布前闸门。
    """
    from rag_quota import quota_collection
    corpora = {name: _collection_count(name)
               for name in {quota_collection(),
                            os.environ.get("RAG_PAPER_COLLECTION", "kb_paper_e5_v1")}}
    total_en = sum(corpora.values())
    eps = english_entrypoints()
    ok = (total_en == 0) or any(eps.values())
    detail = ""
    if not ok:
        detail = (
            f"英文语料共 {total_en} 块(集合 {corpora}),但英文召回入口全部关闭 {eps}。"
            "这批语料在生产上完全不可达——检索链路不会报错,只会永远查不到它们。"
            "修法:① 启用配额 RAG_EN_QUOTA=1(推荐,候选只增不替);"
            "② 或启用 RAG_PAPER_ROUTE=1(硬路由,已实测有劫持风险);"
            "③ 或把英文块并入主库并重建索引;④ 或确实不需要它们 → 删除集合以消除歧义。")
    if strict is None:
        strict = os.environ.get("RAG_REACHABILITY_STRICT", "0") == "1"
    if strict and not ok:
        raise RuntimeError(f"{PAPER_RETRIEVAL_UNREACHABLE}: {detail}")
    return {"ok": ok, "code": None if ok else PAPER_RETRIEVAL_UNREACHABLE,
            "detail": detail, "corpora": corpora, "entrypoints": eps}


def check_language_schema(collection_name: str | None = None, sample: int = 50) -> dict:
    """校验集合是否带 `document_language`——缺失时语言过滤会静默返回空(§11.2)。"""
    from rag_quota import quota_collection
    name = collection_name or quota_collection()
    try:
        import chromadb
        col = chromadb.PersistentClient(
            path=os.path.join(BASE, "chroma_db")).get_collection(name)
        metas = [m for m in (col.get(limit=sample, include=["metadatas"])
                             .get("metadatas") or []) if m]
    except Exception as e:
        return {"ok": True, "code": None, "detail": f"集合不可读,跳过:{e}", "collection": name}
    if not metas:
        return {"ok": True, "code": None, "detail": "空集合,跳过", "collection": name}
    n_lang = sum(1 for m in metas if m.get("document_language"))
    ok = n_lang == len(metas)
    return {
        "ok": ok,
        "code": None if ok else COLLECTION_METADATA_SCHEMA_MISMATCH,
        "detail": "" if ok else (
            f"{name}: {len(metas) - n_lang}/{len(metas)} 个抽样块缺 document_language。"
            "语言过滤在这类块上会静默返回空(不报错),表现为'配额通道存在但永远为空'。"
            "修法:用 scripts/build_paper_bge_collection.py 重建,或补写该字段。"),
        "collection": name, "with_language": n_lang, "sampled": len(metas),
    }


if __name__ == "__main__":
    import json
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    r, s = check_reachability(), check_language_schema()
    print(json.dumps({"reachability": r, "schema": s}, ensure_ascii=False, indent=2))
    sys.exit(0 if (r["ok"] and s["ok"]) else 1)
