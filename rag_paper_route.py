# -*- coding: utf-8 -*-
"""rag_paper_route.py — 论文域回退路由(P1 生产接线,2026-08-09)。

架构(方案文档分域 + 三臂 A/B 终裁):
  中文主库(base-zh,已纯化不含论文) → 证据弱/拒答 → 本模块:
  e5 论文域(kb_paper_e5_v1,multilingual-e5-base 按需加载) → 双证据门 → 强则改用论文证据。

零劫持 by construction:只在主库判弱后触发,中文业务问题(主库强证据)永不进入。
双证据门(标定于 docs/rag_eval/m3_ab/REPORT.md 补充章,50 正例 vs 16 近域负例):
  rerank_top ≥ 0.05  OR  最小 e5 距离 ≤ 0.32  → 正例 39/50 过 / 负例 0/16 误入。
  纯距离门不可分(14/16 负例混入正例区间)——"阈值不是旋钮"的 e5 版重演,精排分才是利刃;
  跨语低精排分尾巴(找对但不敢确信的 ~11/50)按家法宁拒不编,翻译通道未来可提置信。
e5 使用契约:query 侧前缀 "query: "(入库侧 passage:,建库时已带)。
"""
from __future__ import annotations

import os

BASE = os.path.dirname(os.path.abspath(__file__))

PAPER_COLLECTION = os.environ.get("RAG_PAPER_COLLECTION", "kb_paper_e5_v1")
BGE_PAPER_COLLECTION = "kb_paper_bge_v1"
E5_MODEL_DEFAULT = os.path.expanduser(
    "~/.cache/modelscope/hub/models/intfloat/multilingual-e5-base")


def paper_profile() -> str:
    """显式论文检索 Profile。默认保持历史基线，避免影响普通生产查询。

    ``paper_quality`` 只由 ``paper_kb`` 的显式路由调用：使用已标定的 BGE
    论文配额集合、英文 ONNX 分工桥和 0.80 英文证据门。它不修改进程级
    主检索配置，因此中文知识库、个人记忆和实时状态查询均为零额外开销。
    """
    value = (os.environ.get("RAG_PAPER_PROFILE", "baseline") or "baseline").strip()
    return value if value in {"baseline", "paper_quality"} else "baseline"


def paper_route_enabled() -> bool:
    """**默认关**(2026-08-18,测正才采纳——同 HyDE / doc2query / CRAG 的处置)。

    两把尺子实测的完整账:
      · 松门(仅"主库判弱"入口):论文向 56%→70%,但中文真实口径 50.0→**42.3**
        (-7.7pp 真回归;当时 10 题守卫太小没抓住,52 题尺子抓住了);
      · 比分门(论文精排分须赢过主库):中文回到 50.0 ✓,论文向回落 56%
        = 与不启用路由持平 → **收益被安全边界吃光,当前净增益为 0**。
    根因:跨语精排惩罚——英文块对中文 query 的 cross-encoder 分天然偏低,
    与"主库弱但相关的中文块"比分时输掉,而放宽比分就会伤主库。
    因此默认关;代码与标定全部保留,`RAG_PAPER_ROUTE=1` 可开。
    解锁条件(下一个杠杆,未实施):翻译通道把 query 转英文再精排,消除跨语惩罚后
    重跑三线(中文不回退 + 论文向显著超 56%)再议默认值。
    """
    return os.environ.get("RAG_PAPER_ROUTE", "0") == "1"


def _paper_gate(best_dist: float, rerank_top: float,
                main_rerank_top: float | None = None) -> bool:
    """双证据门(OR)+ **比分门**(AND):论文证据还必须赢过主库自己的最佳证据。

    2026-08-18 回归修复:此前只有"主库判弱(in_kb=False)才触发"这一道入口,
    被我错误描述为"零劫持 by construction"——它只挡住主库**强**证据的情况;
    主库**弱而正确**的中文问题(如口语化求职提问)照样被论文块顶替。
    52 题真实口径实测:开路由 42.3% vs 关路由 50.0%,**-7.7pp 真回归**
    (当时 10 题守卫太小没抓住,两把尺子里的大尺子抓住了)。

    修法(采纳 review 建议:比分而非放宽入口):精排分同模型同 query 可比,
    要求 paper_rr > main_rr —— 中文求职问题上论文精排分近 0,自然不触发;
    真论文问题上主库精排分近 0,照常触发。
    """
    t_rr = float(os.environ.get("RAG_PAPER_RR", "0.05"))
    t_d = float(os.environ.get("RAG_PAPER_DIST", "0.32"))
    if not ((rerank_top >= t_rr) or (best_dist <= t_d)):
        return False
    if main_rerank_top is not None and rerank_top <= float(main_rerank_top):
        return False        # 主库自己的证据更强 → 不夺权
    return True


_PAPER_COL_CACHE: dict = {}   # (db_path, collection) -> 集合句柄 | None(缺失,短路)


def _get_paper_collection(collection_name: str | None = None):
    """首查探测论文集合并缓存结论(含缺失):无集合的库上先探测再加载,
    避免每次查询白付 1.1GB e5 加载费后才 fail-soft。进程内建好集合需重启生效。"""
    db_path = os.path.join(BASE, "chroma_db")
    name = collection_name or PAPER_COLLECTION
    key = (db_path, name)
    if key not in _PAPER_COL_CACHE:
        try:
            import chromadb
            _PAPER_COL_CACHE[key] = chromadb.PersistentClient(path=db_path) \
                .get_collection(name)
        except Exception:
            _PAPER_COL_CACHE[key] = None
    return _PAPER_COL_CACHE[key]


def _query_papers(question: str, n: int) -> dict | None:
    """e5 按需加载(rag_tools 进程内缓存)→ 查论文域集合。集合/模型缺失返回 None(fail-soft)。"""
    col = _get_paper_collection()
    if col is None:
        return None
    try:
        from rag_tools import _load_local_model
        model_path = os.environ.get("RAG_PAPER_EMBED_MODEL", E5_MODEL_DEFAULT)
        m = _load_local_model(model_path)
        m.max_seq_length = int(os.environ.get("OFFERCLAW_EMBED_MAX_SEQ", "512") or 512)
        emb = m.encode([f"query: {question}"], normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)[0].tolist()
        r = col.query(query_embeddings=[emb], n_results=n,
                      include=["documents", "metadatas", "distances"])
        if not r.get("documents") or not r["documents"][0]:
            return None
        return {"docs": r["documents"][0], "metas": r["metadatas"][0],
                "dists": r["distances"][0]}
    except Exception:
        return None


def _query_papers_bge(question: str, n: int) -> dict | None:
    """在与主库相同的 BGE 空间查询论文分区，供显式 ``paper_quality`` 使用。

    该路径复用已经加载/缓存的主 embedding；collection 或模型不可用时返回
    ``None``。显式 Profile 会据此安全拒答，不绕过 0.80 门回到宽松旧口径。
    """
    col = _get_paper_collection(BGE_PAPER_COLLECTION)
    if col is None:
        return None
    try:
        from rag_tools import get_embeddings_batch
        emb = get_embeddings_batch([question])[0]
        result = col.query(
            query_embeddings=[emb], n_results=max(1, n),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        if not docs:
            return None
        return {
            "docs": docs,
            "metas": (result.get("metadatas") or [[]])[0],
            "dists": (result.get("distances") or [[]])[0],
        }
    except Exception:
        return None


PAPER_RECALL_N = int(os.environ.get("RAG_PAPER_RECALL_N", "20"))


def _retrieve_paper_evidence(question: str, top_k: int = 5,
                             main_rerank_top: float | None = None) -> dict | None:
    """论文域 dense + BM25 → RRF → 精排 → 双证据门的单一实现。"""
    hit = _query_papers(question, n=max(top_k, PAPER_RECALL_N))
    if hit is None:
        return None
    docs_in, metas_in, dists_in = hit["docs"], hit["metas"], hit["dists"]
    try:                                    # 词法通道 + RRF(失败 fail-soft 退纯 dense)
        from rag_bm25 import bm25_enabled, bm25_search, rrf_fuse
        if bm25_enabled():
            lex = bm25_search(question, PAPER_RECALL_N, collection_name=PAPER_COLLECTION)
            if lex:
                docs_in, metas_in, dists_in = rrf_fuse(
                    docs_in, metas_in, dists_in, lex, top_n=max(top_k, PAPER_RECALL_N))
    except Exception:
        pass
    from rag_rerank import rerank
    docs, metas, dists, scores = rerank(question, docs_in, metas_in, dists_in, top_k)
    rr_top = float(max(scores)) if scores else 0.0
    best = float(min(hit["dists"]))   # 门控仍按原始 e5 距离(RRF 会重排但不改距离语义)
    if not _paper_gate(best, rr_top, main_rerank_top):
        return None
    sources = sorted({(m or {}).get("source", "?") for m in metas})
    return {"in_kb": True, "chunks": list(docs), "sources": sources,
            "matched_by": "paper_rerank" if rr_top >= float(os.environ.get("RAG_PAPER_RR", "0.05"))
                          else "paper_dense",
            "best": best, "has_dists": bool(dists), "rerank_top": rr_top,
            "docs": list(docs), "metas": list(metas), "dists": list(dists),
            "paper_route": True}


def _retrieve_paper_quality(question: str, top_k: int = 3) -> dict | None:
    """显式论文场景的低影响跨语言增强路径。

    Round 11/12 的生产口径是：英文候选 K=5 → base reranker 保持原有分数
    → mMiniLM ONNX 只补英文候选 → 英文证据分达到 ``tau=0.80`` 才允许
    grounded。这里不启用自动论文兜底，也不触碰主库候选，因此不会造成中文
    弱证据被英文论文顶替的旧回归。
    """
    bridge_dir = os.path.expanduser(
        (os.environ.get("RAG_RERANK_EN_ONNX_DIR", "") or "").strip()
    )
    if not bridge_dir or not os.path.isdir(bridge_dir):
        return None
    hit = _query_papers_bge(question, n=5)
    if hit is None:
        return None
    docs_in, metas_in, dists_in = hit["docs"], hit["metas"], hit["dists"]
    from rag_rerank import rerank
    docs, metas, dists, scores = rerank(
        question, docs_in, metas_in, dists_in, top_k,
        en_bridge_idx=list(range(len(docs_in))),
    )
    rr_top = float(scores[0]) if scores else 0.0
    try:
        threshold = float(os.environ.get("RAG_EN_GATE_MIN", "0.80") or 0.80)
    except ValueError:
        threshold = 0.80
    if rr_top < threshold:
        return None
    sources = sorted({(meta or {}).get("source", "?") for meta in metas})
    return {
        "in_kb": True,
        "chunks": list(docs),
        "sources": sources,
        "matched_by": "paper_quality_en_gate",
        "best": float(min(dists_in)) if dists_in else 99.0,
        "has_dists": bool(dists),
        "rerank_top": rr_top,
        "docs": list(docs),
        "metas": list(metas),
        "dists": list(dists),
        "paper_route": True,
        "retrieval_profile": "paper_quality",
        "english_gate_threshold": threshold,
    }


def retrieve_papers_explicit(question: str, top_k: int = 3) -> dict | None:
    """显式“论文/文献”意图使用的论文检索；不受自动兜底开关影响，但仍过证据门。"""
    if paper_profile() == "paper_quality":
        # Profile 一旦显式选定，就必须服从其 0.80 英文证据门；不能在门拒绝后
        # 再悄悄回到宽松 E5 口径，否则会把已标定的「宁拒不编」语义绕开。
        return _retrieve_paper_quality(question, top_k=top_k)
    return _retrieve_paper_evidence(question, top_k=top_k, main_rerank_top=None)


def maybe_paper_route(question: str, main_in_kb: bool, top_k: int = 5,
                      main_rerank_top: float | None = None) -> dict | None:
    """主库证据弱时的论文域回退:dense + BM25 → RRF → 精排 → 双证据门。

    2026-08-10 review 修复④⑤:
    ④ 补 BM25 词法通道(论文域术语最密集,`main context` 这类精确术语 dense 易漂到
      语义邻居;词法通道是本项目实测的跨语主力)+ RRF 融合,与主链路同构;
    ⑤ 召回池 5 → 20:原来"5 进 5 出"让交叉编码器毫无挑选空间(等于白跑),
      且语料扩到几十篇论文后 5 个候选必然召不全——现在与主链路 RECALL_N 一致。
    """
    if not paper_route_enabled() or main_in_kb:
        return None
    return _retrieve_paper_evidence(question, top_k=top_k,
                                    main_rerank_top=main_rerank_top)
