"""Round 2：BM25 稀疏检索 + RRF 融合。

补单塔向量（密集）在「同主题多源 / 中文术语」上的召回短板：
- 密集向量擅长语义，但近义子主题（混合检索 / 索引优化 / 查询改写）在语义空间里挤在一起，区分不开；
- BM25（稀疏 / 词法）按词频精确匹配关键词（"稀疏向量""算法转开发"等中文术语），恰好互补；
- 两路各取 RECALL_N，用 RRF（Reciprocal Rank Fusion，倒数排名融合）合并候选池 → 交给 rerank 精排。

现有 rag_gate.query_keywords 只抽英文 token（"中文交给向量"），故中文术语题（rag09 / career）
得不到词法救援。本模块用 jieba 中文分词补齐。对应知识库 all-in-rag ch4「混合检索」。

任意环节失败（依赖缺失 / 库异常）→ 返回空，调用方自动退回纯向量，不影响主链路。
"""
import os
import threading

# 进程级 BM25 索引缓存：按 collection.count() 判失效（增量入库后自动重建）
_CACHE = {"count": None, "bm25": None, "docs": None, "metas": None}
_LOCK = threading.Lock()


def bm25_enabled() -> bool:
    """默认开（生产）；测试 conftest 通过环境变量关掉避免建索引。"""
    return os.environ.get("RAG_BM25", "1") == "1"


def _tokenize(text: str) -> list:
    """中英混合分词：jieba 切中文，英文/数字成词，统一小写。查询与文档共用，保证对齐。"""
    import jieba
    out = []
    for t in jieba.cut(text or ""):
        t = t.strip().lower()
        if t and not t.isspace():
            out.append(t)
    return out


_CACHES: dict = {}      # collection_name -> cache dict(2026-08-10:按集合分桶,论文域可用)


def _build_index(collection_name: str | None = None) -> dict:
    """从 ChromaDB 取全部文档建 BM25 索引；**按集合**进程级缓存，记录数变化即重建。

    2026-08-10 review 修复④:原实现写死主库 collection,于是论文域(术语最密集、
    BM25 最该发挥作用的地方)反而只有纯 dense。参数化后论文路由可复用同一套词法通道
    ——"英文专有名词经词法通道直通英文文档"正是本项目实测的架构级发现。
    """
    from rank_bm25 import BM25Okapi
    from rag_tools import get_collection_name, get_collection_sqlite_documents
    name = collection_name or get_collection_name()
    base = os.path.dirname(os.path.abspath(__file__))
    docs, metas = get_collection_sqlite_documents(os.path.join(base, "chroma_db"), name)
    count = len(docs)
    with _LOCK:
        cache = _CACHES.setdefault(name, {"count": -1, "bm25": None, "docs": [], "metas": []})
        if cache["count"] == count and cache["bm25"] is not None:
            return dict(cache)
        bm25 = BM25Okapi([_tokenize(d) for d in docs])
        cache.update(count=count, bm25=bm25, docs=docs, metas=metas)
        if name == get_collection_name():
            _CACHE.update(cache)     # 兼容既有 _CACHE 观察者
    return dict(cache)


def bm25_search(query: str, top_n: int, collection_name: str | None = None, *,
                source_types: set[str] | None = None,
                exclude_source_types: set[str] | None = None,
                metadata_filters: dict[str, set[str]] | None = None) -> list:
    """BM25 稀疏检索，返回 [(doc, meta, score), ...] 至多 top_n（仅 score>0）。

    Metadata constraints are applied *before* selecting Top-N.  Filtering a
    global Top-N afterwards can lose every relevant chunk from a small source
    batch even though those chunks have strong lexical scores within the
    requested subset.

    任意异常（依赖缺失 / 空库）→ []，调用方退回纯向量。
    """
    if not bm25_enabled():
        return []
    try:
        cache = _build_index(collection_name)
        if not cache["docs"]:
            return []
        scores = cache["bm25"].get_scores(_tokenize(query))
        source_types = {str(value) for value in (source_types or set()) if value}
        exclude_source_types = {
            str(value) for value in (exclude_source_types or set()) if value
        }
        metadata_filters = {
            str(key): {str(value) for value in values if value}
            for key, values in (metadata_filters or {}).items() if values
        }

        def allowed(index: int) -> bool:
            meta = cache["metas"][index] or {}
            source_type = str(meta.get("source_type") or "")
            if source_types and source_type not in source_types:
                return False
            if exclude_source_types and source_type in exclude_source_types:
                return False
            return all(str(meta.get(key, "")) in values
                       for key, values in metadata_filters.items())

        eligible = [index for index in range(len(scores)) if allowed(index)]
        order = sorted(eligible, key=lambda index: float(scores[index]), reverse=True)[:top_n]
        return [(cache["docs"][i], cache["metas"][i], float(scores[i]))
                for i in order if scores[i] > 0]
    except Exception:
        return []


def rrf_fuse(vec_docs, vec_metas, vec_dists, bm25_results,
             top_n, k: int = 60, bm25_only_dist: float = 1.0,
             dense_weight: float = 1.0, bm25_weight: float = 1.0):
    """RRF 倒数排名融合，返回融合后的 (docs, metas, dists) 至多 top_n。

    RRF(d) = Σ_路  1 / (k + rank_路(d))，rank 从 1 起；按 RRF 降序取 top_n。
    - 去重 key = 文档原文（同一 chunk 两路都召回则 RRF 相加，排得更前）；
    - 向量召回的文档保留**真实向量距离**（供 gate 后 cutoff 用）；
    - 仅 BM25 命中（向量没召回）的文档无真实距离，用 bm25_only_dist 占位
      （调用方传 rescue 阈值：表「词法命中、向量中等距离」，cutoff 行为可控）。
    k=60 为 RRF 论文与工业常用默认，弱化绝对分数尺度差异、只看排名。
    """
    fused = {}  # doc -> [rrf_score, doc, meta, dist]
    for rank, (d, m, dist) in enumerate(zip(vec_docs, vec_metas, vec_dists), start=1):
        ent = fused.setdefault(d, [0.0, d, m, dist])
        ent[0] += dense_weight / (k + rank)
    for rank, (d, m, _score) in enumerate(bm25_results, start=1):
        if d in fused:
            fused[d][0] += bm25_weight / (k + rank)
        else:
            fused[d] = [bm25_weight / (k + rank), d, m, bm25_only_dist]
    ranked = sorted(fused.values(), key=lambda x: x[0], reverse=True)[:top_n]
    return ([r[1] for r in ranked], [r[2] for r in ranked], [r[3] for r in ranked])
