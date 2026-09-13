# -*- coding: utf-8 -*-
"""rag_quota.py — 永不归零的语言配额召回(外部方案报告 v1.0 §9.3/§10.2 Q2/§11)。

**取代硬路由的理由(报告 §8.2,与本项目实测吻合)**:
  硬路由一旦误判 → 某语言候选数直接=0 → BM25/RRF/Reranker/LLM 全都无法补救。
  本项目 2026-08-18 的实测账正是这条路的两个失败面:
  松门=论文块顶替中文弱证据(52 题真实口径 -7.7pp);比分门=收益被安全边界吃光(净增益 0)。
  配额的不同在于**中文候选始终留在池子里同台竞争**——英文只是"增加",不是"顶替",
  所以劫持在构造上不可能发生;而英文候选也永不归零,后面几层才有救回的余地。

**为什么配额必须是独立 RRF 通道(实现要点,踩过就懂)**:
  若把配额结果按距离并进主 dense 列表再融合,英文块在 BGE 空间里距离天然偏大
  (报告 §6:E5 下英文块中位排名 1512,BGE 下 9),排到列表尾部 → RRF 截断时首先被丢掉,
  配额等于没加。作为独立通道后,配额第 1 名拿 1/(k+1),与全局 dense 第 1 名同权,
  必定活到精排——这才叫"永不归零"。

分区依据是 language 而非 source_type(报告 §9.2):英文资料未来还有 blog/manual/api_doc,
按 source_type 枚举会漏项;瓶颈本质是语言竞争,与是不是论文无关。
"""
from __future__ import annotations

import os

BASE = os.path.dirname(os.path.abspath(__file__))

def quota_collection() -> str:
    """英文/混排分区所在集合(**惰性读 env**)。当前=论文域(唯一英文来源);
    未来 blog/manual 一并入此集合即可。

    必须惰性(2026-08-18 实测):原为模块级常量,import 时冻结——Phase 4 的 A/B 换集合
    完全不生效,E5 臂拿 e5 的 query 向量去查了 BGE 的库,跑出 6%(纯 e5 域实测是 96%)。
    "结果离谱到不可能"是发现这类 bug 的信号。同款冻结坑本项目已在 RAG_SYNTH_MODEL、
    RAG_RECALL_N 上各栽过一次。
    """
    return os.environ.get("RAG_QUOTA_COLLECTION", "kb_paper_bge_v1")


def quota_enabled() -> bool:
    """默认关——测正才采纳(同 HyDE/doc2query/CRAG/paper_route 的家法)。"""
    return os.environ.get("RAG_EN_QUOTA", "0") == "1"


def quota_k() -> int:
    """英文/混排保险配额大小。报告 §29 V1 起点=10,§10.2 待扫 5/8/10/15/20。"""
    try:
        return max(0, int(os.environ.get("RAG_EN_QUOTA_K", "10")))
    except ValueError:
        return 10


def rerank_pool_size(main_n: int, quota_n: int) -> int:
    """进精排的候选池大小 = **主池 + 配额池(相加)**,而不是截到某个固定值。

    为什么必须相加(2026-08-18 实测教训):最初按报告 §17 的"24~32"把融合池截成固定 24,
    结果 5 个英文候选**挤掉了 4 个中文候选**——被挤出去的位置腾给了原本进不了精排池的
    另一个中文块,它精排分更高,把正确答案从第 1 顶到第 2。held-out 52 因此 -3.8pp,
    而这与英文毫无关系,纯粹是"顶替"带来的中文子序扰动。
    报告 §10.1 写的本来就是"Top20 + Top20 → 合并 40":配额是**保险池**,语义是增加,
    截断会把"增加"悄悄变回"顶替",正是配额要取代的那个毛病。
    RAG_RERANK_POOL 可显式覆盖(做池深消融时用)。
    """
    override = os.environ.get("RAG_RERANK_POOL", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return main_n + quota_n


_COL_CACHE: dict = {}   # (db_path, name) -> 集合句柄 | None(缺失→短路,fail-soft)


def _get_quota_collection():
    """首查探测并缓存结论(含缺失)。缺失时配额通道静默退化为不加英文候选。"""
    db_path = os.path.join(BASE, "chroma_db")
    name = quota_collection()
    key = (db_path, name)
    if key not in _COL_CACHE:
        try:
            import chromadb
            _COL_CACHE[key] = chromadb.PersistentClient(path=db_path) \
                .get_collection(name)
        except Exception:
            _COL_CACHE[key] = None
    return _COL_CACHE[key]


def quota_query_embedding(question: str, main_emb: list) -> list:
    """配额通道的 query 向量。默认**复用主库向量**(同一空间,距离可比、精排同分布)。

    配 RAG_QUOTA_EMBED_MODEL 则独立编码——供报告 §19/§31 在**统一配额条件下**比较
    多语模型(e5/m3)做英文分区的检索器。报告明令不许拿"BGE+配额 vs M3 无配额"比,
    候选策略必须一致,只换 Dense 模型这一个变量。
    """
    model = os.environ.get("RAG_QUOTA_EMBED_MODEL", "").strip()
    if not model:
        return main_emb
    from rag_tools import _load_local_model
    m = _load_local_model(model)
    try:
        m.max_seq_length = int(os.environ.get("OFFERCLAW_EMBED_MAX_SEQ", "512") or 512)
    except ValueError:
        pass
    prefix = os.environ.get("RAG_QUOTA_EMBED_PREFIX", "")   # e5 契约:query 侧 "query: "
    return [m.encode([f"{prefix}{question}"], normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)[0].tolist()]


def quota_candidates(emb: list, k: int | None = None, question: str = "") -> list:
    """英文/混排保险池:查英文分区,返回 [(doc, meta, dist)]。

    默认与主库同 embedding 空间——距离要能与主池同尺度比较,精排也才在同一分布上打分。
    语言过滤走 metadata(document_language ∈ {en, mixed});老块无该字段则不过滤
    (集合本身即英文分区,过滤只是未来多语混存时的保险)。
    """
    k = quota_k() if k is None else k
    if k <= 0:
        return []
    col = _get_quota_collection()
    if col is None:
        return []
    try:
        emb = quota_query_embedding(question, emb) if question else emb
        from rag_lang import QUOTA_LANGS
        kw = {"query_embeddings": emb, "n_results": k,
              "include": ["documents", "metadatas", "distances"]}
        try:
            r = col.query(where={"document_language": {"$in": list(QUOTA_LANGS)}}, **kw)
        except Exception:
            r = None
        # 空结果**不抛异常**(2026-08-18 实测):没有 document_language 字段的老集合会静静
        # 返回 0 条,于是"配额通道存在但永远为空"——比报错更难发现。故按空结果再降级一次。
        if not r or not (r.get("documents") or [[]])[0]:
            r = col.query(**kw)          # 老集合无语言字段 → 不过滤(集合本身即英文分区)
        docs = (r.get("documents") or [[]])[0]
        metas = (r.get("metadatas") or [[]])[0]
        dists = (r.get("distances") or [[]])[0]
        return list(zip(docs, metas, dists))
    except Exception:
        return []


def append_quota_pool(docs: list, metas: list, dists: list,
                      quota_hits: list, lex_hits: list | None = None,
                      fallback_dist: float = 1.0) -> tuple:
    """把英文保险池**整段追加**到主池尾部(去重),返回 (docs, metas, dists, provenance)。

    为什么是"追加"而不是"与主池一起 RRF 后截断"(2026-08-18 实测教训,已写进 rerank_pool_size):
    任何形式的截断竞争都会让英文候选挤掉中文候选,而腾出的位置会放进基线里本来
    进不了精排池的另一个中文块——它精排分更高,就把正确答案顶下去。held-out 52 实测
    -3.8pp,三题里有两题的"凶手"根本是中文块。配额的语义是**增加**,追加才是它的忠实实现:
    主池与基线逐字节同构(同样的 dense+BM25 融合、同样的深度),英文只多不少,
    要抢第一必须在精排分上真赢过中文——那才是我们想要的竞争。
    """
    prov = {d: ["dense_global"] for d in docs}
    out_d, out_m, out_k = list(docs), list(metas), list(dists)
    seen = set(docs)
    for name, hits in (("dense_en_quota", quota_hits), ("bm25_en", lex_hits or [])):
        for item in hits:
            d, m = item[0], item[1]
            dist = item[2] if len(item) > 2 and item[2] is not None else fallback_dist
            if d in seen:
                prov.setdefault(d, []).append(name)
                continue
            seen.add(d)
            out_d.append(d), out_m.append(m), out_k.append(dist)
            prov[d] = [name]
    return out_d, out_m, out_k, prov


def rrf_multi(channels: dict, top_n: int, k: int = 60,
              fallback_dist: float = 1.0) -> tuple:
    """多通道 RRF:channels = {通道名: [(doc, meta, dist|None), ...]}(各通道内已排序)。

    RRF(d) = Σ_通道 1/(k + 该通道内排名)。去重 key=文档原文;距离取**各通道见过的最小真实距离**
    (同一块被多通道召回时,dense 的真实距离优先于词法通道的占位距离)。
    返回 (docs, metas, dists, provenance):provenance[doc] = 命中它的通道名集合,
    用于报告 §33 的可观测性("该候选为什么进入池子")与 §34 的失败分类。
    """
    fused: dict = {}
    for ch_name, items in channels.items():
        for rank, item in enumerate(items, start=1):
            doc, meta = item[0], item[1]
            dist = item[2] if len(item) > 2 else None
            ent = fused.get(doc)
            if ent is None:
                ent = fused[doc] = {"score": 0.0, "doc": doc, "meta": meta,
                                    "dist": None, "channels": set()}
            ent["score"] += 1.0 / (k + rank)
            ent["channels"].add(ch_name)
            if dist is not None and (ent["dist"] is None or dist < ent["dist"]):
                ent["dist"] = dist
    # 并列打破(指导文档 §18 B1,默认关):多通道各自 rank1 的 RRF 分**完全相同**
    # (都是 1/(k+1)),稳定排序于是恒定偏向先插入的通道(dense_global=中文)。
    # 这就是消融里 S0 "R@1=0% 但 R@3=84%" 的成因——不是 dense 失效,是并列被系统性判负。
    # 开启后用真实向量距离二次排序(配额与主池现已同为 L2² 空间,可比);
    # 无真实距离的纯词法候选按 fallback 处理,并以文档哈希兜底保证确定性。
    if os.environ.get("RAG_RRF_TIEBREAK", "0") == "1":
        # 末位判据用文档串本身而非 hash():CPython 对 str 的 hash 按进程加盐,
        # 用它会让同配置跨进程结果漂移——正是本轮要消除的那类不可复现。
        ranked = sorted(fused.values(),
                        key=lambda e: (-e["score"],
                                       e["dist"] if e["dist"] is not None else fallback_dist,
                                       e["doc"]))[:top_n]
    else:
        ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:top_n]
    docs = [e["doc"] for e in ranked]
    metas = [e["meta"] for e in ranked]
    dists = [(e["dist"] if e["dist"] is not None else fallback_dist) for e in ranked]
    prov = {e["doc"]: sorted(e["channels"]) for e in ranked}
    return docs, metas, dists, prov


def cap_per_source(docs: list, metas: list, dists: list, cap: int | None = None) -> tuple:
    """每来源块数上限(报告 §12):防同一篇文档的多个相似 chunk 占满候选池。

    cap<=0 或未配置 → 不启用(逐字节不变)。保序截断,不改相对排名。
    """
    if cap is None:
        try:
            cap = int(os.environ.get("RAG_MAX_CHUNKS_PER_SOURCE", "0"))
        except ValueError:
            cap = 0
    if not cap or cap <= 0:
        return docs, metas, dists
    seen: dict = {}
    kd, km, kk = [], [], []
    for d, m, dist in zip(docs, metas, dists):
        src = (m or {}).get("source", "?")
        if seen.get(src, 0) >= cap:
            continue
        seen[src] = seen.get(src, 0) + 1
        kd.append(d), km.append(m), kk.append(dist)
    return kd, km, kk


def pool_stats(metas: list, dists: list, prov: dict | None, docs: list) -> dict:
    """候选池指标(报告 §26.1):判断"召回池是否修好"的核心观测,与最终 R@1 分开看。"""
    from rag_lang import is_quota_lang
    langs = [(m or {}).get("document_language") for m in metas]
    first_en = 0
    for i, (m, lg) in enumerate(zip(metas, langs), start=1):
        if is_quota_lang(lg or "") or (m or {}).get("source_type") == "paper":
            first_en = i
            break
    return {
        "pool_size": len(docs),
        "english_candidate_count": sum(
            1 for m, lg in zip(metas, langs)
            if is_quota_lang(lg or "") or (m or {}).get("source_type") == "paper"),
        "first_english_rank": first_en,          # 0=池中无英文候选
        "channels": sorted({c for cs in (prov or {}).values() for c in cs}),
        # 精排前的池内来源顺序:评测器据此算 Candidate Recall 与 Target Chunk Rank
        # (报告 §26.1)。池召回=100% 而最终 R@1 掉 → 损伤在排序层,不是召回层。
        "sources": [(m or {}).get("source", "") for m in metas],
    }
