"""Round 4：元数据 / 文件名感知路由。

针对「同质文档」短板（career 的 05_零基础 / 06_后端转型 / 07_算法转型 三文件约 95% 雷同，
向量 / BM25 / rerank 都分不开唯一区分点——目标人群）。文件名已天然编码人群
（zero_foundation / backend / algorithm），source 又在 ChromaDB meta 里，故无需重新入库：
检索后，若 query 命中人群意图，就在**这组 career 路径文件之间**校正排序——
匹配人群的提权、竞争人群的降权；**其他域文件一律中性、不参与**（避免误伤 algorithm 域八股等）。

这是「检索侧 + 文档侧元数据」结合的精确路由，补 query 侧（HyDE，已证无效）解决不了的同质问题。
默认开（RAG_ROUTE=1）；仅在 query 同时包含人群词和转型/入门路径诉求，且候选含
career 路径文件时才动。人群词只作为比较对象时不得触发（例如“应用岗与算法岗的
学历门槛有何不同”），避免把路径文档错误提升到技术或岗位事实答案之前。
"""
import os

# query 人群词 → 文件名子串标识（career 路径文件名里编码的人群）
_AUDIENCE = {
    "算法岗": "algorithm", "算法工程师": "algorithm", "算法岗位": "algorithm", "算法": "algorithm",
    "后端工程师": "backend", "后端开发": "backend", "服务端": "backend", "后端": "backend",
    "零基础": "zero_foundation", "小白": "zero_foundation", "非科班": "zero_foundation",
    "转行": "zero_foundation", "跨专业": "zero_foundation",
}
# career「学习路径」类文件的共同文件名特征（只在这组内部做人群路由）
_PATH_MARKERS = ("transition_path", "foundation_path")
_ALL_TAGS = ("algorithm", "backend", "zero_foundation")
_PATH_INTENT_TERMS = (
    "怎么转", "如何转", "转型", "转行", "入门", "学习路径", "学习路线",
    "成长路径", "成长路线", "路线怎么走", "路径怎么走", "怎么学", "如何学",
    "转大模型", "转ai", "转 ai", "转人工智能", "转应用开发",
)


def route_enabled() -> bool:
    return os.environ.get("RAG_ROUTE", "1") == "1"


def audience_intent(question: str) -> str:
    """识别 query 的人群意图，返回文件名标识（algorithm/backend/zero_foundation）或 ''。
    只有明确询问转型、入门或学习路径时才启用；单纯提到某个人群可能是在比较。
    长词优先匹配（"算法岗" 先于 "算法"），避免短词误命中。"""
    q = (question or "").lower()
    if not any(term in q for term in _PATH_INTENT_TERMS):
        return ""
    for kw in sorted(_AUDIENCE, key=len, reverse=True):
        if kw.lower() in q:
            return _AUDIENCE[kw]
    return ""


def apply_audience_routing(question, docs, metas, dists):
    """按人群意图在 career 路径文件之间校正排序；非 career 文件保持原相对次序。"""
    if not route_enabled():
        return docs, metas, dists
    intent = audience_intent(question)
    if not intent:
        return docs, metas, dists

    def bucket(m) -> int:
        src = (m or {}).get("source", "").lower()
        is_path = any(mk in src for mk in _PATH_MARKERS)
        if not is_path:
            return 1                       # 非 career 路径文件：中性，不参与路由
        if intent in src:
            return 0                       # 匹配目标人群：提到最前
        if any(t in src for t in _ALL_TAGS):
            return 2                       # 竞争人群的同质文件：降到最后
        return 1

    order = sorted(range(len(docs)), key=lambda i: (bucket(metas[i]), i))  # 稳定排序，桶内保序
    return ([docs[i] for i in order], [metas[i] for i in order], [dists[i] for i in order])
