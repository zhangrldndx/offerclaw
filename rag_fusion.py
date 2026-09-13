# -*- coding: utf-8 -*-
"""rag_fusion.py — 初排/精排的排名融合(指导文档 §18 Experiment 2)。

**为什么需要它**(实证动机,不是设计洁癖):
逐层消融实测(docs/rag_eval/quota/REPORT.md §4)——
  · 初排(dense 配额 + BM25 + RRF)论文向 R@3 = **84%**,候选池召回 100%;
  · 接上现役 cross-encoder 后 R@1 从 0→60%,但 R@3 **掉到 66%**。
即精排把正确候选提到 Top1 的同时,也把另一部分正确候选**踢出了 Top3**。
两种排序各有各对的地方,取其一必然丢另一半——这正是排名融合要解决的事。

四种模式(env `RAG_RANK_FUSION`,默认 `off` = 逐字节保持现状):
  off   : 纯精排序(现状)
  tie   : 仅修 RRF 并列偏置(见 rag_quota.rrf_multi 的 tie_break)
  rrf   : final = RRF(初排名次, 精排名次),k 可调 —— 名次制,不受两侧分数尺度差异影响
  blend : final = α×norm(精排名次) + (1-α)×norm(初排名次),α 可调 —— 可连续调节偏向

名次制优于分数制的理由:cross-encoder 分与 RRF 分尺度完全不可比(前者近似概率、
后者是 1/(k+rank) 的小数),混合分数需要标定;而名次天然同尺度。
"""
from __future__ import annotations

import os


def fusion_mode() -> str:
    """默认 off——测正才采纳(同 HyDE/doc2query/CRAG/quota/translate 的家法)。"""
    return (os.environ.get("RAG_RANK_FUSION", "off") or "off").strip().lower()


def _fusion_k() -> int:
    try:
        return max(1, int(os.environ.get("RAG_FUSION_K", "60")))
    except ValueError:
        return 60


def _alpha() -> float:
    """blend 模式下精排的权重;1.0=纯精排,0.0=纯初排。"""
    try:
        a = float(os.environ.get("RAG_FUSION_ALPHA", "0.5"))
    except ValueError:
        a = 0.5
    return min(1.0, max(0.0, a))


def fuse_prerank_rerank(pre_docs: list, docs: list, metas: list, dists: list,
                        scores: list, mode: str | None = None) -> tuple:
    """把精排结果与初排名次融合后重排,返回 (docs, metas, dists, scores)。

    ``pre_docs`` 是**送进精排前**的文档顺序(初排名次的来源)。
    精排未生效(scores 为空)或模式为 off → 原样返回,调用方无感知。

    注意:返回的 scores 仍是**原始 cross-encoder 分**(不是融合分)，但会与
    重排后的候选一起移动。下游 `_evidence_gate` 对最终 Top-1 取它自己的
    原始 cross-encoder 分与标定阈值(0.85/0.95)比较；不能换成融合分，也不能
    借用融合前另一个候选的分数。
    """
    mode = fusion_mode() if mode is None else mode
    if mode in ("off", "tie") or not scores or not docs:
        return docs, metas, dists, scores

    pre_rank = {d: i + 1 for i, d in enumerate(pre_docs)}
    n = len(docs)
    fallback = len(pre_docs) + 1          # 初排池里没有的(理论上不会有)排到末尾
    k = _fusion_k()

    if mode == "rrf":
        def key(i):
            return -(1.0 / (k + pre_rank.get(docs[i], fallback)) + 1.0 / (k + i + 1))
    elif mode == "blend":
        a = _alpha()

        def key(i):
            # 名次归一化到 [0,1](0=最好);两侧都用各自池深归一,避免池深不同带来的偏置
            r_post = i / max(n - 1, 1)
            r_pre = (pre_rank.get(docs[i], fallback) - 1) / max(len(pre_docs) - 1, 1)
            return a * r_post + (1 - a) * r_pre
    else:                                  # 未知模式 = 不改行为(fail-safe)
        return docs, metas, dists, scores

    order = sorted(range(n), key=key)
    return ([docs[i] for i in order], [metas[i] for i in order],
            [dists[i] for i in order] if dists else dists,
            [scores[i] for i in order] if scores else scores)


def protect_top1_margin(docs: list, metas: list, dists: list, scores: list,
                        margin: float | None = None) -> tuple:
    """English False Winner 保护(指导文档 §30 P1-11):英文候选要抢 Top1,得**赢够**。

    实测动机:配额把英文候选放进池子后,held-out 52 上 `pp19`(Redis 持久化)、
    `h9ba04`(缓存雪崩)两题被英文论文块抢走 Top1——不是因为英文块更对,
    而是 cross-encoder 的跨语分数尺度让它**险胜**了正确的中文块。

    规则:若 Top1 是英文/混排候选,而其后最靠前的中文候选分差 < margin,则让中文候选上位。
    这是**非对称**的,且正是我们想要的非对称:
      · 真论文题上,英文块分数远超中文块(主库压根没有相关内容),margin 拦不住它;
      · 中文业务题上的险胜才会被拦下 —— 用"赢够才换"替代"赢一点就换"。
    margin=0 或未配置 → 不启用(逐字节不变)。
    """
    if margin is None:
        try:
            margin = float(os.environ.get("RAG_EN_TOP1_MARGIN", "0") or 0)
        except ValueError:
            margin = 0.0
    if margin <= 0 or len(docs) < 2 or not scores:
        return docs, metas, dists, scores
    from rag_lang import is_quota_lang

    def _is_en(m):
        m = m or {}
        return is_quota_lang(m.get("document_language") or "") or m.get("source_type") == "paper"

    if not _is_en(metas[0]):
        return docs, metas, dists, scores
    zh = next((i for i in range(1, len(docs)) if not _is_en(metas[i])), None)
    if zh is None or (float(scores[0]) - float(scores[zh])) >= margin:
        return docs, metas, dists, scores      # 没有中文候选,或英文赢得够多 → 不干预
    idx = list(range(len(docs)))
    idx.insert(0, idx.pop(zh))
    return ([docs[i] for i in idx], [metas[i] for i in idx],
            [dists[i] for i in idx] if dists else dists,
            [scores[i] for i in idx] if scores else scores)


def reserve_slot(docs: list, metas: list, dists: list, scores: list,
                 champion: str | None, slot: int | None = None) -> tuple:
    """保底槽位:把英文通道的冠军候选提到第 ``slot`` 位(若它还没进前 slot)。

    **为什么是这个形状而不是等权 RRF**(B3/B4 实测崩盘后的重新设计):
    等权融合会把整个初排顺序的**语言偏置**一起继承——初排前 20 名按构造全是中文
    (多通道 rank1 的 RRF 分完全相同,稳定排序恒定判中文胜),于是融合必然把中文拖回顶部,
    实测论文向全域 0%、中文守卫还从 70% 掉到 50%。

    但初排里真正有价值的信息不是那个全局顺序,而是"**配额通道自己的第 1 名**常常是对的"
    ——消融里 R@3=84% 正是它落在全局第 2 位的结果。所以只保护这一个候选的槽位,
    而不是让有偏的全局初排参与打分。这与配额本身是同一条思路:保底,不顶替
    (放到第 3 位而非第 1 位,精排的 Top1 判断完全不动)。

    代价明确且有界:原本排第 slot 的候选退后一位。中文业务题上这意味着 R@3 可能受损,
    必须用中文大尺子实测,不能只看跨语集。
    """
    if not champion or not docs:
        return docs, metas, dists, scores
    if slot is None:
        try:
            slot = int(os.environ.get("RAG_RESERVE_SLOT", "3"))
        except ValueError:
            slot = 3
    if slot <= 0:
        return docs, metas, dists, scores
    try:
        cur = docs.index(champion)
    except ValueError:
        return docs, metas, dists, scores          # 冠军没进最终池 → 无从保护
    if cur < slot:
        return docs, metas, dists, scores          # 已在前 slot 位,无需干预
    tgt = slot - 1
    idx = list(range(len(docs)))
    idx.insert(tgt, idx.pop(cur))
    return ([docs[i] for i in idx], [metas[i] for i in idx],
            [dists[i] for i in idx] if dists else dists,
            [scores[i] for i in idx] if scores else scores)
