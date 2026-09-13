# -*- coding: utf-8 -*-
"""rag_profiles.py — 检索 Profile 声明与指纹(指导文档 §4.1 / §8)。

**为什么要两个 Profile**(指导 §8):不该继续强迫同一套配置同时做到
"最低中文干扰"与"最强论文检索"——这两个目标在跨语精排的分数校准上直接冲突,
逐轮实测里每次都是按下葫芦浮起瓢。拆开后各自标定,冲突就消失了。

**这不是旧式硬路由**(指导 §8.2 特别强调):Default Profile **仍然保留英文候选**
(配额永不为 0),Paper Profile 只是把别名强度与配额调高。路由是配额调节器,不是候选开关
——这正是本系列开头那个"论文域完全不可达"事故的教训。

Profile 由**显式上下文**选择(用户在论文页 / 绑定了某篇 paper / 显式选深度检索),
不做隐式意图猜测。
"""
from __future__ import annotations

import hashlib
import json
import os

# 每个 Profile = 一组 env 覆盖。空 dict = 用代码默认(即全部实验特性关闭)。
PROFILES: dict = {
    # 中文生产基线:所有跨语特性关闭。用作对照与回滚终点。
    "b0_chinese_baseline": {},

    # 默认混合 Profile(第十一轮改版):配额 + 分工精排(EN 模型桥)。中文全部保护项
    # 实测零代价(同分布/zh_final/held-out/拒答逐题不动),论文域排序可达(34/38 per 50)。
    # 不开英文证据门:默认路径宁拒不编,英文命中走 fallback 标注而非 grounded。
    "default_mixed": {
        "RAG_EN_QUOTA": "1",
        "RAG_EN_QUOTA_K": "5",
        "RAG_RERANK_EN_ONNX_DIR": "~/.cache/modelscope/hub/models/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    },

    # 论文质量 Profile(第十二轮改版):分工桥 + 英文证据门 τ=0.80(显式论文场景用)。
    # τ 标定:正确簇 0.805+/错误簇 ≤0.735/负样本上界 0.401,新放行 19 正确 0 错误。
    # 旧别名方案(质量上限 37~42 但 held-out −1~2)保留为 env 手动组合,不再作为 Profile。
    "paper_quality": {
        "RAG_EN_QUOTA": "1",
        "RAG_EN_QUOTA_K": "5",
        "RAG_RERANK_EN_ONNX_DIR": "~/.cache/modelscope/hub/models/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "RAG_EN_GATE_MIN": "0.80",
    },
}

# Profile 之外仍可能影响检索的旋钮;算指纹时一并纳入,避免"同名 Profile 实际不同行为"。
_EXTRA_FOR_HASH = ("RAG_RERANK_MODEL", "RAG_RERANK_MAX_SEQ", "RAG_RECALL_N",
                   "RAG_QUOTA_COLLECTION", "RAG_QUERY_TRANSLATE",
                   "RAG_ALIAS_SCOPE", "RAG_ALIAS_LIFT_GATE",
                   "RAG_EN_TOP1_MARGIN", "RAG_RESERVE_SLOT", "RAG_RANK_FUSION",
                   "RAG_RERANK_EN_ONNX_DIR", "RAG_EN_GATE_MIN",
                   # 结构块判据会改变门的判据块,同一 Profile 开与关是两套行为;
                   # 不纳入指纹就正好复现这段 docstring 里那个"指标没绑配置"的事故。
                   "RAG_STRUCTURAL_EVIDENCE_MAX")


def profile_env(name: str) -> dict:
    if name not in PROFILES:
        raise KeyError(f"未知 Profile:{name};可选 {sorted(PROFILES)}")
    return dict(PROFILES[name])


def retrieval_profile_hash(name: str) -> str:
    """把「Profile + 别名版本 + 精排配置 + 索引指纹」钉成一个短哈希(指导 §4.1)。

    用途:任何最终指标都必须绑定它。配置一变哈希就变,旧指标自动标记 stale——
    本系列开头那个"文档挂着 56% 而生产实际是 0%"的事故,根因就是指标没绑配置。
    """
    payload = {"profile": name, "env": dict(sorted(profile_env(name).items()))}
    try:
        from rag_alias import ALIAS_VERSION
        from rag_alias_health import prompt_hash
        payload["alias_version"] = ALIAS_VERSION
        payload["alias_prompt_hash"] = prompt_hash()
    except Exception:
        pass
    try:
        from rag_tools import index_fingerprint
        fp = index_fingerprint()
        payload["index"] = {k: fp.get(k) for k in
                            ("collection", "collection_count", "embedding_model",
                             "embedding_dimensions", "rerank_model")}
    except Exception:
        pass
    payload["extra_env"] = {k: os.environ.get(k, "") for k in _EXTRA_FOR_HASH}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def apply_profile(name: str) -> dict:
    """把 Profile 写进当前进程 env,返回其指纹。**仅供评测/脚本使用**——
    生产接线应显式传配置而非改全局 env(指导 §11.1:禁止核心检索逻辑读模块级 env 常量)。"""
    for k, v in profile_env(name).items():
        os.environ[k] = os.path.expanduser(v) if v.startswith("~") else v
    return {"profile": name, "retrieval_profile_hash": retrieval_profile_hash(name)}


def describe(name: str) -> str:
    env = profile_env(name)
    kv = ", ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(全部实验特性关闭)"
    return f"{name}  hash={retrieval_profile_hash(name)}\n    {kv}"


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for n in PROFILES:
        print(describe(n))
