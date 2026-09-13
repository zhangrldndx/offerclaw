# -*- coding: utf-8 -*-
"""rag_rerank.py — 本地 bge-reranker 重排序（RAG 优化 Round 1）。

为什么需要 rerank：向量检索算的是"语义相似度"，但语义最相似 ≠ 最相关。
向量是双塔（query/doc 各自编码后算距离），信息有损；reranker 是**交叉编码器
（cross-encoder）**，把 query+doc 拼在一起进模型，能做细粒度的相关性判断。
做法：向量粗召回 topN（如 20）→ reranker 对每个 (query, doc) 打分 → 精排取 topk。
价值主要体现在 Recall@1 / MRR（把正确文档从第 2-3 名提到第 1 名）。

设计：
- 与本地 bge embedding 一脉相承，用 sentence-transformers 的 CrossEncoder；
- ModelScope 优先下载（国内快）+ 进程内缓存，首次加载慢、之后秒回；
- env 开关 RAG_RERANK（默认 1 开启）+ RAG_RERANK_MODEL（默认 BAAI/bge-reranker-base）；
- 任何失败（模型缺失/打分异常）静默降级为"不重排"，绝不阻断问答。
"""

from __future__ import annotations

import os
import threading

RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
_RERANKER = None
_LOAD_FAILED = False


def rerank_enabled() -> bool:
    return os.environ.get("RAG_RERANK", "1").strip().lower() in ("1", "true", "yes", "on")


_CE_CACHE: dict = {}        # (model, device) -> CrossEncoder | None(加载失败,不再重试)
_LOAD_LOCK = threading.Lock()  # 防并发首问重复下载/加载同一 1GB 模型
_PREDICT_LOCK = threading.Lock()  # 多源路由可并行召回，但同一 PyTorch 模型推理串行更安全


def _load_reranker(model_name: str | None = None):
    """懒加载 CrossEncoder（ModelScope 优先，HF 兜底）。失败置标志位，后续直接降级。

    缓存按 **(模型, 设备)** 分键(2026-08-18):原实现是单个全局 `_RERANKER`,不认模型名——
    进程内改 `RAG_RERANK_MODEL` 会静默沿用旧模型,多语精排 A/B 会得出"两臂一模一样"的
    假结论(与 rag_tools._load_local_model 那次零缓存是同一类缺陷的镜像)。
    """
    global _RERANKER, _LOAD_FAILED
    model = model_name or os.environ.get("RAG_RERANK_MODEL", RERANK_MODEL)
    # ONNX 后端(2026-08-24 探针轮,默认关):RAG_RERANK_ONNX_DIR 指向含
    # onnx/model_qint8_arm64.onnx 的模型目录时,改走 onnxruntime INT8——
    # mMiniLM 实测 36ms/对(base torch 100~126ms)。分数是**原始 logit**,
    # 与 base 的 sigmoid 不同尺度,证据门控阈值不可沿用(启用前须重标定)。
    _onnx_dir = (os.environ.get("RAG_RERANK_ONNX_DIR", "") or "").strip()
    if _onnx_dir:
        _okey = ("onnx", _onnx_dir)
        if _okey not in _CE_CACHE:
            with _LOAD_LOCK:
                if _okey not in _CE_CACHE:
                    try:
                        from rag_rerank_onnx import OnnxCrossEncoder
                        _CE_CACHE[_okey] = OnnxCrossEncoder(_onnx_dir)
                    except Exception as e:
                        import sys as _sys
                        print(f"[rerank] ONNX 加载失败,回退 torch:{e}", file=_sys.stderr)
                        _CE_CACHE[_okey] = None
        if _CE_CACHE[_okey] is not None:
            return _CE_CACHE[_okey]
    _dev_key = os.environ.get("OFFERCLAW_TORCH_DEVICE") or None
    _key = (model, _dev_key)
    if _key in _CE_CACHE:
        _RERANKER = _CE_CACHE[_key]
        _LOAD_FAILED = _RERANKER is None
        return _RERANKER
    with _LOAD_LOCK:
        # 双重检查：等待锁期间另一请求可能已加载完成。
        if _key in _CE_CACHE:
            _RERANKER = _CE_CACHE[_key]
            _LOAD_FAILED = _RERANKER is None
            return _RERANKER
        import contextlib
        import sys as _sys
        from sentence_transformers import CrossEncoder
        # 与 rag_tools._load_local_model 同款设备旋钮：OFFERCLAW_TORCH_DEVICE=cpu 可
        # 避开被并发事故搞坏的 Metal/MPS（事故细节见 rag_tools 注释）。默认自动选。
        _dev = os.environ.get("OFFERCLAW_TORCH_DEVICE") or None
        try:
            with contextlib.redirect_stdout(_sys.stderr):  # 防止下载日志污染 stdout JSON
                if os.path.isdir(model):
                    _RERANKER = CrossEncoder(model, device=_dev)
                else:
                    try:
                        from modelscope import snapshot_download
                        # 只下权重 + 配置 + 分词器，跳过 1G onnx（CrossEncoder 用不到）。
                        # safetensors 必须在列(2026-08-18):bge-reranker-v2-m3 只发 safetensors,
                        # 漏了它会"下载成功"却拿到没有权重的空壳(实测只有 22MB)。
                        local = snapshot_download(model, allow_patterns=[
                            "*.json", "*.txt", "*.model", "pytorch_model.bin", "*.safetensors",
                            "tokenizer*", "sentencepiece*", "vocab*", "special_tokens*"])
                        _RERANKER = CrossEncoder(local, device=_dev)
                    except Exception:
                        _RERANKER = CrossEncoder(model, device=_dev)  # 退 HuggingFace
        except Exception as e:
            _LOAD_FAILED = True
            _CE_CACHE[_key] = None
            print(f"[rerank] 加载失败，降级为不重排：{e}", file=_sys.stderr)
            return None
        _CE_CACHE[_key] = _RERANKER
        return _RERANKER


def rerank(query: str, docs: list, metas: list, dists: list, top_k: int,
           synth_map: dict | None = None,
           alt_query: str | None = None, alt_idx: list | None = None,
           alias_map: dict | None = None, alias_discount: float = 0.0,
           alias_lift_gate: float = 0.0, diag: dict | None = None,
           model_name: str | None = None,
           pair_docs: list[str] | None = None,
           runtime_diag: dict | None = None,
           en_bridge_idx: list | None = None):
    """对 (query, doc) 交叉打分并精排，返回前 top_k 的 (docs, metas, dists, scores)。

    dists 是各 doc 的**原始向量距离**，随 doc 一起重排带走（门控仍按最小向量距离判定，
    不受 rerank 影响）。reranker 不可用或 docs 为空时原样返回（截断到 top_k）。

    ``synth_map``[Round 10 语体桥,默认 None=行为不变]:{doc_key: 该父块最贴近本 query 的
    doc2query 合成问题}。提供时,命中的父块得分取 **max(query↔父块原文, query↔合成问题)**——
    Round 9 双路 A/B 证明瓶颈在交叉编码器对"口语 query↔书面原文"的打分,合成问题与 query
    同语体,给 reranker 递一座桥。额外开销 = 命中父块数条 pair(并入同一次 predict 批)。
    采纳标准(跑 A/B 前预登记):heldout52 R@1≥+4pp 且 bench100 回退≤1pp 且双负样本无净损。

    ``alt_query`` + ``alt_idx``[跨语桥,2026-08-18]:给 ``alt_idx`` 指定的候选**额外**用
    ``alt_query``(query 的英文翻译)打一次分,取 max。与 synth_map 同一机制,桥的对象
    从"语体"换成"语言":实证显示 cross-encoder 在"中文 query↔英文 passage"上打分偏低
    (论文向卡在 60%),而同一模型在"英问英"子域是 100%——所以只对英文候选补一次
    英文 query 的评分,把已知能跑满分的条件创造出来。取 max 而非替换:翻译失准时
    退化为原分,不会因为一次坏翻译把候选打死。

    ``alias_map`` + ``alias_discount``(δ) + ``alias_lift_gate``(τ)[概念别名桥,校准版]:
    与上面两座桥同机制,但**别名分要付代价才算数**。此前无条件 `max(原分, 别名分)` 让
    别名分零成本参与竞争,轻微虚高就能让英文候选抢走中文题的 Top1(held-out 回退 2~3 题)。
    校准后:
        lift = 别名分 - 原分;  lift < τ → 不采信别名分(视为噪声级抬升)
        否则  最终分 = max(原分, 别名分 - δ)
    δ 是**连续可调**的桥分折扣,粒度远细于"Top1 换不换"那种全有全无的名次拦截
    ——实测那种拦截是坏交易(8 道论文题换 1 道中文题)。
    ``diag`` 给出时按 doc 下标回填 {orig, alias, lift, final},供 False Winner 审计。
    """
    requested_model = model_name or os.environ.get("RAG_RERANK_MODEL", RERANK_MODEL)
    if runtime_diag is not None:
        runtime_diag.update({
            "requested": requested_model,
            "actual": "",
            "status": "not_run",
            "scored_count": 0,
        })
    if not docs:
        if runtime_diag is not None:
            runtime_diag["status"] = "skipped_no_candidates"
        return docs, metas, dists, []
    if not rerank_enabled():
        if runtime_diag is not None:
            runtime_diag["status"] = "disabled"
        return docs[:top_k], metas[:top_k], dists[:top_k], []
    model = (_load_reranker(model_name) if model_name else _load_reranker())
    if model is None:
        if runtime_diag is not None:
            runtime_diag["status"] = "load_failed"
        return docs[:top_k], metas[:top_k], dists[:top_k], []
    if runtime_diag is not None:
        runtime_diag["actual"] = requested_model
    # 序列长度旋钮(2026-08-19 剖析驱动):交叉编码器耗时随序列长度**超线性**增长,
    # 实测 25 对 512→256 是 2498ms→1182ms(-53%)。主库真实 token 中位 283、p95 490,
    # 截断确有代价,故按质量实测定档而非拍脑袋(见 docs/rag_eval/quota/REPORT.md)。
    # 每次调用赋值而非加载期冻结:同进程扫描换挡才生效(本项目在 env 冻结上栽过三次)。
    _ms = os.environ.get("RAG_RERANK_MAX_SEQ", "").strip()
    if _ms.isdigit():
        model.max_seq_length = int(_ms)
    try:
        import sys as _sys
        import contextlib
        scoring_docs = pair_docs if pair_docs is not None else docs
        if len(scoring_docs) != len(docs):
            raise ValueError("pair_docs must align one-to-one with docs")
        pairs = [[query, d] for d in scoring_docs]
        bridge_idx: list = []   # (docs 下标, pairs 下标) —— 合成问题/英文 query 附加对
        if synth_map:
            from rag_doc2query import doc_key
            for i, (d, m) in enumerate(zip(docs, metas)):
                sq = synth_map.get(doc_key(d, m))
                if sq:
                    bridge_idx.append((i, len(pairs)))
                    pairs.append([query, sq])
        if alt_query and alt_idx:      # 跨语桥:只给英文候选补 (英文 query, 该 doc) 一对
            for i in alt_idx:
                if 0 <= i < len(docs):
                    bridge_idx.append((i, len(pairs)))
                    pairs.append([alt_query, scoring_docs[i]])
        # 分工精排/英文模型桥(2026-08-24 探针轮,默认关):base 给全部候选打分(中文精度
        # 逐字节不动),多语 mMiniLM(ONNX INT8,36ms/对)只给**英文候选**再打一分,
        # 两边取 sigmoid 概率后 max。30 题探针:xling_para R@1 0→2/R@3 0→3,
        # 中文守卫+陷阱题逐题零扰动、英文抢位 0。旋钮 RAG_RERANK_EN_ONNX_DIR(空=关)。
        en_bridge_model = None
        if en_bridge_idx:
            # `.env.local` 由项目自己的 loader 读取，shell 不会替它展开 `~`。
            # 在唯一模型加载边界统一处理，避免评测脚本可用、真实服务静默跳桥。
            _en_dir = os.path.expanduser(
                (os.environ.get("RAG_RERANK_EN_ONNX_DIR", "") or "").strip()
            )
            if _en_dir:
                _ekey = ("en_bridge", _en_dir)
                if _ekey not in _CE_CACHE:      # 实例级缓存:tokenizer 重载曾拖慢每查询 ~1.5s
                    try:
                        from rag_rerank_onnx import OnnxCrossEncoder
                        _CE_CACHE[_ekey] = OnnxCrossEncoder(_en_dir)
                    except Exception as _e:
                        print(f"[rerank] 英文桥模型加载失败,跳过:{_e}", file=_sys.stderr)
                        _CE_CACHE[_ekey] = None
                en_bridge_model = _CE_CACHE[_ekey]
        alias_idx: list = []           # 概念别名桥:单独记账,因为它要过折扣与 lift 门
        if alias_map:
            from rag_doc2query import doc_key
            for i, (d, m) in enumerate(zip(docs, metas)):
                at = alias_map.get(doc_key(d, m))
                if at:
                    alias_idx.append((i, len(pairs)))
                    pairs.append([query, at])
        with _PREDICT_LOCK:
            with contextlib.redirect_stdout(_sys.stderr):
                scores = model.predict(pairs)
        scores = [float(s) for s in scores]
        if len(scores) < len(docs):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(docs)} candidates"
            )
        if bridge_idx or alias_idx:
            orig = list(scores[:len(docs)])      # lift 要对**原分**算,不能被别的桥污染
            for di, pi in bridge_idx:
                scores[di] = max(scores[di], scores[pi])   # 语体/跨语桥:两侧取 max
            for di, pi in alias_idx:
                a = scores[pi]
                lift = a - orig[di]
                use = lift >= alias_lift_gate
                if use:
                    scores[di] = max(scores[di], a - alias_discount)
                if diag is not None:
                    diag[di] = {"orig": round(orig[di], 4), "alias": round(a, 4),
                                "lift": round(lift, 4), "applied": bool(use),
                                "final": round(scores[di], 4)}
            scores = scores[:len(docs)]
        if en_bridge_model is not None and en_bridge_idx:
            import math as _math
            _valid = [i for i in en_bridge_idx if 0 <= i < len(docs)]
            if _valid:
                _ms = en_bridge_model.predict([[query, docs[i]] for i in _valid])
                for _j, _i in enumerate(_valid):
                    _p = 1.0 / (1.0 + _math.exp(-float(_ms[_j])))   # logit→sigmoid 同尺度
                    scores[_i] = max(scores[_i], _p)
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        rd = [docs[i] for i in order][:top_k]
        rm = [metas[i] for i in order][:top_k]
        rs = [dists[i] for i in order][:top_k] if dists else []
        rscore = [round(float(scores[i]), 4) for i in order][:top_k]
        if runtime_diag is not None:
            runtime_diag.update({
                "status": "ok",
                "scored_count": len(docs),
            })
        return rd, rm, rs, rscore
    except Exception as e:
        import sys as _sys
        print(f"[rerank] 打分异常，降级为不重排：{e}", file=_sys.stderr)
        if runtime_diag is not None:
            runtime_diag.update({"status": "score_failed", "scored_count": 0})
        return docs[:top_k], metas[:top_k], dists[:top_k], []
