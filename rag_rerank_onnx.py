# -*- coding: utf-8 -*-
"""rag_rerank_onnx.py — ONNX 交叉编码器包装(轻量多语精排探路,2026-08-24)。

动机:两条战线的共同卡点是"够强的精排跑不进 CPU 延迟预算"——
v2-m3 质量达标但 40~90s/题;torch 动态 INT8 在本机 ARM 构建无量化引擎(NoQEngine)。
mmarco-mMiniLMv2-L12-H384 在 ModelScope 有**现成的 ARM64 INT8 ONNX**(113MB),
onnxruntime 直接吃,零导出、零训练——是"质量接近多语、延迟接近 base"假设的最便宜试金石。

接口对齐 sentence_transformers.CrossEncoder 的最小面:``predict(pairs) -> [float]``、
``max_seq_length`` 属性。分数是**原始 logit**(排序用途下与概率单调等价);
注意它与 bge-reranker-base 的 sigmoid 分**不同尺度**——证据门控阈值(0.85/0.95)
不可直接沿用,启用前须按 Gate Profile 重标定(指导 §12.4)。默认不接入生产链。
"""
from __future__ import annotations

import os

_SESS_CACHE: dict = {}


class OnnxCrossEncoder:
    """onnxruntime 版交叉编码器。线程数默认 4(8GB MacBook Air 上留余量给系统)。"""

    def __init__(self, model_dir: str, onnx_file: str = "onnx/model_qint8_arm64.onnx",
                 threads: int | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.model_dir = model_dir
        self.max_seq_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        path = os.path.join(model_dir, onnx_file)
        key = (path, threads)
        if key not in _SESS_CACHE:
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = threads or int(
                os.environ.get("OFFERCLAW_ONNX_THREADS", "4"))
            opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _SESS_CACHE[key] = ort.InferenceSession(
                path, sess_options=opt, providers=["CPUExecutionProvider"])
        self.session = _SESS_CACHE[key]
        self._input_names = {i.name for i in self.session.get_inputs()}

    def predict(self, pairs: list, batch_size: int = 8) -> list:
        """对 (query, doc) 对打分,返回原始 logit 列表。分批防峰值内存。"""
        import numpy as np
        out: list = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            enc = self.tokenizer([p[0] for p in chunk], [p[1] for p in chunk],
                                 padding=True, truncation=True,
                                 max_length=self.max_seq_length, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items()
                    if k in self._input_names}
            logits = self.session.run(None, feed)[0]
            out.extend(float(x) for x in logits.reshape(-1))
        return out
