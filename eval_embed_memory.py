# -*- coding: utf-8 -*-
"""eval_embed_memory.py — 8GB 主机的 embedding 常驻内存/延迟实测(m3 定夺依据)。

背景:本地 embedding 模型在**检索服务进程常驻**(首查加载后缓存),不是只在
入库时占内存。本脚本在独立子进程逐配置实测,回答"换 bge-m3 之后,日常同时
开浏览器/GPT/Claude 会不会挤爆 8GB"。

每个配置一个子进程(测完即退,互不污染):
  A. bge-base-zh-v1.5 fp32(现状)
  B. bge-m3 fp32
  C. bge-m3 fp16(推荐的 8GB 姿势;CPU fp16 失败则回退 bf16 并注明)
测量:模型加载后进程 RSS / 加载耗时 / 单条查询编码延迟(x5 中位)。
另记:当前系统内存水位、常驻精排(bge-reranker-base)的参考占用。
输出:docs/rag_eval/ingestion/embed_memory_8gb.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(BASE, ".venv", "bin", "python")

_PROBE = r'''
import json, os, sys, time
import psutil
model_name = sys.argv[1]
dtype = sys.argv[2]                      # fp32 | fp16
# 路径由父进程解析好直接传入(探针进程零网络——modelscope 缓存校验在弱网下会阻塞)
proc = psutil.Process()
rss0 = proc.memory_info().rss
t0 = time.time()
from sentence_transformers import SentenceTransformer
kw = {}
note = ""
if dtype == "fp16":
    import torch
    try:
        m = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.float16})
        _ = m.encode(["半精度试算"], normalize_embeddings=True)
    except Exception as e:
        note = f"fp16 失败({str(e)[:60]}),回退 bfloat16"
        m = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.bfloat16})
else:
    m = SentenceTransformer(model_name)
load_s = time.time() - t0
lat = []
for _ in range(5):
    t = time.time()
    m.encode(["如何评估检索增强生成系统的召回质量与拒答边界"], normalize_embeddings=True)
    lat.append((time.time() - t) * 1000)
lat.sort()
import resource
print(json.dumps({
    "model": model_name, "dtype": dtype, "note": note,
    "rss_now_mb": round(proc.memory_info().rss / 1024 / 1024),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024),
    "load_s": round(load_s, 1),
    "encode_ms_p50": round(lat[2], 1),
}))
'''


_MS = os.path.expanduser("~/.cache/modelscope/hub/models")


def _local_path(model: str) -> str:
    p = os.path.join(_MS, model)
    return p if os.path.isdir(p) else model


def probe(model: str, dtype: str) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    try:
        r = subprocess.run([PY, "-c", _PROBE, _local_path(model), dtype],
                           capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return {"model": model, "dtype": dtype, "error": "timeout(>600s)——按不可用记录"}
    if r.returncode != 0:
        return {"model": model, "dtype": dtype, "error": (r.stderr or "")[-200:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    import psutil
    vm = psutil.virtual_memory()
    out = {
        "machine": {"total_gb": round(vm.total / 1024 ** 3, 1),
                    "available_gb_at_test": round(vm.available / 1024 ** 3, 1)},
        "configs": [],
    }
    for model, dtype in [("BAAI/bge-base-zh-v1.5", "fp32"),
                         ("BAAI/bge-m3", "fp16"),
                         ("BAAI/bge-m3", "fp32")]:
        print(f"[probe] {model} {dtype} ...", flush=True)
        out["configs"].append(probe(model, dtype))

    path = os.path.join(BASE, "docs", "rag_eval", "ingestion", "embed_memory_8gb.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[written] {os.path.relpath(path, BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
