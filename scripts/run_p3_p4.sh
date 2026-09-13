#!/usr/bin/env bash
# =====================================================================
# run_p3_p4.sh — 在(更强的)新机器上一键跑 P4 延迟-精度扫描 + P3 规模压测
#
# 用法(在解包后的项目根目录执行):
#   bash scripts/run_p3_p4.sh
#
# 前置:python3.10+;chroma_db/ 已就位(bundle 附带);联网(首跑自动从
#      ModelScope 下 bge-base-zh-v1.5 与 bge-reranker-base,约 1.2GB)。
# 不需要任何 API key:门控判定(in_kb)纯靠检索证据,LLM 只影响答案文本。
#
# 产出:results_p3p4_<主机>_<日期>.tar.gz —— 拷回原机器交给 Claude 即可。
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "══════════════════ 环境准备 ══════════════════"
PY=python3
$PY --version
if [ ! -d .venv ]; then $PY -m venv .venv; fi
source .venv/bin/activate
pip install -q -r requirements.txt
pip install -q matplotlib psutil   # 实验/绘图依赖(不进生产 requirements)

echo "══════════════════ 前置自检 ══════════════════"
python - << 'EOF'
import chromadb, torch
from rag_tools import get_collection_name
n = chromadb.PersistentClient(path='chroma_db').get_collection(get_collection_name()).count()
assert n > 3000, f"chroma_db 不完整(count={n}),请确认 chroma_db.tar.gz 已解到项目根"
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 向量库 {n} chunks · torch 自动设备 = {dev}")
EOF

STAMP="$(hostname -s)_$(date +%m%d_%H%M)"
OUT="docs/rag_eval/round9"
mkdir -p "$OUT"

echo "══════════════════ P4 延迟-精度扫描(先跑,较快) ══════════════════"
# 扫 4 档精排候选数;每档跑 bench100+heldout52 的 R@1/MRR + heldout 逐题延迟 p50/p95
RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 PYTHONIOENCODING=utf-8 \
  python -u eval_latency_tradeoff.py --settings 5 10 20 40 \
  --out "$OUT/latency_tradeoff_${STAMP}.json" 2>&1 | tee "/tmp/p4_${STAMP}.log"

echo "══════════════════ P3 规模压测(重,末尾自动清理临时集合) ══════════════════"
PYTHONIOENCODING=utf-8 python -u eval_scale_test.py --scales 10000 50000 100000 \
  --out "$OUT/scale_test_${STAMP}.json" 2>&1 | tee "/tmp/p3_${STAMP}.log"

echo "══════════════════ 打包结果 ══════════════════"
RES="results_p3p4_${STAMP}.tar.gz"
tar -czf "$RES" \
  "$OUT/latency_tradeoff_${STAMP}.json" "$OUT/scale_test_${STAMP}.json" \
  "/tmp/p4_${STAMP}.log" "/tmp/p3_${STAMP}.log" 2>/dev/null || \
tar -czf "$RES" "$OUT"/latency_tradeoff_${STAMP}.json "$OUT"/scale_test_${STAMP}.json
python - << EOF
import platform, torch, json
info = {"host": "$STAMP", "machine": platform.platform(),
        "cpu": platform.processor(),
        "torch_device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"}
print(json.dumps(info, ensure_ascii=False, indent=1))
open("machine_info_${STAMP}.json", "w").write(json.dumps(info, ensure_ascii=False))
EOF
tar -rf "${RES%.gz}" "machine_info_${STAMP}.json" 2>/dev/null || true

echo ""
echo "✅ 全部完成。把这个文件拷回原机器给 Claude:"
echo "   $(pwd)/$RES"
echo "   (设备信息在 machine_info_${STAMP}.json,延迟数字必须带设备标注才有意义)"
