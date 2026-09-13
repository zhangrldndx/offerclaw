#!/usr/bin/env bash
set -euo pipefail

# 与稳定 UI 隔离的智能语义路由 Canary。主 UI 继续使用 8000 端口和
# intelligent_shadow；Canary 只供人工验收，避免 8 秒规划 deadline 影响日常使用。
cd "$(dirname "$0")/.."
export RAG_ROUTER_MODE=intelligent
export JD_ANALYZER_MODE=shadow
export RAG_PAPER_PROFILE=paper_quality
export RAG_RERANK_EN_ONNX_DIR="${RAG_RERANK_EN_ONNX_DIR:-$HOME/.cache/modelscope/hub/models/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1}"
export RAG_EN_GATE_MIN="${RAG_EN_GATE_MIN:-0.80}"

exec .venv/bin/python -m uvicorn rag_api:app --host 127.0.0.1 --port "${OFFERCLAW_CANARY_PORT:-8001}"
