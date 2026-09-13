#!/bin/bash
# OfferClaw · 一键启动脚本
# 双击此文件即可启动服务，浏览器会自动打开工作台

cd "$(dirname "$0")"

echo "========================================="
echo "  OfferClaw · 求职作战工作台"
echo "========================================="
echo ""

# 1. 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "[!] 虚拟环境不存在，正在创建..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[*] 安装依赖..."
    pip install -r requirements.txt
else
    source .venv/bin/activate
    echo "[OK] 虚拟环境已激活: $(python3 --version)"
fi

# 2. 检查 .env.local
if [ ! -f ".env.local" ]; then
    echo "[!] 警告: .env.local 不存在，LLM 调用可能失败"
else
    echo "[OK] .env.local 已找到"
fi

# 3. 检查端口
if lsof -i :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo ""
    echo "[!] 端口 8000 已被占用，正在关闭旧进程..."
    kill $(lsof -i :8000 -sTCP:LISTEN -t) 2>/dev/null
    sleep 1
fi

echo ""
echo "[*] 启动 OfferClaw API 服务 (端口 8000)..."
echo "[*] 按 Ctrl+C 停止服务"
echo ""

# 4. 延迟打开浏览器
(sleep 2 && open "http://localhost:8000") &

# 5. 启动 uvicorn
exec uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
