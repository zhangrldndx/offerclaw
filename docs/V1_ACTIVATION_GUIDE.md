# OfferClaw V1 稳定运行与增强功能启用指南

> **历史文档（已失效）**：本文记录 2026-08-25 的灰度方案。当前顶部问答已固定使用
> SemanticQueryPlan v4，`legacy` / `intelligent_shadow` 路由模式和在线回滚路径已经删除。
> 当前配置与验证方式以 `docs/INTELLIGENT_SEMANTIC_ROUTING.md` 为准。

> 状态：2026-08-25 已按本机生产配置复测。目标是让增强能力按范围投入使用，
> 不把实验开关扩散到普通问答，也不让慢规划器阻塞主 UI。

## 1. 当前推荐配置

| 能力 | 主 UI（8000） | 影响范围 | 结论 |
|---|---|---|---|
| 确定性/兼容路由 | 在线回答路径 | 全部问题 | 稳定基线 |
| 智能语义规划 | `intelligent_shadow` | 后台影子采样 | 不增加用户等待，不接管答案 |
| 跨语言增强 | `paper_quality` | 仅显式 `paper_kb` 路由 | 已投入使用 |
| JD 智能分析 | `shadow` | 后台对照 | 暂不接管正式 JD 快照 |
| 自动论文兜底 | 关闭 | 普通知识库弱证据 | 防止论文劫持中文问题 |

主 UI 直接启动：

```bash
cd /Users/<user>/path/to/offerclaw
.venv/bin/python -m uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

`.env.local` 已配置：

```dotenv
RAG_ROUTER_MODE=intelligent_shadow
JD_ANALYZER_MODE=shadow
RAG_PAPER_PROFILE=paper_quality
RAG_RERANK_EN_ONNX_DIR=~/.cache/modelscope/hub/models/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RAG_EN_GATE_MIN=0.80
```

不要在主 UI 全局设置 `RAG_EN_QUOTA=1`，也不要开启 `RAG_PAPER_ROUTE=1`。
前者会给普通查询增加英文候选，后者会恢复曾造成 held-out R@1 回退 7.7pp 的
自动论文兜底。现在的增强由 `rag_paper_route.retrieve_papers_explicit()` 范围化执行。

## 2. 如何触发跨语言增强

问题必须明确要求论文、研究或英文文献，例如：

- “ReAct 论文如何把推理和行动交替起来？”
- “根据论文解释 MemGPT 的分层记忆机制。”
- “有没有英文研究讨论长上下文利用率？”

QueryPlan 选择 `paper_kb.search` 后，系统才会：

1. 在 `kb_paper_bge_v1` 取 5 个英文候选；
2. 用原 bge reranker 保留基线排序；
3. 仅对这 5 个英文候选运行 113MB mMiniLM ONNX 分工桥；
4. 只有最终 Top1 分数不低于 0.80 才作为 grounded 证据回答。

本机端到端复测 xling50：grounded 正确 `30/50`、错误放行 `0`、
p50 `309ms`、p95 `589ms`。普通投递、画像、项目、复盘和中文资料问题不会加载
英文桥，因此没有新增候选数、模型调用或延迟。

回滚只需：

```dotenv
RAG_PAPER_PROFILE=baseline
```

重启服务后生效。

## 3. 如何使用智能语义路由

当前 GPT 规划在真实复杂问题上仍出现约 8 秒 deadline 超时，因此不能在“不影响性能”
的条件下直接全量接管主 UI。主 UI 的 `intelligent_shadow` 已经在使用智能规划器收集
差异，但后台任务不会阻塞旧路径答案。

需要检查智能路由真实回答时，启动隔离 Canary：

```bash
./scripts/start_router_canary.sh
```

然后访问 `http://127.0.0.1:8001/ui`。8000 的日常 UI 不受影响。Canary 中重点复测：

- OfferClaw 功能使用指导（Guide）；
- 个人投递/项目/复盘索引（Recall）；
- 个人事实 + JD + 通用资料的复合建议（Advise）；
- 缺数据时的澄清与安全拒答。

只有同时满足以下条件，才把 8000 的配置改成 `RAG_ROUTER_MODE=intelligent`：

- 真实影子样本 Source-set Macro-F1 不低于 95%；
- 规划 p95 不超过 5 秒、硬上限 8 秒；
- timeout 不超过 1%、fallback 不超过 3%；
- Guide/Recall 误判率低于 1%；
- 个人证据缺失后的错误通用回退为 0。

在此之前，智能语义路由属于“已部署影子验证”，不是“已默认接管”。这是性能边界，
不是功能遗漏。

## 4. 每次发布前的最小检查

```bash
.venv/bin/pytest -q
.venv/bin/python doctor.py
.venv/bin/python verify_pipeline.py
.venv/bin/python verify_docs.py
```

预期口径：pytest 991 passed / 4 skipped、doctor 12 OK、pipeline 6/6、
主库 3327 chunks（候选 JD 不入库）、论文 BGE/E5 collection 各 189 chunks。
