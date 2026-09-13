# OfferClaw · 项目一页纸（Project One-Pager）

> 用途：1-2 分钟内让面试官 / 评委 / 自己理解整个项目。  
> 长版见 `README.md` / `docs/architecture.md` / `docs/verification_report.md`。

---

## 1. 一句话
**OfferClaw 是一个面向求职者的长期执行型 AI Agent，覆盖画像 → 岗位匹配 → 缺口识别 → 学习计划 → 投递跟踪 → 复盘的闭环。**

## 2. 目标用户
正在准备 AI 应用 / Agent 方向实习与校招的工程类学生，特别是简历薄弱、缺乏明确方向的"非科班但目标明确"型用户。

## 3. 解决的问题
| 痛点 | OfferClaw 的应对 |
|---|---|
| JD 看了 100 份还不知道哪些适合 | 三档结论：适合 / 暂不 / 中长期可转向 |
| 知道方向但不知道下一步学什么 | 4 周计划生成（按画像缺口排序） |
| 学了忘了，没复盘 | daily_log + 周度 summary |
| 想问项目状态，要翻 12 份 markdown | RAG 知识库问答（同分布 100 题 R@1 89% / R@5 97% · 混合检索 + answerability 判据重排 + 三票共识门） |
| 投递后没有 tracker | applications.md + 状态机 |

## 4. 系统架构
```
┌──────────┐──▶ 显式 RAG 管线 ──▶ ChromaDB / Judge / Gate
│ FastAPI  │
│ + SSE UI │──▶ CareerFlow LangGraph ──▶ Tools
└──────────┘
      └──────── memory.json / logs / plans
```
详见 `docs/architecture.md` 4 张 Mermaid 图。

## 5. 核心模块
- **match_job.py** — 规则 + LLM 双通路岗位匹配（硬否决在规则层，软评估在 LLM 层）
- **rag_gate.py** — 生产 RAG 显式管线：Dense/BM25/RRF → rerank → answerability → 三票共识门
- **rag_graph.py** — LangGraph RAG 实验/演示工作流；生产 `/api/query` 不由它统一编排
- **rag_api.py** — 91 个路由装饰器 / 86 个唯一项目路径（含多源只读问答 / Agent / trace / JD 版本关系）+ JSON 日志 + request_id 中间件
- **mcp_server.py** — MCP Server（Streamable HTTP 传输，手写 JSON-RPC 协议层，工具与 ReAct 共用 REGISTRY）
- **tools.py** — 6 个 Agent 工具：profile / rules / log / plan / match / summary
- **eval_rag_bench.py** — Recall@1/3/5 + MRR + 拒答门槛 + 难题分桶（100 题同分布 + 52 题口语化回归）
- **static/index.html** — 零依赖前端控制台

## 6. 技术栈
Python 3.10+（当前验证 3.12.5）· FastAPI · LangGraph · ChromaDB · OpenAI-compatible LLM（模型可配置）· bge-base-zh / mMiniLM ONNX · pytest · uvicorn · Vanilla JS

## 7. 当前指标
| 指标 | 值 |
|---|---|
| 测试用例 | **1,659 tests passed**（54 skipped，需显式外部/E2E、私有评估包或本地运行资产） |
| RAG 同分布（n=100） | R@1 **89.0%** · R@3 **95.0%** · R@5 **97.0%** · MRR **0.9195**；标题近义偏多，是偏乐观上限 |
| 口语化回归（n=52） | Recall@1 **75.0%** · R@3 **80.8%** · R@5 **84.6%** · MRR **0.7875**；最初为 held-out，但已参与多轮开发，不再称盲集 |
| Final v4 strict chunk | 历史冻结盲测：Candidate **73/80** · R@1 **59/80** · R@3 **70/80** · R@5 **72/80** · nDCG@5 **0.8317** |
| Quality 的实际收益/代价 | R@1 未提高（A/C 均 59/80）；有效作答 **29/80→59/80** · 历史门控 `correct_premise` **12/12** · 真误纳 **1/28** |
| Quality 检索延迟 | 同分布 100 题冷缓存 p50 **13.9s** · p95 **29.8s** |
| 答案质量 v2（当前 HEAD） | n=48 冻结历史开发回归（40 正 + 8 负，非盲/非 organic）；双裁判 **96/96**：忠实度 **96.9%–97.1%** · 完整度 **65.0%–81.3%** · 负例端到端动作合同 **6/8**（pipeline 路由 **8/8**）；4 个错误前提最终答案仅 **2/4** 真正纠错 |
| 答案端到端延迟 | 同一 n=48：p50 **37.6s** · p95 **97.1s**；独立空判据缓存，包含检索与最终生成 |
| 小型产品验收 | 六域 **36/36**：画像、JD、计划、投递、复盘、简历/CareerFlow 各 6 例；tmp 隔离且无外部 LLM |
| 显式论文跨语言（xling 专项集） | grounded 正确 **60%** · 错误放行 **0** · p95 **589ms** |
| FastAPI 路由 | **91 个装饰器 / 86 个唯一项目路径**（含 SSE 流式） |
| Persona 回归 | 3 personas × multi-JD，结论差异化，见 `docs/persona_compare_report.md` |
| 知识库 | **3327 chunks**（多源真实化；候选 JD 不进入产品 RAG） |
| 工程自检 | doctor **12 OK**（auto-load `.env.local`）· verify_pipeline 6/6 · verify_docs all green |
| 计划/简历生成首字延迟 | < 2s（SSE 流式；不是 Quality RAG 延迟） |
| Playwright SPA | 支持字节/阿里/腾讯等 SPA 招聘页自动渲染 |

> 全部指标的现场命令输出固化在 [`docs/verification_report.md`](verification_report.md)。

## 8. Demo 链路（≤2 分钟）
1. `python -m uvicorn rag_api:app` → 浏览器打开 `http://127.0.0.1:8000/ui`
2. 顶部今日建议横条 → 基于 `career_agent.py` + `applications` / `daily_log` 主动生成
3. 点 **系统健康** → 看主库 `3327 chunks` 与 doctor `12 OK`
4. 粘贴合成 JD → 点 **运行匹配** → 输出三档结论 + 缺口清单
5. 输入 "OfferClaw 主方向是什么？" → 证据问答（当前 Quality 实测 p50 37.6s / p95 97.1s；面试演示前预热，并明确不承诺 RAG 首字 <2s）
6. 点卡片⑥ **针对 JD 生成简历段** → SSE 流式输出定制项目描述

## 9. 关键技术难点 / 取舍
- **规则 vs LLM**：硬否决用规则（确定性 + 可测试），软评估给 LLM。— 见 Story 2
- **RAG chunk 切法**：从 token-based 切到 markdown-header-based，Recall@5 +25 个百分点。— 见 Story 4
- **POST + SSE**：浏览器 EventSource 不支持 POST，前端被迫用 fetch + ReadableStream。— 见 Story 6
- **LangGraph BaseMessage 序列化**：原生 `dict()` 失败，加 `_msg_to_dict()` 适配层。
- **状态契约**：用户层 vs 系统层 vs 运行时分层，详见 `DATA_CONTRACT.md`。

## 10. 当前不足与下一步
| 不足 | V2 计划 |
|---|---|
| Quality RAG 受多次模型调用延迟影响 | 当前答案链 n=48 p95 97.1s；先修 `correct_premise` 动作传递，再提供 Fast/Quality 明确演示档并压低判据调用数 |
| ~~没有 Reranker~~ | ✅ 已加本地 **bge-reranker 交叉编码器**两阶段精排（难题 R@1 82→91） |
| Markdown 事实源需要持续治理 | 保持本地可审计契约；增加 ID、版本、哈希和并发校验，不迁移 SQLite |
| ~~没有 CI~~ / 没有 Docker | ✅ GitHub Actions **CI 门禁**已加（`.github/workflows/ci.yml`，eval_rag_bench --fail-under）；Docker 待做 |
| JD 推荐仍是半自动 | Query Builder + 排序层（见 job_discovery.py 扩展计划） |
| 简历导出 | 支持 Markdown → PDF / Word |
| 没有自动投递 | **保持不做**（见 `docs/ethical_use.md`） |

---

## 11. Portfolio Signal（项目作为求职作品集的信号强度）

> 参照 Career-Ops `modes/project.md` 的 6 维评估标准自评，用于面试和简历附录引用。

| 维度 | 评分 | 说明 |
|---|---|---|
| **Target-role signal** | ⭐⭐⭐⭐⭐ | 直接面向 AI 应用 / Agent / RAG / FastAPI / LangGraph 实习岗，技术栈与岗位 JD 高度对齐 |
| **Uniqueness** | ⭐⭐⭐⭐ | 把"求职过程"本身做成 Agent 系统（画像→匹配→缺口→规划→执行→复盘），区别于纯 RAG demo 或纯简历工具 |
| **Demo-ability** | ⭐⭐⭐⭐⭐ | 本地 `/ui` 8 卡片求职工作台 + 顶部 RAG 问答条 + 今日建议横条 + CareerFlow 流程条，2 分钟可完整演示一遍 |
| **Metrics** | ⭐⭐⭐⭐ | 同分布 R@1/R@5=89%/97% · 口语回归 Recall@1=75% · Final v4 strict R@1=59/80、有效作答=59/80 · 答案忠实度 96.9%–97.1%、完整度 65.0%–81.3% · 1,659 tests · 产品验收 36/36；答案链 p95 97.1s，诚实扣 1 星 |
| **MVP time** | ⭐⭐⭐⭐ | 6-8 周从 Agent Demo → V2 控制台，节奏可追溯（PROJECT_STATUS.md 9 批次推进记录） |
| **STAR potential** | ⭐⭐⭐⭐⭐ | 支持本地私有 STAR+R 故事库绑定技术主题、可回答问题与相关文件；故事内容和数量不进入公开仓库 |

> **总信号**：5 / 6 维 ≥ 4 星 —— 项目已达到"可直接放进简历首屏 + 面试主项目"水平。
> **可演示路径**：`docs/archive/demo_script.md`（已归档）· **可复现路径**：`docs/verification_report.md` · 故事库仅保存在本地私有运行文件

---

GitHub: https://github.com/zhangrldndx/offerclaw
