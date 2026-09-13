# OfferClaw · 系统架构图

> 本文档描述 OfferClaw 的整体架构、数据流、模块职责。
> 配合 README + docs/RESUME_PROJECT.md 使用。

## 1. 顶层架构（C4 Context）

```mermaid
graph TB
    User[👤 求职者<br/>示例用户]
    JVS[JVS Claw 平台<br/>对话入口]
    OC[OfferClaw Agent<br/>本仓库]
    Zhipu[智谱 BigModel<br/>GLM-4-Flash + 本地 bge-base-zh-v1.5 嵌入]
    Files[(本地知识库<br/>已批准资料与个人记忆)]
    Chroma[(ChromaDB<br/>派生向量索引)]

    User --对话--> JVS
    JVS --HTTP--> OC
    User --CLI--> OC
    OC <--JWT 鉴权--> Zhipu
    OC --ingest--> Chroma
    OC --query--> Chroma
    Files --rag_ingest.py--> Chroma
```

## 2. 内部模块依赖

```mermaid
graph LR
    subgraph 入口层
        CLI[rag_agent.py<br/>CLI 单次/交互]
        API[rag_api.py<br/>FastAPI + SSE]
        PIPE[pipeline.py<br/>端到端闭环]
        SUM[summary_tool.py<br/>晚间复盘]
    end

    subgraph 编排层
        GRAPH[rag_graph.py<br/>LangGraph 状态机<br/>retrieve→prompt→llm→tools]
    end

    subgraph 业务层
        JDP[jd_parser.py<br/>JD 原文证据化要求]
        ALIGN[semantic_matcher.py<br/>要求→画像证据对齐]
        MATCH[match_job.py<br/>三档匹配规则]
        PLAN[plan_gen.py<br/>4 周计划生成]
    end

    subgraph 工具层
        TOOLS[rag_tools.py<br/>JWT/Embedding/Chunk/LLM]
        LOG[logging_utils.py<br/>JSON 日志+request_id]
        DEMO[agent_demo.py<br/>智谱 SDK 雏形]
    end

    subgraph 评估层
        EVAL[eval_rag.py<br/>Recall@K + MRR]
        TESTS[tests/<br/>413 pytest 用例]
    end

    CLI --> TOOLS
    API --> GRAPH
    API --> MATCH
    API --> JDP
    JDP --> ALIGN
    ALIGN --> MATCH
    API --> TOOLS
    API --> LOG
    PIPE --> MATCH
    PIPE --> PLAN
    SUM --> DEMO
    GRAPH --> TOOLS
    PLAN --> DEMO
    EVAL --> TOOLS
    TESTS --> MATCH
    TESTS --> TOOLS
    TESTS --> PIPE
    TESTS --> SUM
```

## 3. RAG 数据流（核心闭环）

```mermaid
sequenceDiagram
    participant U as 用户
    participant API as FastAPI
    participant LG as LangGraph
    participant CDB as ChromaDB
    participant Z as 智谱 API

    U->>API: POST /api/stream {query}
    API->>API: middleware 分配 request_id
    API->>LG: invoke(state)
    LG->>LG: embedding(query) [768-d 本地 bge]
    LG-->>LG: vector
    LG->>CDB: query(vec, top_k=5)
    CDB-->>LG: 5 chunks + metadata
    LG->>LG: build_prompt 注入片段
    LG->>Z: chat/completions stream=True
    Z-->>API: SSE token 流
    API-->>U: SSE: data: {delta: "..."}
    Note over API: 全程结构化 JSON 日志<br/>带 request_id
```

## 4. LangGraph 状态机

```mermaid
stateDiagram-v2
    [*] --> retrieve
    retrieve --> build_prompt: docs ∈ State
    build_prompt --> call_llm: messages ∈ State
    call_llm --> execute_tools: tool_calls?
    call_llm --> [*]: 直接答复
    execute_tools --> call_llm: tool result 注入
```

## 5. 关键技术决策

| 选择 | 理由 |
|------|------|
| 智谱 GLM-4-Flash + 本地 bge-base-zh-v1.5 嵌入 | 国产合规、JWT 鉴权清晰、本地 bge-base-zh-v1.5 输出 768 维语义足够 |
| ChromaDB（PersistentClient） | 零运维、SQLite 落盘、本地可重放 |
| LangGraph 状态机 | 显式声明 retrieve→prompt→llm→tools；覆盖常见 JD 的调用链路编排要求 |
| FastAPI + SSE | 流式输出对齐 ChatGPT 体验；middleware 统一日志 |
| 规则 + LLM 混合 | semantic_matcher 只对齐 JD 要求与既有证据；match_job 保留硬门槛和三档裁决权 |
| .env.local + .gitignore | 密钥不入 git，跨设备拉取后只需复制 .env.local |

## 6. 评估指标（当前口径）

| 指标 | 数值 | 数据集 |
|------|------|--------|
| Recall@5 | **0.98** (98/100) | 自建 100 题 3 桶集（fact/explain/cross_doc） |
| MRR | **0.922** | 同上 |
| 分桶命中 | R@1 **89%** · R@3 **96%** · R@5 **98%** | 同上 |
| pytest 通过率 | **437 passed / 3 skipped** | tests/ |
| 端到端流水线时延 | ~30s | pipeline.py（含 1 次 LLM 调用） |
| SSE 首 token 时延 | ~1-2s | /api/stream |

> 现场命令输出见 [`docs/verification_report.md`](verification_report.md)。

下一步优化：见 `eval_rag.py` Q6/Q7 miss 案例 → V2 引入 LLM rerank。
