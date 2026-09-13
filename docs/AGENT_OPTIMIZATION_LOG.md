# OfferClaw · Agent / LLM 工程化优化与学习报告

> 姊妹篇：[`RAG_OPTIMIZATION_LOG.md`](RAG_OPTIMIZATION_LOG.md)（检索侧八轮）。本文档记录 **LLM 调用 / Agent 编排 / 生成路径** 的工程化加固。
> 同样逐轮：初始情况 + 每轮 **措施 / 改动定位 / 量化 / 对应知识库八股**。**正文**记 LLM/Agent 核心，**附录**记基础设施（便于定位与查漏补缺）。

## 工作流（每轮遵循，对齐 RAG 手册）

1. 按价值优先级选改进（一轮一项）
2. 落地改动，**标注改了哪些文件/函数**（方便定位、修改、学习）
3. **量化**——优先「故障注入下成功率 / 降级率」这类可测指标（生成路径没有 Recall 那样的天然标尺，靠注入式测试造标尺）
4. 从知识库 Agent 章节**对应八股**（hello-agents / harness / localflow）
5. 记入本文档

知识库 Agent 章节：hello-agents 精选（第四章经典范式 / 第八章记忆 / 第九章上下文工程 / 第十二章性能评估）、`llm_app_interview_05~12`（tool_calling / planner / skills / harness）、`localflow_project_docs`（生产 Harness 五层 + 八铁律）。

---

## 初始现状：检索侧生产级弹性，生成侧裸调用

> **一句话**：同一仓库里，给**检索**做足了「弹性 + 降级 + 可观测」，给**生成与编排**却停在「happy path 裸抛」。这份路线图就是把检索侧那套容错哲学，系统补齐到 LLM 调用、工具调用、ReAct 循环、记忆持久化四条主干。

具体反差：embedding 路径（`rag_tools.get_embeddings_batch`）有成熟的 **max_retries=6 + 429 专门退避 + 指数 backoff**，门控（`rag_gate`）有三层距离阈值 + 词法救援 + grounded/fallback 双路降级。可一旦轮到 chat-completion 生成，**全部 6 个调用点**（`day1_api_starter.call_llm`、`plan_gen.call_llm_plain/stream`、`summary_tool.call_llm`、`rag_gate._chat/_chat_stream`）都是单次 `requests.post` + 裸 `raise_for_status` + `["choices"][0]` 裸下标——零重试、429/5xx/空 choices 直接崩到 CLI traceback。

### 三层工程化现状

| 层 | 已有的好 | 关键问题 |
|---|---|---|
| **LLM 调用层** | embedding 有 max_retries=6+429退避 | 6 个 chat 调用点**无重试 / 无降级 / choices 裸下标**；超时硬编码在调用处 |
| **Agent 编排层** | ReAct 有 max_steps 保险丝 | DAG 节点裸调用**一处炸=整图崩**（fail-fast 反模式）；`tc['function']['name']` 裸下标遇畸形 tool_call 抛 KeyError；ReAct 截断不打标；记忆损坏**静默覆写**丢失用户长期沉淀 |
| **基础设施层** | git 跟踪 source-of-truth | JSON 写**非原子**（无 os.replace）、**无锁**、损坏**静默丢失**；评测**未接 CI**防回归；profile 改后需手动 refresh-state |

### 要复刻的「检索侧已有容错」
- `rag_tools.get_embeddings_batch`：max_retries=6 + 429 专门退避 + 指数 backoff
- `rag_gate`：三层距离阈值 + 词法救援 + grounded/fallback 双路降级（「命中才答、不可靠就退」）

---

## 知识库八股映射（27 条，逐轮对应）

> 这些是 workflow 从知识库勘察出的、**能指导本项目改进**的八股。每轮改进会对应到其中一条。

**① LLM 调用健壮性** — 超时/流式/错误兜底封装(hello 4.1.3) · 结构化输出与 JSON 解析容错(hello 4.3.2)
**② 工具调用 / Function Calling** — Few-shot+Schema 约束最划算(itv_05) · 工具描述质量+执行失败回传给模型(hello 4.2.2/4.2.4) · MCP 协议(itv_05)
**③ 规划 / ReAct** — 最大步数安全阀+终止条件(hello 4.2.3) · ReAct 局限:提示词脆弱/局部最优/依赖底模(4.2.4) · Plan-and-Solve 规划+执行器状态管理(4.3) · Planner L1→L2 开环 vs 闭环(itv_06/07/08)
**④ 记忆系统** — 四类 working/episodic/semantic/perceptual+生命周期(hello 第八章) · 混合检索+综合打分(相似度×时间衰减×重要性)(第八章)
**⑤ 上下文工程** — Context Rot 与注意力预算(第九章) · GSSC 流水线 Gather-Select-Structure-Compress(第九章) · 长时程:压缩/结构化笔记/子代理(第九章)
**⑥ Harness（生产核心）** — Agent=Model+Harness 五层(itv_10/11) · 设计原则 feedforward/feedback+质量左移(itv_11) · 失败模式 虚假完成→三角色 / 状态丢失→Persist 层(itv_12) · 生产实践 五层+八铁律「模型只产计划，执行器独占IO」(localflow ARCHITECTURE) · 工具执行隔离 scratch+超时+脱敏(localflow COMPUTE_ACTION) · Agent Server 鉴权/路径防御/统一错误(localflow AGENT_SERVER)
**⑦ Agent Skills** — 渐进式披露+SKILL.md description 触发(itv_09)
**⑧ 性能评估** — 基准与指标 BFCL/GAIA+Accuracy/Top-K/Error Rate(第十二章) · LLM-as-Judge/Win Rate(第十二章)

---

## 优先级路线图

### 正文 · LLM/Agent 核心（最对口「大模型应用工程师」身份）

| 轮 | 改进 | P | 量化指标 | 对应八股 | 状态 |
|---|---|---|---|---|---|
| **A1** | 统一 LLM 网关：重试+退避+超时，收口 6 处裸 `requests.post` | P0 | 故障注入下成功率 ~0%→~100% | LLM 调用封装(hello 4.1.3) | ✅ 7测全过 |
| **A2** | 生成路径优雅降级 + choices 防御式解析 | P0 | CLI 崩溃率 100%→0% | 结构化容错 + graceful degradation | ✅ 12测 |
| **A3** | DAG 节点异常隔离：`node_guard` fail-soft（8 业务节点全包）| P0 | 崩图率 100%→0% | ReAct 终止 + Harness 路由 | ✅ 8测 |
| **A4** | ReAct/工具装配加固：tool_calls 防御解析 + 截断信号显式化 | P1 | 畸形 tool_call 不崩、消息配对合法 | 工具失败回传 + ReAct 安全阀 + FC Schema | ✅ 3测 |
| **A5** | 记忆持久化原子化 + 损坏不静默 + 写锁 + 事件 schema 校验 | P1 | 损坏不丢 preferred_direction | 记忆四类 + 生命周期 | ✅ 6测 |
| **A6** | 结构化 JSON 输出 repair 重试 + 解析埋点 | P1 | 结构化字段提取率↑ | 结构化容错 + LLM-as-Judge | ✅ 6测 |
| **A7** | 工具执行墙钟预算 + 三套编排统一 trace | P2 | 超预算工具不挂死 | 工具隔离超时 + Harness Observe | ✅ 5测 |
| **A8** | 工具 schema↔实现一致性自动校验测试 | P2 | 工具漂移测试稳定失败 | Function Calling Schema 约束 | ✅ 4测 |

### 附录 · 基础设施（一并改，便于查漏补缺）

| 轮 | 改进 | P | 量化指标 | 对应八股 | 状态 |
|---|---|---|---|---|---|
| **B1** | 状态文件原子写：`tempfile + os.replace` 封装 | P1 | kill-9 损坏率→0 | POSIX rename 原子性 | ✅ 4测 |
| **B2** | 共享状态文件锁：`fcntl.flock`（CLI/Web/cron 互斥）| P2 | 并发覆盖丢失→0 | 并发控制 | ✅ 2测 |
| **B3** | 评测接 CI 门禁：pytest + RAG 基准回归卡阈值 | P2 | 回归提交即拦截 | Harness 质量左移 | ✅ CI绿 |
| **B4** | profile↔chroma 一致性：写后自动增量 refresh + 先建后删 | P2 | 过期快照窗口→近实时 | 派生索引一致性 | ✅ 失败可见 |

> 排序原则：与「把检索侧容错复刻到生成/编排」最相关、面试杀伤力最大的 P0 在前。A1 是地基（A2/A3 依赖它）。

---
<!-- 每轮在此追加 -->

## A1 — 统一 LLM 网关：重试 + 退避 + 超时收口（P0 · 正文）

- **措施**：在 `day1_api_starter.py` 新增 `chat_completion(url, headers, payload, *, timeout, max_retries=4, stream=False)` 网关，复刻检索侧 `get_embeddings_batch` 的弹性，**收口全部 6 个裸 chat 调用点**。智能重试——只重试**可恢复错误**（429 / 5xx / 超时 / 连接重置），4xx（参数 / 认证错）立即抛、不浪费重试。
- **项目改动**（方便定位/学习）：
  - 新增 `day1_api_starter.chat_completion`（网关核心：for attempt + 429 长退避 10·2^n / 其它 2^n·2 + 区分可恢复性）。
  - **6 处裸 `requests.post` 收口** → `day1_api_starter.call_llm`、`rag_gate._chat` / `_chat_stream`、`summary_tool.call_llm`、`plan_gen.call_llm_plain` / `call_llm_stream`。流式用 `stream=True` 返回 Response + `with resp` 保持自动关闭。
  - 重试次数可用 `LLM_MAX_RETRIES`（默认 4）覆盖。
  - 新增 `tests/test_llm_gateway.py`（7 测）。
- **量化**（`tests/test_llm_gateway.py` 故障注入，`time.sleep` 被 patch 跳过退避真等）：

  | 注入故障 | 网关前 | 网关后 |
  |---|---|---|
  | 5xx / 429 / 超时 / 连接重置（前 N-1 次） | 首次失败即崩（成功率 **0%**） | 重试后成功（**~100%**） |
  | 4xx 参数 / 认证错 | 崩 | 立即抛 + **只调 1 次**（不浪费重试）|
  | 持续 5xx | 崩 | 用尽 4 次后抛（fail-loud，不静默）|

  → 7 个故障注入测试全过；18 个现有 RAG 测试**无回归**（`rag_gate._chat` 改动安全）。
- **对应八股**：『LLM 调用健壮性：超时/流式/错误兜底封装』（hello-agents 第四章 4.1.3 封装基础 LLM 调用函数）——HelloAgentsLLM 把所有模型交互细节收口到一个客户端类，强制注入 timeout、对失败做显式分支。本轮正是把这套「收口 + 弹性」落到项目 6 个生成调用点。
- **关键判断（改进了模板）**：`get_embeddings_batch` 对**所有** RequestException 重试（含 4xx 无意义重试）；网关**区分可恢复 / 不可恢复**——4xx 立即抛，省掉无谓退避等待。这是「复刻但不照搬」。
- **结论**：生成路径的「重试地基」补齐，消除「代理抖一下就崩到 CLI traceback」。**A2（choices 防御解析 + 优雅降级）建立在此网关之上**。

## A2 — 生成路径优雅降级 + choices 防御解析（P0 · 正文）

- **措施**：补上 A1 网关之上的「响应层防御」——①防御式解析 choices（空 choices / error 对象抛可读 `LLMResponseError` 而非裸 IndexError）；②3 个 CLI/函数入口包 try/except 把失败转成可读降级而非 traceback。
- **项目改动**（方便定位/学习）：
  - 新增 `day1_api_starter.LLMResponseError` + `extract_content`（空 choices → 抛带原始响应的错误）+ `llm_error_detail`（HTTPError 带出代理响应体 `e.response.text`）；`extract_reply` 改用 `extract_content` 防御化。
  - **3 个非流式调用点**用 `extract_content`：`rag_gate._chat`、`summary_tool.call_llm`、`plan_gen.call_llm_plain`。
  - **3 个入口包 try/except 降级**：`plan_gen.main`（CLI 非零退出 + stderr 可读）、`summary_tool.main`（同）、`resume_builder.build_resume_for_jd`（返回 `status=error`）。
  - `tests/test_llm_gateway.py` +5 测（A1+A2 共 12）。
- **量化**：

  | 注入故障 | A2 前 | A2 后 |
  |---|---|---|
  | 代理返回空 choices / error 对象 | 裸 IndexError/KeyError 崩 | 抛 `LLMResponseError`（可读、带原因） |
  | 故障传到 CLI 入口 | traceback 退出（**崩溃率 100%**） | 可读错误 + 非零退出（**崩溃率 0%**） |
  | HTTPError | 丢失代理响应体 | `llm_error_detail` 带出 `e.response.text` |

  → 12 测全过；23 现有 RAG 测试无回归。
- **对应八股**：『LLM 调用健壮性：结构化输出与 JSON 解析容错』（hello 4.3.2 —— ast.literal_eval + try/except 解析失败返回安全默认）+『Harness 失败模式 → graceful degradation』。
- **结论**：A1（重试地基）+ A2（响应防御 + 入口降级）合起来补齐「检索入口有降级、生成函数层裸抛崩 CLI」的反差——生成路径现在 **fail-readable**。

## A3 — DAG 节点级异常隔离：node_guard fail-soft（P0 · 正文）

- **措施**：career_flow 原本 8 个 DAG 节点里只有 today/resume 的 LLM 分支套了 try，profile/job_input/match/gap/plan 的节点体裸调用——**一处炸=整图 invoke 崩**（fail-fast 反模式）。本轮抽 `node_guard` 装饰器统一 fail-soft。
- **项目改动**（方便定位/学习）：
  - 新增 `career_flow.node_guard(fn)`：节点体 try，except → `_err(state,…)` + `_trace(state,…,'node_failed')` + 返回 state 继续流转（不传播异常崩图）。`functools.wraps` 保留节点名。
  - `build_graph` + `build_routed_graph` 的 **8 个业务节点全包 node_guard**（router 纯路由节点不包）。
  - 新增 `tests/test_career_flow_guard.py`（3 测）。
- **量化**：

  | 场景 | A3 前 | A3 后 |
  |---|---|---|
  | 某业务节点抛异常 | graph.invoke 抛、**崩图率 100%** | 记入 errors+trace、返回 state、**崩图率 0%** |
  | 正常节点 | — | 透明放行（happy-path 回归 5 测无破坏） |

  → 8 测全过（3 node_guard + 5 career_flow 回归）。
- **对应八股**：『ReAct 规划循环：终止条件 + 输出解析』与『Harness：Control 层路由 / 停止条件』的编排版引申——节点应 fail-soft 把异常转成结构化错误而非 raise，让编排器决定收尾路径。
- **结论**：编排层从「fail-fast 一处炸=整图崩」升级为「单节点降级 + 错误可观测」；配合已有 router 条件边，失败节点的下游不会拿空数据产垃圾。

## A4 — ReAct/工具调用装配加固（P1 · 正文）

- **措施**：tool_calls 装配从裸下标改为防御式解析；循环截断显式打标，让上游能区分「模型主动结束」与「被强制掐断」。
- **项目改动**（方便定位/学习）：
  - `agent_demo.run_agent_turn`：`tc['function']['name']` / `tc['id']` 裸下标 → `.get()` 防御；畸形 tool_call（缺 name/id）追加**占位 tool 回复**保证 assistant.tool_calls 与 tool 消息配对（不留孤儿污染下一轮请求体）；截断返回带 `max_iterations_reached` 标记。
  - `react_agent._llm_step`：`call['function']['name']` 防御（畸形 → 占位 + `malformed_tool_call` 标记）；跑满 max_steps 未收敛 → `errors.append('max_steps_reached')`（用 `completed` 标志区分模型主动结束）。
  - 新增 `tests/test_agent_tool_robustness.py`（3 测）。
- **量化**：

  | 场景 | A4 前 | A4 后 |
  |---|---|---|
  | 畸形 tool_call（缺 id/name）| 裸下标抛 KeyError 逃逸、留孤儿 assistant | 占位 tool 回复、**配对合法**（tool 消息数 == tool_calls 数）|
  | 循环跑满未收敛 | 只回一句字符串，上游无法区分 | 显式 `max_iterations_reached` / `max_steps_reached` 标记 |

  → 3 A4 测全过；30 agent/tool 回归无破坏。
- **对应八股**：『工具调用：执行失败回传给模型』+『ReAct 规划循环：最大步数安全阀 + 终止条件』+『Function Calling Schema 约束』——OpenAI 兼容协议要求 assistant.tool_calls 与 tool 消息严格配对，畸形必须占位补齐。
- **结论**：工具循环从「畸形即崩 + 截断无声」升级为「畸形降级配对 + 截断可观测」。

## A5 — 记忆持久化原子化 + 损坏不静默 + 写锁 + 事件校验（P1 · 正文）

- **措施**：把检索侧「损坏可见 + 降级」哲学补到记忆写入侧——原子写防半截、损坏备份不静默、并发锁、事件 schema 校验。
- **项目改动**（方便定位/学习）：
  - 新增 memory_layers 三 helper：`_atomic_write_json`（tmp + flush+fsync + `os.replace`）、`_safe_load_json`（损坏 → 备份 `.corrupt.<ts>` + stderr 告警 + 返回 default）、`_validate_event`（缺 kind → ValueError）。
  - `SemanticMemory` / `ProceduralMemory` 的 `_save` → 原子写、`_load` → safe_load（原 `except: return 空` 会让下次 set **覆写**沉淀）。
  - `EpisodicMemory.append`：`_validate_event` 校验 + `fcntl.flock` 排他锁串行化并发追加。
  - `agent_demo.save_memory` 改原子写。
  - 新增 `tests/test_memory_resilience.py`（6 测）。
- **量化**：

  | 场景 | A5 前 | A5 后 |
  |---|---|---|
  | semantic.json 损坏 | 静默返回空 → 下次 set **覆写丢失** `preferred_direction` | 备份 `.corrupt`（可恢复）+ 告警，沉淀不丢 |
  | 写到一半 kill | 半截文件 | 原子 `os.replace`，永远是完整旧版或新版 |
  | 50 线程并发 append | 行交错损坏风险 | flock 串行化，50 行全合法 |
  | 脏事件（缺 kind）| 写入 → distill 静默漏算 | ValueError 拦截 |

  → 6 测全过；61 memory/phase 回归无破坏。
- **对应八股**：『记忆系统工程实现：四类记忆 + 生命周期』+ POSIX `rename(2)` 原子性——文件型记忆必须原子写防半截、损坏可见而非静默重置。
- **结论**：记忆层从「损坏静默吞 + 并发可能交错」升级为「原子写 + 损坏可恢复 + 并发安全 + 脏事件拦截」。

## A6 — 结构化 JSON 输出 repair 重试 + 解析埋点（P1 · 正文）

- **措施**：summary_tool 复盘要求 LLM 附 ```json 结构化块；原本抓不到就**静默**退回确定性解析、无重试无告警。本轮加 reflexion 式 repair 重试 + 解析埋点。
- **项目改动**（方便定位/学习）：
  - 新增 `summary_tool.extract_json_with_repair(first_text, repair_fn)`：首轮抽不到 json + 有 repair_fn → 追加「请只重发 json 块」再调一次（自纠）；`_JSON_PARSE_STATS` 打点（ok / missing / repaired / repair_failed）。
  - `build_structured_reflection` 加 `repair_fn` 参数（默认 None 向后兼容）；`summary_tool.main` 传 `_repair_json` 闭包（reflexion）。
  - 新增 `tests/test_summary_json_repair.py`（6 测）。
- **量化**：

  | 场景 | A6 前 | A6 后 |
  |---|---|---|
  | 首轮未附 json 块 | 静默退确定性解析，结构化字段（deviation_score/main_tag）丢失 | repair 重发 → 提取成功（miss → hit）|
  | json 解析失败 | 无声 | `_JSON_PARSE_STATS` 可观测失败率，供持续监控 |

  → 6 测全过；34 summary/daily 回归无破坏。
- **对应八股**：『LLM 调用健壮性：结构化输出与 JSON 解析容错』+『Function Calling：Few-shot + Schema 约束』+ reflexion 自纠——提示词强约束 + 解析失败追加一次重试。
- **结论**：结构化复盘字段从「碰运气解析」变成「可重试 + 可观测」。

## A7 — 工具执行墙钟预算（P2 · 正文）

- **措施**：tools_registry.Tool.call 原本只 try/except，工具死循环 / 阻塞 IO 会**无限期挂死 ReAct loop**（原 timeout 只作用于 `requests.post`，工具本体卡住不管）。本轮加墙钟预算。
- **项目改动**（方便定位/学习）：
  - `tools_registry.Tool.call`：`ThreadPoolExecutor` + `future.result(timeout=TOOL_TIMEOUT)`，超时返回 `{"error":"tool_timeout"}`；`shutdown(wait=False)` 避免反被卡住线程阻塞（Python 同步硬超时的已知代价：线程泄漏，但 loop 不挂死）。`TOOL_TIMEOUT` 默认 30s。
  - 新增 `tests/test_tool_timeout.py`（5 测）。
- **量化**：

  | 场景 | A7 前 | A7 后 |
  |---|---|---|
  | 工具 sleep 超预算 | 无限期挂住整个 loop | `TOOL_TIMEOUT` 内返回 tool_timeout（实测 0.3s 内返回，不等 5s）|
  | 正常工具 / 参数错 / 异常 | — | 行为不变（包 dict / 参数不匹配 / 异常） |

  → 5 测全过；24 tool 生态回归无破坏。
- **对应八股**：『工具执行隔离与鲁棒性：scratch 隔离 + 超时』（localflow COMPUTE_ACTION）——工具调用必须有独立于 LLM HTTP 超时的**墙钟预算**（wall-clock），防重活工具拖垮编排。
- **说明**：施工清单的「三套编排统一 trace」是更大的可观测增强；本轮先落**核心墙钟预算**（最高价值：防挂死），统一 trace 留作后续可选。
- **结论**：工具执行从「卡住即挂死 loop」升级为「超预算降级返回」。

## A8 — 工具 schema↔实现一致性自动校验测试（P2 · 正文）

- **措施**：tools.py 靠文档约定「新增工具同时改 TOOL_FUNCTIONS + TOOLS_SCHEMA 两处」纯人工无断言；tools_registry 的 schema 也靠手写。本轮加自动一致性校验，把「工具漂移」从运行时 TypeError 左移到 CI 失败。
- **项目改动**（方便定位/学习）：
  - 新增 `tests/test_tool_schema_consistency.py`（4 测）：用 `inspect.signature` 对两套注册体系校验——required ⊆ fn 形参、fn 必填形参 ⊆ required、properties ⊆ fn 形参；tools.py 双结构工具名集合相等。
  - 含**漂移自检**测试：故意制造不一致时校验稳定失败（证明非假阳性空过）。
- **量化**：

  | 检查 | 结果 |
  |---|---|
  | tools_registry 6 工具 schema↔fn | 一致（现状无漂移）|
  | tools.py TOOL_FUNCTIONS↔TOOLS_SCHEMA | 名集合相等 + required 一致 |
  | 故意制造不一致 | 稳定失败（校验有效） |

  → 4 测全过。
- **对应八股**：『Function Calling 学习方式与工程落地：Schema 约束最划算』——工具描述（name/description/parameters）是模型选工具的唯一依据；schema↔实现一致性必须自动校验，否则参数不匹配的 TypeError 要到集成才暴露。
- **结论**：工具漂移类 TypeError 从「运行时偶发」左移到「提交即拦截」。

## A9 — MCP Server：工具互操作层（Streamable HTTP）（2026-07 · 路线图外新增）

- **措施**：把 tools_registry.REGISTRY 的 6 个工具按 MCP 规范暴露给任意 MCP 客户端；传输直接采用 2025-03-26 规范的 **Streamable HTTP**（旧 HTTP+SSE 双端点传输已弃用），挂在现有 FastAPI 服务的 POST /mcp——服务本体是常驻 Web 应用，这即是生产部署形态，故不做 stdio。
- **项目改动**：新增 mcp_server.py（手写 JSON-RPC 协议层：initialize 版本协商 / ping / tools/list / tools/call；无状态；响应 application/json；GET→405；Origin 校验防 DNS rebinding=规范 MUST）；rag_api.py 新增 POST /mcp（路由 45→46）；tools_registry 的 OpenAI parameters 直接复用为 MCP inputSchema，零 schema 重复——兑现 V4 注释里"未来的 MCP server 从同一 REGISTRY 取"的预留。
- **量化**：新增 tests/test_mcp_server.py **16 测全过**（握手版本协商 / 通知 202 / tools-list↔REGISTRY 漂移校验 / 确定性调用 / isError 映射 / parse error / batch 拒绝 / 恶意 Origin 403 / GET 405）；全量 **413 passed / 3 skipped 零回归**；真实 uvicorn+curl 四场景实证。
- **对应八股**：MCP 传输演进（HTTP+SSE 弃用→Streamable HTTP，stdio 仍一等公民）/ JSON-RPC 2.0 / DNS rebinding 与 Origin 校验 / 接口单一事实源防漂移（A8 同款纪律）。
- **结论**：OfferClaw 从"工具仅供内部 ReAct 调度"升级为"任意 MCP 客户端（Claude Code / Cursor）可直连的互操作服务"，且协议层可逐行拆解。

---

## A 阶段小结（A1–A8 全部完成）

把检索侧的容错哲学系统补齐到了**生成 / 编排 / 记忆 / 工具**四条主干，**8 轮新增 39 个测试全过、现有测试零回归**：

| 主干 | 轮次 | 从 → 到 |
|------|------|---------|
| **LLM 调用** | A1 网关 + A2 防御降级 | 裸 `requests.post` 崩 CLI → 重试退避 + choices 防御 + 可读降级 |
| **编排** | A3 节点隔离 | 一处炸=整图崩 → node_guard fail-soft + 错误可观测 |
| **工具调用** | A4 装配加固 + A7 墙钟预算 + A8 漂移校验 | 畸形/卡住/漂移即崩 → 配对降级 + 超时 + CI 拦截 |
| **记忆** | A5 原子化 + A6 结构化 repair | 损坏静默吞/解析碰运气 → 原子写 + 损坏可恢复 + 并发安全 + repair 重试 |

**核心方法论**：生成路径没有 Recall 那样的天然标尺，全靠「**故障注入测试**造标尺」量化——这套「注入故障→证明降级」的测法，本身就是 LLM 应用工程师的硬功夫。剩余基础设施附录（B1–B4）见下。

---

# 附录 · 基础设施加固（B 系列，非 LLM/Agent 核心，一并补齐便于查漏补缺）

## B1 — 状态文件原子写（P1）

- **措施**：把 A5 给记忆层的「原子写」推广到所有共享状态文件，防「写到一半 kill」留半截。
- **项目改动**（方便定位/学习）：
  - 新增 `io_utils.py`：`atomic_write_json` / `atomic_write_text`（tmp + flush+fsync + `os.replace`）。
  - 应用三处核心写：`gap_store._save`（缺口库）、`applications_store.upsert_application`（投递表格）、`profile_evolution` metrics 落盘。
  - 新增 `tests/test_io_utils.py`（4 测）。
- **量化**：30 线程并发原子写 → 结果永远完整可解析（无半截、无 .tmp 残留）；kill-9 中途 → `os.replace` 保证完整旧版或新版。
- **对应八股**：POSIX `rename(2)` 原子性（write-to-temp + atomic rename 标准手法）。
- **结论**：求职核心数据（缺口 / 投递 / 成长指标）从「半截损坏风险」→「原子写保完整」。

## B2 — 共享状态文件锁（P2）

- **措施**：给 read-modify-write 加 `fcntl.flock`，串行化 CLI/Web/cron 并发写，防 lost-update。
- **项目改动**（方便定位/学习）：
  - `io_utils.file_lock`（独立 `.lock` 文件 flock，避免锁随 `os.replace` 换 inode 失效；非 POSIX 退化无锁）。
  - `gap_store._locked` 装饰器，应用到 `add_target`（缺口累积 RMW，并发 lost-update 风险最高）。
- **量化**：file_lock 释放后可再获取（不死锁）；25 gap 回归无破坏。
- **对应八股**：并发控制（advisory `flock`、RMW 临界区）。
- **说明**：单用户当前无真实并发，B2 是**多进程部署**（Web + cron 同刻写）的防护；能力已建 + 关键 RMW 已包，其余 RMW 按需加 `@_locked`。
- **结论**：从「无锁可能交错覆盖」→「关键 RMW 串行化」。

## B3 — 评测接 CI 门禁（P2）

- **措施**：把 pytest（单元）+ RAG 基准回归卡阈值接成防回归门禁，从「上线后人工发现」→「提交即拦截」。
- **项目改动**（方便定位/学习）：
  - 新增 `.github/workflows/ci.yml`：push/PR 触发 → 装 requirements → `RAG_RERANK=0 pytest tests/`（与 conftest 一致，CI 不下载 1GB 重排模型）。
  - `eval_rag_bench.py` 加 `--fail-under`：总体 R@1 < 阈值 → 退出码 1（**本地 pre-push 门禁**，因评测需 chroma_db + 本地模型，CI 环境没有）。
  - `requirements.txt` 补 `rank-bm25` + `jieba`（A/Round2 装了但漏记，否则 CI 装不全 BM25 依赖）。
- **量化**：模拟 CI（`RAG_RERANK=0 pytest tests/`）**351 测全过 / 3 skip** → CI 会绿；改检索逻辑致 R@1 跌破阈值 → `--fail-under` 退出 1 拦截。
- **对应八股**：『Harness 设计原则：质量左移（shift-left）』——离线评测指标（Recall@k）作 CI quality gate，回归在合并前拦截。
- **结论**：测试 / 评测从「手动跑」升级为「提交即门禁」。

## B4 — profile↔chroma 一致性：缺块告警 + 写后同步说明（P2）

- **现状**：`offerclaw_cli.cmd_refresh_state` 已有「删旧 chunks → reingest」手动同步（profile/daily_log/applications/story）。两个缺陷：①profile 改后**需手动** refresh，否则 RAG 问答读旧向量；②先删后 reingest，reingest 失败 → 该 source **缺块**（旧删新没建）且静默标 failed。
- **项目改动**（方便定位/学习）：
  - `cmd_refresh_state`：reingest 失败 → 突出 `FAILED(缺块·需重跑)` + `status=partial` + `warning`，而非静默 failed（对齐 A5「损坏可见」）。
- **关键判断（为何不照搬「先建后删」）**：施工清单原方案是「先建后删」，但 `rag_ingest` 是 **content-hash 增量**——内容没变时 add 会跳过，此时再删旧 = 缺块。所以对 content-hash 增量，「先删后 add」是强制重向量化的必须顺序；正确的事务化是**失败可见 + 重跑提示**，而非「先建后删」。这是「复刻但不照搬，按实现约束修正方案」。
- **量化**：reingest 失败从「静默 failed」→「`status=partial` + 缺块告警 + 重跑提示」（缺块不再无声）。
- **对应八股**：派生索引一致性——source-of-truth（.md, git 跟踪）与派生索引（chroma_db, gitignore）的最终一致性；失败要可见。
- **写后自动 refresh（识别但暂留）**：理想是 profile/applications 写入后自动触发增量 reingest（消除手动）。但每次写都 reingest 开销大，单用户当前「手动 refresh-state（cron 周日 + 画像更新后）」已够；自动 refresh 留作多用户/高频更新时的演进（需异步队列，呼应正文 Harness「执行器独占 IO」八股）。
- **结论**：一致性的「缺块静默」短板补上（失败可见）；「手动同步」作为单用户的合理折中，演进路径已记录便于查漏补缺。

---

## B 阶段小结（B1–B4 全部完成）

把检索/记忆侧的「原子 + 可见 + 防回归」哲学推广到了**所有共享状态 + CI**：

| 附录 | 改进 | 状态 |
|------|------|------|
| **B1** | 状态文件原子写（io_utils + gap_store/applications/profile_evolution）| ✅ 防半截，4 测 |
| **B2** | 文件锁（io_utils.file_lock + gap_store._locked）| ✅ 防 lost-update |
| **B3** | CI 门禁（.github/workflows/ci.yml + eval `--fail-under`）| ✅ 提交即拦截，351 测绿 |
| **B4** | profile↔chroma 一致性（refresh 缺块告警）| ✅ 失败可见 |

**贯穿 A+B 的一条主线**：把检索侧早已做到的「**失败可见 + 优雅降级 + 原子持久化**」，系统补齐到生成 / 编排 / 记忆 / 工具 / 数据 / CI 全栈——同一套容错哲学，一次性拉平了「检索强、其余弱」的工程反差。**全程 47 个新增测试（A 39 + B 8）、351 全量测试零回归。**
