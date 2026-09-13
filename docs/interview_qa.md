# OfferClaw · 面试问答卡

> 覆盖技术深度 + 项目动机 + 后端工程视角 + 反问准备。

---

## Q1: 为什么用 LangGraph 而不是直接 if/else 编排？

**A**：手写编排有 3 个痛点：① 工具调用循环要自己写终止条件；② 状态在函数间传递混乱；③ 不可视化没法跟人讲清楚。LangGraph 把"节点 + 边 + 条件分支"声明式建模，4 节点（retrieve / build_prompt / call_llm / execute_tools）+ 1 条件边（是否还有 tool_calls）就描述清楚了整个 RAG-Agent。
**踩过的坑**：`add_messages` 注解会自动把 dict 转成 `BaseMessage` 对象，传给智谱 API 时 `json.dumps` 序列化失败（SystemMessage 不可 JSON 化）→ 我加了一个 `_msg_to_dict()` 适配层。

---

## Q2: 为什么自己写 JWT 而不用 PyJWT？

**A**：智谱的 JWT 规范有两点特殊：① payload 时间戳是**毫秒**不是秒；② signing secret 与 key_id 用 `.` 分隔成单一环境变量。引第三方包要么版本绑定、要么得自己适配——纯标准库 `hmac+hashlib+base64url` 实现 30 行就够，对依赖更可控。

---

## Q3: ChromaDB 3327 chunks 性能够吗？后续怎么扩？

**A**：当前是个人项目，全量 3327 chunks 单次查询数十 ms，PersistentClient SQLite 落盘足够。扩展路径：① chunks > 10 万走 HNSW 索引调参（M、ef_construction）；② chunks > 100 万切到 Milvus / Qdrant；③ 加 metadata filter（按 source 预筛）减小搜索域。

---

## Q4: Recall@5 = 0.98 意味着什么？怎么保持并巩固？

**A**：自建 100 题 3 桶集（fact / explain / cross_doc），当前 Recall@5=0.98（top-5 全部命中期望源），R@1=89%、R@3=96%、MRR=0.922。
<!--HIST-->
**早期两个 miss 案例已定位（历史）**：① `f03`（"目标方向第一优先级"）被 SOUL.md 抢占；② `e05`（"识别伪造信息"）被 DATA_CONTRACT.md 抢占。两者根因都是 chunk 边界把关键句切碎了——已通过调小 chunk_overlap + 按 ## 二级标题保留语义单元修复，Recall@5 从早期 0.96 一路优化到当前 1.00。
<!--/HIST-->
**下一步**：① 加 LLM rerank（top-20 → top-5）进一步抬高 R@1；② 扩评估集到 100 题做 ablation；③ baseline 写进 `tests/rag_eval_baseline.json` 实现回归对比。

---

## Q5: SSE 流式输出 vs WebSocket 怎么选？

**A**：① 单向推流（LLM token 单向往前端推）SSE 足够，WebSocket 是双向；② SSE 走 HTTP，无需额外协议握手，反向代理穿透更友好；③ SSE 自动重连。我这里只需要"一个问题→一段流式回答"，所以 SSE。如果做多轮 stream-in/stream-out 才上 WebSocket。

---

## Q6: 为什么硬门槛规则版要 Python 写、4 周计划要 LLM 写？

**A**：分清"确定性问题"和"生成性问题"。
- **硬门槛**（学历、地域、专业）= 业务规则 → Python 显式判断，可解释、可单测、不允许 LLM 脑补；
- **4 周计划**（叙事、节奏、内容编排）= 生成性 → LLM 拿手；规则写不出来。
这是项目里最重要的设计原则之一：**让 LLM 做它擅长的，规则做它擅长的，别混在一起。**

---

## Q7: 怎么保证 API Key 不进 Git？

**A**：① 把 key 写在 `.env.local`（双扩展名）；② `.gitignore` 加 `.env*` + 常用变体；③ `rag_tools._load_local_env()` 在 import 时读取写到 `os.environ`；④ 加一份 `.env.example` 给协作者参考。**踩过的坑**：`.gitignore` 不支持行内注释，`chroma_db/  # 注释` 会被解析成"忽略名为 chroma_db/  # 注释 的目录"，导致规则失效，要单独成行。

---

## Q8: pytest 437 个用例，哪些是真正"防御性"的？

**A**：核心防御性用例 4 类：
1. **三档结论枚举校验**（`test_match_demo_runs`）—— 防止改规则把结论字符串写错；
2. **persona schema 校验**（`test_persona_schema`）—— 防止新增 persona JSON 漏字段；
3. **multi-persona × multi-JD 参数化**（12 用例）—— 改 match_job 规则后立即看到副作用；
4. **FastAPI TestClient 接口测 + 主链路冒烟**（`tests/test_api.py` + `tests/test_pipeline.py`，新增 19 用例）—— 不依赖 LLM 的 8 个离线 + 3 个 e2e（默认跳过，OFFERCLAW_E2E=1 才跑）+ 6 步主链路 smoke。

总计 413 通过 / 3 skipped，3 e2e 待 flag 触发。

---

## Q9: 这个项目实际用没用？跑了多久？

**A**：从 2026-04-15 开 SOUL.md 起到现在 ~10 天，我每天用它来：① 看我新发现的 JD 适合不适合投；② 把 4 周路线写进 daily_log；③ 周日跑 weekly summary。GitHub commit 历史就是真实使用日志。它**首先**是一个解决我自己问题的产品，**其次**才是一份简历项目——这是我和很多"为了简历而做"的项目最大的区别。

---

## Q10～Q15: 高频追问（六连问）

### Q10: 为什么不直接用 LangChain Agent / LlamaIndex？
**A**：① LangChain Agent 黑盒太多，工具循环和 prompt 拼装藏在内部，调试困难；② LlamaIndex 偏向"知识库 + RAG"重场景，我这里需要"规则 + LLM + RAG"混合编排，LangGraph 的状态机更直白；③ 项目要写进简历，需要"我能讲清每一步为什么"——黑盒越少越好。LangGraph + 直调 requests + 手写 JWT 让每一行都可解释。

### Q11: 为什么评估集只有 100 题？是不是太少？
**A**：完全同意小，所以 README/简历都标了"**自建小规模评估集**"，不冒充通用基准。58 题（精确到目标文件的新版主口径，旧 50 题已降为辅助）是单人项目能负担的标注成本天花板（每题要写 q + 至少一个 expected_source）。3 桶设计是为了能看到分项弱点（fact / explain / cross_doc）。下一步 100 题，再下一步引入合成数据 + 人工抽检。

### Q12: `/api/profile` 是不是写死的 demo 数据？
**A**：之前是，现在不是。当前实现 `_parse_profile()` 用正则从 `user_profile.md` 解析 name / direction / skills / updated_at，user_profile.md 改名（比如 示例用户 → 张三）API 返回会跟着变。可以现场 demo：改文件 → curl /api/profile 立刻看到新值。这是"画像驱动"主张的最小证据。

### Q13: ChromaDB vs FAISS / Milvus / Qdrant 怎么选的？
**A**：① 单人项目 100~10000 chunks 量级，FAISS 要自己管持久化和元数据，麻烦；② Milvus / Qdrant 要起 server，部署成本高；③ ChromaDB PersistentClient 直接 SQLite 落盘，自带元数据 filter，单文件可移植。**取舍**：放弃了"分布式 / 千万级"，换"零部署 / 直接可演示"。这是写在 `docs/postmortem.md` 的第 1 条取舍。

### Q14: Agent 会不会乱改我的画像？怎么保证？
**A**：写在 `DATA_CONTRACT.md` 的 7 条不变量第 1 条：**Agent 永不直接修改 `user_profile.md`**。所有"画像更新建议"通过两条路径：① 写到 `summaries/` 让用户人工确认后回写；② 写到 `memory.json` 作为 Agent 短期记忆，和 user 层物理隔离。User Layer / System Layer / Runtime Layer 的边界写在 contract 里，doctor.py 会检查这三层目录是否齐全。

### Q15: 智谱 `embedding-3` 为什么选这个？换 OpenAI / BGE 行不行？
**A**：当前口径已切到本地 `bge-base-zh-v1.5`（768 维），零 API 成本、可离线演示、语义足够。<!--HIST-->早期一度用智谱 `embedding-3`（2048 维），当时理由是国产合规 + 与 GLM-4-Flash 同栈一份 JWT 签名搞定；后来为省成本/可离线换到本地 BGE。<!--/HIST--> **换不行吗？** 完全可以，`rag_tools.py` 把 embedding 调用收敛到一个函数 `get_embedding()`，换模型只要改这一处 + 重新 ingest 一次 chroma。这是"接口收敛"的好处。

---

## Q16～Q22: 后端工程视角追问（七连问，面试官后端出身时高频）

> 核心立场：**项目没有 MySQL/Redis，但它们背后的每个经典问题我都在文件层亲手解决过一遍**——"为什么不用"的工程判断比"用了"更能证明理解。

### Q16: 为什么不用 MySQL / Redis？数据层怎么设计的？上生产怎么迁？
**A**：单用户本地场景，引入 DB/缓存中间件是过度设计——但对应的问题一个没躲：事务原子性（`io_utils.atomic_write`：tmp+fsync+rename）、并发隔离（flock 文件锁）、UPSERT（`applications_store.upsert_application`，语义同 ON DUPLICATE KEY UPDATE）。**关键设计**：所有状态读写收口在 store 层（`applications_store.py` / `gap_store.py`，本质是 DAO 抽象），换存储后端只动一个文件；roadmap 已明文写"applications 迁 SQLite + 自动化状态机"。迁移路径：SQLite（零部署）→ MySQL（多用户）+ Redis 挂 profile/embedding 缓存。

### Q17: 单机多进程（Web + cron）并发写怎么处理？
**A**：真实风险是 RMW lost update（缺口累积是读-改-写）。方案：`io_utils.file_lock` 用**独立 .lock 文件**做 flock——不锁数据文件本身，因为原子写靠 `os.replace` 换 inode，锁在数据文件上会随 inode 失效（踩坑点）。`gap_store.add_target` 用 `@_locked` 装饰器包临界区，30 线程并发测试无半截/无交错。**映射到 DB 八股**：advisory lock ≈ 悲观锁；上 MySQL 对应 SELECT FOR UPDATE，或版本号 CAS 走乐观锁——单机低冲突场景我选悲观锁因为实现最简单、正确性最好证明。

### Q18: 没有 Redis，缓存和一致性怎么做的？
**A**：两处真实缓存：① `profile_loader` mtime 版本比对——文件改了缓存自动失效，等价 cache-aside 的版本失效策略；② 入库链路 content-hash 增量——内容没变就跳过重向量化，即幂等。**最值钱的一致性 war story**：source-of-truth（.md）与派生索引（chroma_db）的最终一致——通用方案"先建后删"在 content-hash 增量下反而制造缺块（内容未变时 add 会跳过），所以正确顺序是"先删后建 + 失败可见（status=partial + 重跑提示）"。**教训：一致性方案要按存储语义修正，不能照搬模板**。

### Q19: 微服务的超时/重试/熔断，项目里有对应吗？
**A**：全套都有单机版：① `chat_completion` 统一 LLM 网关（`day1_api_starter.py`）——重试 + 指数退避 + 超时 + 可读降级，就是 API 网关弹性三件套；② 工具墙钟预算（`tools_registry`，ThreadPoolExecutor+timeout）——下游超时预算，防一个慢工具拖死整条链路；③ `node_guard` fail-soft——单节点异常不崩全图，等价舱壁隔离（bulkhead）；④ 47 个故障注入测试验证"降级而非崩溃"——混沌工程思想的单机实践。每一条都有测试兜底，可以现场跑。

### Q20: 前端是怎么做的？为什么不用 React / Vue？
**A**：零依赖 vanilla JS：`/ui` 8 卡片求职工作台 + `/ui/console` 8 步 Stepper（前端状态机，随 CareerFlow 节点推进）。**真实的坑**：SSE 流式渲染需要 POST 带 body，但浏览器原生 EventSource 只支持 GET——改用 fetch + ReadableStream 手写流解析。**为什么不用框架**：单页本地控制台引入构建链（node/webpack/依赖树）是过度工程，零依赖 = 克隆即演示。这套是我独立完成的前后端全链路，但我的定位仍是 LLM 应用工程——前端是交付能力的证据，不是方向。

### Q21: 另一个项目 LocalFlow 呢？它跟企业后端有什么关系？
**A**：LocalFlow 对后端出身的面试官甚至更对味——它的每个核心机制都能映射到企业后端的经典设计：
1. **Workspace 四后端统一接口**（Local / Docker / Remote SSH / AgentServer-HTTP）= **适配器模式 + 多环境部署抽象**；AgentServer 用自包含 bundle 注入远端，本质是服务化部署的手工版；
2. **六阶段控制流 + stage 级 checkpoint/resume** = **任务队列的作业状态机**（Celery 任务生命周期同款语义：pending → running → verify → done/rollback，可断点续跑）；
3. **rollback manifest + sha 漂移检测** = **数据迁移回滚 + 校验和一致性**——执行前记录清单，回滚时先校验文件是否被第三方改过；
4. **4 级审批策略 + 路径守卫 + 命令黑白名单** = **权限分级（RBAC 思想）+ 纵深防御**，高危动作必须显式确认；
5. **kernel 边界 + AST 静态 lint 禁止跨层 import + 破例登记 ledger** = **架构治理**（ArchUnit 同款思路）：45 次交付 40 次零 kernel 触碰（88.9%），5 次破例全部留痕；
6. **append-only trace.jsonl** = **审计日志 / event sourcing**——每轮模型决策、工具调用、参数与结果全量留存，支持回放与失败定位。
一句话总结：OfferClaw 把数据层的事务/锁/缓存问题在文件层解决了一遍，LocalFlow 把执行层的隔离/审批/回滚/审计解决了一遍——**两个项目合起来覆盖了后端可靠性工程的读写两侧**。

### Q22: 听说 SSE 被淘汰了？你为什么还在用？
**A**：被弃用的不是 SSE，是 **MCP 的旧 HTTP+SSE 双端点传输**（2025-03-26 规范起由 Streamable HTTP 取代）。SSE（text/event-stream）本身是 WHATWG 现行标准，**OpenAI 和 Anthropic 的 API 今天就是用 SSE 流式返回 token 的**，新的 Streamable HTTP 内部照样可用 SSE 推流。所以我的架构是两层各归其位：①浏览器 UI 的 LLM token 流式走 SSE（与 OpenAI/Anthropic 同款，现行标准）；②**Agent 互操作走 MCP Server，直接按新规范实现了 Streamable HTTP 传输**——单端点 POST /mcp 的 JSON-RPC，手写协议层（initialize/tools/list/tools/call），无状态模式，并按规范 MUST 项做了 Origin 校验防 DNS rebinding。**为什么不做 stdio？** stdio 在规范里仍是一等公民，但它面向本地进程拉起场景；我的服务本体是常驻 FastAPI 应用（46 路由），MCP 挂在同一服务的 /mcp 上才是生产部署形态——传输选型跟着部署形态走。工具层零重复：MCP 与 ReAct Agent 从同一 tools_registry.REGISTRY 取 schema（OpenAI parameters 本就是 JSON Schema，直接映射为 MCP inputSchema），配了 tools/list↔REGISTRY 漂移校验测试。

---

## Q23: 换了生成模型后输出格式变了，你的解析怎么扛？

**A**：这是我踩过的一手事故（2026-08）：主模型从 qwen 切到 gpt-5.6 **当天**，同一份 prompt 生成的学习计划把日块头从 `D1（08-08 周六）` 写成 `### D1（…）`、小节标签变成加粗 bullet——按行首匹配的解析器全部落空，「今日计划」整条静默消失，连日期确定性重写也一并失效。根因：**代码要消费的 LLM 输出就是一个接口，而自由 Markdown 是没有 schema 的接口**——换模型 = 输出格式的分布漂移，在单一模型笔迹上调通的正则是过拟合。修复上我把"保证 JSON 输出"的思路移植过来，做了三层：
① **契约进 prompt**——明写机器解析契约（日期行字样/任务编号样式不可改，Markdown 装饰可容忍）；
② **生成后校验门（fail-visible）**——落盘后立即用解析器数日块，天数随接口/流式 done 事件返回，为 0 前端 toast 警告，绝不静默；
③ **读端容忍 + 风格矩阵回归**——按 Postel 法则容忍已知装饰，5 款笔迹（qwen 裸头 / gpt `###` / 加粗 / 半角括号 / 顿号编号）参数化测试把解析与日期重写两把正则钉在**同一矩阵**上测一致性，新笔迹进矩阵即终身钉死。
**追问：为什么不上 JSON 结构化输出？** 计划是人读文档，叙事段（演进说明/依赖检查）有价值；经代理的 JSON mode 各家支持不一；24 天长输出的转义/截断风险不低；且最关键的字段——日期——本来就由代码确定性重写，不依赖模型自律。若笔迹漂移继续咬人，升级路径是"LLM 出 JSON、代码渲染 Markdown"。
**事故复盘**：`docs/postmortem.md` §2.7。

---

## Q24（反问对方）

可以问招聘官的：
- 团队当前在 RAG 上的主要瓶颈是检索质量、生成质量、工程化，还是评估方法？
- 目标团队的 LLM 应用链路编排使用 LangGraph、自研 DAG，还是 LangChain Expression Language？
- 实习生有机会接触从 0 到 1 的工作流落地，还是更多在已有系统上做局部优化？
- 评估部分（RAGAS / 自定义指标）有没有标准化的内部工具，还是每个项目自己造？
