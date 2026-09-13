# OfferClaw 多 Agent 架构与历史升级记录

> **当前运行版本：Review Workflow v2（2026-09-10）**。
>
> 本页开头是当前权威口径；后续 P0–P4 内容保留为历史设计和实验记录，其中关于
> “Critic LLM 默认关闭”“Writer 只生成项目段”和旧节点名称的描述不再代表 v2 运行行为。

## 当前 v2 架构

只有四个需要独立语言推理并拥有专属输出契约的角色称为 Agent：

| 工作流 | 产物 Agent | 独立 Critic Agent |
|---|---|---|
| 组合计划 | Plan Agent 生成和修改完整 `PlanSpec` | Plan Critic 评审取舍、顺序、可执行性和用户要求 |
| 简历产物 | Resume Agent 按 `full_resume / project_section` 生成和修改 `ResumeSpec` | Resume Critic 按相同范围评审证据、JD 针对性、表达和用户要求 |

上下文组装、Schema 解析、硬规则校验、Markdown 渲染、预算、路由、审批和保存均为工具或
工作流节点，不称为 Agent。`Review` 是工作流，不是第五类 Agent；修改仍回到原产物 Agent，
不设 Reviser Agent。

```
Context tool → Author Agent → Hard Validator tool → Critic Agent
                                      ↑                  │
                                      └── revise once ───┘
                                                   ↓
                                             User approval
```

- 所有可解析产物都必须实际调用 Critic LLM。硬规则即使失败，也会作为输入交给 Critic 汇总；
  Critic 不得覆盖硬规则。模型不可用时标记 `review_unavailable`，不能显示为通过。
- 一次自动周期最多四次 LLM 调用：生成、首次评审、必要时由原 Agent 修改、最终复审。
- 用户默认通过自然语言提出修改要求。Critic 把要求编译进当前轮冻结的 `ReviewContract`，
  原 Agent 修改后使用同一 Contract 复审；完整 Markdown 编辑是高级入口。
- 用户显式选择“以后也遵守”时，记忆服务保存目标范围内的明确偏好；Critic 本身不写记忆。
- 新线程写入 `workflow_version=v2`。已有 v1 审批 checkpoint 继续旧审批路径。

主要契约位于 `review_protocol.py`，图编排位于 `career_multi_agent.py`。所有简历操作都位于
“简历工坊”，界面只有一个 `Resume Agent → Critic` 按钮：用户先选择项目经历或完整简历范围，
项目经历可使用通用模式或真实投递 JD，完整简历必须绑定活动 JD。两个范围都执行硬校验、独立
Critic、至多一次自动修订和人工审批。投递管理只维护投递及活动 JD，不提供简历生成入口。

---

## 历史 P0–P4 蓝图

### 0. 一句话定调

**这不是"从零搭一个多-agent 系统",而是给已经存在且已互联的内环(career_flow routed graph + match_job 规则引擎 + plan_gen + resume Writer + RAG 门控)做统一命名、补两处真缺口(④Critic 独立校验、①JD 结构化)、并用三个预登记实验决定是否升级。** 判断权(三档、拒答、硬门槛)全部留在确定性代码,LLM 只做"文本→结构""素材→表达""语义查编造"这三类可退化的增强,且都能默认关。

---

## 1. 最终组件清单表

| # | 组件 | kind | upgrade_action | 复用来源(锚点) | 复杂度 | 默认状态 |
|---|------|------|----------------|----------------|--------|----------|
| ① | JD 结构化解析 | agent-LLM(可退化) | **light-upgrade**(实为代码去重重构 + 可选 LLM 富化) | `career_flow.job_input_node`、`context_budget.keywords_from`、`rag_gate._chat` | M | **LLM 抽取 default-off**;主链路走 `_jd_struct_fallback` 纯代码 |
| match | 匹配包装层 | **tool-code**(非 agent) | reframe-only | `match_node:209` + conditional edges `:552` + `match_job.decide` | S | 现状(jd_struct 缝默认关) |
| ② | 学习规划 | agent-LLM(单次受约束生成) | reframe-only + 薄新增 | `plan_gen.prepare_plan_messages`、`retrieve_learning_resources` | S | 骨架默认;`skip_llm=False` 才调 LLM |
| ③ | 简历 Writer | agent-LLM | reframe-only | `resume_project.build_project_messages`、`plan_gen.call_llm_stream` | S | 现状(表达内核不改) |
| ④ | 简历 Critic | **agent-LLM(混合体)** | **new-build** | `eval_judge_panel._judge_once`、`profile_loader.load_profile`、黑名单 `resume_builder` | M | 新建;`eval` 达标前 **default-off** |
| ⑤ | CRAG 恢复循环 | **experiment**(起步 tool-code) | new-build | `rag_gate._retrieve_and_classify`、`_evidence_gate`、`rewrite_query/hyde_expand` | M | **`RAG_CRAG=1` 默认关** |
| Sup | 外环 Supervisor | **supervisor-code**(非 LLM) | reframe-only + 薄串行调度 | `build_routed_graph`、`career_agent.compute_execution_tracking`、`memory_layers` | S/M | 现状 |

三本手册纪律延续:任何默认开的 LLM 都必须先过"测正才采纳"eval;否则记诚实负结果(与 HyDE/查询改写/doc2query 同待遇)。

---

## 2. 目标模块结构

### 2.1 新建 / 改动文件清单

```
offerclaw/
├── docs/MULTI_AGENT_UPGRADE.md   [新建·P0 ✅]  本蓝图 + SUPERVISOR 映射台账
├── resume_builder.py             [改·P0 ✅]    _PROJECT_FACTS 数字接 metrics.json 单一真源
├── supervisor.py                 [新建·P1]     薄命名层 + run_supervisor 串行调度
├── jd_parser.py                  [新建·P3]     parse_jd + _jd_struct_fallback(默认路径)
├── resume_critic.py              [新建·P2]     critic_report 三条腿 + verdict 聚合
├── career_flow.py                [改·全程]     CareerState 新增 channel + critic_node + parallel 前置
├── match_job.py                  [改·P3]       run_match 尾部可选 jd_struct 参 + 抽软化词共享 helper
├── plan_gen.py                   [改·P2]       gaps_dict_to_clist 薄序列化器
├── eval_critic_independence.py   [新建·P4-E1]
├── eval_crag_recovery.py         [新建·P4-E2]
└── eval_parallel_multijd.py      [新建·P4-E3]
```

### 2.2 与现有 career_flow 的衔接

- **图位不动**:`job_input` 仍 `add_edge('job_input','router_jd')`;match/gap/plan/today/resume/application_suggest 六节点、三档 conditional edge 保持结构。
- **只加 channel,不改 reducer(P1-P3)**:`CareerState` 新增 `jd_struct: dict`、`plan_md: str`、`critic_report: dict`,均 `TypedDict total=False`。串行阶段无需 reducer。
- **critic_node 插缝**:仅在 `suitable` 路径 `resume → critic → application_suggest`,且只在 `resume_skeleton.mode=='llm'`(真出了 md)时跑,否则透传;`needs_fix` 时在 application_suggest 前打标/降级。包 `node_guard` fail-soft,无 LLM key 时腿2跳过仍返代码腿。
- **P4 才动 reducer**:`trace/errors/requires_confirmation` 改 `Annotated[list, operator.add]`,节点返回 patch-only,`node_guard` 改合并而非 mutate-return。

### 2.3 Supervisor 如实定义(不新建 orchestrator)

`supervisor.py` 里一个 `SUPERVISOR` 映射常量(文档-即-代码),显式声明外环 = 五个**已存在**子系统的组合,每项带 reframe/新增定性 + 锚点:

```
(A) 三档路由       = career_flow.build_routed_graph + _route_after_gap        [reframe]
(B) 预算/checkpoint = career_flow.node_guard + make_budget + resume_career_flow [reframe]
(C) 停滞检测        = career_agent.compute_execution_tracking + ADVICE_ESCALATE_AFTER [light-upgrade: 上提到外环层暴露]
(D) 三层记忆        = memory_layers 全文 + career_flow._learn_from_flow          [reframe]
(E) 节奏触发        = offerclaw_cli 子命令 + OpenClaw cron                        [reframe]
```

`run_supervisor(jds, *, today_iso, ..., skip_llm=True)` = **串行 for 循环**遍历 JD,每份调 `run_career_flow_routed`,收集 per-JD final + 跨 JD 聚合 + escalation。刻意不并行(并行属 P4,且被 CareerState 契约阻塞)。停滞检测上提时**必须与 `get_today_advice` 共享同一 `last_advice` 记录 + 幂等守卫**,避免 `repeat_count` 双 tick。

---

## 3. 分阶段实施顺序

### P1 — 坐实内环 + 命名(近零风险,无新 LLM)
- **改** `career_flow.py`:新增 `plan_md`/`critic_report` channel 占位(空 dict/str,不接线);为 match_node 标 `kind='tool-code'` trace 备注。
- **新建** `supervisor.py`:`SUPERVISOR` 映射 + `run_supervisor` 串行调度 + 停滞检测上提(共享 last_advice)。
- **新建骨架** `jd_parser.py` / `resume_critic.py`:先只实现纯代码腿(`_jd_struct_fallback` 用 `keywords_from`;Critic 腿1 覆盖 + 腿3 黑名单/数字),LLM 腿留降级桩。
- **测**:三档路由映射、跨 JD 串行无串写、预算耗尽跳过、崩溃续跑等价、停滞升级、确定性回归(同批跑两次 route_taken/trace 逐字节一致)。
- **门槛**:pytest 全绿(当前基线 442)。

### P2 — Writer 收口 + Critic 新建(核心真缺口)
- **改** `plan_gen.py`:新增 `gaps_dict_to_clist(gaps)` 薄序列化器(纯代码,补"flow 结构化 gaps 未接进 LLM 规划器"断链);`plan_node` 受 `skip_llm` 门控接 `prepare_plan_messages`。
- **改** `resume_project`:薄封装返回 `{resume_md, mode, origin, material, jd_text, profile}` 结构化契约,供 Critic 消费。**profile 字段统一为 dict**(见 §6.6)。
- **补齐** `resume_critic.py`:腿2(LLM 语义查编造)接 `_judge_once`;verdict 代码聚合。
- **接线** `career_flow`:`critic_node` 插 suitable 路径。
- **测**:Critic 三条腿单测 + 内环集成 + Writer 真值回归(不再出现 118/37/15)。

### P3 — CRAG 条件恢复 + JD 解析(检索/入口)
- **补齐** `jd_parser.py`:`parse_jd` 的 LLM 富化(默认关)。
- **改** `match_job.py`:抽软化词共享 helper;`run_match` 尾部加 `jd_struct: dict|None=None` optional 参(**纯 hint 零位移**,见 §6.1)。
- **改** `career_flow.job_input_node`:guard 后调 `parse_jd`(LLM 默认关);`resume_node/_extract_keywords` 改吃 `jd_struct['keywords']`,`_KEYWORDS` 降级为 fallback allowlist。
- **改** `rag_gate.py`:`_retrieve_and_classify` 加 `_crag_depth=0`,门控后、audience routing 前内联 CRAG 块 + 三 helper。
- **测**:E2 + jd_struct 逐字节相等硬测试(§6.1)。

### P4 — 并行 + 三实验(收官)
- **改** `career_flow.py`:reducer + patch-return 重构(真阻塞,非画布连线);`build_parallel_graph()`。
- **新建** 三个 eval 脚本(见 §5)。
- **顺序**:E2 → E1 → E3(E2 最独立,E3 依赖最重)。

---

## 4. P0 进度(本轮)

- ✅ **4.3 前置 bug 已修**:`resume_builder._PROJECT_FACTS` 的 118 chunks / pytest 37 / 15 路由 / Recall@5 0.96 / MRR 0.74 / bge-small-zh → 运行时读 `metrics.json['current']`(3340 / 442 / 46 / 0.98 / 0.905 / bge-base-zh-v1.5),并去掉不确定的"智谱 GLM-4"具体型号换为"LLM 网关"。pytest 442 全绿,50 简历测试通过。理由:③Writer 长期吃过期事实 = 系统性引用旧数据;④Critic 若信旧数字会误杀正确当前值。
- ✅ **4.1 架构文档已落盘**(本文件)。
- ⏭ **4.2 骨架移至 P1**:jd_parser/resume_critic/supervisor 与其建空壳,不如在 P1 连同纯代码腿一起建(Critic 的 profile 类型 str→dict 统一需在实现时定,见 §6.6)。

---

## 5. 三个决定性实验方案 + 预登记判据

| 实验 | 问题 | 度量 | **预登记 go 判据** | no-go 处置 |
|------|------|------|-------------------|-----------|
| **E1** Critic 独立性 | Writer 自查 vs 独立 Critic 值不值 | Arm A(自查)vs Arm B(独立/跨模型)在 planted-overclaim 上的 `recall` + 对真事实的 `fpr` | **Arm B recall 显著 > Arm A 且 fpr 低** | Critic default-off,记诚实负结果;保留代码腿(黑名单/覆盖)作轻量护栏 |
| **E2** CRAG 救回 | 条件恢复够不够格从 tool 升 agent | held-out 52 题 `recovery_rate`、`r1_tail_delta` | **recovery_rate>0 且三把拒答尺全不降**(abstention 12/12、近似负 ≥11/12、calibrate adv 不塌) | 退回工具,记**第四个诚实负结果**(并列 HyDE/rewrite/doc2query) |
| **E3** 并行多 JD | 扇出并行值不值 reducer 重构 | `speedup=wall_seq/wall_par`、`token_overhead_pct` | **speedup 显著 >1 且 correctness 全等**(并行终态==串行终态、trace multiset 相等) | 若投机 resume token 浪费吃掉收益 → **部分采纳**(不投机),也是诚实结果 |

**共用纪律**:
- E1 语料 = 从 `metrics.json` 派生的干净草稿 + 确定性模板注入编造(ground-truth 由代码保证);**"引用旧数字 118/37/15"本身即一类 overclaim,进 FAB_BANK**。Arm B 跨模型裁判是关键变量(已知自评上浮:qwen 自评 9.92 vs deepseek 交叉 8.83)。
- E2 `RAG_CRAG=0` 必须逐字节等价现状;子查询须在 `_crag_depth==1` 短路防双改写。
- E3 `skip_llm=True` 隔离纯编排开销(token=0),`skip_llm=False` 才测真 token。
- 三实验结果一律写 `docs/rag_eval/roundN/` 或对应台账,真源 = `metrics.json` + REPORT.md。

---

## 6. 诚实边界

### 6.1 身份红线:jd_struct 的自相矛盾 —— **裁决选 (a) 纯 hint 零位移**
若 LLM 能把 regex 漏掉的要求补进 hard_requirements 并触发硬门槛,LLM 就实质参与了裁决,破"规则式/可解释/三档"铁律。**裁决**:jd_struct 只做**重排/标注**,**绝不新增或删除任一硬门槛**,放弃"抓漏硬门槛"卖点。必须加硬测试:**"jd_struct 提供 vs 不提供,在一批 fixture 上 golden match_report 逐字节相等"**。若未来确要让 LLM 参与硬门槛预分类,则如实改标 + 重度 eval 门控,不得再宣称"决不改判定"。

### 6.2 过度 agent 化:①LLM 抽取降为 default-off
①在内环入口主链路、常开、不可关,违背项目自己的"默认关 + 测正才采纳"纪律。**必改**:LLM 抽取 env/skip_llm 门控,`jd_struct` 默认走 `_jd_struct_fallback`(`keywords_from` 纯代码)。只有当 eval 证明"关键词覆盖率提升 / 抓到 regex 漏的质性硬门槛且不回归 match golden"后才提升为常开。

### 6.3 reframe vs 新增(如实台账)
- **真新建 2 项**:④Critic(`resume_critic.py` 确不存在)、①的 LLM 抽取路径(当前 trim-only)。
- **reframe(已存在已互联,只命名/连线)**:match、②plan(plan_gen 早已 LLM)、③Writer(实现皆已 LLM)、Supervisor(A-E 五子系统全已存在)。
- **软性夸大降格**:①原称"唯一无 LLM→有 LLM 的真升级",实质是"去重两份关键词表的代码重构 + 可选 LLM 富化"。如实降格,不抬高。

### 6.4 为什么留代码(不 LLM 化的关键链路)
- **match_job.decide 三档**:唯一必须被信任的裁决,LLM 会注入幻觉 + 抖动 + 破坏 `_route_after_gap` 依赖的确定性 status 字符串。
- **profile / metrics.json 真值**:ground truth 用代码读,绝不让 LLM 凭记忆写 chunks/路由/pytest(本次 bug 教训)。
- **④Critic 的覆盖/黑名单/数字腿 + verdict 聚合**:守"不编造"的防线不能靠会幻觉的 LLM 把关(否则用幻觉守幻觉);LLM 腿只 flag,硬拦交代码。
- **⑤CRAG 循环控制/触发带/深度守卫**:代码化,每次重查重过 `_evidence_gate`(这是与 rag_graph ReAct `should_continue_to_tools` 的分水岭)。
- **Supervisor 全层**:三档路由读规则引擎 status 字符串,零 LLM。

### 6.5 "测正才采纳"落到两项(默认关,达标才开)
1. **④Critic**:先在"注入编造"草稿上证明高召回、低误报(E1),否则 default-off + 记诚实负结果。
2. **⑤CRAG**:先证 held-out 尾部救回 > 0 且拒答不腐蚀(E2),否则退回工具 + 记第四个诚实负结果。

### 6.6 次要修正(审查官指出,已吸收)
- **④Critic profile 类型统一**:`critic_report(..., profile: dict, ...)` 吃 `load_profile()` 出的 dict;Writer 侧 `resume_project` 用的是原文 str——**透传时统一为 dict**,否则 Critic 溯源拿到 str 解析失败。
- **⑤措辞降级**:"边界安全是结构性非经验性" → **"结构性排负样本 + eval 兜底"**。近域对抗负样本改写可能召回 tangential 文档过门控;重过门控只降低非消除风险,真护栏是 abstention eval 硬门(12/12 + 近似负 保留)。
- **④number_check 收紧 scope**:正则严格限定 chunks/pytest/路由/特定百分比,避免误判简历里合法的任意量化数字为 `stale_number` 抬高 fpr。

---

## 7. 面试 / 简历口径(诚实)

> "OfferClaw 我按**双环多-agent**组织:**内环**是 LangGraph 编排的求职专家团——JD 解析、学习规划、简历 Writer、简历 Critic 各是一个带专属上下文的 agent,匹配三档结论我坚持用**规则引擎(工具)**、检索是 RAG 工具,判断权全留确定性代码;遵循**产-查分离**——Writer 只写、独立 Critic 专查编造/证据不足/关键词覆盖,守'不编造'硬边界。**外环**是长期运行的生涯督导:跨天沉淀记忆、检测停滞、驱动节奏。多-agent 不是为用而用——**我只在需要语言推理处用 agent、稳定裁决处坚持用代码**,而且 Critic 与 Agentic-RAG(CRAG)都'测正才采纳',测不出增益就诚实记负结果。"

---

## 8. 实施进度与实验结果（P0–P4）

全部落地并验证:**pytest 442 → 475 全绿零回归、verify_docs 绿**。

| 阶段 | 交付 | 验证 |
|---|---|---|
| **P0** | 架构文档 + 前置 bug（`_PROJECT_FACTS` 接 metrics.json 单一真源） | pytest 绿、verify_docs 绿 |
| **P1** | `jd_parser` / `resume_critic` 纯代码腿 + `supervisor` 台账 + 串行多JD | 15 单测 |
| **P2** | Critic 接进 CareerFlow（两图 `resume→critic→application_suggest`）+ 语义腿（默认关） | 9 单测;默认透传零行为改变（e2e 验证） |
| **P3** | CRAG 恢复循环（`RAG_CRAG=0` 逐字节等价）+ JD LLM 富化（默认关）+ jd_struct（**不喂 match_job**） | 8 单测 + CRAG 冒烟 + 全量套件 |
| **P4** | 并行多JD（`supervisor`,correctness==serial）+ 三实验脚本 | 1 单测 + E3/E1 实跑 |

**决定性实验（预登记判据,测正才采纳）**:
- **E3 并行**（`eval_parallel_multijd.py`）:correctness **PASS**（并行终态==串行终态,逐字段相等）;`skip_llm=True` 下 speedup **0.98x**——CPU 密集受 Python GIL 限制,线程并行不提速属**预期且诚实**;真墙钟收益在 `skip_llm=False`（LLM I/O 密集释放 GIL）,需 key 另测。
- **E1 Critic 代码腿**（`eval_critic_independence.py`）:注入编造召回 **4/4**、干净草稿误报 **0/4**（代码腿结构上独立于写作者）。LLM 自查 vs 独立的跨模型正式对照需 key,另跑（参照裁判团方法）。
- **E2 CRAG 救回**（`eval_crag_recovery.py`）:脚本就绪;held-out 52 题 + reranker 双跑属机器吃紧项,建议异机/挂机单进程跑（与 P3/P4 codex 包同待遇,跑前 pgrep 查场）。

**当前默认状态（全部保守,测正才采纳）**:JD LLM 富化 `JD_PARSE_LLM=0`、Critic 语义腿仅 `skip_llm=False` 开、CRAG `RAG_CRAG=0`。**常开且已验证的是纯代码腿**（Critic 覆盖/黑名单/数字对账、jd_parser fallback、supervisor 编排、匹配规则引擎）。E1-LLM / E2 测正后方可把对应 LLM 腿默认开;测不出增益则记诚实负结果（并列 HyDE / 查询改写 / doc2query）。

### 8.1 实现对原蓝图的诚实偏差（可审计）

实施中有几处**为更稳/更守身份**偏离了 §2–§5 的原计划,如实记录:
1. **match_job 完全不改**（原计划"加 jd_struct optional 参 + 零位移硬测试"）→ 实际**连参数都不加**,jd_struct 根本不传给 match_job。这样"LLM 决不碰硬门槛"是**结构性保证**(match 连看都看不到),比"传了但保证零位移"更干净。硬测试改为 `test_match_job_never_sees_jd_struct`（签名无 jd_struct 参）。
2. **`_extract_keywords` 保持不变**（原计划"改吃 jd_struct['keywords']"）→ curated allowlist 更适合"简历强调哪些技术";jd_struct 改由 **Critic 覆盖检查**消费(更合适的下游)。
3. **并行用 `ThreadPoolExecutor` 跑独立 JD**（原计划"reducer + patch-return 重构 + build_parallel_graph()"）→ 多JD 并行是**独立 state 的并发 invoke**,无跨 JD 共享可变量,**根本不需要 reducer 重构**（那是"单 JD 内并行分支"才需要的更难改造）。更简、更安全,correctness==serial 已单测。
4. **`plan_gen.gaps_dict_to_clist` 未做**（原计划 P2 薄序列化器）→ ②plan 属 reframe（plan_gen 早已 LLM),该序列化器非必需、未接线,不影响 Critic/并行。**如实标注:未实现**,留待真需要时补。
