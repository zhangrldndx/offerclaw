# OfferClaw · Loop Engineering 优化手册

> 第三本工程手册（前两本：`RAG_OPTIMIZATION_LOG.md` 检索侧、`AGENT_OPTIMIZATION_LOG.md` 生成/编排/记忆/工具侧）。
> 本手册聚焦 **loop engineering**——刻意设计 agent 的循环本身：内循环（单任务 ReAct）与外循环（跨天/跨会话长程编排）的
> 终止条件、上下文喂回与压缩、出错恢复、循环级预算、可恢复续跑。
>
> **本文件 = 第 0 步路线图（盘点 + 规划，尚未开做）**。逐项开做时沿用既有五步流程：
> ①按价值优先级选一项 → ②落地并标注改动文件/函数 → ③量化（见下）→ ④对应知识库八股 → ⑤记入本手册。

---

## 核心洞察（为什么这个方向对 OfferClaw 价值最大）

1. **OfferClaw 本质是一个 outer-loop agent**——长期运行、带状态、按天/周唤醒的求职养成 agent。
   大多数 RAG/agent demo 只有 inner loop（单次问答/单次工具循环）；**能把 outer loop 做扎实是大厂面试的稀缺差异化项**。
2. **量化方法**（沿用 Agent 手册的方法论）：循环/生成路径没有 Recall 那样的天然标尺，用两把尺子——
   **① 故障注入**（注入死循环 / 超长上下文 / 中途 kill / 预算超限 / 矛盾反思，证明降级而非崩溃）；
   **② 上下文占用**（注入 LLM 的 token 量、是否触顶、压缩前后对比）。
3. **知识库八股落点扎实**：飞书 KB 已有完整 harness engineering 章节，每项改造都能对标——
   `llm_app_interview_10_harness_engineering.md`（Harness 5 层：Context Injection / Control / Action / Persist / Observe&Verify）、
   `_11_harness_core_workflow.md`（agent loop 运行结构）、`_12_harness_scenarios.md`（失败模式：长任务接力 / context rot / 虚假完成 / 状态丢失 / 时间预算失控）、
   `_04_agent_basics.md`（ReAct / Reflexion / Memory）、`_06~08_agent_planner`（L1–L5 规划演进）、`hello_agents 精选`（第八章记忆 / 第九章上下文工程）。

---

## 现状盘点（基于真实代码，带 file:line 证据）

> 诚实标注：很多循环卫生在 A 系列改造时已顺手做了。本手册做的是**补 delta**，不是从零造。

### 维度 ① 内循环 Inner Loop —— `react_agent.py`

- **已有**：ReAct 主循环 `for step in range(max_steps)`（默认 3，`react_agent.py:189`）；`completed` flag 区分「模型主动结束」vs「被掐断」并记 `max_steps_reached`（A4，`:246-251`）；malformed tool_call 防护（空 fn_name 占位配对 / JSON 解析失败 args={} / 工具不存在 error dict，`:220-224`）；工具结果截 2000 字符拼回（`:240`）；无 key 降级 deterministic。测试 `tests/test_agent_tool_robustness.py`。
- **缺口**：单轮 messages **全量累积、无滑动窗口/token 预检**（`:180-242`）→ 名义 max_steps 越高越易爆窗，实际可用远低于名义；工具结果**硬截非智能摘要**（截断的 JSON/表格丢关键字段）；LLM/tool 失败**无重试**（瞬时抖动即丧失机会）；无单轮 timing/token 开销统计。

### 维度 ② 外循环 Outer Loop + 上下文接地 —— `career_agent.py` / `plan_gen.py` / `summary_tool.py` / cron

- **已有亮点**：`plan_gen.prepare_plan_messages()`（`:676-740`）是 CLI+FastAPI **统一入口**；daily_log **已做窗口化压缩**——近 14 天明细 + 更早按周一行摘要（`digest_history`，`:537-598`、`:690-706`）；复盘沉淀的「次日调整规则」从 semantic 记忆回流注入计划（`:708-716`、`:316-322`，标「必须遵守」）；cron 三任务（工作日 09:00 今日建议 / 22:00 复盘提醒 / 周日 21:00 周度复盘，`setup_openclaw_cron.sh`）。
- **缺口**：① **无「上次到哪了」检查点**——每次唤醒都是 cold start，不知前序建议有没被执行 → 重复建议、建议无递进、复盘无法对标「上次建议 vs 实际执行」（`career_agent.py:181-281`）；② `user_profile` / `applications` / `interview_story_bank` **无压缩**（仅 daily_log 压了），applications **全表扫 O(N)**（`career_agent.py:38-69`），多月后 system_content 50KB+ 对 4K 窗口模型爆窗；③ **无 re-grounding**——用户改 profile 后当天 `/api/today` 仍用旧缓存；④ `resume_builder` profile **硬截 3000 字**（`:77`）丢后段章节；⑤ 压缩掉的历史周摘要**未入向量库**，后续无法 query 回溯；⑥ `prepare_plan_messages` 拼接**无 message size 上界检查**。

### 维度 ③ 反思环 + 记忆回流 —— `summary_tool.py` / `memory_layers.py`

- **已有亮点**：反思**已是 generate→verify→revise**——`extract_json_with_repair`（A6，`:269-286`）首轮 JSON 失败时只要求重发 ```json 块（reflexion 自纠），非全量重跑；Semantic 记忆**读写闭环**（`distill_reflections_to_semantic` 写 → `plan_gen` 注入读，`:328`/`:710`）。
- **缺口**：① **Procedural 记忆完全架空**——`memory_layers.py:204-247` 定义了 SOP/启发式层但**零写零读**（无任何 `.add()`/消费路径）；② 反思的 verify **只在 JSON 结构层、不在内容逻辑层**——「偏离度 0 但列 10 项未完成」这类矛盾无校验；③ Semantic 规则是**文案软建议**非结构化硬约束，LLM 可忽视；④ **Episodic 未回流决策**——`episodic.jsonl` 全量事件流只用于成长日志展示（`profile_evolution.py:54-58`），计划生成只读 semantic 蒸馏摘要，丢失事件级教训。

### 维度 ④ 循环级预算 + 可恢复 —— `tools_registry.py` / `career_flow.py`

- **已有**：**工具级**墙钟预算 `TOOL_TIMEOUT`（A7，`tools_registry.py:55-74`，ThreadPoolExecutor+timeout，超时返 error 不阻塞）；ReAct `max_steps` 次数限制（A4）；CareerFlow **node_guard 单节点 fail-soft**（`career_flow.py:77-93`，单节点异常记 errors 继续流转）；gap_store/applications/memory 原子写（`io_utils.py:17-37`）。
- **缺口**：① **无外循环预算**——CareerFlow `graph.invoke(state)` 无 step/token/wall 总预算（`:481-493`）；② **无中间检查点**——state 仅存内存，崩溃无法恢复；③ **无 replay/resume**——中断后只能重跑全流程；④ node_guard fail-soft 止于单节点，**无流程级降级**（gap 节点失败后 plan/resume 仍按标准路径跑）；⑤ CareerFlow 跑完**不写 episodic**，长程学习看不到跨 JD 演进。

---

## 改造路线图（按价值优先级，逐项开做）

> 每项含：动机 / 改动定位 / 量化方式 / 对应八股 / 简历话术。⏳=待做。

### ✅ L1 — 外循环执行追踪（"上次到哪了" checkpoint）【最高价值】

- **动机**：补维度②缺口①。agent 每次唤醒 cold start，建议无记忆、无递进、复盘无法对标执行——这是 outer-loop agent 最核心的短板。
- **改动定位**：`career_agent.get_today_advice()`（`:181-281`）每次产出建议时写 `memory/last_advice.json`（advice_id + 内容 + 状态）；下次唤醒先读上次建议、对比 daily_log 完成度，决定「续上 / 换新」；`memory_layers` 的 Semantic 存 `last_advice_id` + `execution_status`；cron 任务回传 advice_id。
- **量化**：故障注入式——连续模拟 N 天唤醒，测**重复建议率**（同一建议被反复给出的比例，应↓）+ 建议**递进性**（是否引用前日执行）；对比改造前后。
- **对应八股**：`_12_harness_scenarios.md`（长任务跨 context window 接力：progress file / handoff note / 结构化状态记录）；`_10_harness_engineering.md` 的 **Persist** 层。
- **简历话术**：「为长程 outer-loop 设计**执行追踪检查点**（每次唤醒读上次建议 + 对比实际完成度），消除跨天重复建议、让多日决策有递进」。
- **✅ 完成（2026-06-22）**：
  - 改动：`career_agent.py` 新增纯函数 `_advice_signature` / `compute_execution_tracking` + 常量 `_ADVICE_KEY` / `ADVICE_ESCALATE_AFTER`；`get_today_advice(sem=, today=)` 加执行追踪块——读 `SemanticMemory` 的 `last_advice`，同签名且无推进则 `repeat_count` 累加，达阈值 `escalated=True` 并 insert「【执行追踪】」升级动作（换更小切入口），返回新增 `execution_tracking` 字段。持久化复用 `memory_layers.SemanticMemory`（原子写 + 可注入 base_dir，便于测试）。
  - 量化（故障注入，`tests/test_loop_l1_advice_tracking.py` 7 测全过）：冻结状态连续 5 天唤醒，**「盲目未感知重复」从 4 天 → 被阈值封顶的 2 天**，第 3 天起 `escalated=True`、建议升级不再原样重复；状态变化 → `repeat_count` 归零。相关 54 测零回归。

### ✅ L2 — 检索式上下文压缩 + re-grounding【高价值，复用你的 RAG】

- **动机**：补维度②缺口②③④⑥。daily_log 已压，但 profile/applications/story_bank 全量、无总预算、改 profile 不刷新。
- **改动定位**：`prepare_plan_messages()`（`:676-740`）对 profile/applications **按 JD/缺口相关性筛选**（复用 `retrieve_learning_resources` 范式 `:156-228`，或关键词抽章节）；`resume_builder.py:70-96` 把硬截 3000 改为 JD-profile 相关性抽取；加 **message size budget**（粗算 token，超预算按 plan_prompt>rules>profile>daily_log>resources 逐块降级，返回 `{budget_ok, actual_size, warnings}`）；`get_today_advice()` 集成 `cmd_refresh_state`（`:274-327`）做 lazy re-ground（每早/改档后刷新）。
- **量化**：**上下文占用**——同一任务改造前后注入 LLM 的 token / 字符数（应↓且稳定）、是否触顶；故障注入超长 daily_log（模拟 6 个月）验证不爆窗。
- **对应八股**：`hello_agents 精选`第九章上下文工程；`_12_harness_scenarios.md` 的 **Context Rot**（压缩 / 卸载 / context reset）。
- **简历话术**：「用 **RAG 式上下文压缩**把长程 agent 的注入上下文从全量 dump 改为相关性切片 + 预算降级，多月运行 token 占用从 X 降到 Y、不再触顶」。
- **✅ 完成（2026-06-22）**：
  - 改动：新增可复用件 `context_budget.py`——`fit_to_budget(blocks, max_chars)`（按优先级降级，**预算闸保证 total_after ≤ max**）、`select_relevant_sections(md, keywords, max_chars, always_keep)`（按 markdown 章节相关性挑、还原原文顺序）、`keywords_from`（英文 token + jieba 中文切词，复用 BM25 同款分词）。接入：`resume_builder.build_messages` 把硬截 `profile[:3000]`（无声丢后段章节）→ JD 相关性挑章节 + 5000 字预算；`plan_gen.prepare_plan_messages` 对 profile 超量时按 gaps/方向相关性压到 `OFFERCLAW_PROFILE_BUDGET_CHARS`（默认 6000，小 profile 原样不变）。
  - 量化（`tests/test_loop_l2_context_budget.py` 8 测全过）：故障注入 **50KB profile → 压到 ≤5000 字符**（改造前 = 全量 50KB 爆窗）；`fit_to_budget` 高优先级整保、低优先级截断、预算闸恒成立；resume_builder 验证 JD 相关章节保留、无关大章节被压。resume/plan 相关 59 测零回归。
  - re-grounding 说明：career_agent 的外循环**本就每次 `_read` 现读文件、无陈旧缓存**（已天然再接地）；向量库的再接地由现有 `cmd_refresh_state` 负责。故 L2 净增 = 压缩 + 预算 + 相关性选择。

### ✅ L3 — 外循环预算 + checkpoint/resume【高价值，最"工程"】

- **动机**：补维度④缺口①②③。CareerFlow 无总预算、崩了重跑全流程。
- **改动定位**：`run_career_flow_routed`（`:481-493`）入口加 `BudgetContext`（step/token/wall），每节点后检查决定是否继续；关键节点（gap/plan/resume）后把 `CareerState`（`:40-59` 已是完整 TypedDict）快照落盘（`.offerclaw/checkpoints/{run_id}_{node}.json`，复用 `io_utils` 原子写）；新增 `resume_career_flow(run_id, from_node)` 读检查点续跑。
- **量化**：**故障注入**——跑到第 k 节点 kill → resume 成功率；注入预算超限 → 是否优雅中止记 `budget_exhausted` 而非空转；前序节点失败 → 流程级降级是否跳过依赖分支。
- **对应八股**：`_12_harness_scenarios.md`（时间预算失控、状态丢失）；`_10/11_harness`（Control / Persist 层、resumable execution）。
- **简历话术**：「为 8 节点 CareerFlow 加 **step/wall 预算 + 节点级 checkpoint + 断点续跑**，用**故障注入**（中途 kill、预算超限）验证恢复成功率与优雅降级」。
- **✅ 完成（2026-06-22）**：
  - 改动（`career_flow.py`）：`CareerState` 声明 `_budget`/`_run_id` 两个 channel（否则 LangGraph 不跨节点传）；`node_guard` 扩展为 **fail-soft + 预算闸 + checkpoint** 三合一——预算耗尽则跳过节点记 `skipped_budget_exhausted`（优雅降级），否则执行后计步 + 原子写 state 快照；新增 `make_budget(max_steps, max_wall_s)` / `_budget_exhausted` / `_checkpoint`（复用 `io_utils.atomic_write_json`）/ `load_checkpoint` / `resume_career_flow(run_id)`（从最后检查点 replay 续跑）；`run_career_flow_routed(..., budget=, run_id=)` 新增可选参数。checkpoint 落 `.offerclaw/checkpoints/`（已 gitignore）。
  - 量化（故障注入，`tests/test_loop_l3_budget_resume.py` 7 测全过）：① `max_steps=3` → match 后下游 plan/today/resume **优雅跳过**（`plan_outline` 缺失、trace 有 `skipped_budget_exhausted`）；② 模拟崩溃只剩 checkpoint → `resume_career_flow` **续跑到完成**（`plan_outline` 恢复）；无 checkpoint → `FileNotFoundError`；不传参 → 行为同改造前。career_flow 相关 27 测零回归。
  - 诚实标注：预算实现 **step + wall 两维**；token 维需真实 LLM token 计数，而 CareerFlow 默认 `skip_llm=True` 无 token 流——token 预算见 L5（内循环 ReAct 有真实 LLM 调用）。

### ✅ L4 — Procedural 记忆激活（SOP 闭环）【中价值，补完三层记忆】

- **动机**：补维度③缺口①。三层记忆只活了两层，procedural 架空——"学到的操作规范"无处沉淀复用。
- **改动定位**：`summary_tool` 复盘后把可复用 SOP/启发式（如「投 RL 岗强调 PyTorch」）写入 `ProceduralMemory.add()`（`memory_layers.py:204-247`）；`plan_gen`/`career_agent` 决策时查询并注入；CareerFlow 跑完写 episodic 事件（`kind=career_flow_run`）让长程学习看到跨 JD 演进。
- **量化**：故障注入 + 命中——构造重复情境，测 SOP 是否被复用（命中率从 0→N）；episodic 事件流是否完整记录跨 JD 演进。
- **对应八股**：`_04_agent_basics.md` Memory 三划分（结构/格式/操作）；`hello_agents` 第八章记忆与检索。
- **简历话术**：「打通三层记忆闭环——把架空的 **procedural 层**接入复盘沉淀 SOP→规划复用，agent 跨任务学到的操作规范不再丢失」。
- **✅ 完成（2026-06-22）**：
  - 改动：`memory_layers.py` 新增 `record_career_flow_run`（写 `kind=career_flow_run` 事件）、`distill_procedural_sops`（某方向累计 ≥min_support 次「适合」→ 确定性沉淀 `apply_direction:{方向}` SOP）、`get_active_sops(proc, context)`（trigger `key=value` 命中 context 即返）；`career_flow.py` 新增 `_learn_from_flow`，在 `run_career_flow` / `run_career_flow_routed` 末尾写经验 + 沉淀 SOP（best-effort）；`plan_gen.prepare_plan_messages` 按方向查 SOP 注入到 `adjustments_block`（让"学到的操作规范"作用到排期）。
  - 量化（`tests/test_loop_l4_procedural.py` 7 测全过）：**架空基线 `proc.list()==[]` → 同方向 2 次「适合」后沉淀出 1 条可复用 SOP（命中数 0→1）**；min_support 未达 / 「不适合」不计；`get_active_sops` 按 context 精确过滤；`_learn_from_flow` 写 episodic + 沉淀闭环验证。memory/career_flow/plan 相关 48 测零回归。
  - 闭环：episodic（写经验）→ procedural（沉淀 SOP）→ plan_gen（按方向注入决策）——三层记忆从「只活 2 层」补成全闭环。

### ✅ L5 — 内循环上下文预算 + 智能截断【中价值】

- **动机**：补维度①缺口。单轮 messages 无界增长、工具结果硬截。
- **改动定位**：`react_agent._llm_step()`（`:180-242`）每步估算 messages token，余量 <20% 提前中止记 `budget_exhausted`；工具结果改**结构化优先保留**（dict 体积过大时保 error/status/summary 关键字段再截非关键，`:230-241`）；可选加 LLM/tool 有限重试（2-3 次 backoff，复用 A1 网关退避）。
- **量化**：**故障注入**——注入高 max_steps + 大工具返回，验证是否在 budget 处提前中止而非第 4 轮 LLM 才崩；截断策略对关键字段保留率。
- **对应八股**：`_12_harness_scenarios.md` context window 管理；`_05_agent_tool_calling.md` 工具返回处理。
- **简历话术**：「给 ReAct 内循环加 **token 预算预检 + 工具结果结构化截断**，避免名义 max_steps 因上下文膨胀而虚高」。
- **✅ 完成（2026-06-22）**：
  - 改动（`react_agent.py`）：新增 `REACT_MAX_CTX_CHARS` / `REACT_TOOL_RESULT_LIMIT`（env 可配）、`_messages_chars`、`_truncate_tool_result`（dict 优先保 error/status/summary 等关键字段，超限或非 dict 退回尾截）；`_llm_step` 每步前预算预检超限则 `ctx_budget_exhausted` 提前止损，工具结果改用智能截断替换硬 `[:2000]`。
  - 量化（`tests/test_loop_l5_inner_budget.py` 7 测全过）：**关键 before/after——`{"data": "x"*5000, "error": "CRITICAL"}` 硬截 `[:2000]` 会丢掉 error，智能截断保留 error+status 且 ≤limit**；预算闸可检出超窗；接线源码断言（`ctx_budget_exhausted` + 智能截断已替换硬截、旧 `[:2000]` 已移除，沿用 A4 react_agent 的 mock 成本规避模式）。agent/tool 相关 29 测零回归。
  - 诚实标注：`_llm_step` 走 requests 直连 + `get_llm_config`，HTTP 全链路 mock 成本高，故 budget 止损的**端到端**行为用「纯函数 + 接线断言」覆盖（与现有 `test_agent_tool_robustness` 对 react_agent 的策略一致），未做真实 HTTP 注入。

### ✅ L6 — 反思二阶 verifier + episodic 回流【中价值，深化自纠环】

- **动机**：补维度③缺口②④。反思只验 JSON 格式不验内容逻辑；episodic 只写不回流。
- **改动定位**：`build_structured_reflection()`（`:289-316`）后加**内容一致性 verifier**（偏离度 vs completed/incomplete 比例对账、完成数与日志对账，矛盾则回退/标记）；`plan_gen` 注入 daily_log 时附「episodic 最近 3-5 条反思事件摘要」让 LLM 感知具体历史教训。
- **量化**：**故障注入**——构造「格式完美但逻辑矛盾」的反思，验证 verifier 拦截率；episodic 事件回流后计划是否引用历史教训。
- **对应八股**：`_04_agent_basics.md` Reflexion；`_12_harness_scenarios.md`「虚假完成」检测。
- **简历话术**：「把复盘的 reflexion 自纠从**格式自纠**升级到**内容自纠**（二阶 verifier 对账逻辑一致性）+ 事件级记忆回流」。
- **✅ 完成（2026-06-22）**：
  - 改动：`summary_tool.py` 新增 `verify_reflection_consistency`（校验 deviation_score 与未完成比例自洽，不一致以确定性比例值纠正），`build_structured_reflection` 产出后过 verifier、纠正 score 并记 `_verifier`；`memory_layers.py` 新增 `recent_reflection_lessons`（取最近反思事件简短教训）；`plan_gen.prepare_plan_messages` 把教训回流注入 `adjustments_block`（episodic 此前只写不读）。
  - 量化（`tests/test_loop_l6_reflection_verifier.py` 6 测全过）：**故障注入「格式完美但矛盾」反思（deviation 0 + 5 项未完成）→ verifier 拦截并纠正为 83**；高分零未完成 → 纠正为 0；一致反思不误标；`recent_reflection_lessons` 只回流有未完成项的复盘。summary/reflection/plan 相关 41 测零回归。

---

> **进度：L1–L6 全部完成（6/6）。** 下方为对应八股映射 + 与既有 track 的关系。

---

## 对应八股映射表（改造项 × 知识库 harness 章节）

| 改造项 | 主要对应 KB 八股 |
|---|---|
| L1 执行追踪 checkpoint | `_12_harness_scenarios`（接力/progress file/handoff）· `_10_harness` Persist 层 |
| L2 上下文压缩 + re-ground | `hello_agents` 第九章上下文工程 · `_12` Context Rot |
| L3 外循环预算 + resume | `_12`（预算失控/状态丢失）· `_10/11` Control·Persist·resumable |
| L4 Procedural 记忆 | `_04_agent_basics` Memory 三划分 · `hello_agents` 第八章记忆 |
| L5 内循环 token 预算 | `_12` context window · `_05_agent_tool_calling` |
| L6 反思二阶 verifier | `_04` Reflexion · `_12` 虚假完成 |

> KB 覆盖深度自评（勘察结论）：ReAct/inner loop **深**、失败模式 **高**、Planning(L1-L5) **中**、outer loop 系统框架 **中**（有零散失败模式与解法，缺统一 outer-loop 架构论述——这正好可由本项目的实战反哺补进 KB）。

---

## 优先级建议与排期

- **先做 L1 + L2**：价值最高、最能体现 outer-loop 差异化、且 L2 直接复用你已成熟的 RAG 检索能力（一鱼两吃）。
- **再做 L3**：最"工程"、故障注入量化最漂亮，是简历上「可恢复执行」的硬通货。
- **L4–L6 收尾**：补完记忆闭环与自纠深度。

> 与既有 track 的关系：loop engineering 是 **Agent track（A1–A8）的纵深延伸**——A 系列做的是「单点弹性容错」，本 track 做的是「把这些点织进 inner/outer 两层循环的系统设计」。RAG track 的检索能力是 L2 压缩的现成弹药。
>
> **当前状态：L1–L6 全部完成（2026-06-22），各轮均「改动 + 故障注入量化 + 入册」。**
> 新增 6 个测试文件、42 个循环工程测试，全量 **396 passed** 零回归。可复用件 `context_budget.py` 已落地。
> 后续可选纵深：真实 LLM token 计数的内循环预算端到端注入测试、outer-loop 多 JD 批处理的预算编排、把本项目实战的 outer-loop 设计反哺补进 KB 的 harness 章节。

**推送前对抗审计加固（2026-06-22，4 视角并行复查后修）**：
1. **L5 `_truncate_tool_result` 严格 ≤limit**：原 `full[:limit] + marker` 因 marker 约 6 字符实际超限——改为 `full[:limit-len(marker)] + marker`，测试去掉 +10 容差改严格 `≤limit`。
2. **L3 resume 幂等**：原 `resume_career_flow` 从头重放会重算 match/plan/resume（非确定性场景结果可能不一致）——给 `match_node`/`plan_node`/`resume_node` 加「已算过则跳过」幂等闸，resume 变为「跳过已完成节点真续跑」而非全量重放。
3. **L1 测试加强**：显式断言 `repeat_count==[0,1,2,3,4]`（证明计数持续累加不归零、封顶行为可见），escalation 持续性。
4. **L6 测试加强**：新增端到端——纠正值（83 而非原错值 0）确实落库 episodic 并被回流教训取出。
全部修复后 **397 passed** 零回归。其余审计 warning（部分测试可更强 / fit_to_budget 前置条件未校验 / SOP 注入静默失败）评估为可接受，留作后续。
