# OfferClaw 智能语义路由与统一 JDAnalysis

## 运行架构

顶部问答采用“一次结构化语义规划 + 双注册表校验 + 只读执行”：

```text
所有用户自然语言
  → SemanticQueryPlan v4（一次 LLM 判断语言行为、回合关系与服务模式）
  → ReadSourceRegistry / ActionCapabilityRegistry 校验
  → QueryPlan DAG
  → 多源检索与 Evidence Gate

系统生成的结构化只读请求
  → plan_structured_read（0 embedding / 0 LLM）
```

生产路由不会先按状态、日期或 ID 猜测语言行为。写命令必须形成结构化
`action_request`，并只映射一个实际 UI 能力；顶部问答只渲染操作指引，不调用写 API。
在线入口没有规则回滚或 shadow 答案路径；旧规则函数仅供离线基线评测显式调用。

语言行为与回合关系使用正交字段。`interaction_kind=query|command|how_to` 表示本轮要做
什么；`turn_relation=standalone|continuation|correction` 表示本轮与上一成功回合的关系。
`relation_target_turn_id` 只能由服务端根据会话状态填写，模型不能生成或猜测该 ID。
独立新主题不会把上一轮实体送入模型；只有 continuation/correction 才在规划后绑定候选回合。

服务端按 `conversation_id` 保存成功回合的输出契约、真实实体 ID 和能力 ID。代词只绑定
最近的类型兼容回合；前端不再发送最近三条原始问题。会话状态 24 小时失效，清空对话
不会删除长期记忆。

## 服务模式

- `guide`：解释 OfferClaw 卡片入口、所需输入和边界，不执行写入。
- `recall`：读取用户已保存的画像、计划、投递、项目或复盘。
- `explain`：优先检索策展知识库；证据不足后才由 Evidence Gate 标注通用回答。
- `advise`：组合必需的个人证据与参考资料，输出只读个性化建议。
- `diagnose`：解释数据缺失、未审批、未关联或检索不可用。

规划遵循“最小充分来源”：一个答案对象默认只选一个最专门来源；只有用户明确要求
结合、核验或后续任务依赖前序实体时，才增加来源和 DAG 依赖。

## JDAnalysis

`jd_parser.analyze_jd()` 是 JD 语义分析的唯一入口。它输出带版本和证据位置的：

- 公司、岗位、地点与岗位性质；
- 职责和结构化要求；
- `required / preferred / alternative / context` 措辞；
- 关键词、权重、原文 surface form 与 requirement 关系。

所有正式要求和关键词必须能定位到 JD 原文；无原文证据的模型输出会被代码丢弃。
缓存键包含内容哈希、契约版本和模型。同一 JD 的 CareerFlow、匹配、简历 Writer、
Resume Critic、JD 预览和旧工具均消费同一份分析，不再各自维护技术白名单。

LLM 的 `kind=hard` 只是候选分类。学历、年限、地点、语言等最终硬门槛仍由
`match_job.py` 对原文措辞和用户画像做确定性核验，模型不能直接淘汰投递。

## JD 要求与画像证据对齐

`semantic_matcher.align_requirements()` 补足词面匹配无法处理的跨表达问题，例如 JD 的
“建立检索质量评估体系”与项目中的“用 Recall@K/MRR 分析召回失败”。它是受约束的
语义服务，不是 Agent，也不输出岗位结论：

1. `profile_loader` 将技能、项目、实习、科研和竞赛原文拆成稳定 `evidence_id`；未来计划、
   已知缺口、占位文本和项目数量不作为能力证据。
2. 模型逐项返回 `requirement_id → evidence_id[]`，关系只能是 `direct / transferable /
   partial / unsupported / contradicted`。
3. 代码拒绝不存在的引用；技能名称不能单独证明项目/工作经验；硬门槛不进入此模型。
4. `match_job.py` 只把通过校验的关系用于“技能重叠”和“项目契合”，再由原有规则计算三档。
5. 模型不可用或输出无效时返回 `semantic_status=degraded`，继续确定性匹配；UI 明确显示降级。

`POST /api/match` 的 `use_semantic=true` 只增加一次 LLM 调用。在线匹配使用代码生成的原文
JD 要求，把模型预算用于证据对齐；独立 JD 分析和完整 CareerFlow 仍可按配置使用智能
JD Parser。对齐缓存同时包含 JD 要求哈希、画像证据哈希、模型和契约版本，画像变化后不会
复用旧结果。

## 配置

```dotenv
RAG_ROUTE_MODEL=
RAG_ROUTE_TIMEOUT_SECONDS=60

JD_ANALYZER_MODE=deterministic|shadow|intelligent
JD_ANALYSIS_MODEL=
JD_ANALYSIS_TIMEOUT_SECONDS=30

MATCH_SEMANTIC_MODEL=
MATCH_SEMANTIC_TIMEOUT_SECONDS=30
MATCH_SEMANTIC_REASONING_EFFORT=low

# 可选：让结构化小模型直接复用 LLM_FALLBACK_* 端点
STRUCTURED_LLM_USE_FALLBACK_ENDPOINT=0
```

顶部自然语言固定等待一次 GPT 语义规划并执行其
通过白名单校验的计划；JD 智能分析仍保持影子模式。当前阶段优先保证答案对象与
来源正确，允许语义规划等待最多 60 秒。
若 GPT 规划失败，默认返回可见澄清问题。路由原型只用于离线评测，不具备在线执行资格。

## 验证命令

```bash
.venv/bin/python eval_intelligent_router.py \
  --mode intelligent --output /tmp/intelligent_router.json

JD_ANALYZER_MODE=intelligent .venv/bin/python eval_jd_analysis.py \
  --mode intelligent --output /tmp/jd_analysis.json

JD_ANALYZER_MODE=deterministic .venv/bin/pytest -q
```

冻结集禁止通过话术外壳批量扩写。真实误路由可以进入评测集，但不得为单题增加
关键词分支；修复必须落在答案对象、来源角色、操作契约或证据边界上。

## 2026-08-25 泛化审计与启用门槛

当前生产入口固定使用结构化语义规划。自由文本中的 ID、日期和状态不再绕过语言行为判断；
只有受信系统调用 `plan_structured_read()` 才走零模型路径。环境变量不能把用户流量降级到
旧规则实现。

现有证据必须分层解读：

| 证据 | 当前结果 | 能证明什么 | 不能证明什么 |
|---|---:|---|---|
| `router_surface_invariance_v1` | 480/480，无失败 | 已知 48 个语义种子在 10 种话术外壳下保持稳定 | 开放语言泛化 |
| `router_reviewed_regression_v1` | Source F1 99.58%，Route F1 99.17%，6/240 失败 | 已标注开发边界基本稳定 | 发布门槛已通过；其中关键投递状态召回仅 95.12% |
| 历史 shadow 50 条（切换前） | 28 次 LLM、12 次硬路由、10 次降级 | 旧灰度期曾验证异步隔离 | 当前 v4 的质量与延迟 |
| `router_blind_v1` | 未 provision | 评测契约和仓外保存机制已存在 | 任何盲测质量结论 |

因此当前状态是“智能规划链路已完成、已知边界回归较强、开放世界验收未完成”，不能
以 480 题开发集全通过宣称已经解决开放语言泛化。

启用前按以下顺序收敛，不再为单个问句追加规则：

1. 使用 GPT-5.6 Terra 在 8001 Canary 完成至少 20 个真实复杂问题的人工体验，逐题只记录
   `期望服务模式 / 最小充分路由 / 实际路由 / 是否答对 / 用户可接受延迟`。
2. 对线上 v4 采样进行人工裁决；按证据和最小充分来源判断结果，不把旧规则当作正确答案。
3. 由未阅读路由实现的人独立编写并冻结仓外 `top_chat_v3_blind_v1`：120 组动作/查询
   对照、60 组多轮话题切换、40 组歧义澄清。设置 manifest SHA-256 后运行
   `eval_top_chat_v3.py`；报告只输出聚合指标。任何被公开调试的失败题立即降级为开发集，
   并补充新的盲题。未 provision 的集合明确返回非通过状态。
4. 修复必须落在 Route Registry、ActionCapabilityRegistry、SemanticQueryPlan 契约或证据门；
   自由文本不因 ID、日期或状态直接进入硬路由，确定性快路仅接受系统结构化请求。
5. v4 发布必须同时满足：动作能力匹配率 ≥ 98%，Guide/Recall 互误判 < 1%，写命令进入
   Recall 为 0，Clarify precision/recall ≥ 95%，timeout ≤ 1%，规划 p95 ≤ 5 秒，且 Guide
   评测期间业务文件和记忆事件写入为 0。原有只读路由集继续要求 Source/Route
   Macro-F1 ≥ 95%、required-route recall ≥ 98% 和个人事实错误通用回退为 0。

若智能规划器不可用，在线入口保守返回澄清状态。需要比较旧规则时，评测程序必须直接调用
`rule_plan_query()`，不能让旧结果参与用户答案。
