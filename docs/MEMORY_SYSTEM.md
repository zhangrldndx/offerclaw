# OfferClaw 记忆系统

本文说明当前记忆系统的边界、数据契约、生命周期和运行维护方式。SQLite 是记忆的事实来源；`logs/memory/episodic.jsonl`、`semantic.json` 和 `procedural.json` 是兼容导出，不能作为并发写入入口。

## 1. 三层记忆与业务边界

### 情景记忆

情景层记录已经发生且值得追溯的业务操作：

- 画像保存和修改，包含修改前后快照；
- JD 分析、匹配和用户选择；
- 投递创建、状态、备注、下一步动作、长期关注和投递复盘；
- 计划草稿生成、批准、拒绝和正式计划保存；
- 每日执行、笔记、耗时、阻碍和复盘；
- 顶部问答中已经发送的用户消息、实际生成的助手回复，以及完成、失败或中断状态；
- 简历、知识库材料的提交、采用、晋升和删除。

普通浏览、鼠标轨迹和未提交草稿不进入事件流。系统生成内容使用 `actor=assistant|system`，不会自动成为用户事实。每次业务操作使用稳定的 `operation_id`，重试不会重复记忆。

### 语义记忆

语义层保存带证据的用户事实和认知，包括身份约束、明确偏好、阶段兴趣、当前选择、能力证据和复盘结论。每条记忆都有：

- `memory_type`、结构化 `value` 和 `confidence`；
- 支持证据与反对证据；
- `target_context_id`、生效时间和生命周期；
- 版本、创建时间和更新时间。

正式画像和明确业务选择由确定性规则及时蒸馏。自由文本通过异步模型任务产生候选，再由代码逐条核对事件 ID、用户原文连续片段、独立操作数、日期跨度、目标范围和显式冲突。推断偏好至少需要 3 个独立操作、跨 2 个自然日，且不能有更高优先级冲突。助手自己的总结、重复请求和同一操作的衍生事件不会重复增强证据。模型失败不会阻塞业务保存，失败类型、重试时间和处理游标会写入运行状态。

### 程序记忆

程序层保存“在什么场景采用什么步骤，并出现什么结果”的 SOP。岗位匹配为适合不等于 SOP 成功。只有关联到实际执行的 `success` 或 `failure` 结果才更新效果证据。

SOP 生命周期为 `candidate -> active -> suspended|archived`。默认至少 3 个独立案例、跨 2 天，且 Beta(1,1) 工程评分达到 0.75 才激活；连续两次明确失败会暂停，180 天无有效支持会归档。效果证据按 90 天半衰期参与召回排序。

## 2. 事件契约

事件由服务端生成完整 UUID 和单调序号。主要信封字段如下：

| 字段 | 含义 |
|---|---|
| `event_id` / `seq` | 全局事件身份与增量读取序号 |
| `schema_version` / `event_type` | Schema 版本与类型 |
| `occurred_at` / `recorded_at` / `business_date` | 发生时间、记录时间与业务日期 |
| `actor` / `source` / `traffic_origin` | 操作者、入口和 organic/test/replay 来源 |
| `operation_id` / `correlation_id` / `causation_id` | 幂等、同链路关联和因果关系 |
| `target_context_id` | 事件适用的职业目标 |
| `entity_type` / `entity_id` / `entity_version` | 业务实体及版本 |
| `payload` | 按事件类型校验的内容 |

原文通过 `evidence_snapshots` 按内容哈希去重保存，事件只引用快照。Pydantic 会校验枚举、日期、引用、数值范围和类型化内容；不能迁移的旧记录进入隔离区。

## 3. 状态和任务身份

岗位匹配逻辑只使用机器枚举：

```text
suitable | stretch | not_recommended | unknown
```

旧中文值仅在兼容入口按精确映射转换，“不适合”不会因为包含“适合”而误计。投递进度、任务执行、硬门槛和软条件也使用各自枚举。

任务统计以 `task_id` 为具体任务身份，`task_series_id` 表示长期或重复任务，`skill_id/gap_id` 关联技能。旧文本只做 Unicode、空白和 Markdown 外壳规范化，并使用完整内容与目标范围的指纹精确匹配；不再截取前 12 个字符。相似匹配只能提出人工可核对的关联候选。

## 4. 反思与计划权限

每日执行、用户复盘和投递反馈会形成带证据的调整建议。反思不会直接修改当前计划。用户主动生成或重排时，系统创建计划草稿和变化说明；只有用户批准且目标、计划、来源版本仍一致时才保存为当前计划。

连续高偏离按业务日期聚合，要求三个连续自然日。同一天多次复盘只算一天，周复盘不重复贡献日证据，没有日志也不会被当成未完成。

## 5. 目标隔离和遗忘

职业目标存放在独立上下文中。大幅转向时创建新目标，旧目标的偏好、JD 和 SOP 默认不参与当前建议；历史回忆仍可跨目标查询并显示当时目标。正式切换需要明确的用户操作，聊天中的探索表达不会直接改变正式目标。

记忆采用四种机制：

- 衰减：阶段兴趣默认 30 天半衰期，推断偏好默认 90 天半衰期；
- 失效或替代：新事实或用户纠正让旧结论退出当前使用；
- 归档：180 天无新支持的行为推断和 SOP 默认不再参与建议；
- 删除：删除事件或记忆及其派生关系，并写入最小 tombstone，防止索引重建时复活。

确认事实和原始学习记录不会仅因时间经过自动删除。明确约束在用户修改或有效期结束前不衰减。

## 6. 召回

建议场景默认只读取当前目标和仍生效的记忆。SOP 先按目标、任务类型、岗位类别、阶段、技能、前置条件和禁用条件过滤，再按相关性、效果证据和时效排序，最多返回 3 条。缺少上下文时只返回明确标为通用的 SOP。

个人历史回忆采用结构化、中文词面和向量三路召回：

1. 日期、任务 ID 和实体条件先做结构化过滤；
2. 词面和向量各取最多 20 个候选；
3. 用 RRF 融合、去重和重排；
4. 回到原始快照核对版本、来源状态和删除标记；
5. 默认返回 5 条带日期、来源和目标上下文的证据。

当前提问及其衍生回答会从“以前是否学过”的证据中排除。个人记忆索引与公共知识库分开。个人向量和来源元数据保存在记忆 SQLite 的 `memory_search_chunks` 表中，避免公共 Chroma 索引故障影响个人记忆；向量计算在写事务外完成。向量模型不可用时，接口返回 `dense_status=unavailable|disabled`，并继续使用结构化和词面检索，不能伪造历史。

## 7. 并发和恢复

SQLite 使用事务、唯一约束、乐观版本检查和有界重试。模型调用不能放在写事务中。SQLite 运行时低于 3.51.3 时使用回滚日志；满足版本要求才启用 WAL。

业务 Markdown/JSON 文件和 SQLite 之间由持久操作日志协调：先记录意图和文件指纹，再在跨平台文件锁下原子写文件，最后提交事件和处理任务。进程重启会恢复未完成操作；若文件被第三方修改则标记冲突，不覆盖未知版本。

测试进程根据 `PYTEST_CURRENT_TEST` 自动使用独立数据库。测试和回放事件还必须带对应 `traffic_origin`，它们不能进入生产蒸馏和索引。

## 8. API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/memory/timeline` | 事件时间线 |
| GET | `/api/memory/semantic` | 系统记住的事实、偏好和结论 |
| GET | `/api/memory/semantic/{id}/evidence` | 查看支持与反对证据 |
| GET | `/api/memory/sops` | SOP、评分和生命周期 |
| GET | `/api/memory/search` | 个人历史混合检索 |
| POST | `/api/memory/distill` | 处理待蒸馏事件 |
| POST | `/api/memory/goals/switch` | 创建并切换正式目标上下文 |
| POST | `/api/memory/goals/{id}/activate` | 激活已有目标 |
| POST | `/api/memory/{type}/{id}/lifecycle` | 纠正生效状态、归档或恢复 |
| DELETE | `/api/memory/{type}/{id}` | 删除及级联撤销派生结果 |
| POST | `/api/memory/index/rebuild` | 从有效来源重建个人索引 |

计划生成接口返回草稿；计划决策接口负责批准或拒绝。顶部普通和 SSE 问答均返回会话、消息和操作身份，完成、失败或中断后保存实际回复状态。

## 9. 配置与维护

常用环境变量：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `OFFERCLAW_MEMORY_DIR` | `logs/memory` | 记忆存储目录 |
| `MEMORY_DENSE` | `1` | 是否启用个人记忆向量召回 |
| `OFFERCLAW_MEMORY_MODEL_DISTILL` | `1` | 是否异步运行自由文本模型蒸馏 |
| `OFFERCLAW_SOP_MIN_CASES` | `3` | SOP 激活最少案例数 |
| `OFFERCLAW_SOP_MIN_DAYS` | `2` | SOP 激活最少自然日 |
| `OFFERCLAW_SOP_ACTIVATION_CONFIDENCE` | `0.75` | SOP 激活评分门槛 |
| `OFFERCLAW_SOP_HALF_LIFE_DAYS` | `90` | SOP 证据半衰期 |
| `OFFERCLAW_SOP_ARCHIVE_DAYS` | `180` | SOP 无支持归档天数 |

`POST /api/memory/distill?model_assisted=true` 会同步运行一次人工触发的模型候选提取，并返回每条候选的接受或拒绝原因。没有达到证据门槛时返回 `waiting_for_evidence`；模型不可用时保留确定性蒸馏结果，并在 `memory_health.model_distill` 中暴露降级状态。

迁移先生成只读报告：

```powershell
python scripts/migrate_memory_sqlite.py
```

确认报告后执行可重复迁移：

```powershell
python scripts/migrate_memory_sqlite.py --apply
```

迁移会建立带时间戳的旧数据备份，保留旧 ID 到新 UUID 的映射。来源不明的历史流程事件保留为审计记录，但标记为不可信并退出自动蒸馏。重跑只处理尚未导入的稳定内容快照。
