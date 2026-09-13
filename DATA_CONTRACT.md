# OfferClaw · 数据契约（DATA_CONTRACT）

> 版本：v1.0 · 2026-04-25  
> 用途：明确 OfferClaw 中所有 Markdown / JSON / 目录的边界、写入权限和 Git 提交策略，防止"长期状态层"被误改、误删、误推。

---

## 0. 总原则

OfferClaw 是一个 **长期状态型 Agent 系统**，必须区分两类资产：

| 类别 | 含义 | 写入权限 | 入 Git |
|---|---|---|---|
| **User Layer** | 用户私有的事实状态（画像、日志、投递、故事） | 仅在用户确认后由 Agent 追加；不得静默覆盖 | ❌ 一律不入 Git |
| **System Layer** | 系统规则、Prompt、代码、文档 | 可由开发者直接更新并版本化 | ✅ 全部入 Git |
| **Runtime / Secrets** | 运行时产物、密钥、向量库 | 自动产生 | ❌ 一律 .gitignore |
| **Public Fixtures** | 指定测试目录中的虚构画像、日志和投递样例 | 仅开发者维护；必须显式标注为合成数据 | ✅ 仅白名单路径入 Git |

> 一句话：**用户层负责"我是谁、我做了什么"；系统层负责"OfferClaw 是什么、怎么做"。两者不能互相覆盖。**

---

## 1. User Layer（用户事实状态层）

| 文件 / 目录 | 角色 | 自动写入策略 | Git 策略 |
|---|---|---|---|
| `user_profile.md` | 用户画像（基础信息、技能、方向） | **必须用户确认**；只允许追加 / 更新 §0 元信息 + 已存在字段；不得新增章节 | ❌ 本地私有 |
| `daily_log.md` | 每日学习/求职日志 | Agent 可在用户结束当日操作时追加一行；不得回溯改写 | ❌ 本地私有 |
| `applications.md` | 真实投递追踪表（见 §4.2） | Agent 追加新行；状态变更必须由用户触发 | ❌ 本地私有 |
| `application_jds/` | 用户确认加入投递管理的完整 JD 不可变版本及匹配快照 | 仅确认接口新增版本；禁止覆盖旧版本 | ❌ 用户私有运行时数据 |
| `interview_story_bank.md` | 面试 STAR+R 故事库 | 用户主动写入；Agent 可生成草稿但不直接合并 | ❌ 本地私有 |
| `jd_candidates.md` | 本地候选 JD 与回归输入 | 只允许本地维护；禁止 Agent 静默写入 | ❌ 本地私有 |
| `plans/` | LLM 生成的 N 周学习计划 | Portfolio Plan Agent 只生成 checkpoint 草稿；用户批准后新增版本，不覆盖旧文件 | ❌ .gitignore（产物） |
| `resume_drafts/` | 与投递及活动 JD 绑定的简历草稿 | Writer/Critic 完成后必须经用户批准才新增文件 | ❌ .gitignore（产物） |
| `summaries/` | 单日 / 周度复盘 | Agent 自由写入新文件 | ❌ .gitignore |
| `memory.json` | Agent 长期记忆 KV | Agent 读写 | ❌ .gitignore |
| `memory/` | 历史会话 / 长上下文存档 | Agent 写入 | ❌ .gitignore |
| `logs/` | JSON 日志 | 中间件自动写 | ❌ .gitignore |
| `profiles/p*_*.json` | 多 persona 测试样本 | 仅允许纯合成 persona；私有文件使用 `private_` / `_local` 命名 | ✅ 仅纯合成样本 |
| `integrations/openclaw/lab-fixtures/` | 一次性实验环境输入 | 每个文件必须声明“合成/虚构”，不得从根目录用户状态复制 | ✅ 白名单合成夹具 |

---

## 2. System Layer（系统规则与代码层）

### 2.1 Prompt 与规则文件
| 文件 | 角色 |
|---|---|
| `SOUL.md` | OfferClaw 的产品宪法 / 不可逾越红线 |
| `target_rules.md` | 目标方向白/黑名单、匹配阈值 |
| `source_policy.md` | 信息源可信度分级（A/B/C） |
| `onboarding_prompt.md` | 首次接入新用户的 Prompt |
| `job_match_prompt.md` | LLM 岗位匹配指令 |
| `plan_prompt.md` | 学习计划生成指令 |
| `summary_prompt.md` | 复盘生成指令 |

### 2.2 代码
| 文件 | 角色 |
|---|---|
| `jd_parser.py` | 从 JD 原文生成带 evidence span 的结构化要求，不评价候选人 |
| `semantic_matcher.py` | 将非硬门槛要求对齐到画像证据 ID，不输出最终结论 |
| `match_job.py` | 硬门槛与三档结论的确定性裁决；消费已校验的语义对齐 |
| `plan_gen.py` | 4 周计划生成 |
| `summary_tool.py` | 复盘工具 |
| `pipeline.py` | match→plan→log 流水线 |
| `rag_ingest.py` / `rag_query.py` / `rag_graph.py` / `rag_tools.py` / `rag_agent.py` | RAG 全栈 |
| `rag_api.py` | FastAPI 服务层 |
| `logging_utils.py` | 结构化日志中间件 |
| `eval_rag.py` | RAG 召回 / MRR 评估 |
| `tools.py` | Agent 自定义工具 |
| `agent_demo.py` | 最小 Agent demo |
| `doctor.py` | 工程健康检查（见 §4.6） |
| `verify_pipeline.py` | 主链路端到端验证（见 §4.7） |
| `tests/` | pytest 单测与回归 |
| `static/` | 前端控制台 |

### 2.3 文档
| 文件 | 角色 |
|---|---|
| `README.md` | 项目门面 |
| `PROJECT_STATUS.md` | 进度仪表盘（手动维护，每个 Sprint 末更新） |
| `docs/architecture.md` | 4 张 Mermaid 架构图 |
| `docs/archive/demo.md` | 7 步演示流程（已归档）|
| `docs/archive/resume_pitch.md` | 简历短/中/JD 对照三档（已归档）|
| `docs/interview_qa.md` | 10 题面试卡 |
| `docs/project_one_pager.md` | 一页纸（见 §4.3） |
| `docs/postmortem.md` | 技术复盘（见 §4.9） |
| `docs/ethical_use.md` | 伦理边界（见 §4.5） |
| `DATA_CONTRACT.md` | **本文件** |
| `RAG_QUICKSTART_REPORT.md` | RAG 接入复盘 |
| `AGENT_DEMO.md` | Agent demo 说明 |
| `docs/WECHAT_INTEGRATION.md` | 脱敏后的渠道接入与部署说明 |

---

## 3. Runtime / Secrets（绝不入 Git）

| 文件 / 目录 | 说明 |
|---|---|
| `.env.local` / `.env*` | API Key、私有网关地址与模型配置 |
| `chroma_db/` | 本地向量库，体积大且与机器相关 |
| `__pycache__/` / `.pytest_cache/` / `.venv/` | Python 产物 |
| `.vscode/` / `.idea/` / `.claude/` | IDE / Agent CLI 配置 |
| `logs/` `summaries/` `plans/` `resume_drafts/` `memory.json` `memory/` | 运行时输出 |
| `.offerclaw/career_threads.sqlite3` | 多 Agent checkpoint、审批中断与恢复状态；不是真实投递事实源 |
| `1.txt` / `1` / `2` 等临时文件 | 不要加入 |

**任何含真实邮箱、电话、招聘方内部联系方式、未公开 JD 全文的文件，禁止入 Git。**

---

## 4. 写入与变更追踪规则

### 4.0 写入策略表（11 文件 × 6 列）

> 一站式索引：每个关键文件「能不能读 / 能不能写 / 能不能 Agent 自动写 / 是否需要用户确认 / 是否入 Git / 备注」。

| 文件 / 目录 | 可读 | 可写 | 自动写 | 需用户确认 | 入 Git | 备注 |
|---|---|---|---|---|---|---|
| `user_profile.md` | ✅ | ✅ | ❌ | ✅ | ❌ .gitignore | 仅追加；§0/§1 事实字段不得静默改 |
| `daily_log.md` | ✅ | ✅ | ✅ 仅追加 | ⚠️ 系统改动留"【系统更新】" | ❌ .gitignore | 不可回溯改写 |
| `applications.md` | ✅ | ✅ | ✅ 仅追加新行（状态=已评估） | ✅ 状态前进必须确认 | ❌ .gitignore | 字段见 §4.2；状态必须在合法枚举 |
| `interview_story_bank.md` | ✅ | ✅ | ⚠️ 仅生成草稿 | ✅ 用户合并 | ❌ .gitignore | 每条须含 STAR+R + Metadata |
| `jd_candidates.md` | ✅ | ⚠️ 仅本地 | ❌ | ❌ | ❌ .gitignore | 可能含真实 JD；禁止进入产品 RAG 与 Git |
| `memory.json` | ✅ | ✅ | ✅ | ❌ | ❌ .gitignore | Agent 长期记忆 KV |
| `chroma_db/` | ✅ | ✅ | ✅ | ❌ | ❌ .gitignore | 本地向量库，~3.4MB |
| `.env.local` | ✅ | ✅ | ❌ | ✅ | ❌ .gitignore | 含智谱 API Key |
| `plans/` | ✅ | ✅ | ❌ | ✅ | ❌ .gitignore | Agent 草稿先留在 checkpoint，批准后新增版本 |
| `resume_drafts/` | ✅ | ✅ | ❌ | ✅ | ❌ .gitignore | 绑定 application_id + jd_version_id，批准后写入 |
| `summaries/` `logs/` | ✅ | ✅ | ✅ | ❌ | ❌ .gitignore | 运行时产物 |
| `profiles/p*.json` | ✅ | ✅ | ❌ | ✅ | ✅ 仅合成数据 | `private_*` / `*_local` 始终忽略 |
| `integrations/openclaw/lab-fixtures/` | ✅ | ⚠️ 仅开发者 | ❌ | ❌ | ✅ 白名单 | 必须逐文件声明为合成数据 |
| System Layer (`SOUL.md` `target_rules.md` `*_prompt.md` `*.py` `docs/*`) | ✅ | ✅ | ❌ | ❌（开发者 PR） | ✅ | 任何改动必须 git commit |

> **三条铁律**：
> ① 任何 Agent 自动写 ≠ 静默写 ——`memory.json` / `chroma_db/` 之外，所有用户层文件的改动必须可追溯；
> ② "需用户确认" 的文件，FastAPI 端点必须有显式 confirm 参数，UI 必须有二次点击；
> ③ ❌ 入 Git 的文件不光在 `.gitignore`，CI 还要做 `git ls-files | grep` 兜底。

### 4.1 自动写入边界

| 场景 | 是否允许 Agent 自动写 |
|---|---|
| 在 `daily_log.md` 末尾追加今日复盘 | ✅ |
| 在 `plans/` 新建计划文件 | ❌ Agent 必须先展示范围、草稿并取得批准；旧兼容接口除外 |
| 在 `resume_drafts/` 新建简历文件 | ❌ 必须经过 Writer/Critic 后由用户批准 |
| 在 `summaries/` 新建复盘文件 | ✅ |
| 在 `applications.md` 新增一行投递（状态=已评估） | ✅ |
| 修改 `applications.md` 中投递状态（如→已投递、→面试中） | ❌ 必须用户确认 |
| 修改 `user_profile.md` 中"基础信息"姓名/学校 | ❌ 必须用户确认 |
| 在 `user_profile.md` 中追加"§4 项目"新条目 | ⚠️ Agent 给出草稿，用户合并 |
| 修改 `SOUL.md` / `target_rules.md` / `source_policy.md` | ❌ 仅开发者通过 PR |
| 写入 `interview_story_bank.md` 新故事 | ⚠️ Agent 出草稿，用户终审 |

### 4.2 变更追溯
- 所有 System Layer 改动必须经 git commit；commit message 用前缀：`feat:` / `fix:` / `docs:` / `chore:` / `test:` / `refactor:`
- User Layer 改动如果由 Agent 触发，需在 `daily_log.md` 留一行"【系统更新】§X 由 OfferClaw 在 YYYY-MM-DD 追加"
- `PROJECT_STATUS.md` 在每个 Sprint 收尾时更新一次"最近变更"段

### 4.3 信息源可信度（与 `source_policy.md` 对齐）
- A 级：官网 / 公司官方招聘页 / 政府文件 → 可直接进入 `user_profile.md` 与 `applications.md`
- B 级：知名媒体 / LinkedIn / 脉脉 → 仅用于本次分析；用户确认建立真实投递后，连同来源写入该投递的 JD 版本
- C 级：匿名爆料 / 论坛传闻 → 仅供参考，不写入 user/applications 层

---

## 5. 命名约定

- Markdown 文件：小写下划线 `.md`（例外：`README.md`、`SOUL.md`、`PROJECT_STATUS.md`、`DATA_CONTRACT.md` 全大写传统）
- 计划：`plans/plan_YYYYMMDD_<主题>.md`
- 复盘：`summaries/summary_YYYYMMDD.md` / `summary_week_YYYY_WW.md`
- 测试 persona：`profiles/p<编号>_<标签>.json`
- 测试代码：`tests/test_<模块>.py`

---

## 6. 演进策略

随着项目演进，本契约也会改：
1. 真实画像、日志、投递和面试材料始终留在根目录私有文件；公开演示只维护白名单中的独立合成夹具。
2. 接入数据库后：`memory.json` / `applications.md` 迁到 SQLite，本契约新增 §7 数据库表设计。
3. 引入多用户：把 `user_profile.md` 改造为 `users/<uid>/profile.md`，本契约更新 User Layer 索引方式。

---

## 7. 不变量（任何时候都不能违反）

1. **不得**把含真实 API Key、密码、未脱敏简历 PDF 的文件入 Git。
2. **不得**让 Agent 在用户未确认时覆盖 `user_profile.md` 已有事实字段。
3. **不得**把 `applications.md` 中"已投递 / 面试中"等敏感状态自动改为"已拒绝"。
4. **不得**绕过 `target_rules.md` 与 `SOUL.md` 强制简历内容（杜绝伪造经历）。
5. **必须**在删除 / 重命名任何 System Layer 文件时同步更新本契约 §2 表格。
