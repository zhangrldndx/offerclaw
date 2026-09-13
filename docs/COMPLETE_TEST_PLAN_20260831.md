# OfferClaw 下一阶段完整测试规划

> 版本：2026-08-31  
> 目标：在不污染项目数据、不继续对旧开发集调参的前提下，把 OfferClaw 从“面试可展示”推进到“指标口径清楚、结果可复现、失败可归因”的发布候选。  
> 当前冻结基线以本地私有评估清单为准；公开文档不记录旧提交或本地索引指纹。

---

## 1. 总体判断

OfferClaw 当前已经具备较完整的作品集工程形态：求职闭环、FastAPI/UI、RAG、状态管理、CareerFlow、MCP、审计与回归工具都不是占位实现；`1430 passed / 4 skipped` 和六域小型验收 `24/24` 说明核心代码合同较稳定。

但“测试很多”不等于“产品质量已经完整证明”。当前证据分成四层：

| 层级 | 当前证据 | 可以证明 | 不能证明 |
|---|---|---|---|
| 单元/合同 | 1430 passed / 4 skipped | 大量边界、降级、状态与纯函数行为稳定 | 真实 LLM、浏览器和多步骤链路全部可靠 |
| 小型产品集成 | 六域 24/24 | 画像、JD、计划、投递、复盘、简历/Flow 的关键确定性合同可运行 | 真 API、真模型、真 UI 的完整用户旅程 |
| 检索 | 同分布 R@1 89%；口语回归 R@1 75%；历史 Final v4 strict R@1 59/80 | 系统具备较强同分布与一定口语泛化；历史冻结包可支撑 strict 叙述 | 当前索引/HEAD 在全量 strict 集上的完整复跑结果；全新盲集泛化 |
| 最终答案 | 当前 HEAD 48 题双裁判 96/96 | 忠实度高，且能暴露完整度、动作和延迟问题 | 独立盲集上的发布质量；真实用户总体通过率 |

因此当前最准确的项目判断是：

> **已经可以用于面试展示；下一轮测试的目标不是继续堆单测，而是补齐“当前 HEAD 回归、端到端动作、独立盲测、性能与真实产品旅程”四块证据。**

---

## 2. 当前指标与主要缺口

### 2.1 已有强项

- 全量测试：**1430 passed / 4 skipped**；
- 工程自检：doctor **12/12**，verify_pipeline **6/6**，verify_docs 零漂移；
- 六域确定性验收：**24/24**；
- 同分布 n=100：R@1/R@3/R@5 = **89%/95%/97%**；
- 口语化开发回归 n=52：Recall@1/R@3/R@5 = **75.0%/80.8%/84.6%**；
- Final v4 历史冻结 strict：Candidate **73/80**，R@1 **59/80**，R@3 **70/80**，R@5 **72/80**，MRR **0.8083**，nDCG@5 **0.8317**；
- 当前 HEAD 答案质量 n=48：两位裁判 claim faithfulness **96.9%–97.1%**；
- 48/48 成功生成，正式裁判覆盖 **96/96**，无缺失样本补分。

### 2.2 当前最需要解决的四个问题

#### P0：门控动作没有完整传给生成

48 题中 8 个负例的 pipeline 路由动作与预期 **8/8** 一致，但端到端动作合同只有 **6/8**。4 个 `correct_premise` 均在 trace 层识别成功，最终答案只有 **2/4** 真正首句纠错。

这不是检索问题，而是接口契约问题：当前 `/api/query` 没有稳定地把 trace 中的 `correct_premise` 传给 `_grounded_messages(..., answer_action=...)`。

#### P0：Quality 延迟不适合现场演示

当前 n=48、独立空 answerability cache、模型常驻单进程实测：

| 阶段 | p50 | p95 | max |
|---|---:|---:|---:|
| 检索 + answerability/Gate | 31.0s | 80.4s | 81.4s |
| 最终生成 | 4.35s | 77.4s | 90.9s |
| 端到端 | **37.6s** | **97.1s** | 158.7s |

高忠实度说明 Quality 路径有价值，但当前形态应被定位为质量上界，而不是默认面试演示档。

#### P1：答案完整度不足且裁判分歧较大

- DeepSeek-v3 completeness：**81.25%**；
- GPT-5.5 completeness：**65.0%**；
- 差值：**16.25pp**。

不能挑 81.25% 当唯一 headline，也不能直接认定 65% 就是绝对真值。需要对“两个裁判分歧 + 高影响漏答”做人工仲裁，并区分：

1. exact-gold 没进入上下文；
2. 等价证据进入，但 qrels 只标了一个 chunk；
3. 上下文有答案，生成没有覆盖 `answer_requirements`；
4. 裁判对 partial/full 标准不一致。

#### P1：发布证据仍主要来自开发可见集合

- 口语 52 题已经参与多轮开发；
- answer-quality-v2 48 题来自历史 reranker 数据建设；
- current-head 两题 smoke 只能证明链路可运行；
- Final v4 是历史盲测，但当前索引 content hash 已变化。

下一轮必须增加一套冻结后才运行的独立 Blind A；不再等待 200 条 organic 流量，organic 只作为上线后的长期监测。

---

## 3. 本轮不做什么

为避免再次走偏，本轮测试明确排除以下行为：

- 不训练新的 reranker/student model；
- 不在口语 52、answer-quality-v2 或 Final v4 上继续选阈值；
- 不把两题 smoke 当作当前检索指标；
- 不用同一批题同时调 prompt 又做发布验收；
- 不因某一位裁判分数更高就只引用该裁判；
- 不等待一年才能收满的 200 条 organic 请求；
- 不把测试 raw、问题、片段、答案或裁判理由写进仓库；
- 不对真实画像、计划、投递记录、复盘文件或生产 Chroma collection 执行写操作；
- 不在完整测试期间顺手改模型、pool、阈值或 prompt。

如果测试失败，先按失败类型归因，再决定是否进入下一轮开发；不得通过补简单题或删除失败样本“修复指标”。

---

## 4. 测试分层与发布矩阵

| 层 | 测试内容 | 当前状态 | 下一轮目标 | 发布硬门 |
|---|---|---|---|---|
| L0 静态/谱系 | HEAD、代码、配置、prompt、索引、数据哈希 | 已有冻结器 | 冻结候选 commit 与所有评测工具 | 任一漂移即停止 |
| L1 单元/合同 | pytest 全量 | 1430/1430 | 修改后保持全绿 | 0 fail，skip 只能是预登记 E2E |
| L2 确定性集成 | 六域 24 例 | 24/24 | 扩至 36–48 例，增加 API 状态不变量 | 100% 通过 |
| L3 真 API/LLM | query/search/stream 等 | 当前仅 3 个弱语义 smoke | 12 个强语义 live case | 12/12，无 5xx/空答案/状态污染 |
| L4 strict 检索 | Final v4 120 行 | 当前 HEAD 仅 2 题 smoke | 当前 HEAD 先跑 1 次回归；发布时 3 次 | 见 §7 |
| L5 答案动作 | correct/abstain/answer | 开发回归端到端 6/8 | 24 题动作合同专项集 | correct_premise ≥11/12；abstain 12/12 |
| L6 答案质量 | faithfulness/completeness/citation | 开发回归 n=48 | 新 Blind A：60 正 + 20 负 | 见 §9 |
| L7 性能 | cold/warm/concurrency | 当前只有单进程 n=48 | Fast/Quality 分档基准 | 见 §10 |
| L8 UI 用户旅程 | Playwright 关键流 | 未形成发布套件 | 8 条核心旅程 | 8/8，无真实数据写入 |
| L9 故障/安全 | API、模型、缓存、注入、上传 | 大量单测，缺统一 live 报告 | 20 条发布守卫 | 0 泄漏、0 静默错误放行 |
| L10 跨机复现 | Mac + 高性能电脑 | 分散执行过 | 同 manifest 双机复跑 | 指标方向一致、哈希一致 |

---

## 5. 数据与环境隔离规则

### 5.1 三类评测集必须分开

| 集合 | 用途 | 是否允许调参 | 是否可作发布结论 |
|---|---|---:|---:|
| Development Regression | 现有 52、48、Final v4 回归 | 是，但调后必须重冻候选 | 否，只作回归/诊断 |
| Blind A | 下一轮全新 80 题 | 否 | 是，首次解封结果 |
| Organic Shadow | 上线后真实流量 | 否 | 只作长期监测，不作本轮前置条件 |

### 5.2 文件隔离

- 仓库内只保留协议、无文本 selection manifest、聚合 summary、报告；
- raw 写到仓库外的 `/absolute/external/offerclaw-private-eval/release_v5/<run_id>/`；
- 每个 run 使用自己的 answerability cache；
- 评测时设置 `LLM_USAGE_LOG=0`，避免写入项目运行账本；
- 所有会修改画像/计划/投递/复盘的测试必须用 pytest `tmp_path` 或新增的 test-only data root；
- Chroma 只允许 query/get，禁止 add/upsert/delete；
- 不得使用 `git add .`，避免把已有大量本地评测产物一起纳入提交。

### 5.3 冻结顺序

```text
代码候选完成
→ 全量单测
→ 冻结 commit / runtime / prompt / index / dataset
→ 运行 Development Regression
→ 修复则重新开始冻结
→ 冻结最终候选 SHA
→ 才能解封 Blind A 标签并运行
```

Blind A 运行后不得再修改候选并继续沿用同一 Blind A 发布结论。若修改，Blind A 自动降级为下一轮 Development Regression。

---

## 6. 执行阶段总览

| 阶段 | 工作量 | 目的 | 是否阻塞下一阶段 |
|---|---:|---|---:|
| Phase 0：候选整理与冻结 | 0.5 天 | 得到可复现基线 | 是 |
| Phase 1：P0 动作合同修复与专项测试 | 0.5 天 | 修复 2/4 错误前提生成缺口 | 是 |
| Phase 2：当前 HEAD 完整回归 | 1–2 小时起 | 确认没有 strict/safety 回退 | 是 |
| Phase 3：API/UI/状态验收 | 0.5–1 天 | 证明产品旅程，而不只是纯函数 | 是 |
| Phase 4：构建并预冻结 Blind A | 1–2 天 | 在候选完成前独立准备发布集，但不运行 | 是 |
| Phase 5：性能/故障测试与 Blind A 解封 | 0.5–1 天 + 运行时间 | 先固化产品形态，再给出新发布证据 | 是 |
| Phase 6：双机复现与最终报告 | 0.5 天 | 面试/交付证据收口 | 否 |

完整发布版预计 3–5 个工作日；面试最小闭环见 §13，可在 1 天左右完成。

---

## 7. Phase 0–2：冻结、动作合同与 strict 回归

### Phase 0：候选整理

1. 人工复核本轮新增文件，只暂存本轮明确产物；
2. 不处理或删除用户此前留下的 `docs/rag_eval/colloquial/**` 本地产物；
3. 创建候选 commit 后重新生成冻结清单；
4. 校验 HEAD、tree、runtime 文件、prompt、索引与 qrels 哈希；
5. 运行基础门禁：

```bash
RAG_RERANK=0 .venv/bin/python -m pytest tests/ -q
.venv/bin/python doctor.py
.venv/bin/python verify_pipeline.py
.venv/bin/python verify_docs.py
.venv/bin/python scripts/freeze_current_head_release.py \
  --output docs/rag_eval/release_v5/FROZEN_CONFIG.json \
  --dataset docs/rag_eval/final_v4/final_v4.json
.venv/bin/python scripts/freeze_current_head_release.py \
  --verify docs/rag_eval/release_v5/FROZEN_CONFIG.json
```

硬门：`0 failed`、doctor 12/12、pipeline 6/6、docs 零漂移、freeze verify OK。

### Phase 1：correct_premise 端到端专项集

先加测试，再改生产代码。专项集固定 24 题：

- 12 个库内有明确反证的 `correct_premise`；
- 12 个库内不足、必须 `abstain` 的控制题；
- 同一问题不得通过重复改写扩充分母；
- 同时检查 pipeline action、generation contract action、最终首句动作和是否补充无证据内容。

硬门：

- pipeline action：24/24；
- correct_premise 最终答案：至少 11/12；
- abstain：12/12 不把通用回答冒充 KB 证据；
- correct_premise 不得补充上下文外事实；
- 修复不得改变正常 answer 样本的排序。

若达不到，停止后续 Blind A；先修动作合同，不动 retriever。

### Phase 2：当前 HEAD strict 回归

先跑一次 120 行回归，证明当前候选相对历史 Final v4 没有明显回退：

```bash
LLM_USAGE_LOG=0 .venv/bin/python scripts/run_current_head_regression.py \
  --freeze docs/rag_eval/release_v5/FROZEN_CONFIG.json \
  --outdir /absolute/external/offerclaw-private-eval/release_v5/current_head_run1 \
  --summary docs/rag_eval/release_v5/CURRENT_HEAD_SUMMARY.json \
  --repeats 1
```

该集已经对开发可见，所以名称只能是 `current_head_regression_not_blind`。

建议非劣门槛：

- Candidate：不低于 72/80；
- R@1：不低于 58/80；
- R@3：不低于 69/80；
- R@5：不低于 71/80；
- MRR 不低于 0.79；
- nDCG@5 不低于 0.81；
- 须拒答真误纳不超过 1/28；
- correct_premise 门控识别保持 12/12；
- 必须输出 paired wins/losses，不能只看总分。

这里允许相对历史中位数最多回退 1 题，用于容纳 LLM 判据随机性；安全性不放宽。准备正式发布报告时再跑 `--repeats 3`，取逐指标中位，不能挑最好的一次。

---

## 8. Phase 3：产品功能、API 与 UI 完整验收

### 8.1 扩展确定性验收：24 → 36–48

保留现有六域，每域增加 2–4 个跨模块不变量：

| 域 | 新增重点 |
|---|---|
| Profile | 冲突更新、建议审批、损坏恢复、版本不倒退 |
| JD Match | 真实 JD 文本、硬否决优先级、远程/城市冲突、缺字段 |
| Plan | stale mtime、重复保存幂等、跨天同步、计划快照失效 |
| Application | JD 版本不可变、状态机非法跳转、关联解绑、并发写 |
| Reflection | 重复去重、修正值传播、跨日沉淀、损坏 memory 恢复 |
| Resume/Flow | 编造硬拦、JD 绑定、节点降级、trace 可重放 |

### 8.2 真 API/LLM 12 例

现有 `tests/test_api.py` 的三个 E2E 只检查“有答案/有 delta”，语义门槛太弱。新增 12 个 live case：

- reference_kb 正常回答 3；
- correct_premise 2；
- KB 拒答 2；
- JD match 1；
- plan/summary 1；
- resume generation 1；
- SSE 正常与中断恢复各 1。

每例必须检查结构、语义要求、来源/动作、request_id、超时和无 5xx，不再只断言字符串长度。

现有 live smoke 可先运行：

```bash
OFFERCLAW_E2E=1 .venv/bin/python -m pytest tests/test_api.py -q
```

但它只能作为连通性检查；新增的 12 例完成后才构成发布级 API 验收。

### 8.3 Playwright 8 条用户旅程

在临时数据根目录启动应用，覆盖：

1. 首页与系统健康；
2. 画像读取/建议审批；
3. 粘贴 JD → 匹配结果；
4. 保存今日计划 → 刷新仍一致；
5. 新增投递 → 绑定 JD version；
6. 复盘 → lesson/画像建议；
7. RAG answer / correct / abstain 三态显示；
8. CareerFlow → trace 查看与重放。

硬门：8/8，无 console error、无 5xx、无真实用户文件变化、截图与 trace 均写仓库外。

---

## 9. Phase 4：全新 Blind A 设计与预冻结

### 9.1 规模

总计 80 题：

- 60 个正例；
- 10 个 `correct_premise`；
- 10 个 `abstain`。

60 个正例按以下方式冻结：

- algorithm/backend/career/llm_app 各 15；
- standard/natural/oral/long-noisy 各 15；
- 至少 20 个 source，任一 source 不超过 10%；
- 每个 anchor 只出现一种问法；
- 至少 20 题来自此前 48 题未覆盖的 source；
- 每题有 grade 0–3 qrels、一个或多个等价 relevant target、`answer_requirements` 和 expected action。

### 9.2 出题与冻结纪律

1. 出题者只看原始文档/metadata，不看当前检索排名、分数、Gate 或系统答案；
2. 先人工检查问题自然度、答案要求、等价证据和负例真实性；
3. 冻结 query/anchor/source/target hash 与 selection manifest；
4. 冻结候选代码 SHA 后才运行系统；
5. 运行完成前不看聚合中间分数；
6. 原始文本仍只存仓库外。

### 9.3 Blind A 发布硬门

本阶段可以完成出题、人工复核和 manifest 冻结，但不能提前运行候选。只有 §10 的性能与故障门禁通过、最终候选 SHA 冻结后，才允许一次性解封。

使用两位裁判中较低的一项作为保守 headline：

| 指标 | GO 门槛 |
|---|---:|
| Candidate coverage | ≥ 90% |
| strict R@1 | ≥ 72% |
| strict R@3 | ≥ 85% |
| strict R@5 | ≥ 88% |
| MRR@10 | ≥ 0.80 |
| nDCG@5 | ≥ 0.82 |
| claim faithfulness | ≥ 95% |
| completeness | ≥ 75% |
| effective completeness | ≥ 70% |
| citation precision / recall | 均 ≥ 75% |
| correct_premise | ≥ 9/10 |
| abstain | ≥ 10/10 |
| 任一风格桶 | 不得出现明显净回退 |

两位裁判 completeness 差距若超过 10pp，不直接判模型失败，但必须人工仲裁 20 条：全部分歧样本优先，不足部分按领域/风格分层随机补齐。人工结果单独报告，禁止用人工只改低分而不改高分。

---

## 10. Phase 5：性能、并发、故障与最终解封

### 10.1 Fast / Quality 必须分档

下一轮性能测试不再只报告一个“RAG 延迟”，而是固定两档：

- `fast`：本地召回 + reranker +轻量 Gate，面试与普通交互默认；
- `quality`：当前 teacher answerability + 三票 Gate，作为质量上界。

若代码尚未提供显式模式，本阶段先实现统一 mode 配置；未设置仍保持当前默认行为，禁止通过测试脚本私自拼环境变量冒充产品模式。

### 10.2 延迟样本

固定 30 题：standard/natural/oral 各 10，包含 answer/correct/abstain。分别测：

- 冷启动首题；
- 单进程常驻 p50/p95/max；
- 并发 1/2/4；
- answerability cache 命中/未命中；
- 模型超时、429、无效 JSON；
- 峰值 RSS 与模型加载时间。

建议门槛：

| 模式 | p50 | p95 | 错误率 |
|---|---:|---:|---:|
| Fast | ≤ 5s | ≤ 10s | 0% |
| Quality | ≤ 30s | ≤ 60s | 0% |

Quality 若仍无法达到 p95 60s，可以保留，但不得作为默认面试演示路径；README 必须继续公开真实延迟。

### 10.3 故障与安全守卫 20 例

- 主模型/HyDE/judge 超时、429、空响应、非法 JSON；
- answerability 全不可用时不改变原排序；
- correct_premise 投票故障按拒答处理；
- cache 损坏、索引 fingerprint 漂移；
- prompt injection、要求泄漏系统提示、要求忽略知识库；
- URL 抓取 SSRF 边界、上传扩展名/MIME/体积；
- path traversal、日志 PII、API key/绝对路径泄漏；
- 并发保存、stale mtime、trace 中断恢复。

硬门：0 secret/原文泄漏、0 静默假成功、0 错误证据放行；可读降级必须带 request_id/diagnostic。

---

## 11. Mac 与高性能电脑分工

| 机器 | 负责 | 不负责 |
|---|---|---|
| Mac（实际产品机） | 最终 API/UI、Fast/Quality 延迟、冷启动、内存、Blind A 发布结果 | 大规模临时调参 |
| 高性能电脑 | 120 行 strict 三跑、批量故障注入、跨机复现、长时间稳定性 | 替代 Mac 的产品延迟 headline |

双机必须使用相同：commit SHA、selection/dataset SHA、index content hash、prompt SHA、模型名和 mode。Windows 结果用于复现与吞吐对照；面试中展示的本地延迟必须来自实际演示 Mac。

---

## 12. 最终发布报告结构

最终 `REPORT.md` 固定按以下顺序，避免只展示好看的数字：

1. 候选 SHA、index、prompt、数据 manifest；
2. 全量测试、doctor、pipeline、产品验收；
3. Development Regression（明确非盲）；
4. Blind A 首次结果；
5. strict ranking 漏斗：Candidate → R@1 → Gate → final action；
6. faithfulness/completeness/citation 双裁判与人工仲裁；
7. Fast/Quality latency、并发、内存；
8. failure taxonomy 与逐类计数；
9. 已知限制和 NO-GO 项；
10. raw artifact 哈希与隐私扫描结果。

面试 headline 建议只保留六项：

```text
Blind strict R@1 / R@3 / R@5
Conservative faithfulness / completeness
Correct-premise / abstain action accuracy
Fast p95 / Quality p95
1430+ full tests + product acceptance
一个真实失败与对应工程修复
```

---

## 13. 推荐执行顺序

### A. 面试最小闭环（优先，约 1 天）

1. 修复 `correct_premise` 动作传递；
2. 建 24 题动作专项集并达到 23/24 以上；
3. 运行 120 行 current-head regression 一次；
4. 跑现有 24/24、全量 pytest、doctor、pipeline、docs；
5. 固化 Fast/Quality 各 15 题延迟；
6. 用当前报告更新面试演示脚本。

完成这一档后，项目已经具备更稳的面试证据，不需要等待 Blind A 才能参加面试。

### B. 完整发布闭环（随后，3–5 天）

1. 扩展 36–48 个确定性产品验收；
2. 增加 12 个强语义 API live case；
3. 增加 8 条 Playwright 用户旅程；
4. 完成 20 条故障/安全守卫；
5. 构建、复核、冻结 Blind A 80；
6. 冻结候选后一次性解封运行；
7. Mac 做最终质量/延迟，PC 做三跑与跨机复现；
8. 发布一份统一报告，不再追加临时实验臂。

---

## 14. GO / NO-GO 总规则

### 立即 NO-GO

- freeze/hash 漂移；
- 全量测试或 24 例现有验收失败；
- 任何 secret、原始个人问题或片段进入公开产物；
- 须拒答错误放行增加；
- correct_premise 修复导致正常 answer 排序回退；
- 任一 judge 缺失却仍发布聚合分数；
- 为提升 Blind A 指标而回改候选后仍沿用同一 Blind A 名称。

### 可以发布但必须明示限制

- 忠实度通过、完整度未过 75%；
- Quality p95 仍高于 60s，但 Fast 模式达标；
- 两位裁判分歧大于 10pp，已经完成人工仲裁；
- current-head development regression 通过，但尚无 organic 规模证据。

### 最终 GO

- L0–L9 全部门禁通过；
- Blind A 达到 §9 门槛；
- Mac Fast/Quality 指标达到或明确分档；
- 双机谱系一致、方向一致；
- 报告同时包含收益、延迟、失败与边界。

---

## 15. 文档维护提醒

现有 `docs/RAG_TESTING_GUIDE.md` 仍包含旧的 pool=20、旧 `eval_rag_answer.py` 和早期 HyDE/门控描述，适合作为历史手册，不应继续作为当前发布命令真源。下一步实施本计划时，应将它更新为本文件的操作型子手册，或在顶部明确标记历史口径。

当前事实源与报告：

- `metrics.json`
- `docs/rag_eval/current_head_20260831/FROZEN_CONFIG.json`
- `docs/rag_eval/current_head_20260831/REPORT.md`
- `docs/rag_eval/answer_quality_v2/PROTOCOL.md`
- `docs/rag_eval/answer_quality_v2/SUMMARY.json`
- `docs/rag_eval/answer_quality_v2/REPORT.md`

本计划完成前，简历和 README 继续使用当前已核验指标，不提前写入任何目标值。
