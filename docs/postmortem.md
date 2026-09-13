# OfferClaw · 技术复盘（Postmortem）

> 用途：记录 V1 阶段的关键技术取舍、踩过的坑、未解决的问题。  
> 面试当问"这个项目最难的部分是什么 / 如果重做你会怎么改"时，从这里挑答案。

---

## 1. 关键技术取舍（Trade-offs）

### 1.1 规则版岗位匹配 vs 端到端 LLM
**选了规则**。  
- LLM 在 JD 长文本上的语义理解优于规则，但 (a) 同一 JD 不同 Prompt 给不同结论，(b) 没法回归测试，(c) Token 成本高
- 把"硬否决"（地点 / Java 主线 / 不接受日常实习）抽到 `target_rules.md` + `match_job.py`，LLM 只在规则放行后做"软评估"
- 代价：对 JD 的语义模糊地带（例如"Java 优先但 Python 也行"）要靠人工运维规则
- **如重做**：保留双通路；增加一个规则覆盖率监控（每周看有多少 JD 走到 LLM 兜底，> 50% 就该重写规则）

### 1.2 文件级 RAG vs 数据库 RAG
**选了文件级**。  
- 项目状态全是 markdown，文件级 + ChromaDB 单机部署够用
- 规模到 100+ markdown 时再考虑迁 PostgreSQL + pgvector
- **如重做**：从一开始就用 `rag_ingest.py --watch` 模式自动监听文件变更（现在是手动跑）

### 1.3 LangGraph vs 自写状态机
**选了 LangGraph**（晚引入）。  
- 一开始 `pipeline.py` 是 Python 函数顺序调用，三个分支后开始变成 if/else 地狱
- LangGraph 的强制"显式 State"反而帮我把隐式状态全暴露
- 代价：BaseMessage 序列化要写适配层（`_msg_to_dict`），原生不支持
- **如重做**：可能直接用 LangGraph，不会先走顺序脚本

### 1.4 FastAPI 何时引入
**选在 RAG 完成后引入**。  
- 早引入 → 还没核心功能就在写接口契约，浪费
- 晚引入 → 来不及做前端 demo
- 时机：RAG 跑出第一个 Recall@5 数字后立刻封 API
- **如重做**：同样时机，但接口的 Pydantic Schema 更早冻结，避免后期改 contract

### 1.5 前端：Streamlit vs Vanilla HTML
**选了 Vanilla HTML**。  
- Streamlit 引入 200+ MB 依赖，跟主项目（FastAPI + 智谱）不搭
- Gradio 同理
- 一个零依赖单文件 `static/index.html`，跟 repo 同步推 GitHub
- 代价：要自己处理 SSE（POST + EventSource 不兼容，被迫 fetch + ReadableStream）
- **如重做**：同样选 Vanilla；若上多用户再考虑 React + Vite

### 1.6 评估集已扩到 58 题（此前仅 8 题）
**此限制已解决**。  
- 早期 8 题黄金集是手工标注，覆盖"硬否决规则 / 三档定义 / 4 周计划格式"等元问题
- 当前评测集已扩至 100 题（精确到目标文件·新版主口径），Recall@5 = 0.98 / MRR = 0.922（100 题主口径下），统计置信度已充分
- **已完成**：扩到 100 题；引入分桶（事实型 / 解释型 / 跨文档组合型）

---

## 2. 踩过的坑

### 2.1 中文 markdown 切 chunk
- 一开始用 token-based（每 256 token 一切），结果中文章节边界被破坏，<!--HIST-->Recall@5 卡在 0.50<!--/HIST-->
- 改为 markdown-header-based（`#` `##` `###` 切），不破坏语义单元，<!--HIST-->立刻 +25 个百分点<!--/HIST-->（一路优化到当前 Recall@5 = 0.98）
- **教训**：RAG 90% 的瓶颈在 ingest，不在 retrieval / rerank

### 2.2 智谱 JWT 时间戳必须毫秒
- 用秒级时间戳 401 鉴权失败；查源码发现要 `int(time.time() * 1000)`
- **教训**：第三方 API 集成必读 SDK 源码或官方示例的 timestamp 实现

### 2.3 LangGraph BaseMessage 序列化
- 原生 `state.dict()` 会把 BaseMessage 字段 dump 失败
- 加 `_msg_to_dict()` 把每个 message 转成 `{"type":..., "content":...}` 再 dump
- **教训**：LangGraph 的 State 字段如果含 LangChain 对象，要自己写 serializer

### 2.4 FastAPI POST + SSE 浏览器侧
- 浏览器原生 `EventSource` 只支持 GET
- 前端用 fetch + reader.read() + TextDecoder 自己解析 `event:`/`data:` 块
- **教训**：SSE 标准没说 POST 不行，是浏览器实现限制

### 2.5 Windows 终端编码 cp936
- subprocess.stdout 默认 cp936，print 含 emoji 直接 `UnicodeEncodeError`
- 全部 emoji 替换 ASCII，subprocess decode 加 `errors="replace"`
- **教训**：Windows 上的 Python 项目，默认假设 UTF-8 是错的

### 2.6 用户画像 API 硬编码
- 早期 `/api/profile` 把 `name = "示例用户"` 写死在代码里
- 后来加 `_parse_profile()` 用正则从 `user_profile.md` 抽
- **教训**：任何"演示用"的硬编码必须在 PR 标题里加 `XXX: hardcoded for demo`，避免上线时漏改

### 2.7 换生成模型当天，计划日层解析被"笔迹差异"打穿（2026-08-08）
- 主模型 qwen → gpt-5.6 后，同一份 plan_prompt 生成的计划把日块头从 `D1（08-08 周六）` 写成
  `### D1（08-08 周六）`、小节标签写成加粗 bullet；按行首 `D1（` 匹配的解析器全部落空 →
  「今日计划」整条消失，连日期确定性重写（`normalize_plan_dates`）也一并失效
- 本质：**代码消费的 LLM 输出就是一个接口，而自由 Markdown 是没有 schema 的接口**。
  换模型 = 输出格式的分布漂移；在单一模型笔迹上调通的正则是过拟合
- 修复 + 保障措施（把"保证 JSON 输出"的思路移植到 Markdown 场景，三件套）：
  1. **输出契约进 prompt**：`plan_prompt.md` §5.2 明写机器解析契约（日期行字样/编号样式不可改，装饰可容忍）
  2. **生成后校验门（fail-visible）**：落盘后立即用解析器数日块，`daily_days` 随响应/done 事件
     返回，为 0 时前端 toast 警告——格式被打穿必须当场可见，不静默
  3. **读端容忍 + 风格矩阵回归**：解析器按 Postel 法则容忍已知装饰（###/**/半角括号/顿号编号，
     可选任务用状态机区分标签与内容）；参数化测试把"已知笔迹矩阵"钉死，且 parse 与 normalize
     两把正则用同一矩阵测一致性；UI 规则改为"计划存在则今日条永不消失，解析不出显示暂不可用+原因"
- 为什么**没有**上 JSON 结构化输出（有意取舍）：计划是人读文档（演进说明/依赖检查等叙事段有价值）；
  经代理的 JSON mode 各家支持不一；24 天长输出的 JSON 转义/截断风险不低。文档为主 + 关键字段
  （日期）代码重写 + 容忍解析 + 校验门，是当前体量的性价比解；若笔迹漂移继续咬人，升级路径是
  "LLM 出 JSON、代码渲染 Markdown"
- **教训**：凡是代码要解析的 LLM 输出，写第一个正则之前先回答三问——契约在哪？校验门在哪？换模型会怎样？

---

## 3. 未解决的问题

| 问题 | 影响 | 计划 |
|---|---|---|
| 评估集扩集（已完成，<!--HIST-->8 题<!--/HIST--> → 100 题） | 已解决：指标统计意义已充分 | 已扩到 100 题 + 分桶 |
| 没有 LLM Reranker | 长尾召回偏低 | V2 加 BGE-reranker |
| applications.md 是手维护表 | 真实投递跟踪难规模化 | V2 迁 SQLite |
| 没有 CI / Docker | 无法证明跨环境可复现 | V2 GitHub Actions + Dockerfile |
| 单用户 | 不能多人共用 | V3 用户隔离 |
| Memory.json 无 TTL | 长期会膨胀 | V2 加过期 + 摘要压缩 |
| 没有自动 JD 失效检测 | 可能匹配已下架的岗位 | V2 加 URL 健康检查 |

---

## 4. 给"下一个 V2 起点"的提示

如果你（或一年后的我）要继续做 V2：

1. **RAG 评估集已扩到 58 题**（此前仅 8 题），指标已有统计意义，后续优化不再是凭感觉
2. **再加 LLM Reranker**（BGE-reranker-v2-m3 或 GLM-4 二阶段），不要先调 chunk size
3. **applications.md 迁 SQLite** 之前先把字段冻结；当前字段已经够 V2 用
4. **CI 在 doctor.py + verify_pipeline.py 之上加一层 GitHub Actions**，跑 pytest + verify
5. **Docker 不要一开始打全栈镜像**，先 `python:3.13-slim` + 项目代码，ChromaDB 用 volume 挂载

---

## 5. 我从这个项目学到的（个人）

- **闭环优于功能数**：6 周里我两次想加新技术（多模态 / Function Calling Marketplace），都砍了
- **deterministic first**：能用规则解的别交给 LLM
- **eval 是 RAG 的必须项**：没有黄金集的 RAG 系统等于在黑箱调参
- **状态分层**：用户层 / 系统层 / 运行时层一旦分清，整个项目的可维护性上一个台阶（见 `DATA_CONTRACT.md`）
- **工程可信度 ≠ 功能数**：`doctor.py` + `verify_pipeline.py` 比再加 3 个 FastAPI 接口更能让别人相信这个项目能跑
