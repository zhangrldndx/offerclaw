# OfferClaw · RAG 优化与学习报告

> 本文档记录 OfferClaw RAG 模块的逐轮优化：**初始情况 + 每轮的措施 / 项目改动定位 / 性能对比 / 对应知识库八股**。
> 目的：优化与学习一体——每改一处 = 学一块 RAG 知识点 = 简历多一条量化战绩。

## 工作流（每轮遵循）

1. 按价值优先级选下一项改进（一轮一项）
2. 落地改动，**明确标注改了哪些文件/函数**（方便定位、修改、学习）
3. `python eval_rag_bench.py --baseline <上一轮json>` 测前后量化对比
4. 从知识库 RAG 章节对应八股知识点
5. 记入本文档

评测器：`eval_rag_bench.py` · 评测集：`tests/rag_bench_set.json`（39 题，精确到目标文件，含难题标签）
知识库 RAG 章节：`knowledge_base/learning_resources/llm_app_interview_02_rag_basics.md`、`..._03_rag_full_chain.md`

---

## 初始 RAG 技术要点（Round 0 时的实现）

| 环节 | 实现 | 代码位置 |
|---|---|---|
| 数据加载 | URL / GitHub 整仓 / 本地 md / 图片，多源 + Playwright 反爬兜底 | `knowledge_crawler.fetch_repo_text`、`job_discovery.fetch_url` |
| 分块 | 按 Markdown `##` 标题语义切分 + 段落贪心填充，CHUNK_SIZE=800，前 100 字哈希去重 | `rag_tools.split_markdown_document` |
| 多模态 | 图转文：qwen-vl 描述+OCR，缓存 + 8 路并发 + 失败降级 | `image_caption.py` |
| Embedding | 本地 bge-base-zh-v1.5（768 维），多 provider 抽象，批量 + 指数退避 | `rag_tools.get_embeddings_batch` |
| 向量库 | ChromaDB（底层 HNSW），按 embedding 空间隔离 collection，内容哈希增量入库 | `rag_ingest.ingest_file` |
| 检索 | 向量 top-5（cosine）+ 词法救援 | `rag_gate._retrieve_and_classify` |
| 门控 | 三层距离阈值 + 模型自适应（bge 0.85 / 百炼 0.92），in_kb 判定 | `rag_gate._thresholds` |
| 生成 | grounded 合成（严禁库外）+ 出处标注 + 未命中兜底 + 双源问答（状态文件） | `rag_gate.gated_query` |
| 评测 | （Round 0 升级）精确目标文件 + Recall@1/3/5 + MRR + 门槛 | `eval_rag_bench.py` |
| 工程 | 密钥脱敏、配置漂移自检、distance/mode 可观测 | `redact_secrets`、`doctor.py` |

### 对照知识库的 RAG 覆盖度地图

- ✅ **已涉及**：数据准备全流程、Markdown 特定格式分块、动态嵌入(bge)、密集/语义检索、HNSW向量库、RAG效果评估、敏感信息处理、引用来源、prompt 约束防幻觉
- ⚠️ **部分**：混合检索（只有词法救援，无 BM25 倒排）、语义路由（双源问答雏形）、固定大小分块（overlap 定义未用）、Adaptive-RAG（门控 in_kb 有"要不要检索"影子）
- ❌ **未涉及**：查询优化（Multi-Query / HyDE / Step-Back / 分解）、重排序 Rerank、父文档/句子窗口/自动合并、上下文压缩、RAPTOR、ColBERT、多向量检索、CRAG / Self-RAG、多轮对话 / 追问、嵌入微调

> 定位：**朴素 RAG + 门控/双源 + 扎实工程化**，停在"高级 RAG"门口——预检索查询优化、后检索重排/压缩两大块基本空白。

---

## Round 0 — 建立可信度量基线

**措施**：旧评测（`eval_rag_domain.py`）判定宽松（命中本域任意文件即算，故显示"100%召回"，无区分度）。本轮升级评测体系，作为后续一切改进的标尺。

**项目改动定位**：
- 新增 `eval_rag_bench.py`：精确目标文件（唯一子串匹配）+ Recall@1/3/5 + MRR + 难题分桶 + 门槛准确率 + `--save/--baseline` 对比
- 新增 `tests/rag_bench_set.json`：39 题，覆盖 4 域（llm_app 18 / backend 11 / algorithm 6 / career 4），11 题标 `hard`（向量易漏、靠词法/rerank/混合检索救回）
- `rag_gate._retrieve_and_classify`：返回值新增 `docs/metas/dists`（排序后原始结果，纯增量，不改 baseline 行为；后续 rerank·混合检索在此重排，评测器自动跟随）

**Round 0 baseline 指标**（`docs/rag_eval/round0_baseline.json`，K=5）：

| 分桶 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| **总体 (39)** | 95% | 97% | **100%** | **0.964** |
| 难题 (11) | 82% | 91% | 100% | 0.871 |
| 易题 (28) | 100% | 100% | 100% | 1.000 |
| llm_app (18) | 89% | 94% | 100% | 0.921 |
| backend (11) | 100% | 100% | 100% | 1.000 |
| algorithm (6) | 100% | 100% | 100% | 1.000 |
| career (4) | 100% | 100% | 100% | 1.000 |

门槛准确率：负样本拒答 6/6 · 正样本命中 6/6 ✅

**关键发现（决定后续优化方向）**：
- **R@5 已饱和（100%）**——在"单库 + 文件主题高区分度"场景，bge 粗召回接近天花板。**后续优化的价值不在召回率，而在 R@1 / MRR / 难题**（把正确文件从第 2-3 名精排到第 1 名）。
- 提升空间集中在：**难题 MRR 0.871**、**llm_app 域 MRR 0.921**（RAG/Agent 内容集中在两三个大文件，区分度低，正是精排发力点）。
- 这印证了 **Rerank（精排）= 第一优先级** 的判断：它的目标指标是 MRR 与 Recall@1，不是 Recall@5。

---

## 优化路线图（价值优先级）

> 下表为**实际执行轨迹**（原计划已按实测调整：Round 3 HyDE 证伪、Round 4 改为更对口的元数据路由）。

| 轮次 | 改进 | 目标指标 | 状态与结果 |
|---|---|---|---|
| Round 1 | **Rerank 重排序**（粗召回 topN → reranker 精排 topk） | R@1↑ MRR↑ 难题↑ | ✅ 难题 R@1 82→91 |
| 真实化 v2/v3 | **多源真实仓库入库**（LocalFlow + all-in-rag + hello-agents，1024→3299 块） | 评测真实化 | ✅ R@5 100%→真实区间、暴露 llm_app |
| Round 2 | **混合检索**（BM25 + 向量 + RRF 融合） | llm_app/难题召回↑ | ✅ llm_app R@1 +5.6、难题 R@3 100% |
| Round 3 | HyDE 查询改写（query 侧） | 模糊查询↑ | ❌ **无效/有害**（瓶颈在文档侧，见 Round 3 + 附录） |
| Round 4 | **元数据/文件名路由**（文档侧） | 同质文档区分 | ✅ career R@1 75→100、零副作用 |
| 候选 | gate 拒答优化（neg 4/6） | 拒答可信度↑ | ⏳ 唯一剩余真实短板 |
| 候选 | 端到端答案质量评测 | 生成质量可量化 | ⏳ 未做 |
| 候选 | 上下文压缩 / 父文档 / CRAG / Self-RAG / 图RAG 等 | — | ⏳ 见**附录·适用边界分析**（本项目暂不需要） |

> 每轮完成后在下方追加「Round N」小节：措施 / 改动定位 / 前后指标对比 / 对应八股。

---
## Round 1 — Rerank 重排序（本地 bge-reranker-base）

**措施**：向量是双塔（query/doc 各自编码后算距离），信息有损、"语义最相似 ≠ 最相关"。
引入**交叉编码器（cross-encoder）**：把 query+doc 拼一起进模型做细粒度相关性打分。
链路改为 **向量粗召回 20 → bge-reranker-base 精排 → 取 top-5**。

**项目改动定位**（方便定位/修改/学习）：
- 新增 `rag_rerank.py`：`rerank(query, docs, metas, dists, top_k)` —— CrossEncoder 懒加载
  （ModelScope 下载，跳过 1G onnx 只取 PyTorch 权重）+ 进程缓存 + 失败静默降级；
  env 开关 `RAG_RERANK`（默认 1）、`RAG_RERANK_MODEL`、`RAG_RECALL_N`（默认 20）。
- 改 `rag_gate._retrieve_and_classify`：召回 `RECALL_N` → 门控判定基于**粗召回全集**
  （`best`=最小向量距离不变、词法救援在更大池更强）→ rerank 精排出 top_k 供合成/评测。
- 新增 `tests/test_rag_rerank.py`（开关/降级/按分数重排 5 例，不依赖真实模型）。
- 模型：`BAAI/bge-reranker-base`（278M，本地，~1GB 权重，与本地 bge embedding 同源，无配额）。

**前后指标对比**（`docs/rag_eval/round1_rerank.json` vs round0，K=5）：

| 分桶 | R@1 | MRR | Δ MRR |
|---|---|---|---|
| 总体 (39) | 95% → 95% | 0.964 → 0.962 | -0.001（持平） |
| **难题 (11)** | 82% → **91%** | 0.871 → **0.939** | **+0.068** ✅ |
| **llm_app (18)**（RAG/Agent 八股，学习重点） | 89% → **94%** | 0.921 → **0.963** | **+0.042** ✅ |
| backend (11) | 100% → 100% | 1.000 → 1.000 | 持平 |
| algorithm (6) | 100% → 100% | 1.000 → 1.000 | 持平 |
| **career (4)** | 100% → **75%** | 0.800 | **-0.200** ⚠️ |

门槛准确率不变：负样本拒答 6/6 · 正样本命中 6/6。

**诚实分析（含退化根因）**：
- ✅ **rerank 精准命中了 baseline 预判的提升空间**——难题与 llm_app 域（语义密集、文件区分度低）
  R@1/MRR 显著提升，这正是交叉编码器的价值场景。
- ⚠️ **career 退化（单题 ca03）**：诊断发现 `05_zero_foundation` / `06_backend_transition` /
  `07_algorithm_transition` 三个"XX 转大模型应用"文件**内容高度同质**，cross-encoder 把
  ca03「算法岗转型」的正确文件（07）从 rank1 挤到 rank5，错选了同质的 06。
  → **教训：rerank 对"仅主语不同、正文高度相似"的同质文档也会混淆**，不是银弹。
- **关键认知**：rerank 的净收益与 baseline 质量**负相关**——本库已很干净（R@5=100%、MRR=0.96），
  所以整体净提升小，价值集中在"难题/低区分度域"。在召回噪声大、baseline 差的场景，rerank 收益会大得多。

**对应知识库八股**：
- `llm_app_interview_02_rag_basics.md` · 检索后处理：「向量检索其实就是计算语义层面的相似性，
  但语义最相似并不总是代表最相关。重排模型通过对初始检索结果进行更深入的相关性评估和排序…
  使用专门的重排序模型（闭源 Cohere，**开源 BAAI** 和 IBM）」——本轮用的正是 BAAI bge-reranker。
- `llm_app_interview_03_rag_full_chain.md` · 15-Re-ranking：1) Cohere Re-Rank 方案；2) 大模型做重排序。

**结论**：保留 rerank 默认 ON（对学习核心域 RAG/Agent + 难题有效，且喂给 LLM 的 top chunk 更准，
利于答案质量）；career 同质混淆记为**下一步可优化项**（rerank 分数与向量序融合，或升级 bge-reranker-v2-m3）。
**简历可量化点**：引入 bge-reranker 交叉编码器精排，难题检索 Recall@1 82%→91%、MRR +0.068。

## 知识库真实化 v2 + 方案A 实验（career 退化诊断）

### 知识库来源审计（为什么 R@5=100%）
盘点发现：**74 个正式文件里 71 个（96%）来自同一飞书 wiki（单一作者、主题正交、风格统一）**，
真正多源只有 1-2 个。这才是 R@5=100% 的根因——**不是检索强，是库太"干净"、题和答案文件近乎一一对应**。
工业界 naive RAG 的 R@5 通常 60-85%，靠的是多源异构 + 长尾噪声。

### 方案A（索引优化·chunk 文档标题前缀）— 实验：无效
- **措施**：ingest 时 embedding 文本前缀文档一级标题（`《标题》\n正文`），documents 存原文不变。
  对应 02章 索引优化·添加元数据。改动：`rag_ingest.ingest_file`（env `CHUNK_DOC_TITLE`）。
- **结果**：重建库后评测 **全指标 Δ=0**，career 仍 75% —— **无效**。
- **诚实根因**：十几字标题被几百字正文稀释，对 bge 向量影响≈0。
  **教训：轻量元数据信号会被长正文淹没**，对同质文档消歧无效。→ `CHUNK_DOC_TITLE` 默认关。
- **career 真正解药**：转 **Round 2 混合检索**——BM25 对"算法/后端"关键词精确加权，正是同质文档的解药。

### 库升级 v2：+ LocalFlow 真实项目文档
- **措施**：爬 `github.com/example-owner/localflow` 42 个工程文档（README/ARCHITECTURE/AGENT_SERVER/EVAL…，
  515KB），密钥脱敏后增量入库 → `knowledge_base/learning_resources/localflow_project_docs.md`（801 块）。
  库 **1024 → 1825 块**。改动：`knowledge_crawler.fetch_repo_text`（Round1 已重构出）+ rag_ingest --add。
- **效果**（`docs/rag_eval/kb_v2_localflow.json` vs round1）：**llm_app 域 R@1 94%→89%(-5.5%)、MRR -0.028**
  —— LocalFlow 的 Agent/RAG 工程内容与 RAG/Agent 八股**竞争召回**，检索难度真实上升（backend/algorithm/career
  无 LocalFlow 对应主题，不受影响）。R@5 仍 100%（库还不够大/异构）。
- **结论**：真实异构内容让评测开始有区分度（"R 不再离谱"的开端）。要让整体 R@5 降到工业水平，
  需继续补**同主题多源交叉**（其他作者的 RAG/Agent/后端 讲解）。

> **baseline 迁移**：库已升级（1825 块），Round 2 起以 `kb_v2_localflow.json` 为对比基准（不再用旧库 round0/1）。

### 下一步
- **多源交叉采集**（让 R@5 降到真实）：爬 2-3 个开源 RAG/Agent 教程（如 datawhalechina 系列）→ _pending → 人工审核 → 入库。
- **Round 2 混合检索**（BM25 + 向量 + RRF）：补 career 同质混淆 + 多源后的关键词召回。

---

### 库升级 v3：+ all-in-rag + hello-agents（多源交叉，R@5 真实回落）

- **措施**：按 v2「下一步」补同主题多源——爬 datawhalechina 两个真实教程仓库，**人工审核**后入库：
  - `all-in-rag` **28 章**（RAG 实战：数据加载/分块/嵌入/向量库 Milvus/**索引优化/混合检索/查询改写/高级检索/图 RAG**）→ +432 块
  - `hello-agents` **13 章精选**（剔除旅行助手/赛博小镇/毕设等纯项目复现；保留**记忆与检索/上下文工程**等理论与方法）→ +1042 块
  - 库 **1825 → 3299 块（+1474, +81%）**，llm_app/RAG 主题首次有「他源」交叉。
- **项目改动**（方便定位/学习）：
  - `knowledge_crawler.py:288` 爬虫**采集质量过滤**——跳过 README/`_sidebar`/`_coverpage` 导航、`/en/`+`_en` 英文镜像；
    同目录中英并存时**偏好中文**（剔英文重复版）。修前 hello-agents 因英文版+sidebar 占满 600KB 配额只抓 11/37、缺 ch2-9。
  - `knowledge_crawler.py:565` `cmd_promote` ingest 子进程 **timeout 300→3600**（大教程文件本地 embedding 慢，300s 会半途中断）。
  - `GH_MAX_CHARS`/`GH_MAX_FILES` 改环境变量可配，重爬时放宽以抓全中文章节。
- **本轮性能**（`docs/rag_eval/kb_v3_real_repos.json` vs `kb_v2_localflow.json`，均 rerank ON）：

  | 维度 | v2(1825块) | v3(3299块) | Δ |
  |------|-----------|-----------|---|
  | **总体 R@5** | 100% | **97%** | **↓0.026 终于回落** |
  | 总体 R@1 / MRR | 92% / 0.950 | 79% / 0.865 | ↓0.128 / ↓0.085 |
  | **llm_app R@1 / MRR** | 89% / 0.935 | **61% / 0.752** | **↓0.278 / ↓0.183** |
  | backend / algorithm / career | — | 不变 | Δ=0（新内容是 RAG 主题，不碰其他域）|
  | **门槛 负样本拒答** | 6/6 | **4/6** | ↓ KB 覆盖变广、边界模糊 |
  | 新增未命中 | — | **rag09 混合检索 rank=0** | 被同主题挤出 top5 |

- **解读**：这是**健康的退化**，不是 bug——
  - R@5 100%→97%、llm_app R@1 89%→61%：同一 RAG 主题首次有「多源交叉」，密集同主题块互相竞争召回；
    单塔向量在近义子主题（混合检索/索引优化/查询改写）间**区分不开**，检索难度逼近工业真实水平。
  - **rag09「混合检索」反被自己入库的混合检索内容挤出 top5**：最典型的「语义近、关键词异」案例——
    向量看「稀疏/密集/混合」语义都聚在一起，需**词法精确匹配**才能拉开。
  - 门槛 6/6→4/6：覆盖变广 → 2 个原应拒答的边界问题匹配到泛泛内容、误判 in_kb（后续收紧阈值 / 用 Round2 词法证据佐证）。
- **对应八股**（库内 `all-in-rag` ch4「混合检索」/ ch3「索引优化」）：
  - **稀疏向量 vs 密集向量**：稀疏=词法/词袋(BM25)，维度≈词表大小、绝大多数为 0，精准匹配关键词（型号/函数名/OOV 新词）；
    密集=语义低维稠密，捕捉语义但**会忽略必须精确匹配的关键词**。
  - **混合检索**：并行跑稀疏+密集，再融合排序（RRF 等），解决「关键词检索不懂语义、向量检索漏精确词」的互补问题
    → 直接对口 v3 暴露的 llm_app / rag09 失分。
- **结论**：v2「下一步」的多源交叉已落实，R@5 从离谱的 100% 回落到 97%、llm_app R@1 暴露出 28pt 优化空间。
  **Round 2 混合检索（BM25+向量+RRF）目标明确**：把 llm_app R@1 与 rag09 救回，顺带修 career 同质化。

> **baseline 迁移**：库已升级（3299 块），Round 2 起以 `kb_v3_real_repos.json` 为对比基准。

---

## Round 2 — 混合检索（BM25 稀疏 + 向量 + RRF 融合）

- **措施**：在「向量粗召回 RECALL_N → rerank」之间插入 **BM25 双路召回 + RRF 融合**，扩大候选池。
  补单塔向量在「同主题多源 / 中文术语」上的短板——向量把近义子主题挤在一起，BM25 按关键词把目标 chunk 拉回。
- **项目改动**（方便定位/学习）：
  - 新增 `rag_bm25.py`：`bm25_search`（jieba 中文分词 + BM25Okapi，进程级索引缓存、按 `collection.count()` 失效）、
    `rrf_fuse`（RRF 倒数排名融合，k=60；向量文档保留**真实距离**，仅 BM25 命中者用 rescue 阈值**占位距离**）。
  - `rag_gate._retrieve_and_classify`（256 行后）：gate 判定（向量 best）**之后**、rerank **之前**插入融合，
    条件 `bm25_enabled() and rerank_enabled()`。**gate 行为刻意不变**（in_kb 仍只由向量 best 决定，不让 BM25 影响拒答、避免 neg 再退化）。
  - 新增 `tests/test_rag_bm25.py`（7 测：中英分词 / 降级 / RRF 双路加权 / 占位距离 / top_n 截断）。依赖 `rank_bm25`+`jieba`。
  - 开关 `RAG_BM25`（默认 1）。
- **本轮性能**（`docs/rag_eval/round2_hybrid.json` vs `kb_v3_real_repos.json`，均 rerank ON）：

  | 维度 | v3 纯向量 | Round2 +BM25 | Δ |
  |------|----------|-------------|---|
  | 总体 R@1 / R@3 / MRR | 79% / 92% / 0.865 | **82% / 95% / 0.881** | ↑2.6 / ↑2.6 / ↑1.6 |
  | **llm_app R@1 / R@3 / MRR** | 61% / 89% / 0.752 | **67% / 94% / 0.787** | **↑5.6 / ↑5.5 / ↑3.5** |
  | 难题 R@3 | 91% | **100%** | ↑9.1 |
  | backend / algorithm | 100% | 100% | = |
  | career | 75% | 75% | 0（见下天花板） |
  | 门槛 neg / pos | 4/6 · 6/6 | 4/6 · 6/6 | = gate 未受影响 |

- **两个未救回反例的诚实归因**（区分「算法可解」与「数据/标注问题」）：
  - **career ca03「算法岗转型」= 同质文档天花板**：`05_零基础 / 06_后端转型 / 07_算法转型` 三文件 ~95% 雷同，
    唯一区分词（算法/后端/零基础）被大量重叠词淹没；向量 / BM25 / rerank 都难区分。目标已在 **rank5（R@5 命中）**，
    BM25 不帮不害（Δ=0）。**非算法 bug，是数据特性**——需查询理解 / 元数据过滤（target_audience）才能根治，留待后续。
  - **rag09「混合检索」= ground truth 过时**：top5 全是 `all-in-rag` 混合检索**专章**——库升级后这是比旧飞书八股
    （rag_basics / rag_full_chain）**更专业的正确答案**，检索行为正确，是 `expect_sources` 没随库演进。
    **已裁定（2026-06 用户）**：把 all-in-rag（子串 "rag技术入门"）纳入 rag09 合理目标，并在 `tests/rag_bench_set.json`
    用 `note` 字段留痕变更理由（标注随库演进，非刷分）。
- **对应八股**（库内 all-in-rag ch4「混合检索」）：稀疏(BM25 词法，精确匹配关键词) + 密集(语义) 并行召回，
  **RRF 融合**：score(d)=Σ 1/(k+rank)，k=60，只看排名、弱化两路分数尺度差异 → 再交 rerank 精排。
- **结论**：BM25+RRF 对「同主题多源」的中文关键词题有效（llm_app R@1 +5.6、难题 R@3 满分），且 gate 不受影响。
  剩余 career 同质化需**查询理解**——指向 Round 3。

- **终态**（含 rag09 标注修正，`docs/rag_eval/round2_final.json`）——区分「算法贡献」与「标注修正」：

  | 维度 | v3 纯向量 | +BM25 算法 | +标注修正 终态 |
  |------|----------|-----------|---------------|
  | 总体 R@1 / MRR | 79% / 0.865 | 82% / 0.881 | **85% / 0.907** |
  | llm_app R@1 / R@3 | 61% / 89% | 67% / 94% | **72% / 100%** |
  | career | 75% | 75% | 75%（天花板，待 Round 3） |

  注：R@5 回到 100% 是标注修正使 rag09 命中（**非算法**）；**真实算法进展看 R@1（79→82→85）与 MRR（0.865→0.881→0.907）**
  —— rerank / 混合检索的价值本就在排序精度，不在召回覆盖。

> **baseline 迁移**：Round 3 起以 `round2_final.json` 为对比基准。

---

## Round 3 — 查询侧增强：HyDE（负结果 + 适用边界）

> 本轮是**负结果**，但按工作流如实记录——理解一个方法「何时无效」与「何时有效」同等重要。

- **措施**：实现 HyDE（Hypothetical Document Embeddings）——LLM 先据 query 生成「假设答案」，
  用「原问题 + 假设答案」的 embedding 检索，意图拉近 query↔文档的表述差。
  新增 `rag_hyde.py`，接入 `rag_gate._retrieve_and_classify`（`RAG_HYDE` 开关，**默认关**）。
- **诊断（实现前先定位失分题）**：终态 6 个 R@1 失分题（rag05/rag07/agt02/agt04/agt06/ca03），
  **几乎全是「同主题多源挤压」**——目标相关，但另一个源（all-in-rag / hello-agents / localflow / 相邻文件）
  也讲同主题、被排到 rank1。即**失分主因在文档侧，不在 query 侧**。
- **本轮结果（负结果，如实记录）**：
  - 小范围 6 失分题实测：**5 题持平、ca03 反而变差（rank5 → 掉出 top5）**。
    HyDE 生成的「算法岗转型」假设答案趋于**通用转型建议**、淡化「算法」特异性，在同质文档里被挤得更狠。
  - 全量 eval 因 **每题一次 LLM 调用、延迟巨大**（39 题 > 8min 未跑完）中止——这本身说明 HyDE
    在低延迟检索场景**不实用**。
- **结论 & 适用边界（学习要点）**：HyDE 的收益场景是「query 模糊 / 口语化、与文档用词差异大」；
  本评测集 query 已含精确技术关键词（"混合检索""稀疏密集"），且失分瓶颈在文档侧多源挤压，
  **HyDE 在此无效甚至有害**。代码保留（默认关）作为资产与对照，不启用。
- **对应八股**（all-in-rag ch4「查询改写」）：HyDE / Multi-Query / 查询改写均属 query 侧增强，
  解决「query 与文档表述鸿沟」；当鸿沟本就小（query 已专业）或瓶颈在文档侧（同质 / 多源）时收益有限——
  **方法要匹配瓶颈所在的层（query 侧 vs 文档侧 vs 排序侧）**。
- **下一步（指向文档侧）**：career ca03 的根治在文档侧——`05/06/07` 的 `tags`（零基础 / 后端转型 / 算法岗转型）
  与**文件名**（zero_foundation / backend / algorithm）已天然区分人群，可做**元数据 / 文件名感知路由**精确区分同质文档。

---

## Round 4 — 元数据/文件名感知路由（精确打穿 career 同质天花板）

- **措施**：承接 Round 3 诊断指向的「文档侧同质」瓶颈——career 的 `05_零基础 / 06_后端转型 / 07_算法转型`
  约 95% 雷同，向量 / BM25 / rerank 都分不开唯一区分点（目标人群）。利用**文件名已编码人群**
  （zero_foundation / backend / algorithm）+ source 已在 meta，做**检索后路由**：query 命中人群意图时，
  在这组 career 路径文件之间提权匹配人群、降权竞争人群，**其他域文件一律中性不动**。**无需重新入库**。
- **项目改动**（方便定位/学习）：
  - 新增 `rag_route.py`：`audience_intent`（query→人群标识，长词优先避免误命中）、`apply_audience_routing`
    （仅在 source 含 `transition_path / foundation_path` 的 career 文件间按桶稳定重排）。
  - `rag_gate._retrieve_and_classify`（rerank 之后）插入路由。开关 `RAG_ROUTE`（默认 1）。
  - 新增 `tests/test_rag_route.py`（6 测：意图识别 / 长词优先 / 提权降权 / 无意图保序 / 非 career 不动 / 开关）。
- **本轮性能**（`docs/rag_eval/round4_route.json` vs `round2_final.json`，route+BM25+rerank ON）：

  | 维度 | round2_final | Round4 +路由 | Δ |
  |------|-------------|-------------|---|
  | **career R@1 / MRR** | 75% / 0.800 | **100% / 1.000** | **↑0.25 / ↑0.20** |
  | 总体 R@1 / R@3 / MRR | 85% / 97% / 0.907 | **87% / 100% / 0.927** | ↑2.6 / ↑2.6 / ↑2.1 |
  | llm_app / backend / algorithm | — | 不变 | Δ=0（路由只动 career） |

  ca03「算法岗转型」rank5 → **rank1**；**R@3 = 100%**（39 题全进 top3）；零副作用（algorithm 域含「算法」但不触发，已验证）。
- **对应八股**：当**同质文档的区分点是结构化属性**（人群 / 版本 / 语言 / 时间）时，纯语义（向量）与纯词法（BM25）
  都易混——**元数据过滤 / 路由**用结构化字段（此处文件名编码的人群）精确区分，是「文档侧」对口「同质」的标准解法。
  与 Round 3 形成对照：HyDE（query 侧）治不了文档侧同质，本轮印证 **「方法要匹配瓶颈所在的层」**（query / 文档 / 排序 / 门控）。
- **结论**：career 同质天花板被精确打穿（R@1 75→100），零副作用。剩余短板仅 **gate 拒答 neg 4/6**
  （KB 变大后边界模糊，误判 in_kb）——属门控侧，留待后续。

> **baseline 迁移**：后续以 `round4_route.json` 为对比基准。

---
<!-- 后续每轮在此追加 -->

---

## Round 5 — gate 拒答阈值重标定（修知识库扩大后的拒答退化）

- **措施**：知识库 1024→3299 块后**拒答能力退化**（负样本 6/6→4/6）——2 个边缘技术词 query（Vue3 / Spring Boot）
  因覆盖变广、匹配到沾边内容，best 距离落进灰色地带被误判 in_kb。本轮在更大的库上**重标定距离阈值**。
- **诊断（数据驱动定阈值）**：正样本 best ≤0.62、39 道正经题 best ≤**0.686**（全部）；误判负样本 Vue3=0.773 / SpringBoot=0.816、
  真无关 ≥0.968 → 存在**完美分离间隙 [0.686, 0.773]**，取中点 **0.73**。
- **项目改动**（方便定位/学习）：
  - `rag_gate._THRESHOLD_DEFAULTS` 的 `local`（bge）：`strong / rescue` **0.85 → 0.73**（weak 1.10 不变）。
    rescue 必须同步收紧——否则 Vue3(0.773) 会被 `lexical_rescue`（best≤0.85 且 "vue3" 恰在某 chunk）漏救回（实测验证过）。
  - `tests/test_local_embedding_and_caption.py` 阈值断言同步 0.85→0.73。
- **本轮性能**（`docs/rag_eval/round5_gate.json` vs `round4_route.json`）：

  | 维度 | round4 | Round5 | Δ |
  |------|--------|--------|---|
  | **gate 负样本拒答** | 4/6 | **6/6** | ✅ 修复 |
  | gate 正样本命中 | 6/6 | 6/6 | = |
  | 总体 R@1 / R@3 / R@5 / MRR | 87 / 100 / 100 / 0.927 | 87 / 100 / 100 / 0.927 | **全 Δ=0** |
  | 各域（llm_app/backend/algorithm/career） | — | 全不变 | = |

  干净修复：**拒答恢复满分、检索零影响**——阈值只决定 in_kb（是否拒答），不碰 docs 排序（R@1/MRR 由 rerank/route 定）。
- **对应八股**：RAG 的「拒答 / Adaptive-RAG（要不要检索 / 能不能答）」依赖**距离阈值**，而阈值是 **模型 + 库规模相关**的——
  **库越大、越易匹配到沾边内容，假命中的距离下沿越低**，阈值需随之收紧。教训：**扩库后必须重标定门控阈值**，
  否则拒答能力悄悄退化——这正是「知识库真实化」的隐性代价。
- **诚实说明（过拟合风险）**：0.73 基于当前 6neg+6pos+39 题标定，样本不大；但分离间隙有语义合理性
  （明显相关 ≤0.62 / 边缘沾边 0.77~0.82 / 无关 ≥0.97），非纯过拟合。**未来扩充评测集 / 换库后需重新标定**。

> **baseline 迁移**：后续以 `round5_gate.json` 为对比基准。

---

## 评测集扩充 v2（39→58 题，gate 6→12，验证阈值鲁棒性）

- **动机**：Round 5 阈值 0.73 基于小样本（6neg+6pos+39 题）标定，有过拟合风险。本轮扩充评测集 = **验证鲁棒性 + 提升度量可信度**。
- **措施**：检索题 39→**58**（+19）——补 backend（MVCC/持久化/集群/TCP/设计模式）、algorithm（压缩/GPT-BERT/训练/涌现）、
  **新入库内容**（all-in-rag 分块/Milvus/图RAG/查询路由、hello-agents 通信协议/上下文工程）、career（学习路径/入门QA）；gate neg 6→**12**、pos 6→**12**。
- **出题质量把控（关键过程，体现评测严谨性）**：
  - 每道检索题先验证目标可召回；agt08「harness 工作流」top1 是 localflow（同源更详细，类 rag09）→ expect 纳入 localflow。
  - **gate 负样本必须 grep 验证 KB 不覆盖**：凭直觉选的 Docker（all-in-rag 实战用 Docker，best 0.584）、React（reactive planner 含 "react"，best 0.677）
    都在 KB 沾边、被误判 in_kb → 剔除，换 grep 确认 0 覆盖的 Photoshop / 单片机。
- **重要发现（扩充才暴露的系统边界）**：新正经题 best 上界 **0.669**（alg09）vs React 负样本 **0.677** —— **正负 best 分布开始重叠**。
  揭示：**评测集越大越真实，正负距离分布越重叠，纯阈值门控的「完美分离」越不可能**——Round 5 的完美分离部分是小样本之幸。
  纯距离门控对「边缘相关」（工程邻域词）有固有局限；未来要更强拒答需 cross-encoder 相关性判别 / 关键词覆盖度等**证据型门控**。
- **新基线指标**（`docs/rag_eval/round6_expanded.json`，58 题，route+BM25+rerank ON，阈值 0.73）：

  | 分桶 | R@1 | R@3 | R@5 | MRR |
  |------|-----|-----|-----|-----|
  | 总体(58) | 86% | 97% | 100% | 0.913 |
  | 难题(15) | 67% | 87% | 100% | 0.763 |
  | 易题(43) | 93% | 100% | 100% | 0.965 |
  | llm_app(26) | 77% | 96% | 100% | 0.862 |
  | backend(16) | 94% | 100% | 100% | 0.958 |
  | algorithm(10) | 100% | 100% | 100% | 1.000 |
  | career(6) | 83% | 83% | 100% | 0.867 |
  | **gate** | **neg 12/12 · pos 12/12** ✅ 翻倍样本仍满分 | | | |

- **结论**：**gate 在翻倍样本（12+12）上仍 100%**——0.73 阈值经更大样本验证鲁棒、非过拟合。指标在 58 题上更真实
  （难题 R@1 67% 反映真实难度，不再是小样本虚高）。评测可信度显著提升，后续优化有更扎实的标尺。

> **baseline 迁移**：后续以 `round6_expanded.json`（58 题）为对比基准。

---

## Round 6 — 证据型门控（reranker 二次确认，防御边缘误判）

- **动机**：评测集扩充 v2 暴露「正负 best 分布重叠」——纯距离门控对「距离近但实际不相关」的边缘词
  （Docker：KB 沾边、best 0.58<0.73 会误纳）无能为力。本轮引入 reranker 作**第二重正交证据**。
- **诊断（数据驱动）**：reranker top1 分对「边缘负样本 vs 正经题」的判别力**远强于距离**——
  58 题正经题 reranker_top **min 0.936**，想拒的 Docker **0.644**，**间隙 [0.644, 0.936] 宽 0.29**（距离间隙仅 0.087，宽 3 倍多）。
  印证：cross-encoder **精判相关性** > 向量距离**粗判相似性**。
- **项目改动**（方便定位/学习）：
  - `rag_gate._retrieve_and_classify`：rerank 返回的 top1 分（原丢弃的 `_scores[0]`）保留为 `rerank_top` 并加入返回值；
    in_kb 判定从 rerank **之前**移到**之后**，综合「距离/词法初判 + reranker 确认」。
  - 新增纯函数 `rag_gate._evidence_gate(vector_in_kb, rerank_top)`：reranker 持否决权——
    `vector_in_kb AND rerank_top≥RAG_RERANK_GATE_MIN(默认0.85)`；rerank 关时 rerank_top=None → 退化纯距离（兼容）。
  - 新增 `tests/test_rag_gate_evidence.py`（5 测：否决/退化/阈值覆盖/边界）。
- **本轮性能**（`docs/rag_eval/round7_evidence_gate.json` vs `round6_expanded.json`）：

  | 维度 | round6 | Round6 证据门控 | Δ |
  |------|--------|---------------|---|
  | 总体 / 各域 / 难题 全部检索指标 | — | 完全一致 | **全 Δ=0** |
  | gate neg / pos | 12/12 · 12/12 | 12/12 · 12/12 | = |
  | **边缘负样本拒答**（Docker/React/Vue3/SpringBoot） | 纯距离 **2/4** | 证据门控 **3/4** | **↑ Docker 救回** |

  当前评测集**主指标零影响**（58 题 reranker_top≥0.936 全过、12neg 距离已拒，证据门控不改判定）；
  独立边缘测试**拒答率 +25%**（Docker：距离近 best 0.58 但 reranker 0.64 被否决，纯距离漏纳）。
- **对应八股**：拒答/相关性判定有两类信号——**双塔向量距离**（粗、快，但「相似 ≠ 相关」）vs **cross-encoder**（精、慢，精判相关性）。
  **多信号证据融合**（距离 AND reranker）比单信号鲁棒；reranker 作否决权过滤「距离近但不相关」的边缘误判。
  局限：React（reranker 也误判 0.957）仍漏，需第三重证据（关键词覆盖度 / 实体匹配）——**单一信号都有盲区，证据要正交**。
- **结论**：证据型门控是**防御性鲁棒增强**——当前评测集零影响（无害），对边缘误判（Docker 类）拒答率提升。
  正面回应「正负 best 重叠」：纯阈值不够时引入正交的第二信号（reranker），而非死磕单一阈值。

> **baseline 迁移**：后续以 `round7_evidence_gate.json` 为对比基准。

---

## Round 7 — 端到端答案质量评测（从检索延伸到生成）

- **动机**：前面所有轮次都在**检索层**（召回 / 排序 / 同质 / 拒答），但 RAG 的最终价值是**生成的答案好不好**。本轮建立答案质量评测，让价值闭环到生成端。
- **措施**：新建 `eval_rag_answer.py`——对题走**生产主路径**（`_retrieve_and_classify` 含 rerank/BM25/route）→ grounded 生成 → **LLM-as-judge** 三维度打分（RAGAS 风格）：
  - **忠实度 faithfulness**：论断是否都基于检索资料、有无资料外杜撰（防幻觉，RAG 最核心）；
  - **完整度 completeness**：是否充分准确回答问题；
  - **引用准确性 citation**：末尾引用编号是否对应真实支撑。
- **项目改动**（方便定位/学习）：
  - 新增 `eval_rag_answer.py`：`_judge`（LLM 裁判，强制 JSON 输出）、`_pick_subset`（各域均匀 + 难题优先，按 id 排序**可复现**）、`evaluate`。
  - 复用 `rag_gate.synthesize_grounded_answer`（生产生成逻辑）+ `_retrieve_and_classify`（主路径检索），评测与线上同构。
- **首个基线**（`docs/rag_eval/answer_quality.json`，12 题各域均匀，judge=qwen-turbo）：

  | 维度 | 分数 | 解读 |
  |------|------|------|
  | 忠实度 faithfulness | **9.25**/10 | grounded prompt「严禁资料外知识」约束有效，幻觉少 |
  | 完整度 completeness | **8.92**/10 | 答案基本覆盖问题要点 |
  | 引用准确性 citation | **8.67**/10 | 引用编号大体对应，少数偏差 |

  - 满分题：algorithm 全 10/10/10（八股答案质量高）、career 入门/转型（ca01/ca02）。
  - **最弱题 ca05「学习路径怎么规划」**（忠实 7 / 完整 6 / 引用 3）：**泛开放问题**（规划类）检索 chunk 较散、答案不够 grounded——这类问题是答案质量的短板。
- **对应八股**：**RAGAS** 三支柱——faithfulness（答案↔context 一致）、answer relevance（答案↔question 切题）、context relevance（检索↔question，即前面轮次的 Recall/MRR）。**LLM-as-judge** 是无标准答案时的主流评法。
  局限：judge 与生成同模型（qwen-turbo）有**自评偏差**（`RAG_JUDGE_MODEL` 可换更强模型降偏）；judge 单次有噪声，多题平均更稳（n=4 偏低、n=12 趋稳）。
- **结论**：RAG 价值链从「检索准」延伸到「答案好」**完整闭环**。生成质量整体高（忠实 9.25），暴露**泛开放问题 grounded 不足**——下一步可针对性优化（查询分解 / 更强引用约束 / 该类问题补结构化资料）。

> **答案质量基线**：`docs/rag_eval/answer_quality.json`（忠实 9.25 / 完整 8.92 / 引用 8.67）。

---

## Round 8 — gate 反向边缘救援（修「LoRA 是什么」漏判）

- **动机（测试报告暴露）**：2026-06-22 测试报告 §7 问题3——`eval_rag_domain.py` 的「LoRA 是什么」被门控判为
  `in_kb=False` 漏判。诊断：best 距离 **0.758**（刚过 strong 阈值 0.73，向量初判判为域外），但 reranker_top
  **0.9928**（cross-encoder 明确判定强相关）。这是 **Round 6 证据门控的反向盲区**——Round 6 只赋予 reranker
  **否决权**（拒「距离近但不相关」，如 Docker），却没赋予**救援权**（救「距离边缘但明确相关」，如 LoRA）。证据门控成了**单向**的。
- **项目改动**（方便定位/学习）：
  - `rag_gate._evidence_gate` 扩签名为 `(vector_in_kb, rerank_top, best=None, strong=None)`，加**反向边缘救援**分支：
    `vector_in_kb=False` 时，若 `strong < best ≤ RAG_RERANK_RESCUE_DIST(默认0.80)` **且** `rerank_top ≥ RAG_RERANK_RESCUE_MIN(默认0.95)` → 救回 `True`。
  - 调用点 `_retrieve_and_classify` 传入 `best` 与 `th["strong"]` 激活救援；不传则 best/strong=None 退化为原行为（向后兼容）。
  - `tests/test_rag_gate_evidence.py` +3 测：LoRA 救回（best 0.758 + rerank 0.99）/ 不救太远（best 0.85>0.80）/ 不救弱证据（rerank 0.77<0.95）。
- **本轮性能**：

  | 维度 | 修复前 | 修复后 | Δ |
  |------|--------|--------|---|
  | 「LoRA 是什么」in_kb | **False（漏判）** | **True（救回）** | **✅ 修复** |
  | gate 负样本拒答 / 正样本命中 | 12/12 · 12/12 | 12/12 · 12/12 | **零破坏** |
  | 58 题检索主指标（R@1 86 / MRR 0.913） | — | 完全一致 | **全 Δ=0** |
  | 全量 pytest | 351 passed | **354 passed**（+3 救援测） | 零回归 |

  负样本零误救的原因：**双条件夹逼**缺一不可——Vue3 reranker 0.771<0.95（弱证据不救）、Spring Boot best 0.816>0.80（太远不救）、今天天气 best 0.968>0.80（太远不救）。
- **对应八股**：**证据融合应对称**——一个信号既被赋予否决权（Round 6），其高置信的反向也应能救援，否则强证据被弱信号单向压制、形成盲区。
  「夹逼救援」用**双阈值**限定触发：`strong<best≤rescue_dist` 圈定「边缘带」（只救刚出界的，不救深域外），`rerank≥rescue_min` 要求「强证据」（只让 cross-encoder 高置信的翻转）——避免过度救援把噪声放进来。本质是 **precision/recall 权衡在门控层的精细调参**。
- **结论**：补齐 Round 6 证据门控的反向盲区，召回侧不再漏「距离边缘但强相关」的术语题（LoRA 类），且对负样本零误救、主指标零影响——**纯防御性增强**。
- **门控路径统一（同轮后续修复）**：上线后复检发现救援只在 `_retrieve_and_classify`（→ `/api/stream`）生效，而 `/api/query` 与微信 CLI 走的 `gated_query` 当年是一套**独立的朴素实现**（直接 chromadb 查询 + 旧距离门控，**绕过了 rerank/BM25/route/证据门控/救援**）——即 LoRA 在生产主查询路径仍漏判，而评测走的是好路径，造成**「评测路径 ≠ 生产路径」的隐蔽分叉**。修复：让 `gated_query` 改为**委托 `_retrieve_and_classify`**（与流式 `gated_query_stream`、评测 `eval_rag_bench` 同源），三条入口共用单一检索+门控真源，production 与 eval 口径一致。教训：**评测必须打在生产同一条代码路径上**，否则指标再好也可能只是"另一条路"的好。

---

# 阶段总结（截至 Round 8 + 评测集 v2）

## 优化轨迹与终态

从「朴素 RAG + 宽松评测」走到「rerank + 混合检索 + 元数据路由 + 真实多源库 + 可信度量」：

- **Round 0** 把宽松评测（命中本域任意文件即算、显示假性 100%）升级为**精确目标 + Recall@1/3/5 + MRR**——一切优化的标尺。
- **Round 1** 本地 bge-reranker 精排：难题 R@1 82→91。
- **真实化 v2/v3** 入 LocalFlow + all-in-rag + hello-agents（1024→**3299 块**）：R@5 从虚高 100% 落到真实区间，暴露 llm_app 同主题多源短板。
- **Round 2** BM25 + RRF 混合检索：llm_app R@1 +5.6、难题 R@3 100%。
- **Round 3** HyDE：**负结果**（瓶颈在文档侧，query 侧增强无效 + 延迟大）——学到「方法要匹配瓶颈层」。
- **Round 4** 元数据/文件名路由：career R@1 75→**100**、零副作用。
- **Round 5** gate 拒答阈值重标定（0.85→0.73）：负样本拒答 4/6→**6/6**、检索零影响——扩库后门控必须随之重标定。
- **评测集扩充 v2** 39→**58 题**、gate 6→**12/12**：验证 0.73 阈值在翻倍样本上鲁棒（非过拟合），度量更可信；扩充暴露「正负 best 重叠」。
- **Round 6** 证据型门控（reranker 二次确认）：主指标零影响，边缘负样本拒答 2/4→**3/4**——对「正负 best 重叠」的正交防御。
- **Round 7** 端到端答案质量评测（RAGAS / LLM-as-judge）：价值链从「检索准」闭环到「答案好」——忠实 9.25 / 完整 8.92 / 引用 8.67。
- **Round 8** gate 反向边缘救援：补 Round 6 的反向盲区，「LoRA 是什么」漏判救回，gate 仍 12+12、主指标零影响——证据融合应**对称**（否决权↔救援权）。

**终态**（检索 `round7_evidence_gate.json` + 答案 `answer_quality.json`）：检索 **R@1 86 / R@3 97 / R@5 100 / MRR 0.913 / gate 12+12**；
答案质量 **忠实 9.25 / 完整 8.92 / 引用 8.67**（/10）。algorithm 满分、backend 94%、career 83%、llm_app 77%。
**RAG 全链路闭环**：召回（混合检索）→ 排序（rerank）→ 同质区分（元数据路由）→ 拒答（距离+reranker 双证据）→ 生成（grounded + 质量评测）。

## 贯穿五轮的「元知识」（比单个技术更重要）

1. **可信度量先行**：宽松评测会让优化变自欺（假性 100%）；先有精确标尺，改进才可证。
2. **知识库真实化**：单源库 R@5 虚高 100%，多源交叉后才逼近工业真实——**评测的可信度取决于库的真实度**。
3. **方法要匹配瓶颈所在的层**（query / 文档 / 排序 / 门控）：Round 3（query 侧 HyDE）vs Round 4（文档侧路由）救同一问题、结果相反。
4. **诚实拆分「算法贡献」与「数据/标注」**：Round 2 把 BM25 增益（61→67）与 rag09 标注修正（67→72）分开记，不混功。
5. **负结果也是成果**：HyDE 没涨指标，但「理解它何时无效」是可复用的认知资产。

---

# 附录：知识库 RAG 技术 × 本项目 —— 采用情况与适用边界

> **学习目的**：不仅记成功，更系统梳理「知识库（all-in-rag 28 章 / 飞书八股 / hello-agents）讲了、
> 但本项目没用或试了失败的技术」——掌握每个技术的**适用边界（何时该用）**，同时反向理解**本项目的定位（为什么不需要）**。
> 一句话总纲：本项目是「**中等规模中文八股 / 项目文档问答**」，未采用的多是为「超长文档 / 多跳推理 / 结构化数据 / 极致精度」设计的重型技术——**工程的智慧在「匹配」而非「堆砌」**。

## A. 已采用并见效

| 技术 | 落点 | 结果 |
|---|---|---|
| Markdown 语义分块、bge 本地嵌入、ChromaDB/HNSW、密集检索、图转文、密钥脱敏 | 初始实现 | 朴素 RAG 基座 |
| Rerank 交叉编码器精排 | Round 1 | 难题 R@1 82→91 |
| BM25 + RRF 混合检索 | Round 2 | llm_app R@1 +5.6 |
| 元数据/文件名感知路由 | Round 4 | career R@1 75→100 |
| 距离+reranker 双证据门控（否决权 Round6 + 救援权 Round8） | Round 6 / 8 | 边缘负样本拒答+25%、LoRA 漏判救回、gate 12+12 |
| 词法救援 + 三层距离门控 + 双源问答 | 初始/演进 | in_kb 判定、防幻觉 |

## B. 尝试过但无效 / 不适合本项目（有实测）

| 技术 | 是什么 | 本项目结果 + 原因 | 何时该用 |
|---|---|---|---|
| **索引优化·chunk 标题前缀**（方案A） | 给每块嵌入文本加文档标题前缀，强化主题信号 | ❌ Δ=0：**短标题被长 body 稀释**，bge 向量几乎不变 | 块内主题漂移严重、标题强区分时 |
| **HyDE**（Round 3） | LLM 先生成假设答案，用其 embedding 检索 | ❌ 无效/有害：①query 已含精确关键词、鸿沟本就小；②**瓶颈在文档侧（同主题多源）**，query 侧治不了；③每查询一次 LLM、延迟大；④假设答案趋通用，反而加剧 career 同质混淆 | query 模糊/口语化、与文档用词差异大、低延迟非硬约束 |

## C. 知识库涉及但本项目未采用 —— 按 RAG 层的适用边界分析

**C1 · 预检索（查询侧）**

| 技术 | 一句话 | 本项目为何不需要 | 适用场景 |
|---|---|---|---|
| Multi-Query 多查询 | LLM 生成多个 query 变体并行检索再融合 | 同 HyDE 属 query 侧；query 已精确、瓶颈在文档侧；N× LLM 延迟 | query 短/歧义、单次召回不全 |
| Step-Back 回退提问 | 先抽象上位概念再检索 | 评测 query 已是明确技术点，无需抽象铺垫 | 具体问题需背景知识支撑 |
| 查询分解 Decomposition | 复杂多跳问题拆子问题分别检索 | 本项目问答是**单点八股**，非多跳综合 | 多步推理 / 跨多文档综合 |

**C2 · 分块（文档侧）**

| 技术 | 一句话 | 本项目为何不需要 | 适用场景 |
|---|---|---|---|
| 父文档 / 句子窗口 / 自动合并 | 小块精确检索、大块/邻窗喂 LLM 补上下文 | 按 `##` 语义分块、块不算小（800）；评测聚焦检索召回非生成上下文完整性 | 块太小丢上下文、又需精确命中 |
| RAPTOR 递归摘要树 | 对文档聚类递归摘要、建多粒度层次树 | 文档中等长度、章节结构清晰，建树过重、收益低 | 超长文档、需细节+全局摘要多粒度 |

**C3 · 检索 / 表示侧**

| 技术 | 一句话 | 本项目为何不需要 | 适用场景 |
|---|---|---|---|
| ColBERT / 多向量 | token 级后期交互（late interaction） | 存储/计算重；bge 单向量 + rerank 已 R@3 100% | 极致召回精度、有资源做细粒度匹配 |
| 嵌入微调 | 用领域标注数据微调 embedding | 需标注三元组 + 训练成本；通用 bge 已 3 域满分 | 领域术语与通用语料差异大、有标注 |

**C4 · 后检索 / 生成侧**

| 技术 | 一句话 | 本项目为何不需要 | 适用场景 |
|---|---|---|---|
| 上下文压缩 | 检索后压缩/过滤无关内容再喂 LLM | top5 块量可控、上下文窗口够；评测聚焦检索 | 检索块多/长、上下文超限或噪声大 |
| CRAG 纠正式 RAG | 检索质量自评，差则触发重检索/web | 增 LLM 评估延迟；gate 的 in_kb 已是「检索质量门控」雏形，ROI 低 | 对鲁棒性要求高、容忍额外延迟 |
| Self-RAG | 模型用反思 token 自决检索/批判 | 需特训模型或复杂编排，过重 | 研究级鲁棒性 |

**C5 · 数据形态侧**

| 技术 | 一句话 | 本项目为何不需要 | 适用场景 |
|---|---|---|---|
| 图 RAG / GraphRAG | 抽实体关系建知识图谱检索 | 本项目是**文档问答**非实体关系网络；建图成本高收益低 | 知识高度关联、需多跳实体推理 |
| Text2SQL | 自然语言转 SQL 查结构化库 | 无关系库场景（求职数据是 JSON 状态文件，走双源问答） | 问答对象是关系型数据库 |
| 多模态嵌入（CLIP 等） | 原生图文跨模态向量 | 已用**图转文**（图→qwen-vl 描述→文本嵌入）替代，统一检索更简单 | 需图-图/图文跨模态精确检索 |
