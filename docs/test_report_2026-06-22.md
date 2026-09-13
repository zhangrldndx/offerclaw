# OfferClaw 项目测试报告（2026-06-22 快照）

> 🗄️ **2026-06-22 测试快照 · 历史留存**。文中 pytest 351/354 等为当日值；此后已增至 **397 passed / 3 skipped**。当前真值见 [`verification_report.md`](verification_report.md)。

测试时间：2026-06-22 18:35 CST  
测试环境：macOS 本机，Python 3.12.5，项目虚拟环境 `.venv`  
重点范围：RAG 检索/门控/问答、ReAct Agent、CareerFlow/tool ecology、FastAPI API 层

## 1. 总体结论

当前项目整体可运行，主链路、全量单测、API E2E、RAG 新版 benchmark、Agent 工具链均通过。最重要的健康指标如下：

| 项目 | 结果 |
|---|---:|
| `doctor.py` | 12 OK · 0 WARN · 0 ERR |
| `verify_pipeline.py` | 6/6 通过 |
| 全量 pytest | 351 passed · 3 skipped · 2 warnings |
| RAG 专项单测 | 51 passed · 2 warnings |
| Agent/CareerFlow 专项 | 42 passed · 1 skipped |
| API E2E (`OFFERCLAW_E2E=1`) | 15 passed |
| 新版 RAG benchmark | R@1 86% · R@3 97% · R@5 100% · MRR 0.913 |
| 新版 RAG 门控 | 负样本 12/12 拒答 · 正样本 12/12 命中 |

主要风险不是“功能不可用”，而是评估口径和文档指标已经出现漂移：README 顶部仍写 `130 passed` / `160 chunks` / `RAG Recall@5 0.96`，但当前实际是 `351 passed`、collection `3299 chunks`；旧版 `eval_rag.py` 在当前扩展知识库上 Recall@5 为 `0.88`。

## 2. 环境与基础健康

执行命令：

```bash
.venv/bin/python --version
.venv/bin/python doctor.py
.venv/bin/python verify_pipeline.py
```

结果：

- Python：3.12.5。
- 当前 RAG collection：`offerclaw_local_bge_base_zh_768`。
- collection 记录数：3299 chunks。
- `doctor.py`：12 OK，0 WARN，0 ERR。
- `verify_pipeline.py`：6/6 通过。

注意：`verify_pipeline.py` 的 `rag_query` 步骤显示“模块导入成功（未找到公开查询函数，跳过执行）”，所以它不能单独证明 RAG 查询链路有效；本报告用后续 benchmark 和 `/api/query` smoke 补齐验证。

## 3. 全量测试

执行命令：

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

结果：

```text
351 passed, 3 skipped, 2 warnings in 8.27s
```

警告：

- `StarletteDeprecationWarning`：`fastapi.testclient` 当前依赖的 `httpx` 用法即将弃用，提示安装/迁移 `httpx2`。
- `jieba` 通过 `pkg_resources` 触发 deprecation warning，来源于依赖内部。

## 4. RAG 测试

### 4.1 RAG 专项单测

执行命令：

```bash
.venv/bin/python -m pytest \
  tests/test_rag_bm25.py \
  tests/test_rag_route.py \
  tests/test_rag_rerank.py \
  tests/test_rag_gate_evidence.py \
  tests/test_plan_gen_rag.py \
  tests/test_kb_and_plan_api.py \
  -q -rs --tb=short
```

结果：

```text
51 passed, 2 warnings in 0.97s
```

覆盖面包括 BM25 分词与 RRF 融合、audience route、rerank 开关/降级、证据门控、RAG 计划生成和知识库/API 相关逻辑。

### 4.2 新版生产链路 benchmark

执行命令：

```bash
RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 \
  .venv/bin/python eval_rag_bench.py \
  --baseline docs/rag_eval/round7_evidence_gate.json
```

结果：

| 指标 | 当前值 | 相对 baseline |
|---|---:|---:|
| Overall R@1 | 86% | +0.000 |
| Overall R@3 | 97% | +0.000 |
| Overall R@5 | 100% | +0.000 |
| MRR | 0.913 | +0.000 |
| Hard R@5 | 100% | +0.000 |
| Gate negatives | 12/12 | 通过 |
| Gate positives | 12/12 | 通过 |

分域结果：

| 域 | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| llm_app | 77% | 96% | 100% | 0.862 |
| backend | 94% | 100% | 100% | 0.958 |
| algorithm | 100% | 100% | 100% | 1.000 |
| career | 83% | 83% | 100% | 0.867 |

结论：新版 benchmark 与 `round7_evidence_gate` 完全一致，生产检索链路未出现回归。

### 4.3 rerank 关闭对照

执行命令：

```bash
RAG_RERANK=0 RAG_BM25=1 RAG_ROUTE=1 \
  .venv/bin/python eval_rag_bench.py \
  --baseline docs/rag_eval/round7_evidence_gate.json
```

结果：

| 指标 | rerank on | rerank off | 变化 |
|---|---:|---:|---:|
| Overall R@1 | 86% | 81% | -5.2pt |
| Overall R@3 | 97% | 97% | 0 |
| Overall R@5 | 100% | 98% | -1.7pt |
| MRR | 0.913 | 0.886 | -0.026 |
| Gate negatives | 12/12 | 12/12 | 0 |
| Gate positives | 12/12 | 12/12 | 0 |

未命中样本：`rag12`，问题为“上下文压缩 context compression 在 RAG 里的...”。

结论：rerank 对总体排序质量仍有明确正贡献，尤其是 `llm_app` 域；但关闭 rerank 后 hard 分桶的 R@1/MRR 局部更高，说明 rerank 的排序收益不是对每个子集单调提升。

### 4.4 旧版 `eval_rag.py`

执行命令：

```bash
.venv/bin/python eval_rag.py --k 5
```

结果：

| bucket | N | Recall@5 | MRR |
|---|---:|---:|---:|
| overall | 50 | 0.880 | 0.660 |
| cross_doc | 15 | 0.933 | 0.817 |
| explain | 18 | 0.889 | 0.631 |
| fact | 17 | 0.824 | 0.551 |

主要 miss：`f02`, `f07`, `f12`, `e11`, `e13`, `c02`。

结论：旧版评估口径在当前 3299 chunks 知识库上低于 README 中的 `Recall@5=0.96`。需要判断是旧评估集 expect source 已过时，还是扩库后引入了相似文档干扰。

### 4.5 分域诊断

执行命令：

```bash
.venv/bin/python eval_rag_domain.py --k 5
```

结果：

| 域 | N | Recall@5 | 平均本域纯度 |
|---|---:|---:|---:|
| llm_app | 6 | 100% | 73% |
| backend | 6 | 100% | 97% |
| algorithm | 5 | 100% | 92% |
| career | 4 | 100% | 95% |
| overall | 21 | 100% | 89% |

门控结果：

- 负样本拒答：5/5。
- 正样本命中：4/5。
- 漏判：`LoRA 是什么` 被判为 `in_kb=False`。

结论：分域召回正常，但门控在该小评估集上存在一个正样本漏判；建议把 `LoRA 是什么` 加入新版 gate regression，或重新校准阈值/同义词召回。

## 5. Agent 与 Tool 测试

### 5.1 Agent/CareerFlow 专项

执行命令：

```bash
.venv/bin/python -m pytest \
  tests/test_tool_ecology.py \
  tests/test_agent_tool_robustness.py \
  tests/test_tool_schema_consistency.py \
  tests/test_career_flow.py \
  tests/test_career_flow_routing.py \
  tests/test_career_flow_guard.py \
  -q -rs --tb=short
```

结果：

```text
42 passed, 1 skipped, 1 warning in 0.31s
```

跳过项：

- `tests/test_tool_ecology.py::test_react_llm_mode_real_smoke`：pytest 进程没有预置 `OPENAI_API_KEY`，因此跳过真实 LLM mode smoke。

覆盖面包括：

- 6 个 OpenAI-compatible tools 注册与 schema 一致性。
- tool 参数错误包装。
- tool 超时与异常不炸主循环。
- deterministic ReAct 路由。
- 无 key 时 LLM mode fallback 到 deterministic。
- malformed tool_call 与 max_steps 防御。
- CareerFlow 端到端、条件路由和 fail-soft guard。

### 5.2 真实 LLM mode 直接 smoke

执行命令：直接调用 `react_agent.run("今天我应该做什么？", mode="llm", max_steps=2)`。

结果：

```json
{
  "mode": "llm",
  "steps": 1,
  "errors": [],
  "tool_names": ["today_advice"],
  "answer_len": 226
}
```

结论：真实 LLM smoke 属于本地可选验证；公开报告不记录所用端点、凭据元数据或本机连通状态。

## 6. API 层测试

### 6.1 TestClient smoke

直接检查：

- `GET /health`：200，`healthy`，collection records 为 3299。
- `POST /api/agent` deterministic：200，调用 `today_advice`，无错误。
- `POST /api/agent` llm：200，调用 `today_advice`，无错误。
- `POST /api/query`：200，`in_kb=true`，返回有效答案。

### 6.2 显式 E2E

执行命令：

```bash
OFFERCLAW_E2E=1 .venv/bin/python -m pytest tests/test_api.py -q -rs --tb=short
```

结果：

```text
15 passed, 1 warning in 8.07s
```

结论：默认跳过的 `/api/query`、`/api/search`、`/api/stream` E2E 路径本次均跑通。

## 7. 主要发现与建议

1. README/项目状态指标需要更新  
   当前 README 顶部仍写 `130 passed`、`160 chunks`、`RAG Recall@5 0.96`，与本次实测 `351 passed`、`3299 chunks`、新版 benchmark `R@5 100%`、旧版 eval `R@5 0.88` 不一致。

2. 旧版 RAG 评估集存在口径漂移  
   `eval_rag.py` 的 50 题在扩库后 Recall@5 为 0.88，miss 集中在 profile/status/data contract/application 相关问题。建议逐条复核 expect source 是否仍合理，并把新版 `eval_rag_bench.py` 作为主口径。

3. 门控有一个分域正样本漏判  
   `eval_rag_domain.py` 中 `LoRA 是什么` 被漏判为 out-of-KB。建议加入 gate regression，并检查 query expansion、metadata route 或门控阈值是否对算法术语偏严。

4. RAG benchmark 缺少进度输出  
   开启 rerank 的 `eval_rag_bench.py` 运行耗时较长，期间没有逐题进度，容易被误判为卡死。建议增加 `--verbose` 或简单进度计数。

5. 依赖警告需要排期处理  
   `fastapi.testclient` 的 Starlette/httpx deprecation warning 未来可能影响测试稳定性；`jieba/pkg_resources` 警告来自依赖内部，可先记录，后续通过依赖版本约束处理。

## 8. 复测建议

- 短周期回归：`doctor.py`、全量 pytest、`eval_rag_bench.py --baseline docs/rag_eval/round7_evidence_gate.json`。
- RAG 改动后必跑：RAG 专项单测、新版 benchmark、`eval_rag_domain.py`。
- Agent 改动后必跑：tool ecology、schema consistency、agent robustness、`OFFERCLAW_E2E=1 tests/test_api.py`。
- 文档发布前：同步 README/PROJECT_STATUS 的测试数量、chunks 数、RAG 指标和主评估口径。

## 9. 修复闭环（2026-06-22 当日处理）

> 复测更正：本节记录的是提交 `3578b06` 的修复说明。19:16 之后复测发现，其中 README 指标同步、pytest 数量、benchmark 进度输出、底层 `_retrieve_and_classify` 的 LoRA 救援均已生效；但 `verify_docs.py` 的期望口径没有同步，且非流式 `gated_query` / `/api/query` 仍未接入 Round8 救援。详见 §10。

§7 五项发现已全部落地修复并验证，全量测试从 351 → **354 passed · 3 skipped**（+3 救援测）、零回归。

| # | §7 发现 | 修复 | 改动定位 | 验证 |
|---|---|---|---|---|
| 1 | README 指标漂移（130 passed / 160 chunks / Recall@5 0.96） | 批量同步为当前实测值：pytest **354 passed**、**3299 chunks**、检索 **R@1 86% / MRR 0.913**，并新增「⚙️ 工程亮点」段 | `README.md`（badge / 数据表 / 架构图 / quick start / Evaluation 表 / tech stack） | 人工核对，残留仅 V4 历史里程碑标题（无害） |
| 2 | 旧版 `eval_rag.py` 口径漂移（Recall@5 0.88） | 旧评测降为**辅助口径**、加 deprecation 注释指向主口径；README 已明确 `eval_rag_bench.py` 为主口径 | `eval_rag.py` / `eval_rag_domain.py` docstring 顶部 | 注释内联，指向 58 题精确评测 |
| 3 | 门控漏判「LoRA 是什么」（in_kb=False） | **gate 反向边缘救援**（RAG Round 8）：reranker 除否决权外加救援权，best∈(strong,0.80] 且 rerank≥0.95 时救回 | `rag_gate._evidence_gate`（+best/strong 参数）+ 调用点；`tests/test_rag_gate_evidence.py` +3 测 | LoRA in_kb **False→True**；gate **12/12·12/12** 零破坏；负样本零误救 |
| 4 | benchmark 无进度、易误判卡死 | `evaluate(verbose=)` + `--verbose`；默认覆盖式 `评测中 i/n`、verbose 逐题到 stderr（不污染 stdout 结果） | `eval_rag_bench.py`（evaluate 循环 / argparse / 调用） | 冒烟：`[1/58] rag01 …` 正常输出 |
| 5 | 依赖 deprecation 警告（starlette/httpx、jieba/pkg_resources） | 建 `pytest.ini` filterwarnings 记录在案、不污染输出；附注「待依赖版本约束彻底处理」 | 新增 `pytest.ini` | 全量 warning 2→1 |

修复对应的知识沉淀：问题3 已作为 **Round 8** 写入 `docs/RAG_OPTIMIZATION_LOG.md`（含动机 / 改动 / 双条件夹逼救援 / 对应八股「证据融合应对称」）。

## 10. 二次复检发现的两个未闭环问题（2026-06-22 当日二修）

首次修复推送后复检，发现 §9 的修复有两处未真正闭环，已补修并验证：

| # | 问题 | 根因 | 修复 | 验证 |
|---|---|---|---|---|
| A | `doctor.py` 失败（`verify_docs` 1 ERR） | `verify_docs.py` 的 `EXPECTED` 仍是旧 V4 口径（Recall@5 0.96 / MRR 0.67 / 160 chunks / 28 路由 / doctor 10 / cross_doc 桶）；更深层是 README 改了、另 3 份文档（verification_report / project_one_pager / PROJECT_STATUS）仍全停旧口径 | `EXPECTED` 更新到当前真值（Recall@5 1.00 / MRR 0.913 / 3299 chunks / 45 路由 / doctor 12，去掉过时 cross_doc）；**4 份文档全量重刷到当前口径**（verification_report 重跑命令粘新鲜输出；PROJECT_STATUS 顶部加当前口径横幅、dated 历史批次保留为可追溯记录） | `verify_docs` **0 不一致**；`doctor` **12 OK · 0 WARN · 0 ERR** |
| B | LoRA 救援只在 `_retrieve_and_classify`/`/api/stream` 生效，`/api/query`、微信 CLI 仍漏判 | `gated_query()` 当年是**独立的朴素实现**（直接 chromadb 查询 + 旧距离门控，绕过 rerank/BM25/route/证据门控/救援）；评测走好路径、生产走旧路径——「评测路径 ≠ 生产路径」的隐蔽分叉 | 让 `gated_query` **委托 `_retrieve_and_classify`**，三条入口（同步/流式/评测）共用单一检索+门控真源 | `gated_query("LoRA 是什么")` → `in_kb=True · kb_grounded`；354 passed 零回归 |

教训：① 文档口径要**单一事实来源 + 自动巡检**，改一处必须同步全部活文档；② **评测必须打在生产同一条代码路径上**，否则指标再好也可能只是"另一条路"的好。B 项已补记入 `RAG_OPTIMIZATION_LOG.md` 的 Round 8「门控路径统一」。

## 10. 复测补充（2026-06-22 19:16 后）

复测对象：`3578b06 fix: 修复测试报告 §7 五项发现（README 指标 + gate 反向救援 Round8 + 评测口径/进度/警告）`。  
复测结论：核心 pytest、agent、API E2E、新版 RAG benchmark 均通过；但本轮修复仍有两个未闭环问题。

### 10.1 通过项

| 项目 | 结果 |
|---|---:|
| `verify_pipeline.py` | 6/6 通过 |
| 全量 pytest | 354 passed · 3 skipped · 1 warning |
| RAG gate/rerank/route/BM25 专项 | 26 passed |
| Agent/CareerFlow/tool 专项 | 42 passed · 1 skipped |
| API E2E (`OFFERCLAW_E2E=1`) | 15 passed |
| Agent 真实 LLM mode smoke | mode=llm · steps=1 · errors=[] · tool=`today_advice` |
| 新版 RAG benchmark | R@1 86% · R@3 97% · R@5 100% · MRR 0.913 |
| 新版 benchmark gate | 负样本 12/12 拒答 · 正样本 12/12 命中 |

`eval_rag_bench.py --verbose` 已能输出 `[1/58] ...` 逐题进度，解决了之前“长时间无输出像卡死”的体验问题。指标与 `round7_evidence_gate` baseline 完全一致。

### 10.2 未闭环问题 A：doctor / verify_docs 仍失败

执行：

```bash
.venv/bin/python doctor.py
.venv/bin/python verify_docs.py
```

结果：

```text
doctor.py: 11 OK · 0 WARN · 1 ERR
verify_docs.py: 发现 3 处不一致
```

具体差异：

- `Recall@5`：`verify_docs.py` 仍期望 `0.96`，但 README 现在采集到 `1.00`。
- `MRR`：仍期望 `0.67`，但 README 现在采集到 `0.913`。
- `chunks`：仍期望 `160`，但 README 现在采集到 `3299`。

判断：README 已更新到当前口径，但 `verify_docs.py` 的 `EXPECTED` 常量仍是旧 V4 口径，导致 doctor 红灯。需要同步 `verify_docs.py` 的期望值，或让它区分“当前主口径”和历史文档中的旧指标。

### 10.3 未闭环问题 B：LoRA 救援只进入底层分类，未进入非流式问答入口

底层分类已生效：

```text
_retrieve_and_classify("LoRA 是什么")
=> in_kb=True, matched_by=lexical_rescue, best=0.7581, rerank_top=0.9928
```

但顶层非流式入口仍失败：

```text
gated_query("LoRA 是什么")
=> in_kb=False, mode=general_fallback, best_distance=0.7581
```

API 层也复现该分叉：

| 入口 | LoRA 结果 |
|---|---|
| `/api/query` | `in_kb=false`, `source_count=0` |
| `/api/stream` | `in_kb=true`, `mode=kb_grounded`, `matched_by=lexical_rescue`, `best_distance=0.7581` |

原因定位：`gated_query_stream()` 已调用 `_retrieve_and_classify()`，所以吃到了 Round8 反向救援；但 `gated_query()` 内部仍保留一套旧的向量检索 + 词法门控实现，没有调用 `_evidence_gate()`，也没有 rerank 救援权。因此 `eval_rag_domain.py` 仍显示：

```text
门槛：负样本拒答 5/5 · 正样本命中 4/5
[正] ✗ 漏判 LoRA 是什么
```

建议：把 `gated_query()` 改成复用 `_retrieve_and_classify()`，与 `gated_query_stream()` 共用同一条生产检索/门控链路；同时增加一个回归测试覆盖 `gated_query("LoRA 是什么")` 和 `/api/query`。

### 10.4 辅助口径状态

`eval_rag.py --k 5` 仍为辅助旧口径，结果未变化：

| bucket | N | Recall@5 | MRR |
|---|---:|---:|---:|
| overall | 50 | 0.880 | 0.660 |
| cross_doc | 15 | 0.933 | 0.817 |
| explain | 18 | 0.889 | 0.631 |
| fact | 17 | 0.824 | 0.551 |

这与 §9 的“旧评测降为辅助口径”一致，但仍建议后续清理 README / verify_docs / verification_report 的指标来源，避免主辅口径混在一起。

### 10.5 额外观察

- `pytest.ini` 后全量 pytest 警告从 2 个降到 1 个；剩余为 `fastapi.testclient` / Starlette `httpx` deprecation warning。
- 非 pytest 命令仍会显示 `jieba/pkg_resources` warning，`pytest.ini` 不会影响普通脚本输出。
- `eval_rag_bench.py` 在打印完整总表后，进程短时间内未自然退出，本次手动中断会话后退出码为 0；建议后续复查是否有模型/线程资源释放滞后。
