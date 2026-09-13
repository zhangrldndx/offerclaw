# RAG 测试与验证手册

> ⚠️ **历史手册（时点口径，非当前发布真源）**：本文写于早期轮次，仍保留旧
> pool、旧 `eval_rag_answer.py` 与早期 HyDE/门控描述，适合追溯"每个措施当时
> 怎么验证"。**当前发布口径以 `metrics.json`（单一事实源）与
> `docs/COMPLETE_TEST_PLAN_20260831.md` 为准**；答案质量评测已由
> `eval_answer_quality_v2.py` 接替，产品验收由
> `scripts/run_product_acceptance.py` 接替。本文数字不再随版本更新。

> 目的：**一处一命令**，让你快速定位每个 RAG 措施的代码、亲手跑出效果、对照面试讲解。
> 配合 `RAG_OPTIMIZATION_LOG.md`（为什么做）+ `RAG_INTERVIEW_GUIDE.md`（怎么讲）。
> 所有命令在项目根目录、`.venv` 下执行。评测跑 rerank 时本地模型较慢，耐心等或加 `python -u` 看进度。

---

## 0. 开关总览（做对照实验的钥匙）

每个措施都是一个**可开关的独立模块**，关掉某个开关再跑评测，就能量化它的贡献。

| 环境变量 | 默认 | 作用 | 代码入口 |
|---|---|---|---|
| `RAG_RERANK` | 1 | 交叉编码器精排 | `rag_rerank.rerank_enabled()` |
| `RAG_BM25` | 1 | BM25 + RRF 混合检索 | `rag_bm25.bm25_enabled()` |
| `RAG_ROUTE` | 1 | 元数据/文件名路由 | `rag_route.route_enabled()` |
| `RAG_HYDE` | 0 | HyDE 查询改写（已证伪，默认关）| `rag_hyde.hyde_enabled()` |
| `RAG_RERANK_GATE_MIN` | 0.85 | 证据型门控的 reranker 阈值 | `rag_gate._evidence_gate()` |
| `RAG_RELEVANCE_MAX_DIST` | 0.73 | 距离门控 strong 阈值 | `rag_gate._thresholds()` |
| `RAG_RECALL_N` | 20 | 粗召回候选数（喂给 rerank）| `rag_gate.RECALL_N` |

**对照实验范式**：先跑全开基线，再关一个开关跑，`--baseline` 对比 Δ 就是该措施的净贡献。
```bash
# 例：量化 BM25 混合检索的贡献
RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 .venv/bin/python eval_rag_bench.py --save /tmp/full.json
RAG_RERANK=1 RAG_BM25=0 RAG_ROUTE=1 .venv/bin/python eval_rag_bench.py --baseline /tmp/full.json
# 输出每个指标的 Δ↓ 箭头，就是 BM25 关掉后掉了多少
```

---

## 1. 跑评测（两套标尺）

**检索质量** —— `eval_rag_bench.py`（Recall@1/3/5 + MRR + gate 门槛）
```bash
.venv/bin/python eval_rag_bench.py                          # 跑当前配置
.venv/bin/python eval_rag_bench.py --baseline docs/rag_eval/round7_evidence_gate.json  # 对比基准看 Δ
.venv/bin/python eval_rag_bench.py --save docs/rag_eval/myrun.json --label "我的实验"   # 存为新基准
```
- 评测集：`tests/rag_bench_set.json`（100 题精确到目标文件 + 12 neg/12 pos 门槛样本）
- 接入点：`eval_rag_bench._ranked_sources()` → `rag_gate._retrieve_and_classify()`，所以**评测走的就是生产检索链路**。

**答案质量** —— `eval_rag_answer.py`（RAGAS 风格 LLM-as-judge）
```bash
.venv/bin/python eval_rag_answer.py --n 12 --save docs/rag_eval/answer_quality.json
```
- 三维度：忠实度 / 完整度 / 引用准确性。子集由 `_pick_subset()` 各域均匀抽样（可复现）。

---

## 2. 逐措施验证（代码位置 + 一行命令 + 看什么）

### ① 召回 · 混合检索（BM25 + 向量 + RRF）
- **代码**：`rag_bm25.py` → `bm25_search()`（jieba 分词 + BM25Okapi）、`rrf_fuse()`（RRF 融合）；
  接入在 `rag_gate._retrieve_and_classify()` 的「gate 判定后、rerank 前」。
- **单独看 BM25 召回了什么**：
```bash
.venv/bin/python -c "
from rag_bm25 import bm25_search
for d,m,s in bm25_search('混合检索 稀疏 密集', 5):
    print(round(s,2), m.get('source'))
"
```
- **看 RRF 融合逻辑**：`rrf_fuse()` 里 `score += 1/(k+rank)`，k=60。单测 `tests/test_rag_bm25.py::test_rrf_both_paths_rank_first`。

### ② 排序 · 交叉编码器精排
- **代码**：`rag_rerank.py` → `rerank()`（bge-reranker-base 对 query+doc 打分重排）；惰性加载、失败自动降级。
- **验证它在重排**：`tests/test_rag_rerank.py::test_rerank_reorders_by_score`。
- **量化贡献**：`RAG_RERANK=0` 跑 eval 对比——难题 R@1 会从 91 掉回 ~82。

### ③ 同质区分 · 元数据路由
- **代码**：`rag_route.py` → `audience_intent()`（query→人群标识）、`apply_audience_routing()`（career 路径文件间按桶重排）。
- **看路由命中**：
```bash
.venv/bin/python -c "from rag_route import audience_intent; print(audience_intent('算法岗怎么转大模型'))"  # → algorithm
```
- **量化贡献**：`RAG_ROUTE=0` 跑 eval——career R@1 会从 100 掉回 75（ca03 掉出 rank1）。

### ④ 拒答 · 双证据门控
- **代码**：`rag_gate._thresholds()`（距离阈值，按模型/库标定）+ `rag_gate._evidence_gate()`（reranker 二次确认，持否决权）。
- **看门控判定**：`_retrieve_and_classify()` 返回的 `in_kb / best / rerank_top`。
- **验证证据门控救边缘词**：
```bash
RAG_RERANK=1 .venv/bin/python -c "
from rag_gate import _retrieve_and_classify
g=_retrieve_and_classify('Docker 容器化部署',5)
print('in_kb=',g['in_kb'],'best=',round(g['best'],3),'rerank_top=',round(g['rerank_top'],3))
"   # 距离近(0.58)但 reranker 低(0.64)→ 被否决拒答
```
- 单测 `tests/test_rag_gate_evidence.py`（否决/退化/阈值/边界）。

### ⑤ 生成 · Grounded 合成
- **代码**：`rag_gate.synthesize_grounded_answer()` → `_grounded_messages()`（system prompt 强约束「只能基于资料、严禁资料外知识、末尾列引用编号」）。
- **看一道题的完整答案**：
```bash
RAG_RERANK=1 .venv/bin/python -c "
from rag_gate import gated_query
r=gated_query('什么是 RAG 检索增强生成')
print('mode=',r['mode']); print(r['answer'][:400])
"
```

---

## 3. 单题全过程诊断（面试现场可演示）

把一道 query 的「召回→融合→精排→门控」全摊开看：
```bash
RAG_RERANK=1 RAG_BM25=1 RAG_ROUTE=1 .venv/bin/python -c "
from rag_gate import _retrieve_and_classify
g=_retrieve_and_classify('混合检索 hybrid search 稀疏密集结合', 5)
print('in_kb   :', g['in_kb'])
print('best    :', round(g['best'],3), '(最小向量距离，门控用)')
print('rerank  :', round(g['rerank_top'],3), '(top1 交叉编码器分)')
print('matched :', g['matched_by'])
print('top5 sources:')
for i,m in enumerate(g['metas'],1): print('  ', i, m.get('source'))
"
```

---

## 4. 单元测试地图（每个文件测什么）

```bash
.venv/bin/python -m pytest tests/test_rag_bm25.py tests/test_rag_route.py \
  tests/test_rag_rerank.py tests/test_rag_gate_evidence.py -q
```
| 测试文件 | 覆盖 |
|---|---|
| `test_rag_bm25.py` | 中英分词 / 降级 / RRF 双路加权 / 占位距离 / 截断 |
| `test_rag_rerank.py` | 禁用截断 / 空输入 / 模型缺失降级 / 按分重排 / 开关 |
| `test_rag_route.py` | 意图识别 / 长词优先 / 提权降权 / 非 career 不动 / 开关 |
| `test_rag_gate_evidence.py` | 否决 / rerank 关退化 / 阈值覆盖 / 边界 |
| `tests/conftest.py` | 默认 `RAG_RERANK=0`（CI 不下载 1GB 模型；生产默认开） |

> **注意**：`conftest.py` 让单测默认关 rerank，所以单测里 BM25 融合/证据门控不触发——它们靠**纯函数单测**（`rrf_fuse`/`_evidence_gate`/`audience_intent`）覆盖逻辑，靠**手动跑 eval（rerank 开）**覆盖端到端。这是「快单测 + 慢集成」的常见分层。
