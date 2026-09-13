# OfferClaw PDF / Word RAG 入库优化指导文档

> ⚠️ **收录说明(2026-08-08,先读这段再读正文)**
>
> - **来源与定位**:本文为外部 AI 生成的入库优化**蓝图/路线图**,描述**目标态而非当前实现**——引用时严禁把其中能力当作"已实现"(本文 §21.12 自身的禁令)。当前真实现状以代码与 [metrics.json](../metrics.json) 为准。
> - **分层采纳决定(2026-08-08 评审)**:
>   - ✅ **第 1 层·已实施**:文件级内容去重(§5.1 → `knowledge_crawler.find_duplicate_content`,打分前拦截,重复上传零 LLM 消耗)· 表格检索文本句子化(§9.2 → `extract_text_for_kb`,docx 行级自含表头,任意分块边界语义不丢)· 入库 Golden Set 基线(§19 → `eval_ingestion.py`,基线 [rag_eval/ingestion/baseline_p0.json](rag_eval/ingestion/baseline_p0.json):文字 PDF 中英文句召回 1.0、docx 全召回;**表格 PDF 单元格召回 1.0 但行序仅 0.5 —— P0 结构性弱点的预登记靶点**)。
>   - ⏸ **第 2 层·条件触发**:Docling 结构化解析 + 页面预检。触发条件 = 开始高频喂复杂 PDF,或需要 row_order 指标显著改善;采纳前必须与 baseline_p0 同口径 A/B(测正才采纳)。
>   - 📦 **第 3 层·远期归档**:Canonical 全套模型 / 文档类型分块器 / 三文本 / 状态机+Manifest / OCR / VLM 增强 / 结构化第二轨道——单用户、语料以 md 为主的现状下暂不实施。
> - **仓库适配修正**:正文中以【仓库适配注】标出,共 5 处(许可证 / 目录风格 / 单用户口径 / 检索复用 / 分块边界)。


> 文档用途：作为后续开发 AI、代码助手和项目维护者实施 OfferClaw 文档入库优化时的统一指导。  
> 适用范围：PDF、DOCX 文档的解析、标准化、切块、索引、质量检测与可追溯存储。  
> 核心原则：**结构优先、按需增强、成本可控、来源可追溯、渐进式演进。**

---

## 1. 项目背景与本次优化目标

OfferClaw 面向个人求职场景，需要处理的知识材料主要包括：

- 个人简历；
- 岗位描述（JD）；
- 项目说明、项目复盘与技术文档；
- 面试资料、学习笔记和行业报告；
- 含表格、图片、公式或扫描页的 PDF / Word 文档。

本次工作的目标不是建设一个产品级文档管理平台，也不是一次性覆盖所有复杂版式，而是形成一套**工程完整、低成本、可逐步扩展、能够经受技术追问**的 RAG 入库链路。

最终应做到：

1. PDF 和 DOCX 可以稳定上传并入库；
2. 普通文本、标题、列表、表格、图片和公式不再被简单压成一段无结构文本；
3. 扫描页可以通过 OCR 兜底，但正常页面不重复 OCR；
4. 每个检索块都可以追溯到原文件、页码、章节和原始元素；
5. 入库结果既能服务稀疏检索，也能服务现有或未来的向量检索；
6. 解析器、切块策略或 embedding 升级后，可以有控制地重建索引；
7. 整体成本以本地处理为主，VLM 和云端解析仅作为可选增强。

---

## 2. 必须先统一的核心认识

### 2.1 “解析文档”和“切块入库”不是同一件事

错误做法：

```text
PDF / DOCX
    ↓
提取一大段字符串
    ↓
每 500 字切一块
    ↓
生成向量并入库
```

推荐做法：

```text
PDF / DOCX
    ↓
文件预检与处理路由
    ↓
恢复标题、段落、列表、表格、图片、公式及阅读顺序
    ↓
转换为统一的结构化中间表示
    ↓
根据文档类型和章节结构切块
    ↓
生成检索文本、元数据与来源信息
    ↓
写入稀疏索引 / 向量索引
```

解析阶段负责回答：

- 文档里有哪些元素？
- 元素的顺序是什么？
- 元素属于哪个章节？
- 表格的行列关系是什么？
- 图片、公式与正文是什么关系？
- 该内容位于哪一页、哪个坐标区域？

切块阶段负责回答：

- 哪些元素应该组合成一个检索单元？
- 一个块是否具备完整语义？
- 块太长时应该在哪里拆分？
- 块太短时能否与相邻内容合并？
- 检索时应该使用什么文本表示？

---

### 2.2 `BaseModel` 在这里表示“数据模型基类”

后续可以使用 Pydantic `BaseModel` 定义 OfferClaw 内部统一的数据结构。

这里的 `model` 是 **data model**，不是大语言模型，也不是函数形参。

例如：

```python
class DocumentBlock(BaseModel):
    block_id: str
    doc_id: str
    block_type: str
    raw_text: str
    page_number: int | None
```

其含义是：

> OfferClaw 中的每个文档块都必须符合这套字段与类型约束，解析器输出进入后续流程之前，需要先通过该数据模型校验。

---

### 2.3 LangChain / LangGraph Loader 不是完整入库方案

Loader 只能解决“把文件读进来”的一部分问题。完整的入库工程还必须包含：

- 文件去重与版本管理；
- 文档类型识别；
- 页面级预检；
- 解析策略路由；
- OCR 触发条件；
- 结构化中间表示；
- 文档类型感知切块；
- 索引写入；
- 解析质量检测；
- 来源追踪；
- 失败恢复；
- 重新入库策略。

因此，后续开发中不要把“换一个 Loader”误认为完成了 RAG 入库优化。

---

## 3. 总体架构

推荐链路：

```text
文件上传
   ↓
文件登记、哈希计算与安全检查
   ↓
Preflight：格式、页数、文本层、图片占比、乱码情况
   ↓
Ingestion Router：选择解析路径
   ├── PDF 正常文本路径
   ├── PDF 扫描页 OCR 路径
   ├── DOCX 原生结构解析路径
   └── 复杂元素选择性增强路径
   ↓
结构化解析
   ↓
质量门控与必要的局部回退
   ↓
统一 Canonical Document Model
   ↓
文档分类：resume / jd / project / general
   ↓
结构感知切块
   ↓
生成 retrieval_text、search_text 与 metadata
   ↓
写入原始数据表、稀疏索引和向量索引
   ↓
一致性校验与 Ingestion Manifest
```

建议按以下模块拆分代码，具体目录可结合现有仓库调整：

```text
offerclaw/
└── rag/
    ├── ingestion/
    │   ├── service.py
    │   ├── router.py
    │   ├── preflight.py
    │   ├── quality.py
    │   └── manifest.py
    ├── parsers/
    │   ├── base.py
    │   ├── pdf_parser.py
    │   ├── docx_parser.py
    │   ├── ocr.py
    │   └── enrichers.py
    ├── models/
    │   ├── document.py
    │   ├── block.py
    │   ├── chunk.py
    │   └── quality.py
    ├── chunkers/
    │   ├── base.py
    │   ├── resume.py
    │   ├── jd.py
    │   ├── project.py
    │   └── general.py
    ├── indexers/
    │   ├── sparse.py
    │   ├── dense.py
    │   └── coordinator.py
    └── evaluation/
        ├── golden_set.py
        ├── ingestion_eval.py
        └── retrieval_eval.py
```

不要为了匹配该示例而一次性重构整个项目。首先检查现有仓库，将新能力接入现有模块，避免无必要的目录和接口迁移。

> 【仓库适配注 2】本仓库为根目录平铺风格(`rag_ingest.py` / `knowledge_crawler.py` / `rag_gate.py` …),新能力按功能就近接入既有模块,**不**新建 `offerclaw/rag/` 目录树。

---

## 4. 推荐技术策略

### 4.1 主解析器与辅助组件

推荐组合：

| 职责 | 推荐方案 | 定位 |
|---|---|---|
| PDF 轻量预检 | PyMuPDF | 快速读取页数、文本层、图片、坐标等信息 |
| PDF / DOCX 结构化解析 | Docling | 主解析器，恢复标题、段落、表格、图片等结构 |
| OCR | 本地 RapidOCR、PaddleOCR 或兼容后端之一 | 只处理扫描页或解析失败区域 |
| 数据校验 | Pydantic | 定义统一中间模型与运行时校验 |
| 稀疏检索 | 现有 BM25 或 SQLite FTS5 | 低成本关键词召回 |
| ↑适配注 | 【仓库适配注 4】现有 jieba-BM25 已在役,**不**另起 FTS5 | 复用优先 |
| 向量检索 | 复用 OfferClaw 已有方案 | 基于统一 chunk 生成 embedding |
| 融合与重排 | 复用现有 RRF / Rerank | 本文重点不是重写检索层 |

> 【仓库适配注 1】上表推荐的 PyMuPDF 为 **AGPL** 许可,本仓库以 MIT 公开——预检实现改用 pypdf(BSD)近似(页字符数/图片对象数可得)或直接用 Docling 内置预检,不引入 PyMuPDF。

约束：

1. 不要同时接入多个功能重叠的 OCR 引擎；
2. 不要让每个解析器直接输出各自格式并进入索引；
3. 所有解析结果必须先转换成统一 Pydantic 数据结构；
4. 如果 OfferClaw 已有 BM25、向量检索、RRF 和 Rerank，不应为实现本文方案而删除；
5. 当前优先级是提升入库质量，而不是先更换向量数据库或 embedding 模型。

---

### 4.2 默认处理策略

默认策略应为：

```text
本地结构化解析
    >
页面级按需 OCR
    >
元素级 VLM 增强
    >
整页云端视觉解析
```

即：

- 普通文本型 PDF：直接结构化解析；
- 扫描 PDF：只 OCR 扫描页；
- 混合 PDF：正常页直接解析，异常页 OCR；
- DOCX：优先原生结构解析，不先转换成 PDF；
- 图片：先提取图注、附近正文和 OCR 文本；
- 公式：优先保存解析器已有的 LaTeX / 文本结果；
- VLM：仅用于无法通过普通解析表达的重要图片或复杂页面。

---

## 5. 文件登记、去重与版本管理

每个文件在解析前至少登记：

```text
doc_id
user_id
original_file_name
mime_type
file_size
sha256
source_path
upload_time
ingestion_status
parser_name
parser_version
chunker_version
embedding_version
active_version
```

> 【仓库适配注 3】本仓库当前诚实口径 = **单用户本地 + 租户接缝预留**(`X-OfferClaw-User` seam,未接数据路径)。清单中 `user_id` 类字段实现时一律降为可选、默认 `local`,不为本文档提前实现多租户。

### 5.1 精确去重

使用文件二进制内容的 SHA-256：

```text
相同 SHA-256 + 相同用户空间
→ 默认视为同一份文件
→ 不重复解析和 embedding
```

重复上传时可以返回已有入库记录，而不是重新消耗计算资源。

### 5.2 版本管理

同名文件不一定相同，相同文件名不能作为去重依据。

如果文件发生修改：

```text
新 SHA-256
→ 生成新的 document version
→ 新版本成功索引后标记为 active
→ 旧版本保留但默认不参与检索
```

### 5.3 处理幂等性

同一份文档使用相同的：

```text
file_hash
parser_version
chunker_version
embedding_version
```

重复执行时，应尽可能得到相同的 block_id、chunk_id 与索引结果。

推荐基于稳定字段生成 ID：

```text
block_id = hash(doc_id + page + order_index + normalized_content)
chunk_id = hash(doc_id + ordered_block_ids + chunker_version)
```

---

## 6. PDF 预检与路由

### 6.1 页面级预检指标

对每一页采集：

- 可提取字符数；
- 非空单词或中文字符数量；
- 替换字符 `�`、乱码字符或控制字符比例；
- 页面图片数量；
- 最大图片覆盖面积；
- 是否存在表格迹象；
- 文本块数量与坐标；
- 是否存在大量重复页眉、页脚；
- 是否疑似双栏；
- 解析出的阅读顺序是否异常。

### 6.2 页面类型建议

```text
TEXTUAL
    文本层正常，可以直接提取

SCANNED
    文本极少且页面主要由大图组成，需要 OCR

SUSPICIOUS
    有文本层但乱码严重、字符顺序异常或提取内容明显不完整

MIXED
    页面同时包含有效文本和局部扫描区域
```

可以使用以下规则作为第一版启发式起点，但不得当作行业标准：

```text
若可提取有效字符 < 30，且最大图片覆盖页面面积 > 70%
→ 倾向判定为 SCANNED

若替换字符或明显异常字符比例 > 2%
→ 倾向判定为 SUSPICIOUS

若文本正常但存在局部图片表格或截图
→ 倾向判定为 MIXED
```

所有阈值必须放入配置，并通过 OfferClaw 自己的测试文档校准。

### 6.3 路由逻辑

```python
if page.type == "TEXTUAL":
    use_structured_parser()
elif page.type == "SCANNED":
    use_page_ocr()
elif page.type == "SUSPICIOUS":
    try_structured_parser()
    compare_quality()
    fallback_to_ocr_if_needed()
elif page.type == "MIXED":
    parse_text_layer()
    extract_and_process_relevant_regions()
```

禁止默认对所有 PDF 页面全量 OCR。

---

## 7. DOCX 处理策略

DOCX 本身包含段落、标题样式、表格、关系文件和图片资源，优先保留这些原生语义。

推荐顺序：

```text
DOCX 原生结构解析
    ↓
恢复段落、标题、列表、表格和内嵌图片顺序
    ↓
识别是否存在复杂浮动对象
    ↓
必要时增加“渲染为 PDF 后的布局解析”作为补充
```

不要默认执行：

```text
DOCX → PDF → 再解析
```

因为这会把原本明确的 Word 语义结构转换成视觉坐标，可能造成：

- 标题层级丢失；
- 表格结构退化；
- 列表关系不清；
- 段落边界变化；
- 图片与文字关系变弱。

以下场景可以考虑可选的渲染回退：

- 大量文本框；
- 浮动图片和复杂环绕；
- SmartArt；
- 页面布局本身具有重要语义；
- 原生解析结果与用户看到的页面明显不一致。

该回退应是第二条路径，而不是替代原生 DOCX 解析。

---

## 8. 统一的 Canonical Document Model

### 8.1 为什么必须有统一中间表示

统一模型可以解耦：

```text
解析器
    与
切块器、索引器、检索器
```

以后即使把 Docling 替换成其他解析器，只要仍然输出同一套 `DocumentBlock`，后续模块不需要全部重写。

### 8.2 推荐 Pydantic 模型

以下代码是结构示例，开发时应根据仓库现有字段命名调整，而不是机械照搬。

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DocumentType = Literal["resume", "jd", "project", "general", "unknown"]

BlockType = Literal[
    "title",
    "heading",
    "paragraph",
    "list_item",
    "table",
    "picture",
    "formula",
    "header",
    "footer",
]


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    user_id: str
    original_file_name: str
    mime_type: str
    file_size: int
    sha256: str

    document_type: DocumentType = "unknown"
    source_path: str

    parser_name: str
    parser_version: str
    chunker_version: str

    created_at: datetime
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    doc_id: str
    block_type: BlockType
    order_index: int

    raw_text: str = ""
    retrieval_text: str = ""

    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None

    structured_data: dict[str, Any] = Field(default_factory=dict)
    asset_ids: list[str] = Field(default_factory=list)

    parser_name: str
    parser_version: str
    ocr_used: bool = False
    confidence: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    chunk_type: str

    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str]

    content: str
    retrieval_text: str
    search_text: str

    token_count: int
    page_start: int | None = None
    page_end: int | None = None

    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_count: int
    extracted_char_count: int
    empty_page_count: int
    ocr_page_numbers: list[int] = Field(default_factory=list)

    invalid_char_ratio: float
    repeated_line_ratio: float
    table_count: int
    picture_count: int

    passed: bool
    warnings: list[str] = Field(default_factory=list)
```

---

## 9. 各类元素的入库方法

### 9.1 标题、正文和列表

标题应形成章节路径：

```text
项目文档
└── 系统架构
    └── 记忆模块
```

对应块中保存：

```json
{
  "section_path": ["系统架构", "记忆模块"]
}
```

列表项不能简单地与前后正文完全脱离。检索文本中应包含列表所属标题，例如：

```text
章节：岗位要求 > 必须条件
要求：熟悉 Python、FastAPI 和关系型数据库。
```

页眉、页脚和页码通常不进入检索文本，但可以保留在原始解析结果中。对于每页重复的公司名、保密声明和页码，应识别后去除，避免污染 embedding 和 BM25 权重。

---

### 9.2 表格

表格必须同时保存两种表示。

#### 原始结构表示

```json
{
  "headers": ["技能", "熟练度", "使用项目"],
  "rows": [
    ["LangGraph", "熟悉", "OfferClaw"],
    ["FastAPI", "熟悉", "LocalFlow"]
  ],
  "merged_cells": []
}
```

用于：

- 表格展示；
- 精确字段提取；
- 后续重新生成检索文本；
- 审计解析是否正确。

#### 检索文本表示

```text
表格标题：技术能力
技能：LangGraph；熟练度：熟悉；使用项目：OfferClaw。
技能：FastAPI；熟练度：熟悉；使用项目：LocalFlow。
```

规则：

1. 不要只保存 HTML 或 Markdown 表格；
2. 不要把所有单元格无分隔地拼成一行；
3. 长表格按若干行切分；
4. 每个子块重复表头；
5. 保留表格所属章节；
6. 合并单元格应尽可能展开成可理解的上下文；
7. 表格图片如果结构解析失败，可以先 OCR，再决定是否调用视觉模型。

建议第一版将长表格控制在约 10～30 行一个块，实际数值根据 token 数和表格宽度调整。

---

### 9.3 图片

图片分为两类。

#### 装饰图片

例如：

- 简历头像；
- 公司 Logo；
- 背景图；
- 小图标；
- 页眉装饰。

处理方式：

```text
保存资源引用
不生成独立检索块
不调用 VLM
```

#### 信息图片

例如：

- 项目架构图；
- 流程图；
- 产品截图；
- 实验结果图；
- 证书扫描件；
- 表格截图；
- 含大量文字的图片。

推荐增强顺序：

```text
图注
    ↓
附近标题和正文
    ↓
图片 OCR 文本
    ↓
仍无法表达语义时才调用 VLM
```

图片检索文本示例：

```text
图片类型：项目架构图
图片标题：OfferClaw 工作流架构
所在章节：系统设计 > Agent 编排
图片文字：Supervisor、Profile、JD Parser、Retriever、Writer、Evaluator
上下文：该图展示了基于 LangGraph 条件路由的任务流转关系。
```

禁止：

- 把 Base64 写入检索文本；
- 对所有图片无差别调用 VLM；
- 只存图片路径而不给任何可检索语义；
- 把图片描述当作原始事实，忽略其可能存在的模型误差。

如果使用 VLM，应明确记录：

```text
description_source = "vlm"
vlm_model
prompt_version
generated_at
confidence / warning
```

---

### 9.4 公式

公式不是 OfferClaw 求职文档的核心，因此第一版不应投入过高成本。

处理原则：

1. 解析器能得到 LaTeX、MathML 或文本时直接保留；
2. 公式与前后解释段落组合；
3. 不把单独公式图片无上下文地做成 chunk；
4. 只有论文、技术报告或公式密集材料才启用专门公式识别；
5. 无法识别但不影响主要语义时，记录资源和警告，不阻断整份文档入库。

检索文本示例：

```text
章节：混合检索
公式：score = α × dense_score + β × sparse_score
解释：α 和 β 分别控制向量召回与关键词召回的权重。
```

---

## 10. 文档类型识别

OfferClaw 的入库不是纯通用问答，应识别文档类型：

```text
resume
jd
project
general
unknown
```

第一版可以使用：

```text
文件名规则
+ 标题关键词
+ 文档结构特征
+ 小模型或一次轻量 LLM 分类
```

示例特征：

### Resume

- 教育经历；
- 工作经历；
- 项目经历；
- 专业技能；
- 求职意向；
- 联系方式。

### JD

- 岗位职责；
- 任职要求；
- 加分项；
- 工作地点；
- 岗位级别；
- 学历和经验要求。

### Project

- 项目背景；
- 架构设计；
- 核心模块；
- 技术难点；
- 实验结果；
- 局限性；
- 后续计划。

文档类型判断结果应允许人工或上层业务逻辑修正。

---

## 11. 结构感知切块策略

### 11.1 总体原则

优先级：

```text
自然语义边界
    >
章节边界
    >
元素边界
    >
token 长度
```

不要把固定 token 长度作为第一切分依据。

推荐初始范围：

```text
普通文本块：约 350～700 tokens
强制拆分时 overlap：约 10%～15%
自然章节之间：默认不做 overlap
```

这些数值是第一版起点，不是固定标准，应通过检索评测调整。

> 【仓库适配注 5】本仓库首要分块边界 = **markdown 标题**(结构优先已成立);上述 token 数值仅作过长块**二次拆分**的参考,不作首要切分依据。

### 11.2 父子块

推荐保存：

- 父块：完整章节或完整项目；
- 子块：用于召回的较小语义单元。

检索时可以：

```text
先召回子块
    ↓
按需要补充父块上下文
```

这样既能保持召回精度，又能避免生成时上下文不足。

---

### 11.3 简历切块

推荐单位：

```text
基本信息
教育经历：每条经历一个块
工作经历：每段经历一个块
项目经历：每个项目一个父块，职责或成果可作为子块
技能：按类别或整体形成块
论文、奖项、证书：按条目或小组形成块
```

一个项目的以下内容尽量不要被任意拆散：

```text
项目目标
技术方案
个人职责
量化结果
项目限制
```

当项目描述过长时，可以拆为：

```text
项目概述
技术架构
核心实现
指标与结果
问题与改进
```

每个子块都重复项目名。

---

### 11.4 JD 切块

推荐单位：

```text
岗位概述
岗位职责
必须条件
加分条件
学历与年限
技术栈
地点和岗位类型
```

每个 bullet 可以保留为独立元素，但检索块可以合并若干相邻 bullet。

每个块应带岗位名称和公司名，避免不同 JD 混淆。

---

### 11.5 项目文档切块

推荐单位：

```text
项目背景
问题定义
用户流程
系统架构
核心模块
技术难点
异常处理
评测设计
实验指标
局限性
后续规划
```

对于 OfferClaw 和 LocalFlow 这类项目，应特别保留：

- 设计动机；
- 模块边界；
- 关键工程决策；
- 对照实验；
- 已实现能力；
- 尚未完成能力；
- 证据来源。

不要把尚未实现的规划与已经完成的能力混在同一个事实块中。

---

### 11.6 通用长文档

推荐流程：

```text
按 H1 / H2 / H3 分段
    ↓
章节内按段落、列表、表格、图片组织
    ↓
过长章节再按 token 限制拆分
    ↓
过短相邻元素在同一章节下合并
```

---

## 12. `raw_text`、`retrieval_text` 与 `search_text`

三者不能混为一谈。

### raw_text

解析器恢复出的原始文字，用于：

- 原文展示；
- 审计；
- 高亮；
- 重新生成检索文本。

### retrieval_text

用于 embedding 和提供给 LLM 的自然语言表示，通常包含：

```text
文档标题
文档类型
章节路径
元素类型
正文
必要的表头、图注或上下文
```

示例：

```text
文档：OfferClaw 项目说明
章节：系统架构 > 三层记忆
类型：项目模块
内容：系统将记忆分为事件记忆、偏好记忆和 SOP 记忆……
```

### search_text

用于稀疏检索的文本。中文场景可以：

```text
对 retrieval_text 做中文分词
或
使用支持连续字符匹配的索引策略
```

不要将为了 BM25 分词而加入的空格传给最终生成模型。

---

## 13. 索引设计

### 13.1 事实源与派生索引分离

推荐：

```text
结构化数据库
    保存 DocumentRecord、DocumentBlock、RetrievalChunk、Asset、Manifest

稀疏索引
    保存 chunk_id 与 search_text

向量索引
    保存 chunk_id、embedding 与检索 metadata
```

结构化数据库是事实源，BM25/FTS 和向量索引都是可重建的派生数据。

### 13.2 索引元数据

每个索引项至少保存：

```text
chunk_id
doc_id
user_id
document_type
file_name
section_path
page_start
page_end
active_version
parser_version
chunker_version
embedding_version
```

用于：

- 按用户隔离检索；
- 按文档类型过滤；
- 按指定文件检索；
- 返回页码和章节；
- 排除旧版本；
- 定位需要重建的索引。

### 13.3 已有混合检索的处理

如果 OfferClaw 当前已经实现：

```text
BM25
+ 向量召回
+ RRF
+ Rerank
```

本次不要先重写检索算法，而应确保两种召回都使用同一套高质量 `RetrievalChunk`。

建议先比较：

```text
旧入库结果 + 原检索链路
vs
新入库结果 + 原检索链路
```

如果召回质量已经明显提升，再决定是否调整 embedding、融合权重或 reranker。

---

## 14. 领域结构化数据：第二条入库轨道

OfferClaw 不应只保存通用文本块，还可以逐步增加业务结构化数据。

### 14.1 CandidateProfile

```text
Education[]
Experience[]
Project[]
Skill[]
Publication[]
Award[]
Certificate[]
```

### 14.2 JobSpec

```text
job_title
company
responsibilities[]
required_skills[]
preferred_skills[]
education_requirement
experience_requirement
location
employment_type
```

### 14.3 ProjectEvidence

```text
project_name
problem
architecture
technologies[]
responsibilities[]
metrics[]
limitations[]
future_work[]
evidence_block_ids[]
```

每个结构化字段必须保留证据引用：

```text
source_doc_id
source_block_id
page_number
```

禁止只保存：

```text
用户熟悉 LangGraph
```

而没有记录该结论来自哪份文件、哪个项目和哪段原文。

推荐顺序：

```text
规则或正则提取明显字段
    ↓
小模型或 LLM 做结构化补充
    ↓
字段冲突或低置信度时再使用更强模型
```

结构化提取应按“每份文档或每个大章节”进行，避免对每个小 chunk 重复调用 LLM。

---

## 15. 质量门控

### 15.1 解析质量检查

至少检查：

- 文档是否存在有效文本；
- 空白页比例；
- 乱码比例；
- 是否有大量重复行；
- 是否提取到标题；
- 表格数量是否异常为零；
- OCR 页是否合理；
- 页数是否与原文一致；
- block 顺序是否连续；
- 关键资源是否存在；
- Pydantic 校验是否通过。

### 15.2 切块质量检查

至少检查：

- 空 chunk；
- 极短 chunk；
- 超长 chunk；
- 只有页眉页脚的 chunk；
- 表格没有表头；
- 图片 chunk 没有任何文字代理；
- section_path 为空但原文有明显标题；
- 同一内容被大量重复切入；
- chunk 无法追溯到 block；
- page_start 大于 page_end。

### 15.3 索引一致性检查

```text
数据库有效 chunk 数
==
稀疏索引有效记录数
==
向量索引有效记录数（启用 dense 时）
```

如果不一致，应将该次入库标记为失败或部分失败，不要直接暴露给正常检索流量。

---

## 16. 状态机、失败恢复与 Manifest

推荐状态：

```text
RECEIVED
PREFLIGHTED
PARSED
NORMALIZED
CHUNKED
INDEXED
COMPLETED
FAILED
```

每个阶段结束后记录结果，避免发生失败后从头重跑。

### Ingestion Manifest 建议内容

```json
{
  "doc_id": "doc_xxx",
  "file_hash": "sha256...",
  "status": "COMPLETED",
  "parser": {
    "name": "docling",
    "version": "..."
  },
  "chunker_version": "resume-v1",
  "embedding_version": "current-model-v1",
  "page_count": 12,
  "block_count": 86,
  "chunk_count": 31,
  "ocr_pages": [7],
  "warnings": [],
  "started_at": "...",
  "completed_at": "..."
}
```

处理失败时，至少保存：

```text
失败阶段
异常类型
可读错误信息
是否可重试
已完成阶段
中间产物路径
```

---

## 17. 成本控制策略

必须落实以下原则。

### 17.1 OCR 页面级触发

错误：

```text
整份 PDF 全量 OCR
```

正确：

```text
先预检
只 OCR 扫描页、乱码页或缺失区域
```

### 17.2 VLM 元素级触发

错误：

```text
每一页都上传到 VLM
```

正确：

```text
只对高价值且普通解析无法理解的信息图片调用
```

### 17.3 缓存中间结果

缓存：

- 文件预检结果；
- 原始解析 JSON；
- OCR 结果；
- 图片描述；
- chunk 结果；
- embedding。

解析器升级时，不一定需要重做 OCR；切块器升级时，不需要重新解析文件；embedding 模型升级时，只需要重算向量。

### 17.4 分阶段重建

```text
parser_version 变化
→ 从解析阶段重建

chunker_version 变化
→ 从切块阶段重建

embedding_version 变化
→ 仅重建向量索引
```

---

## 18. 隐私与安全边界

OfferClaw 处理简历、联系方式和求职材料，应默认采用本地优先策略。

必须注意：

- 用户文件按 user_id 隔离；
- 检索必须带用户域过滤；
- 日志中不要完整打印简历内容；
- 不把原始文档默认发送到外部 VLM；
- 启用远程解析或视觉模型时必须有显式配置；
- 文件名和路径需要清理，防止路径穿越；
- 限制单文件大小、页数和处理时长；
- DOCX 本质上是压缩包，应防止异常压缩文件消耗资源；
- 失败文件不得进入正常索引；
- 删除文档时同步失效其稀疏和向量索引。

不要将该系统描述为“安全沙箱”。更准确的表述是：

```text
本地优先
用户域隔离
受控外部调用
可追溯处理
```

---

## 19. 评测集与验收指标

### 19.1 最小 Golden Set

至少准备以下文档：

1. 普通文本型 PDF；
2. 扫描 PDF；
3. 文本与扫描页混合 PDF；
4. 双栏 PDF；
5. 表格密集 PDF；
6. 图片或流程图密集 PDF；
7. 普通简历 DOCX；
8. 带表格和图片的 DOCX；
9. 格式异常或损坏文件；
10. 同一文件的重复上传和修改版本。

建议加入真实的：

- 一份中文简历；
- 一份英文简历；
- 两份不同公司的 JD；
- OfferClaw 项目文档；
- LocalFlow 项目文档；
- 一份包含架构图和表格的技术报告。

### 19.2 入库质量指标

建议记录：

```text
parse_success_rate
document_text_coverage
ocr_page_ratio
invalid_char_ratio
heading_retention_rate
table_retention_rate
source_location_coverage
empty_chunk_rate
overlong_chunk_rate
duplicate_chunk_rate
ingestion_latency
optional_external_call_count
```

其中：

- `source_location_coverage`：有页码或明确来源定位的 chunk 比例；
- `table_retention_rate`：人工标注表格中成功恢复结构的比例；
- `heading_retention_rate`：关键标题层级被正确恢复的比例。

### 19.3 检索质量指标

为每份测试文档构造问题与预期证据块，例如：

```text
这份 JD 的必需技能有哪些？
我的哪个项目证明我使用过 LangGraph？
OfferClaw 的记忆系统分为哪三层？
LocalFlow 是否已经实现 Repo 级代码 Agent？
项目文档中提到的量化指标来自哪一页？
```

建议观察：

```text
Recall@K
MRR
expected_evidence_hit_rate
citation_hit_rate
answer_supported_rate
```

第一阶段重点不是追求一个漂亮的总分，而是比较：

```text
旧入库方案
vs
新结构化入库方案
```

在同一套检索与生成配置下的差异。

---

## 20. 分阶段实施计划

### Phase 0：审计现有实现

任务：

1. 阅读当前文件上传、Loader、splitter、embedding 和索引代码；
2. 画出真实的数据流；
3. 找出已有能力与本文方案的重叠；
4. 确认当前索引的数据结构；
5. 建立基线测试结果。

交付物：

```text
当前架构说明
问题清单
可复用模块
最小改造路径
基线评测
```

禁止在未完成审计前直接全量重写。

---

### Phase 1：统一数据模型与 Manifest

任务：

- 新增 `DocumentRecord`；
- 新增 `DocumentBlock`；
- 新增 `RetrievalChunk`；
- 新增 `ParseQualityReport`；
- 新增入库状态和 Manifest；
- 实现 SHA-256 去重；
- 确保现有文本文件也能转换到统一模型。

验收：

- 数据模型通过单元测试；
- 相同文件不会重复入库；
- 每个 chunk 可追溯到 block 和 document；
- 失败有明确阶段和错误信息。

---

### Phase 2：PDF / DOCX 结构化解析

任务：

- 接入 PDF 预检；
- 接入主结构化解析器；
- DOCX 使用原生解析路径；
- 实现页面级 OCR 路由；
- 表格保存结构化 JSON；
- 图片资源单独保存；
- 清理重复页眉页脚；
- 输出质量报告。

验收：

- 正常 PDF 不触发全量 OCR；
- 扫描页可以恢复主要文字；
- 混合 PDF 只 OCR 必要页面；
- DOCX 标题、段落和表格顺序基本正确；
- 解析失败不会生成伪正常索引。

---

### Phase 3：文档类型感知切块

任务：

- 实现 resume chunker；
- 实现 JD chunker；
- 实现 project chunker；
- 实现 general chunker；
- 支持父子块；
- 生成 `retrieval_text` 与 `search_text`；
- 长表格拆分时重复表头。

验收：

- 一个完整项目不会被无意义截断；
- JD 的职责与要求不会混成一个无结构大块；
- chunk 保留文档标题和章节路径；
- 空块、重复块和超长块比例可控。

---

### Phase 4：接入现有检索链路

任务：

- 稀疏索引使用 `search_text`；
- 向量索引使用 `retrieval_text`；
- 保留 metadata filters；
- 返回结果带页码、章节和文件名；
- 旧版本文档不参与默认检索；
- 完成索引一致性检查。

验收：

- 原 BM25 / dense / RRF / rerank 链路可以继续工作；
- 检索结果可定位到原文；
- 相同问题在新入库方案下证据命中率优于基线；
- 删除或替换文档后不会检索到失效版本。

---

### Phase 5：评测与质量门控

任务：

- 建立 Golden Set；
- 建立入库质量指标；
- 建立检索问题集；
- 输出旧方案与新方案对比；
- 调整 OCR、切块和文档分类阈值。

验收：

- 有可重复运行的评测脚本；
- 每次 parser/chunker 修改后可以回归；
- 改动结果用数据证明，而不是只凭主观阅读；
- 失败案例有归因与修复记录。

---

### Phase 6：可选多模态增强

只有在评测证明存在明确收益时再实施：

- 信息图片 OCR；
- VLM 图片描述；
- 复杂表格视觉恢复；
- 公式识别；
- 页面级视觉检索；
- 云端高精度文档解析。

验收必须包含：

```text
提升了哪些问题的证据召回
新增多少计算或 API 成本
平均延迟增加多少
失败率和误描述风险如何控制
```

没有量化收益时，不应为了“多模态”标签默认开启。

---

## 21. 明确禁止的做法

后续 AI 不得默认执行以下方案：

1. 所有 PDF 全页 OCR；
2. 所有图片调用 VLM；
3. DOCX 先统一转 PDF；
4. 只保存 Markdown，不保存结构化数据；
5. 所有文档统一固定长度切块；
6. 表格只保存成无结构字符串；
7. chunk 不保存页码和章节；
8. 文件重复上传就重新 embedding；
9. 解析器升级后无法识别旧索引；
10. 失败文档仍进入检索；
11. 为了实现新方案删除现有可用检索链路；
12. 把规划中的能力写成已实现能力；
13. 没有评测就宣称“显著提高检索效果”；
14. 为追求技术栈丰富度同时接入多个同类库；
15. 一次性把入库、检索、Agent 编排和 UI 全部重构。

---

## 22. 面向开发 AI 的执行要求

后续开发 AI 在开展每一个入库优化任务时，应遵守以下流程。

### 22.1 修改前

必须先：

1. 检查仓库当前真实实现；
2. 说明现有链路；
3. 指出本次改动对应本文哪个阶段；
4. 列出准备复用、修改和新增的模块；
5. 判断是否会影响已有索引兼容性。

### 22.2 修改中

必须：

- 优先小步提交；
- 保持接口清晰；
- 对解析器输出做 Pydantic 校验；
- 对外部调用设置显式开关；
- 对关键启发式阈值使用配置项；
- 给关键路由与回退逻辑写测试；
- 避免在业务逻辑中散落解析器专属数据结构；
- 保留来源与版本信息。

### 22.3 修改后

必须输出：

```text
1. 修改了哪些文件
2. 每个文件的作用
3. 数据流发生了什么变化
4. 新增了哪些测试
5. 测试结果
6. 对旧数据或旧索引是否兼容
7. 当前仍未覆盖的情况
8. 下一步最有价值的优化
```

禁止只回复“已完成”而不说明验证结果。

---

## 23. 可直接提供给代码 AI 的总指令

以下内容可以作为后续开发任务的总提示词前缀：

```text
你正在优化 OfferClaw 的 PDF / DOCX RAG 入库链路。

请将《OfferClaw PDF / Word RAG 入库优化指导文档》视为本次工作的架构约束和实施依据。你的目标不是重写整个项目，也不是堆叠更多框架，而是在检查当前仓库真实实现后，以最小、可测试、可回滚的方式逐步改进入库质量。

必须遵守：

1. 区分文件解析、结构标准化、切块和索引四个阶段。
2. 所有解析器输出先转换成统一的 Pydantic DocumentBlock，不允许直接进入索引。
3. PDF 先预检，正常页面不 OCR；扫描或异常页面才局部 OCR。
4. DOCX 优先原生结构解析，不默认转 PDF。
5. 表格同时保存结构化表示和检索文本。
6. 图片默认只保存资源、图注、上下文和 OCR；只有高价值且普通方法无法表达时才考虑 VLM。
7. chunk 必须保留 doc_id、block_ids、章节路径、页码和版本信息。
8. 优先按简历、JD、项目文档的业务结构切块，固定 token 仅用于二次拆分。
9. 复用现有 BM25、向量召回、RRF 和 rerank，不因本次任务无必要重写检索层。
10. 使用文件哈希去重，使用 parser/chunker/embedding 版本支持可控重建。
11. 失败文档不得进入正常索引。
12. 每次改动必须有单元测试、最小端到端测试和基线对比。
13. 不把计划能力描述为已完成能力，不在没有指标时声称效果显著提升。
14. 如果仓库现状与指导文档不同，先说明差异，并选择兼容现状的最小改造方案。

每次执行任务时，请按以下格式工作：
- 当前实现审计
- 问题定位
- 本次最小方案
- 代码改动
- 测试与结果
- 兼容性与风险
- 后续建议
```

---

## 24. 最终架构结论

OfferClaw 的推荐 RAG 入库策略可以概括为：

> 以本地结构化解析为主，通过 PDF 页面级预检决定是否局部 OCR；DOCX 优先保留原生结构；将标题、正文、列表、表格、图片和公式统一转换成带页码、章节、资源和版本信息的 Pydantic 文档块；再根据简历、JD、项目材料和通用文档采用不同切块策略，生成可同时服务 BM25 与向量检索的统一 RetrievalChunk；通过哈希去重、版本管理、质量门控、来源追踪和可重复评测，逐步提升入库效果；VLM 与视觉检索仅在明确存在收益时作为增强能力加入。

实现重点不是“用了哪个 PDF Loader”，而是建立以下完整工程能力：

```text
文档预检
+ 策略路由
+ 统一数据模型
+ 结构感知切块
+ 可追溯索引
+ 质量门控
+ 成本控制
+ 可重复评测
```

这套方案既能提升 OfferClaw 的实际检索质量，也能够形成一段准确、不过度包装、可以承受面试追问的工程叙述。
