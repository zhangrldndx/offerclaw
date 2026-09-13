#!/usr/bin/env python3
"""Build reviewer-A's chunk-level qrels overlay from manually selected spans.

This script never runs retrieval and never inspects ranked results.  Each
``TargetSpec`` below is an independent reviewer decision made from the query,
its answer-location note (when present), and the approved source document.
The script only resolves those chosen spans to stable IDs in the frozen Chroma
collection and computes reproducible hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_qrels import (  # noqa: E402
    answer_span_hash,
    normalize_answer_span,
    validate_qrels_against_collection,
    validate_qrels_overlay,
)


COLLECTION = "offerclaw_local_bge_base_zh_768"
DEFAULT_OUTPUT = ROOT / "docs" / "rag_eval" / "qrels" / "rag_bench_paraphrase_reviewer_a.json"


@dataclass(frozen=True)
class TargetSpec:
    source: str
    heading_path: tuple[str, ...]
    excerpt: str
    note: str
    relevance: str = "direct"


@dataclass(frozen=True)
class ReviewSpec:
    outcome: str
    note: str
    targets: tuple[TargetSpec, ...] = ()


def T(source: str, headings: tuple[str, ...], excerpt: str, note: str, relevance: str = "direct") -> TargetSpec:
    return TargetSpec(source, headings, excerpt, note, relevance)


def A(note: str, *targets: TargetSpec) -> ReviewSpec:
    return ReviewSpec("accepted", note, tuple(targets))


def P(note: str, *targets: TargetSpec) -> ReviewSpec:
    return ReviewSpec("partial", note, tuple(targets))


def U(note: str) -> ReviewSpec:
    return ReviewSpec("unsupported", note)


RAG_BASIC = "llm_app_interview_02_rag_basics.md"
RAG_CHAIN = "llm_app_interview_03_rag_full_chain.md"
TOOL = "llm_app_interview_05_agent_tool_calling.md"
TRANSFORMER = "llm_algorithm_basic_03_深度学习笔记——Transformer.md"
GPT_BERT = "llm_algorithm_basic_04_深度学习笔记——GPT_BERT_T5.md"
COMPRESS = "llm_algorithm_basic_05_深度学习笔记——模型压缩和优化技术_蒸馏_剪枝_量化.md"
LORA = "llm_algorithm_basic_06_模型微调之LoRA.md"


# Reviewer-selected spans.  Short spans are intentionally complete claims,
# never headings alone and never chunks selected because a retriever ranked them.
SPECS: dict[str, ReviewSpec] = {
    "pp01": A("基础文档给出数据准备与应用两个阶段的完整链路。",
        T(RAG_BASIC, ("正文采集", "RAG应用流程"), "完整的RAG应用流程主要包含两个阶段：\n- 数据准备阶段：数据提取——>文本分割——>向量化（embedding）——>数据入库\n- 应用阶段：用户提问——>数据检索（召回）——>注入Prompt——>LLM生成答案", "直接列出端到端步骤。")),
    "pp02": A("同一正文连续解释知识边界、幻觉与私域安全。",
        T(RAG_BASIC, ("正文采集", "RAG出现的原因"), "实时性、离线数据无法解答，模型自身的知识完全源于它的训练数据，而现有的主流大模型（ChatGPT、文心一言、通义千问…）的训练集基本都是构建于网络公开的数据，对于一些实时性的、非公开的或离线的数据是无法获取到的，这部分知识也就无从具备。\n\n幻觉问题\n\n所有的AI模型的底层原理都是基于数学概率，其模型输出实质上是一系列数值运算，大模型也不例外，所以它有时候会一本正经地胡说八道", "直接解释外挂知识的必要性与模型幻觉。")),
    "pp03": A("固定长度与语义分割分别有明确原理说明。",
        T(RAG_BASIC, ("正文采集", "RAG优化方法", "文本分割"), "固定长度分割：根据embedding模型的token长度限制，将文本分割为固定长度（例如256/512个tokens），这种切分方式会损失很多语义信息，一般通过在头尾增加一定冗余量来缓解。", "直接说明固定切分及其代价。"),
        T(RAG_BASIC, ("正文采集", "RAG优化方法", "文本分割"), "语义分割：通过计算向量化后的文本的相似度来进行语义层面的分割。", "直接说明语义切分依据。")),
    "pp04": A("HyDE 段落明确描述先生成假设文档再检索。",
        T(RAG_CHAIN, ("正文采集", "HyDE"), "但是，如果让大模型生成一个假设的相关文档，然后使用它来执行相似性检索可能会得到意想不到的结果。这就是 假设性文档嵌入（Hypothetical Document Embeddings，HyDE） 背后的关键思想。", "直接定义问题所述方法。")),
    "pp05": A("RAG Fusion 段落同时包含多查询和 RRF 融合。",
        T(RAG_CHAIN, ("正文采集", "Multi Query 多查询", "RAG Fusion"), "RAG Fusion 和 MultiQueryRetriever 基于同样的思路，在 Multi Query 多查询策略生成子问题并检索的基础上，它对检索结果执行 倒数排名融合（Reciprocal Rank Fusion，RRF） 算法【后面再讲】，使得检索效果更好。", "直接覆盖改写、多次检索和倒数排名融合。")),
    "pp06": A("窗口搜索与父文档搜索直接解决小块上下文丢失。",
        T(RAG_BASIC, ("正文采集", "RAG优化方法", "文本分割"), "句子窗口搜索：相反，文档文块太小会导致上下文的缺失。其中一种解决方案就是窗口搜索，该方法的核心思想是当提问匹配好文档块后，将该文档块周围的块作为上下文一并交给LLM进行输出", "直接支持命中小块后带周围块。"),
        T(RAG_BASIC, ("正文采集", "RAG优化方法", "文本分割"), "父文档搜索先将文档分为尺寸更大的主文档，再把主文档分割为更短的子文档两个层级，用户问题会与子文档匹配，然后将该子文档所属的主文档发送给LLM。", "直接支持命中子块后展开父块。")),
    "pp07": A("混合检索正文说明稀疏与密集并行和结果融合。",
        T("2026-06-19_rag技术入门与架构演进_a9feef.md", ("混合检索",), "混合检索便是基于这个思路，通过结合多种搜索算法（最常见的是稀疏与密集检索）来提升搜索结果相关性和召回率。", "直接定义混合检索。"),
        T("2026-06-19_rag技术入门与架构演进_a9feef.md", ("混合检索",), "混合检索通常并行执行两种检索算法，然后将两组异构的结果集融合成一个统一的排序列表。", "直接说明落地方式。")),
    "pp08": A("Function Calling 文档明确说明自然语言到结构化调用以及四种学习方式。",
        T(TOOL, ("正文采集", "大模型学习 Function Calling 能力的方式"), "Function Calling 的能力，本质是 将自然语言映射为结构化的函数调用格式。模型需要在理解用户意图的同时，结合工具 Schema，正确输出函数名和参数。其学习方式主要包括 微调、少样本提示、强化学习、工程侧优化 四类。", "直接回答工具调用能力如何形成。")),
    "pp09": A("L1 与 L2 各有独立正文定义，需共同作为答案证据。",
        T("llm_app_interview_07_agent_planner_l1_linear.md", ("正文采集",), "规划在执行前完成，且不依赖执行反馈进行调整，中间步骤的结果不会反向影响原始计划，因此本质上属于开环规划。", "直接定义预先规划、执行中不调整。"),
        T("llm_app_interview_08_agent_planner_l2_reactive.md", ("正文采集",), "规划不再一次性生成，而是通过与环境或工具的交互，根据反馈动态调整后续行动，形成闭环控制。", "直接定义基于反馈动态调整。")),
    "pp10": A("Transformer 正文直接说明全局关注，并给出 Q/K/V 权重机制。",
        T(TRANSFORMER, ("正文", "自注意力机制"), "自注意力机制允许输入序列中的每个单词可以“关注”到其他单词，从而捕捉全局上下文信息。", "直接说明词间关注机制。")),
    "pp11": A("问题要求比较三种架构，分别标注三处直接定义。",
        T(GPT_BERT, ("正文",), "GPT使用的是Transformer的解码器（Decoder） 部分。这个架构主要由自注意力机制（self-attention）和前馈神经网络（Feedforward Neural Networks）组成。", "直接支持 GPT 解码器架构。"),
        T(GPT_BERT, ("正文",), "BERT使用的是Transformer的编码器（Encoder）部分。与解码器不同，编码器主要用于表示输入序列中的每个词，并关注该词与序列中其他词之间的关系。", "直接支持 BERT 编码器架构。"),
        T(GPT_BERT, ("正文", "T5", "T5 的架构"), "T5 使用完整的 Transformer 序列到序列（Seq2Seq）架构，包括编码器和解码器，适合处理生成和理解类任务。", "直接支持 T5 编解码架构。")),
    "pp12": P("文档正文直接覆盖蒸馏和剪枝，但量化只有目录标题、没有可验证原理正文；不能补造量化金标。",
        T(COMPRESS, ("正文", "知识蒸馏"), "知识蒸馏是一种重要的模型压缩方法，通过让小模型（学生模型）学习大模型（教师模型）的知识，达到模型精简和高效推理的目的。", "直接支持蒸馏。"),
        T(COMPRESS, ("正文", "权重剪枝"), "模型剪枝是一种减少模型冗余参数的方法，通常通过移除对模型性能影响较小的权重或神经元来降低模型的计算复杂度和存储需求。", "直接支持剪枝。")),
    "pp13": A("LoRA 正文直接说明冻结 W 并训练低秩 A、B。",
        T(LORA, ("正文", "核心原理"), "LoRA 的基本思想是将它的更新部分 ΔW 分解为两个秩更小的矩阵 A 和 B 的乘积，从而减少需要更新的参数量，并且仅仅更新这两个矩阵。而不是更新整个 W 矩阵，保持预训练模型的原始性能。", "直接回答低秩参数高效微调。")),
    "pp14": U("指定 MVCC 文档只有 MVCC 标题和锁分类，未出现版本链、ReadView 或读写不互斥机制；标题不能作为答案金标。"),
    "pp15": U("指定联合索引文档只有“最左前缀原则”标题，没有解释联合索引排序或为何跳过首列无法定位；不以标题冒充直接证据。"),
    "pp16": P("事务文档直接覆盖原子性和回滚，但隔离级别只有目录标题，没有各级别语义正文。",
        T("backend_basic_03_05_mysql_transactions.md", ("正文", "事务的特性：ACID", "原子性"), "事务中的所有操作要么全部完成，要么全部不做，不能只执行部分操作。换句话说，事务是“原子”级的，事务的操作不可分割。\n- 如果事务在执行过程中发生错误，所有已执行的操作都会回滚，数据会恢复到事务开始之前的状态。", "直接支持要么全成要么回滚。")),
    "pp17": A("Redis 单线程文档直接给出内存访问、无切换和 I/O 多路复用原因。",
        T("backend_basic_04_14_redis_single_thread.md", ("正文",), "CPU瓶颈不明显：在 Redis 中，瓶颈通常不是 CPU 的计算能力，而是 网络 I/O 和 内存访问。大多数操作都可以在内存中进行，且 Redis 的数据结构和操作都经过高度优化。", "直接支持内存访问优势。"),
        T("backend_basic_04_14_redis_single_thread.md", ("正文",), "减少多线程开销：在多线程模型中，操作系统需要进行 线程切换 和 锁管理，这些都会带来一定的性能开销。", "直接支持单线程减少切换和锁管理。"),
        T("backend_basic_04_14_redis_single_thread.md", ("正文",), "高效的 I/O 多路复用：Redis 使用 I/O 多路复用（如 epoll，kqueue 等）技术来处理客户端请求，这意味着 Redis 可以高效地管理多个客户端连接，而不需要为每个客户端分配一个线程。", "直接支持高并发 I/O。")),
    "pp18": A("缓存问题文档分别定义穿透、击穿和雪崩并给出防护。",
        T("backend_basic_04_02_redis_cache_problems.md", ("正文", "缓存穿透"), "定义：用户请求的键在缓存和数据库中都不存在，导致每次请求都会直接访问数据库。", "直接定义不存在 key 的场景。"),
        T("backend_basic_04_02_redis_cache_problems.md", ("正文", "缓存击穿"), "定义：当缓存中某个高频访问的热点键过期时，大量请求同时到达数据库，造成数据库压力骤增。", "直接定义热点 key 过期。"),
        T("backend_basic_04_02_redis_cache_problems.md", ("正文", "缓存雪崩"), "定义：在某一时间段内，缓存集中过期 或 Redis 服务宕机，导致大量请求直接涌向数据库，可能引发数据库崩溃。", "直接定义大量 key 同时失效。")),
    "pp19": A("持久化文档直接对比 RDB 快照和 AOF 写命令日志。",
        T("backend_basic_04_09_redis_persistence.md", ("正文",), "RDB：在指定的时间间隔能对你的数据进行快照存储。", "直接支持定时快照。"),
        T("backend_basic_04_09_redis_persistence.md", ("正文",), "AOF：记录每次对服务器写的操作，当服务器重启的时候会重新执行这些命令来恢复原始的数据。", "直接支持写命令流水。")),
    "pp20": A("ZSet 文档直接解释 score 排序、范围查询和跳表取舍。",
        T("backend_basic_04_07_redis_zset_skiplist.md", ("正文",), "它是一个 没有重复成员 且每个成员都关联着一个 分数（score） 的集合，Redis 会根据成员的分数对它们进行 自动排序。", "直接支持按分数排名。"),
        T("backend_basic_04_07_redis_zset_skiplist.md", ("正文",), "跳表（Skiplist）是一种 概率型的数据结构，旨在提供与平衡二叉树类似的查找性能（$O(\\log N)$），但其实现相对更简单且更易于理解。", "直接支持跳表复杂度和取舍。")),
    "pp21": A("消息队列导论直接说明削峰、解耦和异步三类痛点。",
        T("backend_basic_05_01_mq_intro_part1.md", ("正文",), "消息队列（Message Queue，MQ） 是在微服务系统和分布式架构中实现异步通信的技术。是分布式系统中重要的组件，主要解决应用耦合，异步处理，流量削锋等问题", "直接概括问题中的三项价值。")),
    "pp22": A("零基础路线正文明确建议先应用开发、后后端、再算法加分项。",
        T("llm_app_intro_05_zero_foundation_path.md", ("页面结构目录",), "如果你是零基础，则先学大模型应用开发技术栈，有机会再补后端开发相关的技术栈，最后再去看一些加分项（算法岗的一些通用知识）", "直接给出顺序。")),
    "pp23": A("后端转型路线直接给出应用栈学习顺序。",
        T("llm_app_intro_06_backend_transition_path.md", ("页面结构目录",), "学习顺序：先学大模型应用开发技术栈，最后再去看一些加分项（算法岗的一些通用知识）", "直接给出迁移主线。")),
    "pp24": A("入门问答中岗位职责、技术栈和市场分别有直接正文。",
        T("llm_app_intro_04_entry_qa.md", ("正文采集",), "大模型应用开发岗技术栈 = 后端开发技术栈 + AI 落地相关技术栈", "直接支持技术栈。"),
        T("llm_app_intro_04_entry_qa.md", ("如何区别大模型开发岗和大模型算法岗？",), "大模型算法岗主要负责的是模型的更新迭代，更关注模型的效果。而大模型开发岗主要是使用算法岗产出的模型去解决一些实际问题，使AI应用落地，工程侧进一步优化模型的落地使用。", "直接区分日常工作。"),
        T("llm_app_intro_04_entry_qa.md", ("市场行情怎么样？岗位多不多？",), "目前来说这个岗属于一个新兴岗，是在大厂中这种岗位的比例会更高一点，如果你的目标是大厂，可能这种岗位的竞争压力会比传统的前后端开发要低，上岸几率更高", "直接支持招聘市场判断。")),

    "h9ll01": A("答案位置说明与正文完整对应，三类原因均在同一块。",
        T(RAG_BASIC, ("正文采集", "RAG出现的原因"), "对于一些实时性的、非公开的或离线的数据是无法获取到的，这部分知识也就无从具备。\n\n幻觉问题\n\n所有的AI模型的底层原理都是基于数学概率", "直接支持知识边界与幻觉。"),
        T(RAG_BASIC, ("正文采集", "RAG出现的原因"), "对于企业来说，数据安全至关重要，没有企业愿意承担数据泄露的风险，将自身的私域数据上传第三方平台进行训练。", "直接支持私域数据安全限制。")),
    "h9ll02": A("HNSW 搜索过程与余弦关系各有直接段落。",
        T(RAG_CHAIN, ("正文采集", "HNSW"), "搜索从高层次图开始，通过找到与查询点最相似的顶层节点，逐步下降到底层图。每个层中，HNSW通过邻近节点的遍历来找到与查询点最近的节点，逐层缩小搜索范围，直到找到最相似的候选结果。", "直接支持分层图搜索。"),
        T(RAG_CHAIN, ("正文采集", "HNSW"), "HNSW 算法本身并不限定使用哪种相似性度量，它能够支持多种距离度量方法，包括欧氏距离、余弦相似度、杰卡德距离等。具体应用中，余弦相似度可以作为HNSW中度量向量相似性的一种方法。", "直接支持余弦度量关系。"),
        T(RAG_CHAIN, ("正文采集", "HNSW"), "HNSW通过图结构大大减少了需要计算的候选点数，典型的时间复杂度可以达到接近常数级别 $O(\\log n)$。", "直接支持相对暴力搜索的复杂度收益。")),
    "h9ll03": A("ReAct 正文直接对比仅推理、仅行动与交替循环。",
        T("llm_app_interview_04_agent_basics.md", ("正文采集", "ReAct"), "仅推理（Reasoning Only）：LLM 仅仅基于已有的知识进行推理，生成答案回答这个问题。很显然，如果 LLM 本身不具备这些知识，可能会出现幻觉，胡乱回答一通。", "直接说明仅推理的局限。"),
        T("llm_app_interview_04_agent_basics.md", ("正文采集", "ReAct"), "仅行动（Acting Only）： 大模型不加以推理，仅使用工具（比如搜索引擎）搜索这个问题，得出来的将会是海量的资料，不能直接回到这个问题。", "直接说明仅行动的局限。"),
        T("llm_app_interview_04_agent_basics.md", ("正文采集", "ReAct"), "推理+行动（Reasoning and Acting）： LLM 首先会基于已有的知识，并审视拥有的工具。当发现已有的知识不足以回答这个问题，则会调用工具，比如：搜索工具、生成报告等，然后得到新的信息，基于新的信息重复进行推理和行动，直到完成这个任务。", "直接说明循环结合方式。")),
    "h9ll04": A("Function Calling 正文给出四种方式和低成本常用组合。",
        T(TOOL, ("正文采集", "大模型学习 Function Calling 能力的方式"), "其学习方式主要包括 微调、少样本提示、强化学习、工程侧优化 四类。", "直接列举训练和工程方式。"),
        T(TOOL, ("正文采集", "常用方法"), "在实际工程落地中，Few-shot Prompting + Schema Constraint 往往是成本最小、最容易部署的方案", "直接回答何种方案省钱易部署。")),
    "h9ll05": A("Planner 文档直接给出 Workflow 与 Agent 的边界。",
        T("llm_app_interview_06_agent_planner.md", ("正文采集", "为什么需要 Planning 能力"), "Workflow 是“人先想清楚并且预设所有的步骤，系统照着跑”；", "直接支持预设流程。"),
        T("llm_app_interview_06_agent_planner.md", ("正文采集", "为什么需要 Planning 能力"), "Agent 是“可基于实际反馈（如任务异常、外界干扰）实时调整方案，确保最后能完成目标”。", "直接支持反馈调整。")),
    "h9ll06": A("Harness 定义直接列出模型外系统的职责。",
        T("llm_app_interview_10_harness_engineering.md", ("正文采集", "Harness 的定义"), "Harness 是：所有不属于模型本体、但又参与 Agent 执行的代码、配置和运行逻辑。", "直接定义外部系统。"),
        T("llm_app_interview_10_harness_engineering.md", ("正文采集", "Harness 的定义"), "Harness 负责把智能组织成生产力， 提供的上下文注入、控制机制、行动能力、持久化能力以及观察和验证，把模型的推理能力转化为可持续的任务执行能力。", "直接列出 Harness 职责。")),
    "h9ll07": A("核心工作流正文直接给出两种防复犯形式。",
        T("llm_app_interview_11_harness_core_workflow.md", ("正文采集", "第 5 步：构建 Harness"), "每当你发现 agent 犯了一个错误，你就花时间设计出一个解决方案，让它以后再也不会犯同样的错误。", "直接提出防复犯原则。"),
        T("llm_app_interview_11_harness_core_workflow.md", ("正文采集", "第 5 步：构建 Harness"), "这件事大致有两种形式：\n\n1. 更好的隐式提示（AGENTS.md）", "直接给出第一类沉淀方式。"),
        T("llm_app_interview_11_harness_core_workflow.md", ("正文采集", "第 5 步：构建 Harness"), "2. 真正写出来的工具\n\n例如，用于截图的脚本、运行经过筛选的测试脚本等等。", "直接给出第二类沉淀方式。")),
    "h9ll08": A("记忆章节直接说明 LLM 无状态和引入记忆系统的必要性。",
        T("2026-06-19_hello_agents_rag与agent核心精选_150ed4.md", ("第八章：让智能体拥有记忆", "为何智能体需要记忆与RAG"), "当前的大语言模型虽然强大，但设计上是<strong>无状态的</strong>。这意味着，每一次用户请求（或API调用）都是一次独立的、无关联的计算。模型本身不会自动“记住”上一次对话的内容。", "直接支持跨会话遗忘原因。"),
        T("2026-06-19_hello_agents_rag与agent核心精选_150ed4.md", ("第八章：让智能体拥有记忆",), "要解决这个问题，我们的框架需要引入记忆系统。", "直接支持解决方向。")),
    "h9ll09": A("LocalFlow 项目文档给出优化前后延迟与长驻服务方案。",
        T("localflow_project_docs.md", ("AGENT_SERVER.md",), "The existing `DockerWorkspace` and `RemoteWorkspace` shell out one command (`docker exec` / `ssh`) per Workspace op — that's ~100-300 ms per call.", "直接量化逐命令开销。"),
        T("localflow_project_docs.md", ("AGENT_SERVER.md",), "With an `AgentServer` running long-lived in the container / on the remote, the harness opens one HTTP connection and reuses it for the entire run, dropping per-op latency to network RTT (~1-5 ms on localhost, ~10-50 ms over LAN).", "直接说明长驻服务与连接复用收益。")),
    "h9ll10": A("Agent Skills 文档直接说明三阶段渐进式披露。",
        T("llm_app_interview_09_agent_skills.md", ("正文采集", "渐进式披露"), "Agent Skills 通过 渐进式披露（progressive disclosure） 工作。agent 不会一开始读取所有 Skills 的完整内容，而是分 3 个阶段加载。", "直接说明不会一次加载全部说明。")),
    "h9ll11": A("基础 RAG 文档使用完全相同的意大利指代例子并给出查询压缩。",
        T(RAG_BASIC, ("正文采集", "RAG优化方法", "用户提问", "Follow Up Questions"), "Follow Up Questions/查询问题压缩：使用LLM针对历史对话和当前问题生成一个独立问题。这个方法主要针对以下情况：a. 后续问题建立在前一次对话的基础上，或引用了前一次谈话。", "直接回答多轮指代的处理。")),
    "h9ll12": A("Decomposition 段落明确列出递归和独立两种处理方式。",
        T(RAG_CHAIN, ("正文采集", "Decomposition"), "使用第一个问题的答案 + 检索来回答第二个问题，以此类推。", "直接支持串行依赖子问题。"),
        T(RAG_CHAIN, ("正文采集", "Decomposition"), "独立解决每一个问题，最后将每个答案合并为最终答案。", "直接支持并行独立子问题。")),
    "h9ba01": A("事务原子性正文与答案位置完全对应。",
        T("backend_basic_03_05_mysql_transactions.md", ("正文", "事务的特性：ACID", "原子性"), "如果事务在执行过程中发生错误，所有已执行的操作都会回滚，数据会恢复到事务开始之前的状态。", "直接支持失败自动回滚。")),
    "h9ba02": A("MVCC 文档虽然不支持 MVCC 原理，但其锁分类正文直接支持本题。",
        T("backend_basic_03_11_mysql_mvcc.md", ("正文", "数据库并发控制——锁机制", "锁的分类"), "DQL（Data Query Language，数据查询语言）：查询数据（如 SELECT）时使用 读锁。", "直接给出查询对应读锁。"),
        T("backend_basic_03_11_mysql_mvcc.md", ("正文", "数据库并发控制——锁机制", "锁的分类"), "DML（Data Manipulation Language，数据操作语言）：对数据进行增、删、改操作（如 INSERT、DELETE、UPDATE）时使用 写锁。", "直接给出增删改对应写锁。"),
        T("backend_basic_03_11_mysql_mvcc.md", ("正文", "数据库并发控制——锁机制", "锁的分类"), "DDL（Data Definition Language，数据库定义语言）：定义和修改表结构（如 CREATE TABLE、DROP TABLE）时，通常会使用 元数据锁。", "直接给出结构操作对应元数据锁。")),
    "h9ba03": A("B+ 树文档直接说明非叶子节点只存键和指针的 I/O 收益。",
        T("backend_basic_03_06_mysql_bplus_tree.md", ("正文", "非叶子节点不存储数据"), "B+ 树：非叶子节点只存储键值和指针，不存储实际数据，使得内部非叶子节点更小，单个磁盘块可容纳更多键值，减少树的高度和磁盘 I/O 次数，降低树的高度。", "直接回答结构设计目的。")),
    "h9ba04": A("缓存雪崩正文直接支持同过期时间问题和错峰方案。",
        T("backend_basic_04_02_redis_cache_problems.md", ("正文", "缓存雪崩"), "缓存中的大量数据设置了相同的过期时间，导致集中失效。", "直接支持上线同 TTL 风险。"),
        T("backend_basic_04_02_redis_cache_problems.md", ("正文", "缓存雪崩", "解决方案"), "缓存失效时间错峰：", "直接支持提前防护。")),
    "h9ba05": A("Redis 持久化开篇直接列出磁盘恢复和三种方式。",
        T("backend_basic_04_09_redis_persistence.md", ("正文",), "本文详细介绍Redis的持久化机制，目标是将内存中的数据持久化到磁盘，以保证数据的可靠性和在重启后的恢复能力。", "直接支持重启恢复目的。"),
        T("backend_basic_04_09_redis_persistence.md", ("正文",), "Redis 提供了三种持久化机制：RDB（Redis Database Snapshot） 、AOF（Append Only File）以及二者结合的混合持久化", "直接列举存法。")),
    "h9ba06": P("正文支持分布式哈希分片，但 16384 哈希槽只出现在目录标题，没有可验证的工作原理正文。",
        T("backend_basic_04_13_redis_cluster.md", ("正文",), "Redis Cluster 采用分布式哈希（hashing） 方式来管理数据，通过将数据分片并分配给不同的节点，确保 每个 Redis 节点只管理一部分数据。", "直接支持如何分配到节点。")),
    "h9ba07": P("文档只在目录列出 TIME_WAIT/2MSL，未解释等待原因；但明确列出端口复用设置，可作为部分直接证据。",
        T("backend_basic_01_02_network_tcp_udp.md", ("正文", "TCP 挥手的 TIME_WAIT 状态", "TIME_WAIT 导致的问题及其解决方案"), "（1）启用 TCP 重用（SO_REUSEADDR 和 SO_REUSEPORT）", "直接支持端口复用设置。")),
    "h9ba08": A("进程线程文档直接对比地址空间和崩溃影响。",
        T("backend_basic_02_01_os_process_thread.md", ("正文", "进程 VS 线程", "健全性"), "进程有独立的地址空间，进程崩溃后，在保护模式下不会对其他的进程产生影响；线程有自己的堆栈和局部变量，但没有单独的地址空间，一个线程死掉就等于整个进程死掉，所以多进程的程序要比多线程的程序健壮。", "直接回答内存隔离与崩溃差异。")),
    "h9al01": A("Transformer 正文直接说明架构无序与位置编码相加。",
        T(TRANSFORMER, ("正文", "位置编码"), "由于Transformer结构本身不保留输入序列的顺序信息（不像RNN那样逐步处理输入），需要通过位置编码显式的引入序列的顺序。位置编码是通过正弦和余弦函数生成的，加入到每个输入嵌入向量中", "直接支持为何需要及如何注入位置编码。"),
        T(TRANSFORMER, ("正文", "位置编码"), "x_{final}=x_{embedding}+PE_{position}", "公式直接支持位置编码与词嵌入相加。")),
    "h9al02": A("BERT MLM 段落直接给出 80/10/10 与训练推理差异原因。",
        T(GPT_BERT, ("正文", "BERT", "MLM", "掩码的机制"), "被掩盖的词中：\n- 80%：用 [MASK] 代替这个词。\n- 10%：随机替换为词汇表中的其他词。\n- 10%：保持词语不变。", "直接支持掩码比例。"),
        T(GPT_BERT, ("正文", "BERT", "MLM", "掩码的机制"), "这种机制是为了让模型不仅能学会对 [MASK] 标记进行预测，还能在下游任务中更好地处理实际出现的词，因为在推理阶段不会有 [MASK] 标记。", "直接支持设计原因。")),
    "h9al03": A("LoRA 核心与推理合并分别有直接公式和文字。",
        T(LORA, ("正文", "核心原理"), "LoRA 的基本思想是将它的更新部分 ΔW 分解为两个秩更小的矩阵 A 和 B 的乘积，从而减少需要更新的参数量，并且仅仅更新这两个矩阵。", "直接支持冻结大权重只训练 A/B。"),
        T(LORA, ("正文", "推理过程", "权重相加"), "y=(W+AB)x", "直接给出推理时与原权重合并方式。")),
    "h9al04": A("蒸馏正文直接解释软标签、温度和平滑分布。",
        T(COMPRESS, ("正文", "知识蒸馏", "准备教师模型的输出"), "软标签提供了类别间的细微关系，是学生模型的重要学习目标。", "直接支持为何学习教师概率。"),
        T(COMPRESS, ("正文", "知识蒸馏", "温度调节"), "当 $T>1$ 时，输出分布变得更加平滑，使得非最大类的概率变得较大，利于学生模型捕捉到类间关系。", "直接支持温度作用。")),
    "h9al05": A("训练概要开头直接列出三阶段与各自目标。",
        T("llm_algorithm_basic_02_大模型训练全过程_概要.md", ("正文采集",), "1. 预训练（Pretraining）：赋予语言知识基础\n2. 监督微调（Supervised Fine-Tuning, SFT）：塑造任务执行能力\n3. 对齐（Alignment, RLHF/DPO）：实现输出符合人类偏好与价值观", "直接概括从续写到指令和对齐的三个阶段。")),
    "h9ca01": A("路线索引同时给出零基础定义与对应入口文件。",
        T("llm_app_intro_03_learning_path_index.md", ("正文内容", "适用基础"), "零基础：适合没有任何编程基础，或者只会编程语言、没有系统学习计算机技术栈的人，目标是快速入门。", "直接确认问题中的用户类别。"),
        T("llm_app_intro_03_learning_path_index.md", ("正文内容", "路线入口"), "零基础：见 `llm_app_intro_05_zero_foundation_path.md`", "直接指向应先看的材料。")),
    "h9ca02": A("零基础路径开头明确学习顺序和田忌赛马理由。",
        T("llm_app_intro_05_zero_foundation_path.md", ("页面结构目录",), "如果你是零基础，则先学大模型应用开发技术栈，有机会再补后端开发相关的技术栈，最后再去看一些加分项（算法岗的一些通用知识）", "直接给出先 AI 应用后后端。"),
        T("llm_app_intro_05_zero_foundation_path.md", ("页面结构目录",), "大模型应用开发相关的内容远比后端开发的内容要少，学起来会更快。\n2. 类似于田忌赛马的策略", "直接支持排序理由。")),
    "h9ca03": A("岗位概览直接区分应用、算法和合并岗。",
        T("llm_app_intro_02_role_overview.md", ("岗位说明",), "应用开发岗需要将大模型算法岗产出的模型落地，也就是调用模型 API，开发AI应用。", "直接说明应用岗。"),
        T("llm_app_intro_02_role_overview.md", ("岗位说明",), "有的岗位也会把大模型应用开发岗和大模型算法岗合并到大模型岗位，这个我觉得是一个人干两个人的活儿，类似于AI界的“全栈”", "直接支持合并岗判断。")),
}


def _source_path(source: str) -> Path:
    matches = [
        path for path in (ROOT / "knowledge_base").rglob(source)
        if "assets" not in path.parts and "_pending" not in path.parts and "_archived" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one approved source for {source}, found {matches}")
    return matches[0]


def _index_fingerprint(ids: list[str], docs: list[str], metas: list[dict[str, Any]]) -> str:
    records = []
    for chunk_id, doc, meta in zip(ids, docs, metas):
        records.append((chunk_id, str(meta.get("source", "")), hashlib.sha256(doc.encode("utf-8")).hexdigest()))
    raw = json.dumps(sorted(records), ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _derive_heading_path(source_text: str, excerpt: str) -> list[str]:
    """Derive the actual Markdown heading stack at the reviewed span."""

    anchor = normalize_answer_span(excerpt.splitlines()[0])[:24]
    lines = source_text.splitlines()
    positions = [index for index, line in enumerate(lines) if anchor in normalize_answer_span(line)]
    if len(positions) != 1:
        # A shorter anchor handles excerpts whose first line is itself wrapped,
        # while still rejecting ambiguous review locations.
        anchor = normalize_answer_span(excerpt)[:12]
        positions = [index for index, line in enumerate(lines) if anchor in normalize_answer_span(line)]
    if not positions:
        raise RuntimeError(f"cannot derive unique heading path for excerpt anchor {anchor!r}; positions={positions}")

    def stack_at(position: int) -> list[str]:
        stack: dict[int, str] = {}
        for line in lines[: position + 1]:
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not match:
                continue
            level = len(match.group(1))
            stack = {key: value for key, value in stack.items() if key < level}
            stack[level] = match.group(2).strip()
        return [stack[level] for level in sorted(stack)]

    candidates = [stack_at(position) for position in positions]
    # Some captured pages repeat a paragraph verbatim.  If its heading stacks
    # differ only because the second copy has malformed repeated subheadings,
    # the first source occurrence remains the canonical reviewed location.
    return candidates[0] or [Path("unknown").stem]


def build_payload() -> dict[str, Any]:
    import chromadb

    base_items = json.loads((ROOT / "tests" / "rag_bench_paraphrase_set.json").read_text(encoding="utf-8"))["items"]
    query_ids = [item["id"] for item in base_items]
    if set(query_ids) != set(SPECS):
        raise RuntimeError(f"review spec coverage mismatch: missing={sorted(set(query_ids)-set(SPECS))}, extra={sorted(set(SPECS)-set(query_ids))}")

    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(COLLECTION)
    snapshot = collection.get(include=["documents", "metadatas"])
    ids: list[str] = snapshot["ids"]
    docs: list[str] = snapshot["documents"]
    metas: list[dict[str, Any]] = snapshot["metadatas"]
    by_source: dict[str, list[tuple[str, str]]] = {}
    for chunk_id, doc, meta in zip(ids, docs, metas):
        by_source.setdefault(str(meta.get("source", "")), []).append((chunk_id, doc))

    output_items = []
    for query_id in query_ids:
        spec = SPECS[query_id]
        targets = []
        for target in spec.targets:
            source_text = _source_path(target.source).read_text(encoding="utf-8")
            normalized_excerpt = normalize_answer_span(target.excerpt)
            if normalized_excerpt not in normalize_answer_span(source_text):
                raise RuntimeError(f"{query_id}: reviewed excerpt not present in approved source {target.source}: {target.excerpt[:80]!r}")
            matches = [
                (chunk_id, doc) for chunk_id, doc in by_source.get(target.source, [])
                if normalized_excerpt in normalize_answer_span(doc)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{query_id}: expected reviewed span in one current chunk of {target.source}, found {[m[0] for m in matches]}: {target.excerpt[:80]!r}")
            chunk_id, _ = matches[0]
            targets.append({
                "source": target.source,
                "heading_path": _derive_heading_path(source_text, target.excerpt),
                "chunk_id": chunk_id,
                "answer_span_hash": answer_span_hash(target.excerpt),
                "evidence_excerpt": target.excerpt,
                "relevance": target.relevance,
                "review_note": target.note,
            })
        output_items.append({
            "query_id": query_id,
            "review_outcome": spec.outcome,
            "review_note": spec.note,
            "relevant_targets": targets,
        })

    payload = {
        "schema_version": "rag-qrels-overlay-v1",
        "reviewer_id": "reviewer-A-independent",
        "base_set": "tests/rag_bench_paraphrase_set.json",
        "review_method": "Manual answer-span review against approved Markdown; ranked retrieval output was not used to select targets.",
        "index": {
            "collection": COLLECTION,
            "count": collection.count(),
            "fingerprint": _index_fingerprint(ids, docs, metas),
        },
        "items": output_items,
    }
    validate_qrels_overlay(payload, expected_query_ids=query_ids)
    return validate_qrels_against_collection(payload, collection)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Verify the existing artifact matches a clean rebuild")
    args = parser.parse_args()
    payload = build_payload()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"qrels artifact is stale: {args.output}")
        print(f"verified {len(payload['items'])} qrels at {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(payload['items'])} reviewed qrels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
