# -*- coding: utf-8 -*-
"""Tests for the "structure is not evidence" criterion on the evidence gate.

The defect this criterion exists for is `v2aneg-015`
(docs/rag_eval/colloquial/GATE_20260827.md §1.1): a `wrong_relation` negative
that the **main** gate accepts with dense distance 0.576 and reranker 0.9712.
Both chunks behind those numbers are navigational artefacts of the same
document -- a 167-character title block and a page-structure table of contents.
A table of contents repeats the query's vocabulary, which is exactly why its
vector distance is small, and it asserts nothing that could support an answer.

The criterion must cut only one way.  Three Dev80 cases (ca-015, ca-016,
llm-001) were adjudicated as correct answers whose evidence *is* a chunk full
of section headings -- the difference is that there each heading has prose
underneath it.  So the measurement is not "does this look like a contents
list" but "does this chunk have a body of its own", and the tests below pin
both sides of that against the real chunk text from the frozen 3348-chunk
index (fingerprint sha256:09f97cea...).
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_structural_evidence import (  # noqa: E402
    CALIBRATED_THRESHOLD,
    evidence_anchor,
    is_structural,
    structural_fraction,
    structural_threshold,
    substantive_mask,
)

# --- real chunks from the frozen index -------------------------------------
# backend_basic_03_08_mysql_leftmost_prefix_716d703a1126 -- the Top-1 that
# scored 0.9712 and supplied the whole-index-nearest distance of 0.576.
TITLE_BLOCK = (
    "# 8.MySQL数据库——索引潜规则（最左前缀原则）\n"
    "\n"
    "> 来源：https://example.com/redacted-source\n"
    "> 入库说明：从飞书“后端开发面试八股（基础）”子树采集，经候选预览确认后正式入库；"
    "已去除页面 UI、推荐内容、章节封面/标题卡、二维码广告、装饰图和分隔条。"
)

# backend_basic_03_08_mysql_leftmost_prefix_9914708f72f5 -- the runner-up at
# 0.9687.  Note it is *not* empty of prose: disqualifying it needs the bare
# headings to outweigh what little body it has, not a "no prose at all" rule.
TOC_BLOCK = (
    "## 正文内容\n"
    "\n"
    "### 图示单值索引和联合索引\n### 单值索引\n### 联合索引\n最左前缀原则\n示例分析\n"
    "### 1. 全值匹配查询时\n### 2. 匹配左边的列时\n### 3. 匹配列前缀（%）\n"
    "### 4. 匹配范围值\n### 5. 精确匹配某一列并范围匹配另外一列\n### 6. 排序\n### 总结\n"
    "大家好，这里是Good Note，关注 公主号：Goodnote，专栏文章私信限时Free。"
    "本文详细介绍MySQL索引的关键潜规则——最左前缀原则。\n"
    "### 图示单值索引和联合索引\n### 单值索引\n"
    "- 单值索引（唯一索引、主键索引、全文索引等） 是指在数据库表中创建的、仅涉及单个列的索引。"
    "也就是说，单值索引是基于表中的单一列（例如，单个字段）创建的索引结构。"
    "单值索引底层的 B+ 树如下所示：\n"
    "评论"
)

# patient_agent_interview_handbook_a56cde7bdc96 -- the adjudicated evidence for
# v2a-ca-016 ("患者 Agent 项目当前最值得强调的工程点有哪些？").  Heading-dense and
# answered *by* its heading structure, which is precisely the case a blanket
# contents-list penalty would destroy.
ANSWERED_BY_STRUCTURE = (
    "## 11. 这个项目当前最值得强调的工程点\n"
    "\n### 11.1 不是简单聊天机器人\n\n这个项目不是纯 prompt 问答，而是：\n"
    "- 带工具调用\n- 带记忆\n- 带多模态\n- 带规划\n- 带长期沉淀\n"
    "\n### 11.2 记忆架构是分层的\n\n短期和长期记忆的来源、用途、更新频率都不同，所以分开设计。\n"
    "\n### 11.3 Query 不是只读\n\n它既消费记忆，也回写短期记忆，并能触发长期记忆沉淀。\n"
    "\n### 11.4 混合检索是当前长期事件召回的核心\n"
    "\n不是纯关键词，也不是纯向量，而是两者结合，并把来源显式标记出来。\n"
    "\n### 11.5 工程上保留了可替换边界\n"
    "\n模型、向量检索、多模态、工具服务都不是写死在一个大函数里。"
)

# backend_basic_03_12_mysql_replication_df4a59e42b2b -- the *most* structural
# gold chunk across both guard sets.  It is the entire lower safety margin.
MOST_STRUCTURAL_GOLD = (
    "## 正文内容\n"
    "\n简介\n核心组件\n### 主从复制的原理\n作用\n### 主从复制的线程模型\n### 复制的方式\n"
    "### 基于语句的逻辑复制（Statement-Based Replication, SBR）\n"
    "### 基于行的物理复制（Row-Based Replication, RBR）\n"
    "### 混合复制（Mixed Replication, MIXED）\n### 设计复制机制\n### 主从复制执行流程\n### 总结\n"
    "大家好，这里是编程Cookbook。本文详细介绍 MySQL 的主从复制，从原理到配置再到同步过程。\n"
    "简介\nMySQL 主从复制（Replication）是一种数据分布式存储技术，通过将主库（Master）的数据和"
    "操作复制到一个或多个从库（Slave），实现数据的同步和备份。它常用于读写分离、数据容灾、"
    "数据分布等场景。\n核心组件\n### 1.\n主库（Master）：\n"
    "- 负责记录所有数据变更操作到 Binary Log 中。\n- 通过网络将 Binary Log 提供给从库。\n评论\n"
    "### 2.\n从库（Slave）：\n"
    "- 负责从主库获取 Binary Log，并通过中继日志（Relay Log）将其重放在本地，"
    "最终实现与主库的数据同步。\n"
    "### 3.\n二进制日志（Binary Log）：\n"
    "- 主库记录所有数据的逻辑操作，用于主从复制和增量备份。\n"
    "- 包含数据变更的具体操作（语句或行数据）。\n"
    "### 4.\n中继日志（Relay Log）：\n"
    "- 从库将主库发送的 Binary Log 存储为中继日志。\n"
    "- 从库 SQL 线程根据中继日志执行对应的操作。"
)


@pytest.fixture
def knob(monkeypatch):
    def configure(value=None):
        monkeypatch.delenv("RAG_STRUCTURAL_EVIDENCE_MAX", raising=False)
        if value is not None:
            monkeypatch.setenv("RAG_STRUCTURAL_EVIDENCE_MAX", value)
    return configure


# --- the defect ------------------------------------------------------------

def test_the_two_chunks_that_fooled_the_gate_are_structural():
    """Both v2aneg-015 evidence chunks must land above the threshold.

    Neither can be waved through: skipping only the title block leaves the
    contents block scoring 0.9687, still over the 0.85 reranker gate.
    """
    assert is_structural(TITLE_BLOCK, CALIBRATED_THRESHOLD)
    assert is_structural(TOC_BLOCK, CALIBRATED_THRESHOLD)


def test_a_title_block_is_entirely_structural():
    """A heading plus the ingest provenance banner carries no assertion at all."""
    assert structural_fraction(TITLE_BLOCK) == 1.0


# --- the other direction: this must not become a contents-list penalty ------

def test_a_chunk_whose_headings_have_prose_is_not_structural():
    """v2a-ca-016's adjudicated evidence: six headings, every one with a body.

    This is the case the criterion exists to spare.  An earlier candidate
    (strip the headings and re-score with the cross-encoder) collapsed this
    chunk from 0.9894 to 0.0083 -- the headings *are* the topic here -- which
    is why that approach was rejected in favour of measuring bare headings.
    """
    assert not is_structural(ANSWERED_BY_STRUCTURE, CALIBRATED_THRESHOLD)
    assert structural_fraction(ANSWERED_BY_STRUCTURE) < 0.10


def test_calibration_margins_are_pinned_on_both_sides():
    """τ=0.32 is the midpoint of a measured gap, so both edges need pinning.

    Upper edge: the most structural gold chunk in either guard set, 0.250.
    Lower edge: the v2aneg-015 contents block, 0.391.  Those two numbers are
    the entire justification for the threshold; a chunker change that moves
    either one invalidates it and should fail here rather than in production.
    """
    gold = structural_fraction(MOST_STRUCTURAL_GOLD)
    offender = structural_fraction(TOC_BLOCK)
    assert gold == pytest.approx(0.250, abs=0.005)
    assert offender == pytest.approx(0.391, abs=0.005)
    assert gold < CALIBRATED_THRESHOLD < offender
    assert CALIBRATED_THRESHOLD == pytest.approx((gold + offender) / 2, abs=0.005)


def test_an_all_structural_pool_falls_back_instead_of_refusing():
    """"What does this document cover" has to keep working.

    If every candidate is structural then the contents list is the best answer
    available, and refusing would be the blanket penalty this design rejects.
    ``None`` is the caller's signal to keep the original behaviour.
    """
    assert evidence_anchor([TITLE_BLOCK, TOC_BLOCK], CALIBRATED_THRESHOLD) is None


def test_the_anchor_is_the_highest_ranked_substantive_candidate():
    docs = [TITLE_BLOCK, TOC_BLOCK, ANSWERED_BY_STRUCTURE, MOST_STRUCTURAL_GOLD]
    assert evidence_anchor(docs, CALIBRATED_THRESHOLD) == 2


# --- default off -----------------------------------------------------------

def test_default_is_off(knob):
    knob()
    assert structural_threshold() is None
    assert is_structural(TITLE_BLOCK, structural_threshold()) is False
    assert evidence_anchor([TITLE_BLOCK, TOC_BLOCK], None) == 0
    assert substantive_mask([TITLE_BLOCK, TOC_BLOCK], None) == [True, True]


@pytest.mark.parametrize("value", ["", "   ", "abc", "0", "-0.3", "1.5"])
def test_unparseable_or_out_of_range_configuration_switches_off(knob, value):
    """Off is the safe failure: it restores the pre-change behaviour exactly.

    Refusing instead would turn a typo into a knowledge base that answers
    nothing, which is a worse failure than the one being fixed.
    """
    knob(value)
    assert structural_threshold() is None


def test_the_knob_reads_the_environment_per_call(knob):
    """This repo has shipped three module-level env freezes that silently ran
    the baseline arm; the criterion must not become a fourth."""
    knob("0.32")
    assert structural_threshold() == 0.32
    knob("0.50")
    assert structural_threshold() == 0.50


# --- classification hazards measured in the corpus -------------------------

def test_comments_inside_code_fences_are_not_headings():
    """1016 of the corpus's 5208 heading-shaped lines sit inside ``` fences.

    They are Python and shell comments.  Without fence tracking a code-heavy
    chunk reads as a wall of bare headings and gets disqualified.
    """
    fenced_code = (
        "```python\n"
        "# 执行两种检索\n"
        "# 合并结果\n"
        "docs = retrieve(query)\n"
        "```"
    )
    assert structural_fraction(fenced_code) == 0.0
    assert not is_structural(fenced_code, CALIBRATED_THRESHOLD)


def test_only_ingest_provenance_blockquotes_count_as_structure():
    """The corpus has 621 blockquote lines and they are not one thing.

    ``> 来源：``/``> 入库说明：`` are banners this repo's own ingester writes;
    ``> <strong>提示</strong>`` and ``> [完整代码](...)`` are content.  Counting
    every blockquote as boilerplate would disqualify chunks that quote.
    """
    provenance = "# 标题\n\n> 来源：https://example.com/wiki/x\n> 入库说明：从飞书采集。"
    quoted_content = (
        "# 标题\n\n> <strong>提示</strong>：写入前必须先获取行锁，否则并发更新会丢失。\n"
        "> 该限制在 8.0 之后仍然成立。"
    )
    assert structural_fraction(provenance) == 1.0
    assert structural_fraction(quoted_content) < CALIBRATED_THRESHOLD


def test_a_heading_separated_from_its_body_by_provenance_still_has_a_body():
    """The banner sits between a heading and its prose in every ingested page.

    Counting the heading as bare because the next line is a banner would make
    the first section of every wiki import look like a contents entry.  Only
    the banner's own characters may count, which is what the two variants
    below isolate: identical prose, one with a heading whose body is a line
    further away.
    """
    body = "主库把变更写入 Binary Log，从库拉取后经中继日志重放，最终与主库保持一致。"
    banner = "> 来源：https://example.com/x"
    with_banner = f"## 主从复制\n{banner}\n{body}"
    without_banner = f"## 主从复制\n{body}"
    # the heading contributes nothing either way -- only the banner does
    assert structural_fraction(without_banner) == 0.0
    assert structural_fraction(with_banner) == pytest.approx(
        len(banner) / (len("## 主从复制") + len(banner) + len(body)), abs=1e-6)


def test_empty_and_whitespace_chunks_are_not_structural():
    """Zero-length evidence is somebody else's problem (the reranker's)."""
    for text in ("", "   ", "\n\n", None):
        assert structural_fraction(text or "") == 0.0


# --- wiring ----------------------------------------------------------------

def test_the_gate_anchors_on_the_substantive_candidate_not_the_top_one():
    """Source assertion: the criterion has to move the gate's *judged chunk*.

    Scoring the answer's Top-1 while gating on another chunk's number is the
    identity mix-up a previous round removed; re-introducing it here would be
    a regression that no metric would report.
    """
    source = (Path(__file__).resolve().parents[1] / "rag_gate.py").read_text(
        encoding="utf-8")
    assert "_anchor = evidence_anchor(docs, _struct_tau)" in source
    assert "float(final_scores[_anchor])" in source
    # the English second path must read the language of the same chunk whose
    # score it is testing
    assert "_m0 = (metas[_anchor]" in source
    # and the decision has to be auditable per query
    assert '"structural_evidence_max": _struct_tau,' in source
    assert '"gate_anchor_rank": (_anchor + 1) if _anchor is not None else None,' in source


def test_the_dense_side_is_guarded_too():
    """0.576 -- the whole-index nearest distance -- came from the title block.

    Anchoring only the reranker would leave ``vector_in_kb`` decided by a chunk
    the gate has already ruled inadmissible.
    """
    source = (Path(__file__).resolve().parents[1] / "rag_gate.py").read_text(
        encoding="utf-8")
    assert "_gate_solid = substantive_mask(_gate_docs, _struct_tau)" in source
    assert "if not any(_gate_solid):" in source      # all-structural fallback
    assert "best = min((dist for _d, dist in _gate_evidence), default=99.0)" in source
