# -*- coding: utf-8 -*-
"""test_plan_gen_rag.py — P1：plan_gen 接入 RAG 的纯离线单元测试。

只测不依赖 LLM / embedding API 的纯函数：
- 方向抽取
- 缺口拆分为多 query
- 片段清理（去 frontmatter / 目录）
- 资源块格式化
- 参考资源附录追加
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import plan_gen


def test_extract_direction_only_numbered_items():
    profile = (
        "## 2. 求职方向与偏好\n"
        "- 目标方向（按优先级排序）：\n"
        "  1. 数据工程\n"
        "  2. Python 后端\n"
        "  3. 平台工程\n"
        "- 目标岗位类型：校招\n"
        "- 期望薪资区间：实习无薪资要求\n"
    )
    d = plan_gen._extract_direction(profile)
    assert "数据工程" in d
    assert "Python 后端" in d
    # 不应混入薪资/岗位类型噪音
    assert "薪资" not in d
    assert "实习" not in d


def test_split_gap_queries_per_item():
    gaps = (
        "技能缺口：\n"
        "- 缺少 RAG / 向量检索的系统知识\n"
        "- 缺少 Agent / 工具调用的项目经历\n"
        "经历缺口：\n"
        "- 缺少端到端大模型应用开发项目\n"
    )
    qs = plan_gen._split_gap_queries(gaps, direction="Agent 应用工程")
    # 三条实质缺口 → 三条 query（小标题"技能缺口/经历缺口"不单独成条）
    assert len(qs) == 3
    assert all(q.startswith("Agent 应用工程 ") for q in qs)
    assert any("RAG" in q for q in qs)
    assert any("工具调用" in q for q in qs)


def test_split_gap_queries_drops_bracket_headers():
    """前端缺口卡用【硬门槛缺口】这种方括号小标题；它们是结构标签，不能当检索 query。"""
    ui_gaps = (
        "【硬门槛缺口】\n"
        "- 缺少大规模分布式训练经验\n"
        "- 缺少线上 LLM 服务部署经历\n"
        "【技能缺口】\n"
        "- RAG 工程化深度不足\n"
        "【建议】\n"
        "- 补充一个端到端 RAG 项目\n"
    )
    qs = plan_gen._split_gap_queries(ui_gaps, direction="大模型应用工程师")
    # 4 条实质内容；【硬门槛缺口】【技能缺口】【建议】等纯标题不得出现
    assert len(qs) == 4
    joined = "\n".join(qs)
    assert "【硬门槛缺口】" not in joined
    assert "【技能缺口】" not in joined
    assert "【建议】" not in joined
    assert any("分布式训练" in q for q in qs)
    assert any("RAG 工程化" in q for q in qs)


def test_split_gap_queries_keeps_descriptive_sentence_ending_with_quekou():
    """以「…能力缺口」结尾的描述性长句是实质内容，不能被小标题过滤误杀。"""
    qs = plan_gen._split_gap_queries("- 存在明显的 RAG 工程化端到端能力缺口")
    assert len(qs) == 1
    assert "能力缺口" in qs[0]


def test_clean_snippet_strips_frontmatter_and_blockquote():
    doc = (
        '---\n'
        'title: "7. 大模型 Harness Engineering"\n'
        'source_url: https://x\n'
        'crawl_date: 2026-06-02\n'
        '---\n'
        '# 7. 大模型 Harness Engineering\n'
        '> 来源：https://x\n'
        '## 正文采集\n'
        'Harness 是围绕模型建立的执行闭环系统。\n'
    )
    cleaned = plan_gen._clean_snippet(doc)
    assert not cleaned.startswith("---")
    assert "source_url" not in cleaned
    assert "来源：" not in cleaned
    assert "Harness 是围绕模型建立的执行闭环系统" in cleaned


def test_clean_snippet_empty_for_pure_frontmatter():
    """纯 frontmatter + 标题 + 来源（无正文）清理后应为空，便于上层跳过。"""
    doc = (
        '---\ntitle: "x"\nsource_url: https://x\n---\n'
        '# x\n> 来源：https://x\n'
    )
    assert plan_gen._clean_snippet(doc).strip() == ""


def test_format_resources_block_empty():
    assert plan_gen.format_resources_block([]) == ""


def test_format_resources_block_lists_sources():
    resources = [
        {"source": "a.md", "title": "RAG 全链路", "snippet": "检索增强生成"},
        {"source": "b.md", "title": "", "snippet": "工具调用"},
    ]
    block = plan_gen.format_resources_block(resources)
    assert "RAG 全链路" in block
    assert "a.md" in block
    assert "b.md" in block


def test_append_resources_appendix_guarantees_citation():
    plan = "## 4 周计划\nWeek 1 ...\n"
    resources = [
        {"source": "llm_app_interview_02_rag_basics.md", "title": "RAG 基础", "snippet": "向量检索"},
    ]
    out = plan_gen.append_resources_appendix(plan, resources)
    assert "参考的知识库资源" in out
    assert "llm_app_interview_02_rag_basics.md" in out
    # 原计划内容保留
    assert "Week 1" in out


def test_append_resources_appendix_noop_when_empty():
    plan = "## 4 周计划\n"
    assert plan_gen.append_resources_appendix(plan, []) == plan


def test_retrieve_returns_empty_without_key(monkeypatch):
    """无 embedding key 时检索必须静默返回 []，保证 plan_gen 离线可用。"""
    monkeypatch.setattr(plan_gen, "RESOURCE_SOURCE_TYPES", ["career_knowledge", "resource"])
    import rag_tools
    monkeypatch.setattr(rag_tools, "has_embedding_api_key", lambda: False)
    out = plan_gen.retrieve_learning_resources("缺少 RAG 经验", direction="Agent")
    assert out == []


# ---- P2.5：混合相关性门槛的纯函数测试（统一在 rag_gate）----

import rag_gate as rg


def test_query_keywords_extracts_english_tokens():
    kw = rg.query_keywords("什么是react")
    assert "react" in kw
    # 中文疑问词不参与词面救回
    assert "什么" not in kw


def test_query_keywords_drops_stopwords_and_short():
    kw = rg.query_keywords("what is is a")
    assert "what" not in kw  # 停用词
    assert "is" not in kw    # 太短 + 停用词


def test_lexical_hit_true_when_keyword_in_chunk():
    chunks = ["ReAct 是一种结合推理与行动的框架"]
    assert rg.lexical_hit(["react"], chunks) is True  # 大小写不敏感


def test_lexical_hit_false_when_absent():
    chunks = ["这是关于简历项目栏的内容"]
    assert rg.lexical_hit(["vue3"], chunks) is False


def test_lexical_hit_false_when_no_keywords():
    assert rg.lexical_hit([], ["任意内容"]) is False


# ---- 项目先验 + 计划微信摘要（纯逻辑）----

def test_load_project_context_reads_only_synthetic_fixture(tmp_path, monkeypatch):
    project_dir = tmp_path / "project_context"
    project_dir.mkdir()
    (project_dir / "sample.md").write_text(
        "# SampleProject\n\n这是公开测试使用的合成项目，只读。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(plan_gen, "PROJECT_CONTEXT_DIR", str(project_dir))
    pc = plan_gen.load_project_context()
    assert "SampleProject" in pc
    assert "只读" in pc


def test_build_messages_injects_project_directive():
    msgs = plan_gen.build_messages(
        "画像", "prompt", "log", "policy", "rules", "缺Agent项目经历",
        resources_block="", project_context="LocalFlow 现状...")
    user = msgs[1]["content"]
    sys_c = msgs[0]["content"]
    assert "已有项目" in sys_c or "已有项目" in user
    assert "建议你" in sys_c  # 只读措辞约束
    # 实战编排硬性要求出现在 user 任务里
    assert "在已有项目上推进" in user


def test_build_messages_no_project_directive_when_empty():
    msgs = plan_gen.build_messages(
        "画像", "prompt", "log", "policy", "rules", "gaps",
        resources_block="", project_context="")
    assert "实战编排硬性要求" not in msgs[1]["content"]
