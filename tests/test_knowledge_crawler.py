# -*- coding: utf-8 -*-
"""test_knowledge_crawler.py — P4：半自动知识采集纯逻辑离线单测。

只测不依赖网络/LLM 的纯函数：slug、frontmatter 往返、评分解析、提升路径。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import knowledge_crawler as kc


def test_slugify_basic():
    assert kc.slugify("https://juejin.cn/post/abc") == "juejin_cn_post_abc"


def test_slugify_keeps_chinese():
    s = kc.slugify("大模型 RAG 实战")
    assert "大模型" in s and " " not in s


def test_slugify_empty_fallback():
    assert kc.slugify("") == "untitled"


def test_parse_score_valid_json():
    txt = (
        "审核意见见下：\n```json\n"
        '{"grade":"A","relevance":9,"density":8,"recency_ok":true,'
        '"reason":"干货满满","suggested_subdir":"learning_resources",'
        '"suggested_title":"RAG 全链路实战"}\n```'
    )
    s = kc.parse_score(txt)
    assert s["grade"] == "A"
    assert s["relevance"] == 9
    assert s["density"] == 8
    assert s["suggested_subdir"] == "learning_resources"
    assert s["suggested_title"] == "RAG 全链路实战"


def test_parse_score_fallback_on_garbage():
    s = kc.parse_score("模型没按格式输出任何 JSON")
    assert s["grade"] == "C"  # 保守
    assert s["suggested_subdir"] == "learning_resources"


def test_parse_score_invalid_subdir_normalized():
    txt = '```json\n{"grade":"B","suggested_subdir":"random_dir"}\n```'
    s = kc.parse_score(txt)
    assert s["suggested_subdir"] == "learning_resources"


def test_parse_score_invalid_grade_normalized():
    txt = '```json\n{"grade":"超棒"}\n```'
    s = kc.parse_score(txt)
    assert s["grade"] == "C"


def test_frontmatter_roundtrip():
    meta = {
        "title": "RAG 实战", "source_url": "http://x", "quality": "A",
        "target_role": "大模型应用工程师", "review_status": "pending",
        "tags": ["大模型", "RAG"],
    }
    fm = kc.build_frontmatter(meta)
    back = kc.parse_frontmatter(fm + "\n正文内容")
    assert back["title"] == "RAG 实战"
    assert back["quality"] == "A"
    assert back["review_status"] == "pending"


def test_subdir_source_type_mapping():
    assert kc.SUBDIR_SOURCE_TYPE["career_paths"] == "career_knowledge"
    assert kc.SUBDIR_SOURCE_TYPE["experience_posts"] == "experience"
    assert kc.SUBDIR_SOURCE_TYPE["learning_resources"] == "resource"


def test_promote_rejects_bad_subdir(tmp_path):
    f = tmp_path / "x.md"
    f.write_text("---\ntitle: x\n---\n正文", encoding="utf-8")
    out = kc.cmd_promote(str(f), "not_a_subdir")
    assert out["status"] == "error"


def test_promote_moves_and_fixes_metadata(tmp_path):
    """提升：文件移到目标子目录，review_status→approved，source_type→对齐。"""
    pend = tmp_path / "pending.md"
    pend.write_text(
        '---\ntitle: "测试文章"\nreview_status: "pending"\nsource_type: "resource"\n---\n正文',
        encoding="utf-8",
    )
    # 把 KB 目录临时指向 tmp_path，避免污染真实库
    import knowledge_crawler
    orig_kb = knowledge_crawler.KB_DIR
    knowledge_crawler.KB_DIR = str(tmp_path / "kb")
    try:
        out = kc.cmd_promote(str(pend), "career_paths")
        assert out["status"] == "ok"
        assert out["source_type"] == "career_knowledge"
        moved = tmp_path / "kb" / "career_paths" / "pending.md"
        assert moved.exists()
        text = moved.read_text(encoding="utf-8")
        assert 'review_status: "approved"' in text
        assert 'source_type: "career_knowledge"' in text
        assert not pend.exists()  # 原文件已移走
    finally:
        knowledge_crawler.KB_DIR = orig_kb


def test_from_text_missing_file():
    out = kc.cmd_from_text("http://x", "不存在的文件.md")
    assert out["status"] == "error"


def test_score_and_save_rejects_too_short(tmp_path, monkeypatch):
    """正文过短直接 reject，不落盘、不调 LLM。"""
    monkeypatch.setattr(kc, "PENDING_WEB_DIR", str(tmp_path))
    out = kc._score_and_save("太短", "http://x", "test")
    assert out["status"] == "rejected"
    assert out["saved"] is None


def test_content_stats():
    text = "标题行\n正常正文一句话\nhttps://x.com 链接行\n![img](a.png)\n再一句正文"
    s = kc.content_stats(text)
    assert s["lines"] == 5
    assert s["images"] == 1
    assert s["link_pct"] > 0


def test_content_preview_head_tail():
    text = "\n".join(f"行{i}" for i in range(30))
    p = kc.content_preview(text, head=5, tail=3)
    assert p["head"].startswith("行0")
    assert "行29" in p["tail"]
    assert p["truncated_middle"] == 30 - 5 - 3


def test_review_missing_file():
    assert kc.cmd_review("不存在.md")["status"] == "error"


def test_review_surfaces_source_and_path(tmp_path):
    """review 必须给出 source_url + 本地绝对路径（审核两要素）。"""
    f = tmp_path / "c.md"
    f.write_text(
        '---\ntitle: "测试"\nsource_url: "https://example.com/a"\n'
        'quality: "A"\nsource_type: "resource"\n---\n正文内容若干字用于统计。',
        encoding="utf-8",
    )
    out = kc.cmd_review(str(f))
    assert out["source_url"] == "https://example.com/a"   # ① 来源
    assert out["saved_abs"].endswith("c.md")              # ② 本地路径
    assert "stats" in out and "preview" in out


# ---- GitHub 整仓采集：URL 解析 / 路由 / notebook 提取（纯逻辑）----

def test_parse_github_repo_variants():
    assert kc.parse_github_repo("https://github.com/a/b") == ("a", "b")
    assert kc.parse_github_repo("https://github.com/a/b/tree/main/docs") == ("a", "b")
    assert kc.parse_github_repo("https://raw.githubusercontent.com/a/b/main/README.md") == ("a", "b")
    assert kc.parse_github_repo("https://github.com/a/b.git") == ("a", "b")
    assert kc.parse_github_repo("https://zhuanlan.zhihu.com/p/1") is None


def test_is_github_repo_url_routing():
    # 仓库根 / README → 走整仓采集
    assert kc.is_github_repo_url("https://github.com/a/b") is True
    assert kc.is_github_repo_url("https://raw.githubusercontent.com/a/b/main/README.md") is True
    # 具体单文件 → 单文件抓取
    assert kc.is_github_repo_url("https://raw.githubusercontent.com/a/b/main/docs/c.md") is False
    assert kc.is_github_repo_url("https://github.com/a/b/blob/main/docs/c.md") is False
    # 非 GitHub
    assert kc.is_github_repo_url("https://example.com/x") is False


def test_ipynb_to_text_extracts_md_and_code():
    import json as _json
    nb = _json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# 标题\n", "说明文字"]},
        {"cell_type": "code", "source": ["print('hi')"]},
        {"cell_type": "code", "source": [], "outputs": [{"text": "应被忽略"}]},
    ]})
    out = kc._ipynb_to_text(nb)
    assert "# 标题" in out
    assert "说明文字" in out
    assert "print('hi')" in out
    assert "```python" in out
    assert "应被忽略" not in out  # 输出被丢弃


def test_ipynb_to_text_bad_json():
    assert kc._ipynb_to_text("不是json") == ""


# ---- 图片相对路径 → GitHub raw 绝对 URL ----

def test_rewrite_image_links_relative_to_absolute():
    out = kc.rewrite_image_links(
        "![arch](../figures/structure.jpg)", "o", "r", "main", "docs/C3/3.md")
    assert "https://raw.githubusercontent.com/o/r/main/docs/figures/structure.jpg" in out
    assert "../figures" not in out


def test_rewrite_image_links_keeps_http():
    src = "![x](https://cdn.com/a.png)"
    assert kc.rewrite_image_links(src, "o", "r", "main", "docs/a.md") == src


def test_rewrite_image_links_html_img():
    out = kc.rewrite_image_links('<img src="img/x.png" width="50">', "o", "r", "main", "docs/a.md")
    assert "https://raw.githubusercontent.com/o/r/main/docs/img/x.png" in out


def test_rewrite_image_links_same_dir():
    out = kc.rewrite_image_links("![](pic.png)", "o", "r", "main", "guide.md")
    assert "https://raw.githubusercontent.com/o/r/main/pic.png" in out


# ---- 采集内容密钥脱敏 ----

def test_redact_secrets_openai_key():
    # 用明显伪造的 key（仅匹配 sk- 模式，非真实凭证），避免把真 key 写进仓库
    fake = "sk-" + "FAKEFAKEFAKEFAKEFAKEFAKEFAKE0123456789"
    out, n = kc.redact_secrets(f"openai_api_key='{fake}'")
    assert "FAKEFAKE" not in out
    assert "[REDACTED_SECRET]" in out
    assert n == 1


def test_redact_secrets_keeps_normal_text():
    out, n = kc.redact_secrets("这是正常正文，讲 RAG 和 Agent，没有密钥。")
    assert n == 0
    assert "RAG" in out
