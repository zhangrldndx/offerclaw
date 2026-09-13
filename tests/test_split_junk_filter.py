# -*- coding: utf-8 -*-
"""入库切块的垃圾过滤回归测试(2026-07-04 检索质量修复)。

背景:裁判团评测揪出 be03 的检索结果全是垃圾——一条是文章目录页、一条是文件头
YAML 元数据。全库扫描发现 11.5% 的块属于这两类(标题词密度高,检索时反而压过
真内容)。修复:split_markdown_document ①剥离文件头 YAML;②跳过净正文<80字的
目录型块。本文件钉死这两条规则,防止回潮。
"""

from rag_tools import split_markdown_document

FRONTMATTER_DOC = """---
title: "8.MySQL数据库——索引潜规则（最左前缀原则）"
source_url: https://example.com/wiki/xxx
crawl_date: 2026-06-02
tags: ["后端基础"]
---

# 8.MySQL数据库——索引潜规则

## 正文内容

最左前缀原则是指联合索引会优先按照最左边的列进行排序和匹配。当查询条件从索引的
最左列开始、且连续命中时,联合索引才会生效;跳过左列直接查右列则无法使用该索引。
这一规则决定了联合索引列顺序的设计:把区分度最高、最常用作等值查询的列放在最左。
"""

TOC_DOC = """# 某篇文章

## 页面结构目录

- 图示单值索引和联合索引
- 单值索引
- 联合索引
- 全值匹配查询时
- 匹配左边的列时
- 匹配列前缀
- 匹配范围值
- 排序
- 总结

## 正文内容

单值索引是指在数据库表中创建的、仅涉及单个列的索引结构,例如基于单个字段创建的
主键索引或唯一索引。联合索引则是基于多个列共同创建的索引,其底层 B+ 树按照列的
声明顺序依次排序,因此存在最左前缀匹配的约束,查询时必须从最左列开始连续命中。
"""


def test_frontmatter_is_stripped():
    chunks = split_markdown_document(FRONTMATTER_DOC)
    assert chunks, "正文应该保留"
    for c in chunks:
        assert not c["text"].strip().startswith("---"), "YAML 元数据块不得入库"
        assert "source_url" not in c["text"], "YAML 字段不得混入正文块"
    # 真正的正文还在
    assert any("最左前缀原则是指" in c["text"] for c in chunks)


def test_toc_only_section_is_dropped_content_kept():
    chunks = split_markdown_document(TOC_DOC)
    titles = [c["metadata"]["title"] for c in chunks]
    assert "页面结构目录" not in titles, "纯目录块不得入库"
    assert any("单值索引是指" in c["text"] for c in chunks), "正文段必须保留"


def test_numbered_toc_lines_are_filtered():
    """编号目录行("- 1.MySQL数据库——三范式")的英文句点不能算真内容标点。
    be03 复盘:overview 文件的编号目录第一版过滤没拦住,又挤掉了正文。"""
    doc = """# 总览

## 页面结构目录

- 1.MySQL数据库——数据库三范式
- 2.MySQL数据库——存储引擎
- 3.MySQL数据库——常见的几种锁分类
- 4.MySQL数据库——索引介绍
- 5.MySQL数据库——事务介绍

## 正文内容

掌握 MySQL 是后端开发者构建高效、可靠数据服务的基石。其面试精髓在于考察对数据库
核心机制的理解与实践,包括范式设计、索引原理、事务隔离与锁机制等核心知识点。
"""
    chunks = split_markdown_document(doc)
    titles = [c["metadata"]["title"] for c in chunks]
    assert "页面结构目录" not in titles, "编号目录块不得入库"
    assert any("面试精髓" in c["text"] for c in chunks)


def test_normal_bullet_list_content_not_harmed():
    """真列表内容(带句读/较长)不能被目录过滤误杀。"""
    doc = """# 文档

## Redis 快的原因

- 纯内存操作:所有数据存储在内存中,访问速度远高于磁盘。
- 高效的 I/O 多路复用:采用 epoll 等技术,单线程高效管理大量连接。
- 避免多线程开销:没有线程切换与锁竞争,天然无数据竞争问题。
"""
    chunks = split_markdown_document(doc)
    assert any("纯内存操作" in c["text"] for c in chunks), "带句读的真列表内容必须保留"
