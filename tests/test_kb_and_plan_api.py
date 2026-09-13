# -*- coding: utf-8 -*-
"""新端点（学习计划读取/保存 + 知识库维护）的无副作用回归测试。

只覆盖读取端点与参数校验路径——不触发真实入库/落盘/网络，
happy-path（抓取/上传→入库→查询）已在隔离环境单独验证。
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_api  # noqa: E402

client = TestClient(rag_api.app)


# ---------- 学习计划 ----------

def test_plan_current_shape():
    r = client.get("/api/plan/current")
    assert r.status_code == 200
    j = r.json()
    assert "has_plan" in j and isinstance(j["has_plan"], bool)
    if j["has_plan"]:
        assert "content" in j and "edited_by_user" in j


def test_plan_save_rejects_empty():
    r = client.post("/api/plan/save", json={"content": "   "})
    assert r.status_code == 400


# ---------- 知识库维护：读取端点 ----------

def test_kb_status_shape():
    r = client.get("/api/kb/status")
    assert r.status_code == 200
    j = r.json()
    assert "chunks" in j and isinstance(j["chunks"], int)
    assert "sources" in j and isinstance(j["sources"], dict)


def test_kb_pending_shape():
    r = client.get("/api/kb/pending")
    assert r.status_code == 200
    j = r.json()
    assert "items" in j and isinstance(j["items"], list)


# ---------- 知识库维护：参数校验（无副作用） ----------

def test_kb_add_url_rejects_empty():
    r = client.post("/api/kb/add_url", json={"url": ""})
    assert r.status_code == 400


def test_kb_add_file_rejects_bad_ext():
    r = client.post("/api/kb/add_file", json={"name": "x.pdf", "text": "一些内容"})
    assert r.status_code == 400


def test_kb_add_file_rejects_empty():
    r = client.post("/api/kb/add_file", json={"name": "x.md", "text": ""})
    assert r.status_code == 400


def test_kb_promote_rejects_bad_subdir():
    r = client.post("/api/kb/promote", json={"pending_file": "x.md", "to_subdir": "不存在"})
    assert r.status_code == 400


def test_kb_promote_rejects_missing_fields():
    r = client.post("/api/kb/promote", json={"pending_file": "", "to_subdir": ""})
    assert r.status_code == 400
