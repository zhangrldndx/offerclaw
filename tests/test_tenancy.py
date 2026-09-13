# -*- coding: utf-8 -*-
"""多用户预留层(tenancy seam)契约测试。

单用户零行为变化是硬约束:默认租户下 collection 名必须与现网完全一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import tenancy
from tenancy import (
    DEFAULT_TENANT,
    TenantContext,
    collection_name_for,
    normalize_tenant,
    resolve_tenant,
    usage_record_fields,
)


def test_default_tenant_is_zero_change():
    t = resolve_tenant(None)
    assert t.user_id == DEFAULT_TENANT and t.is_default
    # 现网数据零迁移:默认租户不改 collection 名
    assert collection_name_for("offerclaw_kb", t) == "offerclaw_kb"


def test_header_beats_env_beats_default(monkeypatch):
    monkeypatch.setenv(tenancy.TENANT_ENV, "envuser")
    assert resolve_tenant("alice").user_id == "alice"
    assert resolve_tenant(None).user_id == "envuser"
    monkeypatch.delenv(tenancy.TENANT_ENV)
    assert resolve_tenant(None).user_id == DEFAULT_TENANT


def test_illegal_tenant_falls_back_safely():
    # 进 collection 名/日志的标识必须过白名单;非法值回落默认,不拒绝请求
    for bad in ("../etc", "A B", "中文", "-lead", "x" * 40, "", None, 123):
        assert normalize_tenant(bad) is None
    assert resolve_tenant("../etc").user_id == DEFAULT_TENANT


def test_named_tenant_gets_namespaced_collection():
    t = TenantContext("alice")
    assert collection_name_for("offerclaw_kb", t) == "offerclaw_kb__alice"


def test_usage_record_fields_carry_tenant():
    token = tenancy.set_current_tenant(TenantContext("bob"))
    try:
        assert usage_record_fields() == {"user": "bob"}
    finally:
        tenancy._current.reset(token)


def test_api_stats_endpoint_and_tenant_echo():
    from rag_api import app

    client = TestClient(app)
    resp = client.get("/api/stats", headers={"X-OfferClaw-User": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant"] == "alice"
    for key in ("api", "llm_usage", "knowledge_base", "career_loop", "disclosure"):
        assert key in body
    # 响应头回显租户,便于多用户阶段排查
    assert resp.headers.get("X-OfferClaw-User") == "alice"
    # 不带头 → 默认租户
    assert client.get("/api/stats").json()["tenant"] == DEFAULT_TENANT
