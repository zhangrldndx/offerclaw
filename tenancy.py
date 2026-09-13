# -*- coding: utf-8 -*-
"""租户接缝(tenancy seam)——仅服务于「共享实例」这一可选部署形态。

项目主模式是 OpenClaw 式**本地优先分发**:用户自装自用,数据全部
本地,开发者不可见,一人一实例一画像——那种场景**不需要**本模块,
且本模块对其零行为变化(默认租户 = 原 collection 名,有测试钉死)。

本接缝只为一个真实的中间场景预留:实验室/小团队在内网架一台共享
实例给几个人用。届时需要的最小机制提前立好,升级时不动业务代码:

  1. 请求级租户身份:``X-OfferClaw-User`` 请求头 > ``OFFERCLAW_TENANT``
     环境变量 > 默认 ``local``。经 contextvar 贯穿整条请求链路,与既有
     request_id 同款机制。
  2. 知识库命名空间:``collection_name_for(base)`` —— 默认租户返回原名
     (现网数据零迁移),其他租户追加 ``__<tenant>`` 后缀,即
     per-tenant collection 的最小实现。
  3. 用量归属:``usage_record_fields()`` 给 LLM 用量账本预留 user 维度,
     未来按租户核算成本/配额只需在 meter 落盘处合并该字段。

安全约束:租户名会进入 collection 名与日志,必须过白名单正则;
非法值一律回落默认租户(fail-safe,不拒绝请求)。
"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass

DEFAULT_TENANT = "local"
TENANT_HEADER = "X-OfferClaw-User"
TENANT_ENV = "OFFERCLAW_TENANT"

# 进入 collection 名/日志的标识必须收紧:小写字母数字开头,<=32 字符。
_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class TenantContext:
    user_id: str

    @property
    def is_default(self) -> bool:
        return self.user_id == DEFAULT_TENANT


_current: ContextVar[TenantContext] = ContextVar(
    "offerclaw_tenant", default=TenantContext(DEFAULT_TENANT)
)


def normalize_tenant(raw: object) -> str | None:
    """合法化租户名;不合法返回 None(调用方回落默认)。"""
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    if _TENANT_RE.match(candidate):
        return candidate
    return None


def resolve_tenant(header_value: str | None = None) -> TenantContext:
    """租户解析优先级:请求头 > 环境变量 > 默认 local。"""
    for source in (header_value, os.getenv(TENANT_ENV)):
        user_id = normalize_tenant(source)
        if user_id:
            return TenantContext(user_id)
    return TenantContext(DEFAULT_TENANT)


def current_tenant() -> TenantContext:
    return _current.get()


def set_current_tenant(tenant: TenantContext):
    return _current.set(tenant)


async def tenant_middleware(request, call_next):
    """FastAPI 中间件:每个请求解析并绑定租户,响应头回显便于排查。"""
    tenant = resolve_tenant(request.headers.get(TENANT_HEADER))
    token = set_current_tenant(tenant)
    try:
        response = await call_next(request)
    finally:
        _current.reset(token)
    response.headers[TENANT_HEADER] = tenant.user_id
    return response


def collection_name_for(base: str, tenant: TenantContext | None = None) -> str:
    """知识库集合命名空间:默认租户 = 原名(零迁移);其余加后缀。"""
    t = tenant or current_tenant()
    if t.is_default:
        return base
    return f"{base}__{t.user_id}"


def usage_record_fields() -> dict:
    """LLM 用量账本的租户维度(未来在 meter 落盘处 merge 即可)。"""
    return {"user": current_tenant().user_id}
