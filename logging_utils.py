# -*- coding: utf-8 -*-
"""
OfferClaw · 结构化日志 + 请求追踪中间件

每个请求自动分配 request_id，所有日志带 request_id 和耗时。
覆盖常见 AI 应用 JD 中的系统原型、接口开发和部署职责。
"""
import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": _request_id_ctx.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "offerclaw") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = JsonFormatter()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(LOG_DIR, "api.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    _request_id_ctx.set(rid)
    return rid


def current_request_id() -> str:
    return _request_id_ctx.get()


async def request_logging_middleware(request, call_next):
    """FastAPI middleware：分配 request_id + 记录耗时。"""
    rid = new_request_id()
    logger = get_logger("offerclaw.api")
    t0 = time.time()
    logger.info(f"REQ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"ERR {request.method} {request.url.path}")
        raise
    dt = (time.time() - t0) * 1000
    logger.info(f"RES {request.method} {request.url.path} {response.status_code} {dt:.1f}ms")
    response.headers["X-Request-ID"] = rid
    return response
