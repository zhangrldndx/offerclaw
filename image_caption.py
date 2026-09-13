# -*- coding: utf-8 -*-
"""image_caption.py — 图转文：用视觉模型（qwen-vl）把 markdown 里的图片
转成「描述 + OCR 文字」，使图片内容可被文本 RAG 检索。

设计：
- ingest 前对 .md 内容做一次预处理：``![alt](path)`` → ``[图: <描述/OCR>]``。
- 描述按图片身份（URL 或本地内容哈希）**缓存**到 knowledge_base/_image_captions.json，
  重建索引时复用，不重复调用视觉模型。
- 视觉模型走百炼 OpenAI 兼容接口（qwen-vl-plus，可用 VL_MODEL 覆盖），与现有 LLM 同账号。
- 本地图片用 base64 data URL 传入；http 图片直接传 URL。
- 任何失败（无 key/调用错/图缺失）都**降级为去掉该图**，不阻塞 ingest。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTION_CACHE_PATH = os.path.join(BASE_DIR, "knowledge_base", "_image_captions.json")
VL_MODEL = os.environ.get("VL_MODEL", "qwen-vl-plus")

_MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_IMG_EXT_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".gif": "image/gif", ".webp": "image/webp"}

_CAPTION_PROMPT = (
    "请用中文简洁描述这张图片的内容（1-3 句话），"
    "并提取图中所有可见文字（OCR，含公式、表格、代码）。"
    "只输出描述与文字本身，不要寒暄、不要分点标题。"
)


# ---------------- 缓存 ----------------

def _load_cache() -> dict:
    if os.path.exists(CAPTION_CACHE_PATH):
        try:
            with open(CAPTION_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CAPTION_CACHE_PATH), exist_ok=True)
    with open(CAPTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------- 图片定位 ----------------

def _resolve_image(url: str, md_path: str) -> tuple:
    """返回 (kind, value, cache_key)：
    kind='url' → value 是 http URL；kind='b64' → value 是 data URL；kind=None → 无法定位。
    """
    import posixpath
    if url.startswith(("http://", "https://")):
        return "url", url, "url:" + url
    if url.startswith("data:"):
        return "url", url, "data:" + hashlib.md5(url.encode()).hexdigest()[:16]
    # 本地相对路径：相对 md 文件所在目录解析
    base = os.path.dirname(os.path.abspath(md_path)) if md_path else BASE_DIR
    local = os.path.normpath(os.path.join(base, url))
    if not os.path.exists(local):
        return None, None, None
    with open(local, "rb") as f:
        raw = f.read()
    mime = _IMG_EXT_MIME.get(os.path.splitext(local)[1].lower(), "image/png")
    data_url = f"data:{mime};base64," + base64.b64encode(raw).decode()
    return "b64", data_url, "file:" + hashlib.md5(raw).hexdigest()


# ---------------- 视觉模型调用 ----------------

def _vl_endpoint() -> tuple[str, str]:
    """VL 端点独立解析（与主链路 LLM 配置解耦）。

    主链路 OPENAI_* 切到 GPT 代理/DeepSeek 后，图转文仍需打百炼——
    实测代理对图像输入会"接受请求但看不到图"（返回幻觉描述），DeepSeek 无视觉
    API，故 VL 默认固定百炼端点 + DASHSCOPE_API_KEY；可用 VL_BASE_URL /
    VL_API_KEY 显式覆盖。
    """
    base = (os.environ.get("VL_BASE_URL", "").rstrip("/")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    key = os.environ.get("VL_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    return base, key


def _vl_caption(image_value: str) -> str:
    """调 qwen-vl 生成 描述+OCR。无 key/失败返回空串。"""
    import requests
    from day1_api_starter import load_local_env
    load_local_env()
    base, api_key = _vl_endpoint()
    if not api_key:
        return ""  # 无视觉端点 key：图转文静默降级（图片按规则删除）
    payload = {
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": image_value}},
        ]}],
        "max_tokens": 500, "temperature": 0.1,
    }
    try:
        r = requests.post(f"{base}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}"},
                          json=payload, timeout=45)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content", "") or "").strip()
    except Exception:
        return ""


# ---------------- 主入口 ----------------

def caption_markdown(content: str, md_path: str = "", cache: dict | None = None,
                     stats: dict | None = None) -> str:
    """把 content 里的 ``![](img)`` 替换为 ``[图: 描述/OCR]``。

    cache 复用同一张图的描述；无法定位或视觉模型失败的图片直接删除（降级）。
    stats（可选 dict）记录 {captioned, cached, dropped} 计数。
    """
    own_cache = cache is None
    cache = _load_cache() if own_cache else cache
    counters = stats if stats is not None else {}
    counters.setdefault("captioned", 0)
    counters.setdefault("cached", 0)
    counters.setdefault("dropped", 0)

    matches = list(_MD_IMG.finditer(content))
    if not matches:
        return content

    # 1) 解析每个图片引用 → (key, value)；key=None 表示无法定位
    resolved = []  # 与 matches 顺序一致
    todo: dict[str, str] = {}  # 需新调用视觉模型的 key → image_value（去重）
    for m in matches:
        _kind, value, key = _resolve_image(m.group(2), md_path)
        resolved.append(key)
        if key and key not in cache and key not in todo:
            todo[key] = value

    # 2) 并发调用 qwen-vl 生成描述（多张图同时，避免串行 390 次网络往返）
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        workers = int(os.environ.get("VL_CONCURRENCY", "8"))

        def _one(item):
            k, v = item
            return k, _vl_caption(v)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, cap in ex.map(_one, list(todo.items())):
                if cap:
                    cache[k] = cap  # 失败的不写缓存，下次可重试

    # 3) 顺序替换：命中缓存 → [图: 描述]；无法定位/失败 → 删图（不杜撰）
    new_keys = set(todo.keys())
    it = iter(resolved)

    def repl(_m):
        key = next(it)
        if key and cache.get(key):
            counters["captioned" if key in new_keys else "cached"] += 1
            return f"[图: {cache[key]}]"
        counters["dropped"] += 1
        return ""

    out = _MD_IMG.sub(repl, content)
    if own_cache and counters["captioned"] > 0:
        _save_cache(cache)
    return out


def caption_enabled() -> bool:
    """是否启用图转文：需 IMAGE_CAPTION=1 且视觉端点有 key（独立于主链路 LLM）。"""
    if os.environ.get("IMAGE_CAPTION", "").strip() not in ("1", "true", "yes", "on"):
        return False
    from day1_api_starter import load_local_env
    load_local_env()
    _base, key = _vl_endpoint()
    return bool(key)
