# -*- coding: utf-8 -*-
"""
OfferClaw · Day 1 Task 1 · LLM API 调用最小可运行脚本

目的：
    验证 "Python -> HTTP 请求 -> LLM 云端 -> 解析响应" 全链路可跑通。
    这是所有 LLM 应用和 Agent 项目的第一块积木。
    本脚本本身也会作为 Day 2 做 Agent Demo 的代码底座。

v0.6.3 (2026-05) 起：
    默认 provider 从智谱 GLM-4-Flash 切到 **OpenAI 兼容代理**
    （订阅中转站把 ChatGPT 账号包装成 OpenAI API）。
    模型 ``gpt-5.6-terra``，reasoning_effort ``medium``。
    Embeddings 由 rag_tools.py 的 EMBEDDING_PROVIDER 配置决定。

使用步骤：
    1. 把 ``.env.local`` 里的 ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` 设好。
    2. 安装唯一的第三方依赖：``pip install requests``
    3. 运行：``python day1_api_starter.py``

成功标志：
    控制台打印一段完整的 LLM 响应文本 + token usage 统计。
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time

import requests

# =====================================================
# Provider 配置 —— 想换 Provider 只改这几个常量（或改 .env.local 覆盖）
# =====================================================

# 默认：仅回环可见的 OpenAI 兼容开发服务；实际 provider/model 由本地环境覆盖。
DEFAULT_API_BASE = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TIMEOUT = 60.0

API_KEY_ENV = "OPENAI_API_KEY"

# 从 .env.local 或 shell 环境覆盖（env 优先级 > 这里的 default）
def _resolved(envvar: str, default: str) -> str:
    """``os.environ.get`` with a non-empty default fallback."""
    value = os.environ.get(envvar, "").strip()
    return value if value else default


# 备选 · 智谱 GLM（如果想切回旧路径，把下面 4 行解注释、注释掉上面的 OpenAI 默认即可）：
# DEFAULT_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
# DEFAULT_MODEL = "glm-4-flash"
# DEFAULT_REASONING_EFFORT = ""  # 智谱不支持 reasoning_effort
# API_KEY_ENV = "ZHIPU_API_KEY"


# =====================================================
# 本地密钥加载（从 .env.local 读）
# =====================================================


def load_local_env(path: str = ".env.local") -> None:
    """从同目录下的 .env.local 读取 KEY=VALUE 并注入 os.environ。

    约定：
    - 以 # 开头的行是注释
    - 空行跳过
    - 已在 os.environ 里的 KEY 不会被覆盖（显式设置的终端环境变量优先级更高）
    - 只用 Python 标准库，不引入 python-dotenv
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    # An explicitly present empty value is a deliberate opt-out (notably in
    # tests and offline runs); only consult the credential store when absent.
    if API_KEY_ENV not in os.environ and os.name == "nt":
        try:
            from windows_dpapi_secrets import load_dpapi_secret_into_env
            load_dpapi_secret_into_env(API_KEY_ENV)
        except (OSError, ValueError):
            pass


# =====================================================
# 智谱 JWT 签名（旧路径备用，OpenAI 兼容代理不需要）
# =====================================================


def build_zhipu_jwt(api_key: str, exp_seconds: int = 3600) -> str:
    """把智谱复合 API Key 编码为 JWT，用作 Bearer token。

    智谱的 API Key 形如 ``<api_key_id>.<signing_key>``（用 '.' 分隔）。
    官方规范是把它拆开、构造 JWT、用 signing_key 做 HS256 签名。
    参考：https://bigmodel.cn/dev/api/http-auth#jwt-auth

    v0.6.3 起默认走 OpenAI 兼容代理（直接用 raw API key 做 Bearer），
    这个函数保留供旧 Zhipu 路径或 embeddings 调用复用。
    """
    try:
        api_key_id, signing_key = api_key.split(".", 1)
    except ValueError:
        raise ValueError("智谱 API Key 格式应为 '<api_key_id>.<signing_key>'，请检查 .env.local")

    header = {"alg": "HS256", "sign_type": "SIGN"}
    now_ms = int(round(time.time() * 1000))
    payload = {
        "api_key": api_key_id,
        "exp": now_ms + exp_seconds * 1000,
        "timestamp": now_ms,
    }

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header_b64 = _b64(header)
    payload_b64 = _b64(payload)
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(
        signing_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")

    return f"{header_b64}.{payload_b64}.{signature_b64}"


# =====================================================
# 资源解析器 —— 供其他脚本复用
# =====================================================


def get_llm_config() -> dict:
    """读取当前激活的 LLM 配置（env 优先）。

    所有 chat-completion 调用都应该走这里，避免每个脚本各自硬编码。
    """
    load_local_env()
    return {
        "api_base": _resolved("OPENAI_BASE_URL", DEFAULT_API_BASE),
        "model": _resolved("LLM_MODEL", DEFAULT_MODEL),
        "reasoning_effort": _resolved("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        "timeout": float(_resolved("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))),
        "api_key": os.environ.get(API_KEY_ENV, ""),
        "api_key_env": API_KEY_ENV,
        # Legacy compat for the 智谱 / glm flow
        "is_zhipu": "bigmodel" in _resolved("OPENAI_BASE_URL", DEFAULT_API_BASE).lower(),
    }


# Backwards-compat: existing imports `from day1_api_starter import API_BASE, MODEL`
# still work. They snap to current env at import time.
API_BASE = _resolved("OPENAI_BASE_URL", DEFAULT_API_BASE)
MODEL = _resolved("LLM_MODEL", DEFAULT_MODEL)


# =====================================================
# 最小请求函数
# =====================================================


def _log_llm_usage(payload: dict, data: dict, elapsed_ms: int) -> None:
    """[P5 成本计量] 把每次 LLM 调用的 token 用量追加到 JSONL 台账。

    设计约束：
    - **永不抛错**——计量是旁路，任何异常不得影响主调用（裸 except 收口）；
    - 只记事实字段（模型/往返 token/耗时），**成本换算放在报表脚本**
      ``llm_cost_report.py`` 里做——单价会变，台账存原始 token 永远有效；
    - stream 响应拿不到 usage（SSE 分片无汇总），自然跳过；
    - 开关 ``LLM_USAGE_LOG``（默认开），路径 ``logs/llm_usage.jsonl``，
      单行 JSON append（行 < 4KB，POSIX append 原子性足够）。
    """
    try:
        if os.environ.get("LLM_USAGE_LOG", "1").strip().lower() in ("0", "false", "off"):
            return
        usage = (data or {}).get("usage") or {}
        if not usage:
            return
        import json as _json
        import datetime as _dt
        line = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": payload.get("model", "?"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "elapsed_ms": elapsed_ms,
        }
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "llm_usage.jsonl"), "a", encoding="utf-8") as f:
            f.write(_json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 计量失败绝不影响主链路


def _llm_fallback_config() -> dict | None:
    """Return an explicitly enabled fallback configuration.

    Fallback used to become active merely because three credential variables
    happened to exist in ``.env.local``.  That made a stale DeepSeek block an
    implicit production dependency.  It is now fail-closed: credentials may
    remain configured, but no request can reach the fallback endpoint unless
    ``LLM_FALLBACK_ENABLED=1`` is set deliberately.
    """
    enabled = os.environ.get("LLM_FALLBACK_ENABLED", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    key = os.environ.get("LLM_FALLBACK_API_KEY", "")
    base = os.environ.get("LLM_FALLBACK_BASE_URL", "").rstrip("/")
    model = os.environ.get("LLM_FALLBACK_MODEL", "")
    if not (key and base and model):
        return None
    return {"key": key, "base": base, "model": model}


def _failover_worthy(exc: Exception) -> bool:
    """额度/鉴权/限流/网络/5xx → 值得切兜底；参数类 4xx（400/404/422）→ 原样抛出，
    暴露配置错误（兜底不该掩盖打错的模型名）。"""
    if isinstance(exc, (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return False
    return status in (401, 402, 403, 429) or 500 <= status < 600


def chat_completion(url: str, headers: dict, payload: dict, *,
                    timeout: float, max_retries: int = 4, stream: bool = False):
    """[A1 统一 LLM 网关] 主配置调用 + 额度类失败自动切兜底模型。

    主调用语义见 :func:`_chat_completion_once`（重试/退避/限流分类原样保留）。
    兜底（2026-08-08）：显式开启 ``LLM_FALLBACK_ENABLED=1`` 且配置了
    ``LLM_FALLBACK_*`` 三变量时，主模型在
    额度/鉴权/限流/网络/5xx 类失败后自动改发兜底模型重试一轮（额度类硬错误
    会跳过主配置的长退避直接切换）；参数类 4xx 不兜底，直接暴露。
    流式与非流式同样生效；防自循环：主配置已指向兜底端点时不再切换。
    """
    fb = _llm_fallback_config()
    try:
        return _chat_completion_once(url, headers, payload, timeout=timeout,
                                     max_retries=max_retries, stream=stream,
                                     has_fallback=fb is not None)
    except Exception as e:
        if (fb is None or not url.rstrip("/").endswith("/chat/completions")
                or f"{fb['base']}/chat/completions" == url.rstrip("/")
                or not _failover_worthy(e)):
            raise
        import sys as _sys
        print(f"[LLM网关] 主模型失败（{type(e).__name__}），自动切换兜底模型 "
              f"{fb['model']}", file=_sys.stderr)
        fb_payload = dict(payload)
        fb_payload["model"] = fb["model"]
        fb_payload.pop("reasoning_effort", None)   # 兜底方未必支持该扩展参数
        fb_headers = {"Authorization": f"Bearer {fb['key']}",
                      "Content-Type": "application/json"}
        return _chat_completion_once(f"{fb['base']}/chat/completions", fb_headers,
                                     fb_payload, timeout=timeout,
                                     max_retries=max_retries, stream=stream)


def _chat_completion_once(url: str, headers: dict, payload: dict, *,
                          timeout: float, max_retries: int = 4, stream: bool = False,
                          has_fallback: bool = False):
    """chat-completion 单配置弹性调用（原 A1 网关本体），替换全部裸 requests.post。

    复刻检索侧 ``rag_tools.get_embeddings_batch`` 的重试哲学，但**只重试可恢复错误**：
    - 429 限流 → 长退避（10·2^n 秒）；5xx / 超时 / 连接重置 → 标准退避（2^n·2 秒）；
    - **403 且响应体含限流/配额特征词 → 按限流长退避重试**（2026-07-04 实测：
      dashscope compatible-mode 把 QPM 限流报成 403 而非 429——快速连发时
      eval_rag_answer 首题即炸，间隔后同一请求 200。403 的语义本是"不可恢复"，
      故仅在响应体命中限流特征时才重试，真正的鉴权 403 仍立即抛）；
    - 其余 4xx（参数 / 认证错）→ 立即抛出，不浪费重试（重试也不会变对）。
    ``stream=True`` 返回原始 Response 供 SSE 逐行读取；否则返回解析后的 JSON dict。
    重试次数可用环境变量 ``LLM_MAX_RETRIES`` 覆盖。
    """
    import time as _time
    import sys as _sys
    # ``max_retries=0`` conventionally means one attempt and no retry.  The
    # old ``range(0)`` skipped the HTTP request and ended with ``raise None``.
    attempts = max(1, int(os.environ.get("LLM_MAX_RETRIES", str(max_retries))))
    _THROTTLE_MARKERS = ("throttl", "rate limit", "ratelimit", "requests per",
                         "qpm", "quota", "allocated", "exceeded", "too many",
                         "limit_requests", "flow control", "流控", "限流")
    last_exc = None
    for attempt in range(attempts):
        try:
            _t0 = _time.time()
            from model_call_context import (record_model_request,
                                            record_model_response,
                                            remaining_seconds)
            record_model_request(payload)
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=remaining_seconds(timeout), stream=stream)
            resp.raise_for_status()
            if stream:
                return resp
            data = resp.json()
            record_model_response(data)
            _log_llm_usage(payload, data, int((_time.time() - _t0) * 1000))  # P5 旁路计量
            return data
        except requests.exceptions.RequestException as e:
            last_exc = e
            resp_obj = getattr(e, "response", None)
            status = getattr(resp_obj, "status_code", None)
            body = ""
            try:
                body = (resp_obj.text or "")[:500] if resp_obj is not None else ""
            except Exception:
                pass
            is_net = isinstance(e, (requests.exceptions.Timeout,
                                    requests.exceptions.ConnectionError))
            # 2026-07-04 实测：dashscope 免费档的"free quota has been exhausted"
            # 既可能是真耗尽，也可能是**按时间窗刷新的突发限额**（连发打爆、隔窗恢复
            # ——同一模型 70s 内 403、几分钟后 200）。故按限流长退避重试；重试全部
            # 打完仍失败时带 body 抛出，由人判断是去开通付费还是换模型。
            throttled_403 = (status == 403
                             and any(m in body.lower() for m in _THROTTLE_MARKERS))
            is_throttle = status == 429 or throttled_403
            # 有兜底时：额度类硬错误（欠费/余额/免费档耗尽/401/402）不值得长退避——
            # 立即抛给外层切兜底；无兜底时保持原退避语义（突发限额隔窗自愈有实测前科）。
            _HARD_QUOTA_MARKERS = ("insufficient", "exhausted", "arrears", "balance",
                                   "free tier", "freetieronly", "欠费", "余额不足")
            hard_quota = (status in (401, 402)
                          or (status in (403, 429)
                              and any(m in body.lower() for m in _HARD_QUOTA_MARKERS)))
            if has_fallback and hard_quota:
                if body:
                    print(f"[LLM网关] HTTP {status} 额度类错误，跳过重试直接切兜底: "
                          f"{body[:160]}", file=_sys.stderr)
                raise
            retriable = is_throttle or (status is not None and 500 <= status < 600) or is_net
            if attempt == attempts - 1 or not retriable:
                if status is not None and body:
                    # 失败时带上响应体片段——审查发现 raise_for_status 丢 body，
                    # 排障时分不清"鉴权 403"和"限流 403"。
                    print(f"[LLM网关] HTTP {status} 响应体: {body[:200]}", file=_sys.stderr)
                raise
            wait = 10 * (2 ** attempt) if is_throttle else (2 ** attempt) * 2
            from model_call_context import remaining_seconds
            bounded_wait = remaining_seconds(wait)
            if bounded_wait + 0.001 < wait:
                raise TimeoutError("OfferClaw query deadline exhausted") from e
            print(f"[LLM网关] {status or type(e).__name__} 失败 "
                  f"({attempt + 1}/{attempts})，{wait}s 后重试"
                  f"{'（403 限流特征）' if throttled_403 else ''}", file=_sys.stderr)
            _time.sleep(bounded_wait)
    raise last_exc  # 循环内必已 raise，此处仅为类型完整


def call_llm(prompt: str, api_key: str, *, system: str | None = None) -> dict:
    """发送一次 chat completion 请求，返回解析后的 JSON dict。

    使用当前 :func:`get_llm_config` 的配置（OpenAI 兼容代理 + gpt-5.6-terra +
    reasoning_effort medium 是默认）。

    参数：
        prompt  —— 用户输入的提问文本
        api_key —— Bearer token 用的 API 密钥（OpenAI 路径用 raw；
                   Zhipu 路径会自动 JWT 化）
        system  —— 可选 system 消息；默认是一个简洁助手 prompt

    返回：
        整个 JSON 响应对象（含 choices / usage / id 等）

    可能抛出：
        requests.HTTPError —— HTTP 非 2xx
        json.JSONDecodeError —— 响应不是合法 JSON
    """
    cfg = get_llm_config()
    if cfg["is_zhipu"]:
        bearer_token = build_zhipu_jwt(api_key)
    else:
        bearer_token = api_key

    url = f"{cfg['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    else:
        messages.append({"role": "system", "content": "你是一个严谨的技术助手，回答直接、不废话。"})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": cfg["model"],
        "messages": messages,
    }
    # Reasoning effort —— gpt-5.x 系列特性；OpenAI 兼容代理支持。
    # 智谱 / 普通 chat model 不认这个字段就忽略（多数代理会优雅丢弃）。
    if cfg["reasoning_effort"]:
        payload["reasoning_effort"] = cfg["reasoning_effort"]

    return chat_completion(url, headers, payload, timeout=cfg["timeout"])


class LLMResponseError(RuntimeError):
    """[A2] LLM 响应结构异常（空 choices / error 对象）。携带原始响应片段供排障。"""


def extract_content(data: dict) -> str:
    """[A2] 防御式提取 choices[0].message.content。

    代理因安全策略拦截 / 限流常返回**空 choices** 或 ``{"error": ...}``，裸下标会变成
    IndexError/KeyError 冒泡崩 CLI。这里统一校验：空 choices → 抛 :class:`LLMResponseError`
    （带原始响应片段），由上层 try/except 转成可读降级而非 traceback。
    """
    if not isinstance(data, dict) or not data.get("choices"):
        err = data.get("error") if isinstance(data, dict) else None
        raise LLMResponseError(
            f"LLM 响应缺少 choices（可能被代理拦截/限流）: {err or str(data)[:200]}")
    return data["choices"][0].get("message", {}).get("content", "") or ""


def extract_reply(data: dict) -> str:
    """从响应 JSON 里提取 LLM 的文本回复（防御式，见 :func:`extract_content`）。"""
    return extract_content(data)


def llm_error_detail(e) -> str:
    """[A2] 把 LLM 调用异常转成可读信息：HTTPError 带上代理响应体（当前裸抛会丢失这条），
    其它错误给「类型: 消息」。供 CLI/入口降级时呈现给用户而非 traceback。"""
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "text", None):
        return f"HTTP {getattr(resp, 'status_code', '?')}: {resp.text[:300]}"
    return f"{type(e).__name__}: {e}"


# =====================================================
# 主入口
# =====================================================


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_local_env()
    cfg = get_llm_config()

    api_key = cfg["api_key"]
    if not api_key:
        print(f"[ERROR] 未检测到环境变量 {cfg['api_key_env']}")
        print()
        print("有两种设置方式：")
        print(f"  A. 在 .env.local 文件里加一行：{cfg['api_key_env']}=你的密钥")
        print(f"  B. 在当前终端临时设置：")
        print(f"     PowerShell: $env:{cfg['api_key_env']} = '你的密钥'")
        print(f"     CMD:        set {cfg['api_key_env']}=你的密钥")
        sys.exit(1)

    prompt = "我想去洗车，但洗车行离我很近，那我是走过去还是开车过去？"

    print(f"[INFO] Provider         : {cfg['api_base']}")
    print(f"[INFO] Model            : {cfg['model']}")
    if cfg["reasoning_effort"]:
        print(f"[INFO] Reasoning effort : {cfg['reasoning_effort']}")
    print(f"[INFO] Prompt           : {prompt}")
    print("[INFO] 正在发送请求...")
    print()

    try:
        data = call_llm(prompt, api_key)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        reply = extract_reply(data)

        print("[RESPONSE] >>>")
        print(reply)
        print("<<< [END]")

        usage = data.get("usage", {})
        print()
        print(f"[USAGE] tokens = {usage}")

    except requests.HTTPError as e:
        print(f"[HTTP ERROR] {e}")
        print(f"响应体：{e.response.text if e.response is not None else 'N/A'}")
        print()
        print("常见原因：")
        print(f"  - API Key 错误（检查 {cfg['api_key_env']}）")
        print(f"  - 模型名称代理不支持（当前 model={cfg['model']}；请检查代理是否支持 gpt-5.6-terra）")
        print(f"  - 代理 endpoint 不通（当前 base={cfg['api_base']}）")
        sys.exit(1)

    except requests.Timeout:
        print(f"[TIMEOUT] 请求超过 {cfg['timeout']} 秒未响应，检查网络连接")
        sys.exit(1)

    except KeyError as e:
        print(f"[PARSE ERROR] 响应结构不符合预期：缺字段 {e}")
        print(f"原始响应：{json.dumps(data, ensure_ascii=False, indent=2)}")
        sys.exit(1)

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
