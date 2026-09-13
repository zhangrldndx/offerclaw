# -*- coding: utf-8 -*-
"""
OfferClaw · RAG 工具模块

提供：
1. 智谱 JWT Bearer Token 签名（纯标准库，无 PyJWT，作为 legacy fallback）
2. 可配置 Embedding API 调用（智谱 / 百炼 / OpenAI 兼容）
3. Markdown 文档分块（LangChain + 自定义过滤）
4. ChromaDB 入库 / 检索封装

设计原则：
- 复用项目 1 的手写 JWT 风格，不引入 PyJWT
- embedding provider 由 .env.local 配置，切换 provider 时使用独立 Chroma collection
- 批量调用减少 HTTP 请求次数
- 错误重试 + 指数退避
- 分块按 Markdown 标题天然切分，过滤无效块
"""

import hashlib
import hmac
import base64
import time
import json
import os
import requests
import threading


def get_collection_sqlite_stats(db_dir: str, collection_name: str) -> tuple[int, dict[str, int]]:
    """Read Chroma collection metadata without loading its native vector client."""
    import sqlite3

    path = os.path.join(db_dir, "chroma.sqlite3")
    if not os.path.isfile(path):
        return 0, {}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        count = int(connection.execute(
            """SELECT COUNT(*) FROM embeddings e
               JOIN segments s ON s.id=e.segment_id
               JOIN collections c ON c.id=s.collection
               WHERE c.name=? AND s.scope='METADATA'""",
            (collection_name,),
        ).fetchone()[0])
        sources = {str(row[0] or "unknown"): int(row[1]) for row in connection.execute(
            """SELECT m.string_value,COUNT(*) FROM embedding_metadata m
               JOIN embeddings e ON e.id=m.id
               JOIN segments s ON s.id=e.segment_id
               JOIN collections c ON c.id=s.collection
               WHERE c.name=? AND s.scope='METADATA' AND m.key='source_type'
               GROUP BY m.string_value""",
            (collection_name,),
        )}
        connection.close()
        return count, sources
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return 0, {}


def get_collection_sqlite_fingerprint(
        db_dir: str, collection_name: str) -> tuple[int | None, str, list[str]]:
    """Hash one persistent collection through SQLite, safe across reader processes."""
    import sqlite3

    path = os.path.join(db_dir, "chroma.sqlite3")
    if not os.path.isfile(path):
        return None, "", []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        exists = connection.execute(
            "SELECT 1 FROM collections WHERE name=?", (collection_name,)
        ).fetchone()
        if not exists:
            connection.close()
            return None, "", []
        rows = connection.execute(
            """SELECT e.embedding_id,m.key,m.string_value,m.int_value,
                      m.float_value,m.bool_value
               FROM embeddings e
               JOIN segments s ON s.id=e.segment_id
               JOIN collections c ON c.id=s.collection
               LEFT JOIN embedding_metadata m ON m.id=e.id
               WHERE c.name=? AND s.scope='METADATA'
               ORDER BY e.embedding_id,m.key""",
            (collection_name,),
        )
        digest = hashlib.sha256()
        chunk_ids: set[str] = set()
        versions: set[str] = set()
        for embedding_id, key, string_value, int_value, float_value, bool_value in rows:
            chunk_ids.add(str(embedding_id))
            value = next((item for item in (string_value, int_value, float_value, bool_value)
                          if item is not None), None)
            if key == "chunker_version" and value is not None:
                versions.add(str(value))
            digest.update(json.dumps(
                [embedding_id, key, value], ensure_ascii=False, default=str,
                separators=(",", ":"),
            ).encode("utf-8"))
            digest.update(b"\n")
        connection.close()
        return len(chunk_ids), digest.hexdigest()[:16], sorted(versions)
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None, "", []


def get_collection_sqlite_documents(
        db_dir: str, collection_name: str) -> tuple[list[str], list[dict]]:
    """Load documents and metadata for lexical search without native Chroma calls."""
    import sqlite3

    path = os.path.join(db_dir, "chroma.sqlite3")
    if not os.path.isfile(path):
        return [], []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        rows = connection.execute(
            """SELECT e.embedding_id,m.key,m.string_value,m.int_value,
                      m.float_value,m.bool_value
               FROM embeddings e
               JOIN segments s ON s.id=e.segment_id
               JOIN collections c ON c.id=s.collection
               LEFT JOIN embedding_metadata m ON m.id=e.id
               WHERE c.name=? AND s.scope='METADATA'
               ORDER BY e.embedding_id,m.key""",
            (collection_name,),
        )
        by_id: dict[str, dict] = {}
        for embedding_id, key, string_value, int_value, float_value, bool_value in rows:
            item = by_id.setdefault(str(embedding_id), {"document": "", "metadata": {}})
            value = next((part for part in (string_value, int_value, float_value, bool_value)
                          if part is not None), None)
            if key == "chroma:document":
                item["document"] = str(value or "")
            elif key:
                item["metadata"][str(key)] = value
        connection.close()
        ordered = [by_id[key] for key in sorted(by_id)]
        return ([item["document"] for item in ordered],
                [item["metadata"] for item in ordered])
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return [], []


# =====================================================
# 密钥加载（复用 agent_demo.py 同款逻辑）
# =====================================================

def _load_local_env(path: str = ".env.local") -> None:
    """从同目录 .env.local 读取 KEY=VALUE 并注入 os.environ（已有的不覆盖）。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_local_env()


def _env(name: str, default: str = "") -> str:
    """Read a non-empty environment value with a default fallback."""
    value = os.environ.get(name, "").strip()
    return value if value else default


def _normalise_provider(provider: str) -> str:
    p = (provider or "zhipu").strip().lower().replace("-", "_")
    aliases = {
        "dashscope": "bailian",
        "aliyun": "bailian",
        "ali": "bailian",
        "zhipuai": "zhipu",
        "bigmodel": "zhipu",
        "openai": "openai_compatible",
        "openai_compat": "openai_compatible",
        # 本地 sentence-transformers（bge 等）：无额度、无 API、向量稳定
        "sentence_transformers": "local",
        "sentencetransformers": "local",
        "st": "local",
        "bge": "local",
        "huggingface": "local",
        "hf": "local",
    }
    return aliases.get(p, p)


# 默认 provider：**本地 bge**（无额度、无网络、稳定）。可用 EMBEDDING_PROVIDER 覆盖回 bailian/zhipu。
EMBEDDING_PROVIDER = _normalise_provider(_env("EMBEDDING_PROVIDER", "local"))


def _default_embedding_model(provider: str) -> str:
    if provider == "bailian":
        return "text-embedding-v4"
    if provider == "openai_compatible":
        return "text-embedding-3-small"
    if provider == "local":
        return "BAAI/bge-base-zh-v1.5"  # 默认本地模型：体积小、下载稳、中文强
    return "embedding-3"


def _default_embedding_base_url(provider: str) -> str:
    if provider == "bailian":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai_compatible":
        return "https://api.openai.com/v1"
    return "https://open.bigmodel.cn/api/paas/v4"


def _default_embedding_dimensions(provider: str, model: str) -> int | None:
    if provider == "bailian":
        return 1024
    if provider == "local":
        if "bge-m3" in model:
            return 1024
        if "bge-base" in model or "bge-large" in model:
            return 768   # bge-base/large-zh-v1.5 均 768 维
        return None
    if provider == "zhipu" and model == "embedding-3":
        return 2048
    return None


def _default_embedding_batch_size(provider: str) -> int:
    # 百炼 text-embedding 系列单次批量上限随模型变化，10 是稳妥默认值。
    if provider == "bailian":
        return 10
    if provider == "local":
        return 32  # 本地批量无网络往返，可大一些
    return 50


def _optional_int(name: str, default: int | None = None) -> int | None:
    value = _env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 应为整数，当前值: {value!r}") from exc


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def get_embedding_config() -> dict:
    """Return the active embedding provider config from environment variables."""
    provider = _normalise_provider(_env("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER))
    model = _env("EMBEDDING_MODEL", _default_embedding_model(provider))
    base_url = _env("EMBEDDING_BASE_URL", _default_embedding_base_url(provider)).rstrip("/")
    dimensions = _optional_int(
        "EMBEDDING_DIMENSIONS",
        _default_embedding_dimensions(provider, model),
    )
    batch_size = _optional_int(
        "EMBEDDING_BATCH_SIZE",
        _default_embedding_batch_size(provider),
    )
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "endpoint": f"{base_url}/embeddings",
        "dimensions": dimensions,
        "batch_size": batch_size or _default_embedding_batch_size(provider),
    }


def get_collection_name() -> str:
    """Return the Chroma collection name for the active embedding provider.

    The legacy Zhipu embedding-3 collection stays as offerclaw_docs for
    backwards compatibility. Other providers get isolated collections so
    vectors from different embedding spaces are never mixed.
    """
    configured = _env("RAG_COLLECTION_NAME")
    if configured:
        return configured

    cfg = get_embedding_config()
    if cfg["provider"] == "zhipu" and cfg["model"] == "embedding-3":
        return "offerclaw_docs"

    dim = f"_{cfg['dimensions']}" if cfg.get("dimensions") else ""
    return f"offerclaw_{_slug(cfg['provider'])}_{_slug(cfg['model'])}{dim}"


def _get_provider_api_key(provider: str) -> str:
    generic = _env("EMBEDDING_API_KEY")
    if generic:
        return generic
    if provider == "bailian":
        return _env("DASHSCOPE_API_KEY") or _env("BAILIAN_API_KEY")
    if provider == "openai_compatible":
        return _env("OPENAI_API_KEY")
    return _env("ZHIPU_API_KEY")


def has_embedding_api_key() -> bool:
    cfg = get_embedding_config()
    if cfg["provider"] == "local":
        return True  # 本地模型无需 API key，始终"可用"
    return bool(_get_provider_api_key(cfg["provider"]))


def describe_embedding_config() -> str:
    cfg = get_embedding_config()
    dim = f", dim={cfg['dimensions']}" if cfg.get("dimensions") else ""
    key_state = "key=已配置" if has_embedding_api_key() else "key=未配置"
    return f"{cfg['provider']}/{cfg['model']}{dim}, {key_state}, collection={get_collection_name()}"


def _get_embedding_api_key(provider: str) -> str:
    key = _get_provider_api_key(provider)
    if key:
        return key
    if provider == "bailian":
        hint = "DASHSCOPE_API_KEY=你的百炼API Key"
    elif provider == "openai_compatible":
        hint = "OPENAI_API_KEY=你的 API Key"
    else:
        hint = "ZHIPU_API_KEY=你的key_id.你的signing_secret"
    raise RuntimeError(f"未找到 embedding API Key。请在 .env.local 里加一行：\n  {hint}")


def _get_api_key() -> str:
    """读取智谱 API Key（格式：<key_id>.<signing_secret>）。"""
    return _get_embedding_api_key("zhipu")


# 模型配置
_EMBEDDING_CFG = get_embedding_config()
EMBEDDING_MODEL = _EMBEDDING_CFG["model"]
# 仅作为 ``chat_with_llm`` 的默认参数哨兵；真实调用仍从
# day1_api_starter.get_llm_config() 解析活动配置。
LLM_MODEL = "gpt-5.6-terra"
EMBEDDING_ENDPOINT = _EMBEDDING_CFG["endpoint"]
CHAT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 分块配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MIN_CHUNK_CHARS = 50  # 小于此字符数的块将被过滤

# =====================================================
# 1. JWT Bearer Token 签名（纯标准库实现，无 PyJWT 依赖）
# =====================================================

def _base64url_encode(data: bytes) -> str:
    """URL-safe Base64 编码，去掉尾部 ="""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def generate_zhipu_token(exp_seconds: int = 3600) -> str:
    """
    生成智谱 API 的 JWT Bearer Token。
    纯标准库实现，无 PyJWT 依赖。与 agent_demo.py 行为完全一致：
      - 从 ZHIPU_API_KEY 环境变量读取，格式 "<key_id>.<signing_secret>"
      - exp / timestamp 用毫秒（智谱要求）
    """
    raw_key = _get_api_key()
    try:
        api_key_id, signing_secret = raw_key.split(".", 1)
    except ValueError:
        raise ValueError("ZHIPU_API_KEY 格式应为 '<key_id>.<signing_secret>'，请检查 .env.local")

    header = {"alg": "HS256", "sign_type": "SIGN"}
    now_ms = int(round(time.time() * 1000))           # 毫秒，与智谱要求一致
    payload = {
        "api_key": api_key_id,
        "exp": now_ms + exp_seconds * 1000,
        "timestamp": now_ms,
    }

    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    signing_input = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        signing_secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{signing_input}.{signature_b64}"


# =====================================================
# 2. Embedding API 调用
# =====================================================

def get_embedding(
    text: str,
    model: str | None = None,
    max_retries: int = 3,
) -> list[float]:
    """
    获取单段文本的 embedding 向量。
    失败自动重试，指数退避。
    """
    return get_embeddings_batch([text], model=model, max_retries=max_retries)[0]


_LOCAL_MODEL_CACHE: dict = {}
_LOCAL_EMBED_LOCK = threading.Lock()


def _local_embed(texts: list[str], model: str) -> list[list[float]]:
    """用本地 sentence-transformers 模型（bge-m3 等）编码，返回归一化向量。

    模型按名缓存（首次加载较慢，之后常驻）。需 ``pip install sentence-transformers``。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "本地 embedding 需要 sentence-transformers：pip install sentence-transformers"
        ) from e
    m = _LOCAL_MODEL_CACHE.get(model)
    if m is None:
        m = _load_local_model(model)
        # 序列长度上限旋钮(2026-08-09):bge-m3 原生 8192,超长代码块 chunk 会让
        # CPU 注意力计算撞二次方墙(单条几分钟,现象=入库"挂死");512 与 bge-base-zh
        # 的截断行为对齐,也是本库 chunk 尺寸(≈500-800 字)的合理上限。
        _max_seq = os.environ.get("OFFERCLAW_EMBED_MAX_SEQ", "").strip()
        if _max_seq.isdigit():
            try:
                m.max_seq_length = int(_max_seq)
            except Exception:
                pass
        _LOCAL_MODEL_CACHE[model] = m
    _px = os.environ.get("OFFERCLAW_EMBED_PREFIX", "")   # e5 契约:doc 侧 passage: /query 侧由调用方传
    if _px:
        texts = [f"{_px}{t}" for t in texts]
    # SentenceTransformer/PyTorch 的 MPS 后端不保证同一模型实例可被多个线程并发
    # encode。混合路由首问曾在 MetalShaderLibrary 中触发 EXC_BAD_ACCESS，直接杀死
    # uvicorn 进程而不会留下 Python traceback。进程内统一串行化本地编码；远程
    # embedding 不受此锁影响。
    with _LOCAL_EMBED_LOCK:
        vecs = m.encode(texts, normalize_embeddings=True, batch_size=32,
                        show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vecs]


_ST_MODEL_CACHE: dict = {}   # (model, device) -> SentenceTransformer,失败不缓存
_ST_MODEL_LOAD_LOCK = threading.Lock()  # 多路并行首问时防止同一模型重复加载/下载


def _load_local_model(model: str):
    """加载本地 SentenceTransformer 模型，进程内按 (model, device) 缓存。

    优先按给定名/路径从 HuggingFace 加载；国内 HF 不通时自动回退 ModelScope 快照
    （EMBEDDING_MODEL 也可直接填本地快照绝对路径，跳过下载）。
    缓存（2026-08-10 实测驱动）：rag_paper_route 按需加载 e5(1.1GB) 曾因无缓存
    每次主库判弱都重载，单查询拖到分钟级；device 旋钮参与缓存键。
    """
    # 设备旋钮（2026-07-04 事故驱动）：多进程并发打 MPS 曾把 Metal 干进持久坏状态
    # （新 torch-MPS 客户端全部卡死在首个 command buffer 的 waitUntilCompleted，
    # kill -9 都收不掉，重启机器前 GPU 不可用）。OFFERCLAW_TORCH_DEVICE=cpu 强制
    # 走 CPU——慢但不会与 GPU 状态同归于尽；不设 = sentence-transformers 自动选。
    _dev = os.environ.get("OFFERCLAW_TORCH_DEVICE") or None
    _key = (model, _dev)
    cached = _ST_MODEL_CACHE.get(_key)
    if cached is None:
        with _ST_MODEL_LOAD_LOCK:
            # 等锁期间另一条检索路由可能已经完成加载。
            cached = _ST_MODEL_CACHE.get(_key)
            if cached is None:
                cached = _load_local_model_uncached(model, _dev)
                _ST_MODEL_CACHE[_key] = cached
    return cached


def _load_local_model_uncached(model: str, _dev: str | None):
    import contextlib
    import sys as _sys
    from sentence_transformers import SentenceTransformer
    if os.path.isdir(model):  # 直接给本地快照路径
        with contextlib.redirect_stdout(_sys.stderr):
            return SentenceTransformer(model, device=_dev)
    # 国内优先 ModelScope（已缓存则秒回，未缓存下载也快）；不可用再退 HuggingFace
    # 关键：把加载期所有打印重定向到 stderr，避免污染 stdout 的 JSON 输出
    with contextlib.redirect_stdout(_sys.stderr):
        try:
            from modelscope import snapshot_download
            return SentenceTransformer(snapshot_download(model), device=_dev)
        except Exception:
            pass
        try:
            return SentenceTransformer(model, device=_dev)
        except Exception as e:
            raise RuntimeError(
                f"本地 embedding 模型 {model} 加载失败（ModelScope 与 HuggingFace 均不可用）：{e}"
            ) from e


def get_embeddings_batch(
    texts: list[str],
    model: str | None = None,
    batch_size: int | None = None,
    max_retries: int = 6,
    throttle: float = 0.5,
) -> list[list[float]]:
    """
    批量获取 embedding 向量。
    Embedding API 支持 input 数组，一次调用可处理多条文本。
    如果文本量超过 batch_size，自动分批。

    限流处理：
    - 每个成功批次之间 sleep ``throttle`` 秒，平滑请求速率
    - 429（Too Many Requests）使用更长的指数退避（10s, 20s, 40s...）
    - 其它网络错误使用较短退避（2s, 4s, 8s...）
    """
    cfg = get_embedding_config()
    provider = cfg["provider"]
    batch_size = batch_size or int(cfg["batch_size"])
    model = model or cfg["model"]
    endpoint = cfg["endpoint"]

    # 本地 sentence-transformers（bge 等）：本地推理，无 API/额度/限流
    if provider == "local":
        return _local_embed(texts, model)

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                if provider == "zhipu":
                    token = generate_zhipu_token()  # 每次重试都重签，避免 token 过期
                else:
                    token = _get_embedding_api_key(provider)

                payload: dict = {
                    "model": model,
                    "input": batch,
                    "encoding_format": "float",
                }
                if provider == "bailian" and cfg.get("dimensions"):
                    payload["dimensions"] = cfg["dimensions"]

                resp = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                batch_embeddings = [d["embedding"] for d in data["data"]]
                all_embeddings.extend(batch_embeddings)
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                # 429 限流：更长退避；其它错误：标准退避
                is_rate_limit = (
                    getattr(e, "response", None) is not None
                    and e.response.status_code == 429
                )
                if is_rate_limit:
                    wait = 10 * (2 ** attempt)
                    print(f"  [WARN] 触发限流 429 (attempt {attempt+1}/{max_retries})")
                else:
                    wait = (2 ** attempt) * 2
                    print(f"  [WARN] 批量 Embedding 请求失败 (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  [INFO] {wait}s 后重试...")
                time.sleep(wait)

        # 批次间节流，平滑速率，避免连续 burst 触发 429
        if throttle:
            time.sleep(throttle)

    return all_embeddings


def fake_embedding(text: str, dim: int | None = None) -> list[float]:
    """Deterministic fake vector for offline ingest/query smoke tests.

    注意：必须用「无符号整数」解包哈希字节再归一化，**不能**用 ``struct.unpack('f')``
    把任意字节当 IEEE-754 float32 —— 后者约 95% 概率落到 NaN/Inf 位模式，会被
    ChromaDB 直接拒绝（embeddings must not contain NaN/Infinity），导致无 API key
    的离线 ingest 崩溃或污染向量库。整数解包恒为有限值。
    """
    import struct

    dim = dim or get_embedding_config().get("dimensions") or 384
    h = hashlib.sha256(text.encode("utf-8")).digest()
    extended = b""
    while len(extended) < dim * 4:
        h = hashlib.sha256(h).digest()
        extended += h
    ints = struct.unpack(f"{dim}I", extended[: dim * 4])  # 无符号 32-bit，恒有限
    mn, mx = min(ints), max(ints)
    if mx == mn:
        return [0.5] * dim
    span = mx - mn
    return [(v - mn) / span for v in ints]


# =====================================================
# 3. Markdown 文档分块
# =====================================================

def _info_chars(text: str) -> int:
    """信息量度量(2026-08-10 review 修复⑦:英文口径)。

    原过滤门是纯字符数、按中文调的:中文 80 字符 ≈ 80 token 的信息量,英文 80 字符
    仅 ≈13 词 ≈20 token,于是作者名/邮箱/页眉/脚注在英文语料上畅通无阻。

    做法:按文本主体语言换算成"等效中文字符"——
      · CJK 主导(含少量英文术语的中文正文)→ 直接用字符数(与历史行为一致,不误伤);
      · 拉丁主导 → 按词数 × 3.2(一个英文词的信息量 ≈3 个汉字),使 80 门 ≈25 词。
    分语言而非一刀切折算,避免"中文句子里有个 MySQL 就跌破门"的回归(实测踩到)。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if cjk * 3 >= len([c for c in text if not c.isspace()]):   # CJK 占比 ≥1/3 → 中文口径
        return len(text)
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    return int(len(words) * 3.2)


_CITE_PATTERNS = (
    "arxiv preprint", "et al.", "in proceedings", "in advances in neural",
    "doi.org", "conference on", "journal of",
)


def is_citation_dense(text: str, min_hits: int = 6, per_kchar: float = 3.0) -> bool:
    """参考文献段检测(review 修复③:噪声磁铁)。

    参考文献是论文标题的密集堆砌,对"有哪些工作研究 X"这类问题语义密度极高、极易
    冲到 top-1,但内容上只能让 LLM 抄引文或编造。判据:引文标记绝对数与千字密度
    双条件(避免误伤正文里偶尔引用一两篇的段落)。
    """
    low = (text or "").lower()
    hits = sum(low.count(p) for p in _CITE_PATTERNS)
    if hits < min_hits:
        return False
    return hits / max(len(low) / 1000.0, 1.0) >= per_kchar


def _hard_split(text: str, limit: int) -> list:
    """硬上限切分(review 修复②:35% 正文检索时不可见)。

    原逻辑按空行切段,一段内无空行就整段输出、无上限——论文附录 prompt 大表、
    Docling 压平的表格正好是"一大坨无空行文本",实测最大块 18379 字符。
    这种块在两个阶段的行为是反的:embedding 只吃前 ~512 token(89% 内容对检索
    不可见),命中后却把全文塞给 LLM 吃掉大半上下文预算。
    这里按行边界(退化时按硬字符)兜底切到 limit 以内。
    """
    if len(text) <= limit:
        return [text]
    out, buf = [], []
    buf_len = 0
    for line in text.splitlines(keepends=True):
        while len(line) > limit:            # 单行超限:按字符硬切
            if buf:
                out.append("".join(buf)); buf, buf_len = [], 0
            out.append(line[:limit])
            line = line[limit:]
        if buf_len + len(line) > limit and buf:
            out.append("".join(buf)); buf, buf_len = [], 0
        buf.append(line); buf_len += len(line)
    if buf:
        out.append("".join(buf))
    return [c.strip() for c in out if c.strip()]


def split_markdown_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = None,
    strict_hygiene: bool = None,
    preserve_short_sections: bool = False,
) -> list[dict]:
    """
    按 Markdown 二级标题（##）智能切分文档。
    返回 list[dict]，每个 dict 包含：
    - "text": 块内容
    - "metadata": {"char_len": int, "title": str}
    """
    # 硬上限默认 2000 字符 ≈ 512 token 安全线(与 OFFERCLAW_EMBED_MAX_SEQ 对齐):
    # 超过它的块 embedding 只吃前半截,检索时"看不见"后面的内容。
    # 卫生规则按域生效(2026-08-18 实测修正):硬上限/引文过滤/英文口径三项是为**英文论文**
    # 语料设计的(超线块占论文库 35% 字符);中文主库超线块仅 1%,收益微乎其微,而 A/B 实测
    # 把它们套上主库反而 held-out R@1 50.0→44.2、同分布 86.0→85.0(连贯中文段被切碎)。
    # 故默认 strict_hygiene=False = 保持历史行为;论文入库路径显式传 True。
    # 家法:测正才采纳——不因"改都改了"而保留一个实测变差的默认值。
    if strict_hygiene is None:
        strict_hygiene = os.environ.get("OFFERCLAW_CHUNK_STRICT", "0") == "1"
    if max_chars is None:
        max_chars = int(os.environ.get("OFFERCLAW_CHUNK_MAX_CHARS", "2000")) \
            if strict_hygiene else 10 ** 9
    chunks = []

    # 2026-07-04 检索质量修复①：剥掉文件头 YAML 元数据（--- title/source_url/... ---）。
    # 此前它被当正文切成块入库——实测全库有 79 个这类"元数据块"，标题词密度高，
    # 检索时反而挤掉真内容（be03「最左前缀」检索回来的第2条就是它）。
    if text.lstrip().startswith("---"):
        stripped = text.lstrip()
        end = stripped.find("\n---", 3)
        if end != -1 and ("title:" in stripped[:end] or "source_url" in stripped[:end]):
            text = stripped[end + 4:]

    # 按 ## 二级标题分段
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = "__header__"  # 标记文件开头无标题部分
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            # 保存上一段
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip().lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    # 对每个 section 切块
    import re as _re
    for title, section_lines in sections:
        # 注：原"图片素材"段落整段跳过的逻辑已移除——开启图转文后这些段含 [图:描述]
        # 是真内容应保留；未开启时原始 ![](url) 行会被下面过滤掉、空段由 min_chars 兜底跳过。

        # 过滤未转文的原始图片行（![](url)）；保留 [图: …] 这类已转文的图片描述
        # （图转文由 image_caption 在 ingest 前完成，描述含真实内容，应进入索引）
        cleaned_lines = [
            ln for ln in section_lines
            if not ln.strip().startswith("![")
        ]
        section_text = "\n".join(cleaned_lines).strip()

        if title == "__header__":
            title = ""  # 清除标记

        if not section_text or len(section_text) < min_chars:
            continue

        # 2026-07-04 检索质量修复②：跳过"目录型"块——剥掉非正文行后净正文
        # 不足 80 字的块（如飞书文章开头的全目录）。实测全库有 300+ 这类块，
        # 标题词最密、最容易被检索选中，但对回答毫无内容（be03 检索回来的两条
        # 分别是整页目录和文件头元数据，导致系统只能拒答）。
        # 非正文行 = ① # 标题行；② 超短行(<6字)；③ 目录式列表行——以 -/* 开头、
        # 剥掉标记后 <25 字且不含句读（真列表项通常带句号/逗号/冒号或较长）。
        def _is_toc_line(s: str) -> bool:
            if not (s.startswith("-") or s.startswith("*")):
                return False
            body = s.lstrip("-* ").strip()
            # 编号目录行("1.MySQL数据库——三范式")带英文句点,不能当"真内容"标点;
            # 只有句子级标点(句号/逗号/冒号等)才说明是真列表内容。be03 复盘发现
            # overview 文件的编号目录靠 "1." 里的点位逃过了第一版过滤。
            return len(body) < 30 and not any(p in body for p in "。，：；、,;)")
        _content_lines = [
            ln.strip() for ln in section_text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
            and len(ln.strip()) >= 6 and not _is_toc_line(ln.strip())
        ]
        if not preserve_short_sections and sum(len(ln) for ln in _content_lines) < 80:
            continue

        def _emit(txt: str):
            """统一出口:硬上限 + 逐块噪声过滤(信息量/引文密度)。

            2026-08-10 修复:原噪声门只作用于**整节**,节内按空行切出的小段只过
            50 字符 min_chars,于是 55 字符的表题/脚注从缝里漏进库(实测 7 块)。
            """
            for piece in _hard_split(txt.strip(), max_chars):
                if len(piece) < min_chars:
                    continue
                if strict_hygiene and is_citation_dense(piece):
                    continue
                if strict_hygiene:      # 逐块噪声门(严格档);历史档保持整节判定
                    lines = [ln.strip() for ln in piece.splitlines()
                             if ln.strip() and not ln.strip().startswith("#")
                             and len(ln.strip()) >= 6 and not _is_toc_line(ln.strip())]
                    if sum(_info_chars(ln) for ln in lines) < 80:
                        continue
                chunks.append({"text": piece,
                               "metadata": {"char_len": len(piece), "title": title}})

        if len(section_text) <= chunk_size:
            _emit(section_text)
        else:
            paragraphs = section_text.split("\n\n")
            buf: list[str] = []
            buf_len = 0
            for para in paragraphs:
                add_len = len(para) + (2 if buf else 0)
                if buf_len + add_len > chunk_size and buf:
                    _emit("\n\n".join(buf))
                    buf = [para]
                    buf_len = len(para)
                else:
                    buf.append(para)
                    buf_len += add_len
            if buf:
                _emit("\n\n".join(buf))

    return chunks


# =====================================================
# 4. LLM 问答（问答阶段使用）
# =====================================================

def chat_with_llm(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    调用当前配置的 OpenAI 兼容 LLM 进行对话。
    messages 格式：[{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    from day1_api_starter import build_zhipu_jwt, get_llm_config

    cfg = get_llm_config()
    token = build_zhipu_jwt(cfg["api_key"]) if cfg["is_zhipu"] else cfg["api_key"]

    resp = requests.post(
        f"{cfg['api_base']}/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": model if model != LLM_MODEL else cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


_INDEX_FP_CACHE: dict[tuple, tuple[float, dict]] = {}
_GIT_COMMIT_CACHE: str | None = None


def _index_revision_token(collection) -> tuple:
    """Return a cheap cache invalidator without exposing indexed content.

    Chroma does not expose a portable collection generation, so the local
    SQLite/WAL stat is the production signal.  Test adapters and future stores
    may expose ``fingerprint_revision`` directly.
    """

    parts: list[object] = [getattr(collection, "fingerprint_revision", None)]
    db_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    for filename in ("chroma.sqlite3", "chroma.sqlite3-wal"):
        try:
            stat = os.stat(os.path.join(db_root, filename))
            parts.extend((stat.st_mtime_ns, stat.st_size))
        except OSError:
            parts.extend((None, None))
    return tuple(parts)


def index_fingerprint(*, collection=None, cache_ttl: float = 30.0) -> dict:
    """索引指纹(方案文档 P0.3):把"指标属于哪个索引"钉死,杜绝旧索引数字混用。

    评测结果落盘时随 _meta 一并保存;任何两份结果对比前先比指纹——
    collection/模型/维度/截断/精排任一不同,数字就不可直接比。
    """
    import datetime as _dt
    import subprocess as _sp
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is None:
        try:
            _GIT_COMMIT_CACHE = _sp.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).stdout.strip() or "unknown"
        except Exception:
            _GIT_COMMIT_CACHE = "unknown"
    commit = _GIT_COMMIT_CACHE
    name = get_collection_name()
    count = None
    content_hash = ""
    indexed_chunker_versions: list[str] = []
    try:
        native_collection = (collection is not None and
                             type(collection).__module__.startswith("chromadb."))
        if collection is None or native_collection:
            if native_collection:
                name = str(getattr(collection, "name", "") or name)
            count, content_hash, indexed_chunker_versions = get_collection_sqlite_fingerprint(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"), name)
        else:
            count = collection.count()
            cache_key = (name, count, _index_revision_token(collection))
            cached = _INDEX_FP_CACHE.get(cache_key)
            if cached and (time.monotonic() - cached[0]) <= max(0.0, cache_ttl):
                content_hash = cached[1].get("collection_content_hash", "")
                indexed_chunker_versions = list(cached[1].get("indexed_chunker_versions", []))
            else:
                # Hash aligned IDs, document digests and retrieval-relevant metadata.
                got = collection.get(include=["documents", "metadatas"])
                ids = [str(value) for value in (got.get("ids") or [])]
                documents = list(got.get("documents") or [])
                metadatas = list(got.get("metadatas") or [])
                rows = []
                for index, chunk_id in enumerate(ids):
                    document = str(documents[index]) if index < len(documents) else ""
                    metadata = metadatas[index] if index < len(metadatas) else {}
                    metadata = metadata if isinstance(metadata, dict) else {}
                    rows.append((chunk_id, document, metadata))
                rows.sort(key=lambda item: item[0])
                digest = hashlib.sha256()
                for chunk_id, document, metadata in rows:
                    digest.update(chunk_id.encode("utf-8")); digest.update(b"\0")
                    digest.update(hashlib.sha256(document.encode("utf-8")).digest())
                    digest.update(b"\0")
                    digest.update(json.dumps(
                        metadata, ensure_ascii=False, sort_keys=True, default=str,
                        separators=(",", ":"),
                    ).encode("utf-8")); digest.update(b"\n")
                content_hash = digest.hexdigest()[:16]
                indexed_chunker_versions = sorted({
                    str(metadata.get("chunker_version"))
                    for _chunk_id, _document, metadata in rows
                    if metadata.get("chunker_version")
                })
                _INDEX_FP_CACHE[cache_key] = (time.monotonic(), {
                    "collection_content_hash": content_hash,
                    "indexed_chunker_versions": indexed_chunker_versions,
                })
    except Exception:
        pass
    try:
        embedding = get_embedding_config()
    except Exception:
        embedding = {}
    stable = {
        "collection": name,
        "collection_count": count,
        "collection_content_hash": content_hash,
        "embedding_provider": embedding.get("provider", os.environ.get("EMBEDDING_PROVIDER", "")),
        "embedding_model": embedding.get("model", os.environ.get("EMBEDDING_MODEL", "")),
        "embedding_dimensions": embedding.get("dimensions", os.environ.get("EMBEDDING_DIMENSIONS", "")),
        "embed_max_seq": os.environ.get("OFFERCLAW_EMBED_MAX_SEQ", ""),
        "rerank_model": os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base"),
        "rerank_on": os.environ.get("RAG_RERANK", "1"),
        "chunker_version": globals().get("CHUNKER_VERSION", ""),
        "indexed_chunker_versions": indexed_chunker_versions,
    }
    stable_json = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    return {
        **stable,
        # Release identity is audit metadata only.  It must not participate in
        # the index/Gate signature, otherwise committing a calibrated profile
        # changes the signature and makes that profile impossible to activate.
        "git_commit": commit,  # backwards-compatible audit alias
        "audit_git_commit": commit,
        "index_content_fingerprint": content_hash,
        "fingerprint_id": hashlib.sha256(stable_json.encode("utf-8")).hexdigest()[:16],
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


# =====================================================
# Embedding 血缘与集合契约(2026-08-10 review 修复①:防静默毁库)
# =====================================================

CHUNKER_VERSION = "2026-08-10"   # 分块规则变更时手动 +1(硬上限/引文过滤/英文口径)


def embed_profile() -> str:
    """当前 embedding 配置的血缘签名,写进每个 chunk 的 metadata。

    动机(review 实测风险):e5 契约要求入库 `passage: `/查询 `query: ` 前缀,但入库侧
    只靠 env。若某天忘设 EMBEDDING_MODEL/PREFIX 往 e5 集合 --add,**维度同为 768
    不会报错**,向量却落在另一空间——块永远检索不到,还污染距离门,事后无从排查。
    有了签名,`assert_collection_contract` 就能在写入前拦住。
    """
    model = os.environ.get("EMBEDDING_MODEL", "") or "(default)"
    return "|".join([
        os.environ.get("EMBEDDING_PROVIDER", "") or "(default)",
        os.path.basename(model.rstrip("/")),
        os.environ.get("EMBEDDING_DIMENSIONS", "") or "?",
        f"prefix={os.environ.get('OFFERCLAW_EMBED_PREFIX', '') or '-'}",
        f"maxseq={os.environ.get('OFFERCLAW_EMBED_MAX_SEQ', '') or '-'}",
    ])


class EmbeddingContractError(RuntimeError):
    """集合内既有块的 embedding 血缘与当前配置不符——拒绝写入(fail-visible)。"""


def assert_collection_contract(collection, sample: int = 50) -> str | None:
    """写入前校验:集合已有块的 embed_profile 必须与当前配置一致。

    返回既有签名(空集合返回 None,视为首次建库放行)。不一致抛
    EmbeddingContractError,并给出"要么改 env 要么换集合"的可执行指引。
    历史块无签名(老库)→ 只告警不拦(不因治理动作阻断既有工作流)。
    """
    try:
        got = collection.get(limit=sample, include=["metadatas"])
    except Exception:
        return None
    metas = [m for m in (got.get("metadatas") or []) if m]
    if not metas:
        return None
    seen = {m.get("embed_profile") for m in metas if m.get("embed_profile")}
    if not seen:
        print("  [契约] 集合内为历史块(无 embedding 血缘),跳过校验;建议择机 --rebuild 补签名")
        return None
    cur = embed_profile()
    if cur not in seen:
        raise EmbeddingContractError(
            f"embedding 血缘不符,拒绝写入以防污染集合。\n"
            f"    集合既有: {sorted(seen)}\n"
            f"    当前配置: {cur}\n"
            f"    修法:① 按既有签名设 EMBEDDING_MODEL / EMBEDDING_DIMENSIONS / "
            f"OFFERCLAW_EMBED_PREFIX 后重试;② 或换一个新集合名(RAG_COLLECTION_NAME)。")
    return cur
