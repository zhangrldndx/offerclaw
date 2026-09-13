"""
OfferClaw · 工程健康检查（doctor.py）

用法：
    python doctor.py

逐项检查 OfferClaw 是否准备好运行：
- Python 版本、依赖、API Key
- 核心 markdown / 代码 / 测试文件
- ChromaDB 向量库
- pytest 是否可跑（仅 collect）

输出 [OK] / [WARN] / [ERR]，最后返回退出码（0=全 OK，1=有 WARN，2=有 ERR）。
"""
from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
results: list[tuple[str, str]] = []  # (level, msg)


def _load_env_local() -> bool:
    """从 .env.local 读 KEY 注入到当前进程 env。
    - 仅当当前进程未注入时才读
    - 不打印 KEY 内容（只回报 是 / 否 加载成功）
    - .env.local 已被 .gitignore（.env.* 规则）排除，不入 git
    返回 True 表示本次成功补注入。
    """
    env_local = ROOT / ".env.local"
    if not env_local.exists():
        return False
    injected = False
    for raw in env_local.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v
            if k == "ZHIPU_API_KEY":
                injected = True
    return injected


_load_env_local()


def ok(msg: str) -> None: results.append(("OK", msg))
def warn(msg: str) -> None: results.append(("WARN", msg))
def err(msg: str) -> None: results.append(("ERR", msg))


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        err(f"Python {v.major}.{v.minor} 过旧，要求 >= 3.10")


def check_packages() -> None:
    needed = ["fastapi", "uvicorn", "pydantic", "chromadb", "requests", "pytest"]
    missing = []
    for p in needed:
        if importlib.util.find_spec(p) is None:
            missing.append(p)
    if missing:
        err(f"缺依赖：{', '.join(missing)}（请 pip install -r requirements.txt）")
    else:
        ok(f"依赖齐全（{len(needed)} 个核心包）")


def check_env() -> None:
    env_local = ROOT / ".env.local"
    if env_local.exists():
        ok(".env.local 存在（已被 .gitignore 排除，不入 git）")
        # 不打印 KEY 值，仅判定字段存在
        content = env_local.read_text(encoding="utf-8", errors="replace")
        if any(k in content for k in ("OPENAI_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY", "BAILIAN_API_KEY")):
            ok(".env.local 含至少一个 API Key 字段")
        else:
            warn(".env.local 中未发现 API Key 字段")
    else:
        warn(".env.local 不存在（运行时需手动 export API Key）")

    from day1_api_starter import get_llm_config
    from rag_tools import describe_embedding_config, has_embedding_api_key

    cfg = get_llm_config()
    if cfg["api_key"]:
        v = cfg["api_key"]
        masked = f"{v[:3]}***{v[-3:]} · len={len(v)}" if len(v) >= 8 else "***"
        ok(f"{cfg['api_key_env']} 已注入当前 LLM 配置（{masked}）")
    else:
        warn(f"当前 LLM 配置未注入 {cfg['api_key_env']}（LLM 调用会失败）")

    if has_embedding_api_key():
        ok(f"Embedding 配置就绪：{describe_embedding_config()}")
    else:
        warn(f"Embedding API Key 未配置：{describe_embedding_config()}（RAG ingest/query 会使用伪向量）")


def check_core_files() -> None:
    must = [
        "user_profile.md", "SOUL.md", "target_rules.md", "source_policy.md",
        "match_job.py", "plan_gen.py", "summary_tool.py", "pipeline.py",
        "rag_ingest.py", "rag_query.py", "rag_graph.py", "rag_api.py", "mcp_server.py",
        "DATA_CONTRACT.md", "applications.md", "interview_story_bank.md",
        "docs/architecture.md", "docs/archive/demo.md", "docs/archive/resume_pitch.md",
        "docs/interview_qa.md", "docs/project_one_pager.md", "docs/ethical_use.md",
        "static/index.html", "tests/test_offerclaw_core.py",
    ]
    missing = [f for f in must if not (ROOT / f).exists()]
    if missing:
        err(f"缺关键文件 {len(missing)} 个：{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
    else:
        ok(f"核心文件齐全（{len(must)} 个）")


def check_chroma() -> None:
    db = ROOT / "chroma_db"
    if not db.exists():
        warn("chroma_db/ 不存在 → 运行 `python rag_ingest.py` 建索引")
        return
    sqlite = db / "chroma.sqlite3"
    if sqlite.exists() and sqlite.stat().st_size > 1024:
        ok(f"chroma_db 索引存在（{sqlite.stat().st_size // 1024} KB）")
    else:
        warn("chroma_db 目录存在但索引为空 → 重新跑 rag_ingest.py")
        return

    try:
        import chromadb
        from rag_tools import get_collection_name

        collection_name = get_collection_name()
        client = chromadb.PersistentClient(path=str(db))
        all_cols = {c.name: c.count() for c in client.list_collections()}

        if collection_name in all_cols and all_cols[collection_name] > 0:
            ok(f"当前 collection `{collection_name}` 可用（{all_cols[collection_name]} chunks）")
        else:
            # 配置漂移检测：当前(配置)collection 缺/空，但别的 embedding 模型 collection 有数据
            others = {n: c for n, c in all_cols.items() if n != collection_name and c > 0}
            if others:
                detail = "，".join(f"`{n}`({c})" for n, c in others.items())
                warn(f"embedding 配置漂移：当前配置指向 `{collection_name}`（空/缺），"
                     f"但数据在 {detail}。说明你换了 embedding 模型却未重建索引 → "
                     f"`python rag_ingest.py --rebuild`（用新模型重建），或把 .env.local 改回有数据的模型")
            else:
                warn(f"当前 collection `{collection_name}` 为空/缺 → 运行 `python rag_ingest.py --rebuild`")
    except Exception as exc:
        warn(f"当前 collection 不可用 → {exc}")


def check_tests() -> None:
    tests_dir = ROOT / "tests"
    if not tests_dir.exists():
        err("tests/ 目录不存在")
        return
    files = list(tests_dir.glob("test_*.py"))
    if files:
        ok(f"pytest 测试文件 {len(files)} 个：{', '.join(f.name for f in files)}")
    else:
        warn("tests/ 下没有 test_*.py")


def check_gitignore() -> None:
    gi = ROOT / ".gitignore"
    if not gi.exists():
        err(".gitignore 不存在（密钥可能被推到 GitHub！）")
        return
    content = gi.read_text(encoding="utf-8", errors="replace")
    must_ignore = [".env", "chroma_db", "memory.json", "logs", "__pycache__"]
    miss = [m for m in must_ignore if m not in content]
    if miss:
        err(f".gitignore 缺规则：{', '.join(miss)}")
    else:
        ok(".gitignore 关键规则齐全")


def check_docs_consistency() -> None:
    """新增：调用 verify_docs.py 巡检 README/verification_report/project_one_pager/PROJECT_STATUS 指标口径"""
    script = ROOT / "verify_docs.py"
    if not script.exists():
        warn("verify_docs.py 缺失，跳过文档口径检查")
        return
    import subprocess
    r = subprocess.run([sys.executable, str(script), "--json"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        ok("文档指标口径一致（verify_docs 全绿）")
    elif r.returncode == 1:
        # 解析 JSON 拿 issue 数
        try:
            import json as _json
            data = _json.loads(r.stdout)
            n = len(data.get("issues", []))
            err(f"verify_docs 发现 {n} 处指标口径不一致，跑 `python verify_docs.py` 查看")
        except Exception:
            err("verify_docs 报告不一致（详情运行 `python verify_docs.py`）")
    else:
        warn(f"verify_docs 异常退出 rc={r.returncode}")


def main() -> int:
    print("=" * 60)
    print("OfferClaw doctor — 工程健康检查")
    print("=" * 60)
    for fn in [check_python, check_packages, check_env, check_core_files,
               check_chroma, check_tests, check_gitignore, check_docs_consistency]:
        try:
            fn()
        except Exception as e:
            err(f"{fn.__name__} 抛异常：{e}")

    n_ok = sum(1 for lv, _ in results if lv == "OK")
    n_warn = sum(1 for lv, _ in results if lv == "WARN")
    n_err = sum(1 for lv, _ in results if lv == "ERR")

    for lv, msg in results:
        color = {"OK": "\033[32m", "WARN": "\033[33m", "ERR": "\033[31m"}.get(lv, "")
        print(f"{color}[{lv:4}]\033[0m {msg}")

    print("-" * 60)
    print(f"汇总：{n_ok} OK · {n_warn} WARN · {n_err} ERR")
    if n_err:
        return 2
    if n_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
