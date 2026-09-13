# -*- coding: utf-8 -*-
"""knowledge_crawler.py — OfferClaw 半自动知识采集（P4）

定位（与 doc2kb 互补，不重复造轮子）：
    - doc2kb：抓取**需登录/浏览器渲染**的飞书/Notion 文档树（带层级、图片）。
    - 本工具：抓取**公开网页文章**（掘金/知乎专栏/博客等），并加一道
      **LLM 质量门**——按"大模型应用工程师"相关度 / 信息密度 / 时效打分，
      A/B 级落入 _pending 待人工确认，再用 promote 提升到正式知识库。

设计原则：
    - 复用 job_discovery.fetch_url（requests + Playwright 兜底）抓正文。
    - 复用 day1_api_starter 的 LLM 配置打分。
    - 一切落盘到 knowledge_base/_pending/web/，绝不直接进正式库（人在回路）。
    - 纯逻辑（slug / frontmatter / 评分解析 / 提升路径）可离线单测。

用法：
    python knowledge_crawler.py crawl <url>                      # 抓取 + 打分 → _pending/web/
    python knowledge_crawler.py score <pending_file.md>          # 重新打分
    python knowledge_crawler.py promote <pending_file.md> --to learning_resources
"""

import argparse
import datetime
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
PENDING_WEB_DIR = os.path.join(KB_DIR, "_pending", "web")
TARGET_ROLE = "大模型应用工程师"
VALID_SUBDIRS = {
    "career_paths", "experience_posts", "learning_resources", "project_context",
    "resume_rules", "papers",
}
SUBDIR_SOURCE_TYPE = {
    "career_paths": "career_knowledge",
    "experience_posts": "experience",
    "learning_resources": "resource",
    "project_context": "project_context",
    "resume_rules": "resume_rule",
    # 论文域独立 source_type(2026-08-09):与八股/资源隔离——计划的资源检索
    # (RESOURCE_SOURCE_TYPES)不含 paper,学术内容不会混进排期推荐;问答检索仍全域可见。
    "papers": "paper",
}
SOURCE_TYPE_TO_SUBDIR = {v: k for k, v in SUBDIR_SOURCE_TYPE.items()}


# =====================================================
# 纯逻辑（可离线单测）
# =====================================================

def slugify(text: str, maxlen: int = 40) -> str:
    """把标题/URL 压成文件名安全的 slug。"""
    text = re.sub(r"https?://", "", text or "")
    text = re.sub(r"[^\w一-鿿]+", "_", text).strip("_").lower()
    return text[:maxlen] or "untitled"


def build_frontmatter(meta: dict) -> str:
    """生成 YAML frontmatter 字符串。tags 用 JSON 数组写法。"""
    lines = ["---"]
    for k in ["title", "source_url", "crawl_date", "quality", "target_role",
              "source_type", "owner_scope", "review_status", "score_relevance", "score_density",
              "score_recency_ok", "score_reason"]:
        if k not in meta:
            continue
        v = meta[k]
        if isinstance(v, bool):
            v = "true" if v else "false"
        lines.append(f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}")
    if meta.get("tags"):
        lines.append("tags: " + json.dumps(meta["tags"], ensure_ascii=False))
    lines.append("---")
    return "\n".join(lines)


def parse_frontmatter(text: str) -> dict:
    """从一篇 .md 解析 frontmatter（简单 KV，足够本工具用）。"""
    if not text.lstrip().startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta = {}
    for ln in parts[1].splitlines():
        m = re.match(r"^(\w+):\s*(.+)$", ln.strip())
        if m:
            key, val = m.group(1), m.group(2).strip().strip('"')
            meta[key] = val
    return meta


def parse_score(llm_text: str) -> dict:
    """从 LLM 输出里鲁棒解析评分 JSON；失败给保守默认（C / 待审）。"""
    m = re.search(r"```json\s*(\{.*?\})\s*```", llm_text, re.DOTALL) or \
        re.search(r"(\{.*\})", llm_text, re.DOTALL)
    data = {}
    if m:
        try:
            data = json.loads(m.group(1))
        except Exception:
            data = {}
    grade = str(data.get("grade", "C")).upper()
    if grade not in {"A", "B", "C", "REJECT"}:
        grade = "C"
    subdir = data.get("suggested_subdir", "learning_resources")
    if subdir not in VALID_SUBDIRS:
        subdir = "learning_resources"
    return {
        "grade": grade,
        "relevance": int(data.get("relevance", 0) or 0),
        "density": int(data.get("density", 0) or 0),
        "recency_ok": bool(data.get("recency_ok", True)),
        "reason": str(data.get("reason", "") or ""),
        "suggested_subdir": subdir,
        "suggested_title": str(data.get("suggested_title", "") or ""),
    }


# =====================================================
# LLM 质量打分
# =====================================================

_SCORE_PROMPT = """你是 OfferClaw 的知识库质量审核员。请评估下面这篇网页内容对一个
正在求职「{role}」的人是否值得收进学习知识库。

只输出一个 ```json 代码块，字段：
{{
  "grade": "A|B|C|reject",   // A=高质量干货 B=有参考价值 C=一般 reject=广告/无关/低质
  "relevance": 0-10,          // 与 {role} 方向相关度
  "density": 0-10,            // 信息密度（有无具体技术/步骤）
  "recency_ok": true|false,   // 内容是否未明显过时
  "reason": "一句话理由",
  "suggested_subdir": "career_paths|experience_posts|learning_resources|project_context|resume_rules",
  "suggested_title": "一个简洁中文标题"
}}

内容（截断）：
{content}
"""


def score_content(text: str, title: str = "") -> dict:
    """调 LLM 给内容打质量分。无 key / 失败 → 返回保守 C 档。"""
    try:
        import requests
        from day1_api_starter import get_llm_config, build_zhipu_jwt, load_local_env
        load_local_env()
        cfg = get_llm_config()
        api_key = cfg["api_key"]
        if not api_key:
            return {**parse_score(""), "reason": "无 LLM key，未打分，保守置 C"}
        prompt = _SCORE_PROMPT.format(role=TARGET_ROLE, content=text[:4000])
        bearer = build_zhipu_jwt(api_key) if cfg["is_zhipu"] else api_key
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, "max_tokens": 400,
        }
        if cfg.get("reasoning_effort"):
            payload["reasoning_effort"] = cfg["reasoning_effort"]
        resp = requests.post(
            f"{cfg['api_base']}/chat/completions",
            headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
            json=payload, timeout=60,
        )
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"].get("content", "") or ""
        return parse_score(out)
    except Exception as e:
        return {**parse_score(""), "reason": f"打分失败({e})，保守置 C"}


# =====================================================
# 命令
# =====================================================

def content_stats(text: str) -> dict:
    """正文统计，供审核判断"采集是否完整/有无链接堆砌"。"""
    import re
    lines = [l for l in text.splitlines() if l.strip()]
    link_lines = sum(1 for l in lines if "http" in l or "](" in l)
    images = sum(1 for l in lines if l.strip().startswith("!["))
    return {
        "chars": len(text),
        "lines": len(lines),
        "link_pct": (link_lines * 100 // max(len(lines), 1)),
        "images": images,
    }


def content_preview(text: str, head: int = 12, tail: int = 6) -> dict:
    """取正文开头/结尾若干行，便于人工判断有没有截断、缺内容。"""
    lines = [l for l in text.splitlines() if l.strip()]
    return {
        "head": "\n".join(lines[:head]),
        "tail": "\n".join(lines[-tail:]) if len(lines) > head else "",
        "truncated_middle": max(0, len(lines) - head - tail),
    }


def redact_secrets(text: str) -> tuple:
    """把采集到的正文里**疑似密钥/令牌**打码，避免把第三方泄露的 key 入库甚至推到仓库。

    返回 (脱敏后文本, 命中数)。覆盖 OpenAI/通用 sk-、AWS AKIA、GitHub ghp_/gho_ 等常见模式。
    保守匹配，避免误伤正文。
    """
    patterns = [
        r"sk-[A-Za-z0-9_\-]{20,}",          # OpenAI / 兼容
        r"AKIA[0-9A-Z]{16}",                # AWS Access Key
        r"gh[pousr]_[A-Za-z0-9]{30,}",      # GitHub tokens
        r"AIza[0-9A-Za-z_\-]{30,}",         # Google API key
        r"xox[baprs]-[A-Za-z0-9\-]{10,}",   # Slack
    ]
    n = 0
    for pat in patterns:
        text, c = re.subn(pat, "[REDACTED_SECRET]", text)
        n += c
    return text, n


def _score_and_save(text: str, url: str, origin: str, force_keep: bool = False) -> dict:
    """对一段正文打分并存入 _pending/web/（reject 不落盘）。

    crawl（requests/Playwright）与 from-text（浏览器插件采集）共用此后处理，
    保证两条采集轨道走同一质量门 + 同一落盘格式。``origin`` 记入 tags 便于溯源。
    采集内容会先做**密钥脱敏**（第三方文档常含泄露 key），再打分落盘。

    ``force_keep=True``（用户主动上传的本地文档）：即便相关性判为 REJECT 也仍落盘
    （降级为 C 档供人工确认），尊重"用户自己选的资料"；但真正空/过短仍拒绝。

    返回里**显式带上来源 URL、本地绝对路径、采集统计、内容预览**，
    方便审核：①点 source_url 看来源质量 ②开 saved_abs 看采集是否完整。
    """
    import hashlib
    os.makedirs(PENDING_WEB_DIR, exist_ok=True)
    text, _redacted = redact_secrets(text or "")
    # 文件级内容去重(§5.1):同内容已在 候选/归档/正式库 → 不重复打分(省 LLM 调用)、不重复入候选
    dup = find_duplicate_content(text)
    if dup is not None:
        return {"status": "duplicate", "source_url": url, "grade": None,
                "existing": dup, "saved": None, "saved_abs": None,
                "next": f"内容与已有文件一致({dup['where']}:{dup['title']}),未重复入候选"}
    if not text or len(text.strip()) < 80:
        return {"status": "rejected", "source_url": url, "grade": "REJECT",
                "reason": "正文过短或为空", "saved": None, "saved_abs": None,
                "next": "内容不足，未入 _pending"}
    score = score_content(text)
    if score["grade"] == "REJECT" and not force_keep:
        return {"status": "rejected", "source_url": url, "grade": "REJECT",
                "reason": score["reason"], "saved": None, "saved_abs": None,
                "next": "内容无效/无关，已丢弃，未入 _pending"}
    if score["grade"] == "REJECT":  # force_keep：降级 C 仍落盘
        score = {**score, "grade": "C",
                 "suggested_subdir": score.get("suggested_subdir") or "learning_resources",
                 "suggested_title": score.get("suggested_title") or "",
                 "reason": f"用户主动上传（原判：{score.get('reason','')}）"}

    title = score["suggested_title"] or slugify(url, 30)
    today = datetime.date.today().isoformat()
    uhash = hashlib.md5((url or text[:50]).encode("utf-8")).hexdigest()[:6]
    meta = {
        "title": title, "source_url": url or "(browser-capture)", "crawl_date": today,
        "quality": score["grade"], "target_role": TARGET_ROLE,
        "source_type": SUBDIR_SOURCE_TYPE[score["suggested_subdir"]],
        "review_status": "pending",
        "score_relevance": score["relevance"], "score_density": score["density"],
        "score_recency_ok": score["recency_ok"], "score_reason": score["reason"],
        "tags": [TARGET_ROLE, origin],
    }
    fname = f"{today}_{slugify(title)}_{uhash}.md"
    path = os.path.join(PENDING_WEB_DIR, fname)
    body = f"\n# {title}\n\n> 来源：{url or '(browser-capture)'}\n> 建议归类：{score['suggested_subdir']}\n\n{text.strip()}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_frontmatter(meta) + "\n" + body)
    return {
        "status": "ok",
        "title": title,
        "source_url": url or "(browser-capture)",          # ① 审核：来源质量
        "saved": os.path.relpath(path, BASE_DIR),
        "saved_abs": path,                                  # ② 审核：打开看采集质量
        "grade": score["grade"], "relevance": score["relevance"],
        "density": score["density"], "reason": score["reason"],
        "suggested_subdir": score["suggested_subdir"],
        "stats": content_stats(text),
        "preview": content_preview(text),
        "next": f"审核：先看 source_url 判断来源；再开 saved_abs 看采集是否完整。"
                f"满意后 promote --to {score['suggested_subdir']}",
    }


def stage_personal_memory(text: str, title: str, kind: str,
                          source_url: str = "(用户提供)") -> dict:
    """把个人项目/简历规则放入待确认区，不评分、不直接进入正式知识库。

    个人材料的相关性由用户意图决定，不应用通用网页质量门拒收；但仍执行空内容、
    密钥脱敏和内容去重。真正持久化/增量索引继续走 ``cmd_promote`` 人工确认。
    """
    import hashlib
    if kind not in {"project_context", "resume_rules"}:
        return {"status": "error", "error": "kind 仅支持 project_context/resume_rules"}
    text, redacted = redact_secrets(text or "")
    if len(text.strip()) < 80:
        return {"status": "rejected", "reason": "个人材料正文过短（至少 80 字）"}
    dup = find_duplicate_content(text)
    if dup is not None:
        return {"status": "duplicate", "existing": dup, "saved": None,
                "next": f"内容已存在({dup['where']}:{dup['title']})"}
    os.makedirs(PENDING_WEB_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()
    clean_title = (title or ("个人项目材料" if kind == "project_context" else "个人简历规则")).strip()
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
    path = os.path.join(PENDING_WEB_DIR,
                        f"{today}_{slugify(clean_title)}_{digest}.md")
    meta = {
        "title": clean_title,
        "source_url": source_url or "(用户提供)",
        "crawl_date": today,
        "quality": "A",
        "target_role": TARGET_ROLE,
        "source_type": SUBDIR_SOURCE_TYPE[kind],
        "owner_scope": "personal",
        "review_status": "pending",
        "score_relevance": 10,
        "score_density": 0,
        "score_recency_ok": True,
        "score_reason": "用户主动提供的个人材料，待人工确认后持久化",
        "tags": [TARGET_ROLE, "个人记忆", kind],
    }
    body = (f"\n# {clean_title}\n\n> 来源：{source_url or '(用户提供)'}\n"
            f"> 个人记忆类型：{kind}\n\n{text.strip()}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_frontmatter(meta) + "\n" + body)
    return {
        "status": "ok", "title": clean_title, "source_url": source_url,
        "saved": os.path.relpath(path, BASE_DIR), "saved_abs": path,
        "grade": "A", "relevance": 10, "density": 0,
        "reason": "个人材料待确认", "suggested_subdir": kind,
        "stats": content_stats(text), "preview": content_preview(text),
        "redacted_secrets": redacted,
        "next": "请预览并确认后再持久化和增量索引",
    }


# =====================================================
# GitHub 仓库采集（抓真正的教程内容，而非只抓 README）
# =====================================================

import re as _re

_GH_SKIP_NAMES = {"license", "license.md", "contributing.md", "code_of_conduct.md",
                  "security.md", "_sidebar.md", "_coverpage.md", "_navbar.md", ".nojekyll",
                  # README/导航是目录而非正文，跳过避免噪声（正文在 chapterX/）
                  "readme.md", "readme_en.md", "readme_cn.md", "readme_zh.md",
                  "_sidebar_en.md", "_navbar_en.md", "_coverpage_en.md"}
# 跳过：校验目录、英文镜像目录（中英双语仓库的 /en/ 与中文正文重复）
_GH_SKIP_PATH = (".ipynb_checkpoints", "node_modules/", "/.github/", "/test/", "/tests/",
                 "/en/", "/english/", "/.obsidian/")
_GH_MAX_FILES = int(os.environ.get("GH_MAX_FILES", "60"))
_GH_MAX_CHARS = int(os.environ.get("GH_MAX_CHARS", "600000"))
_HAN = _re.compile(r"[一-鿿]")  # 判定文件名是否含中文，用于中英去重


def parse_github_repo(url: str):
    """从各种 GitHub URL 解析出 (owner, repo)；非 GitHub 返回 None。

    支持 github.com/owner/repo[/...]、raw.githubusercontent.com/owner/repo/branch/...
    """
    m = _re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url) or \
        _re.search(r"raw\.githubusercontent\.com/([^/\s]+)/([^/\s]+)", url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    repo = repo.replace(".git", "")
    if owner in {"raw", "gist"}:
        return None
    return owner, repo


def is_github_repo_url(url: str) -> bool:
    """是否是"仓库级"URL（应抓全仓内容，而非单文件）。

    True：github.com/owner/repo 根、或指向 README 的 raw URL。
    False：raw URL 指向某个具体非 README 文件（单文件抓取即可）。
    """
    if not parse_github_repo(url):
        return False
    if "raw.githubusercontent.com" in url:
        return url.rstrip("/").lower().endswith("readme.md")
    # github.com/owner/repo[ 或 /tree/... ]，但不是 /blob/具体文件
    if "/blob/" in url:
        return url.rstrip("/").lower().endswith("readme.md")
    return True


def rewrite_image_links(content: str, owner: str, repo: str, branch: str, file_path: str) -> str:
    """把仓库内 markdown/html 图片的**相对路径**重写成 GitHub raw 绝对 URL，
    使采集后的资源在本地/UI 打开时图片仍能正常显示。

    相对路径相对于该文件所在目录解析（含 ../）；已是 http/data/锚点的保持不变。
    """
    import posixpath
    from urllib.parse import quote
    base_dir = posixpath.dirname(file_path)
    raw_prefix = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"

    def to_abs(rel: str) -> str:
        rel = rel.strip().strip('"').strip("'")
        if rel.startswith(("http://", "https://", "data:", "//", "#", "mailto:")):
            return rel
        resolved = posixpath.normpath(posixpath.join(base_dir, rel))
        return raw_prefix + quote(resolved)

    # markdown ![alt](url "可选title")
    def md_img(m):
        return f"![{m.group(1)}]({to_abs(m.group(2))}{m.group(3) or ''})"
    content = _re.sub(r'!\[([^\]]*)\]\(([^)\s]+)(\s+[^)]*)?\)', md_img, content)
    # html <img ... src="url" ...>
    def html_img(m):
        return m.group(0).replace(m.group(1), to_abs(m.group(1)), 1)
    content = _re.sub(r'<img[^>]+src=["\']([^"\']+)["\']', html_img, content)
    return content


def _ipynb_to_text(raw_json: str) -> str:
    """从 .ipynb（JSON）提取 markdown + code 单元文本，丢弃输出。"""
    try:
        nb = json.loads(raw_json)
    except Exception:
        return ""
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") not in ("markdown", "code"):
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        if not text.strip():
            continue
        if cell.get("cell_type") == "code":
            out.append("```python\n" + text + "\n```")
        else:
            out.append(text)
    return "\n\n".join(out)


def _gh_list_content_files(owner: str, repo: str) -> tuple:
    """返回 (branch, [content_paths])。content = .md/.ipynb，过滤样板/校验/重复目录。"""
    import requests
    h = {"User-Agent": "offerclaw-kb", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=20, headers=h)
    if r.status_code == 403:
        raise RuntimeError("GitHub API 限流（未认证 60/小时），稍后再试")
    r.raise_for_status()
    branch = r.json().get("default_branch", "main")
    t = requests.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
                     timeout=30, headers=h)
    t.raise_for_status()
    blobs = [x["path"] for x in t.json().get("tree", []) if x.get("type") == "blob"]

    def keep(p: str) -> bool:
        low = p.lower()
        if not low.endswith((".md", ".markdown", ".ipynb")):
            return False
        if any(s in low for s in _GH_SKIP_PATH):
            return False
        base = os.path.basename(low)
        if base in _GH_SKIP_NAMES:
            return False
        stem = base.rsplit(".", 1)[0]
        if stem.endswith(("_en", "-en", "_english", "-english", ".en")):
            return False  # 英文镜像文件（README_EN / chapter_en 等）
        return True

    files = [p for p in blobs if keep(p)]
    # 去重①：若存在 docs/ 内容目录，优先只用 docs/（避免 docs 与 notebook 重复章节）
    docs = [p for p in files if p.lower().startswith("docs/")]
    if len(docs) >= 3:
        files = docs
    # 去重②：中英双语仓库同目录常并存「中文文件名章节」与「英文标题章节」，
    # 若某目录有含中文的文件名，则剔除同目录纯 ASCII 文件名的兄弟（视作英文重复版）。
    # 纯中文项目（章节名如 01_xxx.md，无中文兄弟）不受影响。
    by_dir: dict = {}
    for p in files:
        by_dir.setdefault(os.path.dirname(p), []).append(p)
    kept = []
    for _d, ps in by_dir.items():
        has_zh = any(_HAN.search(os.path.basename(x)) for x in ps)
        for x in ps:
            if has_zh and not _HAN.search(os.path.basename(x)):
                continue
            kept.append(x)
    return branch, sorted(kept)


def fetch_repo_text(url: str) -> dict:
    """抓取 GitHub 仓库的内容文件（.md/.ipynb）并拼成一篇长文。

    返回 {status:"ok", text, repo, branch, files_captured, files_total_found}
    或 {status:"error", error}。供 crawl_repo（入知识库）与
    简历项目分析（/api/resume/project）等复用。
    """
    import requests
    parsed = parse_github_repo(url)
    if not parsed:
        return {"status": "error", "error": "不是可识别的 GitHub 仓库 URL"}
    owner, repo = parsed
    branch, files = _gh_list_content_files(owner, repo)
    if not files:
        return {"status": "error", "error": "仓库内未找到 .md/.ipynb 内容文件"}

    h = {"User-Agent": "offerclaw-kb"}
    parts, used, total = [], [], 0
    for path in files:
        if len(used) >= _GH_MAX_FILES or total >= _GH_MAX_CHARS:
            break
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        try:
            rr = requests.get(raw_url, timeout=30, headers=h)
            if rr.status_code != 200:
                continue
            content = rr.text
            if path.lower().endswith(".ipynb"):
                content = _ipynb_to_text(content)
            # 相对图片路径 → GitHub raw 绝对 URL，保证图片可正常显示
            content = rewrite_image_links(content, owner, repo, branch, path)
            content = content.strip()
            if len(content) < 40:
                continue
            parts.append(f"\n\n## {path}\n\n{content}")
            used.append(path)
            total += len(content)
        except Exception:
            continue

    if not parts:
        return {"status": "error", "error": "内容文件抓取失败（网络/限流）"}

    repo_url = f"https://github.com/{owner}/{repo}"
    header = (f"# {repo} —— GitHub 仓库内容采集\n\n"
              f"> 仓库：{repo_url}（分支 {branch}）\n"
              f"> 已采集 {len(used)} 个内容文件，共 {total} 字\n")
    return {
        "status": "ok", "text": header + "".join(parts), "repo": repo_url,
        "branch": branch, "files_captured": len(used), "files_total_found": len(files),
    }


def crawl_repo(url: str) -> dict:
    """抓取整个 GitHub 仓库的**真实内容文件**（章节 .md / notebook），
    拼成一篇知识文档 → 打分 → _pending/web/。修复"只抓 README 简介"的缺陷。
    """
    fetched = fetch_repo_text(url)
    if fetched.get("status") != "ok":
        return fetched
    repo_url = fetched["repo"]
    result = _score_and_save(fetched["text"], repo_url, origin="github仓库采集")
    if result.get("status") == "ok":
        result["repo"] = repo_url
        result["files_captured"] = fetched["files_captured"]
        result["files_total_found"] = fetched["files_total_found"]
    return result


def cmd_crawl(url: str) -> dict:
    """轨道1：抓公开 URL → 打分 → _pending/web/。

    若 URL 是 GitHub 仓库（或其 README），**自动改走整仓内容采集**（crawl_repo），
    避免只抓到 README 简介、漏掉真正的教程章节。
    """
    if is_github_repo_url(url):
        return crawl_repo(url)
    from job_discovery import fetch_url
    return _score_and_save(fetch_url(url), url, origin="web采集")


def cmd_from_text(url: str, text_file: str) -> dict:
    """轨道2：把浏览器插件（Claude-in-Chrome / doc2kb）从登录态/反爬页采集到的
    正文文本接入同一质量门 + 审核流水线。

    text_file：已采集正文的本地文件（.md/.txt）。url：原始页面地址（供溯源）。
    用法：浏览器插件读飞书/知乎等已渲染页面 → 存成文件 → from-text → _pending → 审核 → promote。
    """
    src = text_file if os.path.isabs(text_file) else os.path.join(BASE_DIR, text_file)
    if not os.path.exists(src):
        return {"status": "error", "error": f"文件不存在：{text_file}"}
    with open(src, encoding="utf-8") as f:
        text = f.read()
    return _score_and_save(text, url, origin="浏览器采集")


def cmd_promote(pending_file: str, to_subdir: str, ingest: bool = False) -> dict:
    """把审核通过的 _pending 文件提升到正式知识库子目录。

    ``ingest=True`` 时**增量**入库（rag_ingest --add，不重建、不影响原有内容），
    提升后立即可检索；否则只移动文件，由调用方稍后增量入库。
    """
    if to_subdir not in VALID_SUBDIRS:
        return {"status": "error", "error": f"--to 必须是 {sorted(VALID_SUBDIRS)} 之一"}
    src = pending_file if os.path.isabs(pending_file) else os.path.join(BASE_DIR, pending_file)
    if not os.path.exists(src):
        return {"status": "error", "error": f"文件不存在：{pending_file}"}
    with open(src, encoding="utf-8") as f:
        content = f.read()
    meta = parse_frontmatter(content)
    # 提升时修正 review_status / source_type，与目标子目录对齐
    content = re.sub(r"review_status:\s*\"?pending\"?", 'review_status: "approved"', content)
    expected_st = SUBDIR_SOURCE_TYPE[to_subdir]
    if "source_type:" in content:
        content = re.sub(r"source_type:\s*\"?[^\"\n]+\"?", f'source_type: "{expected_st}"', content)
    owner_scope = "personal" if to_subdir in {
        "experience_posts", "project_context", "resume_rules"
    } else "curated"
    if "owner_scope:" in content:
        content = re.sub(r"owner_scope:\s*\"?[^\"\n]+\"?", f'owner_scope: "{owner_scope}"', content)
    else:
        content = re.sub(r"(source_type:\s*[^\n]+\n)",
                         rf'\1owner_scope: "{owner_scope}"\n', content, count=1)
    dst_dir = os.path.join(KB_DIR, to_subdir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    with open(dst, "w", encoding="utf-8") as f:
        f.write(content)
    os.remove(src)

    out = {
        "status": "ok",
        "promoted_to": os.path.relpath(dst, BASE_DIR),
        "source_type": expected_st,
        "title": meta.get("title", ""),
    }
    if ingest:
        import subprocess
        rel = os.path.relpath(dst, BASE_DIR)
        proc = subprocess.run(
            [os.path.join(BASE_DIR, ".venv/bin/python"), "rag_ingest.py", "--add", rel],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=3600,  # 大教程文件本地 embedding 慢，300s 会半途中断
        )
        tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        out["ingest"] = "ok" if proc.returncode == 0 else "failed"
        out["ingest_detail"] = tail[0]
        out["next"] = "已增量入库，可直接检索（未重建、未影响原有内容）"
    else:
        out["next"] = "增量入库：python rag_ingest.py --add " + os.path.relpath(dst, BASE_DIR)
    return out


def cmd_score(pending_file: str) -> dict:
    """对已存在的 _pending 文件重新打分（只读，不改文件）。"""
    src = pending_file if os.path.isabs(pending_file) else os.path.join(BASE_DIR, pending_file)
    if not os.path.exists(src):
        return {"status": "error", "error": f"文件不存在：{pending_file}"}
    with open(src, encoding="utf-8") as f:
        content = f.read()
    # 去掉 frontmatter 再评分
    body = content.split("---", 2)[-1] if content.lstrip().startswith("---") else content
    score = score_content(body)
    return {"status": "ok", "file": pending_file, **score}


def cmd_review(pending_file: str) -> dict:
    """审核界面：把一篇待审文件的【来源 / 本地路径 / 评分 / 采集统计 / 内容预览】一次性给出，
    方便人工判断 ①来源质量（开 source_url）②采集质量（开 saved_abs，看有没有缺/截断）。只读不改。
    """
    src = pending_file if os.path.isabs(pending_file) else os.path.join(BASE_DIR, pending_file)
    if not os.path.exists(src):
        return {"status": "error", "error": f"文件不存在：{pending_file}"}
    with open(src, encoding="utf-8") as f:
        content = f.read()
    meta = parse_frontmatter(content)
    body = content.split("---", 2)[-1] if content.lstrip().startswith("---") else content
    return {
        "status": "ok",
        "title": meta.get("title", ""),
        "source_url": meta.get("source_url", ""),       # ① 来源：点开判断文章质量
        "saved_abs": os.path.abspath(src),              # ② 本地：打开判断采集质量
        "quality": meta.get("quality", ""),
        "score_relevance": meta.get("score_relevance", ""),
        "score_density": meta.get("score_density", ""),
        "score_reason": meta.get("score_reason", ""),
        "suggested_subdir": SOURCE_TYPE_TO_SUBDIR.get(meta.get("source_type", ""), "learning_resources"),
        "stats": content_stats(body),
        "preview": content_preview(body),
    }


# 工具/索引类文件名，不算审核候选
_NON_CANDIDATE = {"readme.md", "preview.md", "tree.md", "index.md"}


def cmd_list_pending() -> dict:
    """列出待审的"内容候选"（有 source_url 的真实采集文件），方便逐个 review。

    跳过 doc2kb 等工具产生的 README/PREVIEW/TREE/审计等非内容文件。
    """
    items = []
    for root, dirs, files in os.walk(os.path.join(KB_DIR, "_pending")):
        # `_` 前缀子目录 = 归档区(如 _pending/_archived/):审核后判为"已在库副本"的候选
        # 移进去即从待审列表消失,文件保留可追溯(knowledge_base 不入 git,不做硬删除)。
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        for fn in sorted(files):
            if not fn.endswith(".md") or fn.startswith("_"):
                continue
            if fn.lower() in _NON_CANDIDATE or "formula_audit" in fn.lower():
                continue
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8") as f:
                meta = parse_frontmatter(f.read())
            if not meta.get("source_url"):  # 无来源的不是采集候选
                continue
            items.append({
                "title": meta.get("title", fn),
                "source_url": meta.get("source_url", ""),
                "quality": meta.get("quality", ""),
                "saved_abs": os.path.abspath(p),
            })
    return {"status": "ok", "count": len(items), "items": items}


def _out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="OfferClaw 半自动知识采集")
    sub = parser.add_subparsers(dest="cmd")
    p_crawl = sub.add_parser("crawl"); p_crawl.add_argument("url")
    p_ft = sub.add_parser("from-text")  # 浏览器插件采集的正文接入
    p_ft.add_argument("url"); p_ft.add_argument("--file", required=True)
    p_score = sub.add_parser("score"); p_score.add_argument("file")
    p_review = sub.add_parser("review"); p_review.add_argument("file")
    sub.add_parser("list")
    p_prom = sub.add_parser("promote"); p_prom.add_argument("file")
    p_prom.add_argument("--to", required=True, choices=sorted(VALID_SUBDIRS))
    p_prom.add_argument("--ingest", action="store_true", help="提升后立即增量入库（不重建）")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.cmd == "crawl":
        _out(cmd_crawl(args.url))
    elif args.cmd == "from-text":
        _out(cmd_from_text(args.url, args.file))
    elif args.cmd == "score":
        _out(cmd_score(args.file))
    elif args.cmd == "review":
        _out(cmd_review(args.file))
    elif args.cmd == "list":
        _out(cmd_list_pending())
    elif args.cmd == "promote":
        _out(cmd_promote(args.file, args.to, ingest=args.ingest))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


# =====================================================
# 审核辅助(2026-08-08):两条入库途径统一为「落本地 → UI 预览审核 → 确认入库」
#   途径一:文件夹直投(用户把 .md 直接丢进 knowledge_base 子目录)→ list_unindexed 扫出
#   途径二:URL 抓取 / UI 上传 → _pending/web 候选
# 共用 safe_kb_file / read_kb_preview 做卡内本地阅读,审核后再入库。
# =====================================================

def _norm_title(s: str) -> str:
    """标题归一(去空白/标点/大小写),供「疑似已在库」比对。"""
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def safe_kb_file(rel: str) -> str:
    """把 knowledge_base 相对路径(可带 knowledge_base/ 前缀)解析为绝对路径。

    越界(../ 或绝对路径逃逸)/不存在/非文本一律 ValueError——预览与入库的唯一路径入口。
    """
    rel = (rel or "").strip().lstrip("/")
    if rel.startswith("knowledge_base/"):
        rel = rel[len("knowledge_base/"):]
    if not rel:
        raise ValueError("路径为空")
    abs_p = os.path.realpath(os.path.join(KB_DIR, rel))
    kb_root = os.path.realpath(KB_DIR)
    if not abs_p.startswith(kb_root + os.sep):
        raise ValueError("路径越界:仅允许 knowledge_base 子树内的文件")
    if not os.path.isfile(abs_p):
        raise ValueError(f"文件不存在:{rel}")
    if os.path.splitext(abs_p)[1].lower() not in (".md", ".txt", ".markdown"):
        raise ValueError("仅支持 .md / .txt 预览")
    return abs_p


def read_kb_preview(rel: str, max_chars: int = 60000) -> dict:
    """读取候选/库内文件供 UI 卡内审核阅读(超长截断并明示)。"""
    abs_p = safe_kb_file(rel)
    with open(abs_p, encoding="utf-8", errors="replace") as f:
        text = f.read()
    meta = parse_frontmatter(text)
    return {
        "rel": os.path.relpath(abs_p, BASE_DIR),
        "abs": abs_p,
        "title": (meta.get("title") or "").strip() or os.path.basename(abs_p),
        "chars": len(text),
        "truncated": len(text) > max_chars,
        "content": text[:max_chars],
    }


def _walk_formal_kb():
    """遍历正式库(排除 _pending / assets / 隐藏目录 / _ 前缀文件)的 .md 文件。"""
    for root, dirs, files in os.walk(KB_DIR):
        dirs[:] = [d for d in dirs
                   if d not in ("_pending", "assets", "papers") and not d.startswith(".")]
        # papers/ 由论文域集合(kb_paper_e5_v1)独立治理,不进主库"未入库"扫描
        for fn in sorted(files):
            if fn.endswith(".md") and not fn.startswith("_"):
                yield os.path.join(root, fn), fn


def list_unindexed(indexed_sources: set) -> list:
    """文件夹直投轨道:正式库里「磁盘上有、索引里没有」的文件清单(待审核入库)。"""
    out = []
    for abs_p, fn in _walk_formal_kb():
        if fn in indexed_sources:
            continue
        rel = os.path.relpath(abs_p, KB_DIR).replace(os.sep, "/")
        out.append({
            "file": fn,
            "rel": rel,
            "subdir": rel.split("/", 1)[0] if "/" in rel else "",
            "chars": os.path.getsize(abs_p) if os.path.exists(abs_p) else 0,
        })
    return out


def formal_kb_title_marks() -> set:
    """正式库全部文件的 归一化文件名+frontmatter 标题 集合,供候选标记「疑似已在库」。"""
    marks = set()
    for abs_p, fn in _walk_formal_kb():
        marks.add(_norm_title(os.path.splitext(fn)[0]))
        try:
            with open(abs_p, encoding="utf-8", errors="replace") as f:
                head = f.read(600)
        except OSError:
            continue
        m = re.search(r'^title:\s*"?([^"\n]+)"?', head, re.M)
        if m:
            marks.add(_norm_title(m.group(1)))
    return marks


def flag_existing_in_kb(items: list) -> list:
    """给候选清单打「疑似已在库」标记(同名/同标题即命中,答'这 88 个是不是重复暂存')。"""
    marks = formal_kb_title_marks()
    for it in items:
        t = _norm_title(it.get("title", ""))
        it["existing_in_kb"] = bool(t) and t in marks
    return items


def extract_text_for_kb(filename: str, raw: bytes) -> str:
    """PDF / Word 上传 → 纯文本(供打分与候选入库)。抽不出有效文字抛 ValueError。

    诚实边界:PDF 只抽文字层——扫描版(图片型)PDF 无文字层,明确拒绝并提示,
    不做 OCR(避免"接受了却入库空壳/乱码"的静默劣化);Word 仅支持 .docx。
    """
    import io
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ValueError("缺 pypdf 依赖:pip install pypdf") from e
        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise ValueError("PDF 已加密,无法抽取文字")
            pages = [(p.extract_text() or "") for p in reader.pages]
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"PDF 解析失败:{e}") from e
        text = "\n\n".join(t.strip() for t in pages if t.strip())
        if len(text.strip()) < 30:
            raise ValueError("PDF 未抽取到有效文字——可能是扫描版(图片型),暂不支持 OCR;"
                             "请先转成 md/txt 再上传")
        return text
    if ext == ".docx":
        try:
            from docx import Document
        except ImportError as e:
            raise ValueError("缺 python-docx 依赖:pip install python-docx") from e
        try:
            doc = Document(io.BytesIO(raw))
        except Exception as e:
            raise ValueError(f"Word 解析失败:{e}(仅支持 .docx,旧版 .doc 请先另存为 .docx)") from e
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # 表格句子化(入库指导 §9.2 第 1 层采纳):首行为表头时,每个数据行生成
        # "表头: 值；表头: 值。"的自含句——行级自含使任意分块边界都不丢表头语义,
        # 是"长表拆分须重复表头"的更强形式。单行表无表头语义,保持竖线拼接。
        for ti, table in enumerate(doc.tables, 1):
            rows = [[c.text.strip() for c in r.cells] for r in table.rows]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            if len(rows) >= 2:
                headers = [h or f"列{i + 1}" for i, h in enumerate(rows[0])]
                parts.append(f"【表格{ti}】表头：{'、'.join(headers)}")
                for r in rows[1:]:
                    pairs = [f"{headers[i] if i < len(headers) else f'列{i + 1}'}: {v}"
                             for i, v in enumerate(r) if v]
                    if pairs:
                        parts.append("；".join(pairs) + "。")
            else:
                parts.append(" | ".join(v for v in rows[0] if v))
        text = "\n\n".join(parts)
        if len(text.strip()) < 30:
            raise ValueError("Word 未抽取到有效文字")
        return text
    raise ValueError(f"不支持的扩展名:{ext}(PDF/Word 之外请用 .md/.txt)")


# =====================================================
# 文件级内容去重(2026-08-08,入库指导文档 §5.1 第 1 层采纳)
# =====================================================

def _content_fingerprint(text: str) -> str:
    """正文内容指纹:剥 frontmatter 后做空白归一化再 sha256。

    空白归一化让"同一内容重新排版/换行不同"仍判重复;不做更激进的归一
    (如去标点),避免把改写过的内容误判为重复。
    """
    import hashlib
    body = text or ""
    if body.lstrip().startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
    normalized = re.sub(r"\s+", " ", body).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def find_duplicate_content(text: str) -> dict | None:
    """在 候选区(_pending 全树,含 _archived) + 正式子目录 中查同内容文件。

    返回 {"rel", "where", "title"} 或 None。单用户本地规模(百级文件)全量
    扫描哈希 <1s,不建持久登记表(那是指导文档第 3 层的产品级形态)。
    """
    target = _content_fingerprint(text)
    zones = []
    pend = os.path.join(KB_DIR, "_pending")
    if os.path.isdir(pend):
        zones.append((pend, "pending"))
    zones.append((KB_DIR, "formal"))
    seen_formal_skip = {os.path.realpath(pend)} if os.path.isdir(pend) else set()
    for root_dir, where in zones:
        for root, dirs, files in os.walk(root_dir):
            if where == "formal":
                dirs[:] = [d for d in dirs if d not in ("assets",) and not d.startswith(".")
                           and os.path.realpath(os.path.join(root, d)) not in seen_formal_skip]
            else:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fn in files:
                if not fn.endswith(".md") or fn.startswith("_") or fn.lower() == "readme.md":
                    continue
                p = os.path.join(root, fn)
                try:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        raw = f.read()
                except OSError:
                    continue
                if _content_fingerprint(raw) == target:
                    meta = parse_frontmatter(raw)
                    rel = os.path.relpath(p, BASE_DIR)
                    zone = where
                    if where == "pending" and f"{os.sep}_archived{os.sep}" in p:
                        zone = "archived"
                    return {"rel": rel, "where": zone,
                            "title": meta.get("title", fn)}
    return None


def extract_text_structured(filename: str, raw: bytes) -> str:
    """Docling 结构化解析(opt-in,2026-08-09):PDF/Word → Markdown(标题层级+表格)。

    与文字层引擎(extract_text_for_kb)的关系:A/B 实测(docs/rag_eval/ingestion/
    docling_ab.json)显示内容零回退、真实论文表格结构大胜,但速度慢一个量级且依赖
    本地版面模型——故 **默认仍走文字层,论文/含表格文档由用户勾选本引擎**。
    依赖缺失/模型未就绪时抛 ValueError(带安装指引),绝不静默降级。
    """
    import tempfile
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx"):
        raise ValueError(f"结构化解析仅支持 .pdf/.docx,收到 {ext}")
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        raise ValueError("结构化解析需要 docling:pip install docling"
                         "(首次使用还需下载版面模型,见 requirements.txt 注释)") from e
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        # 富化档:公式→LaTeX 文本(可检索)+ 导出图片供 qwen-vl 识别;
        # 富化模型缺失/失败自动降到基础档(fail-soft,只降增强不降正文)。
        doc = None
        try:
            # 公式富化(→LaTeX)实测在 CPU 上一篇 5 页论文 >15 分钟——自动路由里
            # 默认关闭(KB_PDF_FORMULA=1 显式开启,适合挂机/强机器);图片导出恒开
            # (qwen-vl 每图秒级,成本可控)。诚实边界:默认档公式仍会丢失。
            _formula = os.environ.get("KB_PDF_FORMULA", "0") == "1"
            rich = PdfPipelineOptions(do_ocr=False, do_table_structure=True,
                                      do_formula_enrichment=_formula,
                                      generate_picture_images=True, images_scale=2.0)
            conv = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=rich)})
            doc = conv.convert(tmp).document
        except Exception:
            doc = None
        if doc is None:
            opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
            conv = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
            try:
                doc = conv.convert(tmp).document
            except Exception as e:
                raise ValueError(f"结构化解析失败:{str(e)[:120]}(可改用默认文字层引擎重试)") from e
        md = doc.export_to_markdown()
        md += _picture_captions_appendix(doc)
    finally:
        os.unlink(tmp)
    if len(md.strip()) < 30:
        raise ValueError("结构化解析未得到有效文本——可能是扫描版(无文字层),暂不支持 OCR")
    return md


def _picture_captions_appendix(doc, cap: int = 12, min_px: int = 96) -> str:
    """把 Docling 抽出的图片交给 qwen-vl 生成"描述+OCR"附录(检索可命中图中信息)。

    诚实边界:附录形式(不保留图文位置关系);仅取前 cap 张、过滤 <min_px 的装饰小图;
    无 VL key / 调用失败逐图静默跳过,绝不阻断正文入库。
    """
    try:
        pics = list(getattr(doc, "pictures", []) or [])[:cap]
        if not pics:
            return ""
        import base64
        import io
        from image_caption import _vl_caption
        lines = []
        for i, pic in enumerate(pics, 1):
            try:
                img = pic.get_image(doc)
                if img is None or min(img.size) < min_px:
                    continue
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                cap_text = _vl_caption(uri)
                if cap_text:
                    lines.append(f"【图{i}】{cap_text}")
            except Exception:
                continue
        if not lines:
            return ""
        return ("\n\n## 图片内容(qwen-vl 自动识别,附录形式)\n\n"
                + "\n\n".join(lines) + "\n")
    except Exception:
        return ""
