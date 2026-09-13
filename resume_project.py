# -*- coding: utf-8 -*-
"""resume_project.py — 项目 → 简历项目经历段 生成器。

用户流程（用户定义）：
- 用户提供项目素材（三选一）：①项目仓库地址（GitHub 公开仓库，自动抓 README/docs）
  ②项目介绍 .md/.txt 文件 ③直接粘贴项目介绍文本；
- OfferClaw 分析项目要点（定位/技术栈/难点/量化结果），按从用户简历材料中
  **学习到的格式模式**输出可直接粘贴进简历的项目经历段。

模板学习（resume_templates/ 目录，个人材料不入 git）：
- **写作指导**（文件名含 note/写法/指导）→ 作为写作原则注入；
- **真实简历范例**（其余 .md）→ 抽取其中的「项目经历」条目块作为格式范例注入；
- 以上材料存在时，按下方 _LEARNED_PATTERN（从用户材料蒸馏出的结构）严格输出；
  目录为空时退回内置默认三段式。
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "resume_templates")

_MAX_MATERIAL_CHARS = 16000     # 项目素材注入上限
_MAX_GUIDANCE_CHARS = 2800      # 写作指导注入上限
_MAX_EXAMPLE_CHARS = 6000       # 范例条目合计注入上限

_GUIDANCE_HINTS = ("note", "notes", "写法", "指导", "guide", "writing")


def _project_evidence_title(content: str, fallback: str) -> str:
    match = re.search(r"^#{1,4}\s+(.+)$", str(content or ""), re.M)
    return (match.group(1).strip() if match else fallback)[:120]


def load_approved_project_evidence(*, base_dir: str | None = None,
                                   resume_draft_dir: str | None = None,
                                   max_items: int = 8,
                                   max_chars: int = 24000) -> list[dict]:
    """Load project sources that already passed a human approval boundary.

    Approved project-context documents and approved project-section snapshots are
    separate evidence types. Plain files in the draft or pending directories are
    deliberately ignored because their presence does not prove approval.
    """
    root = Path(base_dir or BASE_DIR).resolve()
    draft_root = Path(resume_draft_dir or root / "resume_drafts").resolve()
    candidates: list[dict] = []

    try:
        from memory_store import MemoryStore

        memory_store = MemoryStore()
        events = memory_store.list_events(
            limit=300, kind="resume_artifact_changed", include_archived=False,
        )
        for event in reversed(events):
            if event.get("action") != "approved":
                continue
            saved_path = str(event.get("saved_path") or "").strip()
            if not saved_path:
                continue
            resolved = (Path(saved_path).resolve() if Path(saved_path).is_absolute()
                        else (root / saved_path).resolve())
            try:
                resolved.relative_to(draft_root / "projects")
            except ValueError:
                continue
            snapshot = memory_store.get_snapshot(str(event.get("snapshot_id") or ""))
            content = str((snapshot or {}).get("content") or "").strip()
            if not content:
                continue
            artifact_id = str(event.get("entity_id") or "").strip()
            candidates.append({
                "evidence_ref": f"project_section:{artifact_id or hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}",
                "source_type": "approved_project_section",
                "title": _project_evidence_title(content, resolved.stem),
                "content": content,
                "source_path": saved_path,
                "approved_at": str(event.get("occurred_at") or ""),
            })
    except Exception:
        # Approved knowledge sources below remain usable when memory storage is
        # temporarily unavailable; callers still receive only explicit approvals.
        pass

    project_dir = root / "knowledge_base" / "project_context"
    if project_dir.is_dir():
        from knowledge_crawler import parse_frontmatter

        for path in sorted(project_dir.rglob("*.md"), key=lambda item: item.stat().st_mtime,
                           reverse=True):
            try:
                content = path.read_text(encoding="utf-8").strip()
                meta = parse_frontmatter(content)
            except OSError:
                continue
            if (meta.get("review_status") != "approved"
                    or meta.get("source_type") != "project_context"):
                continue
            relative = path.relative_to(root).as_posix()
            candidates.append({
                "evidence_ref": "project_context:" + hashlib.sha256(
                    relative.encode("utf-8")).hexdigest()[:16],
                "source_type": "approved_project_source",
                "title": str(meta.get("title") or _project_evidence_title(content, path.stem)),
                "content": content,
                "source_path": relative,
                "approved_at": str(meta.get("crawl_date") or ""),
            })

    selected: list[dict] = []
    seen_hashes: set[str] = set()
    used = 0
    for item in candidates:
        content = str(item.get("content") or "").strip()
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not content or digest in seen_hashes:
            continue
        remaining = max_chars - used
        if remaining <= 0 or len(selected) >= max_items:
            break
        clipped = content[:remaining]
        selected.append({**item, "content": clipped, "content_hash": digest})
        seen_hashes.add(digest)
        used += len(clipped)
    return selected

# ============ 从用户简历材料（写作指导 + 4 份真实简历）蒸馏出的格式模式 ============
# 内容组织：项目名(副标题) → 时间|角色 → 项目简介 → 技术栈 → 编号技术亮点；
# 亮点句式：**要点名：**动作+对象+技术手段+效果；量化单列或嵌入，绝不编造。
_LEARNED_PATTERN = """【输出结构（从用户提供的真实简历范例中学习，必须严格遵循）】

#### <项目名>（可带一句话副标题，如"基于 RAG 与 Agent 的智能模拟面试系统"）

| 角色 | 时间 |
| --- | --- |
| <角色，如 AI 应用开发 / 独立开发> | YYYY.MM - YYYY.MM（素材中没有时间就写"时间待填"，不要编造） |

- 项目简介：面向<场景>，构建/实现<什么系统>，重点解决<2-3 个核心痛点>，提升<价值>。（1-2 句）
- 技术栈：<技术1>、<技术2>、…（顿号分隔，按与目标岗位的相关度排序，8-12 项）
- 技术亮点：
  1. **<要点名>**：设计/实现/构建<做了什么>，通过/基于/结合<关键技术与方案细节>，解决/提升/保障<效果>。
  2. （共 3-5 条，每条都是「**要点名：**描述」结构——要点名是名词短语，
     如"混合检索""上下文治理""增量索引"，不是动词开头的流水句）
  3. …
  4. **量化数据**：素材中有评测/性能/规模数字时**单列一条**集中呈现
     （如 RAGAS 指标、耗时对比、召回率、测试用例数）；没有数字就省略此条，**绝不编造**。

【句式与密度要求】
- 描述句 = 动作（设计/实现/构建/搭建）+ 对象 + 技术手段 + 效果，每条 60~120 字，信息密度高；
- 中文为主，技术名词保留英文（RAG、Agent、Rerank、Embedding…）；
- 技能用词分级语义：熟悉 > 掌握 > 了解，不要乱用；
- 让面试官 10 秒看懂：项目干什么、为什么做、你负责哪块、用了什么技术、做成什么样。"""

# 用户材料缺失时的兜底（旧版默认）
_DEFAULT_PATTERN = """（内置默认模式 · 经典三段式）
**项目名 | 角色 | 起止时间**
一句话定位：这个项目是什么、解决什么问题（含核心技术栈）。
- 要点 1：动词开头 + 做了什么 + 用了什么技术 + 量化结果（数字/比例/规模）
- 要点 2：突出难点与你的解法
- 要点 3：工程化/协作亮点（部署、测试、CI、文档）
成果：1 句话总结可验证的产出。"""


def load_materials() -> dict:
    """读取 resume_templates/ 用户材料，分两类返回 {guidance: [...], examples: [...]}。"""
    guidance, examples = [], []
    for p in sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.md"))):
        name = os.path.basename(p)
        if name.lower() == "readme.md":
            continue
        try:
            with open(p, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        item = {"name": name, "content": content}
        if any(h in name.lower() for h in _GUIDANCE_HINTS):
            guidance.append(item)
        else:
            examples.append(item)
    return {"guidance": guidance, "examples": examples}


def load_templates() -> list[dict]:
    """兼容旧接口：返回全部用户材料列表（guidance + examples）。"""
    m = load_materials()
    return m["guidance"] + m["examples"]


def extract_project_blocks(md: str, max_chars: int = _MAX_EXAMPLE_CHARS) -> list[str]:
    """从简历范例中抽取「项目经历」条目块（#### 级标题到下一个标题之间）。

    范例文件是整份简历（含教育/实习/技能），但本任务只需要项目条目的写法，
    抽出来注入能省 token 并让 LLM 聚焦正确的格式。
    """
    blocks: list[str] = []
    lines = md.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.match(r"^####\s+\S", ln)]
    total = 0
    for idx, s in enumerate(starts):
        end = len(lines)
        for j in range(s + 1, len(lines)):
            if re.match(r"^#{1,4}\s+\S", lines[j]):
                end = j
                break
        block = "\n".join(lines[s:end]).strip()
        # 只要"像项目条目"的块（含 技术栈/亮点/实现 关键词），跳过实习小节等
        if not re.search(r"技术栈|技术亮点|技术实现|项目简介|项目描述|背景", block):
            continue
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return blocks


def gather_material(repo_url: str = "", text: str = "") -> dict:
    """汇集项目素材：直接文本优先；否则按仓库地址抓取（GitHub 整仓 README/docs）。

    返回 {status:"ok", material, origin} 或 {status:"error", error}。
    """
    text = (text or "").strip()
    if text:
        return {"status": "ok", "material": text[:_MAX_MATERIAL_CHARS], "origin": "用户提供文本"}
    repo_url = (repo_url or "").strip()
    if not repo_url:
        return {"status": "error", "error": "请提供 仓库地址 / 项目介绍文件 / 项目介绍文本 之一"}
    from knowledge_crawler import is_github_repo_url, fetch_repo_text
    if is_github_repo_url(repo_url):
        fetched = fetch_repo_text(repo_url)
        if fetched.get("status") != "ok":
            return {"status": "error", "error": f"仓库抓取失败：{fetched.get('error', '未知')}"}
        return {"status": "ok", "material": fetched["text"][:_MAX_MATERIAL_CHARS],
                "origin": f"GitHub 仓库（{fetched['files_captured']} 个文件）"}
    # 非 GitHub：按普通网页抓正文
    try:
        from job_discovery import fetch_url
        body = fetch_url(repo_url)
        return {"status": "ok", "material": body[:_MAX_MATERIAL_CHARS], "origin": "网页抓取"}
    except Exception as e:
        return {"status": "error", "error": f"页面抓取失败：{e}"}


def build_project_messages(material: str, project_name: str = "",
                           profile: str = "", jd_text: str = "") -> list:
    """组装"项目分析 → 简历段"的 LLM messages（按用户材料学习格式）。

    ``jd_text`` 非空时进入 **JD 定制模式**：
    - 内部提炼 JD 的考察能力与技术栈关键词；
    - 技术亮点优先挑选并前置与 JD 相关的点，措辞向 JD 用词对齐（事实仍只能来自素材）；
    - 简历段之后单独输出「定制建议」：JD 要求但项目不突出的能力 → 怎么补强。
    本函数纯生成，不触发任何知识库/缺口库写入。"""
    mats = load_materials()
    sections: list[str] = []

    if mats["guidance"]:
        g = "\n\n".join(t["content"][:_MAX_GUIDANCE_CHARS] for t in mats["guidance"][:2])
        sections.append("【写作原则（用户提供的简历写作指导，必须遵守）】\n" + g)

    example_blocks: list[str] = []
    for ex in mats["examples"][:3]:
        example_blocks.extend(extract_project_blocks(ex["content"]))
    if example_blocks:
        sections.append(
            "【格式范例（来自用户提供的真实简历，输出必须与这些条目**同款式**）】\n"
            "⚠️ 范例只用于学习**格式/句式/排版**——范例中的技术名词、数字、机制细节"
            "（如 JWT、原子写快照、Celery 等）与你要写的项目无关，**严禁混入输出**。\n\n"
            + "\n\n---\n\n".join(example_blocks[:4]))

    if mats["guidance"] or example_blocks:
        sections.append(_LEARNED_PATTERN)
    else:
        sections.append("用户暂未提供简历材料，按以下内置默认模式输出：\n\n" + _DEFAULT_PATTERN)

    jd_text = (jd_text or "").strip()
    if jd_text:
        sections.append(
            "【目标 JD（定制依据）】\n" + jd_text[:3000] +
            "\n\n【JD 定制要求】\n"
            "1) 先内部提炼这份 JD 考察的能力项与技术栈关键词（不要输出提炼过程）；\n"
            "2) 技术亮点**优先挑选并前置**与 JD 相关的能力/技术点，措辞向 JD 用词对齐"
            "（如 JD 说「检索增强」就不要只写 RAG 缩写）——但所有事实仍**只能来自项目素材**，"
            "不许为迎合 JD 编造能力；\n"
            "3) 与 JD 无关的亮点压缩或舍弃，控制在 3-5 条；\n"
            "4) 简历段结束后，另起一节输出：\n"
            "---\n"
            "### 📌 定制建议（仅供参考，不要放进简历）\n"
            "逐条列出 2-4 条：「JD 要求/考察 <能力X>，当前项目素材中 <不突出/缺失>，"
            "建议 <如何在项目中补强（具体可执行的下一步），或如何调整简历表达>」。"
            "建议要具体到能直接动手，不要空话。")

    system = (
        "你是 OfferClaw 的简历工程师，目标岗位方向：大模型应用工程师。\n"
        "任务：从项目素材中提炼要点，生成**可直接粘贴进简历**的「项目经历」条目。\n\n"
        "分析方法（先内部完成，不要输出过程）：\n"
        "1) 项目定位：面向什么场景、解决什么问题；\n"
        "2) 技术要点：技术栈、架构、关键实现（RAG/Agent/工程化优先突出）；\n"
        "3) 难点与解法：挑最有面试讨论价值的 3-5 个，各自命名为名词短语要点名；\n"
        "4) 量化：凡素材中有数字（性能/规模/覆盖率/指标）必须用上且只能来自素材，"
        "**绝不编造数字**。\n\n"
        "【事实纪律（最高优先级）】技术栈、机制名、数字、时间**只能来自「项目素材」**；"
        "素材没写的不要推断（时间缺失就写「时间待填」，机制内部细节素材没展开就不要替它展开）；"
        "格式范例里的任何内容词都不属于本项目。\n"
        "【输出前自检】逐项核对「技术栈」行：每一项必须能在「项目素材」原文中找到；"
        "找不到的（哪怕很常见，如 Redis/MySQL/Docker）一律删除。\n\n"
        + "\n\n".join(sections) +
        "\n\n【输出纪律】只输出简历条目本身（markdown），不要寒暄、不要解释、不要输出分析过程。"
        + (f"\n\n【用户画像（对齐方向与角色措辞）】\n{profile[:2000]}" if profile else "")
    )
    user = (f"项目名称：{project_name or '（素材中自行识别）'}\n\n"
            f"========== 项目素材 ==========\n{material}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
