# -*- coding: utf-8 -*-
"""
OfferClaw · 路线规划最小代码版（plan_gen.py）

职责：
    给定一份 JD 匹配产生的"缺口清单"，按 plan_prompt.md 的契约调用
    LLM 生成一份 4 周可执行计划。

设计取舍（V1 阶段）：
    - 不在本地复现 plan_prompt.md 的全部 9 步逻辑；把 prompt 整篇喂给 LLM，让 LLM 走流程
    - 本脚本只负责：组装上下文（profile + plan_prompt + 缺口清单）→ 调 LLM → 落盘
    - 不做缺口的本地解析校验（那是 plan_prompt.md 第 2 步的职责，由 LLM 在内部做）

输入：
    1. user_profile.md（自动读取）
    2. plan_prompt.md（自动读取）
    3. 缺口清单：命令行参数 --gaps <file> 或 stdin 粘贴

输出：
    plans/plan_<YYYYMMDD_HHMMSS>.md

使用：
    python plan_gen.py --gaps gaps.md
    或
    python plan_gen.py            # 然后从 stdin 粘贴，输入 EOF（Ctrl+Z 回车 / Ctrl+D）结束
"""

import argparse
import datetime
import json
import os
import re
import sys

import requests

from day1_api_starter import (
    API_KEY_ENV,
    build_zhipu_jwt,
    get_llm_config,
    load_local_env,
)


PROFILE_PATH = "user_profile.md"
PLAN_PROMPT_PATH = "plan_prompt.md"
DAILY_LOG_PATH = "daily_log.md"
SOURCE_POLICY_PATH = "source_policy.md"
TARGET_RULES_PATH = "target_rules.md"
OUTPUT_DIR = "plans"
PROJECT_CONTEXT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "knowledge_base", "project_context")


def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"必需文件缺失：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_project_context() -> str:
    """读取 knowledge_base/project_context/ 下的"已有项目现状"先验，拼成上下文块。

    这些是用户自有的在建项目（如 LocalFlow），OfferClaw **只读**：规划时据此建议
    "在现有项目上推进/扩展"，不让用户从零重造，也绝不代为开发。无则返回空串。
    """
    if not os.path.isdir(PROJECT_CONTEXT_DIR):
        return ""
    blocks = []
    for fn in sorted(os.listdir(PROJECT_CONTEXT_DIR)):
        if fn.endswith(".md") and not fn.startswith("_"):
            with open(os.path.join(PROJECT_CONTEXT_DIR, fn), encoding="utf-8") as f:
                blocks.append(f.read())
    return "\n\n".join(blocks)


def read_gaps(args) -> str:
    if args.gaps:
        return read_text(args.gaps)
    print("请粘贴缺口清单，输入完成后按 Ctrl+Z 回车（Windows）或 Ctrl+D（Unix）结束：")
    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("缺口清单为空，已退出")
    return data


# =====================================================
# P1：RAG 检索学习资源（让计划基于真实知识库，而非 LLM 即兴编）
# =====================================================

# 只检索"岗位知识 / 学习资源"类内容，避免把画像/日志等内部文件混进推荐
RESOURCE_SOURCE_TYPES = ["career_knowledge", "resource"]

_KB_TITLE_CACHE: dict | None = None


def _kb_title_map() -> dict:
    """构建 {文件名: frontmatter title} 映射，用于给资源显示真实主题标题。

    chunk metadata 里的 title 往往是 ``## 正文采集`` 这类切块小标题，对 LLM
    没有指示性；文件级 frontmatter 的 title（如「7. 大模型 Harness Engineering」）
    才能让 LLM 把资源对应到缺口。一次性扫描 knowledge_base，缓存结果。
    """
    global _KB_TITLE_CACHE
    if _KB_TITLE_CACHE is not None:
        return _KB_TITLE_CACHE
    import re
    mapping = {}
    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
    for root, _dirs, files in os.walk(kb_dir):
        if "_pending" in root:
            continue
        for fn in files:
            if not fn.endswith(".md") or fn.startswith("_"):
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    head = f.read(600)
                m = re.search(r'^title:\s*"?([^"\n]+)"?\s*$', head, re.MULTILINE)
                if m:
                    mapping[fn] = m.group(1).strip()
            except Exception:
                continue
    _KB_TITLE_CACHE = mapping
    return mapping


def _split_gap_queries(gaps: str, direction: str = "") -> list[str]:
    """把缺口清单拆成若干条独立检索 query。

    每条缺口（以 -/* 或数字开头的行，或冒号后的短句）单独成一条 query，
    并各自拼上方向语境。这样能为「RAG 缺口」「Agent 缺口」分别检索到
    各自最相关的章节，而不是用一个大杂烩 query 把它们平均掉。
    """
    items = []
    for ln in gaps.splitlines():
        s = ln.strip().lstrip("-*0123456789.、 ").strip()
        # 去掉整行包裹的【】小标题括号（前端缺口卡用 【硬门槛缺口】 这种格式）
        s = s.strip("【】 ").strip()
        # 去掉"技能缺口："这类冒号前缀小标题
        if "：" in s and len(s.split("：", 1)[0]) <= 6:
            s = s.split("：", 1)[1].strip()
        if not s:
            continue
        # 跳过纯分类小标题（硬门槛缺口/技能缺口/经历缺口/建议…），它们是结构标签而非检索内容。
        # 限长 ≤8，避免误杀「…能力缺口」这类描述性长句。
        if (s.endswith("缺口") and len(s) <= 8) or s in ("建议", "总结", "说明", "备注"):
            continue
        if len(s) >= 6:
            items.append(s)
    if not items:
        items = [gaps.strip()]
    prefix = f"{direction} " if direction else ""
    return [f"{prefix}{it}" for it in items]


def retrieve_learning_resources(
    gaps: str,
    direction: str = "",
    per_query: int = 4,
    top_files: int = 6,
) -> list[dict]:
    """基于缺口清单做 RAG 检索，返回去重后的学习资源片段。

    返回 ``[{"source": 文件名, "title": 标题, "snippet": 正文片段}, ...]``。
    设计要点：
    - **逐条缺口分别检索**（multi-query），再合并去重，保证每个缺口都能
      命中针对性资源，而非一个大杂烩 query 把不同主题平均掉
    - 只检索 ``career_knowledge`` / ``resource`` 两类 source_type
    - 按 source 文件去重；跳过纯 frontmatter / 纯目录等低密度片段
    - 任何失败（无 KEY / 集合不存在 / 依赖缺失）都静默返回 []，
      让 plan_gen 在离线 / 无 RAG 时仍能正常出计划
    """
    try:
        import chromadb
        from rag_tools import get_collection_name, get_embeddings_batch, has_embedding_api_key

        if not has_embedding_api_key():
            return []

        client = chromadb.PersistentClient(
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        )
        col = client.get_collection(get_collection_name())
        if col.count() == 0:
            return []

        queries = _split_gap_queries(gaps, direction)
        embeddings = get_embeddings_batch(queries)

        seen_sources = set()
        out = []
        # 轮转：每条 query 先各取最佳 1 个不重复来源，再回头补第 2 个……
        # 这样保证「每个缺口都有资源」，而不是某个缺口霸占全部名额。
        ranked_per_query = []
        for emb in embeddings:
            res = col.query(
                query_embeddings=[emb],
                n_results=per_query * 3,
                where={"source_type": {"$in": RESOURCE_SOURCE_TYPES}},
                include=["documents", "metadatas"],
            )
            ranked_per_query.append(list(zip(res.get("documents", [[]])[0],
                                             res.get("metadatas", [[]])[0])))

        for round_i in range(per_query * 3):
            for hits in ranked_per_query:
                if round_i >= len(hits):
                    continue
                doc, meta = hits[round_i]
                src = meta.get("source", "?")
                if src in seen_sources:
                    continue
                snippet = _clean_snippet(doc)
                if len(snippet) < 40:
                    continue
                seen_sources.add(src)
                # 优先用文件级 frontmatter 标题（指示性强），退回 chunk 标题
                file_title = _kb_title_map().get(src, "")
                out.append({
                    "source": src,
                    "title": file_title or meta.get("title", "") or "",
                    "snippet": snippet[:300],
                })
                if len(out) >= top_files:
                    return out
        return out
    except Exception:
        return []


def _clean_snippet(doc: str) -> str:
    """清理片段用于展示：去掉 frontmatter、来源 blockquote、页面目录标题，留真正正文。"""
    import re
    text = doc
    # 去掉开头的 YAML frontmatter
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        # 跳过来源/路径/入库说明等 blockquote 与一级标题、目录标记
        if s.startswith(">") or s.startswith("# "):
            continue
        if s in ("## 页面结构目录", "## 正文采集", "## 正文内容", "## 图片素材"):
            continue
        lines.append(s)
    return " ".join(lines).strip()


def _extract_direction(profile: str) -> str:
    """从 user_profile.md 抽取目标方向（仅取编号的方向条目，不含薪资/岗位类型噪音）。

    抓「目标方向」标题后紧跟的数字编号行，遇到第一个非编号行即停止。
    抓不到就返回空串（不影响检索）。
    """
    import re
    lines = profile.splitlines()
    out = []
    capture = False
    for ln in lines:
        s = ln.strip()
        if "目标方向" in s:
            capture = True
            continue
        if capture:
            m = re.match(r"^\d+[\.、]\s*(.+)$", s)
            if m:
                out.append(m.group(1).strip())
            elif s and not s.startswith(("-", "*")):
                break
            elif out:  # 已抓到编号项后遇到 - 开头的下一字段，停止
                break
    return " ".join(out)


def format_resources_block(resources: list[dict]) -> str:
    """把检索到的资源格式化成喂给 LLM 的上下文块。"""
    if not resources:
        return ""
    lines = ["以下是从本地知识库检索到的、与缺口相关的真实学习资源（请在计划中优先引用，标注来源文件名）：\n"]
    for i, r in enumerate(resources, 1):
        title = f"《{r['title']}》" if r["title"] else ""
        lines.append(f"[资源{i}] {title}（来源：{r['source']}）\n  {r['snippet']}\n")
    return "\n".join(lines)


def build_messages(profile: str, plan_prompt: str, daily_log: str,
                   source_policy: str, target_rules: str, gaps: str,
                   resources_block: str = "", project_context: str = "",
                   prev_plan: str = "", revision_note: str = "",
                   start_date: str = "", end_date: str = "",
                   adjustments_block: str = "") -> list:
    """组装要发给 LLM 的 messages。

    设计：把 5 份依赖文件作为 system 上下文，缺口清单作为 user 消息。
    LLM 内部按 plan_prompt 的 9 步流程执行。
    P1：若 ``resources_block`` 非空，追加 RAG 检索到的真实学习资源。
    项目先验：若 ``project_context`` 非空，追加用户已有项目现状，要求把实战任务
    优先落到"在已有项目上推进"，且 OfferClaw 只建议不动手（只读）。
    """
    system_content = (
        "你是 OfferClaw，部署在长期会话中的求职作战官。\n"
        "你接下来要严格按 plan_prompt.md 的指令执行一次路线规划。\n"
        "下面是你必须读取的依赖文件全文。\n\n"
        f"========== plan_prompt.md ==========\n{plan_prompt}\n\n"
        f"========== user_profile.md ==========\n{profile}\n\n"
        f"========== target_rules.md ==========\n{target_rules}\n\n"
        f"========== daily_log.md ==========\n{daily_log}\n\n"
        f"========== source_policy.md ==========\n{source_policy}\n"
    )

    if adjustments_block:
        system_content += (
            f"\n========== 复盘沉淀的调整规则（来自用户实际执行情况）==========\n{adjustments_block}\n"
            "这些规则是用户多日复盘的真实教训，排期时**必须遵守**："
            "如提示减量就降低单日任务量、提示拆细就把大任务拆成小步、"
            "提示某类任务反复未完成就调整其时段或方式。\n"
        )

    if resources_block:
        system_content += (
            f"\n========== 知识库检索资源（RAG） ==========\n{resources_block}\n"
            "规划每周/每日任务时，凡涉及上面资源覆盖的主题，"
            "必须引用对应资源并标注「来源：<文件名>」；不要编造不存在的资源链接。\n"
        )

    if project_context:
        system_content += (
            f"\n========== 用户已有项目现状（只读先验）==========\n{project_context}\n"
            "规划实战任务时的硬性要求：\n"
            "1) 优先把实战落到上面已有项目上（基于其现状给出『下一步推进建议』），"
            "不要让用户从零重造已有项目已具备的东西；\n"
            "2) 你对这些项目【只读】——只给学习/推进建议，绝不代为编码、修改、提交，"
            "措辞用『建议你/可以尝试』，不要写成『我来实现』；\n"
            "3) 若用户明确表示想『从零搭一个新项目来学习』，则可另规划新建项目（二者都支持）。\n"
        )

    if prev_plan:
        system_content += (
            f"\n========== 用户当前计划（最新版，可能含用户手动调整）==========\n{prev_plan}\n"
            "再规划时的要求：在这份现有计划基础上**演进**——保留仍然有效的安排，"
            "只针对新缺口/新调整说明做增量更新；**尊重用户手动改过的内容**（不要无故推翻）。\n"
            "⚠️ 演进指的是**内容连续**（衔接已完成的进度、保留有效安排），"
            "**日期与周次必须重排**：新计划一律从开始日期起、从 Week 1 连续编号且 Week 1 包含开始日期；"
            "绝不允许沿用旧计划的周次、从 Week 2 开始输出、或把起始日推迟到未来。\n"
            "完成度感知（中途重生成时遵循事实）：对照 system 里 daily_log.md 的实际记录——"
            "**已完成的任务不要重复排期**；未完成/部分完成的结合优先级合理重排进剩余天数；"
            "已过去的日期不出现在新计划里（历史以 daily_log 为准，不重写历史）。\n"
        )

    project_directive = ""
    if project_context:
        project_directive = (
            "\n\n【实战编排硬性要求】用户已有在建项目（见 system 的『用户已有项目现状』）。\n"
            "凡『缺项目经历 / 端到端项目 / Agent / 工具调用 / 工作流 / Harness』类实战缺口，"
            "**必须编排为『在已有项目上推进的下一步』**（明确写出在哪个项目、加什么、达到什么），"
            "不要让用户从零再造一个同类项目。每条这类任务用『建议你…』措辞，"
            "OfferClaw 只建议、不代为开发。纯基础类缺口（如 Python 语法）可正常安排学习任务。"
        )
    revision_directive = ""
    if (revision_note or "").strip():
        revision_directive = (
            "\n\n【用户本次修改要求（一次性指令，仅本次生效，无历史可关联）】\n"
            + revision_note.strip() + "\n"
            "处理方式：在 system 里『用户当前计划』的基础上，**优先理解并满足上述修改要求**"
            "（哪里不满意、希望怎么改），其余仍然有效的安排保留；事实与资源仍只能来自"
            "画像/知识库/缺口清单，不为迎合要求而编造。"
        )
    start = (start_date or "").strip() or datetime.date.today().isoformat()
    if (end_date or "").strip():
        period_directive = (
            f"计划开始日期：{start}（用户指定）；结束日期：{end_date.strip()}（用户指定）。"
            "周期严格按这两个日期排，Week 1 从开始日期起。"
        )
    else:
        period_directive = (
            f"计划开始日期：{start}（用户指定）；结束日期：用户未指定——"
            "由你按『任务总量 ÷ 每周可投入工时』估算合适周数（向上取整，通常 2-8 周），"
            "并在计划开头一句话写明估算依据。"
        )
    user_content = (
        "请按 plan_prompt.md 的 9 步流程，基于下面这份缺口清单生成学习计划。\n"
        "今天日期是 " + datetime.date.today().isoformat() + "。\n"
        + period_directive
        + project_directive + revision_directive + "\n\n"
        "========== 缺口清单 ==========\n" + gaps
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def call_llm_plain(messages, api_key, max_tokens: int = 4000) -> str:
    """调用 LLM，不带 tools（规划是单次纯文本生成）。"""
    cfg = _resolve_chat_config(api_key)
    url = f"{cfg['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['bearer']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    from day1_api_starter import chat_completion, extract_content  # A1 网关 + A2 防御解析
    data = chat_completion(url, headers, payload, timeout=120)
    return extract_content(data)


def call_llm_stream(messages, api_key, max_tokens: int = 4000):
    """调用 LLM，stream=True，逐 token yield str。供 SSE 端点消费。"""
    import json as _json
    cfg = _resolve_chat_config(api_key)
    url = f"{cfg['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['bearer']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    from day1_api_starter import chat_completion  # A1 统一 LLM 网关（重试+退避）
    resp = chat_completion(url, headers, payload, timeout=120, stream=True)
    with resp:
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    chunk = _json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    pass


def _resolve_chat_config(api_key: str) -> dict:
    """Resolve chat endpoint/auth for both current proxy and legacy Zhipu callers."""
    cfg = get_llm_config()
    zhipu_key = os.environ.get("ZHIPU_API_KEY", "")
    if api_key and api_key == zhipu_key:
        return {
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4-flash",
            "bearer": build_zhipu_jwt(api_key),
            "reasoning_effort": "",
        }
    bearer = build_zhipu_jwt(api_key) if cfg["is_zhipu"] else api_key
    return {**cfg, "bearer": bearer}


def normalize_plan_dates(plan_md: str, start_iso: str = "") -> str:
    """确定性重写计划日期：周期/周界/日标签全部由「开始日期 + 序号」推导。

    LLM 连续生成几十个日期极易算错（错日/重日/错星期），导致"总计划与
    每日执行对不上"。日期不交给 LLM：生成后以本函数为唯一事实源重写——
      - 第 i 个日块标签 → D{i}（MM-DD 周X），日期 = start + (i-1)
      - Week N 头 → start+(N-1)*7 … min(start+N*7-1, end)
      - 计划周期行 → start → start + 总天数 - 1
    没有日标签的文本原样返回。
    """
    import re
    import datetime

    lines = plan_md.splitlines()
    # 前缀容忍 Markdown 装饰(### / **):不同生成模型风格不同——qwen 裸头,
    # gpt-5.6 实测输出 "### D1（08-08 周六）";装饰保留原样,只重写日期部分。
    day_re = re.compile(r"^(\s*(?:#{1,6}\s+)?(?:\*\*)?\s*)D\d+\s*[（(][^）)]*[）)]")
    day_count = sum(1 for ln in lines if day_re.match(ln))
    if day_count == 0:
        return plan_md

    start = None
    if start_iso:
        try:
            start = datetime.date.fromisoformat(start_iso)
        except ValueError:
            pass
    if start is None:
        m = re.search(r"计划周期[:：]\s*(\d{4}-\d{2}-\d{2})", plan_md)
        if m:
            start = datetime.date.fromisoformat(m.group(1))
    if start is None:
        return plan_md

    wd = "一二三四五六日"

    def _fmt(d: "datetime.date") -> str:
        return f"{d.month:02d}-{d.day:02d} 周{wd[d.weekday()]}"

    out, i_day = [], 0
    for ln in lines:
        if day_re.match(ln):
            i_day += 1
            d = start + datetime.timedelta(days=i_day - 1)
            ln = day_re.sub(lambda mo: f"{mo.group(1)}D{i_day}（{_fmt(d)}）", ln, count=1)
        out.append(ln)
    plan_md = "\n".join(out)

    end = start + datetime.timedelta(days=day_count - 1)
    plan_md = re.sub(
        r"(计划周期[:：]\s*)\d{4}-\d{2}-\d{2}(\s*→\s*)\d{4}-\d{2}-\d{2}",
        lambda mo: f"{mo.group(1)}{start.isoformat()}{mo.group(2)}{end.isoformat()}",
        plan_md)

    def _wk(mo):
        n = int(mo.group(1))
        op = mo.group(2)                     # 保留原括号风格(qwen 半角/gpt 全角)
        cl = "）" if op == "（" else ")"
        sep = "" if op == "（" else " "
        ws = start + datetime.timedelta(days=(n - 1) * 7)
        we = min(ws + datetime.timedelta(days=6), end)
        return (f"Week {n}{sep}{op}{ws.month:02d}-{ws.day:02d} → "
                f"{we.month:02d}-{we.day:02d}{cl}")

    plan_md = re.sub(
        r"Week\s*(\d+)\s*([（(])\s*\d{1,2}-\d{1,2}\s*→\s*\d{1,2}-\d{1,2}\s*[）)]", _wk, plan_md)
    return plan_md


def digest_history(full_log: str, today_iso: str = "", window_days: int = 14,
                   max_weeks: int = 12) -> str:
    """把窗口外的历史留痕**按周压缩成摘要**（确定性，不调 LLM、不需存储）。

    信息分级而非丢弃：近 window_days 天给明细（调用方负责），更早的每周一行
    （留痕天数/完成数/未完成数/主线分布），超过 max_weeks 的折叠成一行合计。
    原始明细永久保存在 daily_log.md 与向量库，随时可查。
    """
    import re
    import datetime

    today = (datetime.date.fromisoformat(today_iso)
             if today_iso else datetime.date.today())
    cutoff = today - datetime.timedelta(days=window_days - 1)

    # 切块：## YYYY-MM-DD … 到下一个日期头
    blocks = re.split(r"(?=^## \d{4}-\d{2}-\d{2})", full_log, flags=re.M)
    weeks: dict[tuple, dict] = {}
    older_days = 0
    for b in blocks:
        m = re.match(r"^## (\d{4}-\d{2}-\d{2})", b)
        if not m:
            continue
        try:
            d = datetime.date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if d >= cutoff:
            continue        # 窗口内明细由调用方注入，不进摘要
        older_days += 1
        key = d.isocalendar()[:2]   # (year, week)
        w = weeks.setdefault(key, {"days": 0, "done": 0, "todo": 0,
                                   "tags": {}, "start": d, "end": d})
        w["days"] += 1
        w["start"] = min(w["start"], d)
        w["end"] = max(w["end"], d)
        try:
            from summary_tool import _parse_log_block
            parsed = _parse_log_block(b, m.group(1))
            w["done"] += len(parsed.get("completed") or [])
            w["todo"] += len(parsed.get("incomplete") or [])
            tag = (parsed.get("main_tag") or "").strip()
            if tag:
                w["tags"][tag] = w["tags"].get(tag, 0) + 1
        except Exception:
            pass

    if not weeks:
        return ""
    ordered = sorted(weeks.items(), key=lambda kv: kv[0], reverse=True)  # 新→旧
    lines = ["【更早历史·周摘要】（明细永久保存在 daily_log.md，可随时查阅）"]
    for (y, wk), w in ordered[:max_weeks]:
        tags = "、".join(f"{t}×{n}" for t, n in
                        sorted(w["tags"].items(), key=lambda kv: -kv[1])[:2]) or "—"
        lines.append(
            f"- {y}-W{wk:02d}（{w['start'].strftime('%m-%d')}~{w['end'].strftime('%m-%d')}）："
            f"留痕{w['days']}天 · 完成{w['done']}项 · 未完成{w['todo']}项 · 主线[{tags}]")
    if len(ordered) > max_weeks:
        rest = ordered[max_weeks:]
        rest_days = sum(w["days"] for _k, w in rest)
        lines.append(f"- 更早 {len(rest)} 周合计：留痕 {rest_days} 天（已折叠）")
    return "\n".join(lines)


def summarize_plan_changes(old_md: str, new_md: str) -> list[str]:
    """重排后输出"较上一版变化"的确定性摘要（不调 LLM）：周期 + 各周主题对比。"""
    import re

    def _themes(md: str) -> list[str]:
        return [t.strip() for t in re.findall(r"Week\s*\d+[^\n]*主题[:：]\s*([^\n]+)", md or "")]

    def _period(md: str) -> str:
        m = re.search(r"计划周期[:：]\s*([\d\-]+\s*→\s*[\d\-]+)", md or "")
        return m.group(1).replace(" ", "") if m else ""

    changes: list[str] = []
    po, pn = _period(old_md), _period(new_md)
    if po and pn and po != pn:
        changes.append(f"周期：{po} → {pn}")
    to, tn = _themes(old_md), _themes(new_md)
    for i in range(max(len(to), len(tn))):
        a = to[i] if i < len(to) else "（无）"
        b = tn[i] if i < len(tn) else "（无）"
        if a != b:
            changes.append(f"Week{i + 1}：{a[:28]} → {b[:28]}")
    if not changes:
        changes.append("与上一版整体一致（细节微调）")
    return changes


def is_degenerate_plan(content: str) -> bool:
    """判定一份"计划"是否是退化产物（LLM 拒绝/报错文本，而非真计划）。

    特征：含输入检查拒绝话术，或没有任何 Week N 主题 周结构。
    退化产物不应作为 prev_plan 注入（会让 LLM 照着拒绝文案自我复制），
    也不应落盘成"当前计划"。
    """
    import re
    c = content or ""
    if "依赖文件检查未通过" in c or "无法执行路线规划" in c:
        return True
    return not re.search(r"Week\s*\d+[^\n]*主题", c)


def ensure_gap_metadata(gaps: str) -> str:
    """给缺口条目自动补齐 plan_prompt 第 2 步要求的元数据标签。

    match / 缺口库 / 画像默认 产出的缺口都是纯文本，不带
    ``[致命度: ...] [短期性: ...]``，会被 plan_prompt 的输入检查拒绝。
    这里按所属分类补默认值（确定性，不调 LLM）：
      硬门槛缺口 → [致命度: 高]；技能/经历缺口 → [致命度: 中]；其他 → [致命度: 低]；
      短期性统一默认 [短期性: 可补]（不可补的硬门槛本就无法靠 4 周计划解决）。
    已带标签的条目原样保留。
    """
    import re
    cat_level = {"硬门槛": "高", "技能": "中", "经历": "中"}
    level = "中"
    out: list[str] = []
    for ln in gaps.splitlines():
        s = ln.strip()
        # 分类标题行：更新当前默认致命度
        header = re.sub(r"[\s:：\[\]【】]+", "", s)
        if header.endswith("缺口") and len(header) <= 8:
            for key, lv in cat_level.items():
                if key in header:
                    level = lv
                    break
            else:
                level = "低"
            out.append(ln)
            continue
        # 条目行（- / * / 数字开头）且未带标签 → 行内补默认元数据
        if re.match(r"^\s*[-*\d]", ln) and "致命度" not in ln:
            out.append(f"{ln.rstrip()} [致命度: {level}] [短期性: 可补]")
        else:
            out.append(ln)
    return "\n".join(out)


def prepare_plan_messages(gaps: str, revision_note: str = "",
                          start_date: str = "", end_date: str = "") -> tuple[list, list[dict]]:
    """读依赖文件 + RAG 检索资源 + 组装 messages 的统一入口。

    CLI 与 FastAPI（/api/plan、/api/plan/stream）共用此函数，确保三条
    路径的 RAG 接入行为一致（避免只有 CLI 享受 P1）。

    返回 ``(messages, resources)``。``resources`` 供调用方追加确定性附录。
    """
    profile = read_text(PROFILE_PATH)
    plan_prompt = read_text(PLAN_PROMPT_PATH)
    source_policy = read_text(SOURCE_POLICY_PATH)
    target_rules = read_text(TARGET_RULES_PATH)

    # daily_log 窗口化注入：近 14 天明细 + 历史省略说明（防上下文随月份无限膨胀）
    daily_log_full = read_text(DAILY_LOG_PATH)
    daily_log = daily_log_full
    try:
        import re
        from summary_tool import extract_recent_blocks
        recent = extract_recent_blocks(daily_log_full, days=14)
        if recent and len(recent) < len(daily_log_full) - 200:
            total_days = len(re.findall(r"^## \d{4}-\d{2}-\d{2}", daily_log_full, re.M))
            digest = digest_history(daily_log_full, window_days=14)
            daily_log = (
                f"（历史已分级注入：近 14 天给明细、更早按周压缩成摘要；"
                f"全部 {total_days} 天原始明细永久保存在 daily_log.md）\n\n"
                + (digest + "\n\n" if digest else "")
                + "【近 14 天明细】\n" + recent)
    except Exception:
        pass

    # 复盘沉淀的调整规则 → 注入计划生成（让"越来越懂用户"真正作用到排期上）
    adjustments_block = ""
    try:
        from memory_layers import SemanticMemory, get_active_adjustments
        adj = get_active_adjustments(SemanticMemory())
        if adj:
            adjustments_block = "\n".join(f"- {a}" for a in adj)
    except Exception:
        pass

    # 自动补齐元数据标签，避免被 plan_prompt 第 2 步输入检查拒绝
    gaps = ensure_gap_metadata(gaps)

    direction = _extract_direction(profile)
    # [L4] 注入 procedural SOP（方向级经验）到调整块——让"学到的操作规范"真正作用到排期
    try:
        from memory_layers import ProceduralMemory, get_active_sops
        _sops = get_active_sops(ProceduralMemory(), context=(direction or ""))
        if _sops:
            _sop_text = "\n".join(f"- [SOP] {s}" for s in _sops)
            adjustments_block = (adjustments_block + "\n" + _sop_text) if adjustments_block else _sop_text
    except Exception:
        pass
    # [L6] episodic 回流：把最近复盘事件的具体教训注入，让 LLM 看到事件级历史而非仅蒸馏摘要
    try:
        from memory_layers import EpisodicMemory, recent_reflection_lessons
        _lessons = recent_reflection_lessons(EpisodicMemory(), n=3)
        if _lessons:
            _les_text = "\n".join(f"- [近期复盘] {x}" for x in _lessons)
            adjustments_block = (adjustments_block + "\n" + _les_text) if adjustments_block else _les_text
    except Exception:
        pass
    resources = retrieve_learning_resources(gaps, direction=direction)
    resources_block = format_resources_block(resources)
    project_context = load_project_context()

    # 再规划以"最新计划（含用户手动调整）"为基础演进；截断以控制上下文体积。
    # 退化产物（拒绝文案/无周结构）绝不注入——否则 LLM 会照着它自我复制拒绝。
    prev_plan = ""
    latest = load_latest_plan()
    if latest and latest.get("content") and not is_degenerate_plan(latest["content"]):
        prev_plan = latest["content"][:4000]

    # [L2] profile 上下文预算：仅当超量时按 gaps/方向相关性挑章节，防长期膨胀爆窗（小 profile 原样不变）
    import os as _os
    from context_budget import select_relevant_sections, keywords_from
    _profile_budget = int(_os.environ.get("OFFERCLAW_PROFILE_BUDGET_CHARS", "6000"))
    profile_view = select_relevant_sections(
        profile, keywords=keywords_from(f"{gaps} {direction or ''}"),
        max_chars=_profile_budget, always_keep=("基础信息", "元信息", "方向", "技能"))

    messages = build_messages(
        profile_view, plan_prompt, daily_log, source_policy, target_rules,
        gaps, resources_block=resources_block, project_context=project_context,
        prev_plan=prev_plan, revision_note=revision_note,
        start_date=start_date, end_date=end_date,
        adjustments_block=adjustments_block,
    )
    return messages, resources


def append_resources_appendix(plan_text: str, resources: list[dict]) -> str:
    """在 LLM 生成的计划末尾追加一个确定性的「参考资源」附录。

    LLM 是否在正文里引用资源不稳定（同一输入可能引用 0~N 次），因此由代码
    确定性地把 RAG 检索到的真实资源附在计划末尾，保证每份计划都能落到
    可追溯的知识库来源（满足 P1：计划输出必含知识库资源引用）。
    """
    if not resources:
        return plan_text
    lines = [
        plan_text.rstrip(),
        "",
        "---",
        "",
        "## 📚 本计划参考的知识库资源（RAG 自动检索）",
        "",
        "> 由 OfferClaw 基于上面的缺口清单从本地知识库自动检索，按相关度排序；"
        "学习对应主题时优先参考。",
        "",
    ]
    for i, r in enumerate(resources, 1):
        title = r["title"] or r["source"]
        lines.append(f"{i}. **{title}**")
        lines.append(f"   - 来源：`{r['source']}`")
        lines.append(f"   - 摘要：{r['snippet'][:140]}")
        lines.append("")
    return "\n".join(lines)


_TARGET_SNAPSHOT_MARKER = "OFFERCLAW_TARGET_SNAPSHOT="
_PROFILE_HASH_MARKER = "OFFERCLAW_PROFILE_HASH="


def append_target_trace(plan_text: str, snapshot: dict | None = None) -> str:
    """Freeze the application/JD target set used by this generated plan."""
    if _TARGET_SNAPSHOT_MARKER in plan_text:
        return plan_text
    from application_jd_store import render_target_appendix, target_snapshot
    snapshot = snapshot or target_snapshot()
    appendix = render_target_appendix(snapshot)
    if not appendix:
        return plan_text
    encoded = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    return (plan_text.rstrip() + "\n\n---\n\n" + appendix.rstrip()
            + f"\n\n<!-- {_TARGET_SNAPSHOT_MARKER}{encoded} -->\n")


def extract_target_snapshot(plan_text: str) -> dict:
    marker = f"<!-- {_TARGET_SNAPSHOT_MARKER}"
    start = plan_text.rfind(marker)
    if start < 0:
        return {}
    raw = plan_text[start + len(marker):].split("-->", 1)[0].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def plan_target_status(plan_text: str) -> dict:
    """Compare the plan's frozen target set with the current application projection."""
    frozen = extract_target_snapshot(plan_text)
    if not frozen:
        return {"traceable": False, "stale": True,
                "reason": "旧版计划，目标 JD 不可完整追溯", "snapshot": {}}
    try:
        from application_jd_store import target_snapshot
        current = target_snapshot()
    except Exception:
        return {"traceable": True, "stale": True,
                "reason": "当前投递目标暂时不可读取", "snapshot": frozen}
    stale = frozen.get("snapshot_id") != current.get("snapshot_id")
    return {"traceable": True, "stale": stale,
            "reason": "目标已变化，建议重排" if stale else "目标与当前投递一致",
            "snapshot": frozen, "current_snapshot_id": current.get("snapshot_id", "")}


def extract_plan_profile_hash(plan_text: str) -> str:
    match = re.search(r"<!--\s*" + re.escape(_PROFILE_HASH_MARKER) + r"([a-f0-9]{64})\s*-->",
                      plan_text or "")
    return match.group(1) if match else ""


def plan_profile_status(plan_text: str) -> dict:
    frozen = extract_plan_profile_hash(plan_text)
    try:
        from profile_store import read_profile
        current = read_profile()["base_hash"]
    except Exception:
        return {"traceable": bool(frozen), "stale": True, "reason": "当前画像暂时不可读取"}
    if not frozen:
        return {"traceable": False, "stale": True,
                "reason": "旧版计划未记录画像版本，建议确认或局部调整"}
    stale = frozen != current
    return {"traceable": True, "stale": stale,
            "reason": "画像已变化：可保持、局部调整或重新规划" if stale else "画像与计划生成时一致",
            "profile_hash": frozen, "current_profile_hash": current}


def append_profile_trace(plan_text: str) -> str:
    if _PROFILE_HASH_MARKER in plan_text:
        return plan_text
    try:
        from profile_store import read_profile
        digest = read_profile()["base_hash"]
    except Exception:
        return plan_text
    return plan_text.rstrip() + f"\n\n<!-- {_PROFILE_HASH_MARKER}{digest} -->\n"


def _plans_dir() -> str:
    """plans/ 目录的绝对路径（相对 plan_gen.py 所在目录，避免受 CWD 影响）。"""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, OUTPUT_DIR))


def save_plan(content: str, edited_by_user: bool = False,
              operation_id: str | None = None, note: str = "") -> str:
    """落盘一份计划到 plans/。edited_by_user=True 时文件名带 _user 后缀，便于回读时标识。"""
    from plan_daily import ensure_task_ids
    content = append_profile_trace(ensure_task_ids(content))
    if operation_id:
        from memory_layers import EpisodicMemory
        existing = EpisodicMemory().store.get_event_by_operation(operation_id)
        if existing:
            saved_path = str(existing.get("saved_path") or "")
            replay_path = os.path.abspath(os.path.join(os.path.dirname(__file__), saved_path))
            if os.path.isfile(replay_path):
                return replay_path
            raise FileNotFoundError(f"幂等操作对应的计划文件已不存在: {saved_path}")
    out_dir = _plans_dir()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_user" if edited_by_user else ""
    path = os.path.join(out_dir, f"plan_{ts}{suffix}.md")
    try:
        from memory_layers import EpisodicMemory
        epi = EpisodicMemory()
        snapshot = epi.store.put_snapshot(content, media_type="text/markdown",
                                          source_path=os.path.relpath(path, os.path.dirname(__file__)))
        from memory_transactions import write_text_with_memory
        result = write_text_with_memory(
            path, content, event_kind="plan_saved",
            event_payload={"edited_by_user": edited_by_user,
                           "saved_path": os.path.relpath(path, os.path.dirname(__file__)),
                           "snapshot_id": snapshot["snapshot_id"],
                           "content_hash": snapshot["content_hash"],
                           "note": note[:500]},
            event_options={"actor": "user" if edited_by_user else "assistant",
                           "source": "plan_gen", "entity_type": "plan",
                           "entity_id": os.path.basename(path)},
            operation_id=operation_id,
        )
        if result.get("memory_error"):
            import logging
            logging.getLogger(__name__).warning("plan memory event queued for recovery: %s",
                                                result["memory_error"])
    except Exception:
        if not os.path.exists(path):
            raise
        import logging
        logging.getLogger(__name__).warning("plan memory write failed", exc_info=True)
    return path


def summarize_plan_for_automation(today_iso: str | None = None) -> dict:
    """把"当前（最新）学习计划"提炼成自动化任务可直接引用的结构。

    供 today_advice / 微信推送 / 提醒 / 复盘 / 再规划统一引用——
    确保所有自动化都以**最新计划（含用户手动调整）**为参考。

    返回（无计划时 has_plan=False）：
      has_plan, plan_file, edited_by_user, period, expired,
      current_week={n, theme, tags, deliverable} | None, weeks=[...], week_count
    """
    import re
    import datetime

    latest = load_latest_plan()
    if not latest:
        return {"has_plan": False}
    content = latest["content"]
    today = datetime.date.fromisoformat(today_iso) if today_iso else datetime.date.today()

    # 计划周期：YYYY-MM-DD → YYYY-MM-DD
    period = ""
    period_start = period_end = None
    mp = re.search(r"计划周期[:：]\s*(\d{4}-\d{2}-\d{2})\s*[→\-~>]+\s*(\d{4}-\d{2}-\d{2})", content)
    if mp:
        try:
            period_start = datetime.date.fromisoformat(mp.group(1))
            period_end = datetime.date.fromisoformat(mp.group(2))
            period = f"{mp.group(1)} → {mp.group(2)}"
        except ValueError:
            pass

    # 周计划层：Week N (可含日期) 主题：xxx  + 随后的 主线标签 / 交付物
    wk_re = re.compile(r"Week\s*(\d+)[^\n主]*主题[:：]\s*(.+)")
    lines = content.splitlines()
    weeks: list[dict] = []
    seen: set[int] = set()
    for i, ln in enumerate(lines):
        m = wk_re.search(ln)
        if not m:
            continue
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        theme = m.group(2).strip().strip("：: ")
        tags = deliverable = ""
        for j in range(i + 1, min(i + 8, len(lines))):
            if wk_re.search(lines[j]):
                break
            t1 = re.search(r"主线标签[:：]\s*(.+)", lines[j])
            t2 = re.search(r"交付物[:：]\s*(.+)", lines[j])
            if t1 and not tags:
                tags = t1.group(1).strip().strip("[]")
            if t2 and not deliverable:
                deliverable = t2.group(1).strip()
        weeks.append({"n": n, "theme": theme, "tags": tags, "deliverable": deliverable})

    # 当前周：按 计划周期 起点 + 今天推算（不依赖每周日期范围，兼容退化格式）
    current_week = None
    if weeks:
        if period_start:
            idx = (today - period_start).days // 7 + 1
        else:
            idx = 1
        idx = max(1, min(idx, len(weeks)))
        current_week = next((w for w in weeks if w["n"] == idx), weeks[idx - 1])

    # 日计划层：定位"今天"的具体任务（供每日执行栏逐日对照，而非只到周粒度）。
    # 支持 D1（06-05 周五）/ 2026-06-05（周五）等日期行格式。
    today_tasks: list[str] = []
    day_re = re.compile(
        r"(?:D\d+\s*[（(]\s*)?(?:(\d{4})-)?(\d{1,2})-(\d{1,2})\s*[ ）)（(]*周[一二三四五六日天]")
    day_marks: list[tuple[int, datetime.date]] = []
    base_year = period_start.year if period_start else today.year
    for i, ln in enumerate(lines):
        m = day_re.search(ln)
        if not m:
            continue
        y = int(m.group(1)) if m.group(1) else base_year
        try:
            d = datetime.date(y, int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        day_marks.append((i, d))
    for k, (i, d) in enumerate(day_marks):
        if d != today:
            continue
        end = day_marks[k + 1][0] if k + 1 < len(day_marks) else len(lines)
        for ln in lines[i + 1:end]:
            s = ln.strip()
            if re.match(r"^(—|##|###|Week\s*\d)", s):   # 出了日块就停
                break
            mt = re.match(r"^\d+\.\s*(.+)", s)
            if mt:
                today_tasks.append(re.sub(r"\s*<!--\s*task_id:.*?-->\s*$", "",
                                          mt.group(1)).strip())
            if len(today_tasks) >= 5:
                break
        break

    return {
        "has_plan": True,
        "plan_file": latest["filename"],
        "edited_by_user": latest["edited_by_user"],
        "period": period,
        "expired": bool(period_end and today > period_end),
        "current_week": current_week,
        "weeks": weeks,
        "week_count": len(weeks),
        "today_tasks": today_tasks,
        "target_status": plan_target_status(content),
        "profile_status": plan_profile_status(content),
    }


def load_latest_plan() -> dict | None:
    """读取 plans/ 下最新的 plan_*.md。返回 {content, path, filename, mtime, edited_by_user}，
    没有任何计划时返回 None。"""
    import glob
    out_dir = _plans_dir()
    files = glob.glob(os.path.join(out_dir, "plan_*.md"))
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        content = f.read()
    fname = os.path.basename(latest)
    return {
        "content": content,
        "path": latest,
        "filename": fname,
        "mtime": int(os.path.getmtime(latest)),
        "edited_by_user": fname.endswith("_user.md"),
        "target_status": plan_target_status(content),
        "profile_status": plan_profile_status(content),
    }


def main():
    parser = argparse.ArgumentParser(description="OfferClaw 路线规划生成器")
    parser.add_argument("--gaps", "-g", help="缺口清单文件路径（默认从 stdin 读）")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_local_env()
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[ERROR] 未检测到环境变量 {API_KEY_ENV}")
        sys.exit(1)

    print("[1/4] 读取缺口清单...")
    gaps = read_gaps(args)

    print("[2/4] 读取依赖文件 + RAG 检索相关学习资源...")
    messages, resources = prepare_plan_messages(gaps)
    if resources:
        print(f"      ✓ 命中 {len(resources)} 份知识库资源：")
        for r in resources:
            print(f"        - {r['title'] or r['source']}（{r['source']}）")
    else:
        print("      （未命中知识库资源，将生成不带引用的计划）")

    print("[3/4] 调用 LLM 生成计划（最长 120 秒）...")
    try:
        plan_text = call_llm_plain(messages, api_key)
    except Exception as e:  # A2: LLM 调用/解析失败 → 可读降级 + 非零退出，而非 traceback 崩 CLI
        import sys as _sys
        from day1_api_starter import llm_error_detail
        print(f"\n✗ LLM 调用失败，计划未生成：{llm_error_detail(e)}", file=_sys.stderr)
        _sys.exit(1)
    # 确定性追加参考资源附录，保证计划必含可追溯的知识库来源
    plan_text = append_resources_appendix(plan_text, resources)

    print("[4/4] 写入文件...")
    out_path = save_plan(plan_text)
    print(f"\n✓ 计划已生成：{out_path}")
    print("=" * 60)
    print(plan_text[:2000])
    if len(plan_text) > 2000:
        print(f"\n...（截断，完整内容见 {out_path}，共 {len(plan_text)} 字符）")


if __name__ == "__main__":
    main()
