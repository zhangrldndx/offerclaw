# -*- coding: utf-8 -*-
"""
OfferClaw · 学习留痕复盘工具 (summary_tool.py)

职责：
    自动化执行 summary_prompt.md 的复盘流程。
    - 单日模式：读 daily_log.md 中指定日期块（默认今天）→ LLM 复盘 → 落盘
    - 周度模式：读最近 7 天 → LLM 周度复盘 → 落盘

使用：
    python summary_tool.py                  # 今日复盘
    python summary_tool.py --date 2026-04-25
    python summary_tool.py --weekly         # 本周复盘
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import uuid

import requests

from day1_api_starter import API_KEY_ENV, build_zhipu_jwt, get_llm_config, load_local_env

DAILY_LOG_PATH = "daily_log.md"
SUMMARY_PROMPT_PATH = "summary_prompt.md"
SOURCE_POLICY_PATH = "source_policy.md"
TARGET_RULES_PATH = "target_rules.md"
OUTPUT_DIR = "summaries"


def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"必需文件缺失：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_date_blocks(log: str, date_str: str) -> list[str]:
    """返回 ``daily_log.md`` 中某一天的全部块，保留历史重复日期。

    旧写入器每次结构化留痕都会追加一个新的 ``## YYYY-MM-DD`` 标题；此前
    ``extract_date_block`` 只取第一块，导致同一天后续留痕在 UI、复盘和 RAG 中
    静默丢失。读取层聚合可以在不改写用户历史文件的前提下修复这个问题。
    """
    pattern = r"(^##\s+(\d{4}-\d{2}-\d{2})[^\n]*\n.*?)(?=^##\s+\d{4}-\d{2}-\d{2}|\Z)"
    return [m.group(1).strip() for m in re.finditer(pattern, log or "", re.DOTALL | re.MULTILINE)
            if m.group(2) == date_str]


def extract_date_block(log: str, date_str: str) -> str:
    """兼容接口：聚合同一天的全部块，而不是只返回第一块。"""
    return "\n\n".join(extract_date_blocks(log, date_str))


_LOG_META_RE = re.compile(r"<!--\s*offerclaw-log:\s*(\{.*?\})\s*-->")


def extract_log_entries(log: str, date_str: str) -> list[dict]:
    """把某日所有 Markdown 块转换为带稳定 ID 的执行事实视图。"""
    entries: list[dict] = []
    for idx, block in enumerate(extract_date_blocks(log, date_str)):
        meta: dict = {}
        mm = _LOG_META_RE.search(block)
        if mm:
            try:
                meta = json.loads(mm.group(1))
            except json.JSONDecodeError:
                meta = {}
        content_hash = hashlib.sha256(block.encode("utf-8")).hexdigest()
        parsed = _parse_single_log_block(block, date_str)
        entries.append({
            "log_id": meta.get("log_id") or f"legacy_{date_str.replace('-', '')}_{content_hash[:12]}",
            "date": date_str,
            "created_at": meta.get("created_at", ""),
            "plan_file": meta.get("plan_file", ""),
            "plan_hash": meta.get("plan_hash", ""),
            "task_id": meta.get("task_id", ""),
            "status": meta.get("status", "done" if parsed.get("completed") else "blocked"),
            "minutes": meta.get("minutes"),
            "attachment_refs": list(meta.get("attachment_refs") or []),
            "content_hash": content_hash,
            "block_index": idx,
            "block": block,
            **parsed,
        })
    return entries


def extract_recent_blocks(log: str, days: int = 7, end_date: str = "") -> str:
    """抓最近 days 天的所有 ## <YYYY-MM-DD> 块。"""
    today = datetime.date.fromisoformat(end_date) if end_date else datetime.date.today()
    blocks = []
    for i in range(days):
        d = (today - datetime.timedelta(days=i)).isoformat()
        b = extract_date_block(log, d)
        if b:
            blocks.append(b)
    return "\n\n".join(blocks)


def _profile_evidence_source_ids(log_block: str) -> list[str]:
    """Resolve log IDs embedded in Markdown to authoritative event IDs."""
    ids: list[str] = []
    for match in _LOG_META_RE.finditer(log_block or ""):
        try:
            log_id = str(json.loads(match.group(1)).get("log_id") or "")
        except (TypeError, json.JSONDecodeError):
            log_id = ""
        if not log_id:
            continue
        try:
            from memory_store import MemoryStore
            with MemoryStore()._connect() as conn:
                row = conn.execute(
                    "SELECT event_id FROM events WHERE entity_id=? AND kind='daily_log_recorded' "
                    "AND deleted_at IS NULL ORDER BY seq DESC LIMIT 1", (log_id,),
                ).fetchone()
            ids.append(str(row[0]) if row else log_id)
        except Exception:
            ids.append(log_id)
    return list(dict.fromkeys(ids))


def build_messages(prompt: str, source_policy: str, target_rules: str,
                   log_block: str, mode: str, date_str: str) -> list:
    system = (
        "你是 OfferClaw，按 summary_prompt.md 的 9 步流程做一次复盘。\n\n"
        f"========== summary_prompt.md ==========\n{prompt}\n\n"
        f"========== source_policy.md ==========\n{source_policy}\n\n"
        f"========== target_rules.md ==========\n{target_rules}\n"
    )
    source_event_ids = _profile_evidence_source_ids(log_block)
    structured_ask = (
        "\n\n在正文复盘之后，另起一段，用如下 ```json 代码块输出结构化复盘"
        "（供系统沉淀「次日调整规则」）：\n"
        "```json\n"
        '{\n'
        '  "main_tag": "补技能|补项目|补面试|岗位调研|投递准备",\n'
        '  "deviation_score": 0-100,\n'
        '  "completed": ["..."],\n'
        '  "incomplete": ["..."],\n'
        '  "blockers": ["..."],\n'
        '  "next_day_suggestion": "...",\n'
        '  "skill_evidence": [{"skill": "...", "level": "observed|practiced|verified", '
        '"evidence": "必须引用当天可核对的行为或产物"}],\n'
        '  "evidence_candidates": [{\n'
        '    "claim": "这条证据能证明什么",\n'
        '    "source_event_id": "从允许列表选择",\n'
        '    "source_quote": "必须逐字复制当天日志原文",\n'
        '    "capability_id": "已有稳定能力 ID；不知道就留空",\n'
        '    "new_capability_candidate": "没有已有 ID 时填写能力名称",\n'
        '    "activity_level": "observed|practiced|delivered",\n'
        '    "scope": "证据适用范围",\n'
        '    "verification_candidate": "source_grounded|artifact_checked|test_result|external_result|user_attested",\n'
        '    "criteria_ids": [], "explicit_statement": false,\n'
        '    "relation": "direct|partial|transferable|unrelated",\n'
        '    "rationale": "简短判断理由"\n'
        '  }]\n'
        '}\n'
        "```\n"
        "deviation_score：0=完全按计划，100=完全偏离。skill_evidence 只记录证据等级，"
        "禁止输出 mastered/已掌握；仅阅读为 observed，实际练习为 practiced，"
        "有测试、项目产物或量化结果才可标 verified。附件链接本身不是技能证据；"
        "未解析附件内容时不得据此标 practiced/verified。evidence_candidates 只能引用这些"
        f" source_event_id：{json.dumps(source_event_ids, ensure_ascii=False)}；"
        "若列表为空则 evidence_candidates 必须为空。source_quote 不得改写或概括原文。"
    )
    if mode == "daily":
        user = (
            f"请按 summary_prompt.md 单日模式复盘 {date_str}。\n"
            "下面是 daily_log.md 中该日期块的全文：\n\n"
            f"========== daily_log [{date_str}] ==========\n{log_block}"
            + structured_ask
        )
    else:
        user = (
            f"请按 summary_prompt.md 周度模式跑本周复盘（截至 {date_str}）。\n"
            "下面是最近 7 天的 daily_log.md 内容：\n\n"
            f"========== daily_log [recent 7d] ==========\n{log_block}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(messages, api_key) -> str:
    cfg = _resolve_chat_config(api_key)
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 3000,
    }
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    from day1_api_starter import chat_completion, extract_content  # A1 网关 + A2 防御解析
    data = chat_completion(
        f"{cfg['api_base']}/chat/completions",
        {"Authorization": f"Bearer {cfg['bearer']}", "Content-Type": "application/json"},
        payload, timeout=120,
    )
    return extract_content(data)


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


def _sig_tokens(text: str) -> set:
    """抽取显著词：英文/数字技术词（≥2 字符）+ 中文 2-gram。用于完成度比对。"""
    t = str(text)
    toks = set(m.lower() for m in re.findall(r"[A-Za-z][A-Za-z0-9_+\-\.]{1,}", t))
    # 中文按标点切块后取 2-gram
    for chunk in re.split(r"[\s\d\.,，。、:：;；()（）\[\]【】\-—/·]+", t):
        chunk = re.sub(r"[A-Za-z0-9_+\-\.]+", "", chunk)
        for i in range(len(chunk) - 1):
            toks.add(chunk[i:i + 2])
    return toks


# 比对时忽略的"框架性"高频词：几乎每条计划/留痕里都有，不构成完成证据
_STOP_2GRAMS = {"推进", "本周", "主线", "交付", "完成", "学习", "今日", "今天",
                "任务", "计划", "建议", "继续", "开始", "进行", "第周"}


def analyze_incomplete(done: list, planned: list) -> list:
    """对照"OfferClaw 今日计划"判定未完成项（确定性启发式，不调 LLM）。

    规则：planned 中某项，若没有任一 done 条目与其共享 ≥2 个显著词
    （或 ≥1 个英文技术词），则判为未完成。done 为空 → 全部未完成。
    晚间 LLM 复盘会在此基础上做更细的偏离度分析；这里只做即时反馈。
    """
    done = [str(d).strip() for d in (done or []) if str(d).strip()]
    planned = [str(p).strip() for p in (planned or []) if str(p).strip()]
    if not planned:
        return []
    if not done:
        return planned[:]
    done_toks = set()
    done_tech = set()
    for d in done:
        toks = _sig_tokens(d)
        done_toks |= toks
        done_tech |= {t for t in toks if re.match(r"^[a-z]", t)}
    incomplete = []
    for p in planned:
        p_toks = _sig_tokens(p) - _STOP_2GRAMS
        p_tech = {t for t in p_toks if re.match(r"^[a-z]", t)}
        overlap = p_toks & done_toks
        tech_hit = bool(p_tech & done_tech)
        if tech_hit or len(overlap) >= 2:
            continue
        incomplete.append(p)
    return incomplete


def append_structured_daily_log(tag: str = "", done=None, todo=None,
                                notes: str = "", date_str: str = None, *,
                                task_id: str = "", status: str = "done",
                                minutes: int | None = None,
                                plan_file: str = "", plan_hash: str = "",
                                attachment_refs=None, operation_id: str | None = None) -> dict:
    """把结构化留痕写入 daily_log.md，字段用 ### 分节（与 P2 _parse_log_block 对齐）。

    CLI（offerclaw_cli.cmd_log）、Web 表单（/api/daily/log）、GitHub 同步共用此写入器，
    保证全项目留痕格式一致、能被晚间复盘正确解析。
    """
    import datetime as _dt
    done = [d for d in (done or []) if str(d).strip()]
    todo = [t for t in (todo or []) if str(t).strip()]
    notes = (notes or "").strip()
    date_str = date_str or _dt.date.today().isoformat()
    if status not in {"done", "partial", "blocked", "deferred"}:
        raise ValueError("status 必须是 done / partial / blocked / deferred")
    if minutes is not None and (int(minutes) < 0 or int(minutes) > 1440):
        raise ValueError("minutes 必须在 0~1440 之间")
    log_id = f"log_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}"
    created_at = _dt.datetime.now().isoformat(timespec="seconds")
    attachment_refs = [str(x).strip() for x in (attachment_refs or []) if str(x).strip()]
    meta = {
        "log_id": log_id, "created_at": created_at, "status": status,
        "task_id": (task_id or "").strip(), "minutes": minutes,
        "plan_file": (plan_file or "").strip(), "plan_hash": (plan_hash or "").strip(),
        "attachment_refs": attachment_refs,
    }

    lines = [f"\n## {date_str}\n", f"<!-- offerclaw-log: {json.dumps(meta, ensure_ascii=False)} -->"]
    if tag:
        lines += ["### 今日主线标签", tag, ""]
    lines += ["### 已完成"]
    lines += [f"- {d}" for d in done] or ["- 【待补充】"]
    lines += ["", "### 未完成"]
    lines += [f"- {t}" for t in todo] or ["- （无）"]
    if notes:
        lines += ["", "### 学习留痕", notes]
    entry = "\n".join(lines) + "\n"

    payload = {
        "log_id": log_id, "date": date_str, "status": status, "task_id": task_id,
        "tag": tag, "done": done, "incomplete": todo, "notes": notes,
        "minutes": minutes, "plan_file": plan_file, "plan_hash": plan_hash,
        "attachment_refs": attachment_refs,
    }
    from memory_transactions import write_text_with_memory
    memory_result = write_text_with_memory(
        DAILY_LOG_PATH, entry, event_kind="daily_log_recorded",
        event_payload=payload,
        event_options={"actor": "user", "source": "daily_log_service",
                       "entity_type": "daily_log", "entity_id": log_id,
                       "business_date": date_str},
        operation_id=operation_id, append=True,
    )
    result = {
        "status": "ok", "date": date_str, "log_id": log_id,
        "created_at": created_at, "main_tag": tag,
        "done_count": len(done), "todo_count": len(todo), "has_notes": bool(notes),
    }
    result.update(memory_result)
    return result


def _parse_single_log_block(block: str, date_str: str) -> dict:
    """从 daily_log 日期块里确定性抽取结构化字段（不依赖 LLM）。

    抓取：主线标签、实际完成项、未完成/偏离信号。daily_log 模板含
    「今日主线标签 / 实际完成 / 偏离度判断 / 明日建议」等字段。
    """
    main_tag = ""
    m = re.search(r"主线标签[：:\s]*([^\n（(]+)", block)
    if m:
        main_tag = m.group(1).strip(" `*")

    def _section(name: str) -> list[str]:
        # 抓 "### <name>" 或 "<name>：" 后面的列表/段落，到下一个 ## / ### 为止。
        # 注意：行尾 \n 已被 .*\n? 吃掉，故下一标题的 lookahead 不带前导 \n。
        pat = rf"(?:#+\s*{name}|{name})[：:\s]*\n?((?:.*\n?)*?)(?=#{{1,6}}\s|\Z)"
        mm = re.search(pat, block)
        if not mm:
            return []
        items = []
        for ln in mm.group(1).splitlines():
            s = ln.strip(" -*•\t")
            if s and not s.startswith("#") and len(s) >= 2:
                items.append(s)
        return items[:8]

    # 字段名与 daily_log 模板（已完成/未完成）及微信留痕（cmd_log）对齐
    completed = _section("已完成") or _section("实际完成") or _section("实际")
    incomplete = _section("未完成")
    return {
        "date": date_str,
        "main_tag": main_tag,
        "completed": completed,
        "incomplete": incomplete,
    }


def _parse_log_block(block: str, date_str: str) -> dict:
    """确定性聚合同一天的全部留痕块，并去重保序。"""
    chunks = extract_date_blocks(block, date_str)
    if not chunks:
        chunks = [block] if block.strip() else []
    parsed = [_parse_single_log_block(chunk, date_str) for chunk in chunks]

    def _merge(key: str) -> list[str]:
        return list(dict.fromkeys(item for row in parsed for item in (row.get(key) or [])))

    tags = [row.get("main_tag", "") for row in parsed if row.get("main_tag")]
    return {"date": date_str, "main_tag": tags[-1] if tags else "",
            "completed": _merge("completed"), "incomplete": _merge("incomplete")}


def _extract_llm_json(summary_text: str) -> dict:
    """若 LLM 输出里带 ```json ... ``` 结构化块，鲁棒解析出来；失败返回 {}。"""
    import json as _json
    m = re.search(r"```json\s*(\{.*?\})\s*```", summary_text, re.DOTALL)
    if not m:
        m = re.search(r"(\{[^{}]*\"deviation_score\"[^{}]*\})", summary_text, re.DOTALL)
    if not m:
        return {}
    try:
        return _json.loads(m.group(1))
    except Exception:
        return {}


# A6: 结构化 JSON 解析埋点（让「LLM 没给 json 块」可观测而非无声）
_JSON_PARSE_STATS = {"ok": 0, "missing": 0, "repaired": 0, "repair_failed": 0}


def extract_json_with_repair(first_text: str, repair_fn=None) -> dict:
    """[A6] 抽结构化 json；首轮失败且提供 repair_fn 时，调一次 repair_fn() 拿重发文本再抽
    （reflexion 式自纠）。同时打点供监控「json 解析失败率」。"""
    js = _extract_llm_json(first_text)
    if js:
        _JSON_PARSE_STATS["ok"] += 1
        return js
    if repair_fn is None:
        _JSON_PARSE_STATS["missing"] += 1
        return {}
    try:
        retry = repair_fn() or ""
    except Exception:
        _JSON_PARSE_STATS["repair_failed"] += 1
        return {}
    js2 = _extract_llm_json(retry)
    _JSON_PARSE_STATS["repaired" if js2 else "missing"] += 1
    return js2


def verify_reflection_consistency(reflection: dict, tol: int = 40) -> dict:
    """[L6] 反思二阶 verifier：校验**内容逻辑一致性**（非 JSON 格式层）。

    A6 的 repair 只保证「能抽出 JSON」；本函数进一步防「格式完美但逻辑矛盾」的反思——
    LLM 可能给 deviation_score=0 却列 10 项未完成。检查 score 与未完成比例是否自洽，
    不一致则以**确定性比例值**纠正 score。返回 ``{consistent, issues, deviation_score(纠正后)}``。
    """
    completed = reflection.get("completed") or []
    incomplete = reflection.get("incomplete") or []
    score = int(reflection.get("deviation_score") or 0)
    total = len(completed) + len(incomplete)
    ratio = int(round(100 * len(incomplete) / total)) if total else 0
    issues = []
    if total > 0 and abs(score - ratio) > tol:
        issues.append(f"deviation_score={score} 与未完成比例隐含值 {ratio} 相差>{tol}"
                      f"（completed={len(completed)} incomplete={len(incomplete)}）")
    if score == 0 and len(incomplete) > 0:
        issues.append(f"deviation_score=0 但有 {len(incomplete)} 项未完成")
    if score >= 60 and total > 0 and len(incomplete) == 0:
        issues.append(f"deviation_score={score} 偏高但零未完成")
    consistent = not issues
    corrected = score if consistent else ratio
    return {"consistent": consistent, "issues": issues,
            "deviation_score": max(0, min(100, corrected))}


def build_structured_reflection(block: str, date_str: str, summary_text: str,
                                repair_fn=None) -> dict:
    """合并确定性解析 + LLM JSON 增强，产出一条结构化复盘。

    deviation_score 优先用 LLM 给的；没有就按 incomplete/completed 比例估算。
    A6: 传 repair_fn 时，summary_text 抽不到 json 块会自纠重试一次。
    L6: 产出后过二阶 verifier，纠正「格式完美但逻辑矛盾」的 deviation_score。
    """
    base = _parse_log_block(block, date_str)
    llm = extract_json_with_repair(summary_text, repair_fn)

    completed = llm.get("completed") or base["completed"]
    incomplete = llm.get("incomplete") or base["incomplete"]

    if "deviation_score" in llm:
        score = int(llm.get("deviation_score") or 0)
    else:
        total = len(completed) + len(incomplete)
        score = int(round(100 * len(incomplete) / total)) if total else 0

    allowed_levels = {"observed", "practiced", "verified"}
    skill_evidence = []
    for raw in (llm.get("skill_evidence") or [])[:12]:
        if not isinstance(raw, dict):
            continue
        skill = str(raw.get("skill", "")).strip()[:100]
        level = str(raw.get("level", "")).strip().lower()
        evidence = str(raw.get("evidence", "")).strip()[:500]
        if skill and level in allowed_levels and evidence:
            skill_evidence.append({"skill": skill, "level": level, "evidence": evidence})

    evidence_candidates = []
    from profile_review import EvidenceCandidate
    for raw in (llm.get("evidence_candidates") or [])[:20]:
        try:
            evidence_candidates.append(
                EvidenceCandidate.model_validate(raw).model_dump(mode="json")
            )
        except Exception:
            # One malformed model item must not discard other valid evidence.
            continue

    reflection = {
        "date": date_str,
        "date_from": date_str,
        "date_to": date_str,
        "kind": "daily",
        "main_tag": llm.get("main_tag") or base["main_tag"],
        "deviation_score": max(0, min(100, score)),
        "completed": completed,
        "incomplete": incomplete,
        "blockers": llm.get("blockers", []) or [],
        "next_day_suggestion": llm.get("next_day_suggestion", "") or "",
        "skill_evidence": skill_evidence,
        "evidence_candidates": evidence_candidates,
    }
    # [L6] 二阶 verifier：内容逻辑一致性，不一致则纠正 score 并记 _verifier
    v = verify_reflection_consistency(reflection)
    reflection["deviation_score"] = v["deviation_score"]
    if not v["consistent"]:
        reflection["_verifier"] = {"consistent": False, "issues": v["issues"]}
    return reflection


def record_and_distill(reflection: dict) -> dict:
    """把结构化复盘写入分层 memory 并沉淀调整规则。失败静默（不阻塞复盘落盘）。"""
    try:
        from memory_layers import (
            EpisodicMemory, SemanticMemory,
            record_reflection, distill_reflections_to_semantic, get_active_adjustments,
        )
        epi, sem = EpisodicMemory(), SemanticMemory()
        event = record_reflection(epi, reflection)
        distill_reflections_to_semantic(epi, sem)
        evidence_result = {"status": "skipped"}
        try:
            from profile_review import ProfileRepository
            evidence_result = ProfileRepository(epi.store).ingest_reflection_evidence(reflection)
        except Exception as exc:
            evidence_result = {"status": "error", "error": str(exc)}
        return {"ok": True, "event_id": event.get("event_id", ""),
                "evidence": evidence_result,
                "active_adjustments": get_active_adjustments(sem)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def save(content: str, mode: str, date_str: str, metadata: dict | None = None) -> str:
    """版本化保存完整复盘；同一天重复执行不会覆盖旧版。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "weekly" if mode == "weekly" else "daily"
    base = os.path.join(OUTPUT_DIR, f"summary_{suffix}_{date_str}.md")
    path = base
    if os.path.exists(path):
        stamp = datetime.datetime.now().strftime("%H%M%S_%f")
        path = os.path.join(OUTPUT_DIR, f"summary_{suffix}_{date_str}_{stamp}.md")
    body = content
    if metadata:
        body = (f"<!-- offerclaw-reflection: {json.dumps(metadata, ensure_ascii=False)} -->\n\n"
                + content.lstrip())
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def main():
    parser = argparse.ArgumentParser(description="OfferClaw 复盘工具")
    parser.add_argument("--date", help="单日模式日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--weekly", action="store_true", help="周度复盘")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_local_env()
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[ERROR] 未检测到 {API_KEY_ENV}")
        sys.exit(1)

    today = datetime.date.today().isoformat()
    date_str = args.date or today
    mode = "weekly" if args.weekly else "daily"

    print(f"[1/3] 读取依赖文件 + daily_log...")
    prompt = read_text(SUMMARY_PROMPT_PATH)
    sp = read_text(SOURCE_POLICY_PATH)
    tr = read_text(TARGET_RULES_PATH)
    log = read_text(DAILY_LOG_PATH)

    if mode == "daily":
        block = extract_date_block(log, date_str)
        if not block:
            print(f"[ERROR] daily_log.md 中找不到 {date_str} 的执行记录；为避免伪造日期复盘，本次不生成。")
            sys.exit(2)
    else:
        block = extract_recent_blocks(log, days=7, end_date=date_str)
        if not block:
            print("[ERROR] 最近 7 天无执行记录；为避免用错误日期内容生成周复盘，本次不生成。")
            sys.exit(2)

    print(f"[2/3] 调用 LLM ({mode}, {date_str}) ...")
    messages = build_messages(prompt, sp, tr, block, mode, date_str)
    try:
        out = call_llm(messages, api_key)
    except Exception as e:  # A2: LLM 失败给可读降级而非 traceback
        import sys as _sys
        from day1_api_starter import llm_error_detail
        print(f"[ERROR] LLM 调用失败，复盘未生成：{llm_error_detail(e)}", file=_sys.stderr)
        _sys.exit(1)

    print(f"[3/3] 写入文件...")
    reflection: dict = {}
    if mode == "daily":
        def _repair_json():  # A6: 让模型只重发结构化 json 块（reflexion 自纠）
            return call_llm(messages + [
                {"role": "assistant", "content": out},
                {"role": "user", "content": "上一次复盘未按要求附带 ```json 结构化块。请只重新输出那段 "
                 "```json 代码块（含 deviation_score / main_tag / completed / incomplete / blockers / "
                 "next_day_suggestion / skill_evidence），不要别的内容；skill_evidence 等级只能是 "
                 "observed / practiced / verified。"},
            ], api_key)
        reflection = build_structured_reflection(block, date_str, out, repair_fn=_repair_json)
        log_entries = extract_log_entries(log, date_str)
        reflection["source_log_ids"] = [e["log_id"] for e in log_entries]
        reflection["source_status"] = "valid" if log_entries else "orphaned"
        seed = f"daily|{date_str}|{datetime.datetime.now().isoformat()}|{out}"
        reflection["reflection_id"] = "refl_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        reflection["content_hash"] = hashlib.sha256(out.encode("utf-8")).hexdigest()
        path = save(out, mode, date_str, reflection)
        reflection["summary_path"] = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
        result = record_and_distill(reflection)
        if result.get("ok"):
            print(f"[MEMORY] 已记录结构化复盘（偏离度={reflection['deviation_score']}，"
                  f"完成 {len(reflection['completed'])} / 未完成 {len(reflection['incomplete'])}）")
            adj = result.get("active_adjustments", [])
            if adj:
                print("[ADJUST] 当前生效的次日调整规则：")
                for a in adj:
                    print(f"  - {a}")
        else:
            print(f"[MEMORY] 跳过沉淀：{result.get('error')}")
    else:
        dates = sorted(set(re.findall(r"^##\s+(\d{4}-\d{2}-\d{2})", block, re.MULTILINE)))
        meta = {
            "reflection_id": "refl_" + hashlib.sha256(
                f"weekly|{date_str}|{datetime.datetime.now().isoformat()}|{out}".encode("utf-8")
            ).hexdigest()[:20],
            "kind": "weekly", "date_from": dates[0] if dates else date_str,
            "date_to": dates[-1] if dates else date_str,
            "source_log_ids": [e["log_id"] for d in dates for e in extract_log_entries(log, d)],
            "source_status": "valid" if dates else "orphaned",
            "content_hash": hashlib.sha256(out.encode("utf-8")).hexdigest(),
        }
        path = save(out, mode, date_str, meta)
        meta.update({"date": meta["date_to"], "summary_path": os.path.relpath(
            path, os.path.dirname(os.path.abspath(__file__))), "main_tag": "周度复盘",
            "completed": [], "incomplete": [], "blockers": [], "next_day_suggestion": "",
            "skill_evidence": []})
        try:
            from memory_layers import EpisodicMemory, record_reflection
            record_reflection(EpisodicMemory(), meta)
        except Exception:
            pass

    print(f"[OK] 复盘已保存：{path}")

    print("-" * 60)
    print(out[:1500])
    if len(out) > 1500:
        print(f"\n...（截断，完整 {len(out)} 字符见 {path}）")


if __name__ == "__main__":
    main()
