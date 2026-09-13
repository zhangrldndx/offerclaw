"""OfferClaw 运行数据报告 —— 一条命令出「个人生产环境」运营账本。

只读统计,不改任何状态。数据源(全部真实落盘产物):
  logs/api.log          结构化 API 请求日志(REQ/RES JSON 行)
  logs/llm_usage.jsonl  LLM 用量账本(Round 9 usage meter)
  ~/.openclaw/cron/     定时任务注册与运行状态(微信推送链路)
  daily_log.md / applications.md   求职外循环真实使用痕迹
  knowledge_base/ + doctor 口径     知识库规模与岗位方向覆盖

用途:面试现场回答「系统有没有运行数据」——python usage_report.py
诚实口径:这是单用户个人生产工具,数字是自用+开发流量,不冒充多用户产品。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def api_stats() -> dict:
    reqs, days, eps, biz = 0, set(), Counter(), 0
    p = ROOT / "logs" / "api.log"
    if not p.exists():
        return {}
    for line in p.open(errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = r.get("msg", "")
        if msg.startswith("REQ "):
            reqs += 1
            days.add(r.get("ts", "")[:10])
            ep = msg.split()[2].split("?")[0]
            eps[ep] += 1
            if ep.startswith("/api/"):
                biz += 1
    return {
        "total_requests": reqs,
        "business_api_requests": biz,
        "active_days": len(days),
        "first_day": min(days) if days else "-",
        "last_day": max(days) if days else "-",
        "top_endpoints": eps.most_common(6),
    }


def llm_stats() -> dict:
    p = ROOT / "logs" / "llm_usage.jsonl"
    if not p.exists():
        return {}
    calls, tokens, days, models = 0, 0, set(), Counter()
    for line in p.open(errors="replace"):
        try:
            u = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls += 1
        tokens += u.get("total_tokens") or 0
        days.add(u.get("ts", "")[:10])
        models[u.get("model", "?")] += 1
    return {"calls": calls, "tokens": tokens, "days": len(days), "models": models.most_common(3)}


def cron_stats() -> dict:
    p = Path.home() / ".openclaw" / "cron" / "jobs.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(errors="replace"))
    jobs = data.get("jobs", data)
    names = []
    if isinstance(jobs, list):
        names = [j.get("name", j.get("id", "?")) for j in jobs]
    elif isinstance(jobs, dict):
        names = [v.get("name", k) for k, v in jobs.items()]
    return {"jobs": len(names), "names": names[:5]}


def loop_stats() -> dict:
    out = {}
    for f in ("daily_log.md", "applications.md"):
        p = ROOT / f
        if p.exists():
            t = p.read_text(errors="replace")
            out[f] = len(set(re.findall(r"20\d\d-\d\d-\d\d", t)))
    return out


def kb_stats() -> dict:
    kb = ROOT / "knowledge_base"
    if not kb.exists():
        return {}
    cats = {
        d.name: sum(1 for _ in d.rglob("*.md"))
        for d in sorted(kb.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    }
    return {"categories": cats, "docs_ingested": sum(cats.values())}


def main() -> None:
    a, l, c, lp, kb = api_stats(), llm_stats(), cron_stats(), loop_stats(), kb_stats()
    print("=" * 62)
    print("OfferClaw 运行数据报告(单用户个人生产环境,真实落盘统计)")
    print("=" * 62)
    if a:
        print(f"API 服务   : {a['total_requests']:,} 次请求(业务端点 {a['business_api_requests']:,} 次)")
        print(f"             {a['active_days']} 个活跃日,{a['first_day']} → {a['last_day']}")
        print(f"             Top 端点: {', '.join(f'{e} ×{n}' for e, n in a['top_endpoints'][:4])}")
    if l:
        print(f"LLM 用量   : {l['calls']:,} 次调用 / {l['tokens']:,} tokens(账本跨 {l['days']} 天,持续累积)")
    if c:
        print(f"自动化链路 : {c['jobs']} 个 OpenClaw 定时任务 → 微信推送(2026-07-03 上线)")
    if lp:
        print(f"求职外循环 : daily_log {lp.get('daily_log.md', 0)} 个活跃日 · 投递跟踪 {lp.get('applications.md', 0)} 个活跃日")
    if kb:
        cats = " / ".join(f"{k} {v}" for k, v in kb["categories"].items())
        print(f"知识库覆盖 : {kb['docs_ingested']} 篇已入库({cats});3 个岗位方向路由(backend/algorithm/career)")
    print("-" * 62)
    print("口径声明:单用户工具;请求数含开发调试流量;不声称多用户规模。")


if __name__ == "__main__":
    main()
