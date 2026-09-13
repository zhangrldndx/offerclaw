# -*- coding: utf-8 -*-
"""github_tracker.py — 从 GitHub 公开活动自动抓取学习留痕（P3）

把用户的 GitHub 公开 commit / 建仓活动转成 daily_log 的"已完成"留痕，
减少手动记录负担（写代码这件事本身就是最好的留痕）。

- 公开事件无需 token（GitHub 未认证限流 60 次/小时，个人够用）。
- 复用 summary_tool.append_structured_daily_log 写入，格式与 CLI/Web 表单一致。
- 纯解析逻辑（parse_push_events / 日期过滤）可离线单测。

用法：
    python github_tracker.py preview <username> [YYYY-MM-DD]   # 只看不写
    python github_tracker.py sync <username> [YYYY-MM-DD]      # 写入 daily_log
（username 省略时读环境变量 GITHUB_USERNAME）
"""

import argparse
import datetime
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

GITHUB_EVENTS_API = "https://api.github.com/users/{user}/events/public?per_page=100"


# =====================================================
# 纯解析逻辑（可离线单测）
# =====================================================

def parse_push_events(events: list, date_str: str = None) -> list:
    """从事件列表里抽取 commit 留痕条目。

    返回 ``["<repo>: <commit message 首行>", ...]``。
    - 只取 PushEvent（提交）与 CreateEvent（建仓/建分支）。
    - 传 ``date_str``（YYYY-MM-DD）时只保留当天事件。
    - commit message 只取首行、去重。
    """
    items = []
    seen = set()
    for ev in events or []:
        etype = ev.get("type")
        created = (ev.get("created_at") or "")[:10]
        if date_str and created != date_str:
            continue
        repo = (ev.get("repo") or {}).get("name", "?").split("/")[-1]
        if etype == "PushEvent":
            for c in (ev.get("payload") or {}).get("commits", []):
                msg = (c.get("message") or "").strip().splitlines()
                first = msg[0].strip() if msg else ""
                if not first:
                    continue
                line = f"{repo}: {first}"
                if line not in seen:
                    seen.add(line)
                    items.append(line)
        elif etype == "CreateEvent":
            ref_type = (ev.get("payload") or {}).get("ref_type", "")
            if ref_type in ("repository", "branch"):
                line = f"{repo}: 新建{('仓库' if ref_type=='repository' else '分支')}"
                if line not in seen:
                    seen.add(line)
                    items.append(line)
    return items


# =====================================================
# I/O
# =====================================================

def fetch_events(username: str) -> list:
    """拉取 GitHub 公开事件（未认证）。失败抛 RuntimeError。"""
    import requests
    url = GITHUB_EVENTS_API.format(user=username)
    r = requests.get(url, timeout=20, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "OfferClaw-github-tracker",
    })
    if r.status_code == 404:
        raise RuntimeError(f"GitHub 用户不存在或无公开活动：{username}")
    if r.status_code == 403:
        raise RuntimeError("GitHub API 限流（未认证 60 次/小时），稍后再试。")
    r.raise_for_status()
    return r.json()


def _resolve_username(username: str = None) -> str:
    if username:
        return username
    from day1_api_starter import load_local_env
    load_local_env()
    u = os.environ.get("GITHUB_USERNAME", "").strip()
    if not u:
        raise RuntimeError("未指定 GitHub 用户名（命令行参数或 .env.local 的 GITHUB_USERNAME）")
    return u


def cmd_preview(username: str = None, date_str: str = None) -> dict:
    user = _resolve_username(username)
    date_str = date_str or datetime.date.today().isoformat()
    items = parse_push_events(fetch_events(user), date_str)
    return {"status": "ok", "user": user, "date": date_str,
            "count": len(items), "items": items}


def cmd_sync(username: str = None, date_str: str = None) -> dict:
    """抓当日 commit → 写入 daily_log 的"已完成"（主线标签：补项目）。"""
    user = _resolve_username(username)
    date_str = date_str or datetime.date.today().isoformat()
    items = parse_push_events(fetch_events(user), date_str)
    if not items:
        return {"status": "empty", "user": user, "date": date_str,
                "message": "当天无 GitHub 公开提交，未写入。"}
    from summary_tool import append_structured_daily_log
    notes = f"（GitHub @{user} 自动同步 {len(items)} 条提交）"
    result = append_structured_daily_log(
        tag="补项目", done=items, notes=notes, date_str=date_str,
    )
    return {"status": "ok", "user": user, "date": date_str,
            "synced": len(items), "items": items, "write": result}


def _out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="OfferClaw GitHub 留痕同步")
    sub = parser.add_subparsers(dest="cmd")
    for name in ("preview", "sync"):
        p = sub.add_parser(name)
        p.add_argument("username", nargs="?", default=None)
        p.add_argument("date", nargs="?", default=None)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        if args.cmd == "preview":
            _out(cmd_preview(args.username, args.date))
        elif args.cmd == "sync":
            _out(cmd_sync(args.username, args.date))
        else:
            parser.print_help(); sys.exit(1)
    except Exception as e:
        _out({"status": "error", "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
