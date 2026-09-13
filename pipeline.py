# -*- coding: utf-8 -*-
"""
OfferClaw · 端到端一键流水线 (pipeline.py)

职责：
    一条命令完成 "JD → 匹配 → 缺口 → 4 周计划 → 写回 daily_log" 全流程。
    把散落的 match_job.py / plan_gen.py 串成可演示的闭环。

使用：
    python pipeline.py                   # 用 match_job.DEMO_JD
    python pipeline.py --jd path/to.txt  # 自定义 JD 文本文件
    python pipeline.py --no-plan         # 只跑匹配，不生成计划
"""

import argparse
import datetime
import os
import sys

from match_job import run_match, format_report, DEMO_JD
from profile_loader import load_profile
from plan_gen import (
    PROFILE_PATH, PLAN_PROMPT_PATH, DAILY_LOG_PATH,
    SOURCE_POLICY_PATH, TARGET_RULES_PATH,
    read_text, build_messages, call_llm_plain, save_plan,
)
from agent_demo import API_KEY_ENV, load_local_env


def gaps_to_text(report) -> str:
    """把 MatchReport.gap_list (dict[str, list[str]]) 拍平成 plan_prompt 能吃的文本。"""
    lines = []
    for category, items in report.gap_list.items():
        lines.append(f"## {category}")
        if not items:
            lines.append("- （无）")
        else:
            for it in items:
                lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines).strip()


def append_daily_log(jd_title: str, conclusion: str, plan_path: str | None) -> None:
    """把本次流水线结果追加到 daily_log.md 末尾。"""
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now().strftime("%H:%M")
    block = [
        "",
        f"## {today} · pipeline 自动留痕",
        f"- 时间：{now}",
        f"- 岗位：{jd_title}",
        f"- 匹配结论：{conclusion}",
    ]
    if plan_path:
        block.append(f"- 计划产物：`{plan_path}`")
    else:
        block.append("- 计划：本次未生成（--no-plan）")
    block.append("")
    with open(DAILY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(block))


def main():
    parser = argparse.ArgumentParser(description="OfferClaw 端到端流水线")
    parser.add_argument("--jd", help="JD 文本文件路径（默认用 match_job.DEMO_JD）")
    parser.add_argument("--title", default="AI 应用开发工程师（实习）", help="JD 名称")
    parser.add_argument("--no-plan", action="store_true", help="只跑匹配，不生成 4 周计划")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_local_env()
    if not args.no_plan and not os.environ.get(API_KEY_ENV):
        print(f"[ERROR] 未检测到 {API_KEY_ENV}，无法生成计划。")
        print("       可加 --no-plan 仅跑匹配。")
        sys.exit(1)

    # === Step 1: 匹配 ===
    print("=" * 60)
    print("[1/3] 运行岗位匹配...")
    print("=" * 60)
    jd_text = read_text(args.jd) if args.jd else DEMO_JD
    profile = load_profile()
    report = run_match(profile, jd_text, jd_title=args.title)
    print(format_report(report))

    # === Step 2: 计划 ===
    plan_path = None
    if not args.no_plan:
        print()
        print("=" * 60)
        print("[2/3] 调用 LLM 生成 4 周计划...")
        print("=" * 60)
        api_key = os.environ.get(API_KEY_ENV)
        gaps_text = gaps_to_text(report)
        messages = build_messages(
            profile=read_text(PROFILE_PATH),
            plan_prompt=read_text(PLAN_PROMPT_PATH),
            daily_log=read_text(DAILY_LOG_PATH),
            source_policy=read_text(SOURCE_POLICY_PATH),
            target_rules=read_text(TARGET_RULES_PATH),
            gaps=gaps_text,
        )
        plan_text = call_llm_plain(messages, api_key)
        plan_path = save_plan(plan_text)
        print(f"[OK] 计划已生成：{plan_path}")
        print("-" * 60)
        print(plan_text[:1200])
        if len(plan_text) > 1200:
            print(f"\n...（截断，完整 {len(plan_text)} 字符见 {plan_path}）")

    # === Step 3: 写回 daily_log ===
    print()
    print("=" * 60)
    print("[3/3] 写回 daily_log.md ...")
    print("=" * 60)
    append_daily_log(args.title, report.conclusion, plan_path)
    print(f"[OK] 已追加到 {DAILY_LOG_PATH}")
    print()
    print("流水线完成。")


if __name__ == "__main__":
    main()
