# -*- coding: utf-8 -*-
"""eval_match.py — OfferClaw 匹配结论评估（确定性 baseline + 可选 LLM-as-judge）

为什么需要：
    既有 RAG eval 只评检索召回，``match_job`` 的"三档结论"质量没有自动化门禁。
    本脚本提供两类评估：
      1) **确定性 baseline**：把 ``tests/match_eval_set.json`` 喂给 ``match_job.run_match``，
         比对 ``expected_status`` / ``expected_direction``，跑出整体 / 分桶准确率与
         混淆矩阵。不依赖 LLM，任何时候都能跑（CI / doctor / 回归）。
      2) **LLM-as-judge**：可选 ``--judge`` 开关。把 (profile, jd, baseline_conclusion)
         交给 LLM 评 1-5 分（仅做合理性二次审视，不替代规则结论），用于发现
         规则版输出的"看起来对但其实勉强"的边界样本。当 ``ZHIPU_API_KEY`` 缺失
         时直接 skip，不抛错。

CLI：
    python eval_match.py              # 跑 baseline，打印 markdown 报告
    python eval_match.py --judge      # 加 LLM-as-judge
    python eval_match.py --json out.json  # 同时落盘机器可读结果
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

EVAL_SET_PATH = os.path.join(BASE_DIR, "tests", "match_eval_set.json")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

STATUSES = ("当前适合投递", "当前暂不建议投递", "中长期可转向")
DIRECTIONS = ("主方向", "派生方向", "不考虑")


def load_eval_set(path: str | None = None) -> dict:
    with open(path or EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profile_fixture(profile_id: str) -> dict:
    p = os.path.join(PROFILES_DIR, f"{profile_id}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"profile fixture not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(items: list[dict]) -> dict:
    """跑确定性 baseline，返回 {results, summary}."""
    from match_job import run_match

    results: list[dict] = []
    for it in items:
        profile = load_profile_fixture(it["profile_id"])
        report = run_match(profile, it["jd_text"], jd_title=it.get("jd_title", ""))
        ok_status = report.conclusion == it["expected_status"]
        ok_direction = report.direction == it["expected_direction"]
        results.append({
            "id": it["id"],
            "profile_id": it["profile_id"],
            "jd_title": it.get("jd_title", ""),
            "expected_status": it["expected_status"],
            "actual_status": report.conclusion,
            "status_ok": ok_status,
            "expected_direction": it["expected_direction"],
            "actual_direction": report.direction,
            "direction_ok": ok_direction,
            "rationale": it.get("rationale", ""),
            "gap_total": sum(len(v) for v in report.gap_list.values()
                             if isinstance(v, list)),
        })

    n = len(results)
    status_acc = sum(r["status_ok"] for r in results) / n if n else 0.0
    direction_acc = sum(r["direction_ok"] for r in results) / n if n else 0.0

    # 按 expected_status 分桶
    per_status: dict[str, dict] = {}
    for s in STATUSES:
        bucket = [r for r in results if r["expected_status"] == s]
        per_status[s] = {
            "n": len(bucket),
            "status_acc": (sum(r["status_ok"] for r in bucket) / len(bucket))
                          if bucket else 0.0,
        }

    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        confusion[r["expected_status"]][r["actual_status"]] += 1

    return {
        "results": results,
        "summary": {
            "n": n,
            "status_acc": status_acc,
            "direction_acc": direction_acc,
            "per_status": per_status,
            "confusion": {k: dict(v) for k, v in confusion.items()},
        },
    }


# =====================================================
# 可选 LLM-as-judge
# =====================================================

_JUDGE_PROMPT = """你是 OfferClaw 的资深求职顾问，需要审视一份"规则版"匹配结论是否合理。

【用户画像简版】
{profile_json}

【JD 原文】
{jd_text}

【规则版给出的结论】
status = {actual_status}
direction = {actual_direction}
gap_total = {gap_total}

【任务】
1. 仅判断该结论是否合理（不要重新做匹配）。
2. 输出 JSON：{{"score": 1-5 整数, "verdict": "合理|勉强|不合理", "reason": "1-2 句"}}
3. 不要输出 JSON 以外的任何内容。

注意：合理=4-5 分，勉强=3 分，不合理=1-2 分。"""


def llm_judge_one(profile: dict, jd_text: str, baseline: dict) -> dict:
    """单条 LLM 二次审视。失败时返回 {error: ...}，不抛异常。

    v0.6.3 起走 OpenAI 兼容代理（gpt-5.4 + medium effort）。
    """
    from day1_api_starter import API_KEY_ENV, build_zhipu_jwt, get_llm_config

    cfg = get_llm_config()
    api_key = cfg["api_key"]
    if not api_key:
        return {"error": f"no_{API_KEY_ENV.lower()}", "score": None}
    bearer = build_zhipu_jwt(api_key) if cfg["is_zhipu"] else api_key

    profile_simple = {k: profile.get(k) for k in (
        "学历", "专业", "所在地", "可接受地域", "方向优先级",
        "明确不做", "熟练技能", "会用技能", "项目数量", "实习数量")}
    prompt = _JUDGE_PROMPT.format(
        profile_json=json.dumps(profile_simple, ensure_ascii=False),
        jd_text=jd_text[:1200],
        actual_status=baseline["actual_status"],
        actual_direction=baseline["actual_direction"],
        gap_total=baseline.get("gap_total", 0),
    )
    try:
        import requests as _r
        body: dict = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        if cfg["reasoning_effort"]:
            body["reasoning_effort"] = cfg["reasoning_effort"]
        resp = _r.post(
            f"{cfg['api_base']}/chat/completions",
            headers={
                "Authorization": f"Bearer {bearer}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        txt = resp.json()["choices"][0]["message"]["content"].strip()
        # 容错：抠出第一个 { ... }
        l, r = txt.find("{"), txt.rfind("}")
        if l == -1 or r == -1:
            return {"error": "no_json", "raw": txt[:200]}
        return json.loads(txt[l:r + 1])
    except Exception as e:
        return {"error": str(e)[:200]}


def run_with_judge(items: list[dict]) -> dict:
    """跑 baseline + LLM judge。LLM 失败时只标 skipped，不影响 baseline 结论。"""
    base = run_baseline(items)
    for r in base["results"]:
        idx = next(i for i, it in enumerate(items) if it["id"] == r["id"])
        it = items[idx]
        profile = load_profile_fixture(it["profile_id"])
        r["judge"] = llm_judge_one(profile, it["jd_text"], r)
    valid = [r for r in base["results"]
             if isinstance(r.get("judge"), dict)
             and isinstance(r["judge"].get("score"), int)]
    if valid:
        base["summary"]["judge_mean_score"] = sum(
            r["judge"]["score"] for r in valid) / len(valid)
        base["summary"]["judge_pass_rate"] = sum(
            1 for r in valid if r["judge"]["score"] >= 4) / len(valid)
        base["summary"]["judge_n"] = len(valid)
    else:
        base["summary"]["judge_mean_score"] = None
        base["summary"]["judge_pass_rate"] = None
        base["summary"]["judge_n"] = 0
    return base


# =====================================================
# CLI 报告
# =====================================================

def format_report(out: dict) -> str:
    s = out["summary"]
    lines = []
    lines.append(f"## match_job 评估报告（n={s['n']}）")
    lines.append("")
    lines.append(f"- status 准确率：{s['status_acc']:.2%}")
    lines.append(f"- direction 准确率：{s['direction_acc']:.2%}")
    lines.append("")
    lines.append("### 分桶")
    for k, v in s["per_status"].items():
        lines.append(f"- {k}: n={v['n']}, acc={v['status_acc']:.2%}")
    lines.append("")
    lines.append("### 混淆矩阵（expected → actual）")
    for k, v in s["confusion"].items():
        lines.append(f"- {k} → {v}")
    if s.get("judge_n"):
        lines.append("")
        lines.append("### LLM-as-judge")
        lines.append(f"- 评估样本数：{s['judge_n']}")
        lines.append(f"- 平均分：{s['judge_mean_score']:.2f} / 5")
        lines.append(f"- 合理率（≥4 分）：{s['judge_pass_rate']:.2%}")
    lines.append("")
    lines.append("### 失败样本")
    fails = [r for r in out["results"] if not r["status_ok"]]
    if not fails:
        lines.append("（无）")
    for r in fails:
        lines.append(
            f"- {r['id']} [{r['profile_id']}] {r['jd_title']}: "
            f"expected={r['expected_status']} actual={r['actual_status']}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="叠加 LLM-as-judge")
    ap.add_argument("--json", default="", help="把结果落盘为 JSON")
    args = ap.parse_args()

    eval_set = load_eval_set()
    items = eval_set["items"]
    out = run_with_judge(items) if args.judge else run_baseline(items)
    print(format_report(out))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {args.json}")
    return 0 if out["summary"]["status_acc"] >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
