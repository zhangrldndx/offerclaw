# -*- coding: utf-8 -*-
"""llm_cost_report.py — LLM 调用成本报表（P5）。

读取网关旁路计量台账 ``logs/llm_usage.jsonl``（由 ``day1_api_starter._log_llm_usage``
在每次非流式调用成功后追加），按模型聚合 token 用量与耗时，并可选换算成本。

诚实性设计：
- 台账只存**原始事实**（token 数 / 耗时），不存换算金额——单价会变，事实不会；
- 单价从 ``llm_prices.json`` 读取（用户按官网现价维护，格式见 ``--init-prices``）；
  没有价目文件就只输出 token 统计，**绝不用猜测的单价出金额**。

用法：
    python llm_cost_report.py                 # 全量台账报表
    python llm_cost_report.py --since 2026-07-05T00:00:00   # 只看某时刻之后
    python llm_cost_report.py --init-prices   # 生成价目文件模板
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USAGE_PATH = os.path.join(BASE_DIR, "logs", "llm_usage.jsonl")
PRICES_PATH = os.path.join(BASE_DIR, "llm_prices.json")

_PRICE_TEMPLATE = {
    "_说明": "单位=元/每千 token。按百炼官网现价手动维护，改价后无需动台账。",
    "_更新日期": "填写你核对官网的日期",
    "示例-模型名": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
}


def load_rows(since: str | None):
    if not os.path.exists(USAGE_PATH):
        print(f"台账不存在：{USAGE_PATH}（先跑一次任意 LLM 调用）", file=sys.stderr)
        return []
    rows = []
    with open(USAGE_PATH, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue  # 半截行（进程被杀时可能出现）：跳过不炸
            if since and str(r.get("ts", "")) < since:
                continue
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM token/成本报表")
    ap.add_argument("--since", help="ISO 时间下界，如 2026-07-05T00:00:00")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--init-prices", action="store_true", help="生成价目文件模板后退出")
    args = ap.parse_args()

    if args.init_prices:
        if os.path.exists(PRICES_PATH):
            print(f"已存在：{PRICES_PATH}")
        else:
            with open(PRICES_PATH, "w", encoding="utf-8") as f:
                json.dump(_PRICE_TEMPLATE, f, ensure_ascii=False, indent=2)
            print(f"已生成模板：{PRICES_PATH}，请按官网现价填写")
        return 0

    rows = load_rows(args.since)
    if not rows:
        print("台账为空（或全部早于 --since）")
        return 0

    prices = {}
    if os.path.exists(PRICES_PATH):
        try:
            prices = {k: v for k, v in json.load(open(PRICES_PATH, encoding="utf-8")).items()
                      if not k.startswith("_")}
        except (json.JSONDecodeError, OSError):
            print("⚠️ llm_prices.json 解析失败，忽略成本换算", file=sys.stderr)

    by_model: dict = {}
    for r in rows:
        m = by_model.setdefault(r.get("model", "?"),
                                {"calls": 0, "prompt": 0, "completion": 0, "lat": []})
        m["calls"] += 1
        m["prompt"] += r.get("prompt_tokens") or 0
        m["completion"] += r.get("completion_tokens") or 0
        if r.get("elapsed_ms") is not None:
            m["lat"].append(r["elapsed_ms"])

    out = {"span": {"from": rows[0].get("ts"), "to": rows[-1].get("ts"), "calls": len(rows)},
           "models": {}}
    total_cost = 0.0
    priced_any = False
    for name, m in sorted(by_model.items()):
        lat = sorted(m["lat"])
        entry = {
            "calls": m["calls"],
            "prompt_tokens": m["prompt"],
            "completion_tokens": m["completion"],
            "avg_tokens_per_call": round((m["prompt"] + m["completion"]) / m["calls"], 1),
            "latency_p50_ms": lat[len(lat) // 2] if lat else None,
            "latency_p95_ms": lat[int(len(lat) * 0.95)] if lat else None,
        }
        p = prices.get(name)
        if p:
            cost = (m["prompt"] / 1000 * p.get("prompt_per_1k", 0)
                    + m["completion"] / 1000 * p.get("completion_per_1k", 0))
            entry["est_cost_yuan"] = round(cost, 4)
            entry["est_cost_per_call_yuan"] = round(cost / m["calls"], 4)
            total_cost += cost
            priced_any = True
        out["models"][name] = entry
    if priced_any:
        out["total_est_cost_yuan"] = round(total_cost, 4)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"# LLM 用量报表  {out['span']['from']} → {out['span']['to']}  共 {out['span']['calls']} 次调用\n")
    for name, e in out["models"].items():
        print(f"## {name}")
        print(f"  调用 {e['calls']} 次 · 输入 {e['prompt_tokens']:,} tok · 输出 {e['completion_tokens']:,} tok"
              f" · 均值 {e['avg_tokens_per_call']} tok/次")
        print(f"  延迟 p50={e['latency_p50_ms']}ms  p95={e['latency_p95_ms']}ms")
        if "est_cost_yuan" in e:
            print(f"  估算成本 {e['est_cost_yuan']} 元（{e['est_cost_per_call_yuan']} 元/次，按 llm_prices.json 单价）")
        else:
            print("  （无单价：跑 `python llm_cost_report.py --init-prices` 并按官网填价后可出金额）")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
