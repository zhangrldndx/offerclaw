# -*- coding: utf-8 -*-
"""supervisor.py — OfferClaw 外环生涯督导（多-agent 升级 Supervisor，docs/MULTI_AGENT_UPGRADE.md）

**如实定义,不新建 orchestrator。** 外环督导 = 五个**已存在**子系统的组合（``SUPERVISOR`` 台账,
文档-即-代码,可审计），外加一个串行多-JD 调度器 ``run_supervisor``（并行属 P4）。
判断权（三档路由）读规则引擎 status 字符串,**零 LLM**。
"""
from __future__ import annotations

# 文档-即-代码:外环督导 = 五个已存在子系统的组合（每项带 reframe/新增定性 + 锚点）
SUPERVISOR = {
    "A_三档路由":       {"where": "career_flow.build_routed_graph + _route_after_gap", "kind": "reframe"},
    "B_预算_checkpoint": {"where": "career_flow.node_guard + make_budget + resume_career_flow", "kind": "reframe"},
    "C_停滞检测":       {"where": "career_agent.compute_execution_tracking + ADVICE_ESCALATE_AFTER", "kind": "light-upgrade"},
    "D_三层记忆":       {"where": "memory_layers + career_flow._learn_from_flow", "kind": "reframe"},
    "E_节奏触发":       {"where": "offerclaw_cli 子命令 + OpenClaw cron", "kind": "reframe"},
}


def _rank_key(status: str) -> int:
    """三档结论排序键。旧文案仅通过精确适配器转换。"""
    from domain_status import MatchStatusCode, match_status_code
    return {
        MatchStatusCode.SUITABLE: 0,
        MatchStatusCode.STRETCH: 1,
        MatchStatusCode.NOT_RECOMMENDED: 2,
        MatchStatusCode.UNKNOWN: 9,
    }[match_status_code(status)]


def _run_one_jd(jd, i, *, skip_llm, budget) -> dict:
    """跑单个 JD 的完整 CareerFlow。每份 JD **独立 state + 独立 budget 副本**,无共享可变量。"""
    from career_flow import run_career_flow_routed
    if isinstance(jd, str):
        jd_text, jd_title = jd, f"JD#{i + 1}"
    else:
        jd_text = jd.get("jd_text", "")
        jd_title = jd.get("jd_title", f"JD#{i + 1}")
    final = run_career_flow_routed(
        jd_text, jd_title=jd_title, skip_llm=skip_llm,
        budget=dict(budget) if budget else None)
    return {"jd_title": jd_title,
            "status": (final.get("match_report") or {}).get("status", ""),
            "route_taken": final.get("route_taken", ""),
            "final": final}


def run_supervisor(jds, *, skip_llm: bool = True, budget: dict = None,
                   parallel: bool = False, max_workers: int = 4) -> dict:
    """遍历多个 JD,每份跑 ``run_career_flow_routed``,聚合成横向对比 + 优先级排序。

    ``parallel=True``（结构腿 E3）:每份 JD **独立 state**,用线程池并发——**仅墙钟提速,不改任一
    JD 结果**。``ex.map`` 保序,故 parallel 终态与 serial **逐字段相等**（E3 correctness 判据）。
    ``jds``: ``[{"jd_text":..., "jd_title":...}, ...]`` 或 ``[str, ...]``。
    """
    jds = list(jds or [])
    if parallel and len(jds) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jds))) as ex:
            runs = list(ex.map(
                lambda p: _run_one_jd(p[1], p[0], skip_llm=skip_llm, budget=budget),
                list(enumerate(jds))))
    else:
        runs = [_run_one_jd(jd, i, skip_llm=skip_llm, budget=budget)
                for i, jd in enumerate(jds)]
    ranked = sorted(runs, key=lambda r: _rank_key(r["status"]))
    buckets = {}
    for r in runs:
        buckets.setdefault(r["status"] or "未知", []).append(r["jd_title"])
    return {
        "runs": runs,
        "ranked_titles": [r["jd_title"] for r in ranked],
        "buckets": buckets,
        "n": len(runs),
        "parallel": bool(parallel and len(jds) > 1),
    }
