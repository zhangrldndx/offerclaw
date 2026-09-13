# -*- coding: utf-8 -*-
"""Independent qualitative Critic Agent for portfolio plans."""
from __future__ import annotations

import json

from review_protocol import CriticOutput, finalize_review, parse_critic_output


def build_plan_critic_messages(*, plan_spec: dict, plan_md: str,
                               context: dict, hard_report: dict,
                               review_contract: dict,
                               previous_report: dict | None = None,
                               final_review: bool = False) -> list[dict]:
    system = (
        "你是 OfferClaw Plan Critic Agent，与 Plan Agent 相互独立。"
        "你只评审计划，不得直接改计划，也不得保存产物。"
        "判断岗位优先级、共同与特有缺口取舍、学习依赖、计划可执行性、"
        "复盘与记忆建议的使用，以及用户要求是否真正落实。"
        "hard_validation 的 error 拥有否决权，你不得覆盖。"
        "首次评审可细化已有标准并增加以 jd_ 开头的岗位标准；不得删除固定标准或伪造用户要求。"
        "最终复审沿用冻结标准，新的轻微风格建议不得阻塞通过，除非修改造成回归。"
        "revision_brief 只给出修改位置、目标和验收条件，不输出替代计划。只返回严格 JSON。"
    )
    payload = {
        "review_phase": "final" if final_review else "initial",
        "scope": context.get("scope") or {},
        "applications": context.get("applications") or [],
        "gaps": context.get("gaps") or [],
        "profile": context.get("profile") or {},
        "memory_context": context.get("memory_context") or {},
        "recent_log": context.get("recent_log") or "",
        "plan_spec": plan_spec,
        "plan_markdown": plan_md,
        "hard_validation": hard_report,
        "review_contract": review_contract,
        "previous_review": previous_report,
        "output_schema": CriticOutput.model_json_schema(),
    }
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def run_plan_critic_agent(*, plan_spec: dict, plan_md: str,
                          context: dict, hard_report: dict,
                          review_contract: dict, artifact_revision: int,
                          caller, previous_report: dict | None = None,
                          final_review: bool = False) -> dict:
    raw = caller(build_plan_critic_messages(
        plan_spec=plan_spec, plan_md=plan_md, context=context,
        hard_report=hard_report, review_contract=review_contract,
        previous_report=previous_report, final_review=final_review,
    ), max_tokens=3000)
    output = parse_critic_output(raw)
    return finalize_review(
        kind="portfolio_plan", artifact_revision=artifact_revision, output=output,
        base_contract=review_contract, hard_report=hard_report,
        final_review=final_review, previous_report=previous_report,
    )
