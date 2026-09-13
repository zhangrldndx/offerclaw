# -*- coding: utf-8 -*-
"""P1-a — 记忆/执行追踪消融(实验规划 v2)。回答面试追问「三层记忆到底好在哪、代价多少」。

两个子实验,全部**确定性**(零 LLM、零真实文件污染):

A. 跨天重复建议消融(执行追踪层,纯函数模拟)
   场景:用户连续 N 天无推进(最坏情况)。
   - ON :compute_execution_tracking 正常追踪,重复达阈值 → 升级建议(拆更小切入口);
   - OFF:每天都当第一次(无追踪)→ 同一建议无限重复。
   指标:重复建议天数占比、升级触发次数。

B. 记忆注入回流验证 + token 成本(隔离 base_dir)
   种子:2 次「适合」结论 → distill 出方向级 SOP;3 条 reflection 事件 → 近期教训。
   指标:注入文本是否真的回流(SOP/教训非空)+ 注入成本(chars 与估算 tokens)。
   诚实边界:B 只度量「回流成立 + 成本」,不声称对计划**质量**的提升
   (质量评估需 LLM judge,超出本确定性实验范围,如实披露)。

预登记判据:A 的 OFF 臂重复率应为 100%(机制关闭的定义性结果),ON 臂显著更低且
出现升级;B 的 SOP 回流非空、成本如实报。结果无论正负入档。
用法:python eval_memory_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from career_agent import ADVICE_ESCALATE_AFTER, compute_execution_tracking

N_DAYS = 10
PERSONAS = ("p1_ai", "p2_backend", "p3_algo")


# ---------------- A. 执行追踪消融 ----------------

def _simulate_days(tracking_on: bool) -> dict:
    """N 天无推进模拟。返回 {repeat_days, escalations, advice_seq}。"""
    repeat_days = escalations = 0
    per_persona = {}
    for persona in PERSONAS:
        record: dict = {}
        prev_advice = None
        seq = []
        base_sig = f"{persona}:先粘贴一份目标 JD 跑匹配"
        sig = base_sig
        for day in range(1, N_DAYS + 1):
            headline = sig.split(":", 1)[1]
            if tracking_on:
                tracking, escalation, record = compute_execution_tracking(
                    sig, headline, f"2026-08-{day:02d}", record, progressed=False)
                advice = escalation if escalation else headline
                if escalation:
                    escalations += 1
                    # 升级建议改变了签名(拆小切入口),次日按新建议追踪
                    sig = f"{persona}:30 分钟最小步 v{day}"
                    record = {}
            else:
                advice = headline          # 无追踪:天天同一句
            if prev_advice is not None and advice == prev_advice:
                repeat_days += 1
            prev_advice = advice
            seq.append(advice[:18])
        per_persona[persona] = seq
    total = len(PERSONAS) * (N_DAYS - 1)
    return {"repeat_days": repeat_days, "repeat_rate": round(repeat_days / total, 3),
            "escalations": escalations, "per_persona_head": {k: v[:4] for k, v in per_persona.items()}}


# ---------------- B. 记忆注入回流 + 成本 ----------------

def _memory_injection(mem_on: bool) -> dict:
    from memory_layers import (EpisodicMemory, ProceduralMemory,
                                get_active_sops, record_sop_outcome,
                                recent_reflection_lessons)
    with tempfile.TemporaryDirectory() as td:
        epi, proc = EpisodicMemory(base_dir=td), ProceduralMemory(base_dir=td)
        if mem_on:
            # SOP 只有在独立执行结果达到门槛后生效，岗位适合度不算效果证据。
            proc.add("rag_application_review", body="投递前核对项目证据并记录结果",
                     trigger={"direction": "大模型应用开发"})
            for index, day in enumerate(("2026-08-01", "2026-08-02", "2026-08-03")):
                event = epi.append({"kind": "sop_execution", "actor": "user",
                                    "source": "memory_ablation", "case": index})
                record_sop_outcome(proc, "rag_application_review", event["event_id"],
                                   "success", day)
            # 种子:3 条 reflection 事件(近期教训回流)
            for d in ("2026-08-01", "2026-08-02", "2026-08-03"):
                epi.append({"kind": "reflection", "date": d, "deviation_score": 2,
                            "incomplete": ["RAG 评测扩集", "简历初稿"]})
        sops = get_active_sops(proc, context="大模型应用开发")
        lessons = recent_reflection_lessons(epi, n=3)
        injected = "\n".join([f"- [SOP] {s}" for s in sops]
                             + [f"- [近期复盘] {x}" for x in lessons])
        return {"sops": len(sops), "lessons": len(lessons),
                "injected_chars": len(injected),
                "injected_tokens_est": len(injected) // 2,   # 中文粗估 ~2 chars/token
                "sample": injected[:120]}


def main() -> dict:
    on, off = _simulate_days(True), _simulate_days(False)
    mem_on, mem_off = _memory_injection(True), _memory_injection(False)
    result = {
        "config": {"days": N_DAYS, "personas": len(PERSONAS),
                   "escalate_after": ADVICE_ESCALATE_AFTER,
                   "scenario": "连续无推进(最坏情况)"},
        "A_execution_tracking": {"ON": on, "OFF": off},
        "B_memory_injection": {"ON": mem_on, "OFF": mem_off},
        "honesty": ("A 为机制级消融(确定性纯函数模拟);B 只证明回流成立并核算注入成本,"
                    "不声称计划质量提升(需 LLM judge,超出本实验范围)。"),
    }
    print(f"== A. 跨天重复建议消融({len(PERSONAS)} persona × {N_DAYS} 天,无推进) ==")
    print(f"  追踪 ON : 重复率 {on['repeat_rate']:.0%}  升级触发 {on['escalations']} 次")
    print(f"  追踪 OFF: 重复率 {off['repeat_rate']:.0%}  升级触发 {off['escalations']} 次")
    print(f"== B. 记忆注入(隔离目录) ==")
    print(f"  ON : SOP {mem_on['sops']} 条 · 教训 {mem_on['lessons']} 条 · "
          f"注入 {mem_on['injected_chars']} chars(≈{mem_on['injected_tokens_est']} tokens)")
    print(f"  OFF: SOP {mem_off['sops']} 条 · 注入 {mem_off['injected_chars']} chars")
    out = os.path.join(BASE, "docs", "agent_eval")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "memory_ablation.json")
    json.dump(result, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[已存] {path}")
    return result


if __name__ == "__main__":
    main()
