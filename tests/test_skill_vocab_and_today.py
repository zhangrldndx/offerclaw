# -*- coding: utf-8 -*-
"""V3 修复回归：
1. job_discovery 词库覆盖 RLHF / PPO / 微调 / 强化学习 / Coding Agent / 多步推理
2. career_flow.today_node 在本次 JD 适合投递时，把本次 JD 顶到 headline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CODING_AGENT_JD = """【实习】Coding Agent实习生（科学发现）
岗位职责：
- Benchmark 评测与优化、Coding Agent 算法
- LLM 后训练（指令微调、强化学习）、代码生成、工具调用、多步推理
岗位要求：硕士、Python、Transformer、数据结构 / 算法
优先：RLHF / PPO 实践，Kaggle / LeetCode 经验
"""


def test_skill_vocab_covers_coding_agent_jd():
    from job_discovery import discover
    parsed = discover(raw=CODING_AGENT_JD)
    hits = {h.lower() for h in parsed["skills_detected"]}
    # 至少要命中这些（Coding Agent 类岗位的通用关键字）
    expected = {"python", "transformer", "agent", "llm", "rlhf", "ppo",
                "coding agent", "强化学习", "指令微调", "代码生成",
                "工具调用", "多步推理", "kaggle", "leetcode"}
    missing = expected - hits
    assert not missing, f"skills_detected 漏关键字：{missing}; 实际命中={sorted(hits)}"


def test_today_node_overrides_when_this_jd_is_suitable():
    from career_flow import run_career_flow
    out = run_career_flow(CODING_AGENT_JD,
                          jd_title="AI4S · Coding Agent 实习生",
                          skip_llm=True)
    headline = (out.get("today_advice") or {}).get("headline", "")
    status = (out.get("match_report") or {}).get("status", "")
    if "适合" in status:
        assert "AI4S" in headline or "Coding Agent" in headline, (
            f"今日建议没有结合本次 JD：headline={headline!r}, status={status!r}"
        )
        assert "[DEMO]" not in headline, (
            f"今日建议被 applications.md 的 DEMO 行污染：{headline!r}"
        )
