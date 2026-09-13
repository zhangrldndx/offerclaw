# -*- coding: utf-8 -*-
"""[L2] 检索式上下文压缩 + 预算 —— 量化测试（token/字符占用 + 不爆窗故障注入）。

注入场景：profile / 多块上下文随长期运行膨胀到远超窗口。
- 改造前：全量 dump（resume_builder 硬截 3000 会无声丢后段；plan_gen profile 全量）。
- 改造后：按相关性挑章节 + 优先级预算降级，注入量被预算封顶（不爆窗），高优先级整保。
量化：压缩后字符数 ≤ 预算（改造前 = 全量，远超）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_budget import fit_to_budget, select_relevant_sections, keywords_from


# ── fit_to_budget ───────────────────────────────────────────

def test_under_budget_unchanged():
    blocks = [{"name": "a", "priority": 1, "text": "x" * 50}]
    r = fit_to_budget(blocks, max_chars=100)
    assert r["ok"] and r["total_after"] == 50 and r["warnings"] == []
    assert r["kept"][0]["action"] == "keep"


def test_over_budget_degrades_low_priority_first():
    blocks = [
        {"name": "sys", "priority": 10, "text": "A" * 100, "min_keep": 0},
        {"name": "log", "priority": 1, "text": "B" * 5000, "min_keep": 50},
    ]
    r = fit_to_budget(blocks, max_chars=200)
    assert r["total_after"] <= 200                      # 预算闸：必不超
    sys_block = next(k for k in r["kept"] if k["name"] == "sys")
    log_block = next(k for k in r["kept"] if k["name"] == "log")
    assert sys_block["chars"] == 100 and sys_block["action"] == "keep"   # 高优先级整保
    assert log_block["chars"] <= 100 and log_block["action"] == "truncate"
    assert any("log" in w for w in r["warnings"])


def test_budget_guarantee_holds():
    # 只要 sum(min_keep) <= max，total_after 必 <= max
    blocks = [{"name": f"b{i}", "priority": i, "text": "z" * 9000, "min_keep": 10} for i in range(5)]
    r = fit_to_budget(blocks, max_chars=1000)
    assert r["total_after"] <= 1000 and r["ok"]


# ── select_relevant_sections ────────────────────────────────

def _profile_with_sections():
    return (
        "## 基础信息\n示例候选人，计算机本科\n\n"
        "## 熟练技能\nPyTorch、深度学习、Transformer\n\n"
        + "## 无关章节A\n" + ("水" * 4000) + "\n\n"
        + "## 无关章节B\n" + ("填" * 4000) + "\n"
    )


def test_relevant_sections_kept_within_budget():
    md = _profile_with_sections()
    out = select_relevant_sections(md, keywords=["PyTorch"], max_chars=800,
                                   always_keep=("基础信息",))
    assert len(out) <= 800                              # 预算闸
    assert "基础信息" in out and "PyTorch" in out        # always_keep + 相关章节保留
    assert "无关章节A" not in out                         # 无关大章节被丢
    assert out.index("基础信息") < out.index("PyTorch")  # 还原原文顺序


def test_under_budget_returns_unchanged():
    md = "## a\nshort"
    assert select_relevant_sections(md, ["x"], max_chars=999) == md


def test_keywords_from_extracts_tokens():
    kw = keywords_from("要求：熟悉 PyTorch 和 LangGraph，了解大模型应用")
    assert "PyTorch" in kw and "LangGraph" in kw          # 英文技术词
    assert "模型" in kw and "应用" in kw                   # 中文 jieba 切词（"大模型"→"大"+"模型"）


# ── 故障注入：超大 profile 不爆窗 ───────────────────────────

def test_huge_profile_does_not_overflow():
    huge = "".join(f"## 章节{i}\n" + ("内容" * 500) + "\n\n" for i in range(50))  # ~50KB
    before = len(huge)
    out = select_relevant_sections(huge, keywords=["章节3"], max_chars=5000,
                                   always_keep=("章节0",))
    assert before > 40000                               # 确实是超大输入
    assert len(out) <= 5000                             # 改造后被预算封顶（改造前 = 50KB 爆窗）


def test_resume_builder_compresses_profile():
    from resume_builder import build_messages
    profile = (
        "## 基础信息\n张某\n\n## 熟练技能\nPyTorch 强化学习\n\n"
        + "## 获奖经历\n" + ("奖" * 6000) + "\n"        # 大块无关章节
    )
    msgs = build_messages(jd_summary="要求熟悉 PyTorch 做强化学习", profile=profile)
    system = msgs[0]["content"]
    assert "PyTorch" in system                          # JD 相关章节保留
    # profile 部分被预算压住：整条 system 远小于「事实清单 + 全量 profile」
    assert len(system) < len(profile)                   # 没有全量 dump 进来
