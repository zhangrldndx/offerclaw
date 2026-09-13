# -*- coding: utf-8 -*-
""""本系统工程事实"意图的路由归属契约(2026-08-31 UI 实测修复)。

实测病灶(docs/rag_eval/current_head_20260831/UI_JOURNEY_FINDINGS.md 发现 1):
4/4 库内工程事实问题("拒答距离阈值多少""线上 reranker 哪个模型")经 UI 语义
规划器全部被 project_memory/Explain 截胡,挂着 3348 chunks 工程文档的
reference_kb.search 一发未进。零 LLM 原型层取证同病:这一类问题在整张
分类学里**无锚可依**(recall 框架下 reference_kb 几乎不进前三)。

这是一类问题,不是一道题:修的是**分类学**——
  1. reference_kb.search 的原型例句正面认领"系统配置/阈值/指标/选型"类;
  2. project_memory.search 以对比例句把"系统工程事实"推出去(它只管
     已审批的个人项目材料),提示词同步改口;
  3. 规划器规则手册新增一条**类级**规则(工程事实→explain+reference_kb);
  4. 原型缓存把例句哈希进有效性判定——否则今后任何人改例句而忘升
     REGISTRY_VERSION,就会静默用旧向量(本次修复顺手挖出的潜伏雷)。

守基线:反类对照(个人项目量化成果/参考资料解释概念)必须不被劫持——
论文路由 2026-08-18 劫持中文结果(-7.7pp)的教训,同型不许重演。
"""

from __future__ import annotations

import hashlib
import os

import pytest

import rag_route_registry as reg


def _defn(key: str):
    d = reg.get_route_definition(*key.split(".", 1))
    assert d is not None, key
    return d


# ------------------------------------------------------- 分类学:归属契约

def test_reference_kb_owns_the_system_fact_class():
    d = _defn("reference_kb.search")
    blob = d.description + " ".join(d.prototypes)
    # 类的三个侧面都要有锚:配置阈值 / 组件选型 / 评测指标
    assert "阈值" in blob, "缺'配置阈值'类锚点例句"
    assert "模型" in blob or "选型" in blob, "缺'组件选型'类锚点例句"
    assert "指标" in blob, "缺'评测指标'类锚点例句"
    assert "工程" in d.description or "系统" in d.description, \
        "描述未认领系统工程文档"


def test_project_memory_pushes_system_facts_away():
    d = _defn("project_memory.search")
    contrasts = " ".join(d.contrast_prototypes)
    assert "阈值" in contrasts or "配置" in contrasts, \
        "project_memory 没有以对比例句把系统工程事实推出去"
    assert "审批" in d.description or "个人" in d.description, \
        "描述应指明它只管已审批的个人项目材料"


def test_prompt_hints_follow_the_taxonomy():
    from semantic_query_planner import _PROMPT_ROUTE_HINTS as hints

    assert any(w in hints["reference_kb.search"] for w in ("工程", "配置", "系统")), \
        "reference_kb 的提示词仍只认'学习资料',工程事实类无家可归"
    assert "个人" in hints["project_memory.search"] or \
        "审批" in hints["project_memory.search"], \
        "project_memory 的提示词'项目实现细节'仍会截胡系统工程事实"


def test_planner_rulebook_names_the_class():
    from semantic_query_planner import _static_system_prompt

    prompt = _static_system_prompt()
    assert "工程事实" in prompt or "配置/阈值/指标" in prompt, \
        "规划器规则手册没有类级规则:系统工程事实→reference_kb"


# ------------------------------------------------- 缓存:例句变更必须自失效

def _fake_embedder(texts):
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:8]])
    return out


def test_prototype_cache_invalidates_when_examples_change(tmp_path, monkeypatch):
    """改例句而不升版号,缓存必须自己发现不一致并重建。

    旧机制只比对 REGISTRY_VERSION:例句改了、版号没动 → 静默用旧向量,
    分类学修复形同未部署。类级修法 = 例句全文哈希进缓存有效性判定。
    """
    monkeypatch.setattr(reg, "_cache_path",
                        lambda key: tmp_path / f"proto_{key}.json")
    v1, m1 = reg._load_or_build_vectors(_fake_embedder, persist=True)
    assert m1["cache"] == "miss"
    v2, m2 = reg._load_or_build_vectors(_fake_embedder, persist=True)
    assert m2["cache"] == "hit"

    # 模拟"有人改了例句":同版号、不同原型文本
    original = reg._prototype_texts
    def _tweaked():
        texts, owners = original()
        return [t + "!" for t in texts], owners
    monkeypatch.setattr(reg, "_prototype_texts", _tweaked)
    v3, m3 = reg._load_or_build_vectors(_fake_embedder, persist=True)
    assert m3["cache"] == "miss", "例句变了还在用旧缓存——分类学修复会被静默吞掉"


# --------------------------------------- 语义排序:类命中 + 反类不劫持(真嵌入)

_SEMANTIC = os.environ.get("OFFERCLAW_SEMANTIC_ROUTER_EVAL") == "1"

# 措辞与注册表原型例句刻意不同——同句复读只能证明背书,不能证明类泛化。
# 2026-08-31 用户方法论纠偏后取**使用者视角**(问自己资料里讲了什么),
# 不问"系统自身运行设定"——后者不在库里也不是真实用法。
_CLASS_QUESTIONS = [
    "笔记里说拒答的距离门槛设多少合适?",
    "资料里推荐用哪个 reranker 做精排?",
    "教程里候选池大小建议开多大?",
    "文档里口语测试那组的 R@1 是多少?",
    "笔记里 HyDE 通道是默认开的吗?",
]
_CONTROLS = [
    ("我的项目有什么量化成果可以写进简历", "project_memory"),
    ("根据参考资料给我解释一下什么是混合检索", "reference_kb"),
    ("我现在投了哪些公司", "application_state"),
]


@pytest.mark.skipif(not _SEMANTIC, reason="set OFFERCLAW_SEMANTIC_ROUTER_EVAL=1 (真嵌入,较慢)")
@pytest.mark.parametrize("service_mode", ["recall", "explain"])
def test_system_fact_questions_reach_reference_kb(service_mode):
    from types import SimpleNamespace

    reg.prebuild_route_prototype_cache()
    frame = SimpleNamespace(answer_objects=[], answer_object="", tasks=[],
                            operations=["search"], personal_scope="none",
                            service_mode=service_mode)
    hits = 0
    for q in _CLASS_QUESTIONS:
        cands, _ = reg.rank_route_candidates(q, frame, top_k=3)
        if any(c.source == "reference_kb" for c in cands[:2]):
            hits += 1
    assert hits >= 4, f"{service_mode}: 工程事实类 top2 命中 reference_kb 仅 {hits}/5"


@pytest.mark.skipif(not _SEMANTIC, reason="set OFFERCLAW_SEMANTIC_ROUTER_EVAL=1 (真嵌入,较慢)")
def test_counter_class_controls_are_not_hijacked():
    """反类对照:修复不得把个人材料/概念解释/投递状态的问题劫给 reference_kb。"""
    from types import SimpleNamespace

    reg.prebuild_route_prototype_cache()
    frame = SimpleNamespace(answer_objects=[], answer_object="", tasks=[],
                            operations=["search"], personal_scope="none",
                            service_mode="explain")
    for q, expected_source in _CONTROLS:
        cands, _ = reg.rank_route_candidates(q, frame, top_k=3)
        top_sources = [c.source for c in cands[:2]]
        assert expected_source in top_sources, \
            f"对照被劫持:{q!r} top2={top_sources},应含 {expected_source}"
