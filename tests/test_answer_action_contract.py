# -*- coding: utf-8 -*-
"""门动作 → 生成的**交接契约**(2026-08-31,answer-quality-v2 P0 修复)。

实测病灶:8 个负例的 pipeline 路由动作 8/8 正确,但端到端动作合同只有 6/8 ——
4 个 `correct_premise` 在 trace 层全部识别成功,最终答案只有 2/4 真正首句纠偏。
根因不是判据,而是**交接**:门的结构化动作(answer / correct_premise)在
`_retrieve_and_classify` 返回时被丢弃,生成端永远拿到默认 "answer" 合同。

修法是一类问题的修法,不是补一条规则:
  1. 动作派生只有**一份实现** `answer_action_from_retrieval(g)`(从
     eval_answer_quality_v2 收编,评测器改为 import 同一函数——两把尺子教训
     见该文件 LESSONS 注释);
  2. `_finish` 是所有检索返回路径(主返回/CRAG 恢复/论文路)的唯一出口,
     在那里盖章 `g["answer_action"]`——新增返回路径自动继承,不靠人记得;
  3. 两个生成消费点(流式 1 处、非流式 1 例)都必须把动作传进合同,
     本文件用源码钉逐个钉死。
"""

from __future__ import annotations

import inspect

import rag_gate


def _g(in_kb=True, panel=None, grades=None, relations=None, anchor_rank=None):
    """按 _retrieve_and_classify 真实返回形状构造最小 g。"""
    diag = {}
    if panel is not None:
        diag["gate_panel"] = {"action": panel}
    if grades is not None:
        diag["grades"] = grades
    if relations is not None:
        diag["relations"] = relations
    features = {"answerability_rerank": diag}
    if anchor_rank is not None:
        features["gate_anchor_rank"] = anchor_rank
    return {"in_kb": in_kb,
            "retrieval_trace": {"gate_features": features}}


# ------------------------------------------------------- 派生:唯一实现

def test_not_in_kb_is_abstain():
    assert rag_gate.answer_action_from_retrieval(_g(in_kb=False)) == "abstain"


def test_panel_action_wins():
    g = _g(panel="correct_premise")
    assert rag_gate.answer_action_from_retrieval(g) == "correct_premise"


def test_anchor_verdict_backs_up_the_panel():
    """无 panel(单票放行)时回落到锚点判据的 action() 合同。

    grades/relations 的键是**锚点下标的字符串**(JSON 往返后的真实形态),
    gate_anchor_rank 是 1-based —— 派生必须容两种键型,这正是评测器里
    踩过的形状。"""
    g = _g(grades={"1": 3}, relations={"1": "contradicts"}, anchor_rank=2)
    assert rag_gate.answer_action_from_retrieval(g) == "correct_premise"


def test_distance_door_defaults_to_answer():
    """距离门直接放行、判据未跑:没有可用动作证据时维持既有 "answer" 语义,
    检索与门的判定一位都不许被本修复挪动(基线红线)。"""
    assert rag_gate.answer_action_from_retrieval(_g()) == "answer"


def test_judge_abstain_with_open_gate_stays_answer():
    """判据说 abstain 但门(距离)已放行:历史语义 = 照常回答。
    本修复只**传递**已有决定,不新增一道否决权。"""
    g = _g(grades={"0": 1}, relations={"0": "not_established"}, anchor_rank=1)
    assert rag_gate.answer_action_from_retrieval(g) == "answer"


# ------------------------------------------------------- 盖章:唯一出口

def test_every_retrieval_result_is_stamped():
    """_finish 是全部返回路径的收口,盖章必须在那里——新增路径自动继承。"""
    src = inspect.getsource(rag_gate._retrieve_and_classify)
    assert "answer_action_from_retrieval" in src, \
        "检索结果没有在 _finish 收口处盖章 answer_action"


# ------------------------------------------------------- 消费:逐点钉死

def test_stream_synthesis_receives_the_action():
    src = inspect.getsource(rag_gate.gated_query_stream)
    assert 'g.get("answer_action"' in src or 'g["answer_action"]' in src, \
        "流式合成没有把门动作传给 _grounded_messages(P0 原病灶)"


def test_stream_meta_exposes_the_action():
    """meta 事件带 answer_action(加性字段):UI/评测能看见生成用的是哪份合同,
    下次'门说了、生成没做'就在线上可观测,而不是等人工抽查。"""
    src = inspect.getsource(rag_gate.gated_query_stream)
    assert "answer_action" in src.split("yield {\"type\": \"meta\"")[0] or \
        src.count("answer_action") >= 2, "meta 事件未暴露 answer_action"


def test_api_meta_whitelist_lets_the_action_through():
    """第二道接缝:rag_api 的 /api/stream 对 meta 做**字段白名单投影**——
    gate 盖了章、白名单不放行,线上照样看不见(2026-08-31 live 探针实测:
    生成已拿到动作,meta 里却是 None)。"""
    import inspect as _inspect

    import rag_api

    src = _inspect.getsource(rag_api)
    start = src.find('elif t == "meta":')
    assert start != -1
    assert '"answer_action"' in src[start:start + 2000], \
        "/api/stream 的 meta 白名单没放行 answer_action"


def test_plain_query_synthesis_receives_the_action():
    src = inspect.getsource(rag_gate.gated_query)
    assert "answer_action" in src, \
        "非流式 gated_query 没有把门动作传给 synthesize_grounded_answer"


def test_multi_source_execution_carries_the_action():
    """第三道接缝(2026-09-01 live 12 例验收抓到):multi_source 执行分支
    整条没有动作章——ExecutionResult 不带、聚合丢章、合成消息进不了纠偏
    合同。规划器把同一问题规划成 reference-only 还是 multi 有随机性,
    两个分支都真实可达,所以章必须两条路都在。"""
    import rag_multi_source

    from dataclasses import fields
    names = {f.name for f in fields(rag_multi_source.ExecutionResult)}
    assert "answer_action" in names, "ExecutionResult 不携带门动作"
    src = inspect.getsource(rag_multi_source.execute_plan)
    assert "answer_action" in src, "execute_plan 聚合时丢弃了 reference 路的章"
    import inspect as _i
    sig = _i.signature(rag_multi_source.multi_source_messages)
    assert "stance" in sig.parameters, \
        "multi_source_messages 无法接收纠偏合同(stance)"


def test_multi_branches_consume_the_action():
    """gated_query 与 gated_query_stream 的 multi 分支都要:返回/meta 带章
    + 按章向合成注入纠偏合同。"""
    for fn in (rag_gate.gated_query, rag_gate.gated_query_stream):
        src = inspect.getsource(fn)
        multi_part = src.split("_execute_query_plan")[-1]
        assert "answer_action" in multi_part, f"{fn.__name__} multi 分支缺章"
        assert "stance" in multi_part, f"{fn.__name__} multi 分支未注入纠偏合同"


def test_judge_sees_the_users_question_not_the_subquery():
    """第四道接缝(live 12 例验收抓到,2026-09-01):multi 路把规划器改写的
    **中性 subquery** 当判定对象传进检索,用户的错误前提在改写中被抹掉,
    判据永远判不出 correct_premise——而前提恰恰住在原问题里。

    类级修法:检索继续用 subquery(检索行为一位不动),判据与共识 panel
    换用显式 ``judgement_question``(缺省 = 检索问题,直连路径零变化)。
    影子观测早就这么干了(_answerability_shadow_question 记原问题)——
    主判据当初漏了同一课。"""
    import inspect as _i

    import rag_multi_source

    sig = _i.signature(rag_gate._retrieve_and_classify)
    assert "judgement_question" in sig.parameters, \
        "_retrieve_and_classify 缺 judgement_question 参数"
    src = _i.getsource(rag_gate._retrieve_and_classify)
    assert "rerank_by_mode(_jq" in src or "rerank_by_mode(\n                _jq" in src, \
        "判据重排仍用检索问题而非判定问题"
    assert "confirm_action(_jq" in src or "confirm_action(\n                        _jq" in src, \
        "共识 panel 仍用检索问题而非判定问题"
    msrc = _i.getsource(rag_multi_source.execute_plan)
    assert "judgement_question=question" in msrc, \
        "multi 路的 reference_kb 任务没把原问题交给判据"


def test_correction_contract_actually_changes_the_messages():
    plain = rag_gate._grounded_messages("q", ["c1"])
    corrected = rag_gate._grounded_messages("q", ["c1"], "correct_premise")
    assert plain != corrected
    assert "前提" in corrected[0]["content"]
    assert "第一句明确指出前提不成立" in corrected[0]["content"]


def test_eval_uses_the_same_yardstick():
    """评测器与生产共用同一份派生实现——两份实现迟早分家(LESSONS #57 同律)。"""
    import eval_answer_quality_v2 as ev

    assert getattr(ev, "_answer_action") is rag_gate.answer_action_from_retrieval
