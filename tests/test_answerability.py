# -*- coding: utf-8 -*-
"""Tests for the answer-containment judge.

This is the signal every other 2026-08-27 intervention was missing: the
production reranker scores topical relatedness, so on its 20 Dev-New failures
it chose a chunk graded 1 ("mentions the topic, no answer") ten times while
passing over a gold graded 3 in 18 of 19 cases.

Two properties are load-bearing and therefore pinned here: an unparseable or
out-of-range verdict must read as *unavailable* rather than as grade 0, and the
prompt must never be handed the answer.
"""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class _FakeTrace:
    """Minimal stand-in: two near-tied high scorers, channels disagreeing."""

    gate_features = {"best_dense_distance": 0.74}
    gate_decision = True

    class _C:
        def __init__(self, chunk_id, score):
            self.chunk_id = chunk_id
            self.rerank_score = score
            self.document = f"body of {chunk_id}"

    reranked_candidates = [_C("a", 0.99), _C("b", 0.98), _C("c", 0.4)]
    dense_candidates = [_C("a", None), _C("z", None)]
    bm25_candidates = [_C("b", None), _C("a", None)]


class _FinalTrace(_FakeTrace):
    index_fingerprint = "fp-test"
    retrieval_profile = type("Profile", (), {"name": "baseline"})()
    final_candidates = [_FakeTrace._C("final-a", 0.97),
                        _FakeTrace._C("final-b", 0.96)]


from rag_answerability import (  # noqa: E402
    GRADES,
    cache_key,
    enabled,
    grade,
    parse_grade,
)


def _v4_json(grade, question_form, direct_answer, premise_status,
             reason="", **extra):
    return json.dumps({
        "grade": grade,
        "question_form": question_form,
        "direct_answer": direct_answer,
        "premise_status": premise_status,
        "reason": reason,
        **extra,
    }, ensure_ascii=False)


@pytest.mark.parametrize("text,expected_relation", [
    (_v4_json(3, "polar", "proposition_true", "supported", "明确支持"),
     "entails"),
    (_v4_json(3, "polar", "proposition_false", "refuted", "明确反驳"),
     "contradicts"),
    (_v4_json(1, "polar", "unknown", "not_established", "关系未建立"),
     "not_established"),
    (_v4_json(3, "open", "not_applicable", "none", "含完整定义"),
     "entails"),
    (_v4_json(3, "open", "not_applicable", "refuted", "开放问句前提错误"),
     "contradicts"),
    (_v4_json(2, "open", "not_applicable", "none", "仅部分事实"),
     "not_established"),
])
def test_parse_grade_derives_relation_from_the_v4_contract(
        text, expected_relation):
    verdict = parse_grade(text)
    assert verdict["relation"] == expected_relation
    assert set(verdict) == {
        "grade", "question_form", "direct_answer", "premise_status",
        "relation", "reason",
    }


def test_model_supplied_relation_cannot_override_v4_derivation():
    text = _v4_json(
        3, "polar", "proposition_false", "refuted", "片段明确说相反",
        relation="entails",
    )

    assert parse_grade(text)["relation"] == "contradicts"


@pytest.mark.parametrize("text", [
    None, "", "这个片段挺相关的", "{}", '{"grade": 7}', '{"grade": -1}',
    '{"grade": "3"}', '{"grade": true}', '{"grade": 2.5}', "not json at all",
    # v3 output must not be accepted under v4, even if its relation looks valid.
    '{"grade": 3, "relation": "entails", "reason": "old contract"}',
    _v4_json(3, "polar", "proposition_false", "supported"),
    _v4_json(3, "polar", "proposition_true", "refuted"),
    _v4_json(3, "polar", "not_applicable", "none"),
    _v4_json(3, "open", "proposition_true", "supported"),
    _v4_json(3, "other", "unknown", "not_established"),
    _v4_json(3, "polar", "maybe", "not_established"),
    _v4_json(3, "polar", "unknown", "maybe"),
])
def test_an_unusable_verdict_is_unavailable_not_zero(text):
    """`None` and grade 0 mean opposite things to a gate.

    Coercing a judge failure to 0 would make an API outage look like "the
    corpus does not contain the answer" -- i.e. it would turn every query into
    a refusal and call that correct behaviour.
    """
    assert parse_grade(text) is None


def test_grade_returns_none_when_the_judge_is_unusable():
    result = grade("问题", "片段", caller=lambda *a, **k: "抱歉我无法回答",
                   use_cache=False)  # prose, not the JSON contract
    assert result is None


def test_grade_passes_only_the_question_and_the_chunk():
    """No label leakage: the judge must not be shown anything derived from
    the gold, or the feature would have seen the answer."""
    captured = {}

    def caller(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        captured["kwargs"] = kwargs
        return _v4_json(2, "open", "not_applicable", "none", "部分")

    grade("BM25 比向量检索强在哪？", "BM25 是词法检索通道。",
          caller=caller, use_cache=False)
    assert "不要输出 relation 字段" in captured["prompt"]
    prompt = captured["prompt"]
    assert "BM25 比向量检索强在哪？" in prompt
    assert "BM25 是词法检索通道。" in prompt
    # determinism is part of the contract -- a drifting judge makes any A/B
    # built on it unreproducible
    assert captured["kwargs"]["temperature"] == 0.0


def test_yes_no_false_premise_semantics_are_explicit_in_judge_prompt():
    """Answerability and premise truth are orthogonal for yes/no questions."""
    captured = {}

    def caller(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return _v4_json(
            3, "polar", "proposition_false", "refuted", "职责属于存储层",
        )

    result = grade(
        "查询层是否负责存储向量和元数据？",
        "存储层负责存储向量和元数据；查询层负责处理查询请求。",
        caller=caller, use_cache=False,
    )

    assert result["relation"] == "contradicts"
    assert result["direct_answer"] == "proposition_false"
    assert result["premise_status"] == "refuted"
    assert "grade=3 不自动等于 supported" in captured["prompt"]
    assert "relation 由程序计算" in captured["prompt"]
    assert "闭源 LLM 是否依靠压缩加速" in captured["prompt"]
    assert "Planner 的职责是执行子任务而非设计路径吗" in captured["prompt"]
    assert "向量数据库与传统数据库相互替代吗" in captured["prompt"]


def test_v4_cache_lineage_invalidates_old_relation_verdicts():
    from rag_answerability import SCHEMA

    assert SCHEMA == "answerability-v4"

    import hashlib

    old_payload = json.dumps(
        ["answerability-v3", "m", "q", "c"],
        ensure_ascii=False, sort_keys=True,
    )
    old_key = hashlib.sha256(old_payload.encode("utf-8")).hexdigest()
    assert cache_key("m", "q", "c") != old_key


@pytest.mark.parametrize("question", [
    "查询层是否负责存储？",
    "Planner是不是执行子任务？",
    "这个方案能否工作？",
    "两种数据库相互替代吗？",
])
def test_obvious_polar_questions_cannot_be_downgraded_to_open(question):
    wrong_shape = _v4_json(
        3, "open", "not_applicable", "none", "错误地按开放问句处理",
    )

    assert parse_grade(wrong_shape, question=question) is None


@pytest.mark.parametrize("question,chunk,raw,expected_relation,expected_action", [
    (
        "闭源 LLM 是否依靠压缩加速？",
        "模型量化和剪枝等压缩方法可以提升一般 LLM 的推理速度。",
        _v4_json(1, "polar", "unknown", "not_established", "未涉及闭源或依靠关系"),
        "not_established", "abstain",
    ),
    (
        "Planner的职责是执行具体子任务而非设计行动路径吗？",
        "Planner 负责设计行动路径；Executor 才执行具体子任务。",
        _v4_json(3, "polar", "proposition_false", "refuted", "职责与题设相反"),
        "contradicts", "correct_premise",
    ),
    (
        "向量数据库与传统数据库属于相互替代关系吗？",
        "两者并非相互替代，而是分别承担不同职责并形成互补。",
        _v4_json(3, "polar", "proposition_false", "refuted", "证据明确说明互补"),
        "contradicts", "correct_premise",
    ),
])
def test_v4_regresses_the_three_observed_v3_failures(
        question, chunk, raw, expected_relation, expected_action):
    from rag_answerability import action

    verdict = grade(
        question, chunk, model="fixture-judge",
        caller=lambda *args, **kwargs: raw, use_cache=False,
    )

    assert verdict["relation"] == expected_relation
    assert action(verdict) == expected_action


def test_v4_cache_reuses_only_a_complete_structured_verdict(monkeypatch):
    import rag_answerability as answerability

    monkeypatch.setattr(answerability, "_CACHE", {})
    calls = 0

    def caller(messages, **kwargs):
        nonlocal calls
        calls += 1
        return _v4_json(
            3, "polar", "proposition_false", "refuted", "职责与题设相反",
        )

    first = answerability.grade(
        "Planner是否执行子任务？", "Planner只设计路径。",
        model="fixture-judge", caller=caller,
    )
    second = answerability.grade(
        "Planner是否执行子任务？", "Planner只设计路径。",
        model="fixture-judge", caller=caller,
    )

    assert calls == 1
    assert second == first
    assert second["premise_status"] == "refuted"
    assert second["relation"] == "contradicts"


def test_the_four_grades_are_the_whole_scale():
    assert GRADES == (0, 1, 2, 3)


def test_cache_key_separates_model_question_and_chunk():
    base = cache_key("m", "q", "c")
    assert base != cache_key("m2", "q", "c")
    assert base != cache_key("m", "q2", "c")
    assert base != cache_key("m", "q", "c2")
    assert base != cache_key("m", "q", "c", retrieval_question="route scope")
    assert cache_key("m", "q", "c", retrieval_question="q") == base
    assert base == cache_key("m", "q", "c")


def test_prompt_keeps_original_premise_and_retrieval_scope_separate():
    captured = {}

    def caller(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return _v4_json(
            3, "polar", "proposition_false", "refuted", "原命题被证据反驳",
        )

    verdict = grade(
        "向量数据库与传统数据库相互替代吗？",
        "两类数据库承担不同职责，通常形成互补。",
        model="fixture-judge", caller=caller, use_cache=False,
        retrieval_question="向量数据库与传统数据库的职责关系",
    )

    assert verdict["relation"] == "contradicts"
    assert "【原始用户问题（唯一的命题/前提真值来源）】" in captured["prompt"]
    assert "向量数据库与传统数据库相互替代吗？" in captured["prompt"]
    assert "【本路由检索问题（只界定事实覆盖范围，不得改写原命题）】" in captured["prompt"]
    assert "向量数据库与传统数据库的职责关系" in captured["prompt"]


def test_default_judge_model_is_resolved_before_cache_lookup(monkeypatch):
    from rag_answerability import resolve_model

    monkeypatch.delenv("RAG_ANSWERABILITY_MODEL", raising=False)
    monkeypatch.setenv("RAG_SYNTH_MODEL", "lineage-model")
    assert resolve_model() == "lineage-model"


def test_enabled_by_default_and_zero_is_the_opt_out(monkeypatch):
    """Production default flipped ON after the Final v3 verdict (B0 beat the old
    default on every ranking metric of a fresh blind set at zero false-accept
    cost).  The trap this pins: an *unset* variable now means ON, so "delete the
    env var" no longer disables the judge -- only an explicit falsy value does."""
    monkeypatch.delenv("RAG_ANSWERABILITY", raising=False)
    monkeypatch.delenv("RAG_ANSWERABILITY_MODE", raising=False)
    assert enabled() is True
    monkeypatch.setenv("RAG_ANSWERABILITY", "0")
    assert enabled() is False
    monkeypatch.setenv("RAG_ANSWERABILITY", "1")
    assert enabled() is True


def test_frozen_probe_shows_the_reranker_choosing_topic_over_answer():
    """The measurement this module exists to make, pinned to its artifact."""
    path = (Path(__file__).resolve().parents[1]
            / "docs/rag_eval/colloquial/answerability_failures.json")
    if not path.is_file():
        pytest.skip("answerability probe artifact unavailable")
    report = json.loads(path.read_text(encoding="utf-8"))
    gold = report["gold_grade_distribution"]
    top1 = report["reranker_top1_grade_distribution"]
    # the gold the reranker passed over almost always contains the answer ...
    assert int(gold.get("3", 0)) >= 15
    # ... while its own pick is "topic only" in a large share of failures
    assert int(top1.get("1", 0)) >= 8
    assert report["judge_unavailable"] == 0


@pytest.mark.parametrize("verdict,expected", [
    ({"grade": 3, "relation": "entails"}, "answer"),
    ({"grade": 3, "relation": "contradicts"}, "correct_premise"),
    ({"grade": 3, "relation": "not_established"}, "abstain"),
    ({"grade": 1, "relation": "contradicts"}, "abstain"),
    ({"grade": 3, "relation": None}, "abstain"),
    (None, "abstain"),
])
def test_three_state_policy(verdict, expected):
    """A false premise the corpus can refute is a *correction*, not silence.

    User ruling, 2026-08-28: evidence supports the premise -> answer; evidence
    refutes it -> say so and correct; evidence establishes neither -> abstain.
    Collapsing the middle case into abstention would penalise the system for
    using its evidence to correct the user, and would train the gate to go
    quiet whenever a question's premise is false.
    """
    from rag_answerability import action

    assert action(verdict) == expected


def test_grade_two_is_not_enough_to_answer_by_default():
    """Two adversarial negatives graded 2/entails were compound questions the
    corpus only half-covers ("说明采用Leiden社区检测，未给出分辨率参数调法").
    Answering those confidently is exactly the failure the guard exists for."""
    from rag_answerability import action

    partial = {"grade": 2, "relation": "entails"}
    assert action(partial, min_grade=3) == "abstain"
    assert action(partial, min_grade=2) == "answer"


def test_legacy_verdict_never_defaults_to_answer():
    """A pre-v4 verdict cannot bypass the structured premise contract."""
    from rag_answerability import action, parse_grade

    verdict = parse_grade('{"grade": 3, "reason": "no relation field"}')
    assert verdict is None
    assert action(verdict) == "abstain"


@pytest.mark.private_artifact
def test_guard_counts_correctable_rows_separately():
    from scripts.run_negative_guard import expected_actions

    actions = expected_actions()
    # col-neg-053 was approved by the user (the chunk states that hybrid search
    # fuses two result sets into one ranked list, which refutes "concatenates
    # two resumes" outright).  col-neg-056 was *not*: its chunk only says the
    # status change needs confirmation and never defines the reranker's remit,
    # so the refutation is incomplete and it stays pending.
    assert set(actions) == {"v2aneg-012", "v2aneg-016", "v2aneg-017",
                            "col-neg-053"}
    assert set(actions.values()) == {"correct_premise"}
    assert "col-neg-056" not in actions


def test_default_threshold_is_the_strict_side():
    """A caller that forgets ``min_grade`` must get the conservative behaviour.

    ``col-neg-056`` ("RAG 的 reranker 是否负责把投递状态改成已投递？") is graded 2
    by human adjudication: the chunk says changing the status requires user
    confirmation, but never states what the reranker's responsibilities are, so
    the refutation is incomplete.  With a default of 2 that row would have been
    turned into a confident correction by any caller that omitted the argument.
    """
    from rag_answerability import MIN_GRADE_TO_ACT, action

    assert MIN_GRADE_TO_ACT == 3
    assert action({"grade": 2, "relation": "contradicts"}) == "abstain"
    assert action({"grade": 2, "relation": "entails"}) == "abstain"
    assert action({"grade": 3, "relation": "contradicts"}) == "correct_premise"


def test_correct_premise_contract_changes_only_the_stance():
    """The action is a structured field, not a free-text hint, and the evidence
    block must be byte-identical between the two branches -- otherwise an A/B
    on the contract would also be an A/B on the context."""
    from rag_gate import _grounded_messages

    chunks = ["混合检索并行执行多种检索算法，再把结果集融合成统一排序列表。"]
    plain = _grounded_messages("Q", chunks)[0]["content"]
    correcting = _grounded_messages("Q", chunks, "correct_premise")[0]["content"]
    context = "[资料1]\n" + chunks[0]
    assert plain.endswith(context) and correcting.endswith(context)
    assert "第一句明确指出前提不成立" in correcting
    assert "第一句明确指出前提不成立" not in plain
    # an unknown action must not silently enable the correction stance
    assert "第一句明确指出前提不成立" not in _grounded_messages(
        "Q", chunks, "something_else")[0]["content"]


def test_judge_reason_is_audit_metadata_not_an_answer():
    """The generator must read the original evidence.

    Feeding the judge's one-line ``reason`` into the prompt would collapse the
    answer into a paraphrase of the judge and quietly make the judge, rather
    than the corpus, the source of truth.
    """
    from rag_gate import _grounded_messages

    built = _grounded_messages("Q", ["原始证据文本"], "correct_premise")
    joined = " ".join(message["content"] for message in built)
    assert "原始证据文本" in joined
    assert "reason" not in joined


def test_only_declared_modules_consume_the_judge():
    """Wiring contract, updated when the judge entered the retrieval path.

    Two consumers are now legitimate: the record-only shadow, and the rerank
    step in ``rag_gate``.  Anything else importing the judge is a leak of an
    LLM dependency into a path that has not been reviewed for it.

    The scan is recursive; an earlier version globbed ``*.py`` at the repo root
    only (141 of 475 files) and would have missed a consumer in any future
    production subpackage.
    """
    root = Path(__file__).resolve().parents[1]
    # The judge's own subsystem plus the one production path that consumes it.
    # Distillation tooling is declared here rather than exempted by pattern:
    # the point of the contract is that a new consumer has to be noticed.
    allowed = {"rag_answerability.py", "rag_shadow_answerability.py", "rag_gate.py",
               "rag_answerability_student_data.py", "train_answerability_student.py",
               "eval_answer_quality_v2.py"}
    skip_dirs = {".venv", "node_modules", ".git", "tests", "scripts", "docs",
                 "__pycache__", ".claude"}
    consumers = []
    for path in root.rglob("*.py"):
        if skip_dirs & set(path.relative_to(root).parts):
            continue
        if path.name in allowed:
            continue
        if "rag_answerability" in path.read_text(encoding="utf-8"):
            consumers.append(str(path.relative_to(root)))
    assert consumers == [], f"undeclared judge consumer: {consumers}"


def test_explicit_zero_disables_and_costs_nothing(monkeypatch):
    """Disabled must mean *no LLM call*, not "call it and ignore the result".
    The test session pins RAG_ANSWERABILITY=0 in conftest for hermeticity, and
    this is the contract that makes that pin actually work."""
    monkeypatch.setenv("RAG_ANSWERABILITY", "0")
    import rag_gate

    assert rag_gate._answerability_enabled() is False


def test_reranking_orders_by_grade_then_existing_score():
    from rag_answerability import rerank_by_answerability

    docs = ["topic only", "has the answer", "unrelated"]
    grades = {"topic only": 1, "has the answer": 3, "unrelated": 0}
    out_docs, out_metas, out_dists, out_scores = rerank_by_answerability(
        "q", docs, [{"i": 0}, {"i": 1}, {"i": 2}], [0.1, 0.2, 0.3],
        [0.99, 0.30, 0.95],
        grader=lambda q, c: {"grade": grades[c], "relation": "entails"},
    )
    # the cross-encoder had "topic only" first at 0.99; answer containment wins
    assert out_docs[0] == "has the answer"
    # every parallel list is permuted the same way
    assert out_metas[0] == {"i": 1} and out_dists[0] == 0.2 and out_scores[0] == 0.30


def test_ties_inside_a_grade_keep_the_reranker_order():
    from rag_answerability import rerank_by_answerability

    docs = ["a", "b"]
    out, _m, _d, _s = rerank_by_answerability(
        "q", docs, [{}, {}], [0.1, 0.2], [0.4, 0.9],
        grader=lambda q, c: {"grade": 3, "relation": "entails"})
    assert out == ["b", "a"]          # same grade -> higher rerank score first


def test_an_unavailable_judge_is_a_no_op_not_a_reordering():
    """Degrading to today's behaviour is the only safe failure mode.

    Treating an ungraded chunk as grade 0 would let an API outage silently
    produce a *different* ranking than either the old or the new design.
    """
    from rag_answerability import rerank_by_answerability

    docs = ["a", "b", "c"]
    args = ([{}, {}, {}], [0.1, 0.2, 0.3], [0.9, 0.5, 0.1])
    out, metas, dists, scores = rerank_by_answerability(
        "q", docs, *args, grader=lambda q, c: None)
    assert (out, metas, dists, scores) == (docs, *args)


def test_a_partially_available_judge_keeps_ungraded_rows_mid_pack():
    from rag_answerability import rerank_by_answerability

    docs = ["graded_low", "ungraded", "graded_high"]
    grades = {"graded_low": 1, "ungraded": None, "graded_high": 3}
    out, _m, _d, _s = rerank_by_answerability(
        "q", docs, [{}, {}, {}], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7],
        grader=lambda q, c: ({"grade": grades[c], "relation": "entails"}
                             if grades[c] is not None else None))
    # grade 3 first, then the ungraded row, then grade 1 -- an ungraded chunk
    # is not evidence of absence
    assert out == ["graded_high", "ungraded", "graded_low"]


def test_early_exit_stops_after_a_top_grade_incumbent():
    """The incumbent holds the highest reranker score, so a top grade makes its
    sort key minimal -- nothing below can displace it and judging the rest is
    provably wasted work for rank 1."""
    from rag_answerability import GRADES, rerank_by_answerability

    seen, docs = [], ["incumbent", "b", "c", "d"]

    def grader(q, chunk):
        seen.append(chunk)
        return {"grade": max(GRADES), "relation": "entails"}

    stats = {}
    out, _m, _d, _s = rerank_by_answerability(
        "q", docs, [{}] * 4, [0.1] * 4, [0.9, 0.8, 0.7, 0.6],
        grader=grader, early_exit=True, stats=stats)
    assert seen == ["incumbent"] and stats["calls"] == 1
    assert out == docs                     # the tail keeps the reranker order


def test_stats_separate_calls_made_from_verdicts_returned():
    """A judge that 503s still counts as called and silently returns the
    reranker's own order, so a run degraded by an outage has to be
    distinguishable from a clean one after the fact."""
    from rag_answerability import rerank_by_answerability

    docs = ["a", "b", "c", "d"]
    stats = {}
    rerank_by_answerability(
        "q", docs, [{}] * 4, [0.1] * 4, [0.9, 0.8, 0.7, 0.6], stats=stats,
        grader=lambda q, c: ({"grade": 2, "relation": "entails"}
                             if c in {"a", "b"} else None))
    assert stats["calls"] == 4 and stats["graded"] == 2


def test_early_exit_still_judges_the_tail_when_the_incumbent_falls_short():
    from rag_answerability import rerank_by_answerability

    docs = ["topic only", "has the answer", "unrelated"]
    grades = {"topic only": 2, "has the answer": 3, "unrelated": 0}
    stats = {}
    out, _m, _d, _s = rerank_by_answerability(
        "q", docs, [{}] * 3, [0.1] * 3, [0.9, 0.8, 0.7],
        grader=lambda q, c: {"grade": grades[c], "relation": "entails"},
        early_exit=True, stats=stats)
    assert stats["calls"] == 3
    assert out[0] == "has the answer"


def test_the_worker_pool_follows_depth_so_the_head_is_one_round():
    """Latency is rounds, not calls.  A pool smaller than ``depth`` would make
    the early-exit tail (1 + depth-1) cost more rounds than judging everything
    at once -- and a pool pinned to the default would re-introduce that the
    moment ``depth`` is raised for a wider candidate pool."""
    from concurrent.futures import ThreadPoolExecutor

    import rag_answerability as ra

    seen = []
    original = ThreadPoolExecutor

    class Recording(original):
        def __init__(self, max_workers=None, **kwargs):
            seen.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    import concurrent.futures

    concurrent.futures.ThreadPoolExecutor = Recording
    try:
        ra.rerank_by_answerability(
            "q", ["a"] * 12, [{}] * 12, [0.1] * 12, [0.9] * 12, depth=12,
            grader=lambda q, c: {"grade": 2, "relation": "entails"})
    finally:
        concurrent.futures.ThreadPoolExecutor = original
    assert seen == [12]              # one round for a depth-12 head, not two


def test_grades_are_keyed_by_the_position_the_chunk_ends_in():
    """The gate scores one specific chunk (its anchor), so pairing a grade with
    a position is only safe if the key is the *post-reorder* position."""
    from rag_answerability import rerank_by_answerability

    docs = ["topic only", "has the answer", "unrelated"]
    grades = {"topic only": 1, "has the answer": 3, "unrelated": 0}
    stats = {}
    out, _m, _d, _s = rerank_by_answerability(
        "q", docs, [{}] * 3, [0.1] * 3, [0.99, 0.30, 0.95],
        grader=lambda q, c: {"grade": grades[c], "relation": "entails"},
        stats=stats)
    assert out == ["has the answer", "topic only", "unrelated"]
    assert stats["grades"] == {0: 3, 1: 1, 2: 0}


def test_an_ungraded_chunk_has_no_grade_entry_at_all():
    """Absent must not read as grade 0: the gate has to be able to tell 'the
    judge said this contains no answer' from 'the judge never answered'."""
    from rag_answerability import rerank_by_answerability

    stats = {}
    rerank_by_answerability(
        "q", ["a", "b"], [{}] * 2, [0.1] * 2, [0.9, 0.8], stats=stats,
        grader=lambda q, c: ({"grade": 3, "relation": "entails"}
                             if c == "a" else None))
    assert stats["grades"] == {0: 3}


def test_the_answerability_gate_defaults_on_with_explicit_opt_out(monkeypatch):
    """Default ON since the Final v4 verdict: 3-vote consensus + v5 prompt held
    true false-accepts to 1/40 on a fresh blind set while doubling delivered
    answers.  An unset variable now means the production default, so anything
    needing the gate off must say "0" -- deleting the variable no longer
    disables it."""
    import rag_gate

    monkeypatch.delenv("RAG_ANSWERABILITY_GATE", raising=False)
    assert rag_gate._answerability_gate() is True
    monkeypatch.setenv("RAG_ANSWERABILITY_GATE", "0")
    assert rag_gate._answerability_gate() is False
    monkeypatch.setenv("RAG_ANSWERABILITY_GATE", "1")
    assert rag_gate._answerability_gate() is True


def test_the_answerability_gate_rides_the_three_state_contract():
    """A new number here would be a new threshold to overfit, and a grade-only
    rule would quietly collapse the three-state policy back to two: under it a
    false-premise question whose corpus refutes the premise reads the same as
    one the corpus says nothing about."""
    import re
    from pathlib import Path

    from rag_answerability import MIN_GRADE_TO_ACT, action

    source = Path(__file__).resolve().parents[1].joinpath("rag_gate.py").read_text(encoding="utf-8")
    block = source.split("_ans_accept = False")[1].split("_rr_gate_raw")[0]
    assert "_ans_action(" in block and "relations" in block
    assert not re.search(r"RAG_ANSWERABILITY_GATE_(MIN|TAU)", source)
    assert MIN_GRADE_TO_ACT == max(__import__("rag_answerability").GRADES)
    # the two verdicts the gate must separate, and the one it must let through
    top = max(__import__("rag_answerability").GRADES)
    assert action({"grade": top, "relation": "not_established"}) == "abstain"
    assert action({"grade": top, "relation": "contradicts"}) == "correct_premise"
    assert action({"grade": top, "relation": "entails"}) == "answer"


def test_relations_travel_with_grades_out_of_the_reranker():
    from rag_answerability import rerank_by_answerability

    stats = {}
    rerank_by_answerability(
        "q", ["a", "b"], [{}] * 2, [0.1] * 2, [0.9, 0.8], stats=stats,
        grader=lambda q, c: {"grade": 3,
                             "relation": "entails" if c == "a" else "not_established"})
    assert stats["grades"] == {0: 3, 1: 3}
    assert stats["relations"] == {0: "entails", 1: "not_established"}


def test_the_gate_diagnostics_carry_relations_not_just_grades():
    """The gate reads its verdict out of ``gate_features``; dropping the
    relation there is the same two-state collapse one layer down."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("rag_gate.py").read_text(encoding="utf-8")
    diag = source.split('"grades": _stats.get("grades"')[1].split('"top1_changed"')[0]
    assert '"relations": _stats.get("relations"' in diag


def test_the_gate_reads_the_depth_knob(monkeypatch):
    import rag_gate
    from rag_answerability import ANSWERABILITY_DEPTH

    monkeypatch.delenv("RAG_ANSWERABILITY_DEPTH", raising=False)
    assert rag_gate._answerability_depth() == ANSWERABILITY_DEPTH
    monkeypatch.setenv("RAG_ANSWERABILITY_DEPTH", "12")
    assert rag_gate._answerability_depth() == 12
    for junk in ("", "0", "-3", "abc"):
        monkeypatch.setenv("RAG_ANSWERABILITY_DEPTH", junk)
        assert rag_gate._answerability_depth() == ANSWERABILITY_DEPTH


def test_early_exit_is_on_by_default_in_the_gate(monkeypatch):
    """Exact for Top1 (the incumbent is already minimal on both sort keys) and
    measured identical on R@3/R@5 at half the judge calls, so it ships on."""
    import rag_gate

    monkeypatch.delenv("RAG_ANSWERABILITY_EARLY_EXIT", raising=False)
    assert rag_gate._answerability_early_exit() is True
    monkeypatch.setenv("RAG_ANSWERABILITY_EARLY_EXIT", "0")
    assert rag_gate._answerability_early_exit() is False


def test_an_unavailable_judge_under_early_exit_is_still_a_no_op():
    from rag_answerability import rerank_by_answerability

    docs = ["a", "b", "c"]
    args = ([{}, {}, {}], [0.1, 0.2, 0.3], [0.9, 0.5, 0.1])
    out, metas, dists, scores = rerank_by_answerability(
        "q", docs, *args, grader=lambda q, c: None, early_exit=True)
    assert (out, metas, dists, scores) == (docs, *args)


def test_shadow_is_off_by_default_and_makes_no_judge_calls(monkeypatch):
    import rag_shadow_answerability as shadow

    monkeypatch.delenv("RAG_SHADOW_ANSWERABILITY", raising=False)
    before = shadow.stats()["observed"]
    assert shadow.enabled() is False
    assert shadow.observe("问题", _FakeTrace()) is None
    assert shadow.stats()["observed"] == before


def test_shadow_observe_returns_nothing_to_branch_on(monkeypatch):
    """The response cannot depend on the shadow because there is no value to
    depend on -- ``observe`` returns ``None`` whether it judged, sampled, or
    dropped."""
    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_MARGIN", "0.05")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_HIGH_SCORE", "0.9")
    monkeypatch.setattr(shadow, "LOG_PATH", Path("/dev/null"))
    assert shadow.observe("问题", _FakeTrace(), sampler=lambda: 1.0) is None


def test_a_failing_judge_cannot_reach_the_caller(monkeypatch, tmp_path):
    """Timeout, exception and full queue must all be invisible to the request."""
    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")
    monkeypatch.setattr(shadow, "LOG_PATH", tmp_path / "shadow.jsonl")

    class Exploding:
        gate_features = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    assert shadow.observe("问题", Exploding()) is None      # feature extraction
    monkeypatch.setattr(shadow, "trigger_config",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert shadow.observe("问题", _FakeTrace()) is None      # config resolution


def test_a_full_queue_drops_and_counts_rather_than_blocking(monkeypatch, tmp_path):
    import queue as queue_module

    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_MARGIN", "0.05")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_HIGH_SCORE", "0.9")
    monkeypatch.setattr(shadow, "LOG_PATH", tmp_path / "shadow.jsonl")
    monkeypatch.setattr(shadow, "_QUEUE", queue_module.Queue(maxsize=1))
    monkeypatch.setattr(shadow, "_ensure_worker", lambda: None)  # never drained

    before = shadow.stats()["dropped"]
    for _ in range(4):
        assert shadow.observe("问题", _FakeTrace(), sampler=lambda: 0.0) is None
    assert shadow.stats()["dropped"] > before


def test_uncalibrated_thresholds_never_trigger():
    """No tuned default is supplied, because every number available today was
    read off Dev80 or V1 -- the sets used to find this failure."""
    import rag_shadow_answerability as shadow

    config = {"calibrated": False, "margin_below": None,
              "high_score_at_least": None, "control_rate": 0.05}
    assert shadow.should_trigger({"rerank_margin": 0.0, "rerank_top": 1.0},
                                 config) is False


def test_trigger_features_need_no_judge():
    """A trigger that had to call the judge to decide whether to call the judge
    would cost exactly what it exists to save."""
    import rag_shadow_answerability as shadow

    features = shadow.trigger_features(_FakeTrace())
    assert set(features) >= {"rerank_top", "rerank_margin", "high_scorers",
                             "channel_top1_agree", "channel_top5_overlap"}
    assert features["rerank_margin"] == pytest.approx(0.01)
    assert features["high_scorers"] == 2
    assert features["channel_top1_agree"] is False


def test_shadow_uses_final_candidates_and_keeps_raw_text_in_memory_only(
        monkeypatch, tmp_path):
    import queue as queue_module

    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")
    monkeypatch.setenv("RAG_SHADOW_CONTROL_RATE", "1")
    monkeypatch.delenv("RAG_SHADOW_TRIGGER_MARGIN", raising=False)
    monkeypatch.delenv("RAG_SHADOW_TRIGGER_HIGH_SCORE", raising=False)
    monkeypatch.setattr(shadow, "LOG_PATH", tmp_path / "shadow.jsonl")
    monkeypatch.setattr(shadow, "_QUEUE", queue_module.Queue(maxsize=8))
    monkeypatch.setattr(shadow, "_ensure_worker", lambda: None)

    question = "本地私密问题"
    shadow.observe(question, _FinalTrace(), sampler=lambda: 0.0)
    queued = shadow._QUEUE.get_nowait()
    assert queued["question_sha256"] == shadow.hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()
    assert queued["_question"] == question
    assert [row[1] for row in queued["_candidates"]] == ["final-a", "final-b"]
    public = {k: v for k, v in queued.items() if not k.startswith("_")}
    assert question not in json.dumps(public, ensure_ascii=False)


def test_shadow_worker_hashes_chunk_ids_before_logging(monkeypatch, tmp_path):
    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")
    monkeypatch.setenv("RAG_SHADOW_CONTROL_RATE", "1")
    monkeypatch.delenv("RAG_SHADOW_TRIGGER_MARGIN", raising=False)
    monkeypatch.delenv("RAG_SHADOW_TRIGGER_HIGH_SCORE", raising=False)
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(shadow, "LOG_PATH", log)
    monkeypatch.setattr("rag_answerability.grade", lambda *a, **k: {
        "grade": 3, "relation": "entails", "reason": "ok",
    })
    monkeypatch.setattr("rag_answerability.flush_cache", lambda: None)

    # Isolate this test from any worker/queue retained by earlier module tests.
    shadow._QUEUE = None
    shadow._WORKER = None
    shadow.observe("私密问题", _FinalTrace(), sampler=lambda: 0.0)
    shadow.drain(2)
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["schema_version"] == "shadow-answerability-v2"
    encoded = json.dumps(row, ensure_ascii=False)
    assert "final-a" not in encoded and "final-b" not in encoded
    assert set(row["verdicts"][0]) >= {"chunk_id_sha256", "verdict", "action"}


def test_control_sample_exists_so_trigger_misses_are_estimable(monkeypatch, tmp_path):
    """Judging only what the trigger flags makes the trigger unfalsifiable:
    the requests it wrongly passes over are exactly the ones never measured."""
    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_ANSWERABILITY", "1")
    monkeypatch.setenv("RAG_SHADOW_SAMPLING_MODE", "legacy_trigger_control")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_MARGIN", "0.001")
    monkeypatch.setenv("RAG_SHADOW_TRIGGER_HIGH_SCORE", "0.99")
    monkeypatch.setenv("RAG_SHADOW_CONTROL_RATE", "0.5")
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(shadow, "LOG_PATH", log)
    monkeypatch.setattr(shadow, "_ensure_worker", lambda: None)
    monkeypatch.setattr(shadow, "_QUEUE", __import__("queue").Queue(maxsize=8))

    shadow.observe("未触发但被抽为对照", _FakeTrace(), sampler=lambda: 0.0)
    queued = shadow._QUEUE.get_nowait()
    assert queued["triggered"] is False and queued["control_sample"] is True

    shadow.observe("未触发也未抽中", _FakeTrace(), sampler=lambda: 0.99)
    unjudged = shadow._QUEUE.get_nowait()
    # un-sampled requests still record their features, so the denominator for a
    # miss-rate estimate exists
    assert unjudged["triggered"] is False
    assert unjudged["control_sample"] is False
    assert "features" in unjudged
    assert unjudged["_question"] == ""


def test_control_rate_can_be_disabled_and_is_clamped(monkeypatch):
    import rag_shadow_answerability as shadow

    monkeypatch.setenv("RAG_SHADOW_CONTROL_RATE", "0")
    assert shadow.trigger_config()["control_rate"] == 0.0
    monkeypatch.setenv("RAG_SHADOW_CONTROL_RATE", "2")
    assert shadow.trigger_config()["control_rate"] == 1.0


def test_historical_seed_selection_excludes_tests_and_truncated_queries(tmp_path):
    from scripts.build_answerability_shadow_seed import extract_stream_history

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_api.py").write_text(
        'query = "测试里的问题"\n', encoding="utf-8"
    )
    events = [
        {"ts": "2026-01-01T00:00:00", "msg": "stream start q='真实问题'"},
        {"ts": "2026-01-01T00:00:01", "msg": "stream start q='真实问题'"},
        {"ts": "2026-01-01T00:00:02", "msg": "stream start q='测试里的问题'"},
        {"ts": "2026-01-01T00:00:03", "msg": "stream start q=" + repr("长" * 60)},
        {"ts": "2026-01-01T00:00:04", "msg": "stream start q='x'"},
    ]
    api_log = tmp_path / "api.log"
    api_log.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in events) + "\n",
        encoding="utf-8",
    )
    selected, audit = extract_stream_history(api_log, tests_dir)
    assert [row["question"] for row in selected] == ["真实问题"]
    assert selected[0]["occurrence_count"] == 2
    assert audit["selected_weighted_events"] == 2
    assert audit["exclusions_by_unique_query"] == {
        "known_automation_query": 1,
        "possibly_truncated_at_60_chars": 1,
        "present_in_test_source": 1,
    }


def test_seed_label_marks_correction_and_unsupported_accepts_as_interventions():
    from scripts.build_answerability_shadow_seed import _query_label

    correction = _query_label(True, [{
        "rank": 1,
        "verdict": {"grade": 3, "relation": "contradicts"},
        "action": "correct_premise",
    }])
    assert correction["intervention_type"] == "correct_premise"
    assert correction["intervention_needed"] is True

    unsupported = _query_label(True, [{
        "rank": 1,
        "verdict": {"grade": 1, "relation": "not_established"},
        "action": "abstain",
    }])
    assert unsupported["intervention_type"] == "block_unsupported"
    assert unsupported["intervention_needed"] is True


def test_the_tiebreak_only_contests_an_exact_top_tie():
    """The listwise call is a tiebreaker, not a second ranker: anything the
    grade already separates must stay separated."""
    import rag_answerability as ra

    seen = {}

    def fake_tiebreak(question, chunks, **kwargs):
        seen["chunks"] = list(chunks)
        return 1

    original, ra.tiebreak = ra.tiebreak, fake_tiebreak
    try:
        docs = ["winner", "gold", "partial", "unrelated"]
        grades = {"winner": (3, "entails"), "gold": (3, "entails"),
                  "partial": (2, "entails"), "unrelated": (0, "not_established")}
        stats = {}
        out, _m, _d, _s = ra.rerank_by_answerability(
            "q", docs, [{}] * 4, [0.1] * 4, [0.9, 0.8, 0.7, 0.6], stats=stats,
            tiebreak_ties=True,
            grader=lambda q, c: {"grade": grades[c][0], "relation": grades[c][1]})
    finally:
        ra.tiebreak = original
    # only the two grade-3 rows were contested
    assert seen["chunks"] == ["winner", "gold"]
    assert out == ["gold", "winner", "partial", "unrelated"]
    assert stats["tiebreak_candidates"] == 2 and stats["tiebreak_moved"] is True


def test_a_tiebreak_that_cannot_choose_leaves_the_order_alone():
    """``None`` has to mean 'no opinion', not 'the first one'."""
    import rag_answerability as ra

    original, ra.tiebreak = ra.tiebreak, lambda q, chunks, **kw: None
    try:
        docs = ["a", "b"]
        out, _m, _d, _s = ra.rerank_by_answerability(
            "q", docs, [{}] * 2, [0.1] * 2, [0.9, 0.8], tiebreak_ties=True,
            grader=lambda q, c: {"grade": 3, "relation": "entails"})
    finally:
        ra.tiebreak = original
    assert out == ["a", "b"]


def test_an_abstaining_relation_is_not_a_tie_candidate():
    """A grade-3 ``not_established`` chunk is an abstention case; contesting the
    top slot with it would smuggle the two-state collapse into ranking."""
    import rag_answerability as ra

    seen = {}
    original = ra.tiebreak
    ra.tiebreak = lambda q, chunks, **kw: seen.setdefault("n", len(chunks)) and None
    try:
        grades = {"a": (3, "not_established"), "b": (3, "not_established")}
        ra.rerank_by_answerability(
            "q", ["a", "b"], [{}] * 2, [0.1] * 2, [0.9, 0.8], tiebreak_ties=True,
            grader=lambda q, c: {"grade": grades[c][0], "relation": grades[c][1]})
    finally:
        ra.tiebreak = original
    assert "n" not in seen                      # never called


def test_tiebreak_rejects_an_out_of_range_or_unparseable_pick():
    from rag_answerability import tiebreak

    assert tiebreak("q", ["a", "b"], caller=lambda m, **k: '{"best": 9}',
                    use_cache=False) is None
    assert tiebreak("q", ["a", "b"], caller=lambda m, **k: "第二个",
                    use_cache=False) is None
    assert tiebreak("q", ["a"], caller=lambda m, **k: '{"best": 0}',
                    use_cache=False) is None    # nothing to break


def test_the_tiebreak_knob_is_off_by_default(monkeypatch):
    import rag_gate

    monkeypatch.delenv("RAG_ANSWERABILITY_TIEBREAK", raising=False)
    assert rag_gate._answerability_tiebreak() is False
    monkeypatch.setenv("RAG_ANSWERABILITY_TIEBREAK", "1")
    assert rag_gate._answerability_tiebreak() is True


def test_the_tiebreak_overrides_early_exit_instead_of_silently_never_firing():
    """Early exit stops once the incumbent holds the top grade, so nothing below
    it is ever graded and a tie cannot be observed.  Combining the two knobs
    without saying so gives a tiebreaker that never fires -- on Dev-New that was
    four of the six in-pool misses, gold sitting at rank 2-6 with no verdict."""
    import rag_answerability as ra

    seen = {}
    original, ra.tiebreak = ra.tiebreak, (
        lambda q, chunks, **kw: seen.setdefault("n", len(chunks)) and 1)
    try:
        docs = ["incumbent", "gold", "c"]
        stats = {}
        out, _m, _d, _s = ra.rerank_by_answerability(
            "q", docs, [{}] * 3, [0.1] * 3, [0.9, 0.8, 0.7], stats=stats,
            early_exit=True, tiebreak_ties=True,
            grader=lambda q, c: {"grade": 3, "relation": "entails"})
    finally:
        ra.tiebreak = original
    assert stats["calls"] == 3          # the whole head, not just the incumbent
    assert seen["n"] == 3               # ... and all three contested the top
    assert out[0] == "gold"


def test_consensus_votes_are_real_resamples_not_cache_echoes():
    """If the vote salt failed to reach the cache key, votes 2..N would be
    answered from vote 1's cache entry and every panel would be unanimous by
    construction -- consensus as decoration, the same silent-no-op shape as a
    dead env knob."""
    from rag_answerability import cache_key

    keys = {cache_key("m", "q", "c", variant=v) for v in ("", "vote2", "vote3")}
    assert len(keys) == 3
    # and the unsalted form is byte-stable so the existing cache stays valid
    assert cache_key("m", "q", "c") == cache_key("m", "q", "c", variant="")


def test_confirm_action_majority_and_failure_semantics():
    from rag_answerability import confirm_action

    ok = {"grade": 3, "relation": "entails"}
    nope = {"grade": 1, "relation": "not_established"}

    def grader_of(seq):
        queue = list(seq)
        return lambda q, c, variant="": queue.pop(0)

    assert confirm_action("q", "c", ok, grader=grader_of([ok, ok]))["action"] == "answer"
    assert confirm_action("q", "c", ok, grader=grader_of([nope, ok]))["action"] == "answer"
    # two dissents kill the acceptance
    assert confirm_action("q", "c", ok, grader=grader_of([nope, nope]))["action"] == "abstain"
    # a failed vote counts as an abstention, never as a smaller panel
    assert confirm_action("q", "c", ok, grader=grader_of([None, None]))["action"] == "abstain"
    # correct_premise survives confirmation as itself, not as "answer"
    cp = {"grade": 3, "relation": "contradicts"}
    assert confirm_action("q", "c", cp, grader=grader_of([cp, cp]))["action"] == "correct_premise"


def test_the_gate_panel_is_asymmetric_by_source():
    """Only an accepting single verdict is escalated: the measured failure mode
    was flips toward acceptance, so refusal must never be argued back open by
    extra votes."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("rag_gate.py").read_text(encoding="utf-8")
    block = source.split("_ans_accept = False")[1].split("_rr_gate_raw")[0]
    # confirm_action is called only inside the branch already gated on the
    # single verdict being actionable
    before_confirm = block.split("confirm_action")[0]
    assert '!= "abstain"' in before_confirm


def test_gate_votes_default_to_three(monkeypatch):
    import rag_gate

    monkeypatch.delenv("RAG_ANSWERABILITY_GATE_VOTES", raising=False)
    assert rag_gate._answerability_gate_votes() == 3
    monkeypatch.setenv("RAG_ANSWERABILITY_GATE_VOTES", "1")
    assert rag_gate._answerability_gate_votes() == 1
    for junk in ("0", "-2", "abc", ""):
        monkeypatch.setenv("RAG_ANSWERABILITY_GATE_VOTES", junk)
        assert rag_gate._answerability_gate_votes() == 3
