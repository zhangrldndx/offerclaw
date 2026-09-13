# -*- coding: utf-8 -*-
"""检索 Profile 声明/指纹 + 别名资产治理 + 英文可达性自检。全部纯逻辑,零模型零网络。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_alias_health as ah  # noqa: E402
import rag_profiles as rp  # noqa: E402
import rag_reachability as rc  # noqa: E402


# ---------------- Profile ----------------

def test_baseline_profile_is_all_off():
    """b0 基线必须是"全部实验特性关闭"——它是回滚终点,不能悄悄带上任何开关。"""
    assert rp.profile_env("b0_chinese_baseline") == {}


def test_both_profiles_keep_english_quota_nonzero():
    """指导 §8.2 的硬要求:Default Profile **也**保留英文候选,配额永不为 0。
    这正是"论文域完全不可达"事故的防线——路由是配额调节器,不是候选开关。"""
    for name in ("default_mixed", "paper_quality"):
        env = rp.profile_env(name)
        assert env["RAG_EN_QUOTA"] == "1"
        assert int(env["RAG_EN_QUOTA_K"]) > 0


def test_profile_hash_changes_with_config(monkeypatch):
    """指标必须绑配置:配置一变指纹就变(旧指标自动 stale)。"""
    h1 = rp.retrieval_profile_hash("paper_quality")
    assert h1 == rp.retrieval_profile_hash("paper_quality")        # 同配置稳定
    assert h1 != rp.retrieval_profile_hash("default_mixed")        # 不同 Profile 不同
    monkeypatch.setenv("RAG_RERANK_MAX_SEQ", "384")                # 非 Profile 内旋钮
    assert rp.retrieval_profile_hash("paper_quality") != h1        # 也要纳入指纹


def test_apply_profile_sets_env():
    for k in ("RAG_EN_QUOTA", "RAG_EN_QUOTA_K", "RAG_CONCEPT_ALIAS"):
        os.environ.pop(k, None)
    meta = rp.apply_profile("paper_quality")
    assert os.environ["RAG_EN_QUOTA"] == "1" and len(meta["retrieval_profile_hash"]) == 12
    for k in rp.profile_env("paper_quality"):
        os.environ.pop(k, None)


def test_unknown_profile_raises():
    try:
        rp.profile_env("不存在的profile")
        assert False, "未知 Profile 应报错而不是静默给空配置"
    except KeyError:
        pass


# ---------------- 别名资产治理 ----------------

def test_validate_flags_missing_fields():
    st, probs = ah.validate({"alias_version": __import__("rag_alias").ALIAS_VERSION,
                             "zh_keywords": ["分页"]})
    assert st == "partial" and any("questions" in p for p in probs)


def test_validate_detects_stale_source_and_prompt():
    from rag_alias import ALIAS_VERSION
    rec = {"alias_version": ALIAS_VERSION, "zh_keywords": ["分页"],
           "zh_candidate_questions": ["怎么分页?"],
           "source_content_hash": ah.content_hash("原文A")}
    assert ah.validate(rec, "原文A")[0] == "succeeded"
    assert ah.validate(rec, "原文B已改")[0] == "stale"        # 原文变了 → 失效
    rec2 = dict(rec, prompt_hash="deadbeef")
    assert ah.validate(rec2, "原文A")[0] == "stale"           # 提示词变了 → 失效


def test_validate_detects_version_mismatch():
    rec = {"alias_version": "1999-01-01", "zh_keywords": ["x"],
           "zh_candidate_questions": ["y"]}
    assert ah.validate(rec)[0] == "stale"


def test_validate_detects_eval_leakage():
    """§10.4 泄漏红线:别名里出现评测题面 = 把答案写进索引,评测全部失效。"""
    from rag_alias import ALIAS_VERSION
    q = "在有限窗口内制造无限上下文的错觉"
    rec = {"alias_version": ALIAS_VERSION, "zh_keywords": ["分页"],
           "zh_candidate_questions": [q]}
    st, probs = ah.validate(rec, leak_terms=[q])
    assert st != "succeeded" and any("泄漏" in p for p in probs)


def test_validate_requires_chinese_content():
    from rag_alias import ALIAS_VERSION
    rec = {"alias_version": ALIAS_VERSION, "zh_keywords": ["paging"],
           "zh_candidate_questions": ["how to page?"]}
    assert any("中文" in p for p in ah.validate(rec)[1])


# ---------------- 可达性自检 ----------------

def test_reachability_flags_unreachable_corpus(monkeypatch):
    """英文语料存在但入口全关 = 生产上完全不可达,必须显式报出来。"""
    monkeypatch.setattr(rc, "_collection_count", lambda name: 194)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": False, "paper_route": False})
    r = rc.check_reachability(strict=False)
    assert r["ok"] is False and r["code"] == rc.PAPER_RETRIEVAL_UNREACHABLE
    assert "RAG_EN_QUOTA=1" in r["detail"]          # 修法要可执行


def test_reachability_ok_when_any_entrypoint_on(monkeypatch):
    monkeypatch.setattr(rc, "_collection_count", lambda name: 194)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": True, "paper_route": False})
    assert rc.check_reachability(strict=False)["ok"] is True


def test_reachability_ok_when_no_english_corpus(monkeypatch):
    """没有英文语料时"入口全关"是正确状态,不该报警。"""
    monkeypatch.setattr(rc, "_collection_count", lambda name: 0)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": False, "paper_route": False})
    assert rc.check_reachability(strict=False)["ok"] is True


def test_reachability_strict_raises(monkeypatch):
    monkeypatch.setattr(rc, "_collection_count", lambda name: 194)
    monkeypatch.setattr(rc, "english_entrypoints",
                        lambda: {"quota": False, "paper_route": False})
    try:
        rc.check_reachability(strict=True)
        assert False, "strict 模式应抛出"
    except RuntimeError as e:
        assert rc.PAPER_RETRIEVAL_UNREACHABLE in str(e)


def test_reachability_entrypoint_on_but_collection_missing(monkeypatch):
    """2026-08-24 实测盲区:开关开着但背后的集合被删(kb_paper_bge_v1 清理事故)——
    通道 fail-soft 成永远空,自检必须仍报不可达,不能被"开关状态"骗过。"""
    monkeypatch.setenv("RAG_EN_QUOTA", "1")
    monkeypatch.delenv("RAG_PAPER_ROUTE", raising=False)
    monkeypatch.setattr(rc, "_collection_count",
                        lambda name: 0 if name == "kb_paper_bge_v1" else 194)
    r = rc.check_reachability(strict=False)
    assert r["entrypoints"]["quota"] is False      # 集合没了 → 入口不算可用
    assert r["ok"] is False and r["code"] == rc.PAPER_RETRIEVAL_UNREACHABLE
