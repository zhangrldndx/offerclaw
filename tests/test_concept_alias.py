# -*- coding: utf-8 -*-
"""概念别名:默认关 / 字段矩阵 / 解析容错 / 版本隔离 / 证据边界。全部打桩零 LLM。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_alias as al  # noqa: E402


def test_default_off(monkeypatch):
    monkeypatch.delenv("RAG_CONCEPT_ALIAS", raising=False)
    assert al.alias_enabled() is False
    monkeypatch.setenv("RAG_CONCEPT_ALIAS", "1")
    assert al.alias_enabled() is True


def test_alias_fields_matrix(monkeypatch):
    """指导 §17 的 A1-A4 靠这个旋钮切换;坏值回退全字段而不是炸掉。"""
    monkeypatch.delenv("RAG_ALIAS_FIELDS", raising=False)
    assert al.alias_fields() == al._ALL_FIELDS
    monkeypatch.setenv("RAG_ALIAS_FIELDS", "keywords,questions")
    assert al.alias_fields() == ("zh_keywords", "zh_candidate_questions")
    monkeypatch.setenv("RAG_ALIAS_FIELDS", "没这个字段")
    assert al.alias_fields() == al._ALL_FIELDS


def test_parse_tolerates_wrapping_and_bad_json():
    raw = '这是说明\n```json\n{"zh_keywords":["分页","记忆"],"zh_summary":"讲分页"}\n```'
    got = al._parse(raw)
    assert got["zh_keywords"] == ["分页", "记忆"] and got["zh_summary"] == "讲分页"
    assert al._parse("完全不是 JSON") == {}       # 失败返回空 → 该块跳过,重跑自动补
    assert al._parse(None) == {}


def test_alias_text_joins_only_selected_fields():
    row = {"zh_concepts": ["虚拟上下文"], "zh_keywords": ["分页", "记忆"],
           "zh_summary": "讲的是分页", "zh_candidate_questions": ["怎么分页?"]}
    t = al._alias_text(row, ("zh_keywords", "zh_candidate_questions"))
    assert "分页、记忆" in t and "怎么分页?" in t
    assert "虚拟上下文" not in t and "讲的是分页" not in t


def test_alias_map_filters_by_version_and_caches(tmp_path, monkeypatch):
    """版本不符的旧别名必须被忽略——改提示词/字段后不能拿旧数据冒充新规格。"""
    p = tmp_path / "alias.jsonl"
    p.write_text(
        json.dumps({"doc_key": "k1", "alias_version": al.ALIAS_VERSION,
                    "zh_keywords": ["分页"]}, ensure_ascii=False) + "\n"
        + json.dumps({"doc_key": "k_old", "alias_version": "1999-01-01",
                      "zh_keywords": ["旧的"]}, ensure_ascii=False) + "\n"
        + "半截行不是 json\n", encoding="utf-8")
    monkeypatch.setattr(al, "CACHE_PATH", str(p))
    al._MAP_CACHE.clear()
    m = al.alias_map(("zh_keywords",))
    assert m == {"k1": "分页"}                    # 旧版本与坏行都被跳过


def test_alias_map_empty_without_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "CACHE_PATH", str(tmp_path / "不存在.jsonl"))
    al._MAP_CACHE.clear()
    assert al.alias_map() == {}                   # 无缓存 → 空 map,调用方静默退化


def test_generate_alias_uses_large_budget(monkeypatch):
    """预算必须足够大:兜底是推理模型,reasoning 与 content 共用 max_tokens,
    给 600 实测 finish_reason=length 且 content 空串(调用成功但产出为 0)。"""
    seen = {}

    def fake_chat(msgs, max_tokens=None, temperature=None, **kw):
        seen["max_tokens"] = max_tokens
        seen["extra"] = kw.get("extra_payload")
        return '{"zh_keywords":["分页"]}'

    monkeypatch.setattr("rag_gate._chat", fake_chat)
    monkeypatch.delenv("RAG_ALIAS_MAX_TOKENS", raising=False)
    assert al.generate_alias("some english text") == {"zh_keywords": ["分页"]}
    assert seen["max_tokens"] >= 3000
    assert seen["extra"] is None                  # 不传 enable_thinking(那是 qwen3 的旋钮)


def test_discount_and_lift_gate_knobs(monkeypatch):
    """δ/τ/N 三个校准旋钮:默认全关(= 无条件 max,与改造前逐字节等价)。"""
    for k in ("RAG_ALIAS_DISCOUNT", "RAG_ALIAS_LIFT_GATE", "RAG_ALIAS_SCOPE"):
        monkeypatch.delenv(k, raising=False)
    assert al.alias_discount() == 0.0 and al.alias_lift_gate() == 0.0
    assert al.alias_scope() == 0
    monkeypatch.setenv("RAG_ALIAS_DISCOUNT", "0.03")
    monkeypatch.setenv("RAG_ALIAS_LIFT_GATE", "0.05")
    monkeypatch.setenv("RAG_ALIAS_SCOPE", "2")
    assert al.alias_discount() == 0.03 and al.alias_lift_gate() == 0.05
    assert al.alias_scope() == 2
    monkeypatch.setenv("RAG_ALIAS_DISCOUNT", "坏值")
    assert al.alias_discount() == 0.0          # 坏值回默认,不炸
    monkeypatch.setenv("RAG_ALIAS_DISCOUNT", "-1")
    assert al.alias_discount() == 0.0          # 负折扣无意义,夹到 0


def test_rerank_applies_alias_discount_and_lift_gate(monkeypatch):
    """桥分校准的核心语义:lift<τ 不采信;否则 max(原分, 别名分-δ)。"""
    import rag_rerank as rr

    class _M:
        max_seq_length = 512

        def predict(self, pairs):
            # pairs = [(q,doc)] + [(q,alias)];原分 0.50,别名分 0.90 → lift 0.40
            return [0.50, 0.90]

    monkeypatch.setattr(rr, "_load_reranker", lambda: _M())
    monkeypatch.setattr(rr, "rerank_enabled", lambda: True)
    docs, metas = ["d1"], [{"source": "p.md"}]
    amap = {__import__("rag_doc2query").doc_key("d1", metas[0]): "中文别名"}

    diag = {}
    _d, _m, _k, s = rr.rerank("q", docs, metas, [0.2], 1, alias_map=amap,
                              alias_discount=0.0, diag=diag)
    assert s == [0.9] and diag[0]["applied"] is True and diag[0]["lift"] == 0.4

    _d, _m, _k, s = rr.rerank("q", docs, metas, [0.2], 1, alias_map=amap,
                              alias_discount=0.05)
    assert s == [0.85]                       # 0.90 - 0.05

    _d, _m, _k, s = rr.rerank("q", docs, metas, [0.2], 1, alias_map=amap,
                              alias_discount=0.0, alias_lift_gate=0.5)
    assert s == [0.5]                        # lift 0.4 < τ 0.5 → 不采信别名分

    _d, _m, _k, s = rr.rerank("q", docs, metas, [0.2], 1, alias_map=amap,
                              alias_discount=0.9)
    assert s == [0.5]                        # 折扣过大 → 退回原分(取 max 保底)
