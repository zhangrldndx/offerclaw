# -*- coding: utf-8 -*-
"""V4 §5 — Profile Schema validation + 分层 Memory 测试。"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# =====================================================
# Profile schema
# =====================================================

def test_schema_file_loads():
    from profile_loader import load_schema
    s = load_schema()
    assert s.get("title") == "OfferClawProfileV1"
    assert "学历" in s["properties"]
    assert "英语自评" in s["properties"]


def test_public_demo_profile_passes_schema():
    """公开合成画像必须满足 schema，否则 schema 与 loader 漂移。"""
    from profile_loader import load_profile, validate_profile
    p = load_profile(
        path=os.path.join(ROOT, "tests", "fixtures", "missing-profile.md"),
        use_cache=False,
    )
    ok, errs = validate_profile(p)
    assert ok, f"合成画像不通过 schema：{errs}"


def test_all_three_persona_fixtures_pass_schema():
    """profiles/p*.json 全部必须通过 schema。"""
    from profile_loader import validate_profile

    failures = []
    for fn in ("p1_demo_ai", "p2_cs_backend_intern", "p3_phd_algo_research"):
        path = os.path.join(ROOT, "profiles", f"{fn}.json")
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        p.pop("persona_id", None)
        p.pop("desc", None)
        ok, errs = validate_profile(p)
        if not ok:
            failures.append((fn, errs))
    assert not failures, f"persona fixture 不通过 schema：{failures}"


def test_schema_rejects_invalid_education_enum():
    from profile_loader import validate_profile
    bad = {
        "学历": "院士",  # 不在 enum
        "可接受地域": ["上海"],
        "方向优先级": [],
        "明确不做": [],
        "熟练技能": [],
        "会用技能": [],
        "项目数量": 0,
        "实习数量": 0,
        "英语自评": 1,
    }
    ok, errs = validate_profile(bad)
    assert not ok
    assert any("学历" in e for e in errs), errs


def test_schema_rejects_out_of_range_english():
    from profile_loader import validate_profile
    bad = {
        "学历": "本科",
        "可接受地域": [],
        "方向优先级": [],
        "明确不做": [],
        "熟练技能": [],
        "会用技能": [],
        "项目数量": 0,
        "实习数量": 0,
        "英语自评": 10,  # 1-5 区间外
    }
    ok, errs = validate_profile(bad)
    assert not ok
    assert any("英语自评" in e for e in errs), errs


def test_schema_rejects_missing_required_field():
    from profile_loader import validate_profile
    bad = {"学历": "本科"}  # 缺其他必填
    ok, errs = validate_profile(bad)
    assert not ok
    assert len(errs) >= 3, f"至少 3 个 required 字段缺失：{errs}"


# =====================================================
# Memory layers
# =====================================================

@pytest.fixture
def tmp_memory(tmp_path):
    return str(tmp_path / "mem")


# --------- Episodic ---------

def test_episodic_append_and_read(tmp_memory):
    from memory_layers import EpisodicMemory
    epi = EpisodicMemory(base_dir=tmp_memory)
    e1 = epi.append({"kind": "match_run", "status": "当前适合投递",
                     "jd_title": "AI 实习"})
    e2 = epi.append({"kind": "match_run", "status": "中长期可转向"})
    assert e1["id"].startswith("ep_")
    assert "ts_iso" in e1

    all_events = epi.all()
    assert len(all_events) == 2
    assert all_events[0]["status"] == "当前适合投递"


def test_episodic_recent_returns_tail(tmp_memory):
    from memory_layers import EpisodicMemory
    epi = EpisodicMemory(base_dir=tmp_memory)
    for i in range(5):
        epi.append({"kind": "match_run", "i": i})
    recent = epi.recent(limit=3)
    assert len(recent) == 3
    assert recent[-1]["i"] == 4


def test_episodic_filter_predicate(tmp_memory):
    from memory_layers import EpisodicMemory
    epi = EpisodicMemory(base_dir=tmp_memory)
    epi.append({"kind": "match_run", "status": "当前适合投递"})
    epi.append({"kind": "daily_log"})
    epi.append({"kind": "match_run", "status": "中长期可转向"})

    suitable = epi.filter(lambda e: e.get("status") == "当前适合投递")
    assert len(suitable) == 1


def test_episodic_count_by(tmp_memory):
    from memory_layers import EpisodicMemory
    epi = EpisodicMemory(base_dir=tmp_memory)
    epi.append({"kind": "match_run", "direction": "主方向"})
    epi.append({"kind": "match_run", "direction": "主方向"})
    epi.append({"kind": "match_run", "direction": "派生方向"})

    dist = epi.count_by("direction")
    assert dist == {"主方向": 2, "派生方向": 1}


# --------- Semantic ---------

def test_semantic_set_get_round_trip(tmp_memory):
    from memory_layers import SemanticMemory
    sem = SemanticMemory(base_dir=tmp_memory)
    sem.set("prefer_remote", True)
    sem.set("avoid_keywords", ["java"])

    assert sem.get("prefer_remote") is True
    assert sem.get("avoid_keywords") == ["java"]
    assert sem.get("missing", "default") == "default"


def test_semantic_updated_at_meta_present(tmp_memory):
    from memory_layers import SemanticMemory
    sem = SemanticMemory(base_dir=tmp_memory)
    sem.set("x", 1)
    data = sem.all()
    assert "_meta" in data
    assert "updated_at" in data["_meta"]


def test_semantic_delete(tmp_memory):
    from memory_layers import SemanticMemory
    sem = SemanticMemory(base_dir=tmp_memory)
    sem.set("x", 1)
    assert sem.delete("x") is True
    assert sem.delete("missing") is False
    assert sem.get("x") is None


# --------- Procedural ---------

def test_procedural_add_get_list(tmp_memory):
    from memory_layers import ProceduralMemory
    proc = ProceduralMemory(base_dir=tmp_memory)
    proc.add("nio_resume",
             body="强调 LangGraph / RAG / FastAPI",
             trigger="JD 含 大模型应用开发实习")
    proc.add("java_avoid", body="跳过", trigger="JD 含 精通 Java")

    sop = proc.get("nio_resume")
    assert sop and sop["trigger"] == "JD 含 大模型应用开发实习"
    sops = proc.list()
    assert len(sops) == 2


def test_procedural_remove(tmp_memory):
    from memory_layers import ProceduralMemory
    proc = ProceduralMemory(base_dir=tmp_memory)
    proc.add("x", body="b")
    assert proc.remove("x") is True
    assert proc.remove("nonexistent") is False
    assert proc.get("x") is None


# --------- Distillation ---------

def test_distill_writes_status_distribution_to_semantic(tmp_memory):
    from memory_layers import (EpisodicMemory, SemanticMemory,
                               distill_to_semantic)
    epi = EpisodicMemory(base_dir=tmp_memory)
    sem = SemanticMemory(base_dir=tmp_memory)
    epi.append({"kind": "match_run", "status": "当前适合投递",
                "direction": "主方向"})
    epi.append({"kind": "match_run", "status": "当前适合投递",
                "direction": "主方向"})
    epi.append({"kind": "match_run", "status": "中长期可转向",
                "direction": "派生方向"})
    epi.append({"kind": "daily_log"})  # 非 match 事件应被忽略

    out = distill_to_semantic(epi, sem)
    assert out["distilled"] is True
    assert out["n_match_events"] == 3

    dist = sem.get("match_status_distribution")
    assert dist == {"suitable": 2, "stretch": 1}
    # 系统给出的岗位匹配结论不能自动改写用户的职业方向偏好。
    assert sem.get("preferred_direction") is None
    assert sem.get("last_distilled_at")


def test_distill_no_match_events_returns_skip(tmp_memory):
    from memory_layers import (EpisodicMemory, SemanticMemory,
                               distill_to_semantic)
    epi = EpisodicMemory(base_dir=tmp_memory)
    sem = SemanticMemory(base_dir=tmp_memory)
    out = distill_to_semantic(epi, sem)
    assert out["distilled"] is False
    assert out["reason"] == "no_trusted_events"


# --------- Path injection isolation ---------

def test_all_three_layers_share_same_base_dir(tmp_memory):
    from memory_layers import EpisodicMemory, SemanticMemory, ProceduralMemory
    epi = EpisodicMemory(base_dir=tmp_memory)
    sem = SemanticMemory(base_dir=tmp_memory)
    proc = ProceduralMemory(base_dir=tmp_memory)

    epi.append({"kind": "test"})
    sem.set("k", "v")
    proc.add("sop", body="b")

    files = os.listdir(tmp_memory)
    assert "episodic.jsonl" in files
    assert "semantic.json" in files
    assert "procedural.json" in files
