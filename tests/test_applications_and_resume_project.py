# -*- coding: utf-8 -*-
"""投递管理（applications_store）+ 简历工坊（resume_project）回归。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import applications_store as ap  # noqa: E402
import resume_project as rp  # noqa: E402

_TABLE = """# 投递追踪

| 日期 | 公司 | 岗位 | 来源 | 地点 | 匹配结论 | 样本定位 | 当前状态 | 下一步动作 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-22 | [DEMO] Acme | LLM 实习 | BOSS | 上海 | 暂不 | x | 不投递 | [示例] | — |
"""


@pytest.fixture()
def tmp_apps(tmp_path, monkeypatch):
    p = tmp_path / "applications.md"
    p.write_text(_TABLE, encoding="utf-8")
    monkeypatch.setattr(ap, "APPLICATIONS_PATH", str(p))
    return p


def test_upsert_adds_new_row(tmp_apps):
    r = ap.upsert_application("字节跳动", "大模型应用实习生", "已投递",
                              date="2026-06-04", next_action="等一周反馈")
    assert r["status"] == "ok" and r["action"] == "added"
    md = tmp_apps.read_text(encoding="utf-8")
    assert "| 字节跳动 | 大模型应用实习生 |" in md.replace("  ", " ")
    assert "已投递" in md and "等一周反馈" in md


def test_upsert_updates_existing_and_appends_timeline(tmp_apps):
    ap.upsert_application("字节跳动", "大模型应用实习生", "已投递", date="2026-06-04")
    r2 = ap.upsert_application("字节跳动", "大模型应用实习生", "面试中",
                               date="2026-06-10", next_action="准备一面八股")
    assert r2["action"] == "updated"
    md = tmp_apps.read_text(encoding="utf-8")
    assert md.count("字节跳动") == 1           # 不重复加行
    assert "面试中" in md and "06-10 面试中" in md  # 备注里有时间线


def test_patch_status_uses_today_and_appends_one_timeline_event(tmp_apps, monkeypatch):
    added = ap.upsert_application("华为", "AI 工程师", "准备投递", date="2026-08-21")
    monkeypatch.setattr(ap, "_today", lambda: "2026-09-09")

    out = ap.patch_application(added["application_id"], status="已投递")

    assert out["status"] == "ok"
    row = ap.get_application(added["application_id"])
    assert row["status"] == "已投递"
    assert row["date"] == "2026-09-09"
    assert [event["status"] for event in row["timeline"]] == ["准备投递", "已投递"]


def test_patch_non_status_field_does_not_duplicate_timeline(tmp_apps):
    added = ap.upsert_application("华为", "AI 工程师", "准备投递", date="2026-08-21")

    out = ap.patch_application(added["application_id"], plan_priority="high")

    assert out["status"] == "ok"
    row = ap.get_application(added["application_id"])
    assert row["date"] == "2026-08-21"
    assert row["plan_priority"] == "high"
    assert [event["status"] for event in row["timeline"]] == ["准备投递"]


def test_upsert_can_backfill_source_and_location(tmp_apps):
    ap.upsert_application("字节跳动", "大模型应用实习生", "准备投递", date="2026-06-04")
    ap.upsert_application("字节跳动", "大模型应用实习生", "准备投递",
                          date="2026-06-05", source="官网", location="上海")
    row = ap.list_applications()[0]
    assert row["来源"] == "官网"
    assert row["地点"] == "上海"


def test_upsert_rejects_bad_status(tmp_apps):
    r = ap.upsert_application("X", "Y", "瞎写的状态")
    assert r["status"] == "error"


def test_demo_rows_filtered(tmp_apps, monkeypatch):
    rows = ap.list_applications()
    assert not any("[DEMO]" in str(r) for r in rows)


def test_save_experience_writes_kb_format(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "EXPERIENCE_DIR", str(tmp_path / "exp"))
    r = ap.save_experience("字节跳动", "大模型应用实习生", "一面",
                           "一面考了 RAG 的 rerank 与多路召回，追问了去重策略；建议先把混合检索吃透。")
    assert r["status"] == "ok"
    body = open(r["saved_abs"], encoding="utf-8").read()
    assert 'source_type: "experience"' in body and "亲历" in body and "rerank" in body


def test_save_experience_rejects_too_short(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "EXPERIENCE_DIR", str(tmp_path / "exp"))
    assert ap.save_experience("A", "B", "一面", "太短")["status"] == "error"


def test_list_experiences_reads_saved_record_and_filters_generic_docs(tmp_path, monkeypatch):
    exp_dir = tmp_path / "exp"
    monkeypatch.setattr(ap, "EXPERIENCE_DIR", str(exp_dir))
    saved = ap.save_experience(
        "华为计算产品线", "AI应用工程师", "机考通过",
        "机考后需要继续准备手撕与基础八股，并记录后续两轮技术面的时间点。",
    )
    assert saved["status"] == "ok"
    (exp_dir / "generic.md").write_text("# 通用项目包装指导\n\n不是个人投递记录", encoding="utf-8")

    rows = ap.list_experiences()
    assert len(rows) == 1
    assert rows[0]["company"] == "华为计算产品线"
    assert rows[0]["position"] == "AI应用工程师"
    assert rows[0]["stage"] == "机考通过"
    assert "继续准备手撕" in rows[0]["summary"]


# ---------------- resume_project ----------------

def test_gather_material_prefers_text():
    out = rp.gather_material(repo_url="https://github.com/x/y", text="项目介绍正文")
    assert out["status"] == "ok" and out["material"] == "项目介绍正文"


def test_gather_material_requires_input():
    assert rp.gather_material()["status"] == "error"


def test_build_messages_uses_default_pattern_without_materials(monkeypatch):
    monkeypatch.setattr(rp, "load_materials", lambda: {"guidance": [], "examples": []})
    msgs = rp.build_project_messages("一个 RAG 项目的介绍", "OfferClaw")
    sysm = msgs[0]["content"]
    assert "内置默认模式" in sysm and "绝不编造数字" in sysm
    assert "OfferClaw" in msgs[1]["content"]


_EXAMPLE_MD = """## 简历X
#### 某 RAG 问答系统
- 项目简介：面向问答场景。
- 技术栈：Python、Chroma。
- 技术亮点：
  1. **混合检索**：实现 BM25 + 向量混合召回。
"""


def test_build_messages_injects_user_materials(monkeypatch):
    monkeypatch.setattr(rp, "load_materials", lambda: {
        "guidance": [{"name": "resume_writing_notes.md", "content": "项目经历最重要，信息密度要高。"}],
        "examples": [{"name": "real_resume_examples.md", "content": _EXAMPLE_MD}],
    })
    msgs = rp.build_project_messages("素材", "")
    sysm = msgs[0]["content"]
    assert "写作原则" in sysm and "信息密度要高" in sysm          # 指导注入
    assert "格式范例" in sysm and "某 RAG 问答系统" in sysm        # 范例条目注入
    assert "输出结构（从用户提供的真实简历范例中学习" in sysm        # 蒸馏模式注入
    assert "严禁混入输出" in sysm                                 # 防串味纪律
    assert "内置默认模式" not in sysm


def test_extract_project_blocks_picks_only_project_entries():
    md = ("### 实习经历\n#### 某公司实习\n- 工作内容：日常开发\n"
          "### 项目经历\n#### 某 Agent 项目\n- 项目简介：x\n- 技术栈：y\n- 技术亮点：z\n")
    blocks = rp.extract_project_blocks(md)
    assert len(blocks) == 1 and "某 Agent 项目" in blocks[0]   # 实习小节（无技术栈等关键词）被跳过


def test_build_messages_jd_mode_adds_customization(monkeypatch):
    """jd_text 非空 → 注入 JD 定制要求 + 补强建议小节；事实纪律保留。"""
    monkeypatch.setattr(rp, "load_materials", lambda: {"guidance": [], "examples": []})
    msgs = rp.build_project_messages("项目素材", "X", jd_text="岗位要求：精通检索增强生成与多路召回")
    sysm = msgs[0]["content"]
    assert "目标 JD（定制依据）" in sysm and "精通检索增强生成" in sysm
    assert "优先挑选并前置" in sysm and "定制建议" in sysm
    assert "不许为迎合 JD 编造能力" in sysm


def test_build_messages_no_jd_mode_by_default(monkeypatch):
    monkeypatch.setattr(rp, "load_materials", lambda: {"guidance": [], "examples": []})
    msgs = rp.build_project_messages("项目素材")
    assert "目标 JD（定制依据）" not in msgs[0]["content"]


def test_guidance_classified_by_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "TEMPLATES_DIR", str(tmp_path))
    (tmp_path / "resume_writing_notes.md").write_text("指导内容" * 20, encoding="utf-8")
    (tmp_path / "my_resume.md").write_text("#### 项目A\n- 技术栈：x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("说明", encoding="utf-8")
    m = rp.load_materials()
    assert [g["name"] for g in m["guidance"]] == ["resume_writing_notes.md"]
    assert [e["name"] for e in m["examples"]] == ["my_resume.md"]
