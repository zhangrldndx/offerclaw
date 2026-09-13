"""产品级 Agent 化指导 §4 验收：状态真实化测试。

覆盖：
1. ``profile_loader.load_profile()`` 返回 13 个必备字段；
2. ``/api/match`` 不再硬编码 ``DEMO_PROFILE``（源码静态扫描 + 行为校验）；
3. 一份合成 JD 与公开合成画像能返回三档结论。
"""

from __future__ import annotations

import os
import re
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------- profile_loader 单测 ---------

def test_load_profile_returns_all_required_keys():
    from profile_loader import REQUIRED_KEYS, load_profile, reset_cache

    reset_cache()
    p = load_profile(
        path=os.path.join(ROOT, "tests", "fixtures", "missing-profile.md"),
        use_cache=False,
    )
    for k in REQUIRED_KEYS:
        assert k in p, f"profile 缺字段：{k}"
    # 类型契约（与 match_job.DEMO_PROFILE 一致）
    assert isinstance(p["学历"], str) and p["学历"]
    assert isinstance(p["可接受地域"], list)
    assert isinstance(p["方向优先级"], list)
    assert isinstance(p["明确不做"], list)
    assert isinstance(p["熟练技能"], list)
    assert isinstance(p["会用技能"], list)
    assert isinstance(p["项目数量"], int)
    assert isinstance(p["实习数量"], int)
    assert isinstance(p["英语自评"], int) and 1 <= p["英语自评"] <= 5


def test_load_profile_parses_synthetic_profile(tmp_path):
    """画像解析回归只使用合成数据，不依赖开发者本机状态。"""
    from profile_loader import load_profile, reset_cache

    profile = tmp_path / "synthetic_profile.md"
    profile.write_text(
        "# 合成画像\n"
        "## 1. 基础信息\n"
        "- 学历层次：本科\n"
        "- 专业：软件工程\n"
        "- 所在地：杭州\n"
        "- 可接受工作地域：杭州/上海/远程\n\n"
        "## 2. 求职方向与偏好\n"
        "- 目标方向（按优先级排序）：\n"
        "  1. 数据工程\n"
        "- 目标岗位类型：校招\n"
        "- 行业偏好：软件服务\n"
        "- 明确不做的方向：\n"
        "  - Go：非目标方向\n"
        "- 工作性质偏好：不限\n"
        "- 期望薪资区间：面议\n\n"
        "## 3. 技能清单\n"
        "- 编程语言：\n"
        "  - 熟练：Python\n"
        "  - 会用：FastAPI / SQL\n"
        "- 工具与框架：RAG / LangGraph\n\n"
        "## 4. 项目经历\n"
        "- 项目 1\n"
        "  - 名称：合成检索服务\n"
        "  - 技术栈：RAG / LangGraph / FastAPI\n"
        "  - 做了什么：实现检索与评测闭环\n\n"
        "## 7. 实习 / 工作经历\n"
        "- 实习 1：【待补充】\n\n"
        "## 9. 英语能力\n"
        "- 英语自评：3\n",
        encoding="utf-8",
    )
    reset_cache()
    p = load_profile(path=str(profile), include_evidence=True, use_cache=False)
    assert p["_source"] == "user_profile.md"
    assert p["学历"] == "本科"
    assert p["专业"] == "软件工程"
    assert p["所在地"] == "杭州"
    assert "上海" in p["可接受地域"]
    assert "数据工程" in p["方向优先级"][0]
    assert p["目标岗位类型"] == "校招"
    assert p["行业偏好"] == "软件服务"
    assert "go" in p["明确不做"]
    assert "Python" in p["熟练技能"]
    assert "FastAPI" in p["会用技能"]
    assert {"RAG", "LangGraph", "FastAPI"}.issubset(set(p["技能证据"]))
    assert {"RAG", "LangGraph"}.issubset(set(p["项目技能"]))
    assert p["项目数量"] >= 1
    project_evidence = [item for item in p["_evidence_catalog"]
                        if item["evidence_type"] == "project"]
    assert project_evidence
    assert any("检索" in item["text"] for item in project_evidence)
    assert all("待补充" not in item["text"] for item in p["_evidence_catalog"])


def test_load_profile_falls_back_when_md_missing(tmp_path):
    from profile_loader import REQUIRED_KEYS, load_profile, reset_cache

    reset_cache()
    fake_md = tmp_path / "missing.md"
    p = load_profile(path=str(fake_md), use_cache=False)
    assert all(k in p for k in REQUIRED_KEYS)
    assert p["_source"] in {"profiles/p1_demo_ai.json", "safe_default"}


def test_load_profile_actually_responds_to_md_edits(tmp_path):
    """状态驱动核心契约：修改 user_profile.md 后，list 字段必须真的变。

    回归 bug：早期版本因为 fallback 永远兜底 ['java']，从画像里删掉 Java
    一项时 ``明确不做`` 仍然返回 ['java']，导致 ``/api/match`` 行为不变，
    'state-driven' 名存实亡。"""
    from profile_loader import load_profile, reset_cache

    md = tmp_path / "p.md"
    md.write_text(
        "# x\n## 1. 基础信息\n- 学历层次：硕士\n- 专业：软件工程\n- 所在地：成都\n"
        "- 可接受工作地域：成都/重庆\n\n"
        "## 2. 求职方向与偏好\n- 目标方向（按优先级排序）：\n  1. 数据工程\n"
        "- 工作性质偏好：不限\n- 期望薪资区间：面议\n"
        "- 明确不做的方向：\n  - Java：业务后端\n  - Go：xx\n",
        encoding="utf-8",
    )
    reset_cache()
    p1 = load_profile(path=str(md), use_cache=False)
    assert p1["明确不做"] == ["java", "go"]

    md.write_text(
        md.read_text(encoding="utf-8").replace(
            "- 明确不做的方向：\n  - Java：业务后端\n  - Go：xx\n",
            "- 明确不做的方向：\n  - （示例：暂无）\n",
        ),
        encoding="utf-8",
    )
    reset_cache()
    p2 = load_profile(path=str(md), use_cache=False)
    assert p2["明确不做"] == [], f"删除条目后 list 应当变空，实际：{p2['明确不做']}"


# --------- rag_api /api/match 不再硬绑 DEMO_PROFILE ---------

def test_rag_api_match_does_not_import_demo_profile():
    """静态扫描 rag_api.py：/api/match 路径函数不应再 import DEMO_PROFILE。"""
    src_path = os.path.join(ROOT, "rag_api.py")
    src = open(src_path, "r", encoding="utf-8").read()
    # /api/match 装饰器到下一个 @app 之间的代码块
    m = re.search(r"@app\.post\(\"/api/match\"[\s\S]*?(?=@app\.|\Z)", src)
    assert m, "未找到 /api/match 路由"
    block = m.group(0)
    assert "DEMO_PROFILE" not in block, "/api/match 仍在引用 DEMO_PROFILE"
    assert "load_profile" in block, "/api/match 应改用 profile_loader.load_profile()"


def test_rag_api_match_uses_real_profile_via_loader(monkeypatch):
    """打桩 load_profile，确认 /api/match 真的调到了 profile_loader.load_profile()。"""
    import profile_loader

    sentinel = {
        "学历": "硕士",
        "专业": "软件工程",
        "所在地": "南京",
        "可接受地域": ["南京", "上海"],
        "方向优先级": ["AI 应用开发"],
        "明确不做": ["java"],
        "工作性质偏好": "不限",
        "期望薪资": "面议",
        "熟练技能": ["Python"],
        "会用技能": ["FastAPI"],
        "项目数量": 2,
        "实习数量": 0,
        "英语自评": 2,
        "_source": "test_stub",
    }
    called = {"n": 0}

    def fake_load(*a, **kw):
        called["n"] += 1
        return dict(sentinel)

    monkeypatch.setattr(profile_loader, "load_profile", fake_load)

    from rag_api import app
    client = TestClient(app)
    jd = (
        "岗位名称：AI 应用开发实习生\n公司：示例\n工作地点：南京\n"
        "学历要求：本科及以上\n专业要求：计算机/通信/电子\n经验要求：实习\n"
        "技术要求：Python / Prompt / Agent / RAG\n工作性质：日常实习"
    )
    resp = client.post("/api/match", json={"jd_text": jd})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in {"当前适合投递", "当前暂不建议投递", "信息不足，建议补充后再判断"}
    assert "direction" in body
    assert called["n"] >= 1, "load_profile 未被调用"


# --------- 端到端：公开合成 profile + 合成 JD ---------

def test_run_match_with_synthetic_profile_returns_three_tier_conclusion():
    from match_job import run_match
    from profile_loader import load_profile, reset_cache

    reset_cache()
    profile = load_profile(
        path=os.path.join(ROOT, "tests", "fixtures", "missing-profile.md"),
        use_cache=False,
    )
    jd = (
        "岗位名称：大模型应用开发实习生\n"
        "公司：ExampleCo\n工作地点：重庆\n"
        "学历要求：本科及以上\n"
        "专业要求：计算机、人工智能等相关专业\n"
        "经验要求：每周实习 4 天及以上\n"
        "技术要求：Python / LangGraph / RAG / Embedding / FastAPI\n"
        "工作性质：实习\n"
    )
    report = run_match(profile, jd, jd_title="ExampleCo AI 应用实习")
    assert report.conclusion in {"当前适合投递", "当前暂不建议投递", "信息不足，建议补充后再判断"}
    assert report.direction
    assert isinstance(report.gap_list, dict)
