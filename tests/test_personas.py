# -*- coding: utf-8 -*-
"""
OfferClaw · 多 Persona 回归测试

用 pytest.mark.parametrize 把 profiles/*.json × 多份 JD 全部跑一遍，
确保 match_job 的规则对不同画像都能稳定输出三档结论。
"""
import glob
import json
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

PROFILES_DIR = os.path.join(BASE, "profiles")


def load_personas():
    files = sorted(glob.glob(os.path.join(PROFILES_DIR, "*.json")))
    out = []
    for f in files:
        name = os.path.basename(f)
        if name.startswith("private_") or name.endswith("_local.json"):
            continue
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        out.append((name, data))
    return out


JD_AI_INTERN = """
岗位名称：AI 应用开发实习生
公司：示例公司
工作地点：上海
学历要求：本科及以上
专业要求：计算机、人工智能、通信、电子等相关专业
经验要求：实习不强制要求年限
技术要求：熟悉 Python；了解 LLM / Prompt / Agent；有项目经验优先
工作性质：驻场
"""

JD_BACKEND = """
岗位名称：Python 后端工程师
公司：示例公司
工作地点：杭州
学历要求：本科及以上
经验要求：1 年以上后端开发经验
技术要求：Python、MySQL、Linux、FastAPI、Redis
工作性质：驻场
"""

JDS = [("ai_intern", JD_AI_INTERN), ("backend", JD_BACKEND)]


@pytest.mark.parametrize("persona_file,persona", load_personas())
@pytest.mark.parametrize("jd_name,jd_text", JDS)
def test_persona_matching_stable(persona_file, persona, jd_name, jd_text):
    """每个 persona × 每份 JD 都能跑出合法三档结论。"""
    from match_job import run_match
    report = run_match(persona, jd_text, jd_title=f"{jd_name}/{persona_file}")
    assert report.conclusion in (
        "当前适合投递",
        "当前暂不建议投递",
        "中长期可转向",
    ), f"非法结论: {report.conclusion}"
    assert isinstance(report.gap_list, dict)
    assert isinstance(report.suggestions, list)


@pytest.mark.parametrize("persona_file,persona", load_personas())
def test_persona_schema(persona_file, persona):
    """persona JSON 必须包含 match_job 需要的全部字段。"""
    required = [
        "学历", "专业", "所在地", "可接受地域", "方向优先级",
        "明确不做", "工作性质偏好", "期望薪资",
        "熟练技能", "会用技能", "项目数量", "实习数量", "英语自评",
    ]
    missing = [k for k in required if k not in persona]
    assert not missing, f"{persona_file} 缺字段: {missing}"
