# -*- coding: utf-8 -*-
"""通用用户兼容扩展契约测试。

这些用例只验证架构底线（能识别、可解释、旧结果不漂移），不声称跨行业
准确率。测试画像全部内联，避免改动现有用户画像与 Persona 数据。
"""
from __future__ import annotations

import json
import os

import pytest

from career_domains import (
    detect_domain,
    detect_role_family,
    generic_direction_alignment,
    load_domain_templates,
    resolve_domain,
)
from job_discovery import extract_jd
from match_job import check_major, judge_direction, soft_direction, soft_programming_language
from match_job import run_match
from job_requirements import analyze_programming_languages


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _profile(direction: str, *, major: str = "工商管理") -> dict:
    return {
        "学历": "本科",
        "专业": major,
        "所在地": "上海",
        "可接受地域": ["上海", "杭州"],
        "方向优先级": [direction],
        "目标岗位类型": "不限",
        "行业偏好": "不限",
        "明确不做": [],
        "工作性质偏好": "不限",
        "期望薪资": "面议",
        "熟练技能": [],
        "会用技能": [],
        "项目数量": 1,
        "实习数量": 0,
        "英语自评": 3,
    }


@pytest.mark.parametrize(
    "target,jd,expected_domain",
    [
        ("产品经理", "岗位名称：高级策略产品经理\n负责用户研究、需求分析和产品迭代", "product"),
        ("用户运营", "岗位名称：社区增长运营\n负责用户分层、活动策划和留存率", "operations"),
        ("UX设计师", "岗位名称：用户体验设计师\n负责交互设计和可用性测试", "design"),
        ("商务拓展", "岗位名称：大客户经理\n负责客户开发、商务谈判与渠道拓展", "marketing_sales"),
        ("财务分析", "岗位名称：经营财务分析\n负责预算管理和财务报表分析", "finance"),
        ("招聘专员", "岗位名称：招聘专员\n负责人才招聘、面试组织和人才盘点", "human_resources"),
    ],
)
def test_common_roles_follow_user_target(target, jd, expected_domain):
    profile = _profile(target)
    result = generic_direction_alignment(profile, jd)
    assert detect_domain(jd) == expected_domain
    assert result["direction"] == "主方向"
    assert result["status"] == "命中"
    assert judge_direction(profile, jd) == "主方向"
    assert soft_direction(profile, jd).status == "命中"


def test_unknown_role_can_match_profile_without_template_entry():
    """模板不要求穷举全行业：用户目标短语直命中仍是第一优先级。"""
    profile = _profile("采购")
    jd = "岗位名称：采购专员\n负责供应商管理、询价和采购合同"
    result = generic_direction_alignment(profile, jd)
    assert result["direction"] == "主方向"
    assert "直接命中" in result["reason"]


def test_llm_role_translation_is_closed_set_and_only_used_after_rule_miss(monkeypatch):
    import career_domains as domains

    unknown = "岗位名称：体验增长伙伴\n负责跨团队推进用户服务策略"
    assert detect_domain(unknown) is None
    monkeypatch.setattr(domains, "_llm_domain_hint", lambda text: "operations")
    assert resolve_domain(unknown, allow_llm=True) == "operations"


def test_related_domain_is_partial_not_main():
    profile = _profile("产品经理")
    jd = "岗位名称：用户运营\n负责活动策划、用户分层和留存率"
    result = generic_direction_alignment(profile, jd)
    assert result["direction"] == "派生方向"
    assert result["status"] == "部分命中"


def test_legacy_ai_keywords_do_not_override_a_general_users_target():
    """AI 是 JD 的属性，不应自动成为所有用户的“主方向”。"""
    profile = _profile("用户运营")
    jd = "岗位名称：大模型应用工程师\n负责 LLM、Agent 与 RAG 系统开发"
    assert judge_direction(profile, jd) == "不考虑"
    direction = soft_direction(profile, jd)
    assert direction.status == "未命中"
    assert "目标方向" in direction.reason


def test_llm_in_backend_jd_does_not_turn_frontend_target_into_ai_main_direction():
    profile = _profile("前端工程师", major="计算机")
    jd = "岗位名称：后端开发工程师\n负责 Java 服务端开发，并接入 LLM API"
    assert detect_role_family(jd) == "backend"
    assert judge_direction(profile, jd) == "派生方向"
    assert soft_direction(profile, jd).status == "部分命中"


def test_ai_application_target_treats_model_training_as_adjacent_family():
    profile = _profile("AI 应用开发", major="计算机")
    jd = "岗位名称：大模型训练平台实习生\n负责预训练、模型微调和分布式训练"
    assert detect_role_family("AI 应用开发") == "llm_application"
    assert detect_role_family(jd) == "model_training"
    assert generic_direction_alignment(profile, jd)["direction"] == "派生方向"


def test_backend_target_treats_data_engineering_as_adjacent_family():
    profile = _profile("Python 后端", major="计算机")
    jd = "岗位名称：数据工程师\n负责 Spark ETL 和数据仓库"
    assert detect_role_family(jd) == "data_engineering"
    assert generic_direction_alignment(profile, jd)["direction"] == "派生方向"


@pytest.mark.parametrize(
    "target,jd,domain",
    [
        ("芯片验证工程师", "岗位名称：数字IC验证工程师\n要求 SystemVerilog、UVM、时序分析", "chip_hardware"),
        ("生物信息工程师", "岗位名称：生物信息工程师\n负责基因组数据分析和医学统计", "biomedical_research"),
    ],
)
def test_non_software_technical_roles_keep_their_own_domain(target, jd, domain):
    profile = _profile(target, major="电子信息")
    assert detect_domain(jd) == domain
    assert judge_direction(profile, jd) == "主方向"


def test_ai_major_whitelist_does_not_leak_into_biomedical_role():
    profile = _profile("生物信息工程师", major="软件工程")
    jd = "岗位名称：生物信息工程师\n专业要求：生物、医学、药学等相关专业"
    result = check_major(profile, jd)
    assert result.status == "?"
    assert "软件工程" in result.reason


def test_general_major_template_only_explains_related_major():
    profile = _profile("产品经理", major="工业设计")
    jd = "岗位名称：产品经理\n专业要求：计算机、管理或其他相关专业\n负责需求分析和产品规划"
    result = check_major(profile, jd)
    assert result.status == "✓"
    assert "产品" in result.reason and "相关专业模板" in result.reason


def test_no_major_requirement_does_not_apply_ai_whitelist_to_general_user():
    profile = _profile("用户运营", major="汉语言文学")
    jd = "岗位名称：内容运营\n负责内容策划和用户增长"
    result = check_major(profile, jd)
    assert result.status == "✓"
    assert "未提出专业限制" in result.reason


def test_job_discovery_extracts_nontechnical_competencies_from_template():
    parsed = extract_jd(
        "岗位名称：用户运营\n工作地点：上海\n岗位职责：负责用户分层、活动策划、留存率复盘\n"
        "任职要求：本科及以上，具备数据分析能力"
    )
    assert {"用户分层", "活动策划", "留存率", "数据分析"}.issubset(
        set(parsed["skills_detected"])
    )


def test_template_contract_is_valid_and_unique():
    domains = load_domain_templates()
    ids = [d["id"] for d in domains]
    assert len(domains) >= 8
    assert len(ids) == len(set(ids))
    assert all(d["title_aliases"] and d["competency_keywords"] for d in domains)


@pytest.mark.parametrize(
    "target,major,skills,jd",
    [
        ("产品经理", "信息管理", ["需求分析", "用户研究"],
         "岗位名称：产品经理\n工作地点：上海\n学历要求：本科及以上\n专业要求：相关专业\n"
         "负责用户研究、需求分析和产品迭代；具备项目经验者优先"),
        ("用户运营", "市场营销", ["活动策划", "数据分析"],
         "岗位名称：用户运营\n工作地点：上海\n学历要求：本科及以上\n专业要求：相关专业\n"
         "负责用户分层、活动策划和留存率复盘；经验不限"),
        ("UI设计师", "视觉传达", ["视觉设计", "Figma"],
         "岗位名称：UI设计师\n工作地点：杭州\n学历要求：本科及以上\n专业要求：相关专业\n"
         "负责视觉设计、设计系统和作品集评审；经验不限"),
    ],
)
def test_full_match_pipeline_accepts_common_role_profiles(target, major, skills, jd):
    profile = _profile(target, major=major)
    profile["熟练技能"] = skills
    report = run_match(profile, jd, jd_title=target)
    assert report.direction == "主方向"
    assert report.conclusion in {"当前适合投递", "中长期可转向", "当前暂不建议投递"}
    assert all(item.reason for item in report.hard_gate + report.soft_dims)
    # 这些 fixture 明确满足学历/专业/地域，通用扩展不能凭空制造硬失败。
    hard = {item.name: item.status for item in report.hard_gate}
    assert hard["学历"] == hard["专业"] == hard["地域"] == "✓"


def test_missing_template_fails_safe_without_breaking_legacy(monkeypatch, tmp_path):
    import career_domains as domains

    domains.clear_domain_caches()
    monkeypatch.setattr(domains, "TEMPLATE_PATH", str(tmp_path / "missing.json"))
    try:
        assert domains.load_domain_templates() == ()
        # 未知通用方向保守返回不考虑，而不是抛异常或臆断。
        result = domains.generic_direction_alignment(_profile("用户运营"), "岗位名称：社区增长运营")
        assert result["direction"] == "不考虑"
    finally:
        domains.clear_domain_caches()


def test_abnormally_long_jd_keeps_front_loaded_domain_result():
    """异常长输入使用有界扫描，标题在前时结果不受尾部垃圾影响。"""
    jd = "岗位名称：用户运营\n负责用户分层和活动策划\n" + ("无关噪声" * 300_000)
    assert detect_domain(jd) == "operations"
    result = generic_direction_alignment(_profile("用户运营"), jd)
    assert result["direction"] == "主方向"


def test_ai_profile_still_matches_ai_application_role_from_its_own_target():
    """AI 画像仍命中 AI 应用岗，但判断依据是画像目标，不只是 JD 热词。"""
    with open(os.path.join(BASE, "profiles", "p1_demo_ai.json"), encoding="utf-8") as f:
        profile = json.load(f)
    jd = (
        "岗位名称：AI 应用开发实习生\n工作地点：上海\n学历要求：本科及以上\n"
        "专业要求：计算机、人工智能、通信、电子等相关专业\n"
        "技术要求：熟悉 Python；了解 LLM / Prompt / Agent"
    )
    direction = soft_direction(profile, jd)
    major = check_major(profile, jd)
    assert judge_direction(profile, jd) == "主方向"
    assert direction.status == "命中"
    assert "profile §2" in direction.reason
    assert major.status == "✓"
    assert "计算机" in major.reason


@pytest.mark.parametrize(
    "jd,policy,hard",
    [
        ("要求：C/C++/Java/Python/Go 等一种以上", "flexible_pool", []),
        ("要求：Python/C++ 至少一门", "flexible_pool", []),
        ("要求：熟练掌握 Python/C++", "flexible_pool", []),
        ("要求：必须熟练掌握 Java", "single_required", ["Java"]),
        ("要求：Java 为主要开发语言，Python 为加分项", "single_required", ["Java"]),
        ("要求：同时精通 Python 和 C++，二者用于核心模块", "multi_required_explicit", ["C++", "Python"]),
        ("团队使用 Java/Python/Go", "context_only", []),
    ],
)
def test_programming_language_policy_uses_explicit_evidence(jd, policy, hard):
    req = analyze_programming_languages(jd)
    assert req.policy == policy
    assert sorted(req.hard_required) == sorted(hard)


def test_explicit_one_of_wording_is_not_marked_ambiguous():
    req = analyze_programming_languages("要求：Java/Python/Go 至少掌握一种")
    assert req.policy == "flexible_pool"
    assert req.ambiguous is False


def test_language_pool_never_requires_every_language():
    profile = _profile("后端工程师", major="计算机")
    profile["会用技能"] = ["Python"]
    result = soft_programming_language(profile, "技术要求：Java、Python、Go，精通一种或多种")
    assert result.status == "命中"
    assert "不要求全部掌握" in result.reason


def test_missing_flexible_language_is_not_a_hard_rejection():
    profile = _profile("后端工程师", major="计算机")
    profile["会用技能"] = ["Ruby"]
    jd = "岗位名称：后端工程师\n技术要求：Java/Python/Go 至少掌握一种"
    result = soft_programming_language(profile, jd)
    assert result.status == "部分命中"
    assert "投递否决" in result.reason


def test_explicit_empty_exclusion_list_is_not_unknown_hard_gate():
    profile = _profile("后端工程师", major="计算机")
    report = run_match(profile, "岗位名称：后端开发\n要求：Python")
    tech = next(x for x in report.hard_gate if x.name == "技术主线")
    assert tech.status == "✓"


def test_single_explicitly_excluded_language_is_a_real_conflict():
    profile = _profile("后端工程师", major="计算机")
    profile["明确不做"] = ["Java"]
    report = run_match(profile, "岗位名称：Java 后端工程师\n要求：必须熟练掌握 Java")
    tech = next(x for x in report.hard_gate if x.name == "技术主线")
    assert tech.status == "✗"


def test_banned_language_inside_flexible_pool_does_not_block_other_options():
    profile = _profile("后端工程师", major="计算机")
    profile["明确不做"] = ["Java"]
    profile["会用技能"] = ["Python"]
    report = run_match(profile, "岗位名称：后端工程师\n要求：Java/Python/Go 至少掌握一种")
    tech = next(x for x in report.hard_gate if x.name == "技术主线")
    lang = next(x for x in report.soft_dims if x.name == "编程语言兼容")
    assert tech.status == "✓"
    assert lang.status == "命中"


def test_match_report_exposes_jd_requirements_and_user_intent():
    profile = _profile("后端工程师", major="计算机")
    profile["目标岗位类型"] = "校招"
    profile["行业偏好"] = "互联网"
    profile["会用技能"] = ["Python"]
    report = run_match(
        profile,
        "岗位名称：后端开发实习生\n要求：Java/Python/Go 至少掌握一种",
    )
    analysis = report.requirement_analysis
    assert analysis["career_domain"] == "software_engineering"
    assert analysis["role_family"] == "backend"
    assert analysis["programming_languages"]["policy"] == "flexible_pool"
    assert analysis["user_targets"] == ["后端工程师"]
    assert analysis["target_job_type"] == "校招"


def test_project_evidence_participates_in_skill_and_project_fit():
    profile = _profile("AI 应用开发", major="计算机")
    profile["技能证据"] = ["Python", "RAG", "FastAPI"]
    profile["项目技能"] = ["RAG", "FastAPI"]
    report = run_match(profile, "岗位名称：AI 应用开发\n要求：Python、RAG、FastAPI")
    dims = {x.name: x for x in report.soft_dims}
    assert dims["技能重叠"].status == "命中"
    assert dims["项目契合"].status == "命中"
