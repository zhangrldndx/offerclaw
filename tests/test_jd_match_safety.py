from jd_parser import analyze_jd
from match_job import run_match, soft_skill_overlap
from resume_critic import keyword_coverage


def _profile(**overrides):
    value = {
        "学历": "专科",
        "专业": "软件工程",
        "项目数量": 0,
        "实习数量": 0,
        "英语自评": 1,
        "可接受地域": ["上海", "远程"],
        "明确不做": [],
        "方向优先级": ["AI 应用开发"],
        "会用技能": ["MongoDB", "storage"],
        "熟练技能": [],
    }
    value.update(overrides)
    return value


def _hard(report, name):
    return next(item for item in report.hard_gate if item.name == name)


def _match(jd, profile=None):
    analysis = analyze_jd(jd, mode="deterministic").model_dump(mode="json")
    return run_match(profile or _profile(), jd, jd_analysis=analysis)


def test_preferred_education_with_explicit_no_restriction_is_not_rejected():
    report = _match("""岗位名称：AI 工程师
任职要求
- 本科优先，但不限学历
""")
    assert _hard(report, "学历").status != "✗"


def test_preferred_experience_with_no_experience_allowed_is_not_rejected():
    report = _match("""岗位名称：AI 工程师
任职要求
- 有相关项目经验者优先，没有相关经验也可
""")
    assert _hard(report, "经验").status != "✗"


def test_language_bonus_without_certificate_requirement_is_not_rejected():
    report = _match("""岗位名称：AI 工程师
任职要求
- 英语能力是加分项，但不要求英语证书
""")
    assert _hard(report, "语言").status != "✗"


def test_explicit_remote_support_overrides_office_city_mismatch():
    report = _match("""岗位名称：AI 工程师
工作地点：北京
岗位说明
- 支持远程办公
""")
    assert _hard(report, "地域").status == "✓"


def test_bare_remote_location_matches_remote_profile_preference():
    report = _match("""岗位名称：AI 工程师
工作地点：远程
任职要求：Python
""")
    assert _hard(report, "地域").status == "✓"


def test_explicit_empty_major_with_colon_is_not_unknown():
    report = _match("""岗位名称：AI 工程师
专业要求：不限
任职要求：Python
""")
    assert _hard(report, "专业").status == "✓"


def test_compound_chinese_exclusion_is_a_hard_conflict():
    report = _match(
        "岗位名称：前端开发实习生\n技术要求：精通 React",
        _profile(**{"明确不做": ["前端"]}),
    )
    assert _hard(report, "技术主线").status == "✗"


def test_short_english_terms_do_not_match_inside_other_words():
    analysis = {
        "keywords": [
            {"canonical_name": "Go", "surface_forms": ["Go"], "importance": 1.0},
            {"canonical_name": "RAG", "surface_forms": ["RAG"], "importance": 1.0},
        ]
    }
    result = soft_skill_overlap(_profile(), "MongoDB storage", analysis)
    assert result.status == "未命中"
    coverage = keyword_coverage("MongoDB storage", analysis["keywords"])
    assert coverage["hit"] == []
    assert coverage["coverage"] == 0.0
