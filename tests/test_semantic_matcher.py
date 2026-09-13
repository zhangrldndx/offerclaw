import json

from jd_parser import analyze_jd
from match_job import run_match
from semantic_matcher import (
    SemanticMatchAssessment, align_requirements, build_profile_evidence,
)


JD = """岗位名称：AI 应用工程师
工作地点：上海
任职要求
- 本科及以上学历
- 能够建立检索质量评估体系并定位召回问题
"""


def _analysis(jd=JD):
    return analyze_jd(jd, mode="deterministic").model_dump(mode="json")


def _profile(**overrides):
    value = {
        "学历": "硕士", "专业": "计算机科学", "所在地": "上海",
        "可接受地域": ["上海"], "方向优先级": ["AI 应用工程"],
        "明确不做": [], "工作性质偏好": "不限", "期望薪资": "面议",
        "熟练技能": [], "会用技能": [], "技能证据": [], "项目技能": [],
        "项目数量": 1, "实习数量": 0, "英语自评": 2,
        "_evidence_catalog": [{
            "evidence_id": "ev_project_eval",
            "evidence_type": "project",
            "source_ref": "user_profile.md#section-4-project-1",
            "text": "构建 100 题离线评测集，使用 Recall@K 和 MRR 分析检索失败并定位召回链路问题。",
        }],
    }
    value.update(overrides)
    return value


def _semantic_requirement(analysis):
    return next(item for item in analysis["requirements"] if item["kind"] != "hard")


def _caller(payload):
    raw = json.dumps(payload, ensure_ascii=False)

    def invoke(_messages, _max_tokens, _temperature, _model):
        return raw

    return invoke


def test_paraphrased_project_evidence_can_support_a_jd_requirement(tmp_path):
    analysis = _analysis()
    requirement = _semantic_requirement(analysis)
    assessment = align_requirements(
        _profile(), analysis, model="fake", cache_dir=tmp_path,
        caller=_caller({"alignments": [{
            "requirement_id": requirement["requirement_id"],
            "relation": "direct", "evidence_ids": ["ev_project_eval"],
            "rationale": "离线指标分析与召回故障定位直接证明检索质量评估能力",
        }]}),
    )
    assert assessment.status == "completed"
    assert assessment.alignments[0].relation == "direct"
    report = run_match(
        _profile(), JD, jd_analysis=analysis,
        semantic_alignment=assessment.model_dump(mode="json"),
    )
    dims = {item.name: item for item in report.soft_dims}
    assert dims["技能重叠"].status == "命中"
    assert dims["项目契合"].status == "命中"
    assert "section-4-project-1" in dims["项目契合"].reason


def test_unknown_evidence_reference_is_rejected_and_cannot_raise_coverage(tmp_path):
    analysis = _analysis()
    requirement = _semantic_requirement(analysis)
    assessment = align_requirements(
        _profile(), analysis, model="fake", cache_dir=tmp_path,
        caller=_caller({"alignments": [{
            "requirement_id": requirement["requirement_id"],
            "relation": "direct", "evidence_ids": ["ev_hallucinated"],
            "rationale": "不存在的证据",
        }]}),
    )
    assert assessment.status == "degraded"
    assert assessment.alignments[0].relation == "unsupported"
    assert assessment.alignments[0].evidence_ids == []
    assert any(value.startswith("unknown_evidence_ref") for value in assessment.warnings)


def test_skill_name_alone_cannot_prove_experience(tmp_path):
    jd = """岗位名称：AI 工程师
任职要求
- 具备复杂检索系统落地经验
"""
    analysis = _analysis(jd)
    requirement = _semantic_requirement(analysis)
    profile = _profile(
        项目数量=0,
        _evidence_catalog=[{
            "evidence_id": "ev_skill_rag", "evidence_type": "declared_skill",
            "source_ref": "profile:会用技能:0", "text": "会用 RAG",
        }],
    )
    assessment = align_requirements(
        profile, analysis, model="fake", cache_dir=tmp_path,
        caller=_caller({"alignments": [{
            "requirement_id": requirement["requirement_id"],
            "relation": "direct", "evidence_ids": ["ev_skill_rag"],
            "rationale": "技能相关",
        }]}),
    )
    assert assessment.alignments[0].relation == "partial"
    assert any(value.startswith("experience_relation_downgraded")
               for value in assessment.warnings)


def test_semantic_alignment_never_overrides_a_hard_gate(tmp_path):
    analysis = _analysis()
    requirement = _semantic_requirement(analysis)
    profile = _profile(学历="专科")
    assessment = align_requirements(
        profile, analysis, model="fake", cache_dir=tmp_path,
        caller=_caller({"alignments": [{
            "requirement_id": requirement["requirement_id"],
            "relation": "direct", "evidence_ids": ["ev_project_eval"],
            "rationale": "项目能力匹配",
        }]}),
    )
    report = run_match(
        profile, JD, jd_analysis=analysis,
        semantic_alignment=assessment.model_dump(mode="json"),
    )
    education = next(item for item in report.hard_gate if item.name == "学历")
    assert education.status == "✗"
    assert report.conclusion == "当前暂不建议投递"


def test_preferred_match_cannot_hide_an_unsupported_required_skill(tmp_path):
    jd = """岗位名称：AI 工程师
任职要求
- 精通分布式训练
加分项
- 有检索系统评测经验优先
"""
    analysis = _analysis(jd)
    required = next(item for item in analysis["requirements"]
                    if item["modality"] == "required")
    preferred = next(item for item in analysis["requirements"]
                     if item["modality"] == "preferred")
    assessment = align_requirements(
        _profile(), analysis, model="fake", cache_dir=tmp_path,
        caller=_caller({"alignments": [
            {"requirement_id": required["requirement_id"],
             "relation": "unsupported", "evidence_ids": [],
             "rationale": "没有分布式训练证据"},
            {"requirement_id": preferred["requirement_id"],
             "relation": "direct", "evidence_ids": ["ev_project_eval"],
             "rationale": "项目包含检索评测"},
        ]}),
    )
    report = run_match(
        _profile(), jd, jd_analysis=analysis,
        semantic_alignment=assessment.model_dump(mode="json"),
    )
    skill = next(item for item in report.soft_dims if item.name == "技能重叠")
    assert skill.status == "部分命中"
    assert "必选 1 项中直接/可迁移证据 0 项" in skill.reason


def test_model_cannot_add_a_final_verdict_to_alignment_schema(tmp_path):
    analysis = _analysis()
    payload = {"alignments": [], "verdict": "当前适合投递"}
    assessment = align_requirements(
        _profile(), analysis, model="fake", cache_dir=tmp_path,
        caller=_caller(payload),
    )
    assert assessment.status == "degraded"
    assert assessment.alignments == []


def test_disabled_alignment_is_explicit_and_does_not_call_model():
    assessment = align_requirements(
        _profile(), _analysis(), enabled=False,
        caller=lambda *_: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    assert assessment.status == "not_requested"
    assert assessment.source == "deterministic"


def test_profile_evidence_does_not_promote_counts_or_future_plans():
    evidence = build_profile_evidence({
        "项目数量": 5,
        "会用技能": ["Python"],
        "_evidence_catalog": [],
    })
    assert [item.text for item in evidence] == ["Python"]


def test_completed_project_is_not_dropped_when_it_describes_a_problem():
    from profile_loader import parse_evidence_catalog

    markdown = """## 4. 项目经历
- 项目 1
  - 名称：检索诊断
  - 做了什么：定位召回性能不足并修复索引配置
  - 后续计划：【待补充】
"""
    rows = parse_evidence_catalog(markdown)
    assert len(rows) == 1
    assert "性能不足" in rows[0]["text"]
    assert "待补充" not in rows[0]["text"]


def test_match_api_exposes_whether_semantic_alignment_ran(monkeypatch):
    import jd_parser
    import semantic_matcher
    from fastapi.testclient import TestClient
    from rag_api import app

    analysis = analyze_jd(JD, mode="deterministic")
    assessment = SemanticMatchAssessment(
        input_hash="i", evidence_hash="e", source="llm", status="completed",
        alignments=[],
    )
    monkeypatch.setattr(jd_parser, "analyze_jd", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(
        semantic_matcher, "align_requirements",
        lambda *_args, **_kwargs: assessment,
    )
    response = TestClient(app).post(
        "/api/match", json={"jd_text": JD, "use_semantic": True},
    )
    assert response.status_code == 200
    assert response.json()["matching_mode"] == "semantic"
    assert response.json()["semantic_status"] == "completed"
