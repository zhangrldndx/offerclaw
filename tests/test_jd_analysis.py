import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from jd_parser import JD_ANALYSIS_SCHEMA_VERSION, analyze_jd, legacy_to_analysis


JD = """岗位名称：AI 应用工程师
公司名称：示例科技
工作地点：深圳
岗位职责
1. 负责 RAG 应用开发
任职要求
1. 熟练掌握 Python 或 Java
2. 不要求工作经验
加分项
1. 熟悉 LangGraph 优先
"""


def _draft(*, ungrounded=False):
    requirement_text = "能够驾驶宇宙飞船" if ungrounded else "熟练掌握 Python 或 Java"
    evidence = requirement_text
    keyword = "QuantumMagic" if ungrounded else "Python"
    return json.dumps({
        "title": "AI 应用工程师", "company": "示例科技", "location": "深圳",
        "job_type": "全职", "responsibilities": ["负责 RAG 应用开发"],
        "requirements": [{
            "text": requirement_text, "kind": "skill", "modality": "alternative",
            "priority": 0.9, "alternatives": ["Python", "Java"],
            "evidence_spans": [evidence],
        }],
        "keywords": [{
            "canonical_name": keyword, "surface_forms": [keyword],
            "category": "technology", "importance": 0.9,
            "requirement_indexes": [0], "evidence_spans": [keyword],
        }],
        "warnings": [],
    }, ensure_ascii=False)


def _caller(output, calls=None):
    def invoke(_messages, _max_tokens, _temperature, _model):
        if calls is not None:
            calls.append(1)
        return output
    return invoke


def test_deterministic_analysis_preserves_modality_and_evidence():
    analysis = analyze_jd(JD, mode="deterministic")
    assert analysis.schema_version == JD_ANALYSIS_SCHEMA_VERSION
    alternative = next(item for item in analysis.requirements if "Python" in item.text)
    no_experience = next(item for item in analysis.requirements if "不要求" in item.text)
    preferred = next(item for item in analysis.requirements if "LangGraph" in item.text)
    assert alternative.modality == "alternative"
    assert set(alternative.alternatives) == {"熟练掌握 Python", "Java"}
    assert no_experience.modality == "context"
    assert preferred.modality == "preferred"
    for requirement in analysis.requirements:
        for span in requirement.evidence_spans:
            assert JD[span.start:span.end] == span.text
    for keyword in analysis.keywords:
        for span in keyword.evidence_spans:
            assert JD[span.start:span.end] == span.text


def test_deterministic_analysis_supports_generic_markdown_headings():
    jd = """# AI 应用开发实习

## 职责
- 参与基于 Python 的 RAG 应用开发。

## 要求
- 熟悉 Python 基础与常见工程实践。
- 理解向量检索和关键词检索的基本原理。
"""
    analysis = analyze_jd(jd, mode="deterministic")
    assert analysis.responsibilities == ["参与基于 Python 的 RAG 应用开发。"]
    assert [item.text for item in analysis.requirements if item.kind != "responsibility"] == [
        "熟悉 Python 基础与常见工程实践。",
        "理解向量检索和关键词检索的基本原理。",
    ]
    assert "deterministic_no_requirement_section" not in analysis.warnings


def test_ungrounded_model_content_is_dropped(tmp_path):
    analysis = analyze_jd(
        JD, mode="intelligent", model="fake", cache_dir=tmp_path,
        caller=_caller(_draft(ungrounded=True)),
    )
    assert analysis.source == "llm"
    assert all("宇宙飞船" not in item.text for item in analysis.requirements)
    assert [item.text for item in analysis.requirements
            if item.kind == "responsibility"] == ["负责 RAG 应用开发"]
    assert analysis.keywords == []
    assert any("dropped_ungrounded_requirement" in warning for warning in analysis.warnings)
    assert any("dropped_ungrounded_keyword" in warning for warning in analysis.warnings)


def test_unrelated_real_span_replaces_hallucinated_requirement_text(tmp_path):
    payload = json.loads(_draft())
    payload["requirements"][0]["text"] = "必须能够驾驶宇宙飞船"
    payload["requirements"][0]["evidence_spans"] = ["熟练掌握 Python 或 Java"]
    analysis = analyze_jd(
        JD, mode="intelligent", model="semantic-grounding", cache_dir=tmp_path,
        caller=_caller(json.dumps(payload, ensure_ascii=False)),
    )
    requirement = next(item for item in analysis.requirements if item.kind == "skill")
    assert requirement.text == "熟练掌握 Python 或 Java"
    assert "宇宙飞船" not in requirement.text
    assert any("normalized_requirement_to_evidence" in value
               for value in analysis.warnings)


def test_ungrounded_header_fields_fall_back_to_source_text(tmp_path):
    payload = json.loads(_draft())
    payload.update({
        "title": "首席宇宙飞船驾驶员", "company": "火星集团",
        "location": "火星", "job_type": "星际派遣",
    })
    analysis = analyze_jd(
        JD, mode="intelligent", model="field-grounding", cache_dir=tmp_path,
        caller=_caller(json.dumps(payload, ensure_ascii=False)),
    )
    assert analysis.title == "AI 应用工程师"
    assert analysis.company == "示例科技"
    assert analysis.location == "深圳"
    assert analysis.job_type != "星际派遣"
    assert sum(value.startswith("dropped_ungrounded_field:")
               for value in analysis.warnings) == 4


def test_cache_avoids_duplicate_llm_call(tmp_path):
    calls = []
    first = analyze_jd(
        JD, mode="intelligent", model="fake", cache_dir=tmp_path,
        caller=_caller(_draft(), calls),
    )
    second = analyze_jd(
        JD, mode="intelligent", model="fake", cache_dir=tmp_path,
        caller=lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert len(calls) == 1
    assert first.source == "llm"
    assert second.source == "cache"
    assert first.input_hash == second.input_hash
    assert first.keyword_names() == second.keyword_names()


def test_concurrent_same_jd_uses_single_flight(tmp_path):
    calls = 0
    guard = threading.Lock()

    def invoke(_messages, _max_tokens, _temperature, _model):
        nonlocal calls
        with guard:
            calls += 1
        time.sleep(0.05)
        return _draft()

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(
            lambda _index: analyze_jd(
                JD, mode="intelligent", model="single-flight",
                cache_dir=tmp_path, caller=invoke,
            ),
            range(5),
        ))
    assert calls == 1
    assert all("Python" in result.keyword_names() for result in results)


def test_invalid_schema_gets_one_repair(tmp_path):
    outputs = iter(["bad json", _draft()])
    calls = []

    def caller(*_args):
        calls.append(1)
        return next(outputs)

    analysis = analyze_jd(
        JD, mode="intelligent", model="repair-model", cache_dir=tmp_path, caller=caller,
    )
    assert len(calls) == 2
    assert analysis.source == "llm"
    assert "Python" in analysis.keyword_names()


def test_legacy_checkpoint_adapter_drops_unverifiable_keyword():
    analysis = legacy_to_analysis(JD, {"keywords": ["Python", "HallucinatedSDK"]})
    assert analysis.source == "legacy"
    assert "Python" in analysis.keyword_names()
    assert "HallucinatedSDK" not in analysis.keyword_names()
    assert any("legacy_keyword_dropped_ungrounded" in value for value in analysis.warnings)
