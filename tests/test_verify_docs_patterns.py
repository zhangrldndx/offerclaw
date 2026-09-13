# -*- coding: utf-8 -*-
"""verify_docs 的指标正则自身的回归测试。

这个门禁此前**没有任何测试**，于是它有一个长期盲区没人发现：pytest 的四条
正则位数写死成 ``\\d{2,3}``，测试数破千之后门禁再也看不见 pytest 漂移——
2026-09-02 实测有 10 处 ``1,430`` 静静躺在 canonical 文档里而扫描报全绿。

盲区不止位数，还有三种真实写法：
  * 千分位 ``1,430``——任何纯 ``\\d`` 正则都匹配不上；
  * shields.io 徽章 ``tests-1430%20passed``——空格是 URL 编码的；
  * README「项目数据」单元格标签是「测试通过」而不是「pytest」。

因此本文件按**真实文档里出现过的写法**逐条钉死，而不是造几个理想字符串。
"""

from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_docs  # noqa: E402


def _hits(metric: str, line: str) -> set[str]:
    """按 verify_docs 的真实口径抽值：捕获后去千分位逗号。"""
    out = set()
    for pat in verify_docs.METRIC_PATTERNS[metric]:
        for m in re.finditer(pat, line, re.IGNORECASE):
            out.add(m.group(1).replace(",", ""))
    return out


# 全部取自 2026-09-02 修复前真实漏扫的行
REAL_PYTEST_LINES = [
    '<img src="https://img.shields.io/badge/tests-1430%20passed-success?style=flat">',
    '  <td align="center"><strong>1,430</strong><br><sub>测试通过</sub></td>',
    "**47 个专项故障注入测试 + 1,430 全量测试通过 + CI 门禁**",
    "python -m pytest tests/ -q        # 1,430 tests passed / 4 skipped",
    "| **全量测试** | **1,430 tests passed** (+4 skipped) |",
    "├─ tests/                # 1,430 tests passed（含故障注入测试）",
    "| 自动化测试 | ✅ **1,430 tests passed / 4 skipped** |",
]


def test_four_digit_and_thousands_separated_pytest_counts_are_visible():
    for line in REAL_PYTEST_LINES:
        assert "1430" in _hits("pytest", line), f"漏扫: {line}"


def test_three_digit_pytest_counts_still_match():
    """放宽位数不能把原来抓得到的短数字放跑了。"""
    assert "734" in _hits("pytest", "全量 734 passed")
    assert "605" in _hits("pytest", "pytest 605 通过")


def test_thousands_separator_is_normalised_before_blacklist_compare():
    """``1,430`` 与 ``1430`` 必须归一到同一个键，否则黑名单永远命不中。"""
    assert _hits("pytest", "1,430 tests passed") == {"1430"}


def test_chunks_pattern_survives_five_digit_corpora():
    """语料规模实验做到过 10 万块；chunks 上限卡在 4 位会重演同一个盲区。"""
    assert "10000" in _hits("chunks", "合成语料 10000 chunks")
    assert "3348" in _hits("chunks", "知识库 3348 chunks")


def test_gate_has_no_stale_value_for_its_own_current_metrics():
    """current 里的值不得同时出现在自己的 stale_blacklist 里（自相矛盾）。"""
    m = verify_docs.load_metrics()
    for key, cur in m["current"].items():
        bad = m["stale_blacklist"].get(key)
        if not bad:
            continue
        assert str(cur) not in bad, f"{key}: 当前值 {cur} 同时被列为旧值"


def test_privacy_gate_rejects_literal_network_services_but_allows_loopback():
    text = "\n".join([
        "internal=http://192.168." + "11.1:8080/v1",
        "public=https://203.0." + "113.10:9443/v1",
        "local=http://127.0.0.1:8080/v1",
        "bind=http://0.0.0.0:8000",
    ])
    assert verify_docs._literal_ip_endpoint_lines(text) == [1, 2]


def test_privacy_gate_rejects_local_identity_paths():
    text = "\n".join([
        "C:" + "\\Users\\private-user\\project",
        "/mnt/c/Users/" + "private-user/project",
        "/Users/" + "private-user/project",
        "dev@" + "workstation.local",
        "C:\\Users\\<user>\\project",
        "/Users/<user>/project",
        "/home/user/project",
    ])
    assert [
        i for i, line in enumerate(text.splitlines(), 1)
        if verify_docs._LOCAL_IDENTITY.search(line)
    ] == [1, 2, 3, 4]


def test_privacy_gate_rejects_private_configured_endpoints():
    text = "\n".join([
        "OPENAI_BASE_URL=https://" + "private-gateway.internal/v1",
        "API_BASE=https://api.openai.com/v1",
        "GPT_PROXY_BASE_URL=https://gateway.example.com/v1",
        "OPENAI_BASE_URL=http://127.0.0.1:8080/v1",
    ])
    assert verify_docs._unapproved_endpoint_lines(text) == [1]

    operational = "\n".join([
        "credential=" + "sk-" + "***masked; len=" + "35",
        "状态：当前本机 OpenAI 兼容" + "代理可用",
        "状态：已完成扫码和" + "本人私聊绑定",
        "验收：投递" + "共 10 条",
        "public docs describe configuration without deployment results",
    ])
    assert verify_docs._private_operational_metadata_lines(operational) == [1, 2, 3, 4]


def test_profile_fact_normalization_ignores_markdown_and_json_formatting():
    markdown = "- 学校：示例私立学院"
    json_line = '"学校": "示例私立学院",'
    assert verify_docs._normalize_profile_fact(markdown) == (
        verify_docs._normalize_profile_fact(json_line)
    )


def test_private_path_gate_cannot_be_bypassed_with_force_add():
    assert verify_docs._is_private_path(".env.local")
    assert verify_docs._is_private_path("profiles/private_user.json")
    assert verify_docs._is_private_path("profiles/candidate_local.json")
    assert verify_docs._is_private_path("logs/query.jsonl")
    assert verify_docs._is_private_path("knowledge_base/source.md")
    assert verify_docs._is_private_path("docs/rag_eval/run.json")
    assert not verify_docs._is_private_path(".env.example")
    assert not verify_docs._is_private_path("docs/rag_eval/README.md")
    assert not verify_docs._is_private_path("knowledge_base/README.md")
    assert not verify_docs._is_private_path("profiles/p1_demo_ai.json")


def test_profile_gate_recognizes_json_identity_fields():
    desc = verify_docs._PROFILE_PRIVATE_LABEL.search(
        '"desc": "synthetic private profile description"'
    )
    direction = verify_docs._PROFILE_PRIVATE_LABEL.search(
        '"方向优先级": ["示例方向"]'
    )
    assert desc and desc.group("label") == "desc"
    assert direction and direction.group("label") == "方向优先级"


def test_current_tracked_tree_and_deleted_history_are_scanned(tmp_path, monkeypatch):
    assert verify_docs.scan_repository_privacy() == []
    assert verify_docs.scan_git_history_privacy(["HEAD"]) == []

    repo = tmp_path / "history-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Privacy Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "privacy@example.com"], cwd=repo, check=True)
    leaked = repo / "notes.md"
    leaked.write_text("API_BASE=https://" + "private-gateway.internal/v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add historical fixture"], cwd=repo, check=True)
    leaked.write_text("public documentation only\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "remove historical fixture"], cwd=repo, check=True)

    monkeypatch.setattr(verify_docs, "ROOT", repo)
    assert verify_docs.scan_repository_privacy() == []
    history_hits = verify_docs.scan_git_history_privacy(["HEAD"])
    assert any(hit["kind"] == "unapproved_service_endpoint" for hit in history_hits)
