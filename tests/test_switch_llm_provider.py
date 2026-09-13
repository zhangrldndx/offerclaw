# -*- coding: utf-8 -*-
"""scripts/switch_llm_provider.py 单元测试(零网络、零真密钥)。

守住四条:切换只改激活三变量+RAG_SYNTH+effort;命名组/embedding 原样;
密钥任何输出必须 <redacted>;缺组报错不写文件。
"""
from pathlib import Path
import importlib.util
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "switch_llm_provider.py"
SPEC = importlib.util.spec_from_file_location("switch_llm_provider", SCRIPT_PATH)
switch_llm_provider = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = switch_llm_provider
SPEC.loader.exec_module(switch_llm_provider)


def _write_env(tmp_path: Path, extra: list[str] = ()) -> Path:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-bailian-active",
                "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "LLM_MODEL=qwen-plus",
                "RAG_SYNTH_MODEL=qwen-plus",
                "LLM_REASONING_EFFORT=",
                "DASHSCOPE_API_KEY=sk-bailian-active",
                "EMBEDDING_PROVIDER=local",
                "EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5",
                "RAG_COLLECTION_NAME=offerclaw_local_bge_base_zh_768",
                "DEEPSEEK_API_KEY=sk-deepseek-secret",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1",
                "DEEPSEEK_MODEL=deepseek-v4-pro",
                *extra,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return env_file


def _values(env_file: Path) -> dict[str, str]:
    _, values = switch_llm_provider._fo.read_env(env_file)
    return values


def test_switch_to_deepseek_updates_active_triplet_only(tmp_path):
    env_file = _write_env(tmp_path)
    assert switch_llm_provider.main(["--env-file", str(env_file), "--provider", "deepseek"]) == 0
    v = _values(env_file)
    assert v["OPENAI_API_KEY"] == "sk-deepseek-secret"
    assert v["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"
    assert v["LLM_MODEL"] == "deepseek-v4-pro"
    assert v["RAG_SYNTH_MODEL"] == "deepseek-v4-pro"
    # 命名组与 embedding/知识库一个字不动
    assert v["DEEPSEEK_API_KEY"] == "sk-deepseek-secret"
    assert v["DASHSCOPE_API_KEY"] == "sk-bailian-active"
    assert v["EMBEDDING_MODEL"] == "BAAI/bge-base-zh-v1.5"
    assert v["RAG_COLLECTION_NAME"] == "offerclaw_local_bge_base_zh_768"


def test_switch_to_gpt_uses_proxy_group_and_reasoning_effort(tmp_path):
    env_file = _write_env(tmp_path, extra=[
        "GPT_PROXY_API_KEY=sk-proxy-secret",
        "GPT_PROXY_BASE_URL=http://127.0.0.1:8080/v1",
        "GPT_PROXY_MODEL=gpt-5.6-terra",
    ])
    assert switch_llm_provider.main(["--env-file", str(env_file), "--provider", "gpt"]) == 0
    v = _values(env_file)
    assert v["OPENAI_API_KEY"] == "sk-proxy-secret"
    assert v["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert v["LLM_MODEL"] == "gpt-5.6-terra"
    assert v["LLM_REASONING_EFFORT"] == "medium"


def test_gpt_provider_fallback_model_is_terra():
    provider = switch_llm_provider.PROVIDERS["gpt"]
    assert provider.fallback_model == "gpt-5.6-terra"


def test_switch_back_to_bailian_roundtrip(tmp_path):
    env_file = _write_env(tmp_path)
    switch_llm_provider.main(["--env-file", str(env_file), "--provider", "deepseek"])
    switch_llm_provider.main(["--env-file", str(env_file), "--provider", "bailian"])
    v = _values(env_file)
    assert v["OPENAI_API_KEY"] == "sk-bailian-active"
    assert v["LLM_MODEL"] == "qwen-plus"
    assert v["LLM_REASONING_EFFORT"] == ""


def test_model_override_and_base_normalization(tmp_path):
    env_file = _write_env(tmp_path)
    # 组里 base 故意不带 /v1 → 主链路直拼需要脚本补上
    text = env_file.read_text(encoding="utf-8").replace(
        "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1",
        "DEEPSEEK_BASE_URL=https://api.deepseek.com")
    env_file.write_text(text, encoding="utf-8")
    switch_llm_provider.main(
        ["--env-file", str(env_file), "--provider", "deepseek", "--model", "deepseek-v4-flash"])
    v = _values(env_file)
    assert v["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"
    assert v["LLM_MODEL"] == "deepseek-v4-flash"


def test_missing_group_errors_and_writes_nothing(tmp_path):
    env_file = _write_env(tmp_path)  # 无 GPT_PROXY_*
    before = env_file.read_text(encoding="utf-8")
    assert switch_llm_provider.main(["--env-file", str(env_file), "--provider", "gpt"]) == 2
    assert env_file.read_text(encoding="utf-8") == before


def test_secrets_never_printed(tmp_path, capsys):
    env_file = _write_env(tmp_path)
    switch_llm_provider.main(["--env-file", str(env_file), "--provider", "deepseek"])
    switch_llm_provider.main(["--env-file", str(env_file), "--list"])
    out = capsys.readouterr().out
    assert "sk-deepseek-secret" not in out
    assert "sk-bailian-active" not in out
    assert "<redacted>" in out  # key 变更行确实走了脱敏


def test_dry_run_writes_nothing(tmp_path):
    env_file = _write_env(tmp_path)
    before = env_file.read_text(encoding="utf-8")
    assert switch_llm_provider.main(
        ["--env-file", str(env_file), "--provider", "deepseek", "--dry-run"]) == 0
    assert env_file.read_text(encoding="utf-8") == before


def test_list_reports_readiness(tmp_path, capsys):
    env_file = _write_env(tmp_path)
    assert switch_llm_provider.main(["--env-file", str(env_file), "--list"]) == 0
    out = capsys.readouterr().out
    assert "bailian" in out and "deepseek" in out and "gpt" in out
    assert "❌缺密钥" in out  # gpt 组未配 → 如实标注
    assert "当前生效" in out  # bailian 是激活组
