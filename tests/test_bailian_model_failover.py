from pathlib import Path
import importlib.util
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bailian_model_failover.py"
SPEC = importlib.util.spec_from_file_location("bailian_model_failover", SCRIPT_PATH)
bailian_model_failover = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = bailian_model_failover
SPEC.loader.exec_module(bailian_model_failover)


def test_collection_name_is_provider_model_and_dimension_scoped():
    assert (
        bailian_model_failover.collection_name("bailian", "text-embedding-v4", "1024")
        == "offerclaw_bailian_text_embedding_v4_1024"
    )


def test_updates_env_without_touching_api_keys(tmp_path, capsys):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-keep-me",
                "OPENAI_BASE_URL=https://old.example/v1",
                "LLM_MODEL=qwen-old",
                "DASHSCOPE_API_KEY=sk-dashscope",
                "EMBEDDING_PROVIDER=bailian",
                "EMBEDDING_MODEL=text-embedding-old",
                "EMBEDDING_DIMENSIONS=768",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = bailian_model_failover.main(
        [
            "--env-file",
            str(env_file),
            "--llm-model",
            "qwen-turbo",
            "--embedding-model",
            "text-embedding-v4",
            "--embedding-dimensions",
            "1024",
        ]
    )

    assert exit_code == 0
    content = env_file.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-keep-me" in content
    assert "DASHSCOPE_API_KEY=sk-dashscope" in content
    assert "LLM_MODEL=qwen-turbo" in content
    assert "RAG_SYNTH_MODEL=qwen-turbo" in content
    assert "EMBEDDING_MODEL=text-embedding-v4" in content
    assert "RAG_COLLECTION_NAME=offerclaw_bailian_text_embedding_v4_1024" in content

    output = capsys.readouterr().out
    assert "sk-keep-me" not in output
    assert "sk-dashscope" not in output


def test_dry_run_does_not_write(tmp_path):
    env_file = tmp_path / ".env.local"
    original = "LLM_MODEL=qwen-old\n"
    env_file.write_text(original, encoding="utf-8")

    exit_code = bailian_model_failover.main(
        ["--env-file", str(env_file), "--llm-model", "qwen-new", "--dry-run"]
    )

    assert exit_code == 0
    assert env_file.read_text(encoding="utf-8") == original
