#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键切换 OfferClaw 的 chat provider(bailian / deepseek / gpt)。

背景:主链路(day1_api_starter.get_llm_config → rag_api / plan_gen …)只认
三变量 OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL。换模型测试效果时,
手改 .env.local 容易漏改 RAG_SYNTH_MODEL 或写错 base。本脚本把每个
provider 的连接信息存成 .env.local 里的命名组,切换时只改激活三变量:

    bailian  ← DASHSCOPE_API_KEY(base 固定 dashscope compatible-mode)
    deepseek ← DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
    gpt      ← GPT_PROXY_API_KEY / GPT_PROXY_BASE_URL / GPT_PROXY_MODEL

用法:
    python scripts/switch_llm_provider.py --list
    python scripts/switch_llm_provider.py --provider deepseek [--model deepseek-v4-flash]
    python scripts/switch_llm_provider.py --provider gpt --dry-run

边界(诚实说明):
- 只切"生成"模型;embedding / 知识库(EMBEDDING_* / RAG_COLLECTION_NAME)
  完全不动——换生成模型不影响已建好的向量库。
- 密钥只在 .env.local 内部搬运,任何输出一律 <redacted>(复用
  bailian_model_failover 的脱敏与原子写)。
- LLM_REASONING_EFFORT 随组管理:gpt → medium(历史代理配置),其余置空。
  注意:置空后主链路 get_llm_config 会落回其内置默认 medium——qwen /
  deepseek 实测忽略该参数(2026-08-08 真实调用验证),行为不受影响。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FAILOVER = _ROOT / "scripts" / "bailian_model_failover.py"
_spec = importlib.util.spec_from_file_location("bailian_model_failover", _FAILOVER)
assert _spec and _spec.loader
_fo = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, _fo)
_spec.loader.exec_module(_fo)

DEFAULT_ENV_FILE = ".env.local"


@dataclass(frozen=True)
class Provider:
    name: str
    key_var: str                 # 命名组里的密钥变量
    base_var: str | None         # 命名组里的 base 变量(None = 用 fixed_base)
    fixed_base: str | None
    model_var: str | None        # 命名组里的默认模型变量
    fallback_model: str
    reasoning_effort: str


PROVIDERS: dict[str, Provider] = {
    "bailian": Provider(
        name="bailian", key_var="DASHSCOPE_API_KEY", base_var=None,
        fixed_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_var=None, fallback_model="qwen-plus", reasoning_effort="",
    ),
    "deepseek": Provider(
        name="deepseek", key_var="DEEPSEEK_API_KEY", base_var="DEEPSEEK_BASE_URL",
        fixed_base=None, model_var="DEEPSEEK_MODEL",
        fallback_model="deepseek-chat", reasoning_effort="",
    ),
    "gpt": Provider(
        name="gpt", key_var="GPT_PROXY_API_KEY", base_var="GPT_PROXY_BASE_URL",
        fixed_base=None, model_var="GPT_PROXY_MODEL",
        fallback_model="gpt-5.6-terra", reasoning_effort="medium",
    ),
}


def _normalize_base(base: str) -> str:
    """主链路直拼 f"{base}/chat/completions",所以 base 必须自带版本段。"""
    base = base.rstrip("/")
    if base.endswith("/v1") or "/api/" in base or "compatible-mode" in base:
        return base
    return base + "/v1"


def resolve(provider: Provider, values: dict[str, str], model_override: str | None) -> dict[str, str] | None:
    """从命名组解析出激活三变量;缺密钥返回 None。"""
    key = values.get(provider.key_var, "")
    if not key:
        return None
    base = provider.fixed_base or values.get(provider.base_var or "", "")
    if not base:
        return None
    model = model_override or (values.get(provider.model_var, "") if provider.model_var else "") \
        or provider.fallback_model
    return {
        "OPENAI_API_KEY": key,
        "OPENAI_BASE_URL": _normalize_base(base),
        "LLM_MODEL": model,
        "RAG_SYNTH_MODEL": model,
        "LLM_REASONING_EFFORT": provider.reasoning_effort,
    }


def cmd_list(values: dict[str, str]) -> int:
    active_base = values.get("OPENAI_BASE_URL", "")
    print("provider  就绪  默认模型                base")
    for p in PROVIDERS.values():
        resolved = resolve(p, values, None)
        ready = "✅" if resolved else "❌缺密钥"
        base = resolved["OPENAI_BASE_URL"] if resolved else "-"
        model = resolved["LLM_MODEL"] if resolved else "-"
        mark = " ← 当前生效" if resolved and base == active_base else ""
        print(f"{p.name:<9} {ready:<5} {model:<22} {base}{mark}")
    print(f"\n当前激活: model={values.get('LLM_MODEL', '?')} · base={active_base}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model", help="覆盖该组默认模型")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--list", action="store_true", help="查看各组就绪状态与当前生效配置")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-update-rag-synth", action="store_true",
                        help="RAG 合成模型保持现状(默认跟随切换)")
    args = parser.parse_args(argv)

    env_path = Path(args.env_file)
    if not env_path.exists():
        print(f"[ERR] 找不到 {env_path}(先从 .env.example 复制并填好密钥)")
        return 2
    lines, values = _fo.read_env(env_path)

    if args.list or not args.provider:
        return cmd_list(values)

    provider = PROVIDERS[args.provider]
    updates = resolve(provider, values, args.model)
    if updates is None:
        print(f"[ERR] {provider.name} 组未配置:.env.local 缺 {provider.key_var}"
              + (f" 或 {provider.base_var}" if provider.base_var else ""))
        return 2
    if args.no_update_rag_synth:
        updates.pop("RAG_SYNTH_MODEL")

    new_lines, changes = _fo.update_lines(lines, updates)
    if not changes:
        print(f"[OK] 已是 {provider.name}({updates['LLM_MODEL']}),无需改动")
        return 0
    for change in changes:
        print(("[dry-run] " if args.dry_run else "") + change.display())
    if not args.dry_run:
        _fo.atomic_write(env_path, new_lines)
        env_path.chmod(0o600)
        print(f"[OK] 已切换到 {provider.name} · model={updates['LLM_MODEL']}"
              f"(embedding/知识库不受影响;重启 API 进程后生效)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
