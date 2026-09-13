#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Switch OfferClaw Bailian model settings in .env.local.

The script deliberately does not scrape Bailian. A Codex/Chrome automation
should inspect the logged-in Bailian page, choose available models, and then
call this script with explicit model names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable


DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ENV_FILE = ".env.local"
SECRET_KEY_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def slug(value: str) -> str:
    """Return the same collection-safe slug shape used by rag_tools.py."""
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return re.sub(r"_+", "_", raw).strip("_")


def collection_name(provider: str, model: str, dimensions: str | int | None) -> str:
    suffix = f"_{dimensions}" if dimensions else ""
    return f"offerclaw_{slug(provider)}_{slug(model)}{suffix}"


def read_env(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.exists():
        return [], {}

    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return lines, values


def redacted(key: str, value: str) -> str:
    if any(marker in key.upper() for marker in SECRET_KEY_MARKERS):
        return "<redacted>"
    return value


@dataclass(frozen=True)
class EnvEdit:
    key: str
    old: str | None
    new: str

    def display(self) -> str:
        old = "<unset>" if self.old is None else redacted(self.key, self.old)
        new = redacted(self.key, self.new)
        return f"{self.key}: {old} -> {new}"


def update_lines(lines: list[str], updates: dict[str, str]) -> tuple[list[str], list[EnvEdit]]:
    remaining = dict(updates)
    edited: list[str] = []
    changes: list[EnvEdit] = []

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            edited.append(raw)
            continue

        key, current = raw.split("=", 1)
        key = key.strip()
        if key not in remaining:
            edited.append(raw)
            continue

        new_value = remaining.pop(key)
        old_value = current.strip().strip('"').strip("'")
        if old_value != new_value:
            changes.append(EnvEdit(key=key, old=old_value, new=new_value))
        edited.append(f"{key}={new_value}")

    if remaining:
        if edited and edited[-1].strip():
            edited.append("")
        edited.append("# Updated by scripts/bailian_model_failover.py")
        for key, value in remaining.items():
            changes.append(EnvEdit(key=key, old=None, new=value))
            edited.append(f"{key}={value}")

    return edited, changes


def atomic_write(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def build_updates(args: argparse.Namespace, current: dict[str, str]) -> dict[str, str]:
    updates: dict[str, str] = {}

    if args.llm_model:
        updates["OPENAI_BASE_URL"] = args.openai_base_url
        updates["LLM_MODEL"] = args.llm_model
        if not args.no_update_rag_synth:
            updates["RAG_SYNTH_MODEL"] = args.rag_synth_model or args.llm_model

    if args.embedding_model:
        provider = args.embedding_provider
        dimensions = args.embedding_dimensions or current.get("EMBEDDING_DIMENSIONS") or "1024"
        updates["EMBEDDING_PROVIDER"] = provider
        updates["EMBEDDING_MODEL"] = args.embedding_model
        updates["EMBEDDING_DIMENSIONS"] = str(dimensions)
        updates["RAG_COLLECTION_NAME"] = collection_name(provider, args.embedding_model, dimensions)

    return updates


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update OfferClaw .env.local after choosing replacement Bailian models.",
    )
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--llm-model", help="Replacement Bailian chat model, e.g. qwen-turbo.")
    parser.add_argument(
        "--rag-synth-model",
        help="Optional RAG synthesis model. Defaults to --llm-model when an LLM is changed.",
    )
    parser.add_argument(
        "--no-update-rag-synth",
        action="store_true",
        help="Keep existing RAG_SYNTH_MODEL untouched.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=DASHSCOPE_OPENAI_BASE_URL,
        help="OpenAI-compatible base URL for Bailian/DashScope.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Replacement Bailian embedding model, e.g. text-embedding-v4.",
    )
    parser.add_argument("--embedding-dimensions", help="Embedding output dimension.")
    parser.add_argument("--embedding-provider", default="bailian")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    env_path = Path(args.env_file)
    lines, current = read_env(env_path)
    updates = build_updates(args, current)

    if not updates:
        print("No model updates requested. Pass --llm-model and/or --embedding-model.")
        return 2

    edited, changes = update_lines(lines, updates)
    if not changes:
        print(f"No changes needed in {env_path}.")
        return 0

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {env_path}:")
    for change in changes:
        print(f"  - {change.display()}")

    if not args.dry_run:
        atomic_write(env_path, edited)

    if args.embedding_model:
        print(
            "Embedding model changed or confirmed. If the collection is new/empty, "
            "run: python rag_ingest.py --rebuild"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
