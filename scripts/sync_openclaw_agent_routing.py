#!/usr/bin/env python3
"""Idempotently install the managed OfferClaw routing block into AGENTS.md."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


START = "<!-- BEGIN OFFERCLAW MANAGED ROUTING -->"
END = "<!-- END OFFERCLAW MANAGED ROUTING -->"


def sync_routing(template_path: Path, agents_path: Path, launcher: str) -> bool:
    block = template_path.read_text(encoding="utf-8").replace(
        "{{OFFERCLAW_LAUNCHER}}", launcher
    ).strip()
    if block.count(START) != 1 or block.count(END) != 1:
        raise ValueError("routing template must contain one managed marker pair")

    existing = agents_path.read_text(encoding="utf-8") if agents_path.exists() else ""
    if START in existing:
        prefix, remainder = existing.split(START, 1)
        if END not in remainder:
            raise ValueError("AGENTS.md contains an unterminated managed routing block")
        _, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + "\n" + suffix.lstrip("\n")
    else:
        separator = "\n\n" if existing.strip() else ""
        updated = existing.rstrip() + separator + block + "\n"

    if updated == existing:
        return False

    agents_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=agents_path.parent, delete=False
    ) as handle:
        handle.write(updated)
        temp_path = Path(handle.name)
    os.replace(temp_path, agents_path)
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: sync_openclaw_agent_routing.py TEMPLATE AGENTS LAUNCHER",
            file=sys.stderr,
        )
        return 64
    sync_routing(Path(argv[1]), Path(argv[2]), argv[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
