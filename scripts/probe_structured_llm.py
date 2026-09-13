#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect or explicitly probe GPT JSON-object support.

Without ``--execute`` this command is guaranteed not to make a network call.
The result contains only public provider metadata and never prints a key or raw
model output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structured_llm import probe_json_object_capability  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe response_format=json_object support (dry by default)."
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="explicitly send one bounded probe and cache the public result",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args(argv)
    result = probe_json_object_capability(
        execute=args.execute, timeout_seconds=args.timeout
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    return 0 if result.get("status") in {"verified", "unsupported"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
