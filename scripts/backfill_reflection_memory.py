#!/usr/bin/env python3
"""盘点或幂等重建长期复盘派生索引；默认 dry-run，不改事实文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reflection_memory import inventory_report, write_derived_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="写入可重建的 logs/reflection/index.json")
    args = parser.parse_args()
    result = write_derived_index() if args.apply else inventory_report()
    if args.apply:
        result = {"status": "written", "stats": result["stats"]}
    else:
        result = {"status": "dry-run", **result}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
