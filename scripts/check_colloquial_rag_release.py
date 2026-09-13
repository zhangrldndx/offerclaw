#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_colloquial_release import load_and_check  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("artifact")
parser.add_argument("--regression")
args = parser.parse_args()
report = load_and_check(args.artifact, args.regression)
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["decision"] == "GO" else 1)
