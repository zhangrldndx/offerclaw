# -*- coding: utf-8 -*-
"""Calibrate a version-bound OfferClaw Evidence Gate from reranker A/B JSON.

This command is intentionally offline-only.  It writes a
``calibration_candidate`` report containing every generated rule.  Direct
promotion to the production registry is disabled until a separate blind/shadow
validation artifact and promotion command exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_gate_profiles import (
    calibrate_gate,
    examples_from_ab_payload,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="reranker-profile-ab-v1 JSON")
    parser.add_argument("--arm", help="candidate arm in a multi-arm A/B JSON")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-output", type=Path,
                        help="disabled: calibration cannot directly write a production registry")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.profile_output:
        raise SystemExit(
            "Direct Gate promotion is disabled: produce a calibration_candidate "
            "report, then validate it on blind/shadow evidence."
        )
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    examples, signature = examples_from_ab_payload(
        payload, arm=args.arm, run_index=args.run_index,
    )
    report = calibrate_gate(
        examples, signature, source=f"{args.input}#{args.arm or 'auto'}",
    )
    _write_json(args.output, report)
    print(
        f"Gate {report['status']}: {report['reason']}; "
        f"candidates={report['candidate_count']}; output={args.output}"
    )


if __name__ == "__main__":
    main()
