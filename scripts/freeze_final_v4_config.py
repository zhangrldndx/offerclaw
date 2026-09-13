#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the three Final v4 arms before any label exists.

Final v4 exists to convert two already-validated candidates into defaults:

  A  production default today: pool28 + compact32 + judge(v4 prompt, d12, early exit)
  B  A with the v5 judge prompt (no-referent rule) -- prompt-flip candidate
  C  quality package: A + HyDE dense+lexical channels + 3-vote consensus gate
     + v5 prompt -- quality-default candidate

All three share one freeze because the comparison is the object being frozen:
code digests, judge prompts (both shas), index fingerprint, and the exact env
that defines each arm.  ``--verify`` fails loudly on drift, as before.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FREEZE_PATH = ROOT / "docs" / "rag_eval" / "final_v4" / "FROZEN_CONFIG.json"

CODE_FILES = (
    "rag_answerability.py",
    "rag_gate.py",
    "rag_colloquial_profiles.py",
    "rag_candidate_pool.py",
    "rag_retrieval_trace.py",
    "rag_hyde.py",
    "eval_colloquial_rag.py",
)

ARMS = {
    "A": {"arm": "compact32_pool28", "env": {}},
    "B": {"arm": "compact32_pool28", "env": {"RAG_ANSWERABILITY_PROMPT": "v5"}},
    "C": {"arm": "compact32_pool28_hydechan_bm25",
          "env": {"RAG_ANSWERABILITY_PROMPT": "v5",
                  "RAG_ANSWERABILITY_GATE": "1",
                  "RAG_ANSWERABILITY_GATE_VOTES": "3"}},
}
# Defined by what is off as much as what is on (defaults are now judge-on).
COMMON_OFF = ("RAG_ANSWERABILITY_TIEBREAK", "RAG_ANSWERABILITY_MODE",
              "RAG_HYDE", "RAG_QUERY_REWRITE", "RAG_DOC2QUERY",
              "RAG_EN_QUOTA", "RAG_EN_GATE_MIN", "RAG_RERANK_EN_ONNX_DIR")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    from rag_answerability import (ANSWERABILITY_DEPTH, MIN_GRADE_TO_ACT,
                                   PROMPT_SHA256, PROMPT_V5_SHA256, SCHEMA,
                                   resolve_model)
    from rag_colloquial_profiles import colloquial_profile
    from rag_tools import get_collection_name, index_fingerprint

    fingerprint = index_fingerprint()
    return {
        "schema_version": "final-v4-frozen-config-v1",
        "arms": {name: {"arm": spec["arm"], "env": dict(spec["env"]),
                        "profile": colloquial_profile(spec["arm"]).to_dict()}
                 for name, spec in ARMS.items()},
        "common_off": list(COMMON_OFF),
        "judge": {
            "model": resolve_model(),
            "schema": SCHEMA,
            "prompt_v4_sha256": PROMPT_SHA256,
            "prompt_v5_sha256": PROMPT_V5_SHA256,
            "min_grade_to_act": MIN_GRADE_TO_ACT,
            "default_depth": ANSWERABILITY_DEPTH,
        },
        "index": {
            "collection": get_collection_name(),
            "count": fingerprint.get("collection_count"),
            "content_hash": fingerprint.get("collection_content_hash"),
            "embedding_model": fingerprint.get("embedding_model"),
            "chunker_version": fingerprint.get("chunker_version"),
        },
        "code": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT,
                capture_output=True, text=True).stdout.strip(),
            "file_sha256": {name: _sha256_file(ROOT / name) for name in CODE_FILES},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    current = collect()
    if not args.verify:
        FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FREEZE_PATH.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"[freeze-v4] wrote {FREEZE_PATH.relative_to(ROOT)}")
        return

    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    drift = []
    for section in ("arms", "judge", "index"):
        blob_frozen = json.dumps(frozen[section], sort_keys=True)
        blob_now = json.dumps(current[section], sort_keys=True)
        if blob_frozen != blob_now:
            drift.append(section)
    for name, want in frozen["code"]["file_sha256"].items():
        got = current["code"]["file_sha256"].get(name)
        if got != want:
            drift.append(f"code.{name}")
    if drift:
        print(f"[freeze-v4] DRIFT: {drift} — Final v4 results not comparable under this tree")
        raise SystemExit(1)
    print("[freeze-v4] OK")


if __name__ == "__main__":
    main()
