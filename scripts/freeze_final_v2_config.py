#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write (or verify) the frozen Final v2 candidate configuration.

Final v2 is a one-shot blind set, so the configuration has to be pinned *before*
any label exists.  Pinning the knobs alone is not enough: this round produced
three separate silent no-ops (``RAG_RECALL_N`` had no effect on this path, HyDE
needed a second env var the profile could not see, and the tiebreak was inert
whenever early exit was on).  Each looked exactly like "the technique does
nothing".  So the freeze records what actually determines behaviour --
knobs, judge model, prompt digest, index fingerprint and the digests of the
source files that implement the path -- and ``--verify`` fails loudly when any
of it has moved.

Run with no arguments to write the freeze; ``--verify`` to check the working
tree still matches it.
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
FREEZE_PATH = ROOT / "docs" / "rag_eval" / "final_v2" / "FROZEN_CONFIG.json"

# The files whose contents decide what the frozen arm does.  A change to any of
# them invalidates the freeze even if every knob still reads the same.
CODE_FILES = (
    "rag_answerability.py",
    "rag_gate.py",
    "rag_colloquial_profiles.py",
    "rag_candidate_pool.py",
    "rag_retrieval_trace.py",
    "rag_hyde.py",
    "eval_colloquial_rag.py",
)

# Every knob the frozen arm depends on, and the value it must hold.  ``""``
# means "must be unset or empty" -- the arm is defined by what is *off* just as
# much as by what is on.
FROZEN_ENV = {
    "RAG_ANSWERABILITY": "1",
    "RAG_ANSWERABILITY_EARLY_EXIT": "1",
    "RAG_ANSWERABILITY_DEPTH": "12",
    "RAG_ANSWERABILITY_MODE": "",        # teacher via the legacy boolean; no student
    "RAG_ANSWERABILITY_GATE": "",        # rejected: accept/refuse flips across cold runs
    "RAG_ANSWERABILITY_TIEBREAK": "",    # rejected: dev +2, held-out -2
    "RAG_HYDE": "",                      # arm C only, never in B
    "RAG_QUERY_REWRITE": "",
    "RAG_DOC2QUERY": "",
}
FROZEN_ARM = "compact32_pool28"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    os.environ["RAG_ANSWERABILITY"] = "1"          # so resolve_model sees teacher
    from rag_answerability import (ANSWERABILITY_DEPTH, MIN_GRADE_TO_ACT,
                                   PROMPT_SHA256, SCHEMA, resolve_model)
    from rag_colloquial_profiles import colloquial_profile
    from rag_tools import get_collection_name, index_fingerprint

    profile = colloquial_profile(FROZEN_ARM)
    fingerprint = index_fingerprint()
    return {
        "schema_version": "final-v2-frozen-config-v1",
        "arm": FROZEN_ARM,
        "retrieval_profile": profile.to_dict(),
        "env": dict(FROZEN_ENV),
        "judge": {
            "model": resolve_model(),
            "schema": SCHEMA,
            "prompt_sha256": PROMPT_SHA256,
            "min_grade_to_act": MIN_GRADE_TO_ACT,
            "module_default_depth": ANSWERABILITY_DEPTH,
            "effective_depth": int(FROZEN_ENV["RAG_ANSWERABILITY_DEPTH"]),
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
            "worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT,
                capture_output=True, text=True).stdout.strip()),
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
        print(f"[freeze] wrote {FREEZE_PATH.relative_to(ROOT)}")
        return

    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    drift = []
    for section in ("retrieval_profile", "env", "judge", "index"):
        for key, want in frozen[section].items():
            got = current[section].get(key)
            if got != want:
                drift.append(f"{section}.{key}: frozen={want!r} now={got!r}")
    for name, want in frozen["code"]["file_sha256"].items():
        got = current["code"]["file_sha256"].get(name)
        if got != want:
            drift.append(f"code.{name}: {want[:12]}... -> {got[:12] if got else None}...")
    if drift:
        print("[freeze] DRIFT — Final v2 results under this tree are not comparable:")
        for line in drift:
            print(f"  - {line}")
        raise SystemExit(1)
    print("[freeze] OK — working tree matches the frozen Final v2 configuration")


if __name__ == "__main__":
    main()
