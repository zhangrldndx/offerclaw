#!/usr/bin/env python3
"""Run the small, isolated OfferClaw product acceptance set.

The manifest points at existing deterministic integration contracts instead of
copying their implementation.  Pytest raw output and temporary files must stay
outside the repository; only an explicitly requested, sanitized JSON summary
may be published under ``docs/``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "product_acceptance_v2.json"


class AcceptanceContractError(ValueError):
    """Raised when the acceptance manifest or isolation boundary is invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside_repo(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved == ROOT or ROOT in resolved.parents


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "offerclaw-product-acceptance-v2":
        raise AcceptanceContractError("unsupported acceptance schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 36:
        raise AcceptanceContractError("acceptance v2 must contain exactly 36 cases")
    ids = [str(row.get("id") or "") for row in cases]
    nodeids = [str(row.get("nodeid") or "") for row in cases]
    if any(not value for value in ids + nodeids):
        raise AcceptanceContractError("every case needs an id and nodeid")
    if len(set(ids)) != len(ids) or len(set(nodeids)) != len(nodeids):
        raise AcceptanceContractError("case ids and nodeids must be unique")
    areas = Counter(str(row.get("area") or "") for row in cases)
    if len(areas) != 6 or set(areas.values()) != {6}:
        raise AcceptanceContractError("acceptance v2 requires six areas x six cases")
    for row in cases:
        nodeid = row["nodeid"]
        if not nodeid.startswith("tests/") or "::" not in nodeid:
            raise AcceptanceContractError(f"unsafe pytest nodeid: {nodeid!r}")
        test_path = ROOT / nodeid.split("::", 1)[0]
        if not test_path.is_file():
            raise AcceptanceContractError(f"missing test file: {test_path}")
        if not str(row.get("contract") or "").strip():
            raise AcceptanceContractError(f"missing contract for {row['id']}")
    isolation = payload.get("isolation") or {}
    if isolation.get("external_llm") is not False:
        raise AcceptanceContractError("default acceptance set must be offline")
    if isolation.get("real_user_writes") is not False:
        raise AcceptanceContractError("acceptance set may not write real user data")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    """True when the working tree differs from HEAD.

    Without this flag a summary produced from a dirty tree reads as the
    recorded commit's result, which it is not.  Untracked eval artifacts and
    local notes are ignored; only tracked-file modifications count.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def _parse_junit(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    rows: dict[str, dict] = {}
    for case in root.iter("testcase"):
        classname = str(case.attrib.get("classname") or "")
        name = str(case.attrib.get("name") or "")
        module = classname.replace(".", "/") + ".py"
        nodeid = f"{module}::{name}"
        status = "passed"
        detail = ""
        for tag in ("failure", "error", "skipped"):
            child = case.find(tag)
            if child is not None:
                status = "failed" if tag in {"failure", "error"} else "skipped"
                detail = str(child.attrib.get("message") or "")[:240]
                break
        rows[nodeid] = {
            "status": status,
            "duration_ms": round(float(case.attrib.get("time") or 0) * 1000, 1),
            "detail": detail,
        }
    return rows


def run(manifest_path: Path, raw_dir: Path, summary_path: Path | None) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    raw_dir = raw_dir.expanduser().resolve()
    if _inside_repo(raw_dir):
        raise AcceptanceContractError("raw acceptance output must stay outside repository")
    if summary_path is not None:
        summary_path = summary_path.expanduser().resolve()
        if not _inside_repo(summary_path):
            raise AcceptanceContractError("published summary must stay in repository")

    manifest = load_manifest(manifest_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    junit_path = raw_dir / "junit.xml"
    stdout_path = raw_dir / "pytest_stdout.txt"
    nodeids = [row["nodeid"] for row in manifest["cases"]]
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
        "RAG_RERANK": "0",
        "OFFERCLAW_E2E": "0",
    })
    command = [
        sys.executable, "-m", "pytest", *nodeids, "-q",
        f"--junitxml={junit_path}", f"--basetemp={raw_dir / 'pytest_tmp'}",
    ]
    proc = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True,
    )
    stdout_path.write_text(
        proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if not junit_path.is_file():
        raise AcceptanceContractError("pytest did not produce JUnit output")
    observed = _parse_junit(junit_path)
    per_case = []
    by_area: dict[str, Counter] = defaultdict(Counter)
    for row in manifest["cases"]:
        result = observed.get(row["nodeid"], {
            "status": "missing", "duration_ms": 0.0,
            "detail": "nodeid absent from JUnit output",
        })
        compact = {
            "id": row["id"], "area": row["area"],
            "nodeid": row["nodeid"], **result,
        }
        per_case.append(compact)
        by_area[row["area"]][result["status"]] += 1

    counts = Counter(row["status"] for row in per_case)
    summary = {
        "schema_version": "offerclaw-product-acceptance-result-v1",
        "release_status": "integration_regression_not_blind",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": _sha256(manifest_path),
            "cases": len(per_case),
        },
        "isolation": manifest["isolation"],
        "counts": dict(sorted(counts.items())),
        "by_area": {area: dict(sorted(values.items()))
                    for area, values in sorted(by_area.items())},
        "duration_ms": round(sum(row["duration_ms"] for row in per_case), 1),
        "cases": per_case,
        "raw_artifacts": {
            "location": "outside_repository",
            "junit_sha256": _sha256(junit_path),
            "stdout_sha256": _sha256(stdout_path),
        },
        "exit_code": proc.returncode,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if proc.returncode != 0 or counts.get("passed", 0) != len(per_case):
        raise SystemExit(proc.returncode or 1)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--summary", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    output = run(args.manifest, args.raw_dir, args.summary)
    print(json.dumps({
        "counts": output["counts"],
        "by_area": output["by_area"],
        "summary": str(args.summary) if args.summary else "",
    }, ensure_ascii=False, indent=2))
