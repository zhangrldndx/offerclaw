# -*- coding: utf-8 -*-
"""Integrity-checked access to private, repository-external evaluation sets.

Only manifests and schemas live in the repository.  The examples themselves
must be supplied through ``OFFERCLAW_PRIVATE_EVAL_ROOT`` and are never copied
into reports or failure output by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PrivateEvalUnavailable(RuntimeError):
    """The private corpus has not been provisioned on this machine."""


class PrivateEvalIntegrityError(RuntimeError):
    """The private corpus does not match its repository-pinned manifest."""


@dataclass(frozen=True)
class PrivateEvalBundle:
    dataset_id: str
    path: Path
    sha256: str
    expected_count: int
    data: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_private_eval(manifest_path: str | Path, *,
                      private_root: str | Path | None = None) -> PrivateEvalBundle:
    """Load and validate a private evaluation package.

    Missing configuration is deliberately an error rather than an empty data
    set.  Callers may present this as an explicit ``skipped`` status, but must
    never turn it into a passing evaluation.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(manifest.get("dataset_id") or manifest.get("version") or "")
    expected_file = str(manifest.get("dataset_file") or "")
    expected_hash = str(manifest.get("sha256") or "").lower()
    expected_count = int(manifest.get("expected_count") or 0)
    if not expected_file or not expected_count:
        raise PrivateEvalIntegrityError(
            f"invalid private-eval manifest: {manifest_path.name}"
        )
    if not SHA256_RE.fullmatch(expected_hash):
        raise PrivateEvalUnavailable(
            f"{dataset_id or expected_file} is not provisioned: manifest SHA-256 is unset"
        )

    root_value = private_root or os.environ.get("OFFERCLAW_PRIVATE_EVAL_ROOT", "")
    if not str(root_value).strip():
        raise PrivateEvalUnavailable(
            "OFFERCLAW_PRIVATE_EVAL_ROOT is not configured"
        )
    root = Path(root_value).expanduser().resolve()
    dataset_path = (root / expected_file).resolve()
    try:
        dataset_path.relative_to(root)
    except ValueError as exc:
        raise PrivateEvalIntegrityError("private dataset path escapes configured root") from exc
    if not dataset_path.is_file():
        raise PrivateEvalUnavailable(f"private dataset is missing: {dataset_path}")

    actual_hash = sha256_file(dataset_path)
    if actual_hash != expected_hash:
        raise PrivateEvalIntegrityError(
            f"SHA-256 mismatch for {dataset_id}: expected {expected_hash}, got {actual_hash}"
        )
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateEvalIntegrityError(
            f"private dataset is not valid UTF-8 JSON: {dataset_path}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise PrivateEvalIntegrityError("private dataset root must contain an items array")
    if len(data["items"]) != expected_count:
        raise PrivateEvalIntegrityError(
            f"private dataset count mismatch: {len(data['items'])} != {expected_count}"
        )
    ids = [str(item.get("id") or "") for item in data["items"] if isinstance(item, dict)]
    if len(ids) != expected_count or any(not item_id for item_id in ids):
        raise PrivateEvalIntegrityError("every private item must have a non-empty id")
    if len(set(ids)) != len(ids):
        raise PrivateEvalIntegrityError("private dataset contains duplicate ids")
    return PrivateEvalBundle(
        dataset_id=dataset_id,
        path=dataset_path,
        sha256=actual_hash,
        expected_count=expected_count,
        data=data,
    )


def unavailable_report(*, dataset_id: str, error: Exception) -> dict[str, Any]:
    """Machine-readable non-pass result for an unavailable private corpus."""
    return {
        "set_version": dataset_id,
        "set_kind": "private_blind",
        "status": "skipped",
        "passed": False,
        "reason": str(error),
    }
