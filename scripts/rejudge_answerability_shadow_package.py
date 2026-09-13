#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay only the answerability judge for an immutable shadow package.

The source package already contains the original user question, the retrieval
subquery, the exact Top-N candidate documents, and all public retrieval/gate
features.  This tool verifies those artifacts and reuses them verbatim.  It
never imports or calls the planner, router, retriever, or reranker paths.

The original user question is the judge's proposition-bearing input.  The
retrieval subquery is retained only to describe what the captured evidence was
retrieved for; query planning is allowed to neutralise a false premise and is
therefore not a valid replacement for the original question during judging.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_answerability_shadow_seed import (  # noqa: E402
    _file_sha256,
    _git_state,
    _json_dump,
    _jsonl_dump,
    _sha256_text,
)


REJUDGE_SCHEMA = "answerability-shadow-rejudge-v1"
SOURCE_JUDGE_SCHEMA = "answerability-v3"
TARGET_JUDGE_SCHEMA = "answerability-v4"
JUDGE_QUESTION_ROLE = "original_user_question"

REQUIRED_SOURCE_FILES = (
    "route_outcomes.jsonl",
    "claude_agent_calibration.jsonl",
    "private_audit.jsonl",
    "summary.json",
    "CLAUDE_HANDOFF.md",
    "generated_queries_private.jsonl",
)
PRIVATE_CANDIDATE_FIELDS = (
    "rank",
    "chunk_id",
    "source",
    "source_type",
    "heading_path",
    "rerank_score",
    "document",
)
PUBLIC_VERDICT_ENUMS = {
    "question_form": frozenset({"polar", "open"}),
    "direct_answer": frozenset({
        "proposition_true", "proposition_false", "unknown", "not_applicable",
    }),
    "premise_status": frozenset({
        "supported", "refuted", "not_established", "none",
    }),
}


class RejudgePackageError(RuntimeError):
    """Raised before publishing when replay provenance is not trustworthy."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RejudgePackageError(f"cannot read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RejudgePackageError(f"JSON artifact {path.name} is not an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RejudgePackageError(f"cannot read JSONL artifact {path.name}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RejudgePackageError(
                f"invalid JSON in {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise RejudgePackageError(
                f"JSONL row is not an object in {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def _public_package_name(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        # Do not put an operator's absolute local path in a public manifest.
        return resolved.name


def verify_source_manifest(source_package: Path) -> dict[str, Any]:
    """Verify every declared source artifact and the required six-file contract."""
    manifest_path = source_package / "manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RejudgePackageError("source manifest has no files object")

    missing = [name for name in REQUIRED_SOURCE_FILES if name not in files]
    if missing:
        raise RejudgePackageError(
            "source manifest is missing required files: " + ", ".join(missing)
        )
    unexpected = sorted(set(files) - set(REQUIRED_SOURCE_FILES))
    if unexpected:
        raise RejudgePackageError(
            "source manifest has unexpected artifact entries: "
            + ", ".join(unexpected)
        )
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise RejudgePackageError("source manifest contains an invalid file entry")
        if Path(name).name != name:
            raise RejudgePackageError(
                f"source manifest artifact is not a basename: {name!r}"
            )
        path = source_package / name
        if not path.is_file():
            raise RejudgePackageError(f"source artifact is missing: {name}")
        expected_bytes = metadata.get("bytes")
        expected_sha256 = metadata.get("sha256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise RejudgePackageError(f"source manifest has invalid bytes for {name}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RejudgePackageError(f"source manifest has invalid sha256 for {name}")
        if path.stat().st_size != expected_bytes:
            raise RejudgePackageError(f"source artifact byte count changed: {name}")
        if _file_sha256(path) != expected_sha256:
            raise RejudgePackageError(f"source artifact sha256 changed: {name}")

    if manifest.get("judge_schema") != SOURCE_JUDGE_SCHEMA:
        raise RejudgePackageError(
            f"source judge schema must be {SOURCE_JUDGE_SCHEMA!r}, got "
            f"{manifest.get('judge_schema')!r}"
        )
    depth = manifest.get("judge_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise RejudgePackageError("source manifest has invalid judge_depth")
    return manifest


def _unique_by_traffic_id(
    rows: list[dict[str, Any]], *, artifact: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        traffic_id = row.get("traffic_id")
        if not isinstance(traffic_id, str) or not traffic_id:
            raise RejudgePackageError(
                f"{artifact}:{line_number} has no valid traffic_id"
            )
        if traffic_id in indexed:
            raise RejudgePackageError(
                f"{artifact} has duplicate traffic_id {traffic_id!r}"
            )
        indexed[traffic_id] = row
    return indexed


def load_replay_rows(
    source_package: Path, source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild judge inputs only after checking public/private row identity."""
    route_rows = _read_jsonl(source_package / "route_outcomes.jsonl")
    public_rows = _read_jsonl(source_package / "claude_agent_calibration.jsonl")
    private_rows = _read_jsonl(source_package / "private_audit.jsonl")
    generated_rows = _read_jsonl(source_package / "generated_queries_private.jsonl")

    if len(public_rows) != len(private_rows):
        raise RejudgePackageError(
            "public calibration and private audit row counts differ"
        )
    generated_by_id = _unique_by_traffic_id(
        generated_rows, artifact="generated_queries_private.jsonl",
    )
    route_by_id = _unique_by_traffic_id(
        route_rows, artifact="route_outcomes.jsonl",
    )
    if set(generated_by_id) != set(route_by_id):
        raise RejudgePackageError(
            "generated query and route outcome traffic_id sets differ"
        )
    for traffic_id, generated in generated_by_id.items():
        question = generated.get("question")
        if not isinstance(question, str) or not question:
            raise RejudgePackageError(
                f"generated query {traffic_id!r} has no original question"
            )
        route = route_by_id[traffic_id]
        if route.get("question_sha256") != _sha256_text(question):
            raise RejudgePackageError(
                f"generated/route original question SHA mismatch for {traffic_id!r}"
            )

    depth = int(source_manifest["judge_depth"])
    replay_rows: list[dict[str, Any]] = []
    for line_number, (public, private) in enumerate(
        zip(public_rows, private_rows, strict=True), 1,
    ):
        traffic_id = public.get("traffic_id")
        if not isinstance(traffic_id, str) or traffic_id != private.get("traffic_id"):
            raise RejudgePackageError(
                f"public/private traffic identity mismatch at reference row {line_number}"
            )
        generated = generated_by_id.get(traffic_id)
        route = route_by_id.get(traffic_id)
        if generated is None or route is None:
            raise RejudgePackageError(
                f"reference row {line_number} has no generated/route parent"
            )
        question = private.get("question")
        retrieval_question = private.get("retrieval_question")
        if not isinstance(question, str) or not question:
            raise RejudgePackageError(
                f"private reference row {line_number} has no original question"
            )
        if not isinstance(retrieval_question, str) or not retrieval_question:
            raise RejudgePackageError(
                f"private reference row {line_number} has no retrieval question"
            )
        if question != generated.get("question"):
            raise RejudgePackageError(
                f"private/generated original question mismatch at row {line_number}"
            )
        if public.get("original_question_sha256") != _sha256_text(question):
            raise RejudgePackageError(
                f"original user question SHA mismatch at reference row {line_number}"
            )
        if public.get("retrieval_question_sha256") != _sha256_text(retrieval_question):
            raise RejudgePackageError(
                f"retrieval question SHA mismatch at reference row {line_number}"
            )
        if route.get("decision") != "answer" or not route.get("has_reference_kb"):
            raise RejudgePackageError(
                f"reference row {line_number} is not backed by an answered reference route"
            )

        candidates = private.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < depth:
            raise RejudgePackageError(
                f"private reference row {line_number} has fewer than Top{depth} candidates"
            )
        public_verdicts = public.get("verdicts")
        if public.get("judge_selected"):
            if not isinstance(public_verdicts, list) or len(public_verdicts) != depth:
                raise RejudgePackageError(
                    f"selected public reference row {line_number} lacks exact Top{depth} identity"
                )
        elif not isinstance(public_verdicts, list):
            raise RejudgePackageError(
                f"public verdicts is not a list at reference row {line_number}"
            )

        rebuilt_candidates: list[dict[str, Any]] = []
        public_by_rank = {
            item.get("rank"): item for item in public_verdicts
            if isinstance(item, dict)
        }
        for candidate_number, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                raise RejudgePackageError(
                    f"candidate is not an object at row {line_number}:{candidate_number}"
                )
            missing_fields = [
                field for field in PRIVATE_CANDIDATE_FIELDS if field not in candidate
            ]
            if missing_fields:
                raise RejudgePackageError(
                    f"candidate at row {line_number}:{candidate_number} is missing "
                    + ", ".join(missing_fields)
                )
            rank = candidate.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank != candidate_number:
                raise RejudgePackageError(
                    f"candidate rank/order mismatch at row {line_number}:{candidate_number}"
                )
            if public.get("judge_selected") and rank <= depth:
                public_identity = public_by_rank.get(rank)
                if public_identity is None:
                    raise RejudgePackageError(
                        f"public Top{depth} rank missing at row {line_number}:{rank}"
                    )
                if public_identity.get("chunk_id_sha256") != _sha256_text(
                    str(candidate["chunk_id"])
                ):
                    raise RejudgePackageError(
                        f"candidate chunk SHA mismatch at row {line_number}:{rank}"
                    )
                if public_identity.get("rerank_score") != candidate.get("rerank_score"):
                    raise RejudgePackageError(
                        f"candidate rerank score mismatch at row {line_number}:{rank}"
                    )
            rebuilt_candidates.append({
                field: candidate[field] for field in PRIVATE_CANDIDATE_FIELDS
            })

        replay_rows.append({
            **public,
            "question": question,
            "retrieval_question": retrieval_question,
            "judge_question_role": JUDGE_QUESTION_ROLE,
            "candidates_private": rebuilt_candidates,
            # Never append v4 verdicts to v3 verdicts.
            "verdicts": [],
        })

    source_summary = _read_json(source_package / "summary.json")
    if source_summary.get("generated_queries") != len(route_rows):
        raise RejudgePackageError("source summary generated query count mismatch")
    if source_summary.get("reference_kb_queries") != len(public_rows):
        raise RejudgePackageError("source summary reference query count mismatch")
    return route_rows, replay_rows, generated_rows


def _publish_public_enums(
    public_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]],
) -> None:
    """Expose only bounded v4 diagnostics, never free-form judge reasoning."""
    if len(public_rows) != len(replay_rows):
        raise RejudgePackageError("packager changed reference row count")
    for line_number, (public, replay) in enumerate(
        zip(public_rows, replay_rows, strict=True), 1,
    ):
        if public.get("traffic_id") != replay.get("traffic_id"):
            raise RejudgePackageError(
                f"packager changed reference row identity at line {line_number}"
            )
        public["judge_question_role"] = JUDGE_QUESTION_ROLE
        raw_by_rank = {
            item["rank"]: item.get("verdict")
            for item in replay.get("verdicts", [])
        }
        for verdict in public.get("verdicts", []):
            raw = raw_by_rank.get(verdict.get("rank"))
            if not isinstance(raw, dict):
                continue
            for field, allowed in PUBLIC_VERDICT_ENUMS.items():
                value = raw.get(field)
                if value not in allowed:
                    raise RejudgePackageError(
                        f"invalid or missing v4 {field} at row {line_number}, "
                        f"rank {verdict.get('rank')}: {value!r}"
                    )
                verdict[field] = value


def _rejudge_handoff(
    base_handoff: str, *, source_package_name: str,
    source_manifest_sha256: str,
) -> str:
    return base_handoff + f"""

## Judge-only replay lineage

This is an `{TARGET_JUDGE_SCHEMA}` judge-only replay of
`{source_package_name}`. The planner, router, retriever, reranker, gate, route
outcomes, candidate identity, and trigger features were **not rerun**. The
source manifest SHA256 is `{source_manifest_sha256}`.

The judge received the original user question as its proposition-bearing
input (`judge_question_role={JUDGE_QUESTION_ROLE}`). The stored retrieval
subquery was supplied only as retrieval-scope context. Public verdicts expose
only `question_form`, `direct_answer`, and `premise_status` enums; free-form
judge reasons and all question/evidence text remain private.

This package supersedes its v3 labels, not its retrieval observations. Any
analysis fitted to the v3 verdicts is stale and must not be reused as evidence
for freezing a trigger rule.
"""


def rejudge_package(
    source_package: Path,
    output_dir: Path,
    *,
    judge_model: str | None = None,
    judge_workers: int = 4,
    judge_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate, judge, package, and atomically publish a v4 replay."""
    source_package = source_package.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_package.is_dir():
        raise RejudgePackageError(f"source package does not exist: {source_package}")
    if output_dir == source_package:
        raise RejudgePackageError("output directory cannot be the source package")
    if output_dir.is_relative_to(source_package):
        raise RejudgePackageError("output directory cannot be inside the source package")
    if output_dir.exists():
        raise RejudgePackageError(f"output directory already exists: {output_dir}")
    if isinstance(judge_workers, bool) or not isinstance(judge_workers, int) or judge_workers < 1:
        raise RejudgePackageError("judge_workers must be a positive integer")

    source_manifest = verify_source_manifest(source_package)
    route_rows, replay_rows, _generated_rows = load_replay_rows(
        source_package, source_manifest,
    )

    from rag_answerability import SCHEMA as judge_schema
    from scripts.run_answerability_shadow_traffic_agent import (
        _handoff,
        judge_selected_rows,
        package_rows,
        validate_public_exports,
    )

    if judge_schema != TARGET_JUDGE_SCHEMA:
        raise RejudgePackageError(
            f"current judge schema must be {TARGET_JUDGE_SCHEMA!r}, got {judge_schema!r}"
        )
    resolved_model = str(
        judge_model or source_manifest.get("judge_model_requested") or ""
    ).strip()
    if not resolved_model:
        raise RejudgePackageError("judge model must be explicit")

    runner = judge_runner or judge_selected_rows
    judge_run = runner(
        replay_rows,
        depth=int(source_manifest["judge_depth"]),
        workers=judge_workers,
        model=resolved_model,
    )
    if judge_run.get("judge_unavailable") != 0:
        raise RejudgePackageError(
            f"v4 judge unavailable for {judge_run.get('judge_unavailable')} job(s)"
        )

    public_routes, public_reference, private_reference, summary = package_rows(
        route_rows,
        replay_rows,
        judge_depth=int(source_manifest["judge_depth"]),
        judge_model=resolved_model,
    )
    _publish_public_enums(public_reference, replay_rows)
    validate_public_exports(public_routes, public_reference, replay_rows)
    summary["judge_run"] = judge_run
    summary["judge_question_role"] = JUDGE_QUESTION_ROLE

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent,
    ))
    try:
        routes_path = staging / "route_outcomes.jsonl"
        calibration_path = staging / "claude_agent_calibration.jsonl"
        private_path = staging / "private_audit.jsonl"
        summary_path = staging / "summary.json"
        handoff_path = staging / "CLAUDE_HANDOFF.md"
        generated_private_path = staging / "generated_queries_private.jsonl"

        # These two artifacts are retrieval/generation observations and must be
        # byte-identical to the source, not merely semantically equivalent JSON.
        shutil.copy2(source_package / routes_path.name, routes_path)
        shutil.copy2(
            source_package / generated_private_path.name, generated_private_path,
        )
        _jsonl_dump(calibration_path, public_reference)
        _jsonl_dump(private_path, private_reference)
        _json_dump(summary_path, summary)

        source_manifest_sha256 = _file_sha256(source_package / "manifest.json")
        source_package_name = _public_package_name(source_package)
        handoff_path.write_text(
            _rejudge_handoff(
                _handoff(summary),
                source_package_name=source_package_name,
                source_manifest_sha256=source_manifest_sha256,
            ),
            encoding="utf-8",
        )
        artifact_paths = (
            routes_path,
            calibration_path,
            private_path,
            summary_path,
            handoff_path,
            generated_private_path,
        )
        rejudge_lineage = {
            "schema_version": REJUDGE_SCHEMA,
            "operation": "judge_only_replay",
            "planner_rerun": False,
            "retrieval_rerun": False,
            "judge_question_role": JUDGE_QUESTION_ROLE,
            "source_package": source_package_name,
            "source_manifest_sha256": source_manifest_sha256,
            "source_judge_schema": source_manifest["judge_schema"],
            "target_judge_schema": judge_schema,
            "source_git": source_manifest.get("git", {}),
            "judge_code_sha256": _file_sha256(ROOT / "rag_answerability.py"),
            "rejudge_script_sha256": _file_sha256(Path(__file__).resolve()),
        }
        manifest = {
            **{
                key: value for key, value in source_manifest.items()
                if key not in {
                    "created_at", "files", "git", "judge_model_requested",
                    "judge_schema",
                }
            },
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git": _git_state(),
            "judge_model_requested": resolved_model,
            "judge_schema": judge_schema,
            "judge_question_role": JUDGE_QUESTION_ROLE,
            "rejudge_lineage": rejudge_lineage,
            "files": {
                path.name: {
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifact_paths
            },
        }
        manifest_path = staging / "manifest.json"
        _json_dump(manifest_path, manifest)

        # Re-read the completed staging package before the atomic publish.
        for path in artifact_paths:
            metadata = manifest["files"][path.name]
            if path.stat().st_size != metadata["bytes"] or _file_sha256(path) != metadata["sha256"]:
                raise RejudgePackageError(
                    f"staged artifact changed before publish: {path.name}"
                )
        if output_dir.exists():
            raise RejudgePackageError(
                f"output directory appeared before publish: {output_dir}"
            )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "manifest": str(output_dir / "manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = rejudge_package(
            args.source_package,
            args.output_dir,
            judge_model=args.judge_model,
            judge_workers=args.judge_workers,
        )
    except RejudgePackageError as exc:
        parser.exit(2, f"rejudge refused: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
