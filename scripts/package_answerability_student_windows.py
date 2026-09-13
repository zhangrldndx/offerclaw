#!/usr/bin/env python3
"""Create a hash-addressed, private Windows CUDA Pilot training bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "train_answerability_student.py",
    "rag_answerability_student.py",
    "rag_answerability_student_data.py",
    "requirements-answerability-student.txt",
    "scripts/run_answerability_student_windows.ps1",
    "scripts/summarize_answerability_student_seeds.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(dataset: Path, dataset_manifest: Path, output: Path) -> dict:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    expected = ((manifest.get("private_artifact") or {}).get("sha256") or "").lower()
    actual = sha256(dataset)
    if not expected or expected != actual:
        raise ValueError(
            f"dataset/manifest SHA256 mismatch: expected={expected!r}, actual={actual}"
        )
    entries: list[tuple[Path, str]] = [
        (dataset, "input/dataset_private.jsonl"),
        (dataset_manifest, "input/dataset_manifest.json"),
    ]
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append((source, relative))
    lineage = {
        "schema_version": "answerability-student-windows-bundle-v1",
        "dataset_sha256": actual,
        "files": {
            archive: {"bytes": source.stat().st_size, "sha256": sha256(source)}
            for source, archive in entries
        },
        "contains_private_text": True,
        "required_device": "cuda",
        "seeds": [17, 29, 43],
    }
    command = (
        "powershell -ExecutionPolicy Bypass -File scripts/"
        "run_answerability_student_windows.ps1 `\n"
        "  -Dataset input/dataset_private.jsonl `\n"
        f"  -DatasetSha256 {actual} `\n"
        "  -OutputRoot output/pilot_models `\n"
        "  -Python .venv/Scripts/python.exe\n"
    )
    instructions = (
        "# Windows CUDA Pilot\n\n"
        "1. 在解压目录创建/启用 Python 环境，并安装 "
        "`requirements-answerability-student.txt`。\n"
        "2. 确保基础模型可通过默认名称加载，或给 PowerShell 命令追加 "
        "`-BaseModel <本地模型目录>`。\n"
        "3. 运行：\n\n```powershell\n" + command + "```\n\n"
        "只有 `output/pilot_models/pilot_seed_summary.json` 的 status 为 "
        "`expand_labeling` 才能继续完整标注；其余状态都必须停止。\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, name in entries:
            archive.write(source, name)
        archive.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr("WINDOWS_RUN.md", instructions)
    try:
        os.chmod(output, 0o600)
    except OSError:
        pass
    bundle_hash = sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{bundle_hash}  {output.name}\n", encoding="utf-8")
    return {
        **lineage,
        "bundle": {
            "name": output.name,
            "bytes": output.stat().st_size,
            "sha256": bundle_hash,
        },
        "checksum_file": checksum_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = package(args.dataset, args.dataset_manifest, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
