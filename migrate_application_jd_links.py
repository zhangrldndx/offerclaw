# -*- coding: utf-8 -*-
"""One-time, idempotent migration for application/JD linkage.

Default mode is dry-run.  ``--apply`` upgrades the Markdown table and moves the
legacy independent gap target store into a recoverable archive.  It never tries
to infer a JD link from anonymous text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import shutil

import applications_store


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_GAPS = os.path.join(BASE_DIR, "gap_store.json")
ARCHIVE_DIR = os.path.join(BASE_DIR, "data", "archive")


def migrate(*, apply: bool = False) -> dict:
    schema = applications_store.ensure_application_schema(dry_run=not apply)
    archive = {"status": "missing"}
    if os.path.exists(LEGACY_GAPS):
        stamp = dt.date.today().strftime("%Y%m%d")
        destination = os.path.join(ARCHIVE_DIR, f"gap_store_legacy_{stamp}.json")
        archived = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "gap_store_legacy_*.json")))
        if archived:
            # 旧独立目标库已退出写入链路；任一历史归档存在即视为完成，
            # 避免脚本跨日期重复执行时制造多份相同归档。
            archive = {"status": "already_archived", "path": archived[-1]}
        elif apply:
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            shutil.copy2(LEGACY_GAPS, destination)
            archive = {"status": "archived", "path": destination,
                       "note": "原文件保留但新计划链路不再读取它"}
        else:
            archive = {"status": "would_archive", "path": destination}
    return {"status": "ok", "mode": "apply" if apply else "dry-run",
            "applications": schema, "legacy_gap_store": archive,
            "guessed_links": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import json
    print(json.dumps(migrate(apply=args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
