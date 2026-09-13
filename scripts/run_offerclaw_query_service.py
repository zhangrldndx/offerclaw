# -*- coding: utf-8 -*-
"""Fixed entry point for the managed loopback OfferClaw query service."""
from __future__ import annotations

import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
os.environ["OFFERCLAW_QUERY_SERVICE"] = "1"

import uvicorn  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(
        "rag_api:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        reload=False,
        access_log=False,
    )
