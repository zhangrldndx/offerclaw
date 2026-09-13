#!/usr/bin/env python3
"""Move OPENAI_API_KEY from .env.local into the Windows user's DPAPI store."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_env(path: Path) -> tuple[list[str], str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    value = ""
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, candidate = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                value = candidate.strip().strip('"').strip("'")
    return lines, value


def _remove_key_line(path: Path, lines: list[str]) -> None:
    kept = []
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                continue
        kept.append(raw)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(kept)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env.local")
    parser.add_argument("--secret-file", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        raise OSError("This migration must run under Windows Python")
    from windows_dpapi_secrets import protect_secret, unprotect_secret

    lines, plaintext = _read_env(args.env_file)
    secret_path = args.secret_file
    if plaintext:
        protect_secret(plaintext, secret_path)
    recovered = unprotect_secret(secret_path)
    if plaintext and recovered != plaintext:
        raise RuntimeError("DPAPI verification failed")
    os.environ["OPENAI_API_KEY"] = recovered
    if args.probe:
        from day1_api_starter import call_llm, extract_content, load_local_env
        load_local_env(str(args.env_file))
        result = call_llm(
            "Return exactly OK.", recovered,
            system="This is a credential health check. Do not include any local data.",
        )
        if not str(extract_content(result) or "").strip():
            raise RuntimeError("provider health probe returned no content")
    if plaintext:
        _remove_key_line(args.env_file, lines)
    print("OfferClaw DPAPI credential is readable; plaintext OPENAI_API_KEY is absent from .env.local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
