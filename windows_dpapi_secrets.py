# -*- coding: utf-8 -*-
"""Windows user-bound DPAPI storage for the OfferClaw provider credential."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import tempfile


MAGIC = b"OFFERCLAW-DPAPI-V1\0"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def default_secret_path() -> Path:
    configured = os.environ.get("OFFERCLAW_DPAPI_SECRET_FILE", "").strip()
    return Path(configured) if configured else Path.home() / ".offerclaw-runtime" / "openai-key.dpapi"


def _blob(raw: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(raw)
    return _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _crypt(raw: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("DPAPI is available only on Windows")
    source, keepalive = _blob(raw)
    target = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if protect:
        ok = function(
            ctypes.byref(source), "OfferClaw provider key", None, None, None, 0,
            ctypes.byref(target),
        )
    else:
        description = wintypes.LPWSTR()
        ok = function(
            ctypes.byref(source), ctypes.byref(description), None, None, None, 0,
            ctypes.byref(target),
        )
    del keepalive
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)


def protect_secret(secret: str, path: str | Path | None = None) -> Path:
    value = str(secret or "")
    if not value:
        raise ValueError("secret cannot be empty")
    destination = Path(path) if path else default_secret_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encrypted = MAGIC + _crypt(value.encode("utf-8"), protect=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encrypted)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def unprotect_secret(path: str | Path | None = None) -> str:
    source = Path(path) if path else default_secret_path()
    raw = source.read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError("unsupported OfferClaw secret format")
    return _crypt(raw[len(MAGIC):], protect=False).decode("utf-8")


def load_dpapi_secret_into_env(env_name: str = "OPENAI_API_KEY") -> bool:
    if os.name != "nt" or os.environ.get(env_name):
        return False
    path = default_secret_path()
    if not path.is_file():
        return False
    os.environ[env_name] = unprotect_secret(path)
    return True


__all__ = [
    "default_secret_path", "load_dpapi_secret_into_env", "protect_secret",
    "unprotect_secret",
]
