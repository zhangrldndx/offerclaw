#!/usr/bin/env python3
"""Fail-closed validation for the OfferClaw OpenClaw reply interceptor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_HOOK = "reply_dispatch"
EXPECTED_PLUGIN_VERSION = "1.1.0"


def _hook_names(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    names: list[str] = []
    for value in values:
        if isinstance(value, str):
            name = value
        elif isinstance(value, dict):
            name = value.get("hookName") or value.get("name") or value.get("hook")
        else:
            name = None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def validate(inspect: Any, state_dir: Path, expected_launcher: Path,
             scope_config: Path | None = None) -> list[str]:
    errors: list[str] = []
    document = inspect if isinstance(inspect, dict) else {}
    plugin = document.get("plugin", document)
    runtime = document if plugin is not document else plugin
    if plugin.get("status") != "loaded":
        errors.append("plugin_not_loaded")
    if plugin.get("version") != EXPECTED_PLUGIN_VERSION:
        errors.append("plugin_version_mismatch")

    if _hook_names(runtime.get("typedHooks")) != [EXPECTED_HOOK]:
        errors.append("typed_hooks_mismatch")
    if plugin.get("hookCount") != 1:
        errors.append("hook_count_mismatch")
    accepted = (
        runtime.get("install", {}).get("acceptedSurface", {}).get("hooks", [])
        if isinstance(runtime.get("install"), dict)
        else []
    )
    accepted_hooks = _hook_names(accepted)
    accepted_surface = (
        runtime.get("install", {}).get("acceptedSurface", {})
        if isinstance(runtime.get("install"), dict)
        else {}
    )
    compatibility_codes = {
        row.get("compatCode")
        for row in runtime.get("compatibility", [])
        if isinstance(row, dict)
    }
    policy = runtime.get("policy") if isinstance(runtime.get("policy"), dict) else {}
    legacy_hook_authorized = (
        accepted_hooks == []
        and runtime.get("shape") == "hook-only"
        and "hook-only-plugin-shape" in compatibility_codes
        and policy.get("allowConversationAccess") is True
        and isinstance(accepted_surface, dict)
        and not any(accepted_surface.get(key) for key in accepted_surface)
    )
    if accepted_hooks != [EXPECTED_HOOK] and not legacy_hook_authorized:
        errors.append("accepted_hooks_mismatch")

    try:
        manifest_path = Path(plugin["rootDir"]) / "openclaw.plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        manifest = {}
    if manifest.get("hooks") != [EXPECTED_HOOK]:
        errors.append("manifest_hooks_mismatch")
    if policy.get("allowConversationAccess") is not True:
        errors.append("conversation_access_not_allowed")

    try:
        root = json.loads((state_dir / "openclaw.json").read_text(encoding="utf-8"))
        entry = root["plugins"]["entries"]["offerclaw-direct-reply"]
        config = entry["config"]
    except (OSError, ValueError, KeyError, TypeError):
        errors.append("plugin_config_unreadable")
        config = {}
        entry = {}

    if entry.get("enabled") is not True:
        errors.append("plugin_not_enabled")
    accounts = config.get("accountIds")
    senders = config.get("senderIds")
    if not (
        isinstance(accounts, list)
        and len(accounts) == 1
        and isinstance(accounts[0], str)
        and accounts[0].strip()
    ):
        errors.append("account_scope_not_exact")
    if not (
        isinstance(senders, list)
        and len(senders) == 1
        and isinstance(senders[0], str)
        and senders[0].strip()
    ):
        errors.append("sender_scope_not_exact")

    configured_launcher = config.get("launcherPath")
    try:
        launcher_matches = (
            isinstance(configured_launcher, str)
            and Path(configured_launcher).resolve(strict=True)
            == expected_launcher.resolve(strict=True)
        )
    except OSError:
        launcher_matches = False
    if not launcher_matches or not os.access(expected_launcher, os.X_OK):
        errors.append("launcher_not_executable")
    if scope_config is not None:
        try:
            scope = json.loads(scope_config.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            scope = {}
        if (
            scope.get("account_ids") != accounts
            or scope.get("sender_ids") != senders
            or not scope_config.is_file()
        ):
            errors.append("launcher_scope_mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--scope-config", type=Path)
    args = parser.parse_args()
    try:
        inspect = json.load(sys.stdin)
    except (ValueError, OSError):
        inspect = {}
    errors = validate(inspect, args.state_dir, args.launcher, args.scope_config)
    document = inspect if isinstance(inspect, dict) else {}
    runtime = document if isinstance(document.get("plugin"), dict) else document
    accepted = (
        runtime.get("install", {}).get("acceptedSurface", {}).get("hooks", [])
        if isinstance(runtime.get("install"), dict)
        else []
    )
    result = {
        "status": "error" if errors else "ok",
        "plugin_loaded": "plugin_not_loaded" not in errors,
        "hook": EXPECTED_HOOK,
        "plugin_version": EXPECTED_PLUGIN_VERSION,
        "capability_accepted": "accepted_hooks_mismatch" not in errors,
        "authorization_mode": "accepted_surface"
        if _hook_names(accepted) == [EXPECTED_HOOK]
        else "hook_only_policy",
        "identity_scope": "exact"
        if not {"account_scope_not_exact", "sender_scope_not_exact"}.intersection(errors)
        else "invalid",
        "launcher_executable": "launcher_not_executable" not in errors,
        "launcher_scope": "exact" if "launcher_scope_mismatch" not in errors else "invalid",
    }
    if errors:
        result["errors"] = errors
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
