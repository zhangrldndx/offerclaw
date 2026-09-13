# -*- coding: utf-8 -*-
"""Deployment contract tests for the isolated OpenClaw WeChat lab."""
from pathlib import Path
import json
import re
import subprocess
import sys

from scripts.openclaw_direct_reply_preflight import validate as validate_preflight
from scripts.sync_openclaw_agent_routing import END, START, sync_routing


ROOT = Path(__file__).resolve().parents[1]


def test_root_script_is_the_only_automation_definition():
    wrapper = (ROOT / "scripts" / "setup_openclaw_cron.sh").read_text(encoding="utf-8")
    assert 'exec bash "$ROOT_DIR/setup_wechat.sh" --cron-only "$@"' in wrapper
    assert "automations add" not in wrapper


def test_jobs_are_exact_named_disabled_and_use_shanghai_time():
    script = (ROOT / "setup_wechat.sh").read_text(encoding="utf-8")
    assert 'readonly TIMEZONE="Asia/Shanghai"' in script
    assert '"0 22 * * *"' in script
    assert "--tz \"$TIMEZONE\" --exact" in script
    assert "--disabled" in script
    assert 'x.get("name")==sys.argv[1]' in script
    assert "--declaration-key" not in script  # exact-name update remains the source of truth
    assert 'local marker="offerclaw://automation/$automation_kind?v=1"' in script
    assert "--command-argv" in script and "--command-input" in script
    assert '"wechat-dispatch", "--stdin", "--reply-text"' in script
    assert "--model \"$MODEL_REF\"" not in script.split("sync_job()", 1)[1]
    assert "--tools exec" not in script
    assert '--account "$delivery_account"' in script
    assert "resolved_delivery_account" in script
    assert "__offerclaw_no_tools__" not in script
    assert "job_needs_command_payload" in script
    assert 'oc automations delete "$id"' in script


def test_current_wechat_login_and_security_contract():
    script = (ROOT / "setup_wechat.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "integrations" / "openclaw" / "offerclaw-launcher.sh.tmpl").read_text(
        encoding="utf-8"
    )
    assert "channels login --channel openclaw-weixin" in script
    assert "channels.openclaw-weixin.dmPolicy" in script and "pairing" in script
    assert "channels.openclaw-weixin.groupPolicy" in script and "disabled" in script
    assert "tools.exec.security" in script and "allowlist" in script
    assert 'approvals allowlist remove --agent "$AGENT_ID" "$LAUNCHER_DST"' in script
    assert "skills.workshop.autonomous.mode" in script and "'\"off\"'" in script
    assert "plugins.entries.memory-core.config.dreaming.enabled false" in script
    assert "agents.defaults.heartbeat.every '\"0m\"'" in script
    assert "torch==$TORCH_CPU_VERSION" in script
    assert "https://download.pytorch.org/whl/cpu" in script
    assert "OFFERCLAW_PYPI_INDEX_URL" in script
    assert 'readonly MODEL_PROVIDER_ID="${OFFERCLAW_MODEL_PROVIDER_ID:-offerclaw-local}"' in script
    assert 'readonly MODEL_ID="${OFFERCLAW_MODEL_ID:-}"' in script
    assert 'models.providers.$MODEL_PROVIDER_ID' in script
    assert "offerclaw-" + "proxy" not in script
    assert "--timeout 120 --retries 5 --progress-bar off" in script
    assert '"agents.entries.$AGENT_ID.skills" \'[]\'' in script
    assert '"agents.entries.$AGENT_ID.tools.allow" \'["web_search","web_fetch"]\'' in script
    assert '"exec","ls","read","edit","write"' in script
    assert "agents bind --agent \"$AGENT_ID\" --bind openclaw-weixin" in script
    assert 'x.get("match", {}).get("channel")=="openclaw-weixin"' in script
    assert "gateway start is deferred" in script
    assert "for attempt in {1..15}" in script
    assert 'say "Gateway RPC is ready on port $GATEWAY_PORT"' in script
    assert "bind_weixin_accounts" in script
    assert 'x.get("match",{}).get("accountId")==sys.argv[2]' in script
    assert '--bind "openclaw-weixin:$account_id"' in script
    assert "bound_weixin_senders" in script
    assert "context-tokens.json" in script
    assert "OPENCLAW_PROFILE_STATE_DIR" in script
    assert 'sender_count" != "1"' in script
    assert "OFFERCLAW_WECHAT_SENDER_IDS" in script
    assert "resolved_private_target" in script
    assert "No unique private WeChat target is available" in script
    assert 'plugins install --link "$DIRECT_PLUGIN_DIR" --force' in script
    assert "--accept-capabilities --acknowledge-install-policy-warning" in script
    assert "--preflight-direct-reply" in script
    assert "ExecStartPre=/bin/bash %s --preflight-direct-reply" in script
    assert "Gateway stopped while direct-reply capabilities are verified" in script
    assert "gateway stop --force" in script
    assert "Gateway RPC is still reachable after stop" in script
    assert "preflight_direct_reply" in script
    assert "plugins inspect offerclaw-direct-reply --runtime --json" in script
    assert "hooks.allowConversationAccess true" in script
    assert 'channels.openclaw-weixin.enabled false' in script
    assert 'cd "$OFFERCLAW_DIR"' in launcher
    assert "OFFERCLAW_LOCAL_EMBEDDING_MODEL" in launcher
    assert 'export EMBEDDING_MODEL="$LOCAL_EMBEDDING_CACHE"' in launcher
    assert 'load_wechat_scope' in launcher
    assert 'OFFERCLAW_WECHAT_ALLOWED_ACCOUNT_IDS' in launcher
    assert '--scope-config "$WECHAT_SCOPE_CONFIG"' in script
    assert "umask 077" in launcher
    assert "UMask=0077" in script
    assert "harden_runtime_permissions" in script
    assert 'chmod 700 "$OPENCLAW_PROFILE_STATE_DIR" "$LAB_ROOT"' in script


def test_direct_reply_preflight_accepts_only_exact_hook_and_identity_scope(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "plugins": {"entries": {"offerclaw-direct-reply": {
            "enabled": True,
            "config": {
                "launcherPath": sys.executable,
                "accountIds": ["account-1"],
                "senderIds": ["self@im.wechat"],
            },
        }}},
    }
    (state / "openclaw.json").write_text(json.dumps(config), encoding="utf-8")
    inspect = {
        "status": "loaded",
        "version": "1.1.0",
        "hookCount": 1,
        "typedHooks": ["reply_dispatch"],
        "rootDir": str(ROOT / "integrations" / "openclaw" / "offerclaw-direct-reply"),
        "policy": {"allowConversationAccess": True},
        "install": {"acceptedSurface": {"hooks": ["reply_dispatch"]}},
    }
    assert validate_preflight(inspect, state, Path(sys.executable)) == []

    for field, value, expected in (
        ("typedHooks", [], "typed_hooks_mismatch"),
        ("typedHooks", ["reply_dispatch", "before_dispatch"], "typed_hooks_mismatch"),
        ("status", "disabled", "plugin_not_loaded"),
    ):
        broken = json.loads(json.dumps(inspect))
        broken[field] = value
        assert expected in validate_preflight(broken, state, Path(sys.executable))

    broken = json.loads(json.dumps(inspect))
    broken["install"]["acceptedSurface"]["hooks"] = []
    assert "accepted_hooks_mismatch" in validate_preflight(
        broken, state, Path(sys.executable)
    )


def test_direct_reply_preflight_rejects_broad_scope_and_bad_launcher(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "plugins": {"entries": {"offerclaw-direct-reply": {
            "enabled": True,
            "config": {
                "launcherPath": str(tmp_path / "missing"),
                "accountIds": ["one", "two"],
                "senderIds": [],
            },
        }}},
    }
    (state / "openclaw.json").write_text(json.dumps(config), encoding="utf-8")
    inspect = {
        "status": "loaded",
        "version": "1.1.0",
        "hookCount": 1,
        "typedHooks": ["reply_dispatch"],
        "rootDir": str(ROOT / "integrations" / "openclaw" / "offerclaw-direct-reply"),
        "policy": {"allowConversationAccess": True},
        "install": {"acceptedSurface": {"hooks": ["reply_dispatch"]}},
    }
    errors = validate_preflight(inspect, state, tmp_path / "missing")
    assert {"account_scope_not_exact", "sender_scope_not_exact", "launcher_not_executable"} <= set(errors)


def test_direct_reply_preflight_requires_launcher_scope_to_match_plugin(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    launcher = tmp_path / "launcher"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o700)
    config = {
        "plugins": {"entries": {"offerclaw-direct-reply": {
            "enabled": True,
            "config": {
                "launcherPath": str(launcher),
                "accountIds": ["account-1"],
                "senderIds": ["self@im.wechat"],
            },
        }}},
    }
    (state / "openclaw.json").write_text(json.dumps(config), encoding="utf-8")
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({
        "account_ids": ["account-1"], "sender_ids": ["other@im.wechat"],
    }), encoding="utf-8")
    inspect = {
        "status": "loaded", "version": "1.1.0", "hookCount": 1,
        "typedHooks": ["reply_dispatch"],
        "rootDir": str(ROOT / "integrations" / "openclaw" / "offerclaw-direct-reply"),
        "policy": {"allowConversationAccess": True},
        "install": {"acceptedSurface": {"hooks": ["reply_dispatch"]}},
    }
    assert "launcher_scope_mismatch" in validate_preflight(
        inspect, state, launcher, scope,
    )


def test_direct_reply_preflight_cli_returns_stable_json(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "plugins": {"entries": {"offerclaw-direct-reply": {
            "enabled": True,
            "config": {
                "launcherPath": sys.executable,
                "accountIds": ["account-1"],
                "senderIds": ["self@im.wechat"],
            },
        }}},
    }
    (state / "openclaw.json").write_text(json.dumps(config), encoding="utf-8")
    inspect = {
        "plugin": {
            "status": "loaded",
            "version": "1.1.0",
            "hookCount": 1,
            "rootDir": str(ROOT / "integrations" / "openclaw" / "offerclaw-direct-reply"),
        },
        "typedHooks": [{"name": "reply_dispatch"}],
        "shape": "hook-only",
        "compatibility": [{"compatCode": "hook-only-plugin-shape"}],
        "policy": {"allowConversationAccess": True},
        "install": {"acceptedSurface": {
            "channels": [], "providers": [], "tools": [], "contracts": [],
            "hooks": [], "mcpServers": [], "cliCommands": [],
            "cliBackends": [], "skills": [], "dangerousConfigFlags": [],
        }},
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "openclaw_direct_reply_preflight.py"),
         "--state-dir", str(state), "--launcher", sys.executable],
        input=json.dumps(inspect), text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["authorization_mode"] == "hook_only_policy"


def test_direct_reply_smoke_uses_stdin_and_redacts_business_output():
    smoke = (ROOT / "scripts" / "smoke_openclaw_direct_reply.mjs").read_text(
        encoding="utf-8"
    )
    assert "for await (const chunk of process.stdin)" in smoke
    assert "createReplyDispatchHandler(config, logger, async" in smoke
    assert "reply_chars" in smoke and "replies[0]" not in smoke
    assert "accountIds[0]" not in smoke.split("console.log", 1)[1]


def test_setup_uses_only_the_real_disposable_index():
    script = (ROOT / "setup_wechat.sh").read_text(encoding="utf-8")
    main_flow = script.split("if [[ $CRON_ONLY -eq 0 ]]", 1)[1]
    assert "ensure_index" not in main_flow
    assert "sync_state_sources" not in main_flow
    assert 'run "$LAUNCHER_DST" wechat-index-sync' in main_flow
    env_file = (ROOT / "integrations" / "openclaw" / "lab-fixtures" / "env.example")
    env_text = env_file.read_text(encoding="utf-8")
    assert "EMBEDDING_DIMENSIONS=768" in env_text
    assert "OFFERCLAW_TORCH_DEVICE=cpu" in env_text


def test_suggestion_seed_refuses_non_lab_paths_and_never_decides():
    script = (ROOT / "scripts" / "seed_openclaw_lab_suggestion.py").read_text(
        encoding="utf-8"
    )
    assert '"offerclaw-lab" / "app"' in script
    assert "Refusing to seed outside the isolated lab" in script
    assert "run_audit" in script
    assert "decide_suggestion" not in script


def test_lab_snapshot_excludes_nested_runtime_caches():
    script = (ROOT / "scripts" / "create_openclaw_lab.sh").read_text(encoding="utf-8")
    assert 'case "$DEST_DIR" in' in script
    assert 'rm -rf -- "$DEST_DIR"' in script
    assert "--exclude='*/__pycache__'" in script
    assert "--exclude='*/.pytest_cache'" in script
    assert 'offerclaw-launcher.sh.tmpl" > "$LAUNCHER_DST"' in script
    assert 'SKILL.md.tmpl" > "$SKILL_DST"' in script
    assert "AGENTS.public-fallback.md.tmpl" in script
    assert "sync_openclaw_agent_routing.py" not in script


def test_public_fallback_agent_has_no_local_business_context():
    policy = (ROOT / "integrations" / "openclaw" / "AGENTS.public-fallback.md.tmpl").read_text(
        encoding="utf-8"
    )
    assert "receives only text" in policy
    assert "Do not use or request local files" in policy
    assert "{{OFFERCLAW_LAUNCHER}}" not in policy
    assert "profile-suggestion" not in policy


def test_lab_fixtures_cover_stable_application_and_attachment_flows():
    fixture_root = ROOT / "integrations" / "openclaw" / "lab-fixtures"
    applications = (fixture_root / "applications.md").read_text(encoding="utf-8")
    assert "投递ID" in applications
    assert "app_wechat_demo_001" in applications
    assert (fixture_root / "attachments" / "sample_jd.md").is_file()
    assert (fixture_root / "attachments" / "sample_project.md").is_file()


def test_skill_uses_one_weekly_command_and_secret_env():
    skill = (ROOT / "integrations" / "openclaw" / "SKILL.md.tmpl").read_text(
        encoding="utf-8"
    )
    assert 'primaryEnv: "OPENAI_API_KEY"' in skill
    assert "profile-suggestion accept <id>" in skill
    assert "只调用一次 `weekly`" in skill
    assert "只有用户随后明确同意" in skill


def test_skill_routes_common_reads_through_the_restricted_launcher():
    skill = (ROOT / "integrations" / "openclaw" / "SKILL.md.tmpl").read_text(
        encoding="utf-8"
    )
    assert "强制命令路由（最高优先级）" in skill
    assert "{{OFFERCLAW_LAUNCHER}} profile" in skill
    assert "{{OFFERCLAW_LAUNCHER}} applications" in skill
    assert "不要先运行 `ls`、`find`、`cat`" in skill
    assert "不得用聊天记忆、周复盘摘要或猜测生成替代答案" in skill


def test_every_launcher_command_is_covered_by_the_skill_router():
    launcher = (ROOT / "integrations" / "openclaw" / "offerclaw-launcher.sh.tmpl").read_text(
        encoding="utf-8"
    )
    skill = (ROOT / "integrations" / "openclaw" / "SKILL.md.tmpl").read_text(
        encoding="utf-8"
    )
    routing = (ROOT / "integrations" / "openclaw" / "AGENTS.routing.md.tmpl").read_text(
        encoding="utf-8"
    )
    commands = set()
    for group in re.findall(r"^  ([a-z0-9_|-]+)\)$", launcher, flags=re.MULTILINE):
        commands.update(group.split("|"))
    assert commands
    internal = {"wechat-dispatch", "wechat-index-sync", "wechat-query-health"}
    assert internal <= commands
    assert all(f"{{{{OFFERCLAW_LAUNCHER}}}} {command}" not in skill for command in internal)
    assert all(f"{{{{OFFERCLAW_LAUNCHER}}}} {command}" not in routing for command in internal)
    assert commands == internal
    assert "do not try to locate or read `SKILL.md`" in routing
    assert "launcher `--help`" in routing
    assert "pending list --status pending" in routing
    assert "profile-suggestion list --status pending" in routing
    assert "when `in_kb=true`" in routing and "when `in_kb=false`" in routing
    assert "主线:补技能\\n完成:item1;item2" in routing
    assert "Do not add a diagnosis or" in routing


def test_managed_agent_routing_is_idempotent_and_preserves_local_instructions(tmp_path):
    template = ROOT / "integrations" / "openclaw" / "AGENTS.routing.md.tmpl"
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Local rules\n\nKeep this line.\n", encoding="utf-8")

    assert sync_routing(template, agents, "/safe/offerclaw-launcher") is True
    assert sync_routing(template, agents, "/safe/offerclaw-launcher") is False

    content = agents.read_text(encoding="utf-8")
    assert content.count(START) == 1
    assert content.count(END) == 1
    assert "Keep this line." in content
    assert "/safe/offerclaw-launcher" in content
