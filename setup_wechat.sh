#!/usr/bin/env bash
# OfferClaw + OpenClaw WeChat lab deployment. Jobs are disabled by default.
set -euo pipefail

readonly OFFERCLAW_DIR="${OFFERCLAW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
readonly OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.9.4}"
readonly NODE_VERSION="${NODE_VERSION:-26.1.0}"
readonly WEIXIN_VERSION="${WEIXIN_VERSION:-2.4.8}"
readonly TORCH_CPU_VERSION="${OFFERCLAW_TORCH_CPU_VERSION:-2.14.0+cpu}"
readonly PYPI_INDEX_URL="${OFFERCLAW_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
readonly WEIXIN_PKG="@tencent-weixin/openclaw-weixin"
readonly OPENCLAW_PROFILE="${OPENCLAW_PROFILE:-offerclaw-lab}"
readonly OPENCLAW_PROFILE_STATE_DIR="${OFFERCLAW_OPENCLAW_STATE_DIR:-$HOME/.openclaw-$OPENCLAW_PROFILE}"
readonly INSTALL_NAME="${OFFERCLAW_INSTALL_NAME:-offerclaw-lab}"
readonly LAB_ROOT="${OFFERCLAW_LAB_ROOT:-$HOME/.local/share/$INSTALL_NAME}"
readonly REAL_SOURCE_CONFIG="$LAB_ROOT/config/real-source.env"
if [[ -f "$REAL_SOURCE_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$REAL_SOURCE_CONFIG"
fi
readonly WINDOWS_REPO_WSL="${OFFERCLAW_WINDOWS_REPO_WSL:-}"
readonly WINDOWS_REPO_WIN="${OFFERCLAW_WINDOWS_REPO_WIN:-}"
readonly WINDOWS_PYTHON="${OFFERCLAW_WINDOWS_PYTHON:-}"
readonly OPENCLAW_PREFIX="${OPENCLAW_PREFIX:-$LAB_ROOT/openclaw-cli}"
readonly OPENCLAW_BIN="${OPENCLAW_BIN:-$OPENCLAW_PREFIX/bin/openclaw}"
readonly NODE_BIN="${OPENCLAW_NODE_BIN:-$OPENCLAW_PREFIX/tools/node-v$NODE_VERSION/bin/node}"
readonly VENV_DIR="${OFFERCLAW_VENV_DIR:-$LAB_ROOT/venv}"
readonly PYTHON_BIN="${OFFERCLAW_PYTHON_BIN:-$VENV_DIR/bin/python}"
readonly AGENT_ID="${OFFERCLAW_AGENT_ID:-offerclaw-lab}"
readonly AGENT_WORKSPACE="${OFFERCLAW_AGENT_WORKSPACE:-$LAB_ROOT/agent-workspace}"
readonly SKILL_TMPL="$OFFERCLAW_DIR/integrations/openclaw/SKILL.md.tmpl"
readonly LAUNCHER_TMPL="$OFFERCLAW_DIR/integrations/openclaw/offerclaw-launcher.sh.tmpl"
readonly AGENT_POLICY_TMPL="$OFFERCLAW_DIR/integrations/openclaw/AGENTS.public-fallback.md.tmpl"
readonly DIRECT_PLUGIN_DIR="$OFFERCLAW_DIR/integrations/openclaw/offerclaw-direct-reply"
readonly DIRECT_PREFLIGHT="$OFFERCLAW_DIR/scripts/openclaw_direct_reply_preflight.py"
readonly SKILL_DST="$AGENT_WORKSPACE/skills/offerclaw/SKILL.md"
readonly LAUNCHER_DST="$LAB_ROOT/bin/offerclaw-launcher"
readonly WECHAT_SCOPE_CONFIG="$LAB_ROOT/config/wechat-private-scope.json"
readonly AGENT_INSTRUCTIONS_DST="$AGENT_WORKSPACE/AGENTS.md"
readonly JOB_PREFIX="${OFFERCLAW_JOB_PREFIX:-offerclaw-lab}"
readonly TIMEZONE="Asia/Shanghai"
readonly MODEL_PROVIDER_ID="${OFFERCLAW_MODEL_PROVIDER_ID:-offerclaw-local}"
readonly MODEL_ID="${OFFERCLAW_MODEL_ID:-}"
readonly MODEL_NAME="${OFFERCLAW_MODEL_NAME:-$MODEL_ID}"
readonly MODEL_CONTEXT_WINDOW="${OFFERCLAW_MODEL_CONTEXT_WINDOW:-131072}"
readonly MODEL_MAX_TOKENS="${OFFERCLAW_MODEL_MAX_TOKENS:-8192}"
readonly MODEL_REF="${OFFERCLAW_MODEL_REF:-$MODEL_PROVIDER_ID/$MODEL_ID}"
readonly MODEL_BASE_URL="${OFFERCLAW_MODEL_BASE_URL:-}"
readonly GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-19051}"

DRY_RUN=0
CRON_ONLY=0
ENABLE_JOBS=0
PREFLIGHT_DIRECT_REPLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --cron-only) CRON_ONLY=1 ;;
    --enable-jobs) ENABLE_JOBS=1 ;;
    --preflight-direct-reply) PREFLIGHT_DIRECT_REPLY=1 ;;
    -h|--help)
      printf 'Usage: bash setup_wechat.sh [--dry-run] [--cron-only] [--enable-jobs] [--preflight-direct-reply]\n'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 64 ;;
  esac
done

if [[ $PREFLIGHT_DIRECT_REPLY -eq 0 && $CRON_ONLY -eq 0 ]]; then
  if [[ -z "$MODEL_BASE_URL" || -z "$MODEL_ID" ]]; then
    printf '[offerclaw-wechat] WARNING: set OFFERCLAW_MODEL_BASE_URL and OFFERCLAW_MODEL_ID locally\n' >&2
    exit 64
  fi
fi

say() { printf '[offerclaw-wechat] %s\n' "$*"; }
warn() { printf '[offerclaw-wechat] WARNING: %s\n' "$*" >&2; }
quote_cmd() { printf '%q ' "$@"; printf '\n'; }
run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[offerclaw-wechat] DRY-RUN: '
    quote_cmd "$@"
  else
    "$@"
  fi
}
oc() {
  local node_dir
  node_dir="$(dirname "$NODE_BIN")"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[offerclaw-wechat] DRY-RUN: '
    quote_cmd env "PATH=$node_dir:$PATH" "$OPENCLAW_BIN" --profile "$OPENCLAW_PROFILE" "$@"
  else
    env "PATH=$node_dir:$PATH" "$OPENCLAW_BIN" --profile "$OPENCLAW_PROFILE" "$@"
  fi
}

install_openclaw() {
  if [[ -x "$OPENCLAW_BIN" ]]; then
    local installed
    installed="$($OPENCLAW_BIN --version | head -1)"
    [[ "$installed" == *"$OPENCLAW_VERSION"* ]] || {
      warn "Expected OpenClaw $OPENCLAW_VERSION, found: $installed"
      exit 1
    }
    [[ -x "$NODE_BIN" ]] || { warn "Private Node binary missing: $NODE_BIN"; exit 1; }
    [[ "$($NODE_BIN --version)" == "v$NODE_VERSION" ]] || {
      warn "Expected Node v$NODE_VERSION, found: $($NODE_BIN --version)"
      exit 1
    }
    say "OpenClaw already installed: $installed; Node $($NODE_BIN --version)"
    return
  fi
  say "Installing OpenClaw $OPENCLAW_VERSION with private Node $NODE_VERSION"
  if [[ $DRY_RUN -eq 1 ]]; then
    say "DRY-RUN: official install-cli.sh -> $OPENCLAW_PREFIX"
    return
  fi
  curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install-cli.sh \
    | bash -s -- --prefix "$OPENCLAW_PREFIX" --version "$OPENCLAW_VERSION" \
      --node-version "$NODE_VERSION" --no-onboard
}

ensure_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    run python3 -m venv "$VENV_DIR"
  fi
  local requirement_hash marker="$VENV_DIR/.offerclaw-requirements.sha256"
  requirement_hash="$(printf '%s  %s\n' "$TORCH_CPU_VERSION" \
    "$(sha256sum "$OFFERCLAW_DIR/requirements.txt" | cut -d' ' -f1)" | sha256sum | cut -d' ' -f1)"
  if [[ ! -f "$marker" || "$(<"$marker")" != "$requirement_hash" ]]; then
    run "$VENV_DIR/bin/pip" install "torch==$TORCH_CPU_VERSION" \
      --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 5 \
      --progress-bar off
    run "$VENV_DIR/bin/pip" install -r "$OFFERCLAW_DIR/requirements.txt" \
      --index-url "$PYPI_INDEX_URL" --timeout 120 --retries 5 --progress-bar off
    if [[ $DRY_RUN -eq 0 ]]; then
      printf '%s\n' "$requirement_hash" > "$marker"
    fi
  fi
}

render_integration() {
  [[ -f "$SKILL_TMPL" ]] || { warn "Missing $SKILL_TMPL"; exit 1; }
  [[ -f "$LAUNCHER_TMPL" ]] || { warn "Missing $LAUNCHER_TMPL"; exit 1; }
  [[ -f "$AGENT_POLICY_TMPL" ]] || { warn "Missing $AGENT_POLICY_TMPL"; exit 1; }
  [[ -x "$WINDOWS_PYTHON" ]] || { warn "Windows bridge Python missing: $WINDOWS_PYTHON"; exit 1; }
  [[ -d "$WINDOWS_REPO_WSL" ]] || { warn "Windows OfferClaw source missing: $WINDOWS_REPO_WSL"; exit 1; }
  if [[ $DRY_RUN -eq 1 ]]; then
    say "DRY-RUN: render restricted launcher -> $LAUNCHER_DST"
    say "DRY-RUN: render skill -> $SKILL_DST"
    say "DRY-RUN: render minimal public fallback policy -> $AGENT_INSTRUCTIONS_DST"
    return
  fi
  mkdir -p "$(dirname "$LAUNCHER_DST")" "$(dirname "$SKILL_DST")"
  sed -e "s|{{OFFERCLAW_DIR}}|$OFFERCLAW_DIR|g" \
      -e "s|{{PYTHON_BIN}}|$PYTHON_BIN|g" \
      -e "s|{{WINDOWS_PYTHON}}|$WINDOWS_PYTHON|g" \
      -e "s|{{WINDOWS_BRIDGE_SCRIPT}}|$WINDOWS_REPO_WIN/wechat_data_bridge.py|g" \
      -e "s|{{WINDOWS_REPO_WSL}}|$WINDOWS_REPO_WSL|g" \
      -e "s|{{REAL_INDEX_DIR}}|$LAB_ROOT/real-chroma|g" \
      -e "s|{{REAL_INDEX_MANIFEST}}|$LAB_ROOT/state/real-index-manifest.json|g" \
      -e "s|{{WECHAT_SCOPE_CONFIG}}|$WECHAT_SCOPE_CONFIG|g" \
      "$LAUNCHER_TMPL" > "$LAUNCHER_DST"
  chmod 700 "$LAUNCHER_DST"
  sed -e "s|{{OFFERCLAW_DIR}}|$OFFERCLAW_DIR|g" \
      -e "s|{{PYTHON_BIN}}|$PYTHON_BIN|g" \
      -e "s|{{OFFERCLAW_LAUNCHER}}|$LAUNCHER_DST|g" \
      "$SKILL_TMPL" > "$SKILL_DST"
  cp "$AGENT_POLICY_TMPL" "$AGENT_INSTRUCTIONS_DST"
  printf '%s\n' '# Public Fallback Tone' 'Be concise, factual, and do not invent local context.' \
    > "$AGENT_WORKSPACE/SOUL.md"
  printf '%s\n' '# Identity' 'You are the public-question fallback for OfferClaw.' \
    > "$AGENT_WORKSPACE/IDENTITY.md"
  printf '%s\n' '# User Context' 'No personal profile is available to this Agent.' \
    > "$AGENT_WORKSPACE/USER.md"
  chmod 600 "$AGENT_INSTRUCTIONS_DST" "$AGENT_WORKSPACE/SOUL.md" \
    "$AGENT_WORKSPACE/IDENTITY.md" "$AGENT_WORKSPACE/USER.md"
}

ensure_agent() {
  if [[ $DRY_RUN -eq 1 ]]; then
    oc agents add "$AGENT_ID" --workspace "$AGENT_WORKSPACE" --non-interactive
  else
    if ! oc agents list --json | "$PYTHON_BIN" -c \
        'import json,sys; d=json.load(sys.stdin); a=d if isinstance(d,list) else d.get("agents",[]); raise SystemExit(0 if any(x.get("id")==sys.argv[1] for x in a) else 1)' \
        "$AGENT_ID"; then
      oc agents add "$AGENT_ID" --workspace "$AGENT_WORKSPACE" --non-interactive
    fi
  fi
  oc config set "agents.entries.$AGENT_ID.model" "\"$MODEL_REF\"" --strict-json
  oc config set "agents.entries.$AGENT_ID.skills" '[]' --strict-json
  oc config set "agents.entries.$AGENT_ID.tools.allow" '["web_search","web_fetch"]' --strict-json
  oc config set "agents.entries.$AGENT_ID.tools.deny" \
    '["exec","ls","read","edit","write","apply_patch","process","terminal","file_fetch","dir_list","dir_fetch","file_write","memory_search","memory_get","secrets"]' --strict-json
}

configure_runtime() {
  local provider_json
  provider_json="$(printf \
    '{"baseUrl":"%s","api":"openai-completions","models":[{"id":"%s","name":"%s","input":["text"],"contextWindow":%s,"maxTokens":%s}]}' \
    "$MODEL_BASE_URL" "$MODEL_ID" "$MODEL_NAME" "$MODEL_CONTEXT_WINDOW" "$MODEL_MAX_TOKENS")"

  oc config set gateway.port "$GATEWAY_PORT" --strict-json
  oc config set gateway.bind '"loopback"' --strict-json
  oc config set secrets.providers.default --provider-source store
  oc config set "models.providers.$MODEL_PROVIDER_ID" "$provider_json" --strict-json --replace
  oc config set "models.providers.$MODEL_PROVIDER_ID.apiKey" \
    --ref-provider default --ref-source store --ref-id OPENAI_API_KEY
  oc config set skills.entries.offerclaw.apiKey \
    --ref-provider default --ref-source store --ref-id OPENAI_API_KEY
  oc config set tools.allow '["web_search","web_fetch"]' --strict-json
  oc config set tools.deny \
    '["exec","ls","read","edit","write","apply_patch","process","terminal","file_fetch","dir_list","dir_fetch","file_write","memory_search","memory_get","secrets"]' --strict-json
  oc config set tools.exec.security '"allowlist"' --strict-json
  oc config set tools.exec.ask '"off"' --strict-json
  oc config set skills.workshop.autonomous.mode '"off"' --strict-json
  oc config set plugins.entries.memory-core.config.dreaming.enabled false --strict-json
  oc config set agents.defaults.heartbeat.every '"0m"' --strict-json
}

plugin_version() {
  oc plugins list --json | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
rows=d if isinstance(d,list) else d.get("plugins",[])
row=next((x for x in rows if x.get("id")=="openclaw-weixin"),None)
print((row or {}).get("version",""))
'
}

configure_weixin() {
  local installed=""
  [[ $DRY_RUN -eq 1 ]] || installed="$(plugin_version)"
  if [[ "$installed" == "$WEIXIN_VERSION" ]]; then
    say "WeChat plugin already locked to $WEIXIN_VERSION"
  else
    say "Installing and locking WeChat plugin to $WEIXIN_VERSION"
    oc plugins install "$WEIXIN_PKG@$WEIXIN_VERSION" --force --pin \
      --accept-capabilities --acknowledge-install-policy-warning
  fi
  oc config set plugins.entries.openclaw-weixin.enabled true --strict-json
  oc config set channels.openclaw-weixin.enabled false --strict-json
  oc config set channels.openclaw-weixin.dmPolicy '"pairing"' --strict-json
  oc config set channels.openclaw-weixin.groupPolicy '"disabled"' --strict-json
  oc config set session.dmScope '"per-account-channel-peer"' --strict-json
  if [[ $DRY_RUN -eq 1 ]]; then
    oc agents bind --agent "$AGENT_ID" --bind openclaw-weixin
  else
    local bindings
    bindings="$(oc agents bindings --json)"
    if ! printf '%s' "$bindings" | "$PYTHON_BIN" -c '
import json,sys
rows=json.load(sys.stdin)
raise SystemExit(0 if any(
    x.get("agentId")==sys.argv[1]
    and x.get("match", {}).get("channel")=="openclaw-weixin"
    for x in rows
) else 1)
' "$AGENT_ID"; then
      oc agents bind --agent "$AGENT_ID" --bind openclaw-weixin
    fi
  fi
}

bound_weixin_accounts() {
  oc agents bindings --json | "$PYTHON_BIN" -c '
import json,sys
for row in json.load(sys.stdin):
    match=row.get("match",{})
    if row.get("agentId")==sys.argv[1] and match.get("channel")=="openclaw-weixin" and match.get("accountId"):
        print(match["accountId"])
' "$AGENT_ID"
}

bound_weixin_senders() {
  local accounts_json="$1"
  if [[ -n "${OFFERCLAW_WECHAT_SENDER_IDS:-}" ]]; then
    printf '%s' "$OFFERCLAW_WECHAT_SENDER_IDS" | "$PYTHON_BIN" -c '
import json,sys
print(json.dumps(sorted({x.strip() for x in sys.stdin.read().split(",") if x.strip()})))
'
    return
  fi
  "$PYTHON_BIN" -c '
import json,re,sys
from pathlib import Path
state_dir=Path(sys.argv[1])
accounts=set(json.loads(sys.argv[2]))
senders=set()
for account in accounts:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
        continue
    path=state_dir / "openclaw-weixin" / "accounts" / f"{account}.context-tokens.json"
    try:
        rows=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        continue
    if isinstance(rows,dict):
        senders.update(k for k,v in rows.items()
                       if isinstance(k,str) and k.endswith("@im.wechat")
                       and isinstance(v,str) and v)
print(json.dumps(sorted(senders)))
' "$OPENCLAW_PROFILE_STATE_DIR" "$accounts_json"
}

configure_direct_reply() {
  [[ -d "$DIRECT_PLUGIN_DIR" ]] || { warn "Missing direct reply plugin"; exit 1; }
  [[ -f "$DIRECT_PREFLIGHT" ]] || { warn "Missing direct reply preflight"; exit 1; }
  oc plugins install --link "$DIRECT_PLUGIN_DIR" --force \
    --accept-capabilities --acknowledge-install-policy-warning
  local accounts_json
  accounts_json="$(bound_weixin_accounts | "$PYTHON_BIN" -c \
    'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))')"
  local account_count
  account_count="$(printf '%s' "$accounts_json" | "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  if [[ "$account_count" != "1" ]]; then
    warn "Expected exactly one account-specific WeChat binding; direct reply plugin remains disabled"
    oc config set plugins.entries.offerclaw-direct-reply.enabled false --strict-json
    oc config set channels.openclaw-weixin.enabled false --strict-json
    return 2
  fi
  local senders_json sender_count
  senders_json="$(bound_weixin_senders "$accounts_json")"
  sender_count="$(printf '%s' "$senders_json" | "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  if [[ "$sender_count" != "1" ]]; then
    warn "Expected exactly one paired private sender, found $sender_count; direct reply plugin remains disabled"
    warn "Set OFFERCLAW_WECHAT_SENDER_IDS explicitly after verifying the private sender identity"
    oc config set plugins.entries.offerclaw-direct-reply.enabled false --strict-json
    oc config set channels.openclaw-weixin.enabled false --strict-json
    return 2
  fi
  if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$(dirname "$WECHAT_SCOPE_CONFIG")"
    printf '%s\n%s\n' "$accounts_json" "$senders_json" | "$PYTHON_BIN" -c '
import json,os,re,sys,tempfile
from pathlib import Path
target=Path(sys.argv[1])
lines=sys.stdin.read().splitlines()
if len(lines) != 2:
    raise SystemExit(1)
accounts=json.loads(lines[0]); senders=json.loads(lines[1])
valid=lambda rows: (isinstance(rows,list) and len(rows)==1
                    and isinstance(rows[0],str)
                    and re.fullmatch(r"[A-Za-z0-9_.@-]{1,256}", rows[0]))
if not valid(accounts) or not valid(senders):
    raise SystemExit(1)
fd,tmp=tempfile.mkstemp(prefix=".wechat-private-scope-", dir=str(target.parent))
try:
    with os.fdopen(fd,"w",encoding="utf-8") as stream:
        json.dump({"account_ids":accounts,"sender_ids":senders},stream,separators=(",",":"))
        stream.flush(); os.fsync(stream.fileno())
    os.chmod(tmp,0o600)
    os.replace(tmp,target)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
' "$WECHAT_SCOPE_CONFIG"
    chmod 600 "$WECHAT_SCOPE_CONFIG"
  fi
  local plugin_json
  plugin_json="$(printf '{"launcherPath":"%s","accountIds":%s,"senderIds":%s,"timeoutMs":45000,"publicLlmFallback":true}' \
    "$LAUNCHER_DST" "$accounts_json" "$senders_json")"
  oc config set plugins.entries.offerclaw-direct-reply.config "$plugin_json" --strict-json --replace
  oc config set plugins.entries.offerclaw-direct-reply.hooks.allowConversationAccess true --strict-json
  oc config set plugins.entries.offerclaw-direct-reply.enabled true --strict-json
}

preflight_direct_reply() {
  [[ -x "$OPENCLAW_BIN" ]] || { warn "OpenClaw executable is unavailable"; return 1; }
  [[ -x "$PYTHON_BIN" ]] || { warn "OfferClaw Python is unavailable"; return 1; }
  [[ -f "$DIRECT_PREFLIGHT" ]] || { warn "Direct reply preflight is unavailable"; return 1; }
  local inspect_json
  inspect_json="$(oc plugins inspect offerclaw-direct-reply --runtime --json)" || {
    warn "Could not inspect direct reply plugin"
    return 1
  }
  printf '%s' "$inspect_json" | "$PYTHON_BIN" "$DIRECT_PREFLIGHT" \
    --state-dir "$OPENCLAW_PROFILE_STATE_DIR" --launcher "$LAUNCHER_DST" \
    --scope-config "$WECHAT_SCOPE_CONFIG" || return 1
  local health_json expected_fingerprint
  health_json="$("$LAUNCHER_DST" wechat-query-health)" || {
    warn "Windows query service authenticated health check failed"
    return 1
  }
  expected_fingerprint="$(cd "$OFFERCLAW_DIR" && "$PYTHON_BIN" -c \
    'from query_service import repository_fingerprint; print(repository_fingerprint())')"
  printf '%s' "$health_json" | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
ok=(d.get("status")=="ok"
    and d.get("query_service_version")=="offerclaw.query-service.v1"
    and d.get("repository_fingerprint")==sys.argv[1]
    and d.get("authentication")=="loopback_token"
    and (d.get("runtime") or {}).get("status")=="ready")
raise SystemExit(0 if ok else 1)
' "$expected_fingerprint" || {
    warn "Windows query service version, authentication, or repository fingerprint mismatch"
    return 1
  }
}

remove_exec_allowlist() {
  if [[ $DRY_RUN -eq 1 ]]; then
    oc approvals allowlist remove --agent "$AGENT_ID" "$LAUNCHER_DST"
    return
  fi
  local existing
  existing="$(oc approvals get --json)"
  if printf '%s' "$existing" | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin).get("file",{}).get("agents",{})
patterns=[x.get("pattern","") for x in d.get(sys.argv[1],{}).get("allowlist",[])]
raise SystemExit(0 if sys.argv[2] in patterns else 1)
' "$AGENT_ID" "$LAUNCHER_DST"; then
    oc approvals allowlist remove --agent "$AGENT_ID" "$LAUNCHER_DST"
  fi
}

harden_runtime_permissions() {
  if [[ $DRY_RUN -eq 1 ]]; then
    say "DRY-RUN: install systemd UMask=0077, direct-reply preflight, and private state permissions"
    return
  fi
  local unit_dropin_dir="$HOME/.config/systemd/user/openclaw-gateway-$OPENCLAW_PROFILE.service.d"
  install -d -m 700 "$unit_dropin_dir"
  printf '[Service]\nUMask=0077\n' > "$unit_dropin_dir/10-offerclaw-private.conf"
  chmod 600 "$unit_dropin_dir/10-offerclaw-private.conf"
  printf '[Service]\nExecStartPre=/bin/bash %s --preflight-direct-reply\n' \
    "$OFFERCLAW_DIR/setup_wechat.sh" > "$unit_dropin_dir/20-offerclaw-direct-reply-preflight.conf"
  chmod 600 "$unit_dropin_dir/20-offerclaw-direct-reply-preflight.conf"

  chmod 700 "$OPENCLAW_PROFILE_STATE_DIR" "$LAB_ROOT" "$LAB_ROOT/config" "$LAB_ROOT/state" 2>/dev/null || true
  chmod 600 "$OPENCLAW_PROFILE_STATE_DIR/openclaw.json" 2>/dev/null || true
  local private_dir
  for private_dir in \
      "$OPENCLAW_PROFILE_STATE_DIR/state" \
      "$OPENCLAW_PROFILE_STATE_DIR/openclaw-weixin/accounts" \
      "$LAB_ROOT/config" \
      "$LAB_ROOT/state" \
      "$LAB_ROOT/real-chroma" \
      "$LAB_ROOT/app/.offerclaw"; do
    [[ -d "$private_dir" ]] || continue
    chmod 700 "$private_dir"
    find "$private_dir" -type d -exec chmod 700 {} +
    find "$private_dir" -type f -exec chmod 600 {} +
  done

  local runtime_log_dir="/tmp/openclaw"
  if [[ -d "$runtime_log_dir" && -O "$runtime_log_dir" ]]; then
    chmod 700 "$runtime_log_dir"
    find "$runtime_log_dir" -type d -exec chmod 700 {} +
    find "$runtime_log_dir" -type f -exec chmod 600 {} +
  fi
  systemctl --user daemon-reload
}

index_is_healthy() {
  [[ $DRY_RUN -eq 1 ]] && return 1
  "$LAUNCHER_DST" health | "$PYTHON_BIN" -c '
import json,sys
raise SystemExit(0 if json.load(sys.stdin).get("status")=="healthy" else 1)
'
}

ensure_index() {
  if index_is_healthy; then
    say "Synthetic ChromaDB collection is healthy"
    return
  fi
  say "Initializing the isolated synthetic ChromaDB collection"
  run "$PYTHON_BIN" "$OFFERCLAW_DIR/rag_ingest.py" --rebuild
  run "$PYTHON_BIN" "$OFFERCLAW_DIR/rag_ingest.py" \
    --add learning_resources/wechat_lab_rag.md --source-type resource --replace
  if [[ $DRY_RUN -eq 0 ]]; then
    index_is_healthy || { warn "Synthetic ChromaDB initialization failed"; exit 1; }
  fi
}

sync_state_sources() {
  if [[ $DRY_RUN -eq 1 ]]; then
    run "$LAUNCHER_DST" refresh-state
    return
  fi
  local result
  result="$("$LAUNCHER_DST" refresh-state)"
  printf '%s\n' "$result"
  if ! printf '%s' "$result" | "$PYTHON_BIN" -c '
import json,sys
raise SystemExit(0 if json.load(sys.stdin).get("status")=="ok" else 1)
'; then
    warn "One or more synthetic state sources failed to refresh; old chunks remain intact"
    exit 1
  fi
}

secret_exists() {
  oc secrets store list --json | "$PYTHON_BIN" -c '
import json,sys
rows=json.load(sys.stdin)
raise SystemExit(0 if any(x.get("name")=="OPENAI_API_KEY" for x in rows) else 1)
'
}

ensure_gateway() {
  if [[ $DRY_RUN -eq 1 ]]; then
    oc gateway install --port "$GATEWAY_PORT"
    return
  fi
  if ! secret_exists; then
    warn "OPENAI_API_KEY is not in Secret Store; gateway start is deferred"
    return
  fi
  local status_json
  status_json="$(oc gateway status --json)"
  if printf '%s' "$status_json" | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
raise SystemExit(0 if d.get("rpc",{}).get("ok") else 1)
'; then
    say "Gateway is already reachable on port $GATEWAY_PORT"
    return
  fi
  if printf '%s' "$status_json" | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
raise SystemExit(0 if d.get("service",{}).get("runtime",{}).get("missingUnit") else 1)
'; then
    oc gateway install --port "$GATEWAY_PORT"
  else
    oc gateway restart
  fi
  local attempt
  for attempt in {1..15}; do
    status_json="$(oc gateway status --json)"
    if printf '%s' "$status_json" | "$PYTHON_BIN" -c '
import json,sys
raise SystemExit(0 if json.load(sys.stdin).get("rpc",{}).get("ok") else 1)
'; then
      say "Gateway RPC is ready on port $GATEWAY_PORT"
      return
    fi
    sleep 2
  done
  oc gateway status --require-rpc --timeout 30000
}

bind_weixin_accounts() {
  if [[ $DRY_RUN -eq 1 ]]; then
    oc agents bind --agent "$AGENT_ID" --bind 'openclaw-weixin:<account-id-after-login>'
    return
  fi
  local bindings account_id
  local -a account_ids=()
  mapfile -t account_ids < <("$PYTHON_BIN" -c '
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]) / "openclaw-weixin" / "accounts"
for path in sorted(root.glob("*.json")):
    if path.name.endswith((".context-tokens.json", ".sync.json")):
        continue
    account_id=path.stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", account_id):
        continue
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        continue
    if isinstance(data,dict):
        print(account_id)
' "$OPENCLAW_PROFILE_STATE_DIR")
  if [[ ${#account_ids[@]} -eq 0 ]]; then
    say "No authenticated WeChat account yet; account binding is deferred"
    return
  fi
  bindings="$(oc agents bindings --json)"
  for account_id in "${account_ids[@]}"; do
    if printf '%s' "$bindings" | "$PYTHON_BIN" -c '
import json,sys
rows=json.load(sys.stdin)
raise SystemExit(0 if any(
    x.get("agentId")==sys.argv[1]
    and x.get("match",{}).get("channel")=="openclaw-weixin"
    and x.get("match",{}).get("accountId")==sys.argv[2]
    for x in rows
) else 1)
' "$AGENT_ID" "$account_id"; then
      say "WeChat account already bound to $AGENT_ID: $account_id"
    else
      oc agents bind --agent "$AGENT_ID" --bind "openclaw-weixin:$account_id"
    fi
  done
}

stop_gateway_for_direct_reply() {
  if [[ $DRY_RUN -eq 1 ]]; then
    oc gateway stop --force
    return
  fi
  oc gateway stop --force >/dev/null
  local status_json
  status_json="$(oc gateway status --json)"
  if printf '%s' "$status_json" | "$PYTHON_BIN" -c '
import json,sys
raise SystemExit(0 if json.load(sys.stdin).get("rpc",{}).get("ok") else 1)
'; then
    warn "Gateway RPC is still reachable after stop"
    return 1
  fi
  say "Gateway stopped while direct-reply capabilities are verified"
}

find_job_id() {
  local name="$1"
  oc automations list --all --json | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
rows=d if isinstance(d,list) else d.get("jobs",d.get("automations",d.get("items",[])))
row=next((x for x in rows if x.get("name")==sys.argv[1]),None)
print((row or {}).get("id",(row or {}).get("jobId","")))
' "$name"
}

job_needs_command_payload() {
  local name="$1"
  oc automations list --all --json | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
rows=d if isinstance(d,list) else d.get("jobs",d.get("automations",d.get("items",[])))
row=next((x for x in rows if x.get("name")==sys.argv[1]),None)
payload=(row or {}).get("payload",{})
raise SystemExit(0 if payload.get("kind") != "command" else 1)
' "$name"
}

resolved_private_target() {
  if [[ -n "${OFFERCLAW_WECHAT_TO:-}" ]]; then
    printf '%s\n' "$OFFERCLAW_WECHAT_TO"
    return
  fi
  local accounts_json senders_json
  accounts_json="$(bound_weixin_accounts | "$PYTHON_BIN" -c \
    'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))')"
  senders_json="$(bound_weixin_senders "$accounts_json")"
  printf '%s' "$senders_json" | "$PYTHON_BIN" -c '
import json,sys
rows=json.load(sys.stdin)
print(rows[0] if len(rows)==1 else "")
'
}

resolved_delivery_account() {
  local accounts_json
  accounts_json="$(bound_weixin_accounts | "$PYTHON_BIN" -c \
    'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))')"
  printf '%s' "$accounts_json" | "$PYTHON_BIN" -c '
import json,sys
rows=json.load(sys.stdin)
print(rows[0] if len(rows)==1 else "")
'
}

sync_job() {
  local name="$1" schedule="$2" description="$3" automation_kind="$4"
  local target="${OFFERCLAW_WECHAT_TO:-}"
  local delivery_account=""
  if [[ $DRY_RUN -eq 1 ]]; then
    delivery_account="<openclaw-weixin-account-id>"
  else
    delivery_account="$(resolved_delivery_account)"
    if [[ -z "$delivery_account" ]]; then
      warn "No unique bound WeChat account is available; no job was changed"
      return 2
    fi
  fi
  if [[ -z "$target" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      target="<openclaw-weixin-private-target>"
    else
      target="$(resolved_private_target)"
      if [[ -z "$target" ]]; then
        warn "No unique private WeChat target is available; set OFFERCLAW_WECHAT_TO after verification"
        return 2
      fi
    fi
  fi

  local marker="offerclaw://automation/$automation_kind?v=1"
  local command_argv request_json
  command_argv="$("$PYTHON_BIN" -c '
import json,sys
print(json.dumps([sys.argv[1], "wechat-dispatch", "--stdin", "--reply-text"], separators=(",", ":")))
' "$LAUNCHER_DST")"
  request_json="$("$PYTHON_BIN" -c '
import json,sys
kind,account,marker=sys.argv[1:]
print(json.dumps({
    "schema_version":"offerclaw.wechat.request.v1",
    "message_id":f"automation:{kind}:v1",
    "conversation_id":f"automation:{kind}",
    "channel":"openclaw-weixin",
    "account_id":account,
    "sender_id":"automation",
    "is_group":False,
    "text":marker,
    "media":[],
    "trigger":"cron",
    "automation_kind":kind,
}, ensure_ascii=False, separators=(",", ":")))
' "$automation_kind" "$delivery_account" "$marker")"

  local id=""
  [[ $DRY_RUN -eq 1 ]] || id="$(find_job_id "$name")"
  if [[ -n "$id" && $DRY_RUN -eq 0 ]] && job_needs_command_payload "$name"; then
    say "Recreating automation as a zero-model command job: $name ($id)"
    oc automations delete "$id"
    id=""
  fi
  local common=(--name "$name" --cron "$schedule" --tz "$TIMEZONE" --exact
    --session isolated --agent "$AGENT_ID" --announce
    --command-argv "$command_argv" --command-input "$request_json"
    --timeout-seconds 45 --no-output-timeout-seconds 45 --output-max-bytes 4096
    --channel openclaw-weixin --account "$delivery_account" --to "$target" --description "$description"
  )
  if [[ -n "$id" ]]; then
    say "Updating exact automation name: $name ($id)"
    oc automations edit "$id" "${common[@]}"
  else
    say "Creating automation: $name"
    oc automations add "${common[@]}" --disabled
    [[ $DRY_RUN -eq 1 ]] || id="$(find_job_id "$name")"
  fi
  if [[ $ENABLE_JOBS -eq 1 ]]; then
    [[ -z "$id" && $DRY_RUN -eq 1 ]] || oc automations enable "${id:-$name}"
  else
    [[ -z "$id" && $DRY_RUN -eq 1 ]] || oc automations disable "${id:-$name}"
  fi
}

configure_jobs() {
  local rc=0
  sync_job "$JOB_PREFIX-morning" "0 9 * * 1-5" "工作日早间今日建议" \
    "morning" || rc=$?
  sync_job "$JOB_PREFIX-evening" "0 22 * * *" "每日晚间留痕提醒" \
    "evening" || rc=$?
  sync_job "$JOB_PREFIX-weekly" "0 21 * * 0" "周日晚间确定性周闭环" \
    "weekly" || rc=$?
  return "$rc"
}

if [[ $PREFLIGHT_DIRECT_REPLY -eq 1 ]]; then
  preflight_direct_reply
  exit $?
fi

DIRECT_REPLY_READY=1
if [[ $CRON_ONLY -eq 0 ]]; then
  install_openclaw
  stop_gateway_for_direct_reply
  ensure_venv
  render_integration
  configure_runtime
  ensure_agent
  remove_exec_allowlist
  configure_weixin
  run "$LAUNCHER_DST" wechat-index-sync
  if [[ $DRY_RUN -eq 0 ]]; then
    oc config validate
  fi
  bind_weixin_accounts
  if ! configure_direct_reply; then
    DIRECT_REPLY_READY=0
  fi
  harden_runtime_permissions
  if [[ $DIRECT_REPLY_READY -eq 1 ]]; then
    if ! preflight_direct_reply; then
      oc config set channels.openclaw-weixin.enabled false --strict-json
      warn "Direct reply preflight failed; gateway and WeChat channel remain disabled"
      exit 1
    fi
    oc config set channels.openclaw-weixin.enabled true --strict-json
    oc config validate
    ensure_gateway
    if ! preflight_direct_reply; then
      oc config set channels.openclaw-weixin.enabled false --strict-json
      oc gateway stop >/dev/null 2>&1 || true
      warn "Post-start direct reply inspection failed; gateway and WeChat channel were stopped"
      exit 1
    fi
  else
    warn "Direct reply preflight was not attempted; gateway and WeChat channel remain disabled"
  fi
fi

if [[ $DIRECT_REPLY_READY -eq 1 ]] && configure_jobs; then
  say "Schedules use $TIMEZONE; jobs are $([[ $ENABLE_JOBS -eq 1 ]] && echo enabled || echo disabled)."
else
  say "No jobs were changed because a private WeChat target is not configured."
fi
if [[ $DRY_RUN -eq 0 ]] && ! secret_exists; then
  say "Manual step 1: $OPENCLAW_BIN --profile $OPENCLAW_PROFILE secrets store set OPENAI_API_KEY"
fi
if [[ $DRY_RUN -eq 0 && -z "$(resolved_private_target)" ]]; then
  say "Manual step 2: $OPENCLAW_BIN --profile $OPENCLAW_PROFILE channels login --channel openclaw-weixin"
fi
