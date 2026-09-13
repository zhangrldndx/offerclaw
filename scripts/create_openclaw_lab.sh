#!/usr/bin/env bash
# Create/update a code snapshot while excluding all live personal state.
set -euo pipefail

readonly SOURCE_DIR="${OFFERCLAW_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly LAB_ROOT="${OFFERCLAW_LAB_ROOT:-$HOME/.local/share/offerclaw-lab}"
readonly DEST_DIR="$LAB_ROOT/app"
readonly FIXTURE_DIR="$SOURCE_DIR/integrations/openclaw/lab-fixtures"
readonly REAL_SOURCE_CONFIG="$LAB_ROOT/config/real-source.env"
readonly WECHAT_SCOPE_CONFIG="$LAB_ROOT/config/wechat-private-scope.json"

[[ -d "$SOURCE_DIR" ]] || { printf 'source not found: %s\n' "$SOURCE_DIR" >&2; exit 1; }
[[ -d "$FIXTURE_DIR" ]] || { printf 'fixtures not found: %s\n' "$FIXTURE_DIR" >&2; exit 1; }
case "$DEST_DIR" in
  "$HOME"/.local/share/offerclaw-lab/app) ;;
  *) printf 'refusing unexpected lab destination: %s\n' "$DEST_DIR" >&2; exit 1 ;;
esac

rm -rf -- "$DEST_DIR"
mkdir -p "$DEST_DIR"
tar -C "$SOURCE_DIR" -cf - \
  --exclude='./.git' --exclude='./.venv' --exclude='./.env' --exclude='./.env.*' \
  --exclude='./.claude' --exclude='./.offerclaw' --exclude='./.playwright-cli' \
  --exclude='./_gpt_exports' --exclude='./_local_notes' --exclude='./docs' \
  --exclude='./knowledge_base' --exclude='./profiles' --exclude='./data' \
  --exclude='./summaries' --exclude='./chroma_db' --exclude='./memory' \
  --exclude='./logs' --exclude='./output' \
  --exclude='./user_profile.md' --exclude='./daily_log.md' --exclude='./applications.md' \
  --exclude='./interview_story_bank.md' --exclude='./plans' --exclude='./application_jds' \
  --exclude='./learning_resources' --exclude='./downloads' --exclude='*/__pycache__' \
  --exclude='*/.pytest_cache' --exclude='./gap_store.json' --exclude='./growth_metrics.json' \
  --exclude='./growth_journal.md' . | tar -C "$DEST_DIR" -xf -

cp "$FIXTURE_DIR/user_profile.md" "$DEST_DIR/user_profile.md"
cp "$FIXTURE_DIR/daily_log.md" "$DEST_DIR/daily_log.md"
cp "$FIXTURE_DIR/applications.md" "$DEST_DIR/applications.md"
cp "$FIXTURE_DIR/interview_story_bank.md" "$DEST_DIR/interview_story_bank.md"
mkdir -p "$DEST_DIR/plans" "$DEST_DIR/learning_resources"
cp "$FIXTURE_DIR/plans/plan_20260901_20260928_user.md" "$DEST_DIR/plans/"
cp "$FIXTURE_DIR/learning_resources/wechat_lab_rag.md" "$DEST_DIR/learning_resources/"
cp "$FIXTURE_DIR/env.example" "$DEST_DIR/.env.local"

# Record executable/source paths only. No live personal content is copied.
windows_user_profile="$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' | tail -1)"
readonly WINDOWS_REPO_WSL="${OFFERCLAW_WINDOWS_REPO_WSL:-$SOURCE_DIR}"
windows_repo_win_raw="${OFFERCLAW_WINDOWS_REPO_WIN:-$(wslpath -w "$WINDOWS_REPO_WSL")}"
readonly WINDOWS_REPO_WIN="${windows_repo_win_raw//\\//}"
readonly WINDOWS_PYTHON="${OFFERCLAW_WINDOWS_PYTHON:-$(wslpath -u "$windows_user_profile\\.offerclaw-runtime\\wechat-bridge-venv\\Scripts\\python.exe")}"
mkdir -p "$(dirname "$REAL_SOURCE_CONFIG")"
{
  printf 'OFFERCLAW_WINDOWS_REPO_WSL=%q\n' "$WINDOWS_REPO_WSL"
  printf 'OFFERCLAW_WINDOWS_REPO_WIN=%q\n' "$WINDOWS_REPO_WIN"
  printf 'OFFERCLAW_WINDOWS_PYTHON=%q\n' "$WINDOWS_PYTHON"
} > "$REAL_SOURCE_CONFIG"
chmod 600 "$REAL_SOURCE_CONFIG"

# Keep the installed gateway entry points aligned with the code snapshot.  The
# first deployment may not have a venv yet; the launcher is still rendered and
# will become executable once setup_wechat.sh creates it.
readonly PYTHON_BIN="$LAB_ROOT/venv/bin/python"
readonly LAUNCHER_DST="$LAB_ROOT/bin/offerclaw-launcher"
readonly SKILL_DST="$LAB_ROOT/agent-workspace/skills/offerclaw/SKILL.md"
mkdir -p "$(dirname "$LAUNCHER_DST")" "$(dirname "$SKILL_DST")"
sed -e "s|{{OFFERCLAW_DIR}}|$DEST_DIR|g" \
    -e "s|{{PYTHON_BIN}}|$PYTHON_BIN|g" \
    -e "s|{{WINDOWS_PYTHON}}|$WINDOWS_PYTHON|g" \
    -e "s|{{WINDOWS_BRIDGE_SCRIPT}}|$WINDOWS_REPO_WIN/wechat_data_bridge.py|g" \
    -e "s|{{WINDOWS_REPO_WSL}}|$WINDOWS_REPO_WSL|g" \
    -e "s|{{REAL_INDEX_DIR}}|$LAB_ROOT/real-chroma|g" \
    -e "s|{{REAL_INDEX_MANIFEST}}|$LAB_ROOT/state/real-index-manifest.json|g" \
    -e "s|{{WECHAT_SCOPE_CONFIG}}|$WECHAT_SCOPE_CONFIG|g" \
    "$DEST_DIR/integrations/openclaw/offerclaw-launcher.sh.tmpl" > "$LAUNCHER_DST"
chmod 700 "$LAUNCHER_DST"
sed -e "s|{{OFFERCLAW_DIR}}|$DEST_DIR|g" \
    -e "s|{{PYTHON_BIN}}|$PYTHON_BIN|g" \
    -e "s|{{OFFERCLAW_LAUNCHER}}|$LAUNCHER_DST|g" \
    "$DEST_DIR/integrations/openclaw/SKILL.md.tmpl" > "$SKILL_DST"
  cp "$DEST_DIR/integrations/openclaw/AGENTS.public-fallback.md.tmpl" \
    "$LAB_ROOT/agent-workspace/AGENTS.md"
  printf '%s\n' '# Public Fallback Tone' 'Be concise, factual, and do not invent local context.' \
    > "$LAB_ROOT/agent-workspace/SOUL.md"
  printf '%s\n' '# Identity' 'You are the public-question fallback for OfferClaw.' \
    > "$LAB_ROOT/agent-workspace/IDENTITY.md"
  printf '%s\n' '# User Context' 'No personal profile is available to this Agent.' \
    > "$LAB_ROOT/agent-workspace/USER.md"
  chmod 600 "$LAB_ROOT/agent-workspace/AGENTS.md" "$LAB_ROOT/agent-workspace/SOUL.md" \
    "$LAB_ROOT/agent-workspace/IDENTITY.md" "$LAB_ROOT/agent-workspace/USER.md"

printf 'OfferClaw lab snapshot ready: %s\n' "$DEST_DIR"
printf 'Restricted launcher and OfferClaw Skill synchronized.\n'
printf 'No live profile, log, plan, ChromaDB, memory, or secret file was copied.\n'
