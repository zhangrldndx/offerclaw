#!/usr/bin/env bash
# Compatibility entry point: the root script is the only automation definition.
set -euo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/setup_wechat.sh" --cron-only "$@"
