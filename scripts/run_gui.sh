#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${GEM_GUI_HOST:-127.0.0.1}"
PORT="${GEM_GUI_PORT:-8080}"
WORKSPACE="${GEM_GUI_WORKSPACE:-$ROOT_DIR}"

printf 'Starting GEM Learning Engine GUI at http://%s:%s\n' "$HOST" "$PORT"
printf 'Workspace: %s\n' "$WORKSPACE"

if command -v uv >/dev/null 2>&1; then
  exec uv run --extra optimization gem-learning-engine-gui \
    --workspace "$WORKSPACE" --host "$HOST" --port "$PORT"
elif [[ -x "$ROOT_DIR/.venv/bin/gem-learning-engine-gui" ]]; then
  exec "$ROOT_DIR/.venv/bin/gem-learning-engine-gui" \
    --workspace "$WORKSPACE" --host "$HOST" --port "$PORT"
else
  echo "ERROR: The project is not installed. Run ./scripts/setup.sh first." >&2
  exit 1
fi
