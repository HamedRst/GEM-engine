#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# GEM_GRAPHVIZ_ONECLICK_V1
if ! source "$ROOT/scripts/ensure_graphviz.sh" "$HOME/Library/Application Support/GEM Engine"; then
  echo "Graphviz renderer setup failed. Check your network connection." >&2
  exit 1
fi
# Launch a separate copy of the application. This does NOT attach to or mutate
# a live seed-43/44 optimizer process, its running jobs or its saved workspace.
export GEM_ENABLE_EXPLORATION=1
export GEM_EXPLORATION_EMBEDDED=1
HOST="${GEM_GUI_HOST:-127.0.0.1}"
PORT="${GEM_GUI_PORT:-8081}"
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" ]]; then
  echo 'Integrated Exploration is localhost-only; use a local browser or SSH tunnel.' >&2
  exit 1
fi
if command -v open >/dev/null 2>&1; then
  (sleep 2; open "http://127.0.0.1:$PORT") &
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run --extra optimization --extra exploration gem-learning-engine-gui \
      --workspace "${GEM_GUI_WORKSPACE:-$ROOT}" --host "$HOST" --port "$PORT"
elif [[ -x "$ROOT/.venv/bin/gem-learning-engine-gui" ]]; then
  exec "$ROOT/.venv/bin/gem-learning-engine-gui" \
      --workspace "${GEM_GUI_WORKSPACE:-$ROOT}" --host "$HOST" --port "$PORT"
else
  echo 'Not installed. Run INSTALL_GEM_ENGINE.command first.' >&2
  exit 1
fi
