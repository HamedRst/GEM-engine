#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if ! command -v uv >/dev/null 2>&1; then
  echo 'For automatic setup, double-click GEM Engine.app. Manual installation requires uv.' >&2
  exit 1
fi
# GEM_GRAPHVIZ_ONECLICK_V1
if ! source "$ROOT/scripts/ensure_graphviz.sh" "$HOME/Library/Application Support/GEM Engine"; then
  echo "Graphviz installation failed." >&2
  exit 1
fi
uv sync --python 3.12 --extra optimization --extra exploration
uv run --extra optimization --extra exploration python scripts/validate_gui_only.py
printf 'Installation and source verification finished. Double-click GEM Engine.app to launch.\n'
