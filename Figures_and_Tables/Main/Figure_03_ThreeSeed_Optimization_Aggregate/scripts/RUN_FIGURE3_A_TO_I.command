#!/bin/bash
# Consolidated ONE generator entry point; keeps existing results and scientific source.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
bash "$HERE/SETUP_PLOT_ENV.command"
PYTHON="$HERE/.plot_venv/bin/python"
[ -x "$PYTHON" ] || { echo 'BLOCKED: plotting-only Python unavailable' >&2; exit 2; }
exec "$PYTHON" "$HERE/generate_figure3_all_panels.py" "$@"
