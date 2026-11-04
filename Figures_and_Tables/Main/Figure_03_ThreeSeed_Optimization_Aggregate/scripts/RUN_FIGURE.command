#!/bin/bash
# Portable one-click launcher; build private plotting-only env on first run.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
bash "$HERE/SETUP_PLOT_ENV.command"
PYTHON="$HERE/.plot_venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "BLOCKED: plotting environment executable missing: $PYTHON" >&2
    exit 2
fi
exec "$PYTHON" "$HERE/figure_3_seeds.py" "$@"
