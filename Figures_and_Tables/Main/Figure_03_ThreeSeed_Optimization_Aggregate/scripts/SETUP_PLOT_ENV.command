#!/bin/bash
# Bootstrap the plotting-only virtual environment. No scientific data touched.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP="$(command -v python)"
else
    echo 'BLOCKED: Python 3 was not found. Install Python 3.10-3.13.' >&2
    exit 2
fi
exec "$BOOTSTRAP" "$HERE/setup_plot_env.py"
