#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="${GEM_ARCHIVE_ROOT:-$HOME/Desktop/GEM_Publication_Archive}"
PY="$HERE/.plot_venv/bin/python"
if [[ ! -x "$PY" ]]; then bash "$HERE/SETUP_PLOT_ENV.command"; fi
"$PY" "$HERE/si_landscape.py" --archive "$ARCHIVE" --mode discover
"$PY" "$HERE/si_landscape.py" --archive "$ARCHIVE" --mode generate
