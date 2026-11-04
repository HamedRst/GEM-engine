#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="$HOME/Desktop/GEM_Publication_Archive"
PY="$ARCHIVE/Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/.plot_venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi
# Rebuild current active folder atomically with current source.
exec "$PY" "$DIR/install.py" --archive "$ARCHIVE" --plot-seed 42
