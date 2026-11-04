#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="${GEM_ARCHIVE_ROOT:-$HOME/Desktop/GEM_Publication_Archive}"
if [[ ! -x "$HERE/.plot_venv/bin/python" ]]; then
  bash "$HERE/SETUP_PLOT_ENV.command"
fi
"$HERE/.plot_venv/bin/python" "$HERE/generate_figure3_jk.py" --archive "$ARCHIVE"
