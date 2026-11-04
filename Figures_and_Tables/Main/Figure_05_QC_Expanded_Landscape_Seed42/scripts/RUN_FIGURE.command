#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="${GEM_ARCHIVE_ROOT:-$HOME/Desktop/GEM_Publication_Archive}"
valid_python() {
  [ -x "$1" ] || return 1
  "$1" -c 'import matplotlib, numpy, pyarrow' >/dev/null 2>&1
}
SHARED="$ARCHIVE/Figures_and_Tables/Supplementary/SI-Landscape/scripts/.plot_venv/bin/python"
FIG3="$ARCHIVE/Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/.plot_venv/bin/python"
OWN="$HERE/.plot_venv/bin/python"
if valid_python "$SHARED"; then
  PY="$SHARED"
elif valid_python "$FIG3"; then
  PY="$FIG3"
elif valid_python "$OWN"; then
  PY="$OWN"
else
  bash "$HERE/SETUP_PLOT_ENV.command"
  PY="$OWN"
fi
echo "Using verified plotting Python: $PY"
"$PY" "$HERE/generate_figure5.py" --archive "$ARCHIVE"
