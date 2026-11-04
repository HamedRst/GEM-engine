#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${GEM_PUBLICATION_ARCHIVE:-$HOME/Desktop/GEM_Publication_Archive}"
ENV="$ARCHIVE/Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/.plot_venv/bin/python"
if [[ ! -x "$ENV" ]]; then
  echo 'BLOCKED: Figure 3 shared plotting-only Python environment unavailable.' >&2
  exit 2
fi
# One canonical output set; no run_YYYYMMDD subfolders.
exec "$ENV" "$HERE/landscape_knowledge_3seeds.py" --archive "$ARCHIVE"
