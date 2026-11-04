#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for PY in \
 "$HOME/Downloads/GEM_EXPLORATION_GENE_ACTIVITY_GUI_ONLY_RERUN_V3/.venv/bin/python" \
 "$HOME/Downloads/GEM_EXPLORATION_SEED42_50K_GUI_ONLY_V2/.venv/bin/python" \
 "$HOME/Library/Application Support/GEM Engine/.venv/bin/python" \
 python3.12 python3; do
  if [[ "$PY" == /* ]] && [[ ! -x "$PY" ]]; then continue; fi
  if "$PY" -c 'import playwright,PIL' >/dev/null 2>&1; then
    echo "Capturing actual GEM Engine UI with $PY"
    exec "$PY" "$HERE/capture_all_gui.py" "$@"
  fi
done
echo "Playwright and Pillow not found. Reuse your established Python 3.12 GUI environment or install them first." >&2
exit 2
