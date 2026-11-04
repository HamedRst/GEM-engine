#!/bin/bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
bash "$HERE/SETUP_PLOT_ENV.command"
exec "$HERE/.plot_venv/bin/python" "$HERE/figure_3_ghi_total_wins_v3.py" "$@"
