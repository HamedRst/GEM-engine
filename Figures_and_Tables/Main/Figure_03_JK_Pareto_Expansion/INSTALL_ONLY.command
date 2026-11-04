#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="${GEM_ARCHIVE_ROOT:-$HOME/Desktop/GEM_Publication_Archive}"
python3 "$HERE/install.py" --archive "$ARCHIVE"
