#!/bin/bash
# Select modern Python; install plot dependencies in a private, replaceable venv.
# STDOUT emits only interpreter pathname so RUN_SI_LANDSCAPE can consume it.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="${GEM_ARCHIVE_ROOT:-$HOME/Desktop/GEM_Publication_Archive}"
ENV="$SCRIPTS/.plot_venv"

compatible_python() {
  local candidate="$1"
  [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1 || return 1
  "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' >/dev/null 2>&1
}

ready_env() {
  [ -x "$ENV/bin/python" ] || return 1
  "$ENV/bin/python" - <<'PYTHON' >/dev/null 2>&1
import sys
from importlib import metadata as m
required={'numpy':'2.2.6','matplotlib':'3.10.7','pyarrow':'23.0.1'}
assert (3,10) <= sys.version_info[:2] < (3,14)
for module,version in required.items():
    assert m.version(module)==version, (module,m.version(module),version)
import numpy,matplotlib,pyarrow
PYTHON
}

choose_python() {
  if [ -n "${GEM_PLOT_PYTHON:-}" ]; then
    if compatible_python "$GEM_PLOT_PYTHON"; then
      printf '%s\n' "$GEM_PLOT_PYTHON"
      return 0
    fi
    echo "ERROR: GEM_PLOT_PYTHON is incompatible: $GEM_PLOT_PYTHON (requires Python 3.10–3.13)." >&2
    return 1
  fi
  local candidate
  for candidate in python3.12 python3.13 python3.11 python3.10 \
    /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
    "$HOME/miniconda3/bin/python" "$HOME/miniforge3/bin/python" \
    "$HOME/anaconda3/bin/python" \
    "$ARCHIVE/Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/.plot_venv/bin/python" \
    python3; do
    if compatible_python "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo 'ERROR: no compatible Python found. Your default python3 is likely Python 3.9.' >&2
  echo 'Install Python 3.12 (python.org / Homebrew), or set GEM_PLOT_PYTHON to Python 3.12.' >&2
  return 1
}

if ready_env; then
  echo "PLOT ENV READY: $ENV" >&2
  printf '%s\n' "$ENV/bin/python"
  exit 0
fi

PYTHON="$(choose_python)" || exit 1
BASE_VERSION="$("$PYTHON" -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')"
echo "Setting up plotting environment using $PYTHON (Python $BASE_VERSION)" >&2
# Only .plot_venv is removed. All archived landscapes and mappings are untouched.
if [ -e "$ENV" ]; then
  echo 'Replacing incompatible/incomplete old .plot_venv' >&2
  rm -rf "$ENV"
fi
"$PYTHON" -m venv "$ENV" || { echo 'ERROR: Unable to create plotting virtual environment' >&2; exit 1; }
"$ENV/bin/python" -m pip install --disable-pip-version-check --upgrade 'pip>=25,<27' >&2 || {
  echo 'ERROR: pip upgrade failed (network access or Python installation may be the issue).' >&2; exit 1;
}
"$ENV/bin/python" -m pip install --disable-pip-version-check -r "$SCRIPTS/requirements.txt" >&2 || {
  echo 'ERROR: Plot dependencies failed. Check Python version and pip network access.' >&2; exit 1;
}
if ! ready_env; then
  echo 'ERROR: Installed plotting environment did not pass version/import checks.' >&2
  exit 1
fi
echo "PLOT ENV READY: $ENV" >&2
printf '%s\n' "$ENV/bin/python"
