#!/bin/bash
set -euo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ENV="$SCRIPTS/.plot_venv"
compatible_python() {
  local candidate="$1"
  [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1 || return 1
  "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' >/dev/null 2>&1
}
ready_env() {
  [ -x "$ENV/bin/python" ] || return 1
  "$ENV/bin/python" - <<'PY' >/dev/null 2>&1
import sys
from importlib import metadata as m
assert (3,10) <= sys.version_info[:2] < (3,14)
for pkg in ('matplotlib','pyarrow','numpy'):
    assert m.version(pkg)
PY
}
choose_python() {
  local candidate
  for candidate in python3.12 python3.13 python3.11 python3.10 \
    /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
    "$HOME/miniconda3/bin/python" "$HOME/miniforge3/bin/python" \
    "$HOME/anaconda3/bin/python" python3; do
    if compatible_python "$candidate"; then printf '%s\n' "$candidate"; return 0; fi
  done
  echo 'ERROR: no compatible Python found (need 3.10–3.13).' >&2
  return 1
}
if ready_env; then echo "PLOT ENV READY: $ENV" >&2; exit 0; fi
PYTHON="$(choose_python)" || exit 1
echo "Setting up plotting environment using $PYTHON" >&2
rm -rf "$ENV"
"$PYTHON" -m venv "$ENV"
"$ENV/bin/python" -m pip install --disable-pip-version-check --upgrade 'pip>=25,<27' >&2
"$ENV/bin/python" -m pip install --disable-pip-version-check matplotlib pyarrow numpy >&2
if ! ready_env; then echo 'ERROR: plotting environment install failed.' >&2; exit 1; fi
echo "PLOT ENV READY: $ENV" >&2
