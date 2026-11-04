#!/usr/bin/env bash
# GEM Engine macOS native network renderer dependency (sourced into launcher).
# The Python `pydot` package does not provide `dot`.
# Installs an isolated Graphviz using official Pixi+conda-forge if system dot is absent.
# Requires: GEM Engine launcher invokes `source .../ensure_graphviz.sh "${SUPPORT}"`.

gem_graphviz_ready() {
  local candidate="$1" format
  [[ -x "$candidate" ]] || return 1
  "$candidate" -V >/dev/null 2>&1 || return 1
  for format in svg pdf png; do
    printf '%s\n' 'digraph G {a -> b}' | "$candidate" "-T${format}" >/dev/null 2>&1 || return 1
  done
  return 0
}

gem_graphviz_bootstrap() {
  local support="${1:-$HOME/Library/Application Support/GEM Engine}"
  local pixi_home="$support/tools/pixi"
  local pixi_bin="$pixi_home/bin/pixi"
  local installer="$support/tools/install-pixi.sh"
  local dot_path=""
  local os="$(uname -s)"
  # Add only application-owned paths; never change user's shell profile or base Conda.
  export PATH="$pixi_home/envs/gem-engine-graphviz/bin:$pixi_home/bin:$PATH"
  if command -v dot >/dev/null 2>&1; then
    dot_path="$(command -v dot)"
    if gem_graphviz_ready "$dot_path"; then
      printf 'Graphviz renderer READY: %s (%s)\n' "$dot_path" "$("$dot_path" -V 2>&1)"
      return 0
    fi
    echo 'An existing Graphviz dot is unusable; checking for an isolated copy.' >&2
  fi
  if [[ "$os" != "Darwin" ]]; then
    echo 'ERROR: This first-launch Graphviz installer supports the macOS GEM Engine.app only.' >&2
    return 1
  fi
  if [[ "${GEM_GRAPHVIZ_AUTO_INSTALL:-1}" == "0" ]]; then
    echo 'ERROR: Graphviz dot missing and GEM_GRAPHVIZ_AUTO_INSTALL=0. Install Graphviz and relaunch.' >&2
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo 'ERROR: curl is missing. Graphviz setup requires internet access on first installation.' >&2
    return 1
  fi
  mkdir -p "$support/tools" || return 1
  if [[ ! -x "$pixi_bin" ]]; then
    echo 'Installing isolated Pixi helper for the Graphviz renderer (no Homebrew or Conda required).'
    if ! curl --fail --location --silent --show-error --retry 2 --connect-timeout 20 \
      https://pixi.sh/install.sh --output "$installer"; then
      echo 'ERROR: Could not download the official Pixi installer. Check network access.' >&2
      return 1
    fi
    if ! env PIXI_HOME="$pixi_home" PIXI_BIN_DIR="$pixi_home/bin" \
         PIXI_NO_PATH_UPDATE=1 PIXI_NO_TELEMETRY=1 \
         bash "$installer"; then
      echo 'ERROR: Pixi could not be installed into the GEM Engine application-support folder.' >&2
      return 1
    fi
  fi
  [[ -x "$pixi_bin" ]] || { echo 'ERROR: Pixi binary not found after setup.' >&2; return 1; }
  echo 'Installing native Graphviz from conda-forge into the isolated GEM Engine tools folder.'
  if ! PIXI_HOME="$pixi_home" PIXI_NO_TELEMETRY=1 \
       "$pixi_bin" global install --environment gem-engine-graphviz --channel conda-forge graphviz; then
    echo 'ERROR: Graphviz installation via Pixi/conda-forge failed.' >&2
    return 1
  fi
  export PATH="$pixi_home/envs/gem-engine-graphviz/bin:$pixi_home/bin:$PATH"
  dot_path="$(command -v dot || true)"
  if [[ -z "$dot_path" ]] || ! gem_graphviz_ready "$dot_path"; then
    echo 'ERROR: Graphviz dot is still unavailable or failed its SVG/PDF/PNG smoke tests.' >&2
    return 1
  fi
  printf 'Graphviz renderer READY: %s (%s)\n' "$dot_path" "$("$dot_path" -V 2>&1)"
  return 0
}
gem_graphviz_bootstrap "${1:-}"
