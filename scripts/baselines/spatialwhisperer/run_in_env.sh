#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
SPATIALWHISPERER_REPO="${SPATIALWHISPERER_REPO:-$WORKSPACE/repos/spatialwhisperer}"

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") <subcommand> [arguments]

Runs the terra-multi SpatialWhisperer adapter in the external repository's
Pixi environment. Example:

  $(basename "$0") check --config $ROOT/configs/baselines/spatialwhisperer.yaml
EOF
}

[ "$#" -gt 0 ] || { usage; exit 2; }
[ -f "$SPATIALWHISPERER_REPO/pixi.toml" ] || {
  echo "SpatialWhisperer pixi.toml not found: $SPATIALWHISPERER_REPO/pixi.toml" >&2
  exit 1
}

if command -v pixi >/dev/null 2>&1; then
  PIXI_BIN="$(command -v pixi)"
elif [ -x "$HOME/.pixi/bin/pixi" ]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
else
  echo "pixi is not installed; see baselines/spatialwhisperer/README.md" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
exec "$PIXI_BIN" run \
  --manifest-path "$SPATIALWHISPERER_REPO/pixi.toml" \
  python -m terra_multi.baselines.spatialwhisperer "$@"
