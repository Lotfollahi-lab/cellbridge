#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
TERRA_ENV="${TERRA_ENV:-$WORKSPACE/envs/terra-dev}"
SPATIALWHISPERER_REPO="${SPATIALWHISPERER_REPO:-$WORKSPACE/repos/spatialwhisperer}"

if [ "$#" -lt 3 ]; then
  echo "Usage: $(basename "$0") <core|method> <command> <config> [arguments]" >&2
  exit 2
fi

runtime="$1"
command="$2"
config="$3"
shift 3
case "$config" in /*) ;; *) config="$ROOT/$config" ;; esac

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
arguments=(python -m cellbridge.inference.run "$command" --config "$config")
if [ "$runtime" != core ]; then
  arguments+=(--method "$runtime")
fi
arguments+=("$@")

if [ "$runtime" = core ]; then
  exec "$TERRA_ENV/bin/python" "${arguments[@]:1}"
fi

if [ "$runtime" = spatialwhisperer ]; then
  PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
  [ -x "$PIXI_BIN" ] || { echo "pixi not found: $PIXI_BIN" >&2; exit 1; }
  [ -f "$SPATIALWHISPERER_REPO/pixi.toml" ] || {
    echo "SpatialWhisperer pixi.toml not found: $SPATIALWHISPERER_REPO" >&2
    exit 1
  }
  exec "$PIXI_BIN" run --manifest-path "$SPATIALWHISPERER_REPO/pixi.toml" \
    "${arguments[@]}"
fi

[ -x "$TERRA_ENV/bin/python" ] || {
  echo "TERRA environment not found: $TERRA_ENV" >&2
  exit 1
}
module load cellgen/uv
source "$TERRA_ENV/bin/activate"

case "$runtime" in
  uni|cellbridge)
    packages=(--with lmdb)
    ;;
  past)
    packages=(
      --with lmdb
      --with 'torch==2.6.0'
      --with 'torchvision==0.21.0'
      --with 'timm==1.0.9'
    )
    ;;
  volta)
    packages=(--with lmdb --with opencv-python-headless)
    ;;
  *)
    echo "Unknown inference runtime: $runtime" >&2
    exit 2
    ;;
esac

exec uv run --active --no-project "${packages[@]}" "${arguments[@]}"
