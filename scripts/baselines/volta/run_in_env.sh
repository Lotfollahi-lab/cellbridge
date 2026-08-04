#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
ENVIRONMENT="${TERRA_ENV:-${ENV_PATH:-$WORKSPACE/envs/terra-dev}}"

[ "$#" -gt 0 ] || {
  echo "Usage: $(basename "$0") <subcommand> [arguments]" >&2
  exit 2
}
[ -x "$ENVIRONMENT/bin/python" ] || {
  echo "TERRA environment not found: $ENVIRONMENT" >&2
  exit 1
}

if command -v module >/dev/null 2>&1; then
  module load cellgen/uv
fi
source "$ENVIRONMENT/bin/activate"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --active --no-project --with lmdb --with opencv-python-headless \
  python -m terra_multi.baselines.volta "$@"
