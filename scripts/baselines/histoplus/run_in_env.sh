#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
HISTOPLUS_REPO="${HISTOPLUS_REPO:-$WORKSPACE/repos/histoplus}"
HISTOPLUS_ENV="${HISTOPLUS_ENV:-$WORKSPACE/envs/histoplus}"
HISTOPLUS_PYTHON="${HISTOPLUS_PYTHON:-3.12}"

[ "$#" -gt 0 ] || {
  echo "Usage: $(basename "$0") <check|prepare|infer-hest|infer-ood|score> [arguments]" >&2
  exit 2
}
[ -f "$HISTOPLUS_REPO/uv.lock" ] || {
  echo "HistoPLUS uv.lock not found: $HISTOPLUS_REPO/uv.lock" >&2
  exit 1
}

module load cellgen/uv
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$HISTOPLUS_ENV"
exec uv run --frozen --python "$HISTOPLUS_PYTHON" --project "$HISTOPLUS_REPO" \
  --with lmdb --with pyarrow --with pyyaml \
  python -m terra_multi.baselines.histoplus "$@"
