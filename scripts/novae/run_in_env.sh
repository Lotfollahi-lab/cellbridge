#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
ENVIRONMENT="${NOVAE_ENV:-$WORKSPACE/envs/novae}"
NOVAE_REPO="${NOVAE_REPO:-$WORKSPACE/repos/novae}"
NOVAE_HF_HOME="${NOVAE_HF_HOME:-$WORKSPACE/cache/huggingface}"
mkdir -p "$NOVAE_HF_HOME"
export HF_HOME="$NOVAE_HF_HOME"
if [ ! -x "$ENVIRONMENT/bin/python" ]; then
  echo "Missing NOVAE environment: $ENVIRONMENT" >&2
  exit 1
fi
source "$ENVIRONMENT/bin/activate"
export PYTHONPATH="$ROOT/src:$NOVAE_REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
