#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_ROOT="${CELLBRIDGE_ENV_ROOT:-$ROOT/../../envs}"
ENV_PATH="${ENV_PATH:-$ENV_ROOT/terra-dev}"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"

if command -v module >/dev/null 2>&1; then
  module load cellgen/uv
fi
source "$ENV_PATH/bin/activate"

cd "$ROOT"
PYTHONPATH=src python3 src/cellbridge/preprocess/01_find_samples.py --config "$CONFIG"
