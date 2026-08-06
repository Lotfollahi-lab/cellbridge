#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${TERRA_MULTI_ENV_ROOT:-$ROOT/../../envs}"
PYTHON="${ENV_PATH:-$ENV_ROOT/terra-dev}/bin/python"
PYTHONPATH="$ROOT/src" "$PYTHON" -m cellbridge.cluster "$@"
