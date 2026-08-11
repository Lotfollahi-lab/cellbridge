#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PATH="${ENV_PATH:-$ROOT/../../envs/terra-dev}"
PYTHONPATH="$ROOT/src" exec "$ENV_PATH/bin/python" -m cellbridge.cluster "$@"
