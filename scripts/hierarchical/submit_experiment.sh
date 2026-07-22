#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/hierarchical/lora_combined.yaml}"
exec "$ROOT/scripts/lora/submit_experiment.sh" "$CONFIG"
