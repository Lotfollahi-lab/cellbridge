#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
train_job="$($SCRIPT_DIR/submit_train.sh "$CONFIG")"
echo "training -> job $train_job" >&2
DEPENDENCY="$train_job" "$SCRIPT_DIR/submit_evaluate.sh" "$CONFIG"
