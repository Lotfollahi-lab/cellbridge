#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/distill/distill_v0_vicreg_mild.yaml}"
train_job="$($SCRIPT_DIR/submit_train.sh "$CONFIG")"
echo "training -> job $train_job" >&2
DEPENDENCY="done($train_job)" "$SCRIPT_DIR/submit_probes.sh" "$CONFIG"
