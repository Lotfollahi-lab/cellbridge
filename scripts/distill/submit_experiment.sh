#!/bin/bash
set -euo pipefail

# Submit training, then all label-test probe arms after its checkpoint is ready.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/distill/distill_v0_vicreg_mild.yaml}"

if ! train_output="$("$SCRIPT_DIR/submit_train.sh" "$CONFIG" 2>&1)"; then
  printf '%s\n' "$train_output" >&2
  exit 1
fi
printf '%s\n' "$train_output"
train_job="$(printf '%s\n' "$train_output" | sed -nE 's/.*Job <([0-9]+)>.*/\1/p' | tail -n1)"
[ -n "$train_job" ] || { echo "could not parse training job ID" >&2; exit 1; }

DEPENDENCY="done(${train_job})" "$SCRIPT_DIR/submit_probes.sh" "$CONFIG"
