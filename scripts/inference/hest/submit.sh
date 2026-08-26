#!/bin/bash
# Submit all reproducibility methods for the annotated HEST cohort.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
SUBMIT="$ROOT/scripts/submit_job.sh"
SUBMIT_RUN="$ROOT/scripts/inference/submit_run.sh"
CHECKPOINTS_ROOT="${CELLBRIDGE_CHECKPOINTS_ROOT:-$WORKSPACE/checkpoints}"
CONFIG="$ROOT/configs/inference/hest.yaml"

samples=("$@")
if [ "${#samples[@]}" -eq 0 ]; then
  mapfile -t samples < <(
    "$SUBMIT" cohort "$CONFIG" data.cohort data.split label_test
  )
fi

for sample_id in "${samples[@]}"; do
  for method in uni past volta spatialwhisperer; do
    "$SUBMIT_RUN" "$CONFIG" "$method" all \
      --sample-id "$sample_id" --conformal-config "$CONFIG"
  done

  for arm in cells_5k cells_10k cells_50k cells_all; do
    checkpoint="$CHECKPOINTS_ROOT/reproducibility/lora/scaling/$arm/best.pt"
    "$SUBMIT_RUN" "$CONFIG" cellbridge "$arm" \
      --sample-id "$sample_id" --checkpoint "$checkpoint" \
      --conformal-config "$CONFIG"
  done
done
