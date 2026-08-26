#!/bin/bash
# Submit all reproducibility methods for one or more OOD datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
SUBMIT_RUN="$ROOT/scripts/inference/submit_run.sh"
CHECKPOINTS_ROOT="${CELLBRIDGE_CHECKPOINTS_ROOT:-$WORKSPACE/checkpoints}"

datasets=("$@")
if [ "${#datasets[@]}" -eq 0 ]; then
  datasets=(lizard nucls pannuke phenocell)
fi

for dataset in "${datasets[@]}"; do
  config="$ROOT/configs/inference/${dataset}.yaml"
  if [ ! -f "$config" ]; then
    echo "missing OOD config: $config" >&2
    exit 1
  fi


  for method in uni past volta spatialwhisperer; do
    "$SUBMIT_RUN" "$config" "$method" all --conformal-config "$config"
  done

  for arm in cells_5k cells_10k cells_50k cells_all; do
    checkpoint="$CHECKPOINTS_ROOT/reproducibility/lora/scaling/$arm/best.pt"
    "$SUBMIT_RUN" "$config" cellbridge "$arm" \
      --checkpoint "$checkpoint" --conformal-config "$config"
  done
done
