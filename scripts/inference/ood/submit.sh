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


  for method in uni past volta spatialwhisperer lemon virchow2 hibou_l; do
    version=all
    if [ "$method" = lemon ]; then version=vits8; fi
    "$SUBMIT_RUN" "$config" "$method" "${version}" --conformal-config "$config"
  done

  "$SUBMIT_RUN" "$config" cellbridge "uni_terra_four_layers_adapted_all_cells" \
    --checkpoint "$CHECKPOINTS_ROOT/uni_terra_four_layers_adapted_all_cells/best.pt" \
    --conformal-config "$config"
done
