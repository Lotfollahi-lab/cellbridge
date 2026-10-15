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
  # Paper Appendix A.1 excludes TENX143.
  if [ "$sample_id" = TENX143 ]; then continue; fi
  for method in uni past volta spatialwhisperer lemon virchow2; do
    version=all
    if [ "$method" = lemon ]; then version=vits8; fi
    "$SUBMIT_RUN" "$CONFIG" "$method" "${version}" \
      --sample-id "$sample_id" --conformal-config "$CONFIG"
  done

  "$SUBMIT_RUN" "$CONFIG" cellbridge "uni_terra_four_layers_adapted_all_cells" \
    --checkpoint "$CHECKPOINTS_ROOT/uni_terra_four_layers_adapted_all_cells/best.pt" --sample-id "$sample_id" \
    --conformal-config "$CONFIG"
done
