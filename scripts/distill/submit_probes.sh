#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/distill/distill_v0_vicreg_mild.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"

if [ "$($SUBMIT value "$CONFIG" distill_version)" = v1 ]; then
  DIM="$($SUBMIT value "$CONFIG" feat_dim)"
  COUNT="$($SUBMIT value "$CONFIG" num_prototypes)"
  ARMS=("prototype_penultimate_$DIM" "prototype_probs_$COUNT")
else
  ARMS=(uni_baseline terra_teacher distilled concat_oracle)
fi
IDS=()
for arm in "${ARMS[@]}"; do
  id="$($SCRIPT_DIR/submit_probe.sh "$CONFIG" "$arm")"
  echo "  $arm -> job $id" >&2
  IDS+=("$id")
done
DEPENDENCY="$(printf 'done(%s)&&' "${IDS[@]}")"
DEPENDENCY="${DEPENDENCY%&&}" "$SCRIPT_DIR/submit_probe.sh" "$CONFIG" aggregate
