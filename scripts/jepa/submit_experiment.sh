#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/jepa/spatial_block.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"

train="${JEPA_TRAIN_JOB_ID:-}"
if [ -z "$train" ]; then
  train="$($SCRIPT_DIR/submit_train.sh "$CONFIG")"
else
  [[ "$train" =~ ^[0-9]+$ ]] || { echo "JEPA_TRAIN_JOB_ID must be numeric" >&2; exit 1; }
fi
echo "  TRAIN -> job $train" >&2
mapfile -t SAMPLES < <("$SUBMIT" cohort "$CONFIG" \
  eval.label_test_fast_path splits_path label_test)
PROBES=()
for sample in "${SAMPLES[@]}"; do
  extract="$(DEPENDENCY="done($train)" "$SCRIPT_DIR/submit_extract.sh" "$CONFIG" "$sample")"
  probe="$(DEPENDENCY="done($extract)" "$SCRIPT_DIR/submit_probe.sh" "$CONFIG" "$sample")"
  echo "  $sample -> extract $extract, probe $probe" >&2
  PROBES+=("$probe")
done
dependency="$(printf 'done(%s)&&' "${PROBES[@]}")"
DEPENDENCY="${dependency%&&}" "$SCRIPT_DIR/submit_aggregate.sh" "$CONFIG"
