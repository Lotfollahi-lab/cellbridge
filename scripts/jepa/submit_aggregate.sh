#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/jepa/spatial_block.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
DEP_ARGS=(); [ -z "${DEPENDENCY:-}" ] || DEP_ARGS=(--dependency "$DEPENDENCY")
"$SUBMIT" submit "$CONFIG" --job "${RUN_NAME}_aggregate" --log-prefix aggregate \
  --queue "${JEPA_AGG_QUEUE:-training-normal}" --gpu "${JEPA_AGG_GPU_SPEC:--}" \
  --memory "${JEPA_AGG_MEM_GB:-8}" --cores "${JEPA_AGG_NCORES:-2}" \
  "${DEP_ARGS[@]}" -- python3 -m cellbridge.jepa.evaluate \
  --config "$CONFIG" --aggregate-only
