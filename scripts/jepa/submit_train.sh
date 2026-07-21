#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/jepa/spatial_block.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
"$SUBMIT" submit "$CONFIG" --job "$RUN_NAME" --log-prefix train \
  --queue "${JEPA_QUEUE:-training-normal}" \
  --gpu "${JEPA_GPU_SPEC:-num=1:gmem=30000}" \
  --memory "${JEPA_TRAIN_MEM_GB:-128}" --cores "${JEPA_TRAIN_NCORES:-12}" -- \
  python3 -m terra_multi.jepa.train --config "$CONFIG"
