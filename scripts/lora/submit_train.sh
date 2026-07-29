#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
"$SUBMIT" submit --job "$RUN_NAME" --log-prefix train \
  --queue "${LORA_QUEUE:-training-normal}" \
  --gpu "${LORA_GPU_SPEC:-num=1:gmem=40000}" \
  --memory "${LORA_TRAIN_MEM_GB:-96}" --cores "${LORA_TRAIN_NCORES:-12}" \
  --environment hest "$CONFIG" -- python -m terra_multi.lora.train --config "$CONFIG"
