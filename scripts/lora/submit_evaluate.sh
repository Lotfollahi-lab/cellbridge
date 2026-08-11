#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$SUBMIT" submit --job "eval_${RUN_NAME}" --log-prefix evaluate \
  --queue "${LORA_EVAL_QUEUE:-auto}" --gmem "${LORA_EVAL_GMEM:-30000}" \
  --memory "${LORA_EVAL_MEM_GB:-96}" --cores "${LORA_EVAL_NCORES:-12}" \
  --environment hest "${DEP_ARGS[@]}" "$CONFIG" -- \
  python -m cellbridge.lora.evaluate --config "$CONFIG"
