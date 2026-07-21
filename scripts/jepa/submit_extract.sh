#!/bin/bash
set -euo pipefail
CONFIG="${1:?usage: submit_extract.sh <config> <sample-id>}"
SAMPLE="${2:?usage: submit_extract.sh <config> <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
DEP_ARGS=(); [ -z "${DEPENDENCY:-}" ] || DEP_ARGS=(--dependency "$DEPENDENCY")
"$SUBMIT" submit "$CONFIG" --job "${RUN_NAME}_extract_${SAMPLE}" \
  --log-prefix "extract_${SAMPLE}" --queue "${JEPA_EXTRACT_QUEUE:-training-normal}" \
  --gpu "${JEPA_EVAL_GPU_SPEC:-num=1:gmem=30000}" \
  --memory "${JEPA_EVAL_MEM_GB:-64}" --cores "${JEPA_EVAL_NCORES:-12}" \
  "${DEP_ARGS[@]}" -- python3 -m terra_multi.jepa.extract \
  --config "$CONFIG" --samples "$SAMPLE"
