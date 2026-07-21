#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
mapfile -t SAMPLES < <("$SUBMIT" cohort "$CONFIG" eval.label_test_fast_path splits_path label_test)

IDS=()
for sample in "${SAMPLES[@]}"; do
  id="$($SUBMIT submit "$CONFIG" --job "${RUN_NAME}_probe_${sample}" \
    --log-prefix "probe_${sample}" --queue "${LORA_PROBE_QUEUE:-training-normal}" \
    --gpu "${LORA_PROBE_GPU_SPEC:--}" --memory "${LORA_PROBE_MEM_GB:-64}" \
    --cores "${LORA_PROBE_NCORES:-12}" --environment hest -- \
    python -m terra_multi.lora.evaluate --config "$CONFIG" --probe-samples "$sample")"
  echo "  PROBE $sample -> job $id" >&2; IDS+=("$id")
done
dependency="$(printf 'done(%s)&&' "${IDS[@]}")"
"$SUBMIT" submit "$CONFIG" --job "${RUN_NAME}_aggregate" --log-prefix aggregate \
  --queue "${LORA_EVAL_QUEUE:-training-normal}" \
  --gpu "${LORA_EVAL_GPU_SPEC:-num=1:gmem=30000}" \
  --memory "${LORA_AGG_MEM_GB:-128}" --cores "${LORA_AGG_NCORES:-16}" \
  --environment hest --dependency "${dependency%&&}" -- \
  python -m terra_multi.lora.evaluate --config "$CONFIG"
