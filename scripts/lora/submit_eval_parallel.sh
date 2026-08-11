#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
mapfile -t SAMPLES < <("$SUBMIT" cohort "$CONFIG" eval.label_test_fast_path splits_path label_test)

submit() {
  local suffix="$1" memory="$2" cores="$3" dependency="$4"; shift 4
  local dep_args=(); [ -z "$dependency" ] || dep_args=(--dependency "$dependency")
  "$SUBMIT" submit "$CONFIG" --job "${RUN_NAME}_${suffix}" --log-prefix "$suffix" \
    --queue "${LORA_EVAL_QUEUE:-auto}" --gmem "${LORA_EVAL_GMEM:-30000}" \
    --memory "$memory" --cores "$cores" --environment hest "${dep_args[@]}" -- \
    python -m cellbridge.lora.evaluate --config "$CONFIG" "$@"
}

IDS=()
for sample in "${SAMPLES[@]}"; do
  id="$(submit "export_hest_${sample}" "${LORA_EVAL_MEM_GB:-96}" \
    "${LORA_EVAL_NCORES:-12}" "" --samples "$sample" --skip-phenocell --skip-probes)"
  echo "  HEST $sample -> job $id" >&2; IDS+=("$id")
done
id="$(submit export_phenocell "${LORA_EVAL_MEM_GB:-96}" \
  "${LORA_EVAL_NCORES:-12}" "" --skip-hest --skip-probes)"
echo "  PHENOCELL -> job $id" >&2; IDS+=("$id")
dependency="$(printf 'done(%s)&&' "${IDS[@]}")"
submit aggregate "${LORA_AGG_MEM_GB:-128}" "${LORA_AGG_NCORES:-16}" \
  "${dependency%&&}"
