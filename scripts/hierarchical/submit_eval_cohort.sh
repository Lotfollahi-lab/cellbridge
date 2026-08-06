#!/bin/bash
# Evaluate one HEST cohort instead of all of them.
#
# A full evaluation runs UNI2-h over every cell of nine sections and takes
# 8-11 hours. Restricting the export to a single cohort cuts that roughly
# threefold, which is the difference between overnight and three days when the
# queue only gives you a couple of GPUs.
#
# Usage: [DEPENDENCY=<jobid>] [COHORT_SAMPLES=A,B,C] submit_eval_cohort.sh <config>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:?usage: submit_eval_cohort.sh <config>}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
SAMPLES="${COHORT_SAMPLES:-TENX200,TENX201,TENX202}"
QUEUE="${LORA_EVAL_QUEUE:-gpu-interactive}"

submit() {
  local suffix="$1" dependency="$2" gpu="$3"; shift 3
  local dep_args=(); [ -z "$dependency" ] || dep_args=(--dependency "$dependency")
  local gpu_args=(); [ -z "$gpu" ] || gpu_args=(--gpu "$gpu")
  "$SUBMIT" submit --job "${RUN_NAME}_${suffix}" --log-prefix "$suffix" \
    --queue "$QUEUE" "${gpu_args[@]}" \
    --memory "${LORA_EVAL_MEM_GB:-96}" --cores "${LORA_EVAL_NCORES:-8}" \
    --environment hest "${dep_args[@]}" "$CONFIG" -- \
    python -m cellbridge.lora.evaluate --config "$CONFIG" "$@"
}

DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
# Export needs the GPU for UNI2-h. Probing only reads cached features, but every
# GPU queue here rejects jobs that request no GPU, so ask for the smallest slice.
export_id="$(submit export "$DEP" "${LORA_EVAL_GPU_SPEC:-num=1:gmem=30000}" \
  --samples "$SAMPLES" --skip-phenocell --skip-probes)"
probe_id="$(submit probe "done($export_id)" "num=1:gmem=1000" \
  --probe-only --probe-samples "$SAMPLES" --skip-phenocell)"
echo "$export_id $probe_id"
