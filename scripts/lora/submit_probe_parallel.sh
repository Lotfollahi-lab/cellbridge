#!/bin/bash
set -euo pipefail

# Fan the probes of one config out across the cluster. Assumes the feature
# exports are already on disk (submit_eval_parallel.sh).
#
#   - one CPU job per HEST label_test_fast sample (probes every arm of it)
#   - one GPU aggregate: HEST probes all cached by then, so it collects them,
#     runs the PhenoCell probes, and writes summary.csv
#
# Probing loads no model, so the per-sample jobs need no GPU and schedule on the
# open CPU queue instead of waiting behind the GPU work.
#
# Usage: submit_probe_parallel.sh [config]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

RUN_NAME="$(sed -nE 's/^[[:space:]]*run_name:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
OUTPUT_DIR="$(sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
FAST_PATH="$(sed -nE 's/^[[:space:]]*label_test_fast_path:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
[ -n "$RUN_NAME" ] && [ -n "$OUTPUT_DIR" ] && [ -n "$FAST_PATH" ] || {
  echo "config must define run_name, output_dir, and eval.label_test_fast_path" >&2
  exit 1
}
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

mapfile -t SAMPLES < <(tail -n +2 "$FAST_PATH" | cut -d, -f1 | grep -v '^$')
[ "${#SAMPLES[@]}" -gt 0 ] || { echo "no samples found in $FAST_PATH" >&2; exit 1; }

# Probing needs no GPU, but the open CPU queue only has 4-core/45G hosts. Sit on
# a GPU *host* without reserving a GPU (-gpu "-"), as the distill probes do.
PROBE_QUEUE="${LORA_PROBE_QUEUE:-training-normal}"
PROBE_GPU_SPEC="${LORA_PROBE_GPU_SPEC:--}"
PROBE_MEM_GB="${LORA_PROBE_MEM_GB:-64}"
PROBE_NCORES="${LORA_PROBE_NCORES:-12}"
BGROUP="${BGROUP:-s10396}"
AGG_QUEUE="${LORA_EVAL_QUEUE:-training-normal}"
AGG_GPU_SPEC="${LORA_EVAL_GPU_SPEC:-num=1:gmem=30000}"
AGG_MEM_GB="${LORA_AGG_MEM_GB:-128}"
AGG_NCORES="${LORA_AGG_NCORES:-16}"

job_ids=()
for sample in "${SAMPLES[@]}"; do
  id="$(bsub <<EOF | sed -nE 's/.*Job <([0-9]+)>.*/\1/p'
#BSUB -J ${RUN_NAME}_probe_${sample}
#BSUB -o ${LOG_DIR}/probe_${sample}_%J.out
#BSUB -e ${LOG_DIR}/probe_${sample}_%J.err
#BSUB -n ${PROBE_NCORES}
#BSUB -q ${PROBE_QUEUE}
#BSUB -gpu "${PROBE_GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${PROBE_MEM_GB}G
#BSUB -R "select[mem>${PROBE_MEM_GB}G] rusage[mem=${PROBE_MEM_GB}G] span[hosts=1]"

set -euo pipefail
module load ISG/conda
conda activate hest
cd "${ROOT}"
PYTHONPATH=src python -m terra_multi.lora.evaluate --config "${CONFIG}" \
  --probe-samples "${sample}"
EOF
)"
  [ -n "$id" ] || { echo "failed to submit probe for $sample" >&2; exit 1; }
  echo "  PROBE ${sample} -> job ${id}"
  job_ids+=("$id")
done

dependency="$(printf 'done(%s)&&' "${job_ids[@]}")"
dependency="${dependency%&&}"

agg="$(bsub <<EOF | sed -nE 's/.*Job <([0-9]+)>.*/\1/p'
#BSUB -J ${RUN_NAME}_aggregate
#BSUB -o ${LOG_DIR}/aggregate_%J.out
#BSUB -e ${LOG_DIR}/aggregate_%J.err
#BSUB -n ${AGG_NCORES}
#BSUB -q ${AGG_QUEUE}
#BSUB -gpu "${AGG_GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${AGG_MEM_GB}G
#BSUB -R "select[mem>${AGG_MEM_GB}G] rusage[mem=${AGG_MEM_GB}G] span[hosts=1]"
#BSUB -w "${dependency}"

set -euo pipefail
module load ISG/conda
conda activate hest
cd "${ROOT}"
PYTHONPATH=src python -m terra_multi.lora.evaluate --config "${CONFIG}"
EOF
)"
[ -n "$agg" ] || { echo "failed to submit aggregate" >&2; exit 1; }
echo "  AGGREGATE -> job ${agg}  (waits on ${#job_ids[@]} probe jobs)"
