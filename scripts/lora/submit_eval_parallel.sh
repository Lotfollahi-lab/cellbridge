#!/bin/bash
set -euo pipefail

# Fan the evaluation of one config out across the cluster:
#   - one GPU job per HEST label_test_fast sample (feature export only)
#   - one GPU job for PhenoCell (feature export only)
#   - one aggregate job, gated on all of the above, that runs every probe
#
# The aggregate re-runs the exports with no --samples filter. Every feature file
# is already current by then, so each one is a signature check rather than a
# forward pass; that rebuilds the full manifest probe_hest requires and falls
# straight through to the probes.
#
# Usage: submit_eval_parallel.sh [config]

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

# The label_test_fast sample IDs, straight from the file the evaluator reads.
mapfile -t SAMPLES < <(tail -n +2 "$FAST_PATH" | cut -d, -f1 | grep -v '^$')
[ "${#SAMPLES[@]}" -gt 0 ] || { echo "no samples found in $FAST_PATH" >&2; exit 1; }

QUEUE="${LORA_EVAL_QUEUE:-training-normal}"
GPU_SPEC="${LORA_EVAL_GPU_SPEC:-num=1:gmem=30000}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${LORA_EVAL_MEM_GB:-96}"
NCORES="${LORA_EVAL_NCORES:-12}"
AGG_MEM_GB="${LORA_AGG_MEM_GB:-128}"
AGG_NCORES="${LORA_AGG_NCORES:-16}"

submit() {
  # submit <job-suffix> <mem-gb> <ncores> <dependency-or-empty> <py-args...>
  local suffix="$1" mem="$2" cores="$3" dependency="$4"; shift 4
  local dep_directive=""
  [ -n "$dependency" ] && dep_directive="#BSUB -w \"${dependency}\""

  bsub <<EOF | sed -nE 's/.*Job <([0-9]+)>.*/\1/p'
#BSUB -J ${RUN_NAME}_${suffix}
#BSUB -o ${LOG_DIR}/${suffix}_%J.out
#BSUB -e ${LOG_DIR}/${suffix}_%J.err
#BSUB -n ${cores}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${mem}G
#BSUB -R "select[mem>${mem}G] rusage[mem=${mem}G] span[hosts=1]"
${dep_directive}

set -euo pipefail
module load ISG/conda
conda activate hest
cd "${ROOT}"
export OMP_NUM_THREADS=${cores} OPENBLAS_NUM_THREADS=${cores} MKL_NUM_THREADS=${cores}
PYTHONPATH=src python -m terra_multi.lora.evaluate --config "${CONFIG}" $*
EOF
}

job_ids=()

for sample in "${SAMPLES[@]}"; do
  id="$(submit "export_hest_${sample}" "$MEM_GB" "$NCORES" "" \
    --samples "$sample" --skip-phenocell --skip-probes)"
  [ -n "$id" ] || { echo "failed to submit HEST export for $sample" >&2; exit 1; }
  echo "  HEST  ${sample}  -> job ${id}"
  job_ids+=("$id")
done

id="$(submit "export_phenocell" "$MEM_GB" "$NCORES" "" --skip-hest --skip-probes)"
[ -n "$id" ] || { echo "failed to submit PhenoCell export" >&2; exit 1; }
echo "  PHENO         -> job ${id}"
job_ids+=("$id")

dependency="$(printf 'done(%s)&&' "${job_ids[@]}")"
dependency="${dependency%&&}"
agg="$(submit "aggregate" "$AGG_MEM_GB" "$AGG_NCORES" "$dependency")"
[ -n "$agg" ] || { echo "failed to submit aggregate" >&2; exit 1; }
echo "  AGGREGATE     -> job ${agg}  (waits on ${#job_ids[@]} jobs)"
