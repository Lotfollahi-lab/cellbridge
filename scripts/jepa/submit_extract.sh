#!/bin/bash
set -euo pipefail

CONFIG="${1:?usage: submit_extract.sh <config> <sample-id>}"
SAMPLE="${2:?usage: submit_extract.sh <config> <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

RUN_NAME="$(sed -nE 's/^[[:space:]]*run_name:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
OUTPUT_DIR="$(sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
[ -n "$RUN_NAME" ] && [ -n "$OUTPUT_DIR" ] || {
  echo "config must define run_name and output_dir" >&2
  exit 1
}
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
mkdir -p "$OUTPUT_DIR/logs"

QUEUE="${JEPA_EXTRACT_QUEUE:-training-normal}"
GPU_SPEC="${JEPA_EVAL_GPU_SPEC:-num=1:gmem=30000}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${JEPA_EVAL_MEM_GB:-64}"
NCORES="${JEPA_EVAL_NCORES:-12}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"
DEPENDENCY_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEPENDENCY_DIRECTIVE="#BSUB -w \"${DEPENDENCY}\""

bsub <<EOF
#BSUB -J ${RUN_NAME}_extract_${SAMPLE}
#BSUB -o ${OUTPUT_DIR}/logs/extract_${SAMPLE}_%J.out
#BSUB -e ${OUTPUT_DIR}/logs/extract_${SAMPLE}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEPENDENCY_DIRECTIVE}

set -euo pipefail
module load cellgen/uv
source "${ENV_PATH}/bin/activate"
cd "${ROOT}"
export OMP_NUM_THREADS=${NCORES} OPENBLAS_NUM_THREADS=${NCORES} MKL_NUM_THREADS=${NCORES}
PYTHONPATH=src python3 -m terra_multi.jepa.extract --config "${CONFIG}" --samples "${SAMPLE}"
EOF
