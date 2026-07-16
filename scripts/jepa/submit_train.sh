#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/jepa/spatial_block.yaml}"
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

QUEUE="${JEPA_QUEUE:-training-normal}"
GPU_SPEC="${JEPA_GPU_SPEC:-num=1:gmem=30000}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${JEPA_TRAIN_MEM_GB:-128}"
NCORES="${JEPA_TRAIN_NCORES:-12}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"

bsub <<EOF
#BSUB -J ${RUN_NAME}
#BSUB -o ${OUTPUT_DIR}/logs/train_%J.out
#BSUB -e ${OUTPUT_DIR}/logs/train_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"

set -euo pipefail
module load cellgen/uv
source "${ENV_PATH}/bin/activate"
cd "${ROOT}"
export OMP_NUM_THREADS=${NCORES} OPENBLAS_NUM_THREADS=${NCORES} MKL_NUM_THREADS=${NCORES}
PYTHONPATH=src python3 -m terra_multi.jepa.train --config "${CONFIG}"
EOF
