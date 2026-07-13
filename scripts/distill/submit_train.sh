#!/bin/bash
set -euo pipefail

# Usage: submit_train.sh [config]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"

CONFIG="${1:-$ROOT/configs/distill/distill_v0_vicreg_mild.yaml}"
case "$CONFIG" in
  /*) ;;
  *) CONFIG="$ROOT/$CONFIG" ;;
esac
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

RUN_NAME="$(sed -nE 's/^[[:space:]]*run_name:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
OUTPUT_DIR="$(sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
[ -n "$RUN_NAME" ] && [ -n "$OUTPUT_DIR" ] || {
  echo "config must define run_name and output_dir" >&2
  exit 1
}
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;;
esac
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE="${DISTILL_QUEUE:-training-normal}"
GPU_SPEC="${DISTILL_GPU_SPEC:--}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${DISTILL_MEM_GB:-64}"
NCORES="${DISTILL_NCORES:-12}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"
GPU_DIRECTIVE="#BSUB -gpu \"${GPU_SPEC}\""

bsub <<EOF
#BSUB -J ${RUN_NAME}
#BSUB -o ${LOG_DIR}/${RUN_NAME}_%J.out
#BSUB -e ${LOG_DIR}/${RUN_NAME}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
${GPU_DIRECTIVE}
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"

set -euo pipefail
module load cellgen/uv
source "${ENV_PATH}/bin/activate"
cd "${ROOT}"
export OMP_NUM_THREADS=${NCORES} OPENBLAS_NUM_THREADS=${NCORES} MKL_NUM_THREADS=${NCORES} NUMEXPR_NUM_THREADS=${NCORES}
PYTHONPATH=src python3 -m terra_multi.distill.train --config "${CONFIG}"
EOF
