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

QUEUE="${JEPA_AGG_QUEUE:-training-normal}"
GPU_SPEC="${JEPA_AGG_GPU_SPEC:--}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${JEPA_AGG_MEM_GB:-8}"
NCORES="${JEPA_AGG_NCORES:-2}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"
DEPENDENCY_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEPENDENCY_DIRECTIVE="#BSUB -w \"${DEPENDENCY}\""

bsub <<EOF
#BSUB -J ${RUN_NAME}_aggregate
#BSUB -o ${OUTPUT_DIR}/logs/aggregate_%J.out
#BSUB -e ${OUTPUT_DIR}/logs/aggregate_%J.err
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
PYTHONPATH=src python3 -m terra_multi.jepa.evaluate --config "${CONFIG}" --aggregate-only
EOF
