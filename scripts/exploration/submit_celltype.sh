#!/bin/bash
set -euo pipefail

MODEL="$1"   # terra | terra_neighbor | uni | concat | neighbor_concat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"

QUEUE="${CELLTYPE_QUEUE:-${QUEUE:-oversubscribed}}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${CELLTYPE_MEM_GB:-${MEM_GB:-8}}"
NCORES="${CELLTYPE_NCORES:-${NCORES:-1}}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"
CONFIG="${CONFIG:-$ROOT/configs/exploration/celltype_tenx116.yaml}"
GPU_SPEC="${CELLTYPE_GPU_SPEC:-${CPU_GPU_SPEC:-}}"
if [ -z "$GPU_SPEC" ]; then
  case ",${QUEUE}," in
    *,training-normal,*|*,training-parallel,*|*,gpu-interactive,*|*,inference,*) GPU_SPEC="-" ;;
  esac
fi
GPU_DIRECTIVE=""
[ -n "$GPU_SPEC" ] && GPU_DIRECTIVE="#BSUB -gpu \"${GPU_SPEC}\""

# Derive the output dir from the config
OUTPUT_DIR="$(sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"

case "$OUTPUT_DIR" in
  /*) : ;;                             # already absolute
  *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; # relative to project root (matches run cwd)
esac
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

bsub <<EOF
#BSUB -J celltype_${MODEL}
#BSUB -o ${LOG_DIR}/celltype_${MODEL}_%J.out
#BSUB -e ${LOG_DIR}/celltype_${MODEL}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
${GPU_DIRECTIVE}
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"

set -euo pipefail
module load cellgen/uv
source "${ENV_PATH}/bin/activate"
which python3
python3 --version

cd "${ROOT}"
# Respect the allocated slots and prevent hidden BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS=${NCORES} OPENBLAS_NUM_THREADS=${NCORES} MKL_NUM_THREADS=${NCORES} NUMEXPR_NUM_THREADS=${NCORES}
PYTHONPATH=src python3 -m terra_multi.evaluation.run --config "${CONFIG}" --model "${MODEL}"
EOF
