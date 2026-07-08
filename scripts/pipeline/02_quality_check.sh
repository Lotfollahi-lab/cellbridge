#!/bin/bash
set -euo pipefail

SAMPLE_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"

QUEUE="${QC_QUEUE:-${QUEUE:-oversubscribed}}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${QC_MEM_GB:-8}"
NCORES="${QC_NCORES:-4}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"
CONFIG="${CONFIG:-$ROOT/configs/pipeline/pipeline.yaml}"
DEP_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEP_DIRECTIVE="#BSUB -w \"done(${DEPENDENCY})\""
GPU_SPEC="${QC_GPU_SPEC:-${CPU_GPU_SPEC:-}}"
if [ -z "$GPU_SPEC" ]; then
  case ",${QUEUE}," in
    *,training-normal,*|*,training-parallel,*|*,gpu-interactive,*|*,inference,*) GPU_SPEC="-" ;;
  esac
fi
GPU_DIRECTIVE=""
[ -n "$GPU_SPEC" ] && GPU_DIRECTIVE="#BSUB -gpu \"${GPU_SPEC}\""

OUTPUT_ROOT="$(sed -nE 's/^[[:space:]]*output_root:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT" ;;
esac
LOG_DIR="$OUTPUT_ROOT/$SAMPLE_ID/logs"
mkdir -p "$LOG_DIR"

bsub <<EOF
#BSUB -J qc_${SAMPLE_ID}
#BSUB -o ${LOG_DIR}/qc_${SAMPLE_ID}_%J.out
#BSUB -e ${LOG_DIR}/qc_${SAMPLE_ID}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
${GPU_DIRECTIVE}
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEP_DIRECTIVE}

set -euo pipefail
if command -v module >/dev/null 2>&1; then module load cellgen/uv; fi
source "${ENV_PATH}/bin/activate"
cd "${ROOT}"
PYTHONPATH=src python3 src/terra_multi/pipeline/02_quality_check.py --config "${CONFIG}" --sample-id "${SAMPLE_ID}"
EOF
