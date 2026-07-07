#!/bin/bash
set -euo pipefail

SAMPLE_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"

QUEUE="${UNI_QUEUE:-${GPU_QUEUE:-gpu-normal}}"
GPU_SPEC="${UNI_GPU_SPEC:-${GPU_SPEC:-num=1:gmem=30000}}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${UNI_MEM_GB:-48}"
NCORES="${UNI_NCORES:-4}"
CONFIG="${CONFIG:-$ROOT/configs/pipeline/pipeline.yaml}"
DEP_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEP_DIRECTIVE="#BSUB -w \"done(${DEPENDENCY})\""

OUTPUT_ROOT="$(sed -nE 's/^[[:space:]]*output_root:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT" ;;
esac
LOG_DIR="$OUTPUT_ROOT/$SAMPLE_ID/logs"
mkdir -p "$LOG_DIR"

bsub <<EOF
#BSUB -J uni_${SAMPLE_ID}
#BSUB -o ${LOG_DIR}/uni_${SAMPLE_ID}_%J.out
#BSUB -e ${LOG_DIR}/uni_${SAMPLE_ID}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEP_DIRECTIVE}

set -euo pipefail
module load ISG/conda
conda activate hest
cd "${ROOT}"
PYTHONPATH=src python3 src/terra_multi/pipeline/05_uni_embed.py --config "${CONFIG}" --sample-id "${SAMPLE_ID}"
EOF
