#!/bin/bash
set -euo pipefail

SAMPLE_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"

QUEUE="${TERRA_QUEUE:-${GPU_QUEUE:-training-normal}}"
GPU_SPEC="${TERRA_GPU_SPEC:-${GPU_SPEC:-num=1:gmem=20000}}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${TERRA_MEM_GB:-16}"
NCORES="${TERRA_NCORES:-12}"
BATCH_SIZE="${TERRA_BATCH_SIZE:-128}"
ENV_PATH="${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
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
#BSUB -J terra_${SAMPLE_ID}
#BSUB -o ${LOG_DIR}/terra_${SAMPLE_ID}_%J.out
#BSUB -e ${LOG_DIR}/terra_${SAMPLE_ID}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEP_DIRECTIVE}

set -euo pipefail
if command -v module >/dev/null 2>&1; then module load cellgen/uv; fi
source "${ENV_PATH}/bin/activate"
cd "${ROOT}"
PYTHONPATH=src python3 src/terra_multi/preprocess/04_terra_embed.py \
  --config "${CONFIG}" \
  --sample-id "${SAMPLE_ID}" \
  --batch-size "${BATCH_SIZE}"
EOF
