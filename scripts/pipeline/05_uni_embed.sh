#!/bin/bash
# Step 4: submit one sample's UNI2-h image embedding job (GPU). Needs {id}_terra.npz.
set -euo pipefail

SAMPLE_ID="$1"   # e.g. TENX133

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SCRIPT_DIR/cluster_env.sh" ] && . "$SCRIPT_DIR/cluster_env.sh"

# GPU job -- queue/GPU/cores/mem come from dedicated GPU_* vars so the CPU-tuned
# QUEUE/MEM_GB/NCORES in cluster_env.sh don't get applied here. Override via env.
QUEUE="${GPU_QUEUE:-gpu-normal}"
GPU_SPEC="${GPU_SPEC:-num=1:gmem=20000}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${GPU_MEM_GB:-32}"
NCORES="${GPU_NCORES:-4}"
CONFIG="${CONFIG:-$ROOT/configs/pipeline/pipeline.yaml}"

# Per-sample outputs (and logs) live under output_root/<sample>.
OUTPUT_ROOT="$(sed -nE 's/^[[:space:]]*output_root:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_ROOT" in
  /*) : ;;                              # already absolute
  *) OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT" ;;  # relative to project root (matches run cwd)
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

set -euo pipefail
module load cellgen/uv
source ${ENV_PATH}/bin/activate
which python3
python3 --version

cd ${ROOT}
PYTHONPATH=src python3 -m terra_multi.pipeline.uni_embed --config ${CONFIG} --sample-id ${SAMPLE_ID}
EOF
