#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac

OUTPUT_DIR="$(awk '
  /^embedding:/ { section=1; next }
  section && /^[^[:space:]]/ { section=0 }
  section && $1 == "output_dir:" { print $2; exit }
' "$CONFIG")"
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
mkdir -p "$OUTPUT_DIR/logs"

QUEUE="${PHENOCELL_QUEUE:-${GPU_QUEUE:-training-normal}}"
GPU_SPEC="${PHENOCELL_GPU_SPEC:-num=1:gmem=30000}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${PHENOCELL_EMBED_MEM_GB:-32}"
NCORES="${PHENOCELL_EMBED_NCORES:-8}"
DEP_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEP_DIRECTIVE="#BSUB -w \"done(${DEPENDENCY})\""

bsub <<EOF
#BSUB -J phenocell_embed
#BSUB -o ${OUTPUT_DIR}/logs/embed_%J.out
#BSUB -e ${OUTPUT_DIR}/logs/embed_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEP_DIRECTIVE}

set -euo pipefail
module load ISG/conda
module load cellgen/uv
conda activate hest
cd "${ROOT}"
PYTHONPATH=src uv run --active --no-project --with lmdb python \
  -m terra_multi.evaluation.embed_phenocell --config "${CONFIG}"
EOF
