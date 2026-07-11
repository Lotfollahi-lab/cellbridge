#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac

OUTPUT_DIR="$(sed -nE 's/^output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
mkdir -p "$OUTPUT_DIR/logs"

QUEUE="${PHENOCELL_PROJECT_QUEUE:-training-normal}"
GPU_SPEC="${PHENOCELL_PROJECT_GPU_SPEC:-num=1:gmem=10000}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${PHENOCELL_PROJECT_MEM_GB:-32}"
NCORES="${PHENOCELL_PROJECT_NCORES:-4}"
DEP_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEP_DIRECTIVE="#BSUB -w \"done(${DEPENDENCY})\""

bsub <<EOF
#BSUB -J phenocell_project
#BSUB -o ${OUTPUT_DIR}/logs/project_%J.out
#BSUB -e ${OUTPUT_DIR}/logs/project_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "${GPU_SPEC}"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEP_DIRECTIVE}

set -euo pipefail
module load cellgen/uv
source "${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}/bin/activate"
cd "${ROOT}"
PYTHONPATH=src python3 -m terra_multi.evaluation.project_phenocell \
  --config "${CONFIG}"
EOF
