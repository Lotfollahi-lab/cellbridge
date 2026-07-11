#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
ARM="${2:-}"
case "$ARM" in ""|uni|distilled) ;; *) echo "arm must be uni or distilled" >&2; exit 1 ;; esac
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac

OUTPUT_DIR="$(sed -nE 's/^output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
mkdir -p "$OUTPUT_DIR/logs"

QUEUE="${PHENOCELL_PROBE_QUEUE:-training-normal}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${PHENOCELL_PROBE_MEM_GB:-64}"
NCORES="${PHENOCELL_PROBE_NCORES:-8}"
DEP_DIRECTIVE=""
[ -n "${DEPENDENCY:-}" ] && DEP_DIRECTIVE="#BSUB -w \"done(${DEPENDENCY})\""
ARM_ARGS=""
[ -n "$ARM" ] && ARM_ARGS="--arm $ARM"

bsub <<EOF
#BSUB -J phenocell_probe${ARM:+_$ARM}
#BSUB -o ${OUTPUT_DIR}/logs/probe${ARM:+_$ARM}_%J.out
#BSUB -e ${OUTPUT_DIR}/logs/probe${ARM:+_$ARM}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "-"
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G] span[hosts=1]"
${DEP_DIRECTIVE}

set -euo pipefail
module load cellgen/uv
source "${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}/bin/activate"
cd "${ROOT}"
export OMP_NUM_THREADS=${NCORES} OPENBLAS_NUM_THREADS=${NCORES} MKL_NUM_THREADS=${NCORES}
PYTHONPATH=src python3 -m terra_multi.evaluation.probe_phenocell \
  --config "${CONFIG}" ${ARM_ARGS}
EOF
