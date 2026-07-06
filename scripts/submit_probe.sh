#!/bin/bash
set -euo pipefail

MODEL="$1"   # terra | uni | concat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SCRIPT_DIR/cluster_env.sh" ] && . "$SCRIPT_DIR/cluster_env.sh"

QUEUE="${QUEUE:-oversubscribed}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${MEM_GB:-4}"
NCORES="${NCORES:-2}"
CONFIG="${CONFIG:-$ROOT/configs/exploration/probe.yaml}"

# Derive the output dir from the config
OUTPUT_DIR="$(sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"

case "$OUTPUT_DIR" in
  /*) : ;;                          # already absolute
  *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;;  # relative to project root (matches run cwd)
esac
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

bsub <<EOF
#BSUB -J probe_${MODEL}
#BSUB -o ${LOG_DIR}/probe_${MODEL}_%J.out
#BSUB -e ${LOG_DIR}/probe_${MODEL}_%J.err
#BSUB -n ${NCORES}
#BSUB -q ${QUEUE}
#BSUB -G ${BGROUP}
#BSUB -M ${MEM_GB}G
#BSUB -R "select[mem>${MEM_GB}G] rusage[mem=${MEM_GB}G]"

set -euo pipefail
module load cellgen/uv
source ${ENV_PATH}/bin/activate
which python3
python3 --version

cd ${ROOT}
PYTHONPATH=src python3 -m terra_multi.exploration.run_linear_probe --config ${CONFIG} --model ${MODEL}
EOF
