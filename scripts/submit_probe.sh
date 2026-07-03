#!/bin/bash
set -euo pipefail

MODEL="$1"   # terra | uni | concat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SCRIPT_DIR/cluster_env.sh" ] && . "$SCRIPT_DIR/cluster_env.sh"

QUEUE="${QUEUE:-normal}"
BGROUP="${BGROUP:-team361}"
MEM_GB="${MEM_GB:-32}"
NCORES="${NCORES:-4}"
CONFIG="${CONFIG:-$ROOT/configs/probe.yaml}"
LOG_DIR="$ROOT/outputs/linear_probe/logs"
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
PYTHONPATH=src python3 -m terra_multi.experiments.run_linear_probe --config ${CONFIG} --model ${MODEL}
EOF
