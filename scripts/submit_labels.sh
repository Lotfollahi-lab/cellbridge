#!/bin/bash
# Step 5: submit one sample's cell-type label transfer (CPU).
set -euo pipefail

SAMPLE_ID="$1"   # e.g. TENX133

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SCRIPT_DIR/cluster_env.sh" ] && . "$SCRIPT_DIR/cluster_env.sh"

QUEUE="${QUEUE:-oversubscribed}"
BGROUP="${BGROUP:-s10396}"
MEM_GB="${MEM_GB:-8}"   # bump for big sections: MEM_GB=32 scripts/submit_labels.sh <id>
NCORES="${NCORES:-1}"
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
#BSUB -J labels_${SAMPLE_ID}
#BSUB -o ${LOG_DIR}/labels_${SAMPLE_ID}_%J.out
#BSUB -e ${LOG_DIR}/labels_${SAMPLE_ID}_%J.err
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
PYTHONPATH=src python3 -m terra_multi.pipeline.celltype --config ${CONFIG} --sample-id ${SAMPLE_ID}
EOF
