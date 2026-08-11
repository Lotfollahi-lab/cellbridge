#!/bin/bash
set -euo pipefail
SAMPLE="${1:?usage: 02_quality_check.sh <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "qc_${SAMPLE}" \
  --log-prefix "qc_${SAMPLE}" --output-key output_root --log-subdir "$SAMPLE/logs" \
  --queue "${QC_QUEUE:-auto}" --memory "${QC_MEM_GB:-8}" \
  --cores "${QC_NCORES:-4}" "${DEP_ARGS[@]}" -- python3 \
  src/cellbridge/preprocess/02_quality_check.py --config "$CONFIG" --sample-id "$SAMPLE"
