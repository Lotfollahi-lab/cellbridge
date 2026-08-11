#!/bin/bash
set -euo pipefail
SAMPLE="${1:?usage: 02_quality_check_streaming.sh <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "qcstream_${SAMPLE}" \
  --log-prefix "qcstream_${SAMPLE}" --output-key output_root \
  --log-subdir "$SAMPLE/logs" --queue "${QC_QUEUE:-auto}" \
  --gmem "${QC_GMEM:-1000}" \
  --memory "${QC_MEM_GB:-64}" --cores "${QC_NCORES:-4}" "${DEP_ARGS[@]}" -- \
  python3 src/cellbridge/preprocess/02_quality_check_streaming.py \
  --config "$CONFIG" --sample-id "$SAMPLE" --batch-rows "${QC_BATCH_ROWS:-5000000}"
