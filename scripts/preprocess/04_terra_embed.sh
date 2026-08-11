#!/bin/bash
set -euo pipefail
SAMPLE="${1:?usage: 04_terra_embed.sh <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "terra_${SAMPLE}" \
  --log-prefix "terra_${SAMPLE}" --output-key output_root --log-subdir "$SAMPLE/logs" \
  --queue "${TERRA_QUEUE:-auto}" --gmem "${TERRA_GMEM:-20000}" \
  --memory "${TERRA_MEM_GB:-16}" --cores "${TERRA_NCORES:-12}" \
  "${DEP_ARGS[@]}" -- python3 src/cellbridge/preprocess/04_terra_embed.py \
  --config "$CONFIG" --sample-id "$SAMPLE" --batch-size "${TERRA_BATCH_SIZE:-128}"
