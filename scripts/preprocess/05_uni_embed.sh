#!/bin/bash
set -euo pipefail
SAMPLE="${1:?usage: 05_uni_embed.sh <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "uni_${SAMPLE}" \
  --log-prefix "uni_${SAMPLE}" --output-key output_root --log-subdir "$SAMPLE/logs" \
  --queue "${UNI_QUEUE:-${GPU_QUEUE:-training-normal}}" \
  --gpu "${UNI_GPU_SPEC:-${GPU_SPEC:-num=1:gmem=30000}}" \
  --memory "${UNI_MEM_GB:-16}" --cores "${UNI_NCORES:-12}" --environment hest \
  "${DEP_ARGS[@]}" -- python3 src/cellbridge/preprocess/05_uni_embed.py \
  --config "$CONFIG" --sample-id "$SAMPLE" --batch-size "${UNI_BATCH_SIZE:-128}"
