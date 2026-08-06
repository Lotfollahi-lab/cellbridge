#!/bin/bash
set -euo pipefail
SAMPLE="${1:?usage: 03_cell_labels.sh <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
QUEUE="${LABEL_QUEUE:-${QUEUE:-oversubscribed}}"
GPU="${LABEL_GPU_SPEC:-${CPU_GPU_SPEC:-}}"
case ",${QUEUE}," in *,training-normal,*|*,training-parallel,*|*,gpu-interactive,*|*,inference,*) GPU="${GPU:--}" ;; esac
GPU_ARGS=(); [ -z "$GPU" ] || GPU_ARGS=(--gpu "$GPU")
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "labels_${SAMPLE}" \
  --log-prefix "labels_${SAMPLE}" --output-key output_root --log-subdir "$SAMPLE/logs" \
  --queue "$QUEUE" "${GPU_ARGS[@]}" --memory "${LABEL_MEM_GB:-8}" \
  --cores "${LABEL_NCORES:-4}" "${DEP_ARGS[@]}" -- python3 \
  src/cellbridge/preprocess/03_cell_labels.py --config "$CONFIG" --sample-id "$SAMPLE"
