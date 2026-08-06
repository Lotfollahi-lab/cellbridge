#!/bin/bash
set -euo pipefail
MODEL="${1:?usage: submit_celltype.sh <model>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${CONFIG:-$ROOT/configs/exploration/celltype_tenx116.yaml}"
QUEUE="${CELLTYPE_QUEUE:-${QUEUE:-oversubscribed}}"
GPU="${CELLTYPE_GPU_SPEC:-${CPU_GPU_SPEC:-}}"
case ",${QUEUE}," in *,training-normal,*|*,training-parallel,*|*,gpu-interactive,*|*,inference,*) GPU="${GPU:--}" ;; esac
GPU_ARGS=(); [ -z "$GPU" ] || GPU_ARGS=(--gpu "$GPU")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "celltype_${MODEL}" \
  --log-prefix "celltype_${MODEL}" --queue "$QUEUE" "${GPU_ARGS[@]}" \
  --memory "${CELLTYPE_MEM_GB:-${MEM_GB:-8}}" \
  --cores "${CELLTYPE_NCORES:-${NCORES:-1}}" -- \
  python3 -m cellbridge.evaluation.run --config "$CONFIG" --model "$MODEL"
